// FILENAME: src/hyper_shared_ptr.cpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#include <pthread.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <new>
#include <theta/hyper_shared_ptr.hpp>
#include <thread>

#if THETA_OS_LINUX
#include <fcntl.h>
#include <linux/membarrier.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#if THETA_HAS_LINUX_RSEQ
#include <sys/rseq.h>
#endif
#endif

namespace theta::hyper_shared_ptr::detail {

constinit const void* g_theta_sp_vptr = nullptr;
constinit std::int64_t* g_rseq_slab_base = nullptr;
constinit std::atomic<std::uint64_t>* g_hsp_slab_base = nullptr;

namespace {

constinit std::atomic<bool> g_runtime_ready{false};
constinit std::atomic<bool> g_runtime_init_lock{false};
constinit bool g_rseq_membarrier_ok = false;
constinit std::int32_t g_max_possible_cpus = 1;

constinit std::atomic<std::uint64_t> g_mb_window_ms{0};
constinit std::atomic<std::uint64_t> g_mb_storm_until_ms{0};
constinit std::atomic<std::uint32_t> g_mb_burst_count{0};

std::uint64_t coarse_monotonic_ms() noexcept {
#if THETA_OS_LINUX
  struct timespec ts{};
  ::clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000ULL +
         static_cast<std::uint64_t>(ts.tv_nsec / 1000000L);
#else
  return 0ULL;
#endif
}

bool in_membarrier_storm() noexcept {
  const std::uint64_t until =
      g_mb_storm_until_ms.load(std::memory_order_relaxed);
  if (until == 0ULL) [[likely]] {
    return false;
  }
  return coarse_monotonic_ms() <= until;
}

void note_involuntary_harvest_membarrier() noexcept {
  const std::uint64_t now = coarse_monotonic_ms();
  const std::uint64_t win = g_mb_window_ms.load(std::memory_order_relaxed);
  if (now - win > 25ULL) {
    g_mb_window_ms.store(now, std::memory_order_relaxed);
    g_mb_burst_count.store(1U, std::memory_order_relaxed);
  } else {
    const std::uint32_t c =
        g_mb_burst_count.fetch_add(1U, std::memory_order_relaxed) + 1U;
    if (c >= 256U) {
      g_mb_storm_until_ms.store(now + 250ULL, std::memory_order_relaxed);
    }
  }
}

void broadcast_rseq_membarrier(bool count_toward_storm) noexcept {
#if THETA_OS_LINUX
#if THETA_HSP_USE_RSEQ_ASM
  ::syscall(SYS_membarrier, MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ, 0, 0);
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
  if (count_toward_storm) {
    note_involuntary_harvest_membarrier();
  }
#else
  (void)count_toward_storm;
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

std::int32_t parse_linux_possible_cpus() noexcept {
#if THETA_OS_LINUX
  const int fd =
      ::open("/sys/devices/system/cpu/possible", O_RDONLY | O_CLOEXEC);
  if (fd >= 0) {
    char buf[128]{};
    const ssize_t n = ::read(fd, buf, sizeof(buf) - 1U);
    ::close(fd);
    if (n > 0) {
      std::int32_t max_id = 0;
      std::int32_t cur = 0;
      bool has_digit = false;
      for (ssize_t i = 0; i < n; ++i) {
        const char ch = buf[i];
        if (ch >= '0' && ch <= '9') {
          cur = cur * 10 + static_cast<std::int32_t>(ch - '0');
          has_digit = true;
        } else {
          if (has_digit && cur > max_id) {
            max_id = cur;
          }
          cur = 0;
          has_digit = false;
        }
      }
      if (has_digit && cur > max_id) {
        max_id = cur;
      }
      const std::int32_t count = max_id + 1;
      if (count >= 1 && count <= kHspMaxCpus) {
        return count;
      }
      if (count > kHspMaxCpus) {
        return kHspMaxCpus;
      }
    }
  }
#endif
  const unsigned int hc = std::thread::hardware_concurrency();
  if (hc == 0U) {
    return 1;
  }
  return hc > static_cast<unsigned int>(kHspMaxCpus)
             ? kHspMaxCpus
             : static_cast<std::int32_t>(hc);
}

void* map_striped_region(std::size_t bytes) noexcept {
#if THETA_OS_LINUX
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
  void* mem = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (mem == MAP_FAILED) {
    return nullptr;
  }
  return mem;
#else
  return std::calloc(1U, bytes);
#endif
}

// ============================================================================
// ABA-Free Striped Slot Allocator + Zero-Guard POD `initial-exec` TLS Cache
// ============================================================================
class StripedSlotAllocator {
 public:
  std::uint32_t alloc_from_global(std::uint32_t* batch_out,
                                  std::uint32_t want) noexcept {
    lock();
    std::uint32_t got = 0U;
    while (got < want && free_top_ > 0U) {
      batch_out[got++] = free_stack_[--free_top_];
    }
    while (got < want && next_fresh_ < kMaxStripedSlots) {
      batch_out[got++] = next_fresh_++;
    }
    unlock();
    return got;
  }

  void free_to_global(const std::uint32_t* slots,
                      std::uint32_t count) noexcept {
    if (count == 0U) {
      return;
    }
    lock();
    for (std::uint32_t i = 0U; i < count; ++i) {
      if (free_top_ < kMaxStripedSlots) {
        free_stack_[free_top_++] = slots[i];
      }
    }
    unlock();
  }

 private:
  void lock() noexcept {
    while (lock_.exchange(1U, std::memory_order_acquire) != 0U) {
      while (lock_.load(std::memory_order_relaxed) != 0U) {
        ::theta::arch::cpu_relax();
      }
    }
  }

  void unlock() noexcept { lock_.store(0U, std::memory_order_release); }

  std::atomic<std::uint8_t> lock_{0};
  std::uint32_t next_fresh_{1};
  std::uint32_t free_top_{0};
  std::uint32_t free_stack_[kMaxStripedSlots]{};
};

constinit StripedSlotAllocator g_rseq_slot_alloc{};
constinit StripedSlotAllocator g_hsp_slot_alloc{};

struct PodTlsSlotState {
  static constexpr std::uint32_t kCap = 32U;
  static constexpr std::uint32_t kRefill = 16U;

  bool cleanup_registered;
  std::uint32_t rseq_count;
  std::uint32_t hsp_count;
  std::uint32_t rseq_ids[kCap];
  std::uint32_t hsp_ids[kCap];
};

#if THETA_OS_LINUX
[[gnu::visibility("hidden")]] __thread PodTlsSlotState g_tls_slots
    __attribute__((tls_model("initial-exec"))) = {};
#else
thread_local PodTlsSlotState g_tls_slots = {};
#endif

constinit pthread_key_t g_tls_cleanup_key{};
constinit bool g_tls_key_created = false;

void on_thread_exit_flush_slots(void* /*arg*/) noexcept {
  if (g_tls_slots.rseq_count > 0U) {
    g_rseq_slot_alloc.free_to_global(g_tls_slots.rseq_ids,
                                     g_tls_slots.rseq_count);
    g_tls_slots.rseq_count = 0U;
  }
  if (g_tls_slots.hsp_count > 0U) {
    g_hsp_slot_alloc.free_to_global(g_tls_slots.hsp_ids, g_tls_slots.hsp_count);
    g_tls_slots.hsp_count = 0U;
  }
}

void ensure_thread_cleanup_registered() noexcept {
  if (!g_tls_slots.cleanup_registered && g_tls_key_created) [[unlikely]] {
    g_tls_slots.cleanup_registered = true;
    (void)::pthread_setspecific(g_tls_cleanup_key, &g_tls_slots);
  }
}

void ensure_runtime_init() noexcept {
  if (g_runtime_ready.load(std::memory_order_acquire)) [[likely]] {
    return;
  }
  bool expected = false;
  while (!g_runtime_init_lock.compare_exchange_weak(
      expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
    expected = false;
    ::theta::arch::cpu_relax();
  }

  if (!g_runtime_ready.load(std::memory_order_relaxed)) {
    g_max_possible_cpus = parse_linux_possible_cpus();
    if (!g_tls_key_created) {
      g_tls_key_created =
          (::pthread_key_create(&g_tls_cleanup_key,
                                &on_thread_exit_flush_slots) == 0);
    }

    const std::size_t total_cells =
        static_cast<std::size_t>(kHspMaxCpus) * kMaxStripedSlots;

    if (g_rseq_slab_base == nullptr) {
      g_rseq_slab_base = static_cast<std::int64_t*>(
          map_striped_region(total_cells * sizeof(std::int64_t)));
    }
    if (g_hsp_slab_base == nullptr) {
      g_hsp_slab_base = static_cast<std::atomic<std::uint64_t>*>(
          map_striped_region(total_cells * sizeof(std::uint64_t)));
    }

#if THETA_HSP_USE_RSEQ_ASM
    bool rseq_tls_ok = false;
    if (__rseq_size > 0U) {
      const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
      const auto* cpu_ptr =
          reinterpret_cast<const std::int32_t*>(tp + __rseq_offset + 4);
      const std::int32_t c = __atomic_load_n(cpu_ptr, __ATOMIC_RELAXED);
      rseq_tls_ok = (c >= 0 && c < kHspMaxCpus);
    }
    const long rc = ::syscall(
        SYS_membarrier, MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ, 0, 0);
    g_rseq_membarrier_ok =
        rseq_tls_ok && (rc == 0) && (g_rseq_slab_base != nullptr);
#else
    g_rseq_membarrier_ok = (g_rseq_slab_base != nullptr);
#endif

    // Also seed `g_theta_sp_vptr` eagerly while single-threaded at startup!
    if (__atomic_load_n(&g_theta_sp_vptr, __ATOMIC_RELAXED) == nullptr) {
      ThetaSpCountedBase sample(nullptr, nullptr, nullptr, nullptr);
      record_theta_sp_vptr(&sample);
    }

    g_runtime_ready.store(true, std::memory_order_release);
  }

  g_runtime_init_lock.store(false, std::memory_order_release);
}

// Register `MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ` and `MAP_NORESERVE`
// slabs at process startup while `mm->mm_users == 1`, avoiding Linux's ~8 ms
// `synchronize_rcu()` grace-period stall when first called after
// `pthread_create`.
[[gnu::constructor(101)]] void eager_init_hsp_runtime() noexcept {
  ensure_runtime_init();
}

std::uint32_t alloc_rseq_slot() noexcept {
  if (g_tls_slots.rseq_count == 0U) [[unlikely]] {
    ensure_thread_cleanup_registered();
    g_tls_slots.rseq_count = g_rseq_slot_alloc.alloc_from_global(
        g_tls_slots.rseq_ids, PodTlsSlotState::kRefill);
    if (g_tls_slots.rseq_count == 0U) [[unlikely]] {
      return 0U;
    }
  }
  return g_tls_slots.rseq_ids[--g_tls_slots.rseq_count];
}

void free_rseq_slot(std::uint32_t slot) noexcept {
  if (slot == 0U) {
    return;
  }
  if (g_tls_slots.rseq_count == PodTlsSlotState::kCap) [[unlikely]] {
    ensure_thread_cleanup_registered();
    g_rseq_slot_alloc.free_to_global(g_tls_slots.rseq_ids,
                                     PodTlsSlotState::kRefill);
    std::memmove(g_tls_slots.rseq_ids,
                 g_tls_slots.rseq_ids + PodTlsSlotState::kRefill,
                 (PodTlsSlotState::kCap - PodTlsSlotState::kRefill) *
                     sizeof(std::uint32_t));
    g_tls_slots.rseq_count -= PodTlsSlotState::kRefill;
  }
  g_tls_slots.rseq_ids[g_tls_slots.rseq_count++] = slot;
}

std::uint32_t alloc_hsp_slot() noexcept {
  if (g_tls_slots.hsp_count == 0U) [[unlikely]] {
    ensure_thread_cleanup_registered();
    g_tls_slots.hsp_count = g_hsp_slot_alloc.alloc_from_global(
        g_tls_slots.hsp_ids, PodTlsSlotState::kRefill);
    if (g_tls_slots.hsp_count == 0U) [[unlikely]] {
      return 0U;
    }
  }
  return g_tls_slots.hsp_ids[--g_tls_slots.hsp_count];
}

void free_hsp_slot(std::uint32_t slot) noexcept {
  if (slot == 0U) {
    return;
  }
  if (g_tls_slots.hsp_count == PodTlsSlotState::kCap) [[unlikely]] {
    ensure_thread_cleanup_registered();
    g_hsp_slot_alloc.free_to_global(g_tls_slots.hsp_ids,
                                    PodTlsSlotState::kRefill);
    std::memmove(g_tls_slots.hsp_ids,
                 g_tls_slots.hsp_ids + PodTlsSlotState::kRefill,
                 (PodTlsSlotState::kCap - PodTlsSlotState::kRefill) *
                     sizeof(std::uint32_t));
    g_tls_slots.hsp_count -= PodTlsSlotState::kRefill;
  }
  g_tls_slots.hsp_ids[g_tls_slots.hsp_count++] = slot;
}

THETA_FORCE_INLINE void lock_cb(ThetaSpCountedBase* cb) noexcept {
  while (cb->reg_lock_.exchange(1U, std::memory_order_acquire) != 0U) {
    while (cb->reg_lock_.load(std::memory_order_relaxed) != 0U) {
      ::theta::arch::cpu_relax();
    }
  }
}

THETA_FORCE_INLINE void unlock_cb(ThetaSpCountedBase* cb) noexcept {
  cb->reg_lock_.store(0U, std::memory_order_release);
}

void append_reg_cpu_locked(ThetaSpCountedBase* cb, std::uint16_t cpu) noexcept {
  const std::uint16_t n = cb->num_reg_cpus_;
  if (n < 8U) [[likely]] {
    cb->inline_cpus_[n] = cpu;
    cb->num_reg_cpus_ = static_cast<std::uint16_t>(n + 1U);
    return;
  }
  const std::uint16_t ext_idx = static_cast<std::uint16_t>(n - 8U);
  if (ext_idx >= cb->overflow_cap_) {
    const std::uint16_t new_cap =
        cb->overflow_cap_ == 0U
            ? 16U
            : static_cast<std::uint16_t>(cb->overflow_cap_ >= 4096U
                                             ? kHspMaxCpus
                                             : cb->overflow_cap_ * 4U);
    auto* next_buf = new (std::nothrow) std::uint16_t[new_cap];
    if (next_buf == nullptr) [[unlikely]] {
      return;
    }
    if (cb->overflow_cpus_ != nullptr) {
      std::memcpy(next_buf, cb->overflow_cpus_,
                  ext_idx * sizeof(std::uint16_t));
      delete[] cb->overflow_cpus_;
    }
    cb->overflow_cpus_ = next_buf;
    cb->overflow_cap_ = new_cap;
  }
  cb->overflow_cpus_[ext_idx] = cpu;
  cb->num_reg_cpus_ = static_cast<std::uint16_t>(n + 1U);
}

std::uint16_t get_reg_cpu_locked(const ThetaSpCountedBase* cb,
                                 std::uint16_t i) noexcept {
  if (i < 8U) [[likely]] {
    return cb->inline_cpus_[i];
  }
  return cb->overflow_cpus_[i - 8U];
}

bool try_register_cpu_cell_locked(ThetaSpCountedBase* cb, std::int32_t cpu,
                                  std::int64_t init_encoded) noexcept {
  const std::uint32_t slot = cb->fast_.slot_id;
  std::int64_t* const cell =
      g_rseq_slab_base + (static_cast<std::size_t>(cpu) << kSlotShift) + slot;
  if (__atomic_load_n(cell, __ATOMIC_RELAXED) != 0) {
    return false;
  }
  append_reg_cpu_locked(cb, static_cast<std::uint16_t>(cpu));
  __atomic_store_n(cell, init_encoded, __ATOMIC_RELEASE);
  return true;
}

// Single-Registered-CPU Quiescence Theorem: if zero CPUs or ONLY our currently
// executing CPU ever registered on `cb`, any prior thread on our CPU was
// already `rseq`-aborted on context switch, so a local `seq_cst` fence after
// `fast_active = 0` achieves global quiescence with 0 `SYS_membarrier` IPIs.
void quiesce_rseq_for_cb_locked(const ThetaSpCountedBase* cb,
                                bool count_toward_storm) noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const std::uint16_t n = cb->num_reg_cpus_;
  if (n == 0U) {
    return;
  }
  if (n == 1U) {
    const std::int32_t self_cpu = current_cpu_id();
    if (cb->inline_cpus_[0] == static_cast<std::uint16_t>(self_cpu)) {
      std::atomic_signal_fence(std::memory_order_seq_cst);
      if (current_cpu_id() == self_cpu) {
        return;
      }
    }
  }
  broadcast_rseq_membarrier(count_toward_storm);
}

bool promote_to_percpu_locked(ThetaSpCountedBase* cb, std::int64_t first_delta,
                              bool force_keepalive) noexcept {
  ensure_runtime_init();
  if (!g_rseq_membarrier_ok || (!force_keepalive && in_membarrier_storm())) {
    __atomic_store_n(&cb->fast_.mode, kModeAtomicRetired, __ATOMIC_RELEASE);
    return false;
  }
  const std::uint32_t slot = alloc_rseq_slot();
  if (slot == 0U) [[unlikely]] {
    __atomic_store_n(&cb->fast_.mode, kModeAtomicRetired, __ATOMIC_RELEASE);
    return false;
  }

  const std::int32_t u_refs = cb->unpromoted_theta_refs_;
  cb->unpromoted_theta_refs_ = 0;

  cb->_M_weak_add_ref();

  if (u_refs == 0) {
    cb->_M_add_ref_copy();
  } else if (u_refs > 1) {
    for (std::int32_t i = 0; i < u_refs - 1; ++i) {
      cb->_M_release();
    }
  }

  cb->central_tickets_.store(static_cast<std::int64_t>(u_refs),
                             std::memory_order_relaxed);
  cb->fast_.slot_id = slot;

  const std::int32_t cpu = current_cpu_id();
  (void)try_register_cpu_cell_locked(cb, cpu, 1 + first_delta);

  __atomic_store_n(&cb->fast_.mode, kModePerCpuActive, __ATOMIC_RELEASE);
  __atomic_store_n(&cb->fast_.fast_active, 1U, __ATOMIC_RELEASE);
  return true;
}

}  // namespace

[[gnu::aligned(64)]] void ThetaSpCountedBase::_M_dispose() noexcept {
  if (dispose_fn_ != nullptr) {
    dispose_fn_(this);
  }
}

[[gnu::aligned(64)]] void record_theta_sp_vptr(
    const ThetaSpCountedBase* sample) noexcept {
  const void* vptr = nullptr;
  std::memcpy(&vptr, static_cast<const void*>(sample), sizeof(vptr));
  __atomic_store_n(&g_theta_sp_vptr, vptr, __ATOMIC_RELEASE);
}

[[gnu::aligned(64)]] std::int32_t current_cpu_id() noexcept {
#if THETA_HAS_LINUX_RSEQ && (THETA_ARCH_X86_64 || THETA_ARCH_AARCH64)
  if (__rseq_size > 0U) [[likely]] {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const auto* cpu_ptr =
        reinterpret_cast<const std::int32_t*>(tp + __rseq_offset + 4);
    const std::int32_t c = __atomic_load_n(cpu_ptr, __ATOMIC_RELAXED);
    if (c >= 0 && c < kHspMaxCpus) [[likely]] {
      return c;
    }
  }
#endif
#if THETA_OS_LINUX
  const int sc = ::sched_getcpu();
  if (sc >= 0 && sc < kHspMaxCpus) {
    return static_cast<std::int32_t>(sc);
  }
#endif
  return 0;
}

[[gnu::aligned(64)]] std::int32_t max_possible_cpus() noexcept {
  ensure_runtime_init();
  return g_max_possible_cpus;
}

[[gnu::aligned(64)]] void sp_add_ref_slow(StdSpCountedBase* pi) noexcept {
  if (!is_theta_sp(pi)) {
    pi->_M_add_ref_copy();
    return;
  }
  auto* cb = static_cast<ThetaSpCountedBase*>(pi);
  if (__atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) ==
      kModeAtomicRetired) {
    pi->_M_add_ref_copy();
    return;
  }

  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode == kModeUnpromoted) {
    if (promote_to_percpu_locked(cb, /*first_delta=*/1,
                                 /*force_keepalive=*/false)) {
      unlock_cb(cb);
      return;
    }
    unlock_cb(cb);
    pi->_M_add_ref_copy();
    return;
  }

  if (mode == kModePerCpuActive) {
    std::int64_t drained = 0;
    if (ArchBackend::try_rseq_drain_local_surplus(&cb->fast_, drained)) {
      cb->central_tickets_.fetch_add(drained + 1, std::memory_order_release);
      unlock_cb(cb);
      return;
    }
    const std::int32_t cpu = current_cpu_id();
    if (!try_register_cpu_cell_locked(cb, cpu, /*init_encoded=*/2)) {
      cb->central_tickets_.fetch_add(1, std::memory_order_relaxed);
    }
    unlock_cb(cb);
    return;
  }

  unlock_cb(cb);
  pi->_M_add_ref_copy();
}

[[gnu::aligned(64)]] void sp_release_slow(StdSpCountedBase* pi) noexcept {
  if (!is_theta_sp(pi)) {
    pi->_M_release();
    return;
  }
  auto* cb = static_cast<ThetaSpCountedBase*>(pi);
  if (__atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) ==
      kModeAtomicRetired) {
    pi->_M_release();
    return;
  }

  // Lockless cross-CPU consumer fast path when `fast_active != 0` and
  // `central_tickets_ >= 2` (fed in 64-item batches by producer
  // `try_rseq_drain_local_surplus`, `0` `SYS_membarrier` IPIs!).
  if (__atomic_load_n(&cb->fast_.fast_active, __ATOMIC_ACQUIRE) != 0U)
      [[likely]] {
    std::int64_t cur_tickets =
        cb->central_tickets_.load(std::memory_order_relaxed);
    while (cur_tickets >= 2) [[likely]] {
      if (cb->central_tickets_.compare_exchange_weak(
              cur_tickets, cur_tickets - 1, std::memory_order_acq_rel,
              std::memory_order_relaxed)) [[likely]] {
        return;
      }
    }
  }

  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode == kModeAtomicRetired) {
    unlock_cb(cb);
    pi->_M_release();
    return;
  }

  if (mode == kModeUnpromoted) {
    --cb->unpromoted_theta_refs_;
    unlock_cb(cb);
    pi->_M_release();
    return;
  }

  const std::int64_t rem =
      cb->central_tickets_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (rem > 0) [[likely]] {
    unlock_cb(cb);
    return;
  }

  __atomic_store_n(&cb->fast_.fast_active, 0U, __ATOMIC_RELEASE);
  quiesce_rseq_for_cb_locked(cb, /*count_toward_storm=*/true);

  const std::uint32_t slot = cb->fast_.slot_id;
  const std::uint16_t n_cpus = cb->num_reg_cpus_;
  std::int64_t delta_sum = 0;
  for (std::uint16_t i = 0U; i < n_cpus; ++i) {
    const std::uint16_t c = get_reg_cpu_locked(cb, i);
    std::int64_t* const cell =
        g_rseq_slab_base + (static_cast<std::size_t>(c) << kSlotShift) + slot;
    const std::int64_t v = __atomic_load_n(cell, __ATOMIC_ACQUIRE);
    if (v >= 1) {
      delta_sum += (v - 1);
    }
  }

  const std::int64_t h_after = rem + delta_sum;
  ++cb->harvest_count_;

  if (h_after == 0) {
    for (std::uint16_t i = 0U; i < n_cpus; ++i) {
      const std::uint16_t c = get_reg_cpu_locked(cb, i);
      std::int64_t* const cell =
          g_rseq_slab_base + (static_cast<std::size_t>(c) << kSlotShift) + slot;
      __atomic_store_n(cell, 0, __ATOMIC_RELEASE);
    }
    cb->num_reg_cpus_ = 0U;
    cb->fast_.slot_id = 0U;
    cb->central_tickets_.store(0, std::memory_order_relaxed);
    free_rseq_slot(slot);
    __atomic_store_n(&cb->fast_.mode, kModeAtomicRetired, __ATOMIC_RELEASE);
    unlock_cb(cb);

    cb->_M_weak_release();
    cb->_M_release();
    return;
  }

  // Retain Per-CPU Mode whenever `h_after >= 1` (e.g. a stationary `g_ptr`
  // whose temporary worker handle migrated once across CPUs!) unless this
  // specific object has exhausted `central_tickets_` repeatedly (`> 4` times,
  // indicating sustained asymmetric unbuffered cross-CPU handoff) or the
  // process is in a membarrier storm.
  if (cb->harvest_count_ <= 4U && !in_membarrier_storm()) {
    for (std::uint16_t i = 0U; i < n_cpus; ++i) {
      const std::uint16_t c = get_reg_cpu_locked(cb, i);
      std::int64_t* const cell =
          g_rseq_slab_base + (static_cast<std::size_t>(c) << kSlotShift) + slot;
      __atomic_store_n(cell, 1, __ATOMIC_RELEASE);
    }
    cb->central_tickets_.store(h_after, std::memory_order_relaxed);
    __atomic_store_n(&cb->fast_.fast_active, 1U, __ATOMIC_RELEASE);
    unlock_cb(cb);
    return;
  }

  for (std::uint16_t i = 0U; i < n_cpus; ++i) {
    const std::uint16_t c = get_reg_cpu_locked(cb, i);
    std::int64_t* const cell =
        g_rseq_slab_base + (static_cast<std::size_t>(c) << kSlotShift) + slot;
    __atomic_store_n(cell, 0, __ATOMIC_RELEASE);
  }
  cb->num_reg_cpus_ = 0U;
  cb->fast_.slot_id = 0U;
  cb->central_tickets_.store(0, std::memory_order_relaxed);
  free_rseq_slot(slot);

  for (std::int64_t i = 0; i < h_after - 1; ++i) {
    cb->_M_add_ref_copy();
  }
  __atomic_store_n(&cb->fast_.mode, kModeAtomicRetired, __ATOMIC_RELEASE);
  unlock_cb(cb);
  cb->_M_weak_release();
}

[[gnu::aligned(64)]] long sp_use_count_slow(
    const StdSpCountedBase* pi) noexcept {
  if (pi == nullptr) {
    return 0L;
  }
  if (!is_theta_sp(pi)) {
    return pi->_M_get_use_count();
  }
  auto* cb = const_cast<ThetaSpCountedBase*>(
      static_cast<const ThetaSpCountedBase*>(pi));
  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode != kModePerCpuActive) {
    const long cnt = pi->_M_get_use_count();
    unlock_cb(cb);
    return cnt;
  }
  const long legacy_plus_lease = pi->_M_get_use_count();
  const long legacy_refs =
      legacy_plus_lease > 0L ? (legacy_plus_lease - 1L) : 0L;
  std::int64_t total =
      legacy_refs + cb->central_tickets_.load(std::memory_order_relaxed);
  const std::uint32_t slot = cb->fast_.slot_id;
  const std::uint16_t n_cpus = cb->num_reg_cpus_;
  for (std::uint16_t i = 0U; i < n_cpus; ++i) {
    const std::uint16_t c = get_reg_cpu_locked(cb, i);
    const std::int64_t* const cell =
        g_rseq_slab_base + (static_cast<std::size_t>(c) << kSlotShift) + slot;
    const std::int64_t v = __atomic_load_n(cell, __ATOMIC_RELAXED);
    if (v >= 1) {
      total += (v - 1);
    }
  }
  unlock_cb(cb);
  return total > 0 ? static_cast<long>(total) : 0L;
}

[[gnu::aligned(64)]] void sp_note_to_std_copy(StdSpCountedBase* pi) noexcept {
  if (is_theta_sp(pi)) {
    auto* cb = static_cast<ThetaSpCountedBase*>(pi);
    __atomic_store_n(&cb->has_weak_or_std_, 1U, __ATOMIC_RELEASE);
  }
  pi->_M_add_ref_copy();
}

[[gnu::aligned(64)]] void sp_note_to_std_move(StdSpCountedBase* pi) noexcept {
  if (!is_theta_sp(pi)) {
    return;
  }
  auto* cb = static_cast<ThetaSpCountedBase*>(pi);
  __atomic_store_n(&cb->has_weak_or_std_, 1U, __ATOMIC_RELEASE);
  if (__atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) ==
      kModeAtomicRetired) {
    return;
  }
  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode == kModeAtomicRetired) {
    unlock_cb(cb);
    return;
  }
  if (mode == kModeUnpromoted) {
    --cb->unpromoted_theta_refs_;
    unlock_cb(cb);
    return;
  }
  cb->_M_add_ref_copy();
  unlock_cb(cb);
  sp_release(pi);
}

[[gnu::aligned(64)]] void sp_add_ref_from_std(StdSpCountedBase* pi) noexcept {
  if (!is_theta_sp(pi)) {
    pi->_M_add_ref_copy();
    return;
  }
  auto* cb = static_cast<ThetaSpCountedBase*>(pi);
  if (__atomic_load_n(&cb->fast_.fast_active, __ATOMIC_RELAXED) != 0U) {
    sp_add_ref(pi);
    return;
  }
  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode == kModeUnpromoted) {
    ++cb->unpromoted_theta_refs_;
    cb->_M_add_ref_copy();
    unlock_cb(cb);
    return;
  }
  unlock_cb(cb);
  sp_add_ref(pi);
}

[[gnu::aligned(64)]] void sp_adopt_from_std_move(
    StdSpCountedBase* pi) noexcept {
  if (!is_theta_sp(pi)) {
    return;
  }
  auto* cb = static_cast<ThetaSpCountedBase*>(pi);
  if (__atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) ==
      kModeAtomicRetired) {
    return;
  }
  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode == kModeAtomicRetired) {
    unlock_cb(cb);
    return;
  }
  if (mode == kModeUnpromoted) {
    ++cb->unpromoted_theta_refs_;
    unlock_cb(cb);
    return;
  }
  cb->central_tickets_.fetch_add(1, std::memory_order_relaxed);
  unlock_cb(cb);
  pi->_M_release();
}

[[gnu::aligned(64)]] bool sp_weak_lock_slow(StdSpCountedBase* pi) noexcept {
  if (!is_theta_sp(pi)) {
    return pi->_M_add_ref_lock_nothrow();
  }
  auto* cb = static_cast<ThetaSpCountedBase*>(pi);
  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode == kModeAtomicRetired) {
    unlock_cb(cb);
    return pi->_M_add_ref_lock_nothrow();
  }
  if (mode == kModeUnpromoted) {
    if (pi->_M_add_ref_lock_nothrow()) {
      ++cb->unpromoted_theta_refs_;
      unlock_cb(cb);
      return true;
    }
    unlock_cb(cb);
    return false;
  }

  cb->central_tickets_.fetch_add(1, std::memory_order_relaxed);
  unlock_cb(cb);
  return true;
}

[[gnu::aligned(64)]] void sp_pin_for_keepalive(
    ThetaSpCountedBase* cb) noexcept {
  lock_cb(cb);
  if (cb->fast_.mode == kModeUnpromoted) {
    (void)promote_to_percpu_locked(cb, /*first_delta=*/0,
                                   /*force_keepalive=*/true);
  } else if (cb->fast_.mode == kModePerCpuActive) {
    const std::int32_t cpu = current_cpu_id();
    (void)try_register_cpu_cell_locked(cb, cpu, /*init_encoded=*/1);
  }
  unlock_cb(cb);
}

[[gnu::aligned(64)]] void sp_register_keepalive_cpu_slow(
    ThetaSpCountedBase* cb) noexcept {
  if (__atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) != kModePerCpuActive) {
    return;
  }
  lock_cb(cb);
  if (cb->fast_.mode == kModePerCpuActive) {
    const std::int32_t cpu = current_cpu_id();
    (void)try_register_cpu_cell_locked(cb, cpu, /*init_encoded=*/1);
  }
  unlock_cb(cb);
}

[[gnu::aligned(64)]] void sp_keepalive_retire_old(
    ThetaSpCountedBase* old_cb) noexcept {
  lock_cb(old_cb);
  quiesce_rseq_for_cb_locked(old_cb, /*count_toward_storm=*/false);
  unlock_cb(old_cb);
}

// ============================================================================
// `HyperControlBlock` (`theta::HyperSharedPtr<T>`) Cold Slow Paths
// ============================================================================
[[gnu::aligned(64)]] std::uint32_t alloc_hsp_slot_and_init_cpu(
    std::int32_t cpu) noexcept {
  ensure_runtime_init();
  if (g_hsp_slab_base == nullptr) [[unlikely]] {
    return 0U;
  }
  const std::uint32_t slot = alloc_hsp_slot();
  if (slot == 0U) [[unlikely]] {
    return 0U;
  }
  const std::size_t idx = (static_cast<std::size_t>(cpu) << kSlotShift) + slot;
  g_hsp_slab_base[idx].store(3ULL, std::memory_order_release);
  return slot;
}

[[gnu::aligned(64)]] void hsp_activate_shard_slow(HyperControlBlock* cb,
                                                  std::int32_t cpu) noexcept {
  if (cb->slot_id_ == 0U || g_hsp_slab_base == nullptr) [[unlikely]] {
    cb->active_shards_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  const std::size_t idx =
      (static_cast<std::size_t>(cpu) << kSlotShift) + cb->slot_id_;
  auto& shard = g_hsp_slab_base[idx];
  std::uint64_t cur = shard.load(std::memory_order_relaxed);
  while (true) {
    if ((cur & 1ULL) != 0ULL) {
      if (shard.compare_exchange_weak(cur, cur + 2ULL,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed)) {
        return;
      }
    } else {
      cb->active_shards_.fetch_add(1U, std::memory_order_relaxed);
      if (shard.compare_exchange_strong(cur, 3ULL, std::memory_order_acq_rel,
                                        std::memory_order_relaxed)) {
        return;
      }
      cb->active_shards_.fetch_sub(1U, std::memory_order_relaxed);
    }
  }
}

[[gnu::aligned(64)]] void hsp_destroy_last_strong_slow(
    HyperControlBlock* cb) noexcept {
  if (cb->slot_id_ != 0U) {
    free_hsp_slot(cb->slot_id_);
    cb->slot_id_ = 0U;
  }
  if (cb->dispose_fn_ != nullptr) {
    cb->dispose_fn_(cb);
  }
  hsp_weak_release_slow(cb);
}

[[gnu::aligned(64)]] void hsp_weak_release_slow(
    HyperControlBlock* cb) noexcept {
  if (cb->weak_refs_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
    if (cb->destroy_fn_ != nullptr) {
      cb->destroy_fn_(cb);
    } else {
      delete cb;
    }
  }
}

[[gnu::aligned(64)]] bool hsp_weak_lock_slow(HyperControlBlock* cb,
                                             std::int32_t cpu) noexcept {
  std::uint32_t active = cb->active_shards_.load(std::memory_order_acquire);
  while (active != 0U) {
    if (cb->active_shards_.compare_exchange_weak(active, active + 1U,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
      if (cb->slot_id_ != 0U && g_hsp_slab_base != nullptr) {
        hsp_inc_on_cpu(cb, cpu);
        cb->active_shards_.fetch_sub(1U, std::memory_order_acq_rel);
      }
      return true;
    }
  }
  return false;
}

[[gnu::aligned(64)]] std::size_t hsp_use_count_slow(
    const HyperControlBlock* cb) noexcept {
  if (cb == nullptr) {
    return 0U;
  }
  const std::uint32_t active =
      cb->active_shards_.load(std::memory_order_acquire);
  if (active == 0U) {
    return 0U;
  }
  if (cb->slot_id_ == 0U || g_hsp_slab_base == nullptr) {
    return active;
  }
  const std::int32_t limit = max_possible_cpus();
  std::size_t sum = 0U;
  std::uint32_t seen_active = 0U;
  for (std::int32_t c = 0; c < limit; ++c) {
    const std::size_t idx =
        (static_cast<std::size_t>(c) << kSlotShift) + cb->slot_id_;
    const std::uint64_t v =
        g_hsp_slab_base[idx].load(std::memory_order_relaxed);
    if ((v & 1ULL) != 0ULL) {
      sum += static_cast<std::size_t>(v >> 1U);
      ++seen_active;
    }
  }
  if (seen_active < active) {
    for (std::int32_t c = limit; c < kHspMaxCpus; ++c) {
      const std::size_t idx =
          (static_cast<std::size_t>(c) << kSlotShift) + cb->slot_id_;
      const std::uint64_t v =
          g_hsp_slab_base[idx].load(std::memory_order_relaxed);
      if ((v & 1ULL) != 0ULL) {
        sum += static_cast<std::size_t>(v >> 1U);
      }
    }
  }
  return sum;
}

}  // namespace theta::hyper_shared_ptr::detail
