// FILENAME: src/portable_shared_ptr.cpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT
//
// Pure C++23 portable runtime for `theta::portable::SharedPtr<T>`,
// `AtomicSharedPtr<T>`, and `KeepAlive<T>` — zero `<sys/rseq.h>`, zero inline
// assembly, zero `<pthread.h>`, and zero `SYS_membarrier` IPIs.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <theta/portable/shared_ptr.hpp>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#endif
#if THETA_OS_LINUX
#include <ctime>
#endif

namespace theta::portable::detail::shared_ptr {

constinit const void* g_portable_sp_vptr = nullptr;
constinit void** g_portable_keepalive_ptrs = nullptr;

#if THETA_OS_LINUX
__thread PortableFastTls g_portable_fast_tls
    __attribute__((tls_model("initial-exec"))) = {nullptr, nullptr};
#else
thread_local PortableFastTls g_portable_fast_tls = {nullptr, nullptr};
#endif

namespace {

inline constexpr std::int64_t kPinnedTicketCushion = 1024;
inline constexpr std::uint32_t kAtomicPinWarmupReads = 2U;
inline constexpr std::uint32_t kMaxHazardRecords = 512U;

struct alignas(64) HazardRecord {
  std::atomic<const void*> protected_cb{nullptr};
  std::atomic<bool> in_use{false};
};

struct alignas(64) PerStripeSlotMeta {
  std::atomic<std::int64_t> central_tickets{0};
  std::uint64_t last_harvest_ms{0};
  std::uint32_t rapid_harvest_streak{0};
  std::uint32_t num_reg_stripes{0};
  std::uint32_t overflow_cap{0};
  std::atomic<std::uint32_t> pin_count{0};
  std::atomic<std::uint32_t> atomic_pin_reads{0};
  std::uint16_t quiescing_unpins{0};
  std::uint8_t keepalive_ptr_set{0};
  std::uint8_t pad_{0};
  std::uint16_t inline_stripes[8]{};
  std::uint16_t* overflow_stripes{nullptr};
};

static_assert(sizeof(PerStripeSlotMeta) == 64);

constinit std::int64_t* g_portable_slab_base = nullptr;
constinit PerStripeSlotMeta* g_portable_slot_meta = nullptr;
constinit bool g_portable_slab_ok = false;

constinit std::atomic<bool> g_runtime_ready{false};
constinit std::atomic<bool> g_runtime_init_lock{false};

constinit HazardRecord g_hazard_records[kMaxHazardRecords]{};
constinit std::atomic<std::uint32_t> g_hazard_high_water{0U};
constinit std::atomic<std::uint32_t> g_next_stripe{0U};

std::uint64_t coarse_monotonic_ms() noexcept {
#if THETA_OS_LINUX
  struct timespec ts{};
  ::clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000ULL +
         static_cast<std::uint64_t>(ts.tv_nsec / 1000000L);
#else
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
#endif
}

void* map_portable_region(std::size_t bytes) noexcept {
#if defined(__unix__) || defined(__APPLE__)
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

class StripedSlotAllocator {
 public:
  std::uint32_t alloc_from_global(std::uint32_t* batch_out,
                                  std::uint32_t want) noexcept {
    lock();
    if (next_fresh_ == 0U) [[unlikely]] {
      next_fresh_ = 1U;
    }
    std::uint32_t got = 0U;
    while (got < want && free_top_ > 0U) {
      batch_out[got++] = free_stack_[--free_top_];
    }
    while (got < want && next_fresh_ < kMaxPortableSlots) {
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
      if (free_top_ < kMaxPortableSlots) {
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
  std::uint32_t next_fresh_{0};
  std::uint32_t free_top_{0};
  std::uint32_t free_stack_[kMaxPortableSlots]{};
};

constinit StripedSlotAllocator g_portable_slot_alloc{};

struct PodTlsSlotCache {
  static constexpr std::uint32_t kCap = 32U;
  static constexpr std::uint32_t kRefill = 16U;

  bool thread_exited;
  bool stripe_assigned;
  std::uint16_t stripe_id;
  std::uint32_t hazard_idx;
  std::uint32_t slot_count;
  std::uint32_t slot_ids[kCap];
};

#if THETA_OS_LINUX
[[gnu::visibility("hidden")]] __thread PodTlsSlotCache g_tls_cache
    __attribute__((tls_model("initial-exec"))) = {};
#else
thread_local PodTlsSlotCache g_tls_cache = {};
#endif

struct PortableThreadExitCleaner {
  bool armed{false};

  ~PortableThreadExitCleaner() noexcept {
    if (!armed) {
      return;
    }
    g_portable_fast_tls = {nullptr, nullptr};
    g_tls_cache.thread_exited = true;
    if (g_tls_cache.slot_count > 0U) {
      g_portable_slot_alloc.free_to_global(g_tls_cache.slot_ids,
                                           g_tls_cache.slot_count);
      g_tls_cache.slot_count = 0U;
    }
    if (g_tls_cache.hazard_idx < kMaxHazardRecords) {
      g_hazard_records[g_tls_cache.hazard_idx].protected_cb.store(
          nullptr, std::memory_order_release);
      g_hazard_records[g_tls_cache.hazard_idx].in_use.store(
          false, std::memory_order_release);
      g_tls_cache.hazard_idx = kMaxHazardRecords;
    }
  }
};

thread_local PortableThreadExitCleaner g_thread_exit_cleaner{};

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
    const std::size_t total_cells =
        static_cast<std::size_t>(kPortableNumStripes) * kMaxPortableSlots;

    if (g_portable_slab_base == nullptr) {
      g_portable_slab_base = static_cast<std::int64_t*>(
          map_portable_region(total_cells * sizeof(std::int64_t)));
    }
    if (g_portable_slot_meta == nullptr) {
      g_portable_slot_meta = static_cast<PerStripeSlotMeta*>(
          map_portable_region(kMaxPortableSlots * sizeof(PerStripeSlotMeta)));
    }
    if (g_portable_keepalive_ptrs == nullptr) {
      g_portable_keepalive_ptrs = static_cast<void**>(
          map_portable_region(kMaxPortableSlots * sizeof(void*)));
    }

    g_portable_slab_ok = (g_portable_slab_base != nullptr) &&
                         (g_portable_slot_meta != nullptr) &&
                         (g_portable_keepalive_ptrs != nullptr);

    if (__atomic_load_n(&g_portable_sp_vptr, __ATOMIC_RELAXED) == nullptr) {
      PortableSpCountedBase sample(0U);
      record_portable_sp_vptr(&sample);
    }

    g_runtime_ready.store(true, std::memory_order_release);
  }

  g_runtime_init_lock.store(false, std::memory_order_release);
}

[[gnu::constructor(102)]] void eager_init_portable_runtime() noexcept {
  ensure_runtime_init();
}

std::uint16_t ensure_thread_tls_initialized() noexcept {
  ensure_runtime_init();
  if (g_tls_cache.thread_exited || !g_portable_slab_ok) [[unlikely]] {
    return 0U;
  }
  if (!g_tls_cache.stripe_assigned) [[unlikely]] {
    const std::uint16_t stripe = static_cast<std::uint16_t>(
        g_next_stripe.fetch_add(1U, std::memory_order_relaxed) &
        (kPortableNumStripes - 1U));
    g_tls_cache.stripe_id = stripe;
    g_tls_cache.stripe_assigned = true;
    g_tls_cache.hazard_idx = kMaxHazardRecords;

    for (std::uint32_t i = 0U; i < kMaxHazardRecords; ++i) {
      bool expected = false;
      if (g_hazard_records[i].in_use.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel,
              std::memory_order_relaxed)) {
        g_tls_cache.hazard_idx = i;
        std::uint32_t hw = g_hazard_high_water.load(std::memory_order_relaxed);
        while (i + 1U > hw && !g_hazard_high_water.compare_exchange_weak(
                                  hw, i + 1U, std::memory_order_release,
                                  std::memory_order_relaxed)) {
        }
        break;
      }
    }

    g_portable_fast_tls.stripe_base =
        g_portable_slab_base +
        (static_cast<std::size_t>(stripe) << kPortableSlotShift);
    g_portable_fast_tls.hazard_ptr =
        (g_tls_cache.hazard_idx < kMaxHazardRecords)
            ? &g_hazard_records[g_tls_cache.hazard_idx].protected_cb
            : nullptr;
    g_thread_exit_cleaner.armed = true;
  }
  return g_tls_cache.stripe_id;
}

std::uint32_t alloc_portable_slot() noexcept {
  if (g_tls_cache.thread_exited) [[unlikely]] {
    std::uint32_t single = 0U;
    return g_portable_slot_alloc.alloc_from_global(&single, 1U) == 1U ? single
                                                                      : 0U;
  }
  if (g_tls_cache.slot_count == 0U) [[unlikely]] {
    (void)ensure_thread_tls_initialized();
    g_tls_cache.slot_count = g_portable_slot_alloc.alloc_from_global(
        g_tls_cache.slot_ids, PodTlsSlotCache::kRefill);
    if (g_tls_cache.slot_count == 0U) [[unlikely]] {
      return 0U;
    }
  }
  return g_tls_cache.slot_ids[--g_tls_cache.slot_count];
}

void free_portable_slot(std::uint32_t slot) noexcept {
  if (slot == 0U) {
    return;
  }
  if (g_portable_keepalive_ptrs != nullptr) {
    __atomic_store_n(&g_portable_keepalive_ptrs[slot], nullptr,
                     __ATOMIC_RELEASE);
  }
  if (g_tls_cache.thread_exited) [[unlikely]] {
    g_portable_slot_alloc.free_to_global(&slot, 1U);
    return;
  }
  if (g_tls_cache.slot_count == PodTlsSlotCache::kCap) [[unlikely]] {
    (void)ensure_thread_tls_initialized();
    g_portable_slot_alloc.free_to_global(g_tls_cache.slot_ids,
                                         PodTlsSlotCache::kRefill);
    std::memmove(g_tls_cache.slot_ids,
                 g_tls_cache.slot_ids + PodTlsSlotCache::kRefill,
                 (PodTlsSlotCache::kCap - PodTlsSlotCache::kRefill) *
                     sizeof(std::uint32_t));
    g_tls_cache.slot_count -= PodTlsSlotCache::kRefill;
  }
  g_tls_cache.slot_ids[g_tls_cache.slot_count++] = slot;
}

THETA_FORCE_INLINE void lock_cb(PortableSpCountedBase* cb) noexcept {
  while (__atomic_exchange_n(&cb->fast_.reg_lock, 1U, __ATOMIC_ACQUIRE) != 0U) {
    while (__atomic_load_n(&cb->fast_.reg_lock, __ATOMIC_RELAXED) != 0U) {
      ::theta::arch::cpu_relax();
    }
  }
}

THETA_FORCE_INLINE void unlock_cb(PortableSpCountedBase* cb) noexcept {
  __atomic_store_n(&cb->fast_.reg_lock, 0U, __ATOMIC_RELEASE);
}

void append_reg_stripe_locked(PerStripeSlotMeta& meta,
                              std::uint16_t stripe) noexcept {
  const std::uint32_t n = meta.num_reg_stripes;
  if (n < 8U) [[likely]] {
    meta.inline_stripes[n] = stripe;
    meta.num_reg_stripes = n + 1U;
    return;
  }
  const std::uint32_t ext_idx = n - 8U;
  if (ext_idx >= meta.overflow_cap) {
    const std::uint32_t new_cap =
        meta.overflow_cap == 0U
            ? 16U
            : std::min<std::uint32_t>(kPortableNumStripes,
                                      meta.overflow_cap * 4U);
    auto* next_buf = new (std::nothrow) std::uint16_t[new_cap];
    if (next_buf == nullptr) [[unlikely]] {
      return;
    }
    if (meta.overflow_stripes != nullptr) {
      std::memcpy(next_buf, meta.overflow_stripes,
                  ext_idx * sizeof(std::uint16_t));
      delete[] meta.overflow_stripes;
    }
    meta.overflow_stripes = next_buf;
    meta.overflow_cap = new_cap;
  }
  meta.overflow_stripes[ext_idx] = stripe;
  meta.num_reg_stripes = n + 1U;
}

std::uint16_t get_reg_stripe_locked(const PerStripeSlotMeta& meta,
                                    std::uint32_t i) noexcept {
  if (i < 8U) [[likely]] {
    return meta.inline_stripes[i];
  }
  return meta.overflow_stripes[i - 8U];
}

bool try_register_stripe_cell_locked(std::uint32_t slot,
                                     PerStripeSlotMeta& meta,
                                     std::uint16_t stripe,
                                     std::int64_t init_encoded) noexcept {
  std::int64_t* const cell =
      g_portable_slab_base +
      (static_cast<std::size_t>(stripe) << kPortableSlotShift) + slot;
  if (__atomic_load_n(cell, __ATOMIC_RELAXED) != 0) {
    return false;
  }
  append_reg_stripe_locked(meta, stripe);
  __atomic_store_n(cell, init_encoded, __ATOMIC_RELEASE);
  return true;
}

void wait_for_hazard_quiescence(const PortableSpCountedBase* old_cb,
                                const PerStripeSlotMeta& meta) noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const std::uint32_t n = g_hazard_high_water.load(std::memory_order_acquire);
  for (std::uint32_t i = 0U; i < n; ++i) {
    std::uint32_t spins = 0U;
    while (g_hazard_records[i].protected_cb.load(std::memory_order_seq_cst) ==
           static_cast<const void*>(old_cb)) {
      if (meta.pin_count.load(std::memory_order_acquire) > 0U) {
        return;
      }
      if ((++spins & 63U) == 0U) {
        std::this_thread::yield();
      } else {
        ::theta::arch::cpu_relax();
      }
    }
  }
}

struct UnpromotedState {
  std::uint32_t raw;
  std::uint32_t copies;
  std::uint32_t extra_refs;
  std::uint32_t theta_refs;
};

UnpromotedState seal_unpromoted_locked(PortableSpCountedBase* cb) noexcept {
  const std::uint32_t raw =
      __atomic_fetch_or(&cb->fast_.slot_id, kWarmupSealedBit, __ATOMIC_ACQ_REL);
  if ((raw & kWarmupActiveBit) != 0U) {
    const std::uint32_t copies = (raw >> 16U) & 0x3FFFU;
    const std::uint32_t extra = raw & 0xFFFFU;
    return {raw, copies, extra, 1U + extra};
  }
  return {raw, 0U, 0U, 1U};
}

void unseal_unpromoted_locked(PortableSpCountedBase* cb, std::uint32_t copies,
                              std::uint32_t extra_refs) noexcept {
  const std::uint32_t word =
      (copies == 0U && extra_refs == 0U)
          ? 0U
          : (kWarmupActiveBit | ((copies & 0x3FFFU) << 16U) |
             (extra_refs & 0xFFFFU));
  __atomic_store_n(&cb->fast_.slot_id, word, __ATOMIC_RELEASE);
}

void retire_unpromoted_locked(PortableSpCountedBase* cb,
                              std::uint32_t add_use_count) noexcept {
  for (std::uint32_t i = 0U; i < add_use_count; ++i) {
    cb->_M_add_ref_copy();
  }
  __atomic_store_n(&cb->fast_.slot_id, kWarmupSealedBit, __ATOMIC_RELAXED);
  __atomic_store_n(&cb->fast_.mode, kModeAtomicRetired, __ATOMIC_RELEASE);
}

bool promote_to_striped_locked(PortableSpCountedBase* cb,
                               std::int64_t initial_tickets,
                               std::int64_t first_delta) noexcept {
  const std::uint16_t stripe = ensure_thread_tls_initialized();
  if (!g_portable_slab_ok) {
    return false;
  }
  const std::uint32_t slot = alloc_portable_slot();
  if (slot == 0U) [[unlikely]] {
    return false;
  }

  cb->_M_weak_add_ref();

  PerStripeSlotMeta& meta = g_portable_slot_meta[slot];
  meta.central_tickets.store(initial_tickets, std::memory_order_relaxed);
  meta.last_harvest_ms = 0ULL;
  meta.rapid_harvest_streak = 0U;
  meta.num_reg_stripes = 0U;
  meta.pin_count.store(0U, std::memory_order_relaxed);
  meta.atomic_pin_reads.store(0U, std::memory_order_relaxed);
  meta.quiescing_unpins = 0U;
  meta.keepalive_ptr_set = 0U;
  __atomic_store_n(&cb->fast_.slot_id, slot, __ATOMIC_RELEASE);

  if (!try_register_stripe_cell_locked(slot, meta, stripe, 1 + first_delta) &&
      first_delta != 0) [[unlikely]] {
    meta.central_tickets.fetch_add(first_delta, std::memory_order_relaxed);
  }

  __atomic_store_n(&cb->fast_.mode, kModePerCpuActive, __ATOMIC_RELEASE);
  __atomic_store_n(&cb->fast_.fast_active, 1U, __ATOMIC_RELEASE);
  return true;
}

alignas(32) const unsigned char g_keepalive_unpinned_sentinel_bytes[sizeof(
    PortableSpCountedBase)]{};

struct alignas(64) AtomicPtrShard {
  std::atomic<std::uint8_t> lock{0};
  std::atomic<std::uint32_t> seq{0};
  std::atomic<std::uint32_t> waiters{0};
};

constexpr std::size_t kAtomicPtrShards = 64U;
constinit AtomicPtrShard g_atomic_ptr_shards[kAtomicPtrShards]{};

AtomicPtrShard& shard_for_addr(const void* addr) noexcept {
  std::uintptr_t v = reinterpret_cast<std::uintptr_t>(addr);
  v = (v >> 4U) ^ (v >> 10U) ^ (v >> 16U);
  return g_atomic_ptr_shards[v & (kAtomicPtrShards - 1U)];
}

}  // namespace

[[gnu::aligned(64)]] void PortableSpCountedBase::_M_dispose() noexcept {
  const std::uintptr_t tag = ops_or_size_;
  if ((tag & 1U) != 0U) {
    return;
  }
  const auto* ops = reinterpret_cast<const PortableColdOps*>(tag);
  if (ops != nullptr && ops->dispose != nullptr) {
    ops->dispose(this);
  }
}

[[gnu::aligned(64)]] void PortableSpCountedBase::_M_destroy() noexcept {
  const std::uint8_t m = __atomic_load_n(&fast_.mode, __ATOMIC_RELAXED);
  const std::uint32_t raw_slot =
      __atomic_load_n(&fast_.slot_id, __ATOMIC_RELAXED);
  const std::uint32_t slot = raw_slot & ~kWarmupTagMask;
  if (m != kModeUnpromoted && slot != 0U && slot < kMaxPortableSlots)
      [[unlikely]] {
    __atomic_store_n(&fast_.slot_id, 0U, __ATOMIC_RELAXED);
    free_portable_slot(slot);
  }
  const std::uintptr_t tag = ops_or_size_;
  if ((tag & 1U) != 0U) {
    ::operator delete(static_cast<void*>(this), tag >> 1U);
    return;
  }
  const auto* ops = reinterpret_cast<const PortableColdOps*>(tag);
  if (ops != nullptr && ops->destroy != nullptr) {
    ops->destroy(this);
  } else {
    ::operator delete(static_cast<void*>(this), sizeof(PortableSpCountedBase));
  }
}

[[gnu::aligned(64)]] void* PortableSpCountedBase::_M_get_deleter(
    const std::type_info& ti) noexcept {
  const std::uintptr_t tag = ops_or_size_;
  if ((tag & 1U) != 0U) {
    return nullptr;
  }
  const auto* ops = reinterpret_cast<const PortableColdOps*>(tag);
  if (ops != nullptr && ops->get_deleter != nullptr) {
    return ops->get_deleter(this, ti);
  }
  return nullptr;
}

[[gnu::aligned(64)]] void record_portable_sp_vptr(
    const PortableSpCountedBase* sample) noexcept {
  const void* vptr = nullptr;
  std::memcpy(&vptr, static_cast<const void*>(sample), sizeof(vptr));
  __atomic_store_n(&g_portable_sp_vptr, vptr, __ATOMIC_RELEASE);
}

[[gnu::aligned(64)]] void sp_add_ref_slow(StdSpCountedBase* pi) noexcept {
  if (!is_portable_sp(pi)) {
    pi->_M_add_ref_copy();
    return;
  }
  auto* cb = static_cast<PortableSpCountedBase*>(pi);
  if (__atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) ==
      kModeAtomicRetired) {
    pi->_M_add_ref_copy();
    return;
  }

  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode == kModeUnpromoted) {
    const UnpromotedState st = seal_unpromoted_locked(cb);
    const std::uint32_t next_copies = st.copies + 1U;
    if (next_copies < kPromotionWarmupCopies && st.extra_refs < 0xFFFDU)
        [[likely]] {
      unseal_unpromoted_locked(cb, next_copies, st.extra_refs + 1U);
      unlock_cb(cb);
      return;
    }
    if (promote_to_striped_locked(cb, static_cast<std::int64_t>(st.theta_refs),
                                  /*first_delta=*/1)) {
      unlock_cb(cb);
      return;
    }
    retire_unpromoted_locked(cb, st.extra_refs + 1U);
    unlock_cb(cb);
    return;
  }

  if (mode == kModePerCpuActive) {
    const std::uint16_t stripe = ensure_thread_tls_initialized();
    const std::uint32_t slot = cb->fast_.slot_id;
    PerStripeSlotMeta& meta = g_portable_slot_meta[slot];
    std::int64_t drained = 0;
    if (PortableBackend::try_drain_local_surplus(&cb->fast_, drained)) {
      meta.central_tickets.fetch_add(drained + 1, std::memory_order_release);
      unlock_cb(cb);
      return;
    }
    if (!g_tls_cache.stripe_assigned ||
        !try_register_stripe_cell_locked(slot, meta, stripe,
                                         /*init_encoded=*/2)) {
      meta.central_tickets.fetch_add(1, std::memory_order_relaxed);
    }
    unlock_cb(cb);
    return;
  }

  unlock_cb(cb);
  pi->_M_add_ref_copy();
}

[[gnu::aligned(64)]] void sp_release_slow(StdSpCountedBase* pi) noexcept {
  if (!is_portable_sp(pi)) {
    pi->_M_release();
    return;
  }
  auto* cb = static_cast<PortableSpCountedBase*>(pi);
  if (__atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) ==
      kModeAtomicRetired) {
    pi->_M_release();
    return;
  }

  if (__atomic_load_n(&cb->fast_.fast_active, __ATOMIC_ACQUIRE) != 0U)
      [[likely]] {
    const std::uint32_t slot =
        __atomic_load_n(&cb->fast_.slot_id, __ATOMIC_ACQUIRE);
    if (slot != 0U && slot < kMaxPortableSlots) [[likely]] {
      auto& tickets = g_portable_slot_meta[slot].central_tickets;
      std::int64_t cur_tickets = tickets.load(std::memory_order_relaxed);
      while (cur_tickets >= 2) [[likely]] {
        if (tickets.compare_exchange_weak(
                cur_tickets, cur_tickets - 1, std::memory_order_acq_rel,
                std::memory_order_relaxed)) [[likely]] {
          return;
        }
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
    const UnpromotedState st = seal_unpromoted_locked(cb);
    if (st.extra_refs > 0U) [[unlikely]] {
      unseal_unpromoted_locked(cb, st.copies, st.extra_refs - 1U);
      unlock_cb(cb);
      return;
    }
    if (__atomic_load_n(&cb->fast_.has_weak_or_std, __ATOMIC_ACQUIRE) == 0U)
        [[likely]] {
      __atomic_store_n(&cb->fast_.slot_id, 0U, __ATOMIC_RELAXED);
      unlock_cb(cb);
      const std::uintptr_t tag = cb->ops_or_size_;
      if ((tag & 1U) != 0U) [[likely]] {
        ::operator delete(static_cast<void*>(cb), tag >> 1U);
      } else {
        reinterpret_cast<const PortableColdOps*>(tag)->dispose_and_destroy(cb);
      }
      return;
    }
    retire_unpromoted_locked(cb, 0U);
    unlock_cb(cb);
    pi->_M_release();
    return;
  }

  const std::uint32_t slot = cb->fast_.slot_id;
  PerStripeSlotMeta& meta = g_portable_slot_meta[slot];

  const std::int64_t rem =
      meta.central_tickets.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (rem > 0) [[likely]] {
    unlock_cb(cb);
    return;
  }

  // Atomically close and drain every registered stripe via `exchange(cell, 0)`.
  // Any concurrent `CAS(cell, cur >= 1, ...)` either ordered before this
  // `exchange` (included in `v`) or sees `0` and falls back to `lock_cb(cb)`.
  __atomic_store_n(&cb->fast_.fast_active, 0U, __ATOMIC_RELEASE);

  const std::uint32_t n_stripes = meta.num_reg_stripes;
  std::int64_t delta_sum = 0;
#pragma clang loop vectorize(disable) unroll(disable)
  for (std::uint32_t i = 0U; i < n_stripes; ++i) {
    const std::uint16_t s = get_reg_stripe_locked(meta, i);
    std::int64_t* const cell =
        g_portable_slab_base +
        (static_cast<std::size_t>(s) << kPortableSlotShift) + slot;
    const std::int64_t v = __atomic_exchange_n(cell, 0, __ATOMIC_ACQUIRE);
    if (v >= 1) {
      delta_sum += (v - 1);
    }
  }

  const std::int64_t h_after = rem + delta_sum;
  const std::uint64_t now_ms = coarse_monotonic_ms();
  if (meta.last_harvest_ms != 0ULL && now_ms == meta.last_harvest_ms) {
    ++meta.rapid_harvest_streak;
  } else {
    meta.rapid_harvest_streak = 1U;
  }
  meta.last_harvest_ms = (now_ms != 0ULL) ? now_ms : 1ULL;

  if (h_after == 0) {
    meta.num_reg_stripes = 0U;
    meta.central_tickets.store(0, std::memory_order_relaxed);
    if (__atomic_load_n(&cb->fast_.has_weak_or_std, __ATOMIC_ACQUIRE) == 0U) {
      __atomic_store_n(&cb->fast_.slot_id, kWarmupSealedBit, __ATOMIC_RELAXED);
      free_portable_slot(slot);
    } else {
      __atomic_store_n(&cb->fast_.slot_id, slot | kWarmupSealedBit,
                       __ATOMIC_RELAXED);
    }
    __atomic_store_n(&cb->fast_.mode, kModeAtomicRetired, __ATOMIC_RELEASE);
    unlock_cb(cb);

    cb->_M_weak_release();
    cb->_M_release();
    return;
  }

  if (meta.pin_count.load(std::memory_order_relaxed) > 0U ||
      meta.quiescing_unpins > 0U || meta.rapid_harvest_streak <= 64U) {
#pragma clang loop vectorize(disable) unroll(disable)
    for (std::uint32_t i = 0U; i < n_stripes; ++i) {
      const std::uint16_t s = get_reg_stripe_locked(meta, i);
      std::int64_t* const cell =
          g_portable_slab_base +
          (static_cast<std::size_t>(s) << kPortableSlotShift) + slot;
      __atomic_store_n(cell, 1, __ATOMIC_RELEASE);
    }
    meta.central_tickets.store(h_after, std::memory_order_relaxed);
    __atomic_store_n(&cb->fast_.fast_active, 1U, __ATOMIC_RELEASE);
    unlock_cb(cb);
    return;
  }

  meta.num_reg_stripes = 0U;
  meta.central_tickets.store(0, std::memory_order_relaxed);
  for (std::int64_t i = 0; i < h_after - 1; ++i) {
    cb->_M_add_ref_copy();
  }
  __atomic_store_n(&cb->fast_.slot_id, slot | kWarmupSealedBit,
                   __ATOMIC_RELAXED);
  __atomic_store_n(&cb->fast_.mode, kModeAtomicRetired, __ATOMIC_RELEASE);
  unlock_cb(cb);
  cb->_M_weak_release();
}

[[gnu::aligned(64)]] long sp_use_count_slow(
    const StdSpCountedBase* pi) noexcept {
  if (pi == nullptr) {
    return 0L;
  }
  if (!is_portable_sp(pi)) {
    return pi->_M_get_use_count();
  }
  auto* cb = const_cast<PortableSpCountedBase*>(
      static_cast<const PortableSpCountedBase*>(pi));
  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode == kModeUnpromoted) {
    const std::uint32_t raw =
        __atomic_load_n(&cb->fast_.slot_id, __ATOMIC_ACQUIRE);
    const long extra_refs = ((raw & kWarmupActiveBit) != 0U)
                                ? static_cast<long>(raw & 0xFFFFU)
                                : 0L;
    const long cnt = pi->_M_get_use_count();
    unlock_cb(cb);
    return cnt > 0L ? (cnt + extra_refs) : 0L;
  }
  if (mode != kModePerCpuActive) {
    const long cnt = pi->_M_get_use_count();
    unlock_cb(cb);
    return cnt;
  }
  const std::uint32_t slot = cb->fast_.slot_id;
  const PerStripeSlotMeta& meta = g_portable_slot_meta[slot];
  const long legacy_plus_lease = pi->_M_get_use_count();
  const long legacy_refs =
      legacy_plus_lease > 0L ? (legacy_plus_lease - 1L) : 0L;
  const std::int64_t active_cushions =
      (meta.pin_count.load(std::memory_order_relaxed) > 0U ? 1 : 0) +
      static_cast<std::int64_t>(meta.quiescing_unpins);
  std::int64_t total = legacy_refs +
                       meta.central_tickets.load(std::memory_order_relaxed) -
                       active_cushions * kPinnedTicketCushion;
  const std::uint32_t n_stripes = meta.num_reg_stripes;
#pragma clang loop vectorize(disable) unroll(disable)
  for (std::uint32_t i = 0U; i < n_stripes; ++i) {
    const std::uint16_t s = get_reg_stripe_locked(meta, i);
    const std::int64_t* const cell =
        g_portable_slab_base +
        (static_cast<std::size_t>(s) << kPortableSlotShift) + slot;
    const std::int64_t v = __atomic_load_n(cell, __ATOMIC_RELAXED);
    if (v >= 1) {
      total += (v - 1);
    }
  }
  unlock_cb(cb);
  return total > 0 ? static_cast<long>(total) : 0L;
}

[[gnu::aligned(64)]] void sp_note_to_std_copy(StdSpCountedBase* pi) noexcept {
  if (!is_portable_sp(pi)) {
    pi->_M_add_ref_copy();
    return;
  }
  auto* cb = static_cast<PortableSpCountedBase*>(pi);
  __atomic_store_n(&cb->fast_.has_weak_or_std, 1U, __ATOMIC_RELEASE);
  const std::uint8_t pre_mode =
      __atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE);
  if (pre_mode == kModeUnpromoted) {
    lock_cb(cb);
    if (cb->fast_.mode == kModeUnpromoted) {
      const UnpromotedState st = seal_unpromoted_locked(cb);
      if (!promote_to_striped_locked(cb,
                                     static_cast<std::int64_t>(st.theta_refs),
                                     /*first_delta=*/0)) {
        retire_unpromoted_locked(cb, st.extra_refs);
      }
    }
    unlock_cb(cb);
  }
  pi->_M_add_ref_copy();
}

[[gnu::aligned(64)]] void sp_note_to_std_move(StdSpCountedBase* pi) noexcept {
  if (!is_portable_sp(pi)) {
    return;
  }
  auto* cb = static_cast<PortableSpCountedBase*>(pi);
  __atomic_store_n(&cb->fast_.has_weak_or_std, 1U, __ATOMIC_RELEASE);
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
    const UnpromotedState st = seal_unpromoted_locked(cb);
    retire_unpromoted_locked(cb, st.extra_refs);
    unlock_cb(cb);
    return;
  }
  cb->_M_add_ref_copy();
  unlock_cb(cb);
  sp_release(pi);
}

[[gnu::aligned(64)]] void sp_add_ref_from_std(StdSpCountedBase* pi) noexcept {
  sp_add_ref(pi);
}

[[gnu::aligned(64)]] void sp_adopt_from_std_move(
    StdSpCountedBase* pi) noexcept {
  if (!is_portable_sp(pi)) {
    return;
  }
  auto* cb = static_cast<PortableSpCountedBase*>(pi);
  if (__atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) ==
      kModeAtomicRetired) {
    return;
  }
  lock_cb(cb);
  if (cb->fast_.mode == kModePerCpuActive) {
    g_portable_slot_meta[cb->fast_.slot_id].central_tickets.fetch_add(
        1, std::memory_order_relaxed);
    unlock_cb(cb);
    pi->_M_release();
    return;
  }
  if (cb->fast_.mode == kModeUnpromoted) {
    const UnpromotedState st = seal_unpromoted_locked(cb);
    retire_unpromoted_locked(cb, st.extra_refs);
  }
  unlock_cb(cb);
}

[[gnu::aligned(64)]] bool sp_weak_lock_slow(StdSpCountedBase* pi) noexcept {
  if (!is_portable_sp(pi)) {
    return pi->_M_add_ref_lock_nothrow();
  }
  auto* cb = static_cast<PortableSpCountedBase*>(pi);
  lock_cb(cb);
  const std::uint8_t mode = cb->fast_.mode;
  if (mode == kModeAtomicRetired) {
    unlock_cb(cb);
    return pi->_M_add_ref_lock_nothrow();
  }
  if (mode == kModeUnpromoted) {
    const UnpromotedState st = seal_unpromoted_locked(cb);
    const std::uint32_t next_copies = st.copies + 1U;
    if (next_copies < kPromotionWarmupCopies && st.extra_refs < 0xFFFDU)
        [[likely]] {
      unseal_unpromoted_locked(cb, next_copies, st.extra_refs + 1U);
      unlock_cb(cb);
      return true;
    }
    if (!promote_to_striped_locked(
            cb, static_cast<std::int64_t>(st.theta_refs + 1U),
            /*first_delta=*/0)) {
      retire_unpromoted_locked(cb, st.extra_refs + 1U);
    }
    unlock_cb(cb);
    return true;
  }

  g_portable_slot_meta[cb->fast_.slot_id].central_tickets.fetch_add(
      1, std::memory_order_relaxed);
  unlock_cb(cb);
  return true;
}

[[gnu::aligned(64)]] PortableSpCountedBase*
sp_keepalive_unpinned_sentinel() noexcept {
  return const_cast<PortableSpCountedBase*>(
      reinterpret_cast<const PortableSpCountedBase*>(
          g_keepalive_unpinned_sentinel_bytes));
}

[[gnu::aligned(64)]] SpPinResult sp_pin_for_keepalive(
    PortableSpCountedBase* cb, void* managed_ptr) noexcept {
  const std::uint16_t stripe = ensure_thread_tls_initialized();
  if (!g_portable_slab_ok) [[unlikely]] {
    return SpPinResult::kThrottledStorm;
  }

  lock_cb(cb);
  if (cb->fast_.mode == kModeUnpromoted) {
    const UnpromotedState st = seal_unpromoted_locked(cb);
    if (!promote_to_striped_locked(cb, static_cast<std::int64_t>(st.theta_refs),
                                   /*first_delta=*/0)) {
      retire_unpromoted_locked(cb, st.extra_refs);
      unlock_cb(cb);
      return SpPinResult::kThrottledStorm;
    }
  } else if (cb->fast_.mode == kModePerCpuActive) {
    if (g_tls_cache.stripe_assigned) {
      const std::uint32_t slot = cb->fast_.slot_id;
      (void)try_register_stripe_cell_locked(slot, g_portable_slot_meta[slot],
                                            stripe, /*init_encoded=*/1);
    }
  } else {
    unlock_cb(cb);
    return SpPinResult::kAliasConflict;
  }

  const std::uint32_t slot = cb->fast_.slot_id;
  PerStripeSlotMeta& meta = g_portable_slot_meta[slot];
  void* existing =
      __atomic_load_n(&g_portable_keepalive_ptrs[slot], __ATOMIC_RELAXED);
  if (meta.keepalive_ptr_set != 0U && existing != managed_ptr) [[unlikely]] {
    unlock_cb(cb);
    return SpPinResult::kAliasConflict;
  }
  __atomic_store_n(&g_portable_keepalive_ptrs[slot], managed_ptr,
                   __ATOMIC_RELEASE);
  meta.keepalive_ptr_set = 1U;
  const std::uint32_t prev_pins =
      meta.pin_count.load(std::memory_order_relaxed);
  if (prev_pins == 0U) {
    meta.central_tickets.fetch_add(kPinnedTicketCushion,
                                   std::memory_order_relaxed);
  }
  meta.pin_count.store(prev_pins + 1U, std::memory_order_release);
  unlock_cb(cb);
  return SpPinResult::kPinned;
}

[[gnu::aligned(64)]] bool sp_try_pin_atomic_fast(PortableSpCountedBase* cb,
                                                 void* managed_ptr,
                                                 bool& out_never_pin) noexcept {
  out_never_pin = false;
  if (managed_ptr == nullptr) [[unlikely]] {
    return false;
  }
  const std::uint16_t stripe = ensure_thread_tls_initialized();
  if (!g_portable_slab_ok) [[unlikely]] {
    return false;
  }
  const std::uint32_t slot =
      __atomic_load_n(&cb->fast_.slot_id, __ATOMIC_ACQUIRE);
  if (slot == 0U || slot >= kMaxPortableSlots ||
      g_portable_slot_meta == nullptr) [[unlikely]] {
    return false;
  }
  const std::uint32_t reads =
      g_portable_slot_meta[slot].atomic_pin_reads.fetch_add(
          1U, std::memory_order_relaxed) +
      1U;
  if (reads < kAtomicPinWarmupReads) {
    return false;
  }

  lock_cb(cb);
  if (cb->fast_.mode != kModePerCpuActive || cb->fast_.slot_id != slot) {
    unlock_cb(cb);
    out_never_pin = (cb->fast_.mode == kModeAtomicRetired);
    return false;
  }
  PerStripeSlotMeta& meta = g_portable_slot_meta[slot];
  void* existing =
      __atomic_load_n(&g_portable_keepalive_ptrs[slot], __ATOMIC_RELAXED);
  if (meta.keepalive_ptr_set != 0U && existing != managed_ptr) [[unlikely]] {
    unlock_cb(cb);
    out_never_pin = true;
    return false;
  }
  __atomic_store_n(&g_portable_keepalive_ptrs[slot], managed_ptr,
                   __ATOMIC_RELEASE);
  meta.keepalive_ptr_set = 1U;
  const std::uint32_t prev_pins =
      meta.pin_count.load(std::memory_order_relaxed);
  if (prev_pins == 0U) {
    meta.central_tickets.fetch_add(kPinnedTicketCushion,
                                   std::memory_order_relaxed);
  }
  meta.pin_count.store(prev_pins + 1U, std::memory_order_release);
  if (g_tls_cache.stripe_assigned) {
    (void)try_register_stripe_cell_locked(slot, meta, stripe,
                                          /*init_encoded=*/1);
  }
  unlock_cb(cb);
  return true;
}

[[gnu::aligned(64)]] void* sp_acquire_pinned_under_hazard_slow(
    PortableSpCountedBase* cb, std::atomic<const void*>* hazard_ptr) noexcept {
  const std::uint16_t stripe = ensure_thread_tls_initialized();
  lock_cb(cb);
  void* managed = nullptr;
  if (cb->fast_.mode == kModePerCpuActive) [[likely]] {
    const std::uint32_t slot = cb->fast_.slot_id;
    managed =
        __atomic_load_n(&g_portable_keepalive_ptrs[slot], __ATOMIC_RELAXED);
    PerStripeSlotMeta& meta = g_portable_slot_meta[slot];
    std::int64_t drained = 0;
    if (PortableBackend::try_drain_local_surplus(&cb->fast_, drained)) {
      meta.central_tickets.fetch_add(drained + 1, std::memory_order_release);
    } else if (!g_tls_cache.stripe_assigned ||
               !try_register_stripe_cell_locked(slot, meta, stripe,
                                                /*init_encoded=*/2)) {
      meta.central_tickets.fetch_add(1, std::memory_order_relaxed);
    }
  } else {
    const std::uint32_t slot = cb->fast_.slot_id & ~kWarmupTagMask;
    if (slot != 0U && slot < kMaxPortableSlots) {
      managed =
          __atomic_load_n(&g_portable_keepalive_ptrs[slot], __ATOMIC_RELAXED);
    }
    cb->_M_add_ref_copy();
  }
  unlock_cb(cb);
  hazard_ptr->store(nullptr, std::memory_order_release);
  return managed;
}

[[gnu::aligned(64)]] void sp_register_keepalive_stripe_slow(
    PortableSpCountedBase* cb) noexcept {
  const std::uint16_t stripe = ensure_thread_tls_initialized();
  if (!g_tls_cache.stripe_assigned ||
      __atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) != kModePerCpuActive) {
    return;
  }
  lock_cb(cb);
  if (cb->fast_.mode == kModePerCpuActive) {
    const std::uint32_t slot = cb->fast_.slot_id;
    (void)try_register_stripe_cell_locked(slot, g_portable_slot_meta[slot],
                                          stripe, /*init_encoded=*/1);
  }
  unlock_cb(cb);
}

[[gnu::aligned(64)]] void sp_unpin_and_retire_old(
    PortableSpCountedBase* old_cb, bool drop_anchor_ref,
    bool quiesce_readers) noexcept {
  lock_cb(old_cb);
  if (old_cb->fast_.mode != kModePerCpuActive) {
    unlock_cb(old_cb);
    if (drop_anchor_ref) {
      sp_release(old_cb);
    }
    return;
  }

  const std::uint32_t slot = old_cb->fast_.slot_id;
  PerStripeSlotMeta& meta = g_portable_slot_meta[slot];

  const std::uint32_t prev_pins =
      meta.pin_count.load(std::memory_order_relaxed);
  if (prev_pins > 1U) {
    meta.pin_count.store(prev_pins - 1U, std::memory_order_release);
    if (drop_anchor_ref) {
      (void)meta.central_tickets.fetch_sub(1, std::memory_order_acq_rel);
    }
    unlock_cb(old_cb);
    return;
  }

  std::int64_t sub = drop_anchor_ref ? 1 : 0;
  if (prev_pins == 1U) {
    meta.pin_count.store(0U, std::memory_order_release);
    sub += kPinnedTicketCushion;
    if (quiesce_readers) {
      ++meta.quiescing_unpins;
      unlock_cb(old_cb);
      wait_for_hazard_quiescence(old_cb, meta);
      lock_cb(old_cb);
      --meta.quiescing_unpins;
    }
  }

  const std::int64_t rem =
      (sub != 0)
          ? (meta.central_tickets.fetch_sub(sub, std::memory_order_acq_rel) -
             sub)
          : meta.central_tickets.load(std::memory_order_acquire);

  if (rem > 0) {
    unlock_cb(old_cb);
    return;
  }

  __atomic_store_n(&old_cb->fast_.fast_active, 0U, __ATOMIC_RELEASE);

  const std::uint32_t n_stripes = meta.num_reg_stripes;
  std::int64_t delta_sum = 0;
#pragma clang loop vectorize(disable) unroll(disable)
  for (std::uint32_t i = 0U; i < n_stripes; ++i) {
    const std::uint16_t s = get_reg_stripe_locked(meta, i);
    std::int64_t* const cell =
        g_portable_slab_base +
        (static_cast<std::size_t>(s) << kPortableSlotShift) + slot;
    const std::int64_t v = __atomic_exchange_n(cell, 0, __ATOMIC_ACQUIRE);
    if (v >= 1) {
      delta_sum += (v - 1);
    }
  }

  const std::int64_t h_after = rem + delta_sum;
  if (h_after <= 0 || (meta.pin_count.load(std::memory_order_relaxed) == 0U &&
                       meta.quiescing_unpins == 0U)) {
    meta.num_reg_stripes = 0U;
    meta.central_tickets.store(0, std::memory_order_relaxed);
    for (std::int64_t i = 0; i < h_after - 1; ++i) {
      old_cb->_M_add_ref_copy();
    }
    if (h_after <= 0 && __atomic_load_n(&old_cb->fast_.has_weak_or_std,
                                        __ATOMIC_ACQUIRE) == 0U) {
      __atomic_store_n(&old_cb->fast_.slot_id, kWarmupSealedBit,
                       __ATOMIC_RELAXED);
      free_portable_slot(slot);
    } else {
      __atomic_store_n(&old_cb->fast_.slot_id, slot | kWarmupSealedBit,
                       __ATOMIC_RELAXED);
    }
    __atomic_store_n(&old_cb->fast_.mode, kModeAtomicRetired, __ATOMIC_RELEASE);
    unlock_cb(old_cb);

    old_cb->_M_weak_release();
    if (h_after <= 0) {
      old_cb->_M_release();
    }
    return;
  }

#pragma clang loop vectorize(disable) unroll(disable)
  for (std::uint32_t i = 0U; i < n_stripes; ++i) {
    const std::uint16_t s = get_reg_stripe_locked(meta, i);
    std::int64_t* const cell =
        g_portable_slab_base +
        (static_cast<std::size_t>(s) << kPortableSlotShift) + slot;
    __atomic_store_n(cell, 1, __ATOMIC_RELEASE);
  }
  meta.central_tickets.store(h_after, std::memory_order_relaxed);
  __atomic_store_n(&old_cb->fast_.fast_active, 1U, __ATOMIC_RELEASE);
  unlock_cb(old_cb);
}

[[gnu::aligned(64)]] void atomic_ptr_lock(const void* addr) noexcept {
  AtomicPtrShard& s = shard_for_addr(addr);
  while (s.lock.exchange(1U, std::memory_order_acquire) != 0U) {
    while (s.lock.load(std::memory_order_relaxed) != 0U) {
      ::theta::arch::cpu_relax();
    }
  }
}

[[gnu::aligned(64)]] void atomic_ptr_unlock(const void* addr) noexcept {
  AtomicPtrShard& s = shard_for_addr(addr);
  s.lock.store(0U, std::memory_order_release);
}

[[gnu::aligned(64)]] void atomic_ptr_unlock_and_notify(
    const void* addr, bool notify_all) noexcept {
  AtomicPtrShard& s = shard_for_addr(addr);
  s.seq.fetch_add(1U, std::memory_order_relaxed);
  const bool has_waiters = s.waiters.load(std::memory_order_acquire) != 0U;
  s.lock.store(0U, std::memory_order_release);
  if (has_waiters) [[unlikely]] {
    if (notify_all) {
      s.seq.notify_all();
    } else {
      s.seq.notify_one();
    }
  }
}

[[gnu::aligned(64)]] std::uint32_t atomic_ptr_seq(const void* addr) noexcept {
  return shard_for_addr(addr).seq.load(std::memory_order_relaxed);
}

[[gnu::aligned(64)]] void atomic_ptr_wait_seq(
    const void* addr, std::uint32_t expected_seq) noexcept {
  AtomicPtrShard& s = shard_for_addr(addr);
  s.waiters.fetch_add(1U, std::memory_order_acq_rel);
  s.seq.wait(expected_seq, std::memory_order_acquire);
  s.waiters.fetch_sub(1U, std::memory_order_release);
}

}  // namespace theta::portable::detail::shared_ptr
