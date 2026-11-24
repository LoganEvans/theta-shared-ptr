// FILENAME: include/theta/hyper_shared_ptr/detail/common.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <theta/arch.hpp>
#include <type_traits>
#include <typeinfo>
#include <utility>

#if defined(__GLIBCXX__)
#define THETA_HSP_HAS_GLIBCXX_ABI 1
#else
#define THETA_HSP_HAS_GLIBCXX_ABI 0
#endif

#if THETA_HAS_LINUX_RSEQ && (THETA_ARCH_X86_64 || THETA_ARCH_AARCH64) && \
    !THETA_TSAN_ENABLED
#define THETA_HSP_USE_RSEQ_ASM 1
#else
#define THETA_HSP_USE_RSEQ_ASM 0
#endif

namespace theta {

inline constexpr std::int32_t kHspMaxCpus = 8192;

namespace hyper_shared_ptr::detail {

// Under sanitizers, keep virtual `MAP_NORESERVE` regions compact (`4,096` slots
// * `8,192` CPUs * `8B` = `256 MiB` VA) so TSan shadow mappings stay tiny while
// still testing all `8,192` CPU indices; in normal builds, expose `131,072`
// slots (`8 GiB` uncommitted VA, `0 B` RSS on unused CPUs/pages).
inline constexpr int kSlotShift =
    (::theta::arch::kTsanEnabled || ::theta::arch::kAsanEnabled) ? 12 : 17;
inline constexpr std::uint32_t kMaxStripedSlots = (1U << kSlotShift);

inline constexpr std::uint8_t kModeUnpromoted = 0U;
inline constexpr std::uint8_t kModePerCpuActive = 1U;
inline constexpr std::uint8_t kModeAtomicRetired = 2U;

// Standard-layout 8-byte hot header embedded inside `ThetaSpCountedBase` so
// inline `rseq` assembly accesses `slot_id` (+0) and `fast_active` (+4) with
// compile-time `offsetof` verification, while `ThetaSpCountedBase` itself fits
// in just 32 bytes (`40B` total with an inplace `uint64_t`).
struct SpFastHeader {
  std::uint32_t slot_id{0};
  std::uint8_t fast_active{0};
  std::uint8_t mode{kModeUnpromoted};
  std::uint8_t has_weak_or_std{0};
  std::uint8_t reg_lock{0};
};

static_assert(std::is_standard_layout_v<SpFastHeader>);
static_assert(sizeof(SpFastHeader) == 8);
static_assert(offsetof(SpFastHeader, slot_id) == 0);
static_assert(offsetof(SpFastHeader, fast_active) == 4);
static_assert(offsetof(SpFastHeader, mode) == 5);
static_assert(offsetof(SpFastHeader, has_weak_or_std) == 6);
static_assert(offsetof(SpFastHeader, reg_lock) == 7);

#if THETA_HSP_HAS_GLIBCXX_ABI
using StdSpCountedBase = std::_Sp_counted_base<__gnu_cxx::_S_atomic>;
#else
class StdSpCountedBase {
 public:
  StdSpCountedBase() noexcept = default;
  virtual ~StdSpCountedBase() noexcept = default;
  virtual void _M_dispose() noexcept = 0;
  virtual void _M_destroy() noexcept { delete this; }
  virtual void* _M_get_deleter(const std::type_info&) noexcept = 0;

  void _M_add_ref_copy() noexcept {
    use_count_.fetch_add(1, std::memory_order_relaxed);
  }
  bool _M_add_ref_lock_nothrow() noexcept {
    int c = use_count_.load(std::memory_order_relaxed);
    while (c != 0) {
      if (use_count_.compare_exchange_weak(c, c + 1, std::memory_order_acq_rel,
                                           std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }
  void _M_release() noexcept {
    if (use_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      _M_dispose();
      _M_weak_release();
    }
  }
  void _M_weak_add_ref() noexcept {
    weak_count_.fetch_add(1, std::memory_order_relaxed);
  }
  void _M_weak_release() noexcept {
    if (weak_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      _M_destroy();
    }
  }
  long _M_get_use_count() const noexcept {
    return use_count_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<int> use_count_{1};
  std::atomic<int> weak_count_{1};
};
#endif

// ============================================================================
// 1. `ThetaSpCountedBase`: Non-template `final` control block deriving directly
//    from `std::_Sp_counted_base<_S_atomic>` for 100% binary `std::shared_ptr`
//    and `std::weak_ptr` ABI compatibility across all `T`, `Deleter`, `Alloc`.
//
//    Because `ThetaSpCountedBase` is a single non-template `final` class, every
//    instance in the process shares the exact same `vptr` (`g_theta_sp_vptr`),
//    allowing O(1) 1-instruction identification without RTTI or pointer tags.
// ============================================================================
class ThetaSpCountedBase;
extern const void* g_theta_sp_vptr;
void record_theta_sp_vptr(const ThetaSpCountedBase* sample) noexcept;

struct alignas(8) SpColdOps {
  void (*dispose)(ThetaSpCountedBase*) noexcept;
  void (*destroy)(ThetaSpCountedBase*) noexcept;
  void (*dispose_and_destroy)(ThetaSpCountedBase*) noexcept;
  void* (*get_deleter)(ThetaSpCountedBase*, const std::type_info&) noexcept;
};

// Compact 32-byte control block (`vptr` [8B] + `_M_use_count/_M_weak_count`
// [8B] + `SpFastHeader` [8B] + `ops_or_size_` [8B]).
// Cold Per-CPU coordination state (`central_tickets`, `inline_cpus`,
// `overflow_cpus`, `harvest_count`) lives out-of-line in `g_rseq_slot_meta`
// indexed by `fast_.slot_id` and is touched only if the object is promoted.
class ThetaSpCountedBase final : public StdSpCountedBase {
 public:
  explicit ThetaSpCountedBase(std::uintptr_t ops_or_size) noexcept
      : ops_or_size_(ops_or_size) {
#if !(defined(__GXX_ABI_VERSION) && !THETA_ASAN_ENABLED)
    if (__atomic_load_n(&g_theta_sp_vptr, __ATOMIC_RELAXED) == nullptr)
        [[unlikely]] {
      record_theta_sp_vptr(this);
    }
#endif
  }

  ~ThetaSpCountedBase() noexcept override = default;

  // Key function defined in `src/hyper_shared_ptr.cpp` to anchor a single
  // canonical vtable address across the entire process.
  void _M_dispose() noexcept override;
  void _M_destroy() noexcept override;
  void* _M_get_deleter(const std::type_info& ti) noexcept override;

  SpFastHeader fast_{};
  // LSB == 1: `(sizeof(SpInplaceStorage<T>) << 1) | 1` for zero-indirect-call
  //           sized `::operator delete` when `T` is trivially destructible and
  //           standard-aligned.
  // LSB == 0: `const SpColdOps*` descriptor in `.rodata`.
  std::uintptr_t ops_or_size_{0};
};

static_assert(sizeof(ThetaSpCountedBase) == 4 * sizeof(void*));

#if defined(__GXX_ABI_VERSION) && !THETA_ASAN_ENABLED
extern const void* const kThetaSpVtableSym[] asm(
    "_ZTVN5theta16hyper_shared_ptr6detail18ThetaSpCountedBaseE");
#endif

// Global runtime pointers exposed for zero-indirection `rseq` inline fast paths
extern const void* g_theta_sp_vptr;
extern std::int64_t* g_rseq_slab_base;
extern void** g_rseq_keepalive_ptrs;
extern std::atomic<std::uint64_t>* g_hsp_slab_base;

void record_theta_sp_vptr(const ThetaSpCountedBase* sample) noexcept;

[[nodiscard]] std::int32_t current_cpu_id() noexcept;
[[nodiscard]] std::int32_t max_possible_cpus() noexcept;

// Cold slow paths for ABI-compatible `ThetaSpCountedBase`
void sp_add_ref_slow(StdSpCountedBase* pi) noexcept;
void sp_release_slow(StdSpCountedBase* pi) noexcept;
[[nodiscard]] long sp_use_count_slow(const StdSpCountedBase* pi) noexcept;
void sp_note_to_std_copy(StdSpCountedBase* pi) noexcept;
void sp_note_to_std_move(StdSpCountedBase* pi) noexcept;
void sp_add_ref_from_std(StdSpCountedBase* pi) noexcept;
void sp_adopt_from_std_move(StdSpCountedBase* pi) noexcept;
[[nodiscard]] bool sp_weak_lock_slow(StdSpCountedBase* pi) noexcept;

// KeepAlive support on `ThetaSpCountedBase`
[[nodiscard]] bool sp_pin_for_keepalive(ThetaSpCountedBase* cb,
                                        void* managed_ptr) noexcept;
void sp_register_keepalive_cpu_slow(ThetaSpCountedBase* cb) noexcept;
void sp_keepalive_retire_old(ThetaSpCountedBase* old_cb) noexcept;

THETA_FORCE_INLINE bool is_theta_sp(const StdSpCountedBase* pi) noexcept {
  if (pi == nullptr) [[unlikely]] {
    return false;
  }
  const void* vptr = nullptr;
  std::memcpy(&vptr, static_cast<const void*>(pi), sizeof(vptr));
#if defined(__GXX_ABI_VERSION) && !THETA_ASAN_ENABLED
  return vptr == static_cast<const void*>(kThetaSpVtableSym + 2);
#else
  return vptr == __atomic_load_n(&g_theta_sp_vptr, __ATOMIC_RELAXED);
#endif
}

// ============================================================================
// 2. `HyperControlBlock`: Handle-Tagged Per-CPU Atomic Shards (`0 .. 8,191`
//    CPUs) — modernizing the original `HyperSharedPointer` design without `64`-
//    CPU bitmask UB, `yield()` spin races, or `4 KiB` per-object `malloc`s.
//
//    Tag layout inside `std::uintptr_t tagged_`:
//      - `HyperControlBlock` is `alignas(128)` -> bits `0..6` (`7` bits) hold
//        `origin_cpu & 0x7F`.
//      - Canonical 48-bit & 57-bit (LA57 / ARM64 LVA) user pointers always have
//        bits `58..63 == 0` (`6` bits) -> hold `(origin_cpu >> 7) & 0x3F`.
//      - Total = `13` bits = `8,192` CPUs (`0 .. 8,191`), while `kHspPtrMask`
//        (`0x03FFFFFFFFFFFF80ULL`) is a 1-instruction ARM64 logical immediate!
// ============================================================================
inline constexpr std::uintptr_t kHspPtrMask = 0x03FFFFFFFFFFFF80ULL;
inline constexpr std::uintptr_t kHspLowCpuMask = 0x7FULL;

class alignas(128) HyperControlBlock {
 public:
  using DisposeFn = void (*)(HyperControlBlock*) noexcept;
  using DestroyFn = void (*)(HyperControlBlock*) noexcept;

  HyperControlBlock(void* managed_ptr, std::uint32_t slot_id,
                    DisposeFn dispose_fn, DestroyFn destroy_fn) noexcept
      : slot_id_(slot_id),
        managed_ptr_(managed_ptr),
        dispose_fn_(dispose_fn),
        destroy_fn_(destroy_fn) {}

  std::uint32_t slot_id_{0};
  std::atomic<std::uint32_t> active_shards_{1};
  std::atomic<std::uint32_t> weak_refs_{1};
  void* managed_ptr_{nullptr};
  DisposeFn dispose_fn_{nullptr};
  DestroyFn destroy_fn_{nullptr};
};

[[nodiscard]] THETA_FORCE_INLINE std::uintptr_t pack_hsp_tag(
    HyperControlBlock* cb, std::int32_t cpu) noexcept {
  if (cb == nullptr) {
    return 0U;
  }
  const auto ucpu =
      static_cast<std::uintptr_t>((cpu >= 0 && cpu < kHspMaxCpus) ? cpu : 0);
  const std::uintptr_t lo = ucpu & kHspLowCpuMask;
  const std::uintptr_t hi = ((ucpu >> 7U) & 0x3FULL) << 58U;
  return (reinterpret_cast<std::uintptr_t>(cb) & kHspPtrMask) | hi | lo;
}

[[nodiscard]] THETA_FORCE_INLINE HyperControlBlock* unpack_hsp_cb(
    std::uintptr_t tagged) noexcept {
  return reinterpret_cast<HyperControlBlock*>(tagged & kHspPtrMask);
}

[[nodiscard]] THETA_FORCE_INLINE std::int32_t unpack_hsp_cpu(
    std::uintptr_t tagged) noexcept {
  const std::uintptr_t lo = tagged & kHspLowCpuMask;
  const std::uintptr_t hi = (tagged >> 58U) & 0x3FULL;
  return static_cast<std::int32_t>((hi << 7U) | lo);
}

[[nodiscard]] std::uint32_t alloc_hsp_slot_and_init_cpu(
    std::int32_t cpu) noexcept;
void hsp_activate_shard_slow(HyperControlBlock* cb, std::int32_t cpu) noexcept;
void hsp_destroy_last_strong_slow(HyperControlBlock* cb) noexcept;
void hsp_weak_release_slow(HyperControlBlock* cb) noexcept;
[[nodiscard]] bool hsp_weak_lock_slow(HyperControlBlock* cb,
                                      std::int32_t cpu) noexcept;
[[nodiscard]] std::size_t hsp_use_count_slow(
    const HyperControlBlock* cb) noexcept;

THETA_FORCE_INLINE void hsp_inc_on_cpu(HyperControlBlock* cb,
                                       std::int32_t cpu) noexcept {
  if (cb->slot_id_ != 0U && g_hsp_slab_base != nullptr) [[likely]] {
    const std::size_t idx =
        (static_cast<std::size_t>(cpu) << kSlotShift) + cb->slot_id_;
    auto& shard = g_hsp_slab_base[idx];
    std::uint64_t cur = shard.load(std::memory_order_relaxed);
    while ((cur & 1ULL) != 0ULL) [[likely]] {
      if (shard.compare_exchange_weak(cur, cur + 2ULL,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed)) [[likely]] {
        return;
      }
    }
  }
  hsp_activate_shard_slow(cb, cpu);
}

// Returns true iff the last strong reference dropped (and `cb` was disposed).
THETA_FORCE_INLINE void hsp_dec_on_origin(HyperControlBlock* cb,
                                          std::int32_t origin_cpu) noexcept {
  if (cb->slot_id_ != 0U && g_hsp_slab_base != nullptr) [[likely]] {
    const std::size_t idx =
        (static_cast<std::size_t>(origin_cpu) << kSlotShift) + cb->slot_id_;
    auto& shard = g_hsp_slab_base[idx];
    std::uint64_t cur = shard.load(std::memory_order_relaxed);
    while (true) {
      if (cur > 3ULL) [[likely]] {
        if (shard.compare_exchange_weak(cur, cur - 2ULL,
                                        std::memory_order_acq_rel,
                                        std::memory_order_relaxed)) [[likely]] {
          return;
        }
      } else {
        if (shard.compare_exchange_weak(cur, 0ULL, std::memory_order_acq_rel,
                                        std::memory_order_relaxed)) {
          if (cb->active_shards_.fetch_sub(1U, std::memory_order_acq_rel) == 1U)
              [[unlikely]] {
            hsp_destroy_last_strong_slow(cb);
          }
          return;
        }
      }
    }
  } else {
    if (cb->active_shards_.fetch_sub(1U, std::memory_order_acq_rel) == 1U)
        [[unlikely]] {
      hsp_destroy_last_strong_slow(cb);
    }
  }
}

}  // namespace hyper_shared_ptr::detail
}  // namespace theta
