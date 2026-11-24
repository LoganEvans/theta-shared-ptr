// FILENAME: include/theta/shared_ptr/detail/common.hpp
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
#define THETA_SHARED_PTR_HAS_GLIBCXX_ABI 1
#else
#define THETA_SHARED_PTR_HAS_GLIBCXX_ABI 0
#endif

#if THETA_HAS_LINUX_RSEQ && (THETA_ARCH_X86_64 || THETA_ARCH_AARCH64) && \
    !THETA_TSAN_ENABLED
#define THETA_SHARED_PTR_USE_RSEQ_ASM 1
#else
#define THETA_SHARED_PTR_USE_RSEQ_ASM 0
#endif

namespace theta {

inline constexpr std::int32_t kSharedPtrMaxCpus = 8192;

namespace detail::shared_ptr {

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
// in 32 bytes (`40B` total with an inplace `uint64_t`).
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

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
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
// `ThetaSpCountedBase`: Non-template `final` control block deriving directly
// from `std::_Sp_counted_base<_S_atomic>` for 100% binary `std::shared_ptr`
// and `std::weak_ptr` ABI compatibility across all `T`, `Deleter`, and `Alloc`.
//
// Because `ThetaSpCountedBase` is a single non-template `final` class, every
// instance in the process shares the exact same canonical `vptr` (`_ZTV + 16`),
// allowing O(1) 1-instruction identification without RTTI or pointer tags.
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
// `overflow_cpus`, `rapid_harvest_streak`) lives out-of-line in
// `g_rseq_slot_meta` indexed by `fast_.slot_id` and is touched only if the
// object is promoted.
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

  // Key function defined in `src/shared_ptr.cpp` to anchor a single canonical
  // vtable address across the entire process.
  void _M_dispose() noexcept override;
  void _M_destroy() noexcept override;
  void* _M_get_deleter(const std::type_info& ti) noexcept override;

  SpFastHeader fast_{};
  // LSB == 1: `(sizeof(SpInplaceStorage<T>) << 1) | 1` for zero-indirect-call
  //           sized `::operator delete` when `T` is trivially destructible and
  //           standard-aligned (with default allocator).
  // LSB == 0: `const SpColdOps*` descriptor in `.rodata`.
  std::uintptr_t ops_or_size_{0};
};

static_assert(sizeof(ThetaSpCountedBase) == 4 * sizeof(void*));

#if defined(__GXX_ABI_VERSION) && !THETA_ASAN_ENABLED
extern const void* const kThetaSpVtableSym[] asm(
    "_ZTVN5theta6detail10shared_ptr18ThetaSpCountedBaseE");
#endif

// Global runtime pointers exposed for zero-indirection `rseq` inline fast paths
extern std::int64_t* g_rseq_slab_base;
extern void** g_rseq_keepalive_ptrs;

[[nodiscard]] std::int32_t current_cpu_id() noexcept;
[[nodiscard]] std::int32_t max_possible_cpus() noexcept;

// Cold slow paths for `ThetaSpCountedBase`
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

// Striped spinlock + futex wait/notify table for `std::atomic<SharedPtr<T>>`
// and `std::atomic<WeakPtr<T>>` (keeps `sizeof(atomic<SharedPtr<T>>) == 16`).
void atomic_ptr_lock(const void* addr) noexcept;
void atomic_ptr_unlock(const void* addr) noexcept;
void atomic_ptr_unlock_and_notify(const void* addr, bool notify_all) noexcept;
[[nodiscard]] std::uint32_t atomic_ptr_seq(const void* addr) noexcept;
void atomic_ptr_wait_seq(const void* addr, std::uint32_t expected_seq) noexcept;

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

}  // namespace detail::shared_ptr
}  // namespace theta
