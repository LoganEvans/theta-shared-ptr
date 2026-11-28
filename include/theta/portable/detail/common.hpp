// FILENAME: include/theta/portable/detail/common.hpp
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
#include <theta/shared_ptr/detail/common.hpp>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace theta::portable {

inline constexpr std::uint32_t kMaxPortableThreads = 512U;

namespace detail::shared_ptr {

using ::theta::detail::shared_ptr::kModeAtomicRetired;
using ::theta::detail::shared_ptr::kModePerCpuActive;
using ::theta::detail::shared_ptr::kModeUnpromoted;
using ::theta::detail::shared_ptr::kPromotionWarmupCopies;
using ::theta::detail::shared_ptr::kWarmupActiveBit;
using ::theta::detail::shared_ptr::kWarmupCopyBase;
using ::theta::detail::shared_ptr::kWarmupCopyRange;
using ::theta::detail::shared_ptr::kWarmupFirstCopyWord;
using ::theta::detail::shared_ptr::kWarmupSealedBit;
using ::theta::detail::shared_ptr::kWarmupTagMask;
using ::theta::detail::shared_ptr::SpFastHeader;
using ::theta::detail::shared_ptr::SpPinResult;
using ::theta::detail::shared_ptr::StdSpCountedBase;

inline constexpr int kPortableSlotShift =
    (::theta::arch::kTsanEnabled || ::theta::arch::kAsanEnabled) ? 12 : 16;
inline constexpr std::uint32_t kMaxPortableSlots = (1U << kPortableSlotShift);

// ============================================================================
// `PortableSpCountedBase`: 32-byte non-template `final` control block deriving
// from `StdSpCountedBase` (`std::_Sp_counted_base<_S_atomic>` on libstdc++) for
// 100% `std::shared_ptr` ABI compatibility without Linux `rseq` or syscalls.
// ============================================================================
class PortableSpCountedBase;
extern const void* g_portable_sp_vptr;
void record_portable_sp_vptr(const PortableSpCountedBase* sample) noexcept;

struct alignas(8) PortableColdOps {
  void (*dispose)(PortableSpCountedBase*) noexcept;
  void (*destroy)(PortableSpCountedBase*) noexcept;
  void (*dispose_and_destroy)(PortableSpCountedBase*) noexcept;
  void* (*get_deleter)(PortableSpCountedBase*, const std::type_info&) noexcept;
};

class PortableSpCountedBase final : public StdSpCountedBase {
 public:
  explicit PortableSpCountedBase(std::uintptr_t ops_or_size) noexcept {
    std::atomic_signal_fence(std::memory_order_relaxed);
    ops_or_size_ = ops_or_size;
#if !(defined(__GXX_ABI_VERSION) && !THETA_ASAN_ENABLED)
    if (__atomic_load_n(&g_portable_sp_vptr, __ATOMIC_RELAXED) == nullptr)
        [[unlikely]] {
      record_portable_sp_vptr(this);
    }
#endif
  }

  ~PortableSpCountedBase() noexcept override = default;

  void _M_dispose() noexcept override;
  void _M_destroy() noexcept override;
  void* _M_get_deleter(const std::type_info& ti) noexcept override;

  SpFastHeader fast_{};
  std::uintptr_t ops_or_size_{0};
};

static_assert(sizeof(PortableSpCountedBase) == 4 * sizeof(void*));

#if defined(__GXX_ABI_VERSION) && !THETA_ASAN_ENABLED
extern const void* const kPortableSpVtableSym[] asm(
    "_ZTVN5theta8portable6detail10shared_ptr21PortableSpCountedBaseE");
#endif

// ============================================================================
// 64B Cache-Line-Isolated Per-Thread 2-Party Dekker Gate + Cold Hazard Slot
//   - Byte 0 (`harvester_busy`): written ONLY by Harvester (`0` = open)
//   - Byte 1 (`owner_busy`):     written ONLY by Owner thread (non-zero offset;
//     acts as both the slab Dekker byte AND the fast-path reader epoch gate!)
//   - `cold_hazard_cb`:          populated ONLY when `.load()`/`.get()` misses
//     the 5-instruction fast cell and drops `owner_busy` before `lock_cb(cb)`.
// ============================================================================
struct alignas(64) PortableThreadGate {
  std::atomic<std::uint8_t> harvester_busy{0};
  std::atomic<std::uint8_t> owner_busy{0};
  std::atomic<std::uint8_t> harvester_lock{0};
  std::atomic<bool> in_use{false};
  std::uint32_t pad0_{0};
  std::atomic<const void*> cold_hazard_cb{nullptr};
};

struct alignas(16) PortableFastTls {
  PortableThreadGate* gate;
  std::int64_t* stripe_base;
};

#if THETA_OS_LINUX
extern __thread PortableFastTls g_portable_fast_tls
    __attribute__((tls_model("initial-exec")));
#else
extern thread_local PortableFastTls g_portable_fast_tls;
#endif

extern void** g_portable_keepalive_ptrs;

// Cold slow paths in `src/portable_shared_ptr.cpp`
void sp_add_ref_slow(StdSpCountedBase* pi) noexcept;
void sp_release_slow(StdSpCountedBase* pi) noexcept;
[[nodiscard]] long sp_use_count_slow(const StdSpCountedBase* pi) noexcept;
void sp_note_to_std_copy(StdSpCountedBase* pi) noexcept;
void sp_note_to_std_move(StdSpCountedBase* pi) noexcept;
void sp_add_ref_from_std(StdSpCountedBase* pi) noexcept;
void sp_adopt_from_std_move(StdSpCountedBase* pi) noexcept;
[[nodiscard]] bool sp_weak_lock_slow(StdSpCountedBase* pi) noexcept;

[[nodiscard]] SpPinResult sp_pin_for_keepalive(PortableSpCountedBase* cb,
                                               void* managed_ptr) noexcept;
[[nodiscard]] PortableSpCountedBase* sp_keepalive_unpinned_sentinel() noexcept;
[[nodiscard]] bool sp_try_pin_atomic_fast(PortableSpCountedBase* cb,
                                          void* managed_ptr,
                                          bool& out_never_pin) noexcept;
[[nodiscard]] void* sp_acquire_pinned_under_hazard_slow(
    PortableSpCountedBase* cb, PortableThreadGate* gate) noexcept;
void sp_register_keepalive_stripe_slow(PortableSpCountedBase* cb) noexcept;
void sp_unpin_and_retire_old(PortableSpCountedBase* old_cb,
                             bool drop_anchor_ref,
                             bool quiesce_readers) noexcept;

void atomic_ptr_lock(const void* addr) noexcept;
void atomic_ptr_unlock(const void* addr) noexcept;
void atomic_ptr_unlock_and_notify(const void* addr, bool notify_all) noexcept;
[[nodiscard]] std::uint32_t atomic_ptr_seq(const void* addr) noexcept;
void atomic_ptr_wait_seq(const void* addr, std::uint32_t expected_seq) noexcept;

THETA_FORCE_INLINE bool is_portable_sp(const StdSpCountedBase* pi) noexcept {
  if (pi == nullptr) [[unlikely]] {
    return false;
  }
  const void* vptr = nullptr;
  std::memcpy(&vptr, static_cast<const void*>(pi), sizeof(vptr));
#if defined(__GXX_ABI_VERSION) && !THETA_ASAN_ENABLED
  return vptr == static_cast<const void*>(kPortableSpVtableSym + 2);
#else
  return vptr == __atomic_load_n(&g_portable_sp_vptr, __ATOMIC_RELAXED);
#endif
}

// ============================================================================
// 2-Party Dekker Gate + Fused Hazard-Slot Backend:
//   - On x86-64: `movb $1, 1(%gate)` + stack red-zone `lock orl $0, -8(%rsp)` +
//     disjoint `cmpb $0, (%gate)` (`0` partial-STLF penalty, `0` `xchgb`).
//   - On AArch64 / C++23: `stlrb` (`owner_busy`) -> `ldarb` (`harvester_busy`)
//     disjoint-byte `RCsc` handshake (`0` `dmb ish`, `0` `cas`/`casl`).
//   - Inside the gate, `slab[tid][slot]` is 100% single-owner non-atomic memory
//     and unlock is a single 1-byte store `owner_busy = 0`.
// ============================================================================
struct PortableBackend {
  static THETA_FORCE_INLINE bool enter_owner_gate(
      PortableThreadGate* gate) noexcept {
#if THETA_ARCH_X86_64 && !THETA_TSAN_ENABLED
    gate->owner_busy.store(1U, std::memory_order_relaxed);
    if constexpr (::theta::arch::kHasRedZoneStlfPause) {
      // WHY ASM: (b) Missing ISA/C++ construct — drains the x86 Store Buffer
      // and AMD Zen 5 Write-Coalescing Buffer on the private M-state stack
      // red-zone line (~4.8 ns vs `mfence` ~14 ns or `xchgb` ~8.3 ns).
      asm volatile("lock orl $0, -8(%%rsp)" ::: "memory", "cc");
    } else {
      std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    return gate->harvester_busy.load(std::memory_order_relaxed) == 0U;
#else
    gate->owner_busy.store(1U, std::memory_order_seq_cst);
    return gate->harvester_busy.load(std::memory_order_seq_cst) == 0U;
#endif
  }

  static THETA_FORCE_INLINE void exit_owner_gate(
      PortableThreadGate* gate) noexcept {
    gate->owner_busy.store(0U, std::memory_order_release);
  }

  static THETA_FORCE_INLINE bool try_stripe_inc_slot(
      std::uint32_t slot) noexcept {
    std::int64_t* const base = g_portable_fast_tls.stripe_base;
    if (base == nullptr) [[unlikely]] {
      return false;
    }
    std::int64_t* const cell = base + slot;
    std::int64_t cur = __atomic_load_n(cell, __ATOMIC_RELAXED);
    while (cur >= 1 && cur < 65) [[likely]] {
      if (__atomic_compare_exchange_n(cell, &cur, cur + 1, true,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED))
          [[likely]] {
        return true;
      }
    }
    return false;
  }

  static THETA_FORCE_INLINE bool try_stripe_dec_slot(
      std::uint32_t slot) noexcept {
    std::int64_t* const base = g_portable_fast_tls.stripe_base;
    if (base == nullptr) [[unlikely]] {
      return false;
    }
    std::int64_t* const cell = base + slot;
    std::int64_t cur = __atomic_load_n(cell, __ATOMIC_RELAXED);
    while (cur >= 2) [[likely]] {
      if (__atomic_compare_exchange_n(cell, &cur, cur - 1, true,
                                      __ATOMIC_RELEASE, __ATOMIC_RELAXED))
          [[likely]] {
        return true;
      }
    }
    return false;
  }

  static THETA_FORCE_INLINE bool try_drain_local_surplus_locked(
      SpFastHeader* fast, std::int64_t& out_drained) noexcept {
    if (__atomic_load_n(&fast->fast_active, __ATOMIC_ACQUIRE) == 0U) {
      return false;
    }
    std::int64_t* const base = g_portable_fast_tls.stripe_base;
    if (base == nullptr) {
      return false;
    }
    const std::uint32_t slot =
        __atomic_load_n(&fast->slot_id, __ATOMIC_RELAXED);
    if (slot == 0U || slot >= kMaxPortableSlots) {
      return false;
    }
    std::int64_t* const cell = base + slot;
    std::int64_t cur = __atomic_load_n(cell, __ATOMIC_RELAXED);
    while (cur >= 2) {
      if (__atomic_compare_exchange_n(cell, &cur, 1, true, __ATOMIC_ACQ_REL,
                                      __ATOMIC_RELAXED)) {
        out_drained = cur - 1;
        return true;
      }
    }
    return false;
  }
};

}  // namespace detail::shared_ptr
}  // namespace theta::portable
