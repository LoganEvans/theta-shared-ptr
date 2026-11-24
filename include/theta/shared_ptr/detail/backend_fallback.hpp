// FILENAME: include/theta/shared_ptr/detail/backend_fallback.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <theta/shared_ptr/detail/common.hpp>

#if !THETA_SHARED_PTR_USE_RSEQ_ASM

namespace theta::detail::shared_ptr {

// Portable C++23 / ThreadSanitizer backend exercising the exact same promotion,
// per-CPU striped slab registration, `central_tickets`, and harvest state
// machine via instrumented atomics when `rseq` inline `asm` is disabled.
struct ArchBackend {
  static THETA_FORCE_INLINE bool try_rseq_inc(SpFastHeader* fast) noexcept {
    if (__atomic_load_n(&fast->fast_active, __ATOMIC_ACQUIRE) == 0U) {
      return false;
    }
    std::int64_t* const base = g_rseq_slab_base;
    if (base == nullptr) {
      return false;
    }
    const std::int32_t cpu = current_cpu_id();
    const std::uint32_t slot =
        __atomic_load_n(&fast->slot_id, __ATOMIC_RELAXED);
    std::int64_t* const cell =
        base + (static_cast<std::size_t>(cpu) << kSlotShift) + slot;
    std::int64_t cur = __atomic_load_n(cell, __ATOMIC_RELAXED);
    while (cur >= 1 && cur < 33) {
      if (__atomic_load_n(&fast->fast_active, __ATOMIC_ACQUIRE) == 0U) {
        return false;
      }
      if (__atomic_compare_exchange_n(cell, &cur, cur + 1, true,
                                      __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
        return true;
      }
    }
    return false;
  }

  static THETA_FORCE_INLINE bool try_rseq_dec(SpFastHeader* fast) noexcept {
    if (__atomic_load_n(&fast->fast_active, __ATOMIC_ACQUIRE) == 0U) {
      return false;
    }
    std::int64_t* const base = g_rseq_slab_base;
    if (base == nullptr) {
      return false;
    }
    const std::int32_t cpu = current_cpu_id();
    const std::uint32_t slot =
        __atomic_load_n(&fast->slot_id, __ATOMIC_RELAXED);
    std::int64_t* const cell =
        base + (static_cast<std::size_t>(cpu) << kSlotShift) + slot;
    std::int64_t cur = __atomic_load_n(cell, __ATOMIC_RELAXED);
    while (cur >= 2) {
      if (__atomic_load_n(&fast->fast_active, __ATOMIC_ACQUIRE) == 0U) {
        return false;
      }
      if (__atomic_compare_exchange_n(cell, &cur, cur - 1, true,
                                      __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
        return true;
      }
    }
    return false;
  }

  static THETA_FORCE_INLINE bool try_rseq_keepalive_acquire(
      ThetaSpCountedBase* const* /*cb_slot*/, ThetaSpCountedBase*& /*out_cb*/,
      std::uint32_t& /*out_slot*/) noexcept {
    return false;
  }

  static THETA_FORCE_INLINE bool try_rseq_atomic_load(
      const std::uintptr_t* /*rep_slot*/, ThetaSpCountedBase*& /*out_cb*/,
      std::uint32_t& /*out_slot*/) noexcept {
    return false;
  }

  static THETA_FORCE_INLINE bool try_rseq_drain_local_surplus(
      SpFastHeader* fast, std::int64_t& out_drained) noexcept {
    if (__atomic_load_n(&fast->fast_active, __ATOMIC_ACQUIRE) == 0U) {
      return false;
    }
    std::int64_t* const base = g_rseq_slab_base;
    if (base == nullptr) {
      return false;
    }
    const std::int32_t cpu = current_cpu_id();
    const std::uint32_t slot =
        __atomic_load_n(&fast->slot_id, __ATOMIC_RELAXED);
    std::int64_t* const cell =
        base + (static_cast<std::size_t>(cpu) << kSlotShift) + slot;
    std::int64_t cur = __atomic_load_n(cell, __ATOMIC_RELAXED);
    while (cur >= 2) {
      if (__atomic_load_n(&fast->fast_active, __ATOMIC_ACQUIRE) == 0U) {
        return false;
      }
      if (__atomic_compare_exchange_n(cell, &cur, 1, true, __ATOMIC_ACQ_REL,
                                      __ATOMIC_RELAXED)) {
        out_drained = cur - 1;
        return true;
      }
    }
    return false;
  }
};

}  // namespace theta::detail::shared_ptr

#endif  // !THETA_SHARED_PTR_USE_RSEQ_ASM
