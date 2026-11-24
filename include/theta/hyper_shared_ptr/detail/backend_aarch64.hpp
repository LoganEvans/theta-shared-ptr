// FILENAME: include/theta/hyper_shared_ptr/detail/backend_aarch64.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <theta/hyper_shared_ptr/detail/common.hpp>

#if THETA_ARCH_AARCH64 && THETA_HSP_USE_RSEQ_ASM

#include <sys/rseq.h>

namespace theta::hyper_shared_ptr::detail {

static_assert(offsetof(struct rseq, cpu_id) == 4,
              "Linux struct rseq::cpu_id must reside at byte offset 4.");
static_assert(offsetof(struct rseq, rseq_cs) == 8,
              "Linux struct rseq::rseq_cs must reside at byte offset 8.");
#ifdef RSEQ_SIG
static_assert(
    RSEQ_SIG == 0xd428bc00U,
    "AArch64 RSEQ_SIG must match 0xd428bc00 in inline assembly abort pad.");
#endif

struct ArchBackend {
  static THETA_FORCE_INLINE std::int32_t fast_cpu_id() noexcept {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const auto* rseq_base = tp + __rseq_offset;
    const auto* cpu_ptr = reinterpret_cast<const std::int32_t*>(rseq_base + 4);
    const std::int32_t c = __atomic_load_n(cpu_ptr, __ATOMIC_RELAXED);
    return (c >= 0 && c < kHspMaxCpus) ? c : current_cpu_id();
  }

  static THETA_FORCE_INLINE bool try_rseq_inc(SpFastHeader* fast) noexcept {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const char* const rseq_base = tp + __rseq_offset;
    std::int64_t* const base = g_rseq_slab_base;
    std::uint64_t idx;
    std::uint64_t slot;
    std::uint64_t val;

    // WHY ASM: (a) OS/ABI control-flow contract — Linux `rseq` `[2f, 3f)`.
    asm volatile goto(
        ".pushsection __rseq_cs, \"aw?\"\n\t"
        ".balign 32\n\t"
        "1:\n\t"
        ".long 0, 0\n\t"
        ".quad 2f, (3f - 2f), 4f\n\t"
        ".popsection\n\t"
        "6:\n\t"
        "adrp %[idx], 1b\n\t"
        "add %[idx], %[idx], :lo12:1b\n\t"
        "str %[idx], [%[rseq], #8]\n\t"
        "2:\n\t"
        "ldrb %w[idx], [%[fast], #4]\n\t"
        "cbz %w[idx], %l[slow_inc_arm]\n\t"
        "ldr %w[idx], [%[rseq], #4]\n\t"
        "cmp %w[idx], #2, lsl #12\n\t"
        "b.hs %l[slow_inc_arm]\n\t"
        "ldr %w[slot], [%[fast]]\n\t"
        "add %[idx], %[slot], %[idx], lsl #%c[shift]\n\t"
        "ldr %[val], [%[base], %[idx], lsl #3]\n\t"
        "cbz %[val], %l[slow_inc_arm]\n\t"
        "cmp %[val], #33\n\t"
        "b.hs %l[slow_inc_arm]\n\t"
        "add %[val], %[val], #1\n\t"
        "str %[val], [%[base], %[idx], lsl #3]\n\t"
        "3:\n\t"
        ".pushsection .text.unlikely, \"ax?\"\n\t"
        ".inst 0xd428bc00\n\t"
        "4:\n\t"
        "b 6b\n\t"
        ".popsection\n\t"
        : [idx] "=&r"(idx), [slot] "=&r"(slot), [val] "=&r"(val)
        : [rseq] "r"(rseq_base), [fast] "r"(fast), [base] "r"(base),
          [shift] "i"(kSlotShift)
        : "memory", "cc"
        : slow_inc_arm);
    return true;
  slow_inc_arm:
    return false;
  }

  static THETA_FORCE_INLINE bool try_rseq_dec(SpFastHeader* fast) noexcept {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const char* const rseq_base = tp + __rseq_offset;
    std::int64_t* const base = g_rseq_slab_base;
    std::uint64_t idx;
    std::uint64_t slot;
    std::uint64_t val;

    // WHY ASM: (a) OS/ABI control-flow contract — Linux `rseq` `[2f, 3f)`.
    asm volatile goto(
        ".pushsection __rseq_cs, \"aw?\"\n\t"
        ".balign 32\n\t"
        "1:\n\t"
        ".long 0, 0\n\t"
        ".quad 2f, (3f - 2f), 4f\n\t"
        ".popsection\n\t"
        "6:\n\t"
        "adrp %[idx], 1b\n\t"
        "add %[idx], %[idx], :lo12:1b\n\t"
        "str %[idx], [%[rseq], #8]\n\t"
        "2:\n\t"
        "ldrb %w[idx], [%[fast], #4]\n\t"
        "cbz %w[idx], %l[slow_dec_arm]\n\t"
        "ldr %w[idx], [%[rseq], #4]\n\t"
        "cmp %w[idx], #2, lsl #12\n\t"
        "b.hs %l[slow_dec_arm]\n\t"
        "ldr %w[slot], [%[fast]]\n\t"
        "add %[idx], %[slot], %[idx], lsl #%c[shift]\n\t"
        "ldr %[val], [%[base], %[idx], lsl #3]\n\t"
        "cmp %[val], #2\n\t"
        "b.lo %l[slow_dec_arm]\n\t"
        "sub %[val], %[val], #1\n\t"
        "str %[val], [%[base], %[idx], lsl #3]\n\t"
        "3:\n\t"
        ".pushsection .text.unlikely, \"ax?\"\n\t"
        ".inst 0xd428bc00\n\t"
        "4:\n\t"
        "b 6b\n\t"
        ".popsection\n\t"
        : [idx] "=&r"(idx), [slot] "=&r"(slot), [val] "=&r"(val)
        : [rseq] "r"(rseq_base), [fast] "r"(fast), [base] "r"(base),
          [shift] "i"(kSlotShift)
        : "memory", "cc"
        : slow_dec_arm);
    return true;
  slow_dec_arm:
    return false;
  }

  static THETA_FORCE_INLINE bool try_rseq_keepalive_acquire(
      ThetaSpCountedBase* const* cb_slot, ThetaSpCountedBase*& out_cb,
      std::uint32_t& out_slot) noexcept {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const char* const rseq_base = tp + __rseq_offset;
    std::int64_t* const base = g_rseq_slab_base;
    if (base == nullptr) [[unlikely]] {
      return false;
    }
    ThetaSpCountedBase* cb;
    std::uint64_t idx;
    std::uint64_t slot;
    std::uint64_t val;

    // WHY ASM: (a) OS/ABI control-flow contract — atomic pointer load +
    // per-CPU counter increment inside one Linux `rseq` `[2f, 3f)` transaction.
    asm volatile goto(
        ".pushsection __rseq_cs, \"aw?\"\n\t"
        ".balign 32\n\t"
        "1:\n\t"
        ".long 0, 0\n\t"
        ".quad 2f, (3f - 2f), 4f\n\t"
        ".popsection\n\t"
        "6:\n\t"
        "adrp %[idx], 1b\n\t"
        "add %[idx], %[idx], :lo12:1b\n\t"
        "str %[idx], [%[rseq], #8]\n\t"
        "2:\n\t"
        "ldr %[cb], [%[cb_slot]]\n\t"
        "cbz %[cb], %l[null_done_arm]\n\t"
        "ldrb %w[idx], [%[cb], #20]\n\t"
        "cbz %w[idx], %l[slow_ka_arm]\n\t"
        "ldr %w[idx], [%[rseq], #4]\n\t"
        "cmp %w[idx], #2, lsl #12\n\t"
        "b.hs %l[slow_ka_arm]\n\t"
        "ldr %w[slot], [%[cb], #16]\n\t"
        "add %[idx], %[slot], %[idx], lsl #%c[shift]\n\t"
        "ldr %[val], [%[base], %[idx], lsl #3]\n\t"
        "cbz %[val], %l[slow_ka_arm]\n\t"
        "add %[val], %[val], #1\n\t"
        "str %[val], [%[base], %[idx], lsl #3]\n\t"
        "3:\n\t"
        "dmb ishld\n\t"
        ".pushsection .text.unlikely, \"ax?\"\n\t"
        ".inst 0xd428bc00\n\t"
        "4:\n\t"
        "b 6b\n\t"
        ".popsection\n\t"
        : [cb] "=&r"(cb), [idx] "=&r"(idx), [slot] "=&r"(slot), [val] "=&r"(val)
        : [rseq] "r"(rseq_base), [cb_slot] "r"(cb_slot), [base] "r"(base),
          [shift] "i"(kSlotShift)
        : "memory", "cc"
        : null_done_arm, slow_ka_arm);
    out_cb = cb;
    out_slot = static_cast<std::uint32_t>(slot);
    return true;
  null_done_arm:
    out_cb = nullptr;
    out_slot = 0U;
    return true;
  slow_ka_arm:
    return false;
  }

  static THETA_FORCE_INLINE bool try_rseq_drain_local_surplus(
      SpFastHeader* fast, std::int64_t& out_drained) noexcept {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const char* const rseq_base = tp + __rseq_offset;
    std::int64_t* const base = g_rseq_slab_base;
    std::uint64_t idx;
    std::uint64_t slot;
    std::uint64_t val;
    std::uint64_t one = 1U;

    // WHY ASM: (a) OS/ABI control-flow contract — Linux `rseq` `[2f, 3f)`
    // resetting the caller CPU's local delta to `0` (`encoded = 1`).
    asm volatile goto(
        ".pushsection __rseq_cs, \"aw?\"\n\t"
        ".balign 32\n\t"
        "1:\n\t"
        ".long 0, 0\n\t"
        ".quad 2f, (3f - 2f), 4f\n\t"
        ".popsection\n\t"
        "6:\n\t"
        "adrp %[idx], 1b\n\t"
        "add %[idx], %[idx], :lo12:1b\n\t"
        "str %[idx], [%[rseq], #8]\n\t"
        "2:\n\t"
        "ldrb %w[idx], [%[fast], #4]\n\t"
        "cbz %w[idx], %l[no_drain_arm]\n\t"
        "ldr %w[idx], [%[rseq], #4]\n\t"
        "cmp %w[idx], #2, lsl #12\n\t"
        "b.hs %l[no_drain_arm]\n\t"
        "ldr %w[slot], [%[fast]]\n\t"
        "add %[idx], %[slot], %[idx], lsl #%c[shift]\n\t"
        "ldr %[val], [%[base], %[idx], lsl #3]\n\t"
        "cmp %[val], #2\n\t"
        "b.lo %l[no_drain_arm]\n\t"
        "str %[one], [%[base], %[idx], lsl #3]\n\t"
        "3:\n\t"
        ".pushsection .text.unlikely, \"ax?\"\n\t"
        ".inst 0xd428bc00\n\t"
        "4:\n\t"
        "b 6b\n\t"
        ".popsection\n\t"
        : [idx] "=&r"(idx), [slot] "=&r"(slot), [val] "=&r"(val)
        : [rseq] "r"(rseq_base), [fast] "r"(fast), [base] "r"(base),
          [one] "r"(one), [shift] "i"(kSlotShift)
        : "memory", "cc"
        : no_drain_arm);
    out_drained = static_cast<std::int64_t>(val - 1U);
    return true;
  no_drain_arm:
    return false;
  }
};

}  // namespace theta::hyper_shared_ptr::detail

#endif  // THETA_ARCH_AARCH64 && THETA_HSP_USE_RSEQ_ASM
