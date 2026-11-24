// FILENAME: include/theta/shared_ptr/detail/backend_x86_64.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <theta/shared_ptr/detail/common.hpp>

#if THETA_ARCH_X86_64 && THETA_SHARED_PTR_USE_RSEQ_ASM

#include <sys/rseq.h>

namespace theta::detail::shared_ptr {

static_assert(offsetof(struct rseq, cpu_id) == 4,
              "Linux struct rseq::cpu_id must reside at byte offset 4.");
static_assert(offsetof(struct rseq, rseq_cs) == 8,
              "Linux struct rseq::rseq_cs must reside at byte offset 8.");
#ifdef RSEQ_SIG
static_assert(
    RSEQ_SIG == 0x53053053,
    "x86-64 RSEQ_SIG must match 0x53053053 in inline assembly abort pad.");
#endif

struct ArchBackend {
  // Zero-atomic, zero-fence per-CPU `rseq` reference increment (`val >= 1` ->
  // `val + 1`, keeping width-matched 8-byte load/store for 4-cycle STLF).
  static THETA_FORCE_INLINE bool try_rseq_inc(SpFastHeader* fast) noexcept {
    const std::ptrdiff_t rseq_off = __rseq_offset;
    std::int64_t* const base = g_rseq_slab_base;
    std::uint64_t idx;
    std::uint64_t slot;
    std::uint64_t val;

    // WHY ASM: (a) OS/ABI control-flow contract — Linux `rseq` requires an
    // exact `[2f, 3f)` instruction window registered in `__rseq_cs` with
    // `RSEQ_SIG` (`0x53053053`) immediately preceding abort pad `4f`.
    asm volatile goto(
        ".pushsection __rseq_cs, \"aw?\"\n\t"
        ".balign 32\n\t"
        "1:\n\t"
        ".long 0, 0\n\t"
        ".quad 2f, (3f - 2f), 4f\n\t"
        ".popsection\n\t"
        "6:\n\t"
        "leaq 1b(%%rip), %[idx]\n\t"
        "movq %[idx], %%fs:8(%[rseq_off])\n\t"
        "2:\n\t"
        "cmpb $0, 4(%[fast])\n\t"
        "je %l[slow_inc]\n\t"
        "movl %%fs:4(%[rseq_off]), %k[idx]\n\t"
        "cmpl %[max_cpus], %k[idx]\n\t"
        "jae %l[slow_inc]\n\t"
        "movl (%[fast]), %k[slot]\n\t"
        "shlq %[shift], %[idx]\n\t"
        "addq %[slot], %[idx]\n\t"
        "movq (%[base], %[idx], 8), %[val]\n\t"
        "testq %[val], %[val]\n\t"
        "jz %l[slow_inc]\n\t"
        "cmpq $33, %[val]\n\t"
        "jae %l[slow_inc]\n\t"
        "incq %[val]\n\t"
        "movq %[val], (%[base], %[idx], 8)\n\t"
        "3:\n\t"
        ".pushsection .text.unlikely, \"ax?\"\n\t"
        ".long 0x53053053\n\t"
        "4:\n\t"
        "jmp 6b\n\t"
        ".popsection\n\t"
        : [idx] "=&r"(idx), [slot] "=&r"(slot), [val] "=&r"(val)
        : [rseq_off] "r"(rseq_off), [fast] "r"(fast), [base] "r"(base),
          [max_cpus] "i"(kSharedPtrMaxCpus), [shift] "i"(kSlotShift)
        : "memory", "cc"
        : slow_inc);
    return true;
  slow_inc:
    return false;
  }

  // Zero-atomic, zero-fence per-CPU `rseq` reference decrement (`val >= 2` ->
  // `val - 1`, leaving `val == 1` registered when local delta returns to `0`).
  static THETA_FORCE_INLINE bool try_rseq_dec(SpFastHeader* fast) noexcept {
    const std::ptrdiff_t rseq_off = __rseq_offset;
    std::int64_t* const base = g_rseq_slab_base;
    std::uint64_t idx;
    std::uint64_t slot;
    std::uint64_t val;

    // WHY ASM: (a) OS/ABI control-flow contract — Linux `rseq` `[2f, 3f)`
    // critical section aborting on preemption or `MEMBARRIER_RSEQ` harvest.
    asm volatile goto(
        ".pushsection __rseq_cs, \"aw?\"\n\t"
        ".balign 32\n\t"
        "1:\n\t"
        ".long 0, 0\n\t"
        ".quad 2f, (3f - 2f), 4f\n\t"
        ".popsection\n\t"
        "6:\n\t"
        "leaq 1b(%%rip), %[idx]\n\t"
        "movq %[idx], %%fs:8(%[rseq_off])\n\t"
        "2:\n\t"
        "cmpb $0, 4(%[fast])\n\t"
        "je %l[slow_dec]\n\t"
        "movl %%fs:4(%[rseq_off]), %k[idx]\n\t"
        "cmpl %[max_cpus], %k[idx]\n\t"
        "jae %l[slow_dec]\n\t"
        "movl (%[fast]), %k[slot]\n\t"
        "shlq %[shift], %[idx]\n\t"
        "addq %[slot], %[idx]\n\t"
        "movq (%[base], %[idx], 8), %[val]\n\t"
        "cmpq $2, %[val]\n\t"
        "jb %l[slow_dec]\n\t"
        "decq %[val]\n\t"
        "movq %[val], (%[base], %[idx], 8)\n\t"
        "3:\n\t"
        ".pushsection .text.unlikely, \"ax?\"\n\t"
        ".long 0x53053053\n\t"
        "4:\n\t"
        "jmp 6b\n\t"
        ".popsection\n\t"
        : [idx] "=&r"(idx), [slot] "=&r"(slot), [val] "=&r"(val)
        : [rseq_off] "r"(rseq_off), [fast] "r"(fast), [base] "r"(base),
          [max_cpus] "i"(kSharedPtrMaxCpus), [shift] "i"(kSlotShift)
        : "memory", "cc"
        : slow_dec);
    return true;
  slow_dec:
    return false;
  }

  // Wait-free `KeepAlive::get()` snapshot load + per-CPU increment inside a
  // single `rseq` window (`0` `LOCK`, `0` `shared_mutex`). If a writer calls
  // `KeepAlive::reset()` concurrently, its `MEMBARRIER_RSEQ` aborts any reader
  // between loading `*cb_slot` and committing `slab[cpu][slot] + 1`.
  static THETA_FORCE_INLINE bool try_rseq_keepalive_acquire(
      ThetaSpCountedBase* const* cb_slot, ThetaSpCountedBase*& out_cb,
      std::uint32_t& out_slot) noexcept {
    const std::ptrdiff_t rseq_off = __rseq_offset;
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
        "leaq 1b(%%rip), %[idx]\n\t"
        "movq %[idx], %%fs:8(%[rseq_off])\n\t"
        "2:\n\t"
        "movq (%[cb_slot]), %[cb]\n\t"
        "testq %[cb], %[cb]\n\t"
        "jz %l[null_done]\n\t"
        "cmpb $0, 20(%[cb])\n\t"
        "je %l[slow_ka]\n\t"
        "movl %%fs:4(%[rseq_off]), %k[idx]\n\t"
        "cmpl %[max_cpus], %k[idx]\n\t"
        "jae %l[slow_ka]\n\t"
        "movl 16(%[cb]), %k[slot]\n\t"
        "shlq %[shift], %[idx]\n\t"
        "addq %[slot], %[idx]\n\t"
        "movq (%[base], %[idx], 8), %[val]\n\t"
        "testq %[val], %[val]\n\t"
        "jz %l[slow_ka]\n\t"
        "incq %[val]\n\t"
        "movq %[val], (%[base], %[idx], 8)\n\t"
        "3:\n\t"
        ".pushsection .text.unlikely, \"ax?\"\n\t"
        ".long 0x53053053\n\t"
        "4:\n\t"
        "jmp 6b\n\t"
        ".popsection\n\t"
        : [cb] "=&r"(cb), [idx] "=&r"(idx), [slot] "=&r"(slot), [val] "=&r"(val)
        : [rseq_off] "r"(rseq_off), [cb_slot] "r"(cb_slot), [base] "r"(base),
          [max_cpus] "i"(kSharedPtrMaxCpus), [shift] "i"(kSlotShift)
        : "memory", "cc"
        : null_done, slow_ka);
    out_cb = cb;
    out_slot = static_cast<std::uint32_t>(slot);
    return true;
  null_done:
    out_cb = nullptr;
    out_slot = 0U;
    return true;
  slow_ka:
    return false;
  }

  // Drains accumulated producer surplus (`val >= 2` -> `1`) on the current CPU
  // inside `rseq` while caller holds `cb->fast_.reg_lock` (`0` `membarrier`!).
  static THETA_FORCE_INLINE bool try_rseq_drain_local_surplus(
      SpFastHeader* fast, std::int64_t& out_drained) noexcept {
    const std::ptrdiff_t rseq_off = __rseq_offset;
    std::int64_t* const base = g_rseq_slab_base;
    std::uint64_t idx;
    std::uint64_t slot;
    std::uint64_t val;

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
        "leaq 1b(%%rip), %[idx]\n\t"
        "movq %[idx], %%fs:8(%[rseq_off])\n\t"
        "2:\n\t"
        "cmpb $0, 4(%[fast])\n\t"
        "je %l[no_drain]\n\t"
        "movl %%fs:4(%[rseq_off]), %k[idx]\n\t"
        "cmpl %[max_cpus], %k[idx]\n\t"
        "jae %l[no_drain]\n\t"
        "movl (%[fast]), %k[slot]\n\t"
        "shlq %[shift], %[idx]\n\t"
        "addq %[slot], %[idx]\n\t"
        "movq (%[base], %[idx], 8), %[val]\n\t"
        "cmpq $2, %[val]\n\t"
        "jb %l[no_drain]\n\t"
        "movq $1, (%[base], %[idx], 8)\n\t"
        "3:\n\t"
        ".pushsection .text.unlikely, \"ax?\"\n\t"
        ".long 0x53053053\n\t"
        "4:\n\t"
        "jmp 6b\n\t"
        ".popsection\n\t"
        : [idx] "=&r"(idx), [slot] "=&r"(slot), [val] "=&r"(val)
        : [rseq_off] "r"(rseq_off), [fast] "r"(fast), [base] "r"(base),
          [max_cpus] "i"(kSharedPtrMaxCpus), [shift] "i"(kSlotShift)
        : "memory", "cc"
        : no_drain);
    out_drained = static_cast<std::int64_t>(val - 1U);
    return true;
  no_drain:
    return false;
  }
};

}  // namespace theta::detail::shared_ptr

#endif  // THETA_ARCH_X86_64 && THETA_SHARED_PTR_USE_RSEQ_ASM
