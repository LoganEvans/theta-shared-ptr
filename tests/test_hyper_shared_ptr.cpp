// FILENAME: tests/test_hyper_shared_ptr.cpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <theta/hyper-shared-ptr/hyper-shared-ptr.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <theta/hyper_shared_ptr.hpp>
#include <thread>
#include <vector>

#include "opaque_std_interop.hpp"

namespace theta {
namespace {

struct TrackedBase {
  static inline std::atomic<int> live{0};
  static inline std::atomic<int> dtors{0};

  int base_val{0};

  explicit TrackedBase(int v) noexcept : base_val(v) {
    live.fetch_add(1, std::memory_order_relaxed);
  }

  virtual ~TrackedBase() noexcept {
    base_val = -1;
    live.fetch_sub(1, std::memory_order_relaxed);
    dtors.fetch_add(1, std::memory_order_relaxed);
  }

  static void reset_counts() noexcept {
    live.store(0, std::memory_order_seq_cst);
    dtors.store(0, std::memory_order_seq_cst);
  }
};

struct TrackedDerived final : TrackedBase {
  int extra{0};
  TrackedDerived(int b, int e) noexcept : TrackedBase(b), extra(e) {}
};

struct ThrowingObj {
  explicit ThrowingObj(bool should_throw) {
    if (should_throw) {
      throw std::runtime_error("ctor boom");
    }
  }
};

// ============================================================================
// 1. Single-Threaded Lifecycle, Custom Deleters, Aliasing, Derived->Base,
//    Exceptions, & Legacy `theta::hsp::HyperSharedPointer` API Compatibility
// ============================================================================
TEST(SharedPtrBasicTest, LifecycleAndConversions) {
  TrackedBase::reset_counts();
  {
    SharedPtr<TrackedDerived> d = make_shared<TrackedDerived>(10, 20);
    EXPECT_TRUE(d);
    EXPECT_EQ(d->base_val, 10);
    EXPECT_EQ(d->extra, 20);
    EXPECT_EQ(d.use_count(), 1L);
    EXPECT_EQ(TrackedBase::live.load(), 1);

    // Copy promotes to Per-CPU Mode
    SharedPtr<TrackedBase> b = d;
    EXPECT_EQ(b.get(), d.get());
    EXPECT_EQ(b.use_count(), 2L);

    // `const T` conversion + aliasing constructor
    SharedPtr<const TrackedBase> cb = b;
    SharedPtr<const int> alias(d, &d->extra);
    EXPECT_EQ(*alias, 20);
    EXPECT_EQ(cb.use_count(), 4L);

    // Self-copy and self-move safety
    auto& b_ref = b;
    b = b_ref;
    b = std::move(b_ref);
    EXPECT_TRUE(b);

    // Move construction and move assignment
    SharedPtr<TrackedBase> moved(std::move(b));
    EXPECT_FALSE(b);
    EXPECT_TRUE(moved);
    EXPECT_EQ(moved.use_count(), 4L);

    d.reset();
    cb.reset();
    alias.reset();
    EXPECT_EQ(moved.use_count(), 1L);
    EXPECT_EQ(TrackedBase::dtors.load(), 0);
  }
  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 1);
}

TEST(SharedPtrBasicTest, CustomDeleterAndExceptionSafety) {
  int deleter_calls = 0;
  struct CustomDel {
    int* counter;
    void operator()(int* p) const noexcept {
      ++(*counter);
      delete p;
    }
  };

  {
    SharedPtr<int> p(new int(77), CustomDel{&deleter_calls});
    auto* del = get_deleter<CustomDel>(p);
    ASSERT_NE(del, nullptr);
    EXPECT_EQ(del->counter, &deleter_calls);

    SharedPtr<int> p2 = p;
    EXPECT_EQ(*p2, 77);
  }
  EXPECT_EQ(deleter_calls, 1);

  // Nullptr with custom deleter still invokes deleter per C++ `shared_ptr` spec
  {
    SharedPtr<int> null_p(static_cast<int*>(nullptr),
                          CustomDel{&deleter_calls});
    EXPECT_FALSE(null_p);
  }
  EXPECT_EQ(deleter_calls, 2);

  EXPECT_THROW((void)make_shared<ThrowingObj>(true), std::runtime_error);
}

TEST(HyperSharedPtrBasicTest, LifecycleAnd8192CpuTagging) {
  TrackedBase::reset_counts();
  {
    HyperSharedPtr<TrackedDerived> p = make_hyper_shared<TrackedDerived>(7, 9);
    EXPECT_EQ(p.use_count(), 1U);
    EXPECT_GE(p.original_cpu(), 0);
    EXPECT_LT(p.original_cpu(), kHspMaxCpus);

    // Test explicit handle tagging across all 13-bit CPU boundaries (`0..8191`)
    constexpr std::array<std::int32_t, 10> kTestCpus = {
        0, 1, 31, 63, 64, 127, 128, 1024, 4095, 8191};
    std::vector<HyperSharedPtr<TrackedDerived>> copies;
    for (std::int32_t c : kTestCpus) {
      auto cp = copy_hyper_on_cpu_for_test(p, c);
      EXPECT_EQ(cp.original_cpu(), c);
      EXPECT_EQ(cp.get(), p.get());
      copies.push_back(std::move(cp));
    }

    EXPECT_EQ(p.use_count(), 1U + kTestCpus.size());

    // Destroy in reverse order on current thread (cross-CPU origin release)
    while (!copies.empty()) {
      copies.pop_back();
    }
    EXPECT_EQ(p.use_count(), 1U);
    EXPECT_EQ(TrackedBase::dtors.load(), 0);
  }
  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 1);
}

TEST(HyperSharedPtrBasicTest, LegacyHspNamespaceCompatibility) {
  TrackedBase::reset_counts();
  {
    hsp::HyperSharedPointer<TrackedBase> p1;
    hsp::HyperSharedPointer<TrackedBase> p2;
    {
      hsp::KeepAlive<TrackedBase> ka(new TrackedBase(4));
      EXPECT_EQ(ka.get()->base_val, 4);
      p1 = ka.get();
      EXPECT_EQ(ka.reset(new TrackedBase(5))->base_val, 5);
      p2 = ka.get();
    }
    EXPECT_EQ(p1->base_val, 4);
    EXPECT_EQ(p2->base_val, 5);
    EXPECT_EQ(TrackedBase::live.load(), 2);
  }
  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 2);
}

// ============================================================================
// 2. Cross-TU Binary `std::shared_ptr` & `std::weak_ptr` ABI Interop Tests
// ============================================================================
#if THETA_HSP_HAS_GLIBCXX_ABI
TEST(AbiInteropTest, UnpromotedAndPromotedCrossTuInterop) {
  using test_opaque::OpaquePayload;
  OpaquePayload::live_count.store(0);
  OpaquePayload::dtor_count.store(0);

  // Case A: Unpromoted `SharedPtr` -> `std::shared_ptr` outliving Theta handle
  {
    std::shared_ptr<OpaquePayload> escaped;
    {
      auto tp = make_shared<OpaquePayload>(123);
      EXPECT_EQ(test_opaque::opaque_read_and_copy_churn(tp, 16), 123);
      escaped = tp.to_std();
      WeakPtr<OpaquePayload> tw(tp);
      auto sw = tw.to_std();
      EXPECT_TRUE(test_opaque::opaque_verify_owner_equivalence(
          escaped, tp.to_std(), sw));
    }
    // `tp` destroyed, `escaped` inside stock `std::shared_ptr` keeps it alive!
    EXPECT_EQ(OpaquePayload::live_count.load(), 1);
    EXPECT_EQ(escaped->value, 123);
  }
  EXPECT_EQ(OpaquePayload::live_count.load(), 0);
  EXPECT_EQ(OpaquePayload::dtor_count.load(), 1);

  // Case B: Promoted Per-CPU `SharedPtr` + Move Conversions Both Ways
  {
    auto tp1 = make_shared<OpaquePayload>(456);
    auto tp2 = tp1;  // Promotes to `kPerCpuActive`
    std::shared_ptr<OpaquePayload> sp1 = std::move(tp2).to_std();
    EXPECT_FALSE(tp2);
    EXPECT_EQ(test_opaque::opaque_clone_by_value(sp1)->value, 456);

    SharedPtr<OpaquePayload> tp3(sp1);  // Copy from `std::shared_ptr`
    SharedPtr<OpaquePayload> tp4(
        std::move(sp1));  // Move from `std::shared_ptr`
    EXPECT_FALSE(sp1);
    EXPECT_EQ(tp3->value, 456);
    EXPECT_EQ(tp4->value, 456);
  }
  EXPECT_EQ(OpaquePayload::live_count.load(), 0);
  EXPECT_EQ(OpaquePayload::dtor_count.load(), 2);

  // Case C: Adopting a native `std::make_shared` object into `theta::SharedPtr`
  {
    std::shared_ptr<OpaquePayload> native =
        test_opaque::opaque_make_native_std(789);
    SharedPtr<OpaquePayload> adopted(native);
    SharedPtr<OpaquePayload> copy2 = adopted;
    native.reset();
    EXPECT_EQ(copy2->value, 789);
    EXPECT_EQ(OpaquePayload::live_count.load(), 1);
  }
  EXPECT_EQ(OpaquePayload::live_count.load(), 0);
  EXPECT_EQ(OpaquePayload::dtor_count.load(), 3);
}
#endif

// ============================================================================
// 3. Multithreaded Concurrency Stress: `0 <-> 1` Churn, Cross-CPU Producer->
//    Consumer Handoff, `WeakPtr::lock()` Races, & `KeepAlive` Snapshot Churn
// ============================================================================
TEST(ConcurrencyStressTest, ZeroToOneHotSharingBothVariants) {
  TrackedBase::reset_counts();
  constexpr int kThreads = 8;
  constexpr int kIters = 15000;

  {
    auto shared_abi = make_shared<TrackedBase>(101);
    auto shared_hsp = make_hyper_shared<TrackedBase>(202);
#if THETA_HSP_HAS_GLIBCXX_ABI
    std::shared_ptr<TrackedBase> std_bridge = shared_abi.to_std();
#endif

    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t]() {
        unsigned int spins = 0U;
        while (!start.load(std::memory_order_relaxed)) {
          if ((++spins & 63U) == 0U) {
            std::this_thread::yield();
          } else {
            arch::cpu_relax();
          }
        }
        (void)start.load(std::memory_order_acquire);
        for (int i = 0; i < kIters; ++i) {
          // 0 -> 1 -> 0 single local copy churn
          {
            auto local_abi = shared_abi;
            ASSERT_EQ(local_abi->base_val, 101);
          }
          {
            auto local_hsp = shared_hsp;
            ASSERT_EQ(local_hsp->base_val, 202);
          }
#if THETA_HSP_HAS_GLIBCXX_ABI
          if ((i & 63) == 0 && (t & 1) == 0) {
            std::shared_ptr<TrackedBase> local_std = std_bridge;
            ASSERT_EQ(local_std->base_val, 101);
          }
#endif
        }
      });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : threads) {
      th.join();
    }
#if THETA_HSP_HAS_GLIBCXX_ABI
    const long expected_abi_cnt = 2L;
#else
    const long expected_abi_cnt = 1L;
#endif
    EXPECT_EQ(shared_abi.use_count(), expected_abi_cnt);
    EXPECT_EQ(shared_hsp.use_count(), 1U);
  }
  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 2);
}

TEST(ConcurrencyStressTest, ProducerConsumerCrossThreadHandoffAndEarlyDrop) {
  TrackedBase::reset_counts();
  constexpr int kPairs = 4;
  constexpr int kItemsPerPair = 4000;

  struct alignas(64) MailSlot {
    std::atomic<bool> full{false};
    SharedPtr<TrackedBase> abi_item;
    HyperSharedPtr<TrackedBase> hsp_item;
  };

  std::array<MailSlot, kPairs> slots{};
  std::vector<std::thread> threads;

  // Also test an object whose creator drops its handle BEFORE consumers drain!
  auto transient_abi = make_shared<TrackedBase>(555);
  auto transient_hsp = make_hyper_shared<TrackedBase>(666);

  for (int p = 0; p < kPairs; ++p) {
    threads.emplace_back(
        [&, p, seed_abi = transient_abi, seed_hsp = transient_hsp]() mutable {
          for (int i = 0; i < kItemsPerPair; ++i) {
            unsigned int spins = 0U;
            while (slots[static_cast<std::size_t>(p)].full.load(
                std::memory_order_acquire)) {
              if ((++spins & 63U) == 0U) {
                std::this_thread::yield();
              } else {
                arch::cpu_relax();
              }
            }
            slots[static_cast<std::size_t>(p)].abi_item = seed_abi;
            slots[static_cast<std::size_t>(p)].hsp_item = seed_hsp;
            slots[static_cast<std::size_t>(p)].full.store(
                true, std::memory_order_release);
          }
          // Drop producer's seed handles immediately!
          seed_abi.reset();
          seed_hsp.reset();
        });

    threads.emplace_back([&, p]() {
      for (int i = 0; i < kItemsPerPair; ++i) {
        unsigned int spins = 0U;
        while (!slots[static_cast<std::size_t>(p)].full.load(
            std::memory_order_acquire)) {
          if ((++spins & 63U) == 0U) {
            std::this_thread::yield();
          } else {
            arch::cpu_relax();
          }
        }
        auto a = std::move(slots[static_cast<std::size_t>(p)].abi_item);
        auto h = std::move(slots[static_cast<std::size_t>(p)].hsp_item);
        slots[static_cast<std::size_t>(p)].full.store(
            false, std::memory_order_release);
        ASSERT_EQ(a->base_val, 555);
        ASSERT_EQ(h->base_val, 666);
      }
    });
  }

  // Drop main thread's handles while producers & consumers are still in flight!
  transient_abi.reset();
  transient_hsp.reset();

  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 2);
}

TEST(ConcurrencyStressTest, WeakLockRacesFinalDestroy) {
  TrackedBase::reset_counts();
  constexpr int kRounds = 400;

  for (int r = 0; r < kRounds; ++r) {
    auto sp = make_shared<TrackedBase>(r);
    auto sp_copy = sp;  // Promote to `kPerCpuActive`
    WeakPtr<TrackedBase> wp(sp);
#if THETA_HSP_HAS_GLIBCXX_ABI
    std::weak_ptr<TrackedBase> std_wp = wp.to_std();
#endif

    auto hp = make_hyper_shared<TrackedBase>(r + 1000);
    HyperWeakPtr<TrackedBase> hwp(hp);

    std::thread locker([&]() {
      for (int k = 0; k < 8; ++k) {
        if (auto locked = wp.lock()) {
          ASSERT_EQ(locked->base_val, r);
        }
#if THETA_HSP_HAS_GLIBCXX_ABI
        const int v = test_opaque::opaque_lock_weak_and_read(
            reinterpret_cast<const std::weak_ptr<test_opaque::OpaquePayload>&>(
                std_wp),
            0);
        (void)v;
#endif
        if (auto hlocked = hwp.lock()) {
          ASSERT_EQ(hlocked->base_val, r + 1000);
        }
      }
    });

    sp_copy.reset();
    sp.reset();
    hp.reset();
    locker.join();

    EXPECT_TRUE(wp.expired());
    EXPECT_EQ(wp.lock().get(), nullptr);
    EXPECT_TRUE(hwp.expired());
    EXPECT_EQ(hwp.lock().get(), nullptr);
  }

  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 2 * kRounds);
}

TEST(ConcurrencyStressTest, KeepAliveConcurrentReadersAndResets) {
  TrackedBase::reset_counts();
  KeepAlive<TrackedBase> ka(new TrackedBase(1));
  std::atomic<bool> stop{false};

  std::vector<std::thread> readers;
  for (int t = 0; t < 6; ++t) {
    readers.emplace_back([&]() {
      unsigned int iters = 0U;
      while (!stop.load(std::memory_order_relaxed)) {
        auto snap = ka.get();
        if (snap) {
          ASSERT_GE(snap->base_val, 1);
        }
        if constexpr (arch::kTsanEnabled) {
          if ((++iters & 15U) == 0U) {
            std::this_thread::yield();
          }
        }
      }
    });
  }

  for (int gen = 2; gen <= 150; ++gen) {
    auto cur = ka.reset(new TrackedBase(gen));
    ASSERT_GE(cur->base_val, gen);
  }
  ka.reset(static_cast<TrackedBase*>(nullptr));

  stop.store(true, std::memory_order_release);
  for (auto& th : readers) {
    th.join();
  }

  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 150);
}

}  // namespace
}  // namespace theta
