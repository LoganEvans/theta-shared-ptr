// FILENAME: tests/test_shared_ptr.cpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <theta/shared_ptr.hpp>
#include <thread>
#include <unordered_set>
#include <vector>

#include "opaque_std_interop.hpp"

namespace theta {
namespace {

struct TrackedBase {
  static inline std::atomic<int> live{0};
  static inline std::atomic<int> dtors{0};

  int base_val{0};

  TrackedBase() noexcept : TrackedBase(0) {}

  explicit TrackedBase(int v) noexcept : base_val(v) {
    live.fetch_add(1, std::memory_order_relaxed);
  }

  TrackedBase(const TrackedBase& other) noexcept : base_val(other.base_val) {
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

struct UnrelatedPolymorphic {
  virtual ~UnrelatedPolymorphic() = default;
};

struct ThrowingObj {
  explicit ThrowingObj(bool should_throw) {
    if (should_throw) {
      throw std::runtime_error("ctor boom");
    }
  }
};

template <typename T>
struct CountingAllocator {
  using value_type = T;

  std::atomic<int>* allocs{nullptr};
  std::atomic<int>* deallocs{nullptr};

  CountingAllocator() noexcept = default;
  CountingAllocator(std::atomic<int>* a, std::atomic<int>* d) noexcept
      : allocs(a), deallocs(d) {}

  template <typename U>
  CountingAllocator(const CountingAllocator<U>& other) noexcept
      : allocs(other.allocs), deallocs(other.deallocs) {}

  T* allocate(std::size_t n) {
    if (allocs != nullptr) {
      allocs->fetch_add(1, std::memory_order_relaxed);
    }
    return std::allocator<T>{}.allocate(n);
  }

  void deallocate(T* p, std::size_t n) noexcept {
    if (deallocs != nullptr) {
      deallocs->fetch_add(1, std::memory_order_relaxed);
    }
    std::allocator<T>{}.deallocate(p, n);
  }

  template <typename U>
  bool operator==(const CountingAllocator<U>& other) const noexcept {
    return allocs == other.allocs && deallocs == other.deallocs;
  }
};

// ============================================================================
// 1. Single-Threaded Lifecycle, Custom Deleters, Allocators, Aliasing, Void
// ============================================================================
TEST(SharedPtrBasicTest, LifecycleConversionsAndSnakeCaseAliases) {
  TrackedBase::reset_counts();
  {
    theta::shared_ptr<TrackedDerived> d = make_shared<TrackedDerived>(10, 20);
    EXPECT_TRUE(d);
    EXPECT_EQ(d->base_val, 10);
    EXPECT_EQ(d->extra, 20);
    EXPECT_EQ(d.use_count(), 1L);
    EXPECT_EQ(TrackedBase::live.load(), 1);

    // Copy promotes to Per-CPU Mode
    SharedPtr<TrackedBase> b = d;
    EXPECT_EQ(b.get(), d.get());
    EXPECT_EQ(b.use_count(), 2L);

    // `SharedPtr<void>` + `const T` conversion + aliasing constructor
    SharedPtr<const TrackedBase> cb = b;
    SharedPtr<void> vp = b;
    EXPECT_EQ(vp.get(), static_cast<void*>(d.get()));
    SharedPtr<const int> alias(d, &d->extra);
    EXPECT_EQ(*alias, 20);
    EXPECT_EQ(cb.use_count(), 5L);

    // Self-copy and self-move safety
    auto& b_ref = b;
    b = b_ref;
    b = std::move(b_ref);
    EXPECT_TRUE(b);

    // Move construction and move assignment
    SharedPtr<TrackedBase> moved(std::move(b));
    EXPECT_FALSE(b);
    EXPECT_TRUE(moved);
    EXPECT_EQ(moved.use_count(), 5L);

    d.reset();
    cb.reset();
    vp.reset();
    alias.reset();
    EXPECT_EQ(moved.use_count(), 1L);
    EXPECT_EQ(TrackedBase::dtors.load(), 0);
  }
  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 1);
}

TEST(SharedPtrBasicTest, CustomDeleterAllocatorsAndUniquePtrAdoption) {
  int deleter_calls = 0;
  struct CustomDel {
    int* counter;
    void operator()(int* p) const noexcept {
      ++(*counter);
      delete p;
    }
    void operator()(std::nullptr_t) const noexcept { ++(*counter); }
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

  // `nullptr` with custom deleter
  {
    SharedPtr<int> null_p(nullptr, CustomDel{&deleter_calls});
    EXPECT_FALSE(null_p);
  }
  EXPECT_EQ(deleter_calls, 2);

  // Custom allocator with `allocate_shared`, `allocate_shared_for_overwrite`,
  // and `SharedPtr(ptr, deleter, alloc)`
  std::atomic<int> alloc_cnt{0};
  std::atomic<int> dealloc_cnt{0};
  CountingAllocator<int> ca(&alloc_cnt, &dealloc_cnt);
  {
    auto ap = allocate_shared<TrackedDerived>(ca, 33, 44);
    EXPECT_EQ(ap->base_val, 33);
    EXPECT_EQ(ap->extra, 44);

    auto ov = allocate_shared_for_overwrite<int>(ca);
    *ov = 99;
    EXPECT_EQ(*ov, 99);

    SharedPtr<int> dp(new int(12), CustomDel{&deleter_calls}, ca);
    EXPECT_EQ(*dp, 12);
  }
  EXPECT_EQ(alloc_cnt.load(), 3);
  EXPECT_EQ(dealloc_cnt.load(), 3);
  EXPECT_EQ(deleter_calls, 3);

  // `std::unique_ptr` adoption (value deleter, reference deleter, and null)
  {
    auto up = std::make_unique<TrackedDerived>(51, 52);
    SharedPtr<TrackedBase> from_up = std::move(up);
    EXPECT_EQ(up.get(), nullptr);
    EXPECT_EQ(from_up->base_val, 51);

    CustomDel ref_del{&deleter_calls};
    std::unique_ptr<int, CustomDel&> up_ref(new int(61), ref_del);
    SharedPtr<int> from_ref(std::move(up_ref));
    EXPECT_EQ(*from_ref, 61);

    std::unique_ptr<int> null_up;
    SharedPtr<int> from_null(std::move(null_up));
    EXPECT_FALSE(from_null);
    EXPECT_EQ(from_null.use_count(), 0L);
  }
  EXPECT_EQ(deleter_calls, 4);
  EXPECT_EQ(TrackedBase::live.load(), 0);

  EXPECT_THROW((void)make_shared<ThrowingObj>(true), std::runtime_error);
}

// ============================================================================
// 2. Bounded & Unbounded Arrays (`T[]`, `T[N]`, `make_shared_for_overwrite`)
// ============================================================================
TEST(SharedPtrDropInFeaturesTest, BoundedAndUnboundedArrays) {
  TrackedBase::reset_counts();
  {
    // Unbounded array value-init & fill-init
    SharedPtr<TrackedBase[]> arr =
        make_shared<TrackedBase[]>(5, TrackedBase(7));
    for (std::ptrdiff_t i = 0; i < 5; ++i) {
      EXPECT_EQ(arr[i].base_val, 7);
      arr[i].base_val = static_cast<int>(100 + i);
    }
    SharedPtr<const TrackedBase[]> carr = arr;
    EXPECT_EQ(carr[3].base_val, 103);
    EXPECT_EQ(carr.use_count(), 2L);

    // Bounded array `U[N]` -> `U[]` -> `const U[]` conversion
    SharedPtr<int[4]> fixed = make_shared<int[4]>(42);
    for (std::ptrdiff_t i = 0; i < 4; ++i) {
      EXPECT_EQ(fixed[i], 42);
    }
    SharedPtr<int[]> dyn_from_fixed = fixed;
    SharedPtr<const int[]> cdyn_from_fixed = fixed;
    EXPECT_EQ(cdyn_from_fixed[2], 42);
    EXPECT_EQ(fixed.use_count(), 3L);

    // `make_shared_for_overwrite` (scalar, bounded array, unbounded array)
    auto raw_scalar = make_shared_for_overwrite<int>();
    *raw_scalar = 1234;
    EXPECT_EQ(*raw_scalar, 1234);

    auto raw_buf = make_shared_for_overwrite<std::uint64_t[]>(16);
    for (std::ptrdiff_t i = 0; i < 16; ++i) {
      raw_buf[i] = static_cast<std::uint64_t>(i * 3);
    }
    EXPECT_EQ(raw_buf[15], 45ULL);

    // Raw `new T[n]` array constructor uses `default_delete<T[]>`
    SharedPtr<TrackedBase[]> raw_arr(new TrackedBase[3]);
    EXPECT_EQ(raw_arr[0].base_val, 0);
  }
  EXPECT_EQ(TrackedBase::live.load(), 0);
}

// ============================================================================
// 3. `EnableSharedFromThis` & `std::enable_shared_from_this` Dual Support
// ============================================================================
struct ThetaEsftNode : enable_shared_from_this<ThetaEsftNode> {
  int id{0};
  explicit ThetaEsftNode(int v) noexcept : id(v) {}
};

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
struct StdEsftNode : std::enable_shared_from_this<StdEsftNode> {
  static inline std::atomic<int> live{0};
  int id{0};
  explicit StdEsftNode(int v) noexcept : id(v) { ++live; }
  ~StdEsftNode() noexcept { --live; }
};
#endif

TEST(SharedPtrDropInFeaturesTest, EnableSharedFromThisBothThetaAndStd) {
  // Unmanaged `EnableSharedFromThis` must throw `std::bad_weak_ptr`
  ThetaEsftNode stack_node(1);
  EXPECT_THROW((void)stack_node.shared_from_this(), std::bad_weak_ptr);
  EXPECT_TRUE(stack_node.weak_from_this().expired());

  {
    auto n = make_shared<ThetaEsftNode>(99);
    auto n2 = n->shared_from_this();
    const ThetaEsftNode* cn = n.get();
    SharedPtr<const ThetaEsftNode> n3 = cn->shared_from_this();
    EXPECT_EQ(n.get(), n2.get());
    EXPECT_EQ(n.get(), n3.get());
    EXPECT_EQ(n.use_count(), 3L);
    EXPECT_FALSE(n->weak_from_this().expired());
  }

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
  StdEsftNode::live.store(0);
  {
    std::shared_ptr<StdEsftNode> escaped_std;
    {
      auto tn = make_shared<StdEsftNode>(777);
      auto tn_copy = tn;  // Promote to `rseq` Per-CPU Mode
      escaped_std = tn->shared_from_this();
      EXPECT_EQ(escaped_std->id, 777);
      EXPECT_EQ(tn_copy->id, 777);
    }
    // Both `theta::SharedPtr` handles dropped; `escaped_std` must keep it alive
    EXPECT_EQ(StdEsftNode::live.load(), 1);
    EXPECT_EQ(escaped_std->id, 777);
  }
  EXPECT_EQ(StdEsftNode::live.load(), 0);
#endif
}

// ============================================================================
// 4. Casts, `<=>`, `std::hash`, `owner_less`/`owner_equal`/`owner_hash`, Stream
// ============================================================================
TEST(SharedPtrDropInFeaturesTest, CastsComparisonsHashingAndOwnerOps) {
  TrackedBase::reset_counts();
  {
    SharedPtr<TrackedDerived> d = make_shared<TrackedDerived>(15, 25);
    SharedPtr<TrackedBase> b = static_pointer_cast<TrackedBase>(d);
    EXPECT_EQ(d.use_count(), 2L);

    auto d_dyn = dynamic_pointer_cast<TrackedDerived>(b);
    ASSERT_TRUE(d_dyn);
    EXPECT_EQ(d_dyn->extra, 25);
    EXPECT_FALSE(dynamic_pointer_cast<UnrelatedPolymorphic>(b));

    // Move-cast (`&&`) transfers ownership without extra refcount bump
    const long before = d_dyn.use_count();
    SharedPtr<const TrackedDerived> cd =
        const_pointer_cast<const TrackedDerived>(std::move(d_dyn));
    EXPECT_FALSE(d_dyn);
    EXPECT_EQ(cd.use_count(), before);

    SharedPtr<TrackedDerived> unconst =
        const_pointer_cast<TrackedDerived>(std::move(cd));
    SharedPtr<int> reinterp = reinterpret_pointer_cast<int>(unconst);
    EXPECT_EQ(reinterp.get(), reinterpret_cast<int*>(unconst.get()));

    // `<=>`, `std::hash`, `std::ostringstream`, `owner_less`, `owner_equal`
    SharedPtr<int> null_sp;
    EXPECT_EQ(null_sp <=> nullptr, std::strong_ordering::equal);
    EXPECT_TRUE((unconst <=> nullptr) > 0);
    EXPECT_GT(unconst, nullptr);

    std::ostringstream oss;
    oss << unconst;
    EXPECT_FALSE(oss.str().empty());

    std::unordered_set<SharedPtr<TrackedDerived>> set;
    set.insert(unconst);
    EXPECT_EQ(set.count(unconst), 1U);

    WeakPtr<TrackedDerived> wp(unconst);
    EXPECT_TRUE(unconst.owner_equal(wp));
    EXPECT_EQ(unconst.owner_hash(), wp.owner_hash());

    std::map<WeakPtr<TrackedDerived>, int, owner_less<>> omap;
    omap[wp] = 42;
    EXPECT_EQ(omap.find(wp)->second, 42);

    SharedPtr<TrackedDerived> locked_ctor(wp);
    EXPECT_EQ(locked_ctor.get(), unconst.get());
  }
  EXPECT_EQ(TrackedBase::live.load(), 0);

  WeakPtr<int> expired_wp;
  EXPECT_THROW((void)SharedPtr<int>(expired_wp), std::bad_weak_ptr);
}

// ============================================================================
// 5. Cross-TU Binary `std::shared_ptr` & `std::weak_ptr` ABI Interop Tests
// ============================================================================
#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
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
      std::weak_ptr<OpaquePayload> sw = tw;
      WeakPtr<OpaquePayload> tw_roundtrip(sw);
      EXPECT_EQ(tw_roundtrip.lock()->value, 123);
      EXPECT_TRUE(test_opaque::opaque_verify_owner_equivalence(
          escaped, tp.to_std(), sw));
    }
    EXPECT_EQ(OpaquePayload::live_count.load(), 1);
    EXPECT_EQ(escaped->value, 123);
  }
  EXPECT_EQ(OpaquePayload::live_count.load(), 0);
  EXPECT_EQ(OpaquePayload::dtor_count.load(), 1);

  // Case B: Promoted Per-CPU `SharedPtr` + Move Conversions Both Ways
  {
    auto tp1 = make_shared<OpaquePayload>(456);
    auto tp2 = tp1;  // Promotes to `kModePerCpuActive`
    std::shared_ptr<OpaquePayload> sp1 = std::move(tp2).to_std();
    EXPECT_FALSE(tp2);
    EXPECT_EQ(test_opaque::opaque_clone_by_value(sp1)->value, 456);

    SharedPtr<OpaquePayload> tp3(sp1);
    SharedPtr<OpaquePayload> tp4(std::move(sp1));
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
// 6. Multithreaded Concurrency Stress: `0 <-> 1` Churn, Cross-CPU Producer->
//    Consumer Handoff, `WeakPtr::lock()`, `KeepAlive`, & `std::atomic`
// ============================================================================
TEST(ConcurrencyStressTest, ZeroToOneHotSharing) {
  TrackedBase::reset_counts();
  constexpr int kThreads = 8;
  constexpr int kIters = 15000;

  {
    auto shared_obj = make_shared<TrackedBase>(101);
#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
    std::shared_ptr<TrackedBase> std_bridge = shared_obj.to_std();
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
          {
            auto local = shared_obj;
            ASSERT_EQ(local->base_val, 101);
          }
#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
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
#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
    const long expected_cnt = 2L;
#else
    const long expected_cnt = 1L;
#endif
    EXPECT_EQ(shared_obj.use_count(), expected_cnt);
  }
  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 1);
}

TEST(ConcurrencyStressTest, ProducerConsumerCrossThreadHandoffAndEarlyDrop) {
  TrackedBase::reset_counts();
  constexpr int kPairs = 4;
  constexpr int kItemsPerPair = 4000;

  struct alignas(64) MailSlot {
    std::atomic<bool> full{false};
    SharedPtr<TrackedBase> item;
  };

  std::array<MailSlot, kPairs> slots{};
  std::vector<std::thread> threads;

  auto transient = make_shared<TrackedBase>(555);

  for (int p = 0; p < kPairs; ++p) {
    threads.emplace_back([&, p, seed = transient]() mutable {
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
        slots[static_cast<std::size_t>(p)].item = seed;
        slots[static_cast<std::size_t>(p)].full.store(
            true, std::memory_order_release);
      }
      seed.reset();
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
        auto a = std::move(slots[static_cast<std::size_t>(p)].item);
        slots[static_cast<std::size_t>(p)].full.store(
            false, std::memory_order_release);
        ASSERT_EQ(a->base_val, 555);
      }
    });
  }

  transient.reset();

  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), 1);
}

TEST(ConcurrencyStressTest, WeakLockRacesFinalDestroy) {
  TrackedBase::reset_counts();
  constexpr int kRounds = 400;

  for (int r = 0; r < kRounds; ++r) {
    auto sp = make_shared<TrackedBase>(r);
    auto sp_copy = sp;
    WeakPtr<TrackedBase> wp(sp);
#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
    std::weak_ptr<TrackedBase> std_wp = wp.to_std();
#endif

    std::thread locker([&]() {
      for (int k = 0; k < 8; ++k) {
        if (auto locked = wp.lock()) {
          ASSERT_EQ(locked->base_val, r);
        }
#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
        const int v = test_opaque::opaque_lock_weak_and_read(
            reinterpret_cast<const std::weak_ptr<test_opaque::OpaquePayload>&>(
                std_wp),
            0);
        (void)v;
#endif
      }
    });

    sp_copy.reset();
    sp.reset();
    locker.join();

    EXPECT_TRUE(wp.expired());
    EXPECT_EQ(wp.lock().get(), nullptr);
  }

  EXPECT_EQ(TrackedBase::live.load(), 0);
  EXPECT_EQ(TrackedBase::dtors.load(), kRounds);
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
        auto snap = ka.load();
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

TEST(ConcurrencyStressTest, StdAtomicSharedPtrCompareExchangeAndWait) {
  TrackedBase::reset_counts();
  static_assert(sizeof(std::atomic<SharedPtr<TrackedBase>>) ==
                sizeof(SharedPtr<TrackedBase>));

  {
    auto init = make_shared<TrackedBase>(0);
    std::atomic<SharedPtr<TrackedBase>> cell(init);
    std::atomic<WeakPtr<TrackedBase>> weak_cell{WeakPtr<TrackedBase>(init)};

    std::thread waiter([&]() { cell.wait(init); });

    constexpr int kWorkers = 4;
    constexpr int kUpdates = 250;
    std::vector<std::thread> workers;
    for (int t = 0; t < kWorkers; ++t) {
      workers.emplace_back([&]() {
        for (int i = 0; i < kUpdates; ++i) {
          SharedPtr<TrackedBase> cur = cell.load();
          while (true) {
            auto next = make_shared<TrackedBase>(cur->base_val + 1);
            if (cell.compare_exchange_weak(cur, std::move(next))) {
              break;
            }
          }
          weak_cell.store(WeakPtr<TrackedBase>(cell.load()));
        }
      });
    }

    for (auto& w : workers) {
      w.join();
    }
    waiter.join();

    EXPECT_EQ(cell.load()->base_val, kWorkers * kUpdates);
    EXPECT_FALSE(weak_cell.load().expired());
  }

  EXPECT_EQ(TrackedBase::live.load(), 0);
}

struct alignas(128) OverAlignedPayload {
  std::uint64_t a{0};
  std::uint64_t b{0};
};

TEST(SharedPtrSemanticsTest, OverAlignedAndAliasedKeepAlive) {
  {
    auto over =
        make_shared<OverAlignedPayload>(OverAlignedPayload{11ULL, 22ULL});
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(over.get()) % 128U, 0U);
    EXPECT_EQ(over->a, 11ULL);
    EXPECT_EQ(over->b, 22ULL);

    auto over_arr = make_shared<OverAlignedPayload[]>(4);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(over_arr.get()) % 128U, 0U);
  }
  {
    struct PairObj {
      int first{101};
      int second{202};
    };
    auto root = make_shared<PairObj>();
    SharedPtr<int> a1(root, &root->first);
    SharedPtr<int> a2(root, &root->second);
    KeepAlive<int> ka1(a1);
    KeepAlive<int> ka2(a2);
    EXPECT_EQ(*ka1.get(), 101);
    EXPECT_EQ(*ka2.get(), 202);
  }
}

}  // namespace
}  // namespace theta
