// FILENAME: benchmarks/bench_hyper_shared_ptr.cpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT
//
// 3-way microarchitectural comparison across:
//   (a) `StdSharedPtr`   — stock `std::shared_ptr<T>` (`LOCK XADD` on 1 line)
//   (b) `HyperSharedPtr` — 8192-CPU handle-tagged per-CPU atomic shards
//   (c) `ThetaSharedPtr` — 100% `std::shared_ptr` ABI-compatible `rseq`
//                          8192-CPU striped per-CPU delta reference counter

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <theta/arch.hpp>
#include <theta/hyper_shared_ptr.hpp>
#include <thread>
#include <vector>

namespace {

void apply_thread_sweep(benchmark::Benchmark* b) {
  const int hc =
      std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  b->Threads(1);
  if (hc >= 2) {
    b->Threads(2);
  }
  if (hc >= 4) {
    b->Threads(4);
  }
  if (hc >= 8 && hc != 8) {
    b->Threads(8);
  }
  if (hc > 4) {
    b->Threads(hc);
  }
  b->UseRealTime();
}

void apply_pair_sweep(benchmark::Benchmark* b) {
  const int hc =
      std::max(2, static_cast<int>(std::thread::hardware_concurrency()));
  b->Threads(2);
  if (hc >= 4) {
    b->Threads(4);
  }
  if (hc > 4) {
    const int even_hc = (hc & ~1);
    if (even_hc > 4) {
      b->Threads(even_hc);
    }
  }
  b->UseRealTime();
}

// ============================================================================
// 01. Unshared Creation + Destruction (`make_*_shared` + ~Ptr)
// ============================================================================
void BM_01a_CreateDestroy_StdSharedPtr(benchmark::State& state) {
  for (auto _ : state) {
    auto p = std::make_shared<std::uint64_t>(42ULL);
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK(BM_01a_CreateDestroy_StdSharedPtr)->Apply(apply_thread_sweep);

void BM_01b_CreateDestroy_HyperSharedPtr(benchmark::State& state) {
  for (auto _ : state) {
    auto p = theta::make_hyper_shared<std::uint64_t>(42ULL);
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK(BM_01b_CreateDestroy_HyperSharedPtr)->Apply(apply_thread_sweep);

void BM_01c_CreateDestroy_ThetaSharedPtr(benchmark::State& state) {
  for (auto _ : state) {
    auto p = theta::make_shared<std::uint64_t>(42ULL);
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK(BM_01c_CreateDestroy_ThetaSharedPtr)->Apply(apply_thread_sweep);

// ============================================================================
// 02. Hot Sharing (`0 -> 1 -> 0` Single Temporary Copy Loop — Real-World
//     `auto cfg = global_cfg;` Service Handle Pattern)
// ============================================================================
void BM_02a_HotShare0to1_StdSharedPtr(benchmark::State& state) {
  static const auto* g_ptr =
      new std::shared_ptr<std::uint64_t>(std::make_shared<std::uint64_t>(7ULL));
  for (auto _ : state) {
    std::shared_ptr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_02a_HotShare0to1_StdSharedPtr)->Apply(apply_thread_sweep);

void BM_02b_HotShare0to1_HyperSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::HyperSharedPtr<std::uint64_t>(
      theta::make_hyper_shared<std::uint64_t>(7ULL));
  for (auto _ : state) {
    theta::HyperSharedPtr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_02b_HotShare0to1_HyperSharedPtr)->Apply(apply_thread_sweep);

void BM_02c_HotShare0to1_ThetaSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::SharedPtr<std::uint64_t>(
      theta::make_shared<std::uint64_t>(7ULL));
  for (auto _ : state) {
    theta::SharedPtr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_02c_HotShare0to1_ThetaSharedPtr)->Apply(apply_thread_sweep);

// ============================================================================
// 03. Pinned Hot Sharing (`1 -> 2 -> 1`, Matching Original README Regime
//     Without `uniform_int_distribution` PRNG Overhead)
// ============================================================================
void BM_03a_HotSharePinned_StdSharedPtr(benchmark::State& state) {
  static const auto* g_ptr =
      new std::shared_ptr<std::uint64_t>(std::make_shared<std::uint64_t>(9ULL));
  std::shared_ptr<std::uint64_t> pinned = *g_ptr;
  benchmark::DoNotOptimize(pinned);
  for (auto _ : state) {
    std::shared_ptr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_03a_HotSharePinned_StdSharedPtr)->Apply(apply_thread_sweep);

void BM_03b_HotSharePinned_HyperSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::HyperSharedPtr<std::uint64_t>(
      theta::make_hyper_shared<std::uint64_t>(9ULL));
  theta::HyperSharedPtr<std::uint64_t> pinned = *g_ptr;
  benchmark::DoNotOptimize(pinned);
  for (auto _ : state) {
    theta::HyperSharedPtr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_03b_HotSharePinned_HyperSharedPtr)->Apply(apply_thread_sweep);

void BM_03c_HotSharePinned_ThetaSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::SharedPtr<std::uint64_t>(
      theta::make_shared<std::uint64_t>(9ULL));
  theta::SharedPtr<std::uint64_t> pinned = *g_ptr;
  benchmark::DoNotOptimize(pinned);
  for (auto _ : state) {
    theta::SharedPtr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_03c_HotSharePinned_ThetaSharedPtr)->Apply(apply_thread_sweep);

// ============================================================================
// 04. Cross-Thread Producer -> Consumer Handoff (Batched Ring Slots)
// ============================================================================
template <typename PtrType>
struct alignas(64) SpscRing {
  static constexpr std::size_t kCap = 64;
  alignas(64) std::atomic<std::uint64_t> head{0};
  alignas(64) std::atomic<std::uint64_t> tail{0};
  alignas(64) std::array<PtrType, kCap> slots{};
};

template <typename PtrType>
void run_prod_cons_bench(benchmark::State& state, const PtrType& proto) {
  static std::array<SpscRing<PtrType>, 128> rings{};
  const int pair_id = state.thread_index() / 2;
  auto& ring = rings[static_cast<std::size_t>(pair_id) & 127U];

  if (state.thread_index() % 2 == 0) {
    // Producer copies `proto` and hands off to sibling consumer thread
    for (auto _ : state) {
      std::uint64_t t = ring.tail.load(std::memory_order_relaxed);
      while (t - ring.head.load(std::memory_order_acquire) >=
             SpscRing<PtrType>::kCap) {
        theta::arch::cpu_relax();
      }
      ring.slots[t & (SpscRing<PtrType>::kCap - 1U)] = proto;
      ring.tail.store(t + 1ULL, std::memory_order_release);
    }
    // Drain leftovers at end of run so next thread-count run starts clean
    while (ring.head.load(std::memory_order_acquire) !=
           ring.tail.load(std::memory_order_relaxed)) {
      theta::arch::cpu_relax();
    }
  } else {
    // Consumer pops and destroys handle on a different thread/CPU
    for (auto _ : state) {
      std::uint64_t h = ring.head.load(std::memory_order_relaxed);
      while (ring.tail.load(std::memory_order_acquire) == h) {
        theta::arch::cpu_relax();
      }
      PtrType item = std::move(ring.slots[h & (SpscRing<PtrType>::kCap - 1U)]);
      ring.head.store(h + 1ULL, std::memory_order_release);
      benchmark::DoNotOptimize(item);
    }
  }
}

void BM_04a_CrossThreadProdCons_StdSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new std::shared_ptr<std::uint64_t>(
      std::make_shared<std::uint64_t>(11ULL));
  run_prod_cons_bench(state, *g_ptr);
}
BENCHMARK(BM_04a_CrossThreadProdCons_StdSharedPtr)->Apply(apply_pair_sweep);

void BM_04b_CrossThreadProdCons_HyperSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::HyperSharedPtr<std::uint64_t>(
      theta::make_hyper_shared<std::uint64_t>(11ULL));
  run_prod_cons_bench(state, *g_ptr);
}
BENCHMARK(BM_04b_CrossThreadProdCons_HyperSharedPtr)->Apply(apply_pair_sweep);

void BM_04c_CrossThreadProdCons_ThetaSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::SharedPtr<std::uint64_t>(
      theta::make_shared<std::uint64_t>(11ULL));
  run_prod_cons_bench(state, *g_ptr);
}
BENCHMARK(BM_04c_CrossThreadProdCons_ThetaSharedPtr)->Apply(apply_pair_sweep);

// ============================================================================
// 05. Read-Mostly Config Snapshot (`KeepAlive::get()` — 99.99% Readers,
//     Periodic Writer Reload Every 16,384 Ops on Thread 0)
// ============================================================================
class StdKeepAlive {
 public:
  explicit StdKeepAlive(std::uint64_t v)
      : ptr_(std::make_shared<std::uint64_t>(v)) {}

  std::shared_ptr<std::uint64_t> get() const {
    std::shared_lock<std::shared_mutex> lock(mtx_);
    return ptr_;
  }

  void reset(std::uint64_t v) {
    auto next = std::make_shared<std::uint64_t>(v);
    std::unique_lock<std::shared_mutex> lock(mtx_);
    ptr_ = std::move(next);
  }

 private:
  mutable std::shared_mutex mtx_;
  std::shared_ptr<std::uint64_t> ptr_;
};

void BM_05a_KeepAliveSnapshot_StdSharedMutex(benchmark::State& state) {
  static auto* g_ka = new StdKeepAlive(1ULL);
  std::uint64_t iter = 0;
  for (auto _ : state) {
    if (state.thread_index() == 0 && ((++iter & 16383ULL) == 0ULL))
        [[unlikely]] {
      g_ka->reset(iter);
    } else {
      auto snap = g_ka->get();
      benchmark::DoNotOptimize(*snap);
    }
  }
}
BENCHMARK(BM_05a_KeepAliveSnapshot_StdSharedMutex)->Apply(apply_thread_sweep);

void BM_05b_KeepAliveSnapshot_HyperKeepAlive(benchmark::State& state) {
  static auto* g_ka =
      new theta::HyperKeepAlive<std::uint64_t>(new std::uint64_t(1ULL));
  std::uint64_t iter = 0;
  for (auto _ : state) {
    if (state.thread_index() == 0 && ((++iter & 16383ULL) == 0ULL))
        [[unlikely]] {
      (void)g_ka->reset(new std::uint64_t(iter));
    } else {
      auto snap = g_ka->get();
      benchmark::DoNotOptimize(*snap);
    }
  }
}
BENCHMARK(BM_05b_KeepAliveSnapshot_HyperKeepAlive)->Apply(apply_thread_sweep);

void BM_05c_KeepAliveSnapshot_ThetaKeepAlive(benchmark::State& state) {
  static auto* g_ka = new theta::KeepAlive<std::uint64_t>(
      theta::make_shared<std::uint64_t>(1ULL));
  std::uint64_t iter = 0;
  for (auto _ : state) {
    if (state.thread_index() == 0 && ((++iter & 16383ULL) == 0ULL))
        [[unlikely]] {
      (void)g_ka->reset(theta::make_shared<std::uint64_t>(iter));
    } else {
      auto snap = g_ka->get();
      benchmark::DoNotOptimize(*snap);
    }
  }
}
BENCHMARK(BM_05c_KeepAliveSnapshot_ThetaKeepAlive)->Apply(apply_thread_sweep);

// ============================================================================
// 06. 2,048-Object Working-Set Table (Cache Footprint & Striped Page Packing)
// ============================================================================
constexpr std::size_t kTableSize = 2048;

void BM_06a_Table2K_StdSharedPtr(benchmark::State& state) {
  static const auto* g_tbl = []() {
    auto* v = new std::array<std::shared_ptr<std::uint64_t>, kTableSize>();
    for (std::size_t i = 0; i < kTableSize; ++i) {
      (*v)[i] = std::make_shared<std::uint64_t>(i);
    }
    return v;
  }();
  std::size_t idx = static_cast<std::size_t>(state.thread_index()) * 131U;
  for (auto _ : state) {
    idx = (idx + 1U) & (kTableSize - 1U);
    auto local = (*g_tbl)[idx];
    benchmark::DoNotOptimize(*local);
  }
}
BENCHMARK(BM_06a_Table2K_StdSharedPtr)->Apply(apply_thread_sweep);

void BM_06b_Table2K_HyperSharedPtr(benchmark::State& state) {
  static const auto* g_tbl = []() {
    auto* v =
        new std::array<theta::HyperSharedPtr<std::uint64_t>, kTableSize>();
    for (std::size_t i = 0; i < kTableSize; ++i) {
      (*v)[i] = theta::make_hyper_shared<std::uint64_t>(i);
    }
    return v;
  }();
  std::size_t idx = static_cast<std::size_t>(state.thread_index()) * 131U;
  for (auto _ : state) {
    idx = (idx + 1U) & (kTableSize - 1U);
    auto local = (*g_tbl)[idx];
    benchmark::DoNotOptimize(*local);
  }
}
BENCHMARK(BM_06b_Table2K_HyperSharedPtr)->Apply(apply_thread_sweep);

void BM_06c_Table2K_ThetaSharedPtr(benchmark::State& state) {
  static const auto* g_tbl = []() {
    auto* v = new std::array<theta::SharedPtr<std::uint64_t>, kTableSize>();
    for (std::size_t i = 0; i < kTableSize; ++i) {
      (*v)[i] = theta::make_shared<std::uint64_t>(i);
    }
    return v;
  }();
  std::size_t idx = static_cast<std::size_t>(state.thread_index()) * 131U;
  for (auto _ : state) {
    idx = (idx + 1U) & (kTableSize - 1U);
    auto local = (*g_tbl)[idx];
    benchmark::DoNotOptimize(*local);
  }
}
BENCHMARK(BM_06c_Table2K_ThetaSharedPtr)->Apply(apply_thread_sweep);

}  // namespace

BENCHMARK_MAIN();
