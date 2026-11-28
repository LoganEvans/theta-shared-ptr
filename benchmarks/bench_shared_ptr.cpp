// FILENAME: benchmarks/bench_shared_ptr.cpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT
//
// Head-to-head microarchitectural benchmark suite comparing:
//   (a) `StdSharedPtr`   — stock `std::shared_ptr<T>` (`LOCK XADD` on 1 line)
//   (b) `ThetaSharedPtr` — 100% `std::shared_ptr` ABI-compatible `rseq`
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
#include <theta/portable/shared_ptr.hpp>
#include <theta/shared_ptr.hpp>
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
// 01. Unshared Creation + Destruction (`make_shared` + ~SharedPtr)
// ============================================================================
void BM_01a_CreateDestroy_StdSharedPtr(benchmark::State& state) {
  for (auto _ : state) {
    auto p = std::make_shared<std::uint64_t>(42ULL);
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK(BM_01a_CreateDestroy_StdSharedPtr)->Apply(apply_thread_sweep);

void BM_01b_CreateDestroy_ThetaSharedPtr(benchmark::State& state) {
  for (auto _ : state) {
    auto p = theta::make_shared<std::uint64_t>(42ULL);
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK(BM_01b_CreateDestroy_ThetaSharedPtr)->Apply(apply_thread_sweep);

void BM_01c_CreateDestroy_PortableSharedPtr(benchmark::State& state) {
  for (auto _ : state) {
    auto p = theta::portable::make_shared<std::uint64_t>(42ULL);
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK(BM_01c_CreateDestroy_PortableSharedPtr)->Apply(apply_thread_sweep);

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

void BM_02b_HotShare0to1_ThetaSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::SharedPtr<std::uint64_t>(
      theta::make_shared<std::uint64_t>(7ULL));
  for (auto _ : state) {
    theta::SharedPtr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_02b_HotShare0to1_ThetaSharedPtr)->Apply(apply_thread_sweep);

void BM_02c_HotShare0to1_PortableSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::portable::SharedPtr<std::uint64_t>(
      theta::portable::make_shared<std::uint64_t>(7ULL));
  for (auto _ : state) {
    theta::portable::SharedPtr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_02c_HotShare0to1_PortableSharedPtr)->Apply(apply_thread_sweep);

// ============================================================================
// 03. Pinned Hot Sharing (`1 -> 2 -> 1`)
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

void BM_03b_HotSharePinned_ThetaSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::SharedPtr<std::uint64_t>(
      theta::make_shared<std::uint64_t>(9ULL));
  theta::SharedPtr<std::uint64_t> pinned = *g_ptr;
  benchmark::DoNotOptimize(pinned);
  for (auto _ : state) {
    theta::SharedPtr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_03b_HotSharePinned_ThetaSharedPtr)->Apply(apply_thread_sweep);

void BM_03c_HotSharePinned_PortableSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::portable::SharedPtr<std::uint64_t>(
      theta::portable::make_shared<std::uint64_t>(9ULL));
  theta::portable::SharedPtr<std::uint64_t> pinned = *g_ptr;
  benchmark::DoNotOptimize(pinned);
  for (auto _ : state) {
    theta::portable::SharedPtr<std::uint64_t> local = *g_ptr;
    benchmark::DoNotOptimize(local);
  }
}
BENCHMARK(BM_03c_HotSharePinned_PortableSharedPtr)->Apply(apply_thread_sweep);

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
    for (auto _ : state) {
      std::uint64_t t = ring.tail.load(std::memory_order_relaxed);
      while (t - ring.head.load(std::memory_order_acquire) >=
             SpscRing<PtrType>::kCap) {
        theta::arch::cpu_relax();
      }
      ring.slots[t & (SpscRing<PtrType>::kCap - 1U)] = proto;
      ring.tail.store(t + 1ULL, std::memory_order_release);
    }
    while (ring.head.load(std::memory_order_acquire) !=
           ring.tail.load(std::memory_order_relaxed)) {
      theta::arch::cpu_relax();
    }
  } else {
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

void BM_04b_CrossThreadProdCons_ThetaSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::SharedPtr<std::uint64_t>(
      theta::make_shared<std::uint64_t>(11ULL));
  run_prod_cons_bench(state, *g_ptr);
}
BENCHMARK(BM_04b_CrossThreadProdCons_ThetaSharedPtr)->Apply(apply_pair_sweep);

void BM_04c_CrossThreadProdCons_PortableSharedPtr(benchmark::State& state) {
  static const auto* g_ptr = new theta::portable::SharedPtr<std::uint64_t>(
      theta::portable::make_shared<std::uint64_t>(11ULL));
  run_prod_cons_bench(state, *g_ptr);
}
BENCHMARK(BM_04c_CrossThreadProdCons_PortableSharedPtr)
    ->Apply(apply_pair_sweep);

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

void BM_05b_KeepAliveSnapshot_ThetaKeepAlive(benchmark::State& state) {
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
BENCHMARK(BM_05b_KeepAliveSnapshot_ThetaKeepAlive)->Apply(apply_thread_sweep);

void BM_05c_KeepAliveSnapshot_PortableKeepAlive(benchmark::State& state) {
  static auto* g_ka = new theta::portable::KeepAlive<std::uint64_t>(
      theta::portable::make_shared<std::uint64_t>(1ULL));
  std::uint64_t iter = 0;
  for (auto _ : state) {
    if (state.thread_index() == 0 && ((++iter & 16383ULL) == 0ULL))
        [[unlikely]] {
      (void)g_ka->reset(theta::portable::make_shared<std::uint64_t>(iter));
    } else {
      auto snap = g_ka->get();
      benchmark::DoNotOptimize(*snap);
    }
  }
}
BENCHMARK(BM_05c_KeepAliveSnapshot_PortableKeepAlive)
    ->Apply(apply_thread_sweep);

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
  std::uint64_t idx =
      static_cast<std::uint64_t>(state.thread_index()) * 2654435761ULL;
  for (auto _ : state) {
    idx = (idx + 17ULL) & (kTableSize - 1U);
    auto cp = (*g_tbl)[idx];
    benchmark::DoNotOptimize(cp);
  }
}
BENCHMARK(BM_06a_Table2K_StdSharedPtr)->Apply(apply_thread_sweep);

void BM_06b_Table2K_ThetaSharedPtr(benchmark::State& state) {
  static const auto* g_tbl = []() {
    auto* v = new std::array<theta::SharedPtr<std::uint64_t>, kTableSize>();
    for (std::size_t i = 0; i < kTableSize; ++i) {
      (*v)[i] = theta::make_shared<std::uint64_t>(i);
    }
    return v;
  }();
  std::uint64_t idx =
      static_cast<std::uint64_t>(state.thread_index()) * 2654435761ULL;
  for (auto _ : state) {
    idx = (idx + 17ULL) & (kTableSize - 1U);
    auto cp = (*g_tbl)[idx];
    benchmark::DoNotOptimize(cp);
  }
}
BENCHMARK(BM_06b_Table2K_ThetaSharedPtr)->Apply(apply_thread_sweep);

void BM_06c_Table2K_PortableSharedPtr(benchmark::State& state) {
  static const auto* g_tbl = []() {
    auto* v =
        new std::array<theta::portable::SharedPtr<std::uint64_t>, kTableSize>();
    for (std::size_t i = 0; i < kTableSize; ++i) {
      (*v)[i] = theta::portable::make_shared<std::uint64_t>(i);
    }
    return v;
  }();
  std::uint64_t idx =
      static_cast<std::uint64_t>(state.thread_index()) * 2654435761ULL;
  for (auto _ : state) {
    idx = (idx + 17ULL) & (kTableSize - 1U);
    auto cp = (*g_tbl)[idx];
    benchmark::DoNotOptimize(cp);
  }
}
BENCHMARK(BM_06c_Table2K_PortableSharedPtr)->Apply(apply_thread_sweep);

// ============================================================================
// 07. Ephemeral Cross-Thread Create + 1-Copy + Immediate Destroy (Max Rate)
//     Producer constructs a fresh object on CPU A, copies it (`1 -> 2`) into
//     an outbound SPSC queue while finishing local scope work, and drops its
//     local ref (`2 -> 1`); Consumer pops on CPU B and immediately destroys the
//     final ref (`1 -> 0`).
// ============================================================================
template <typename PtrType, typename MakeFn>
void run_ephemeral_cross_cpu_bench(benchmark::State& state, MakeFn make_fn,
                                   std::uint32_t work_spins) {
  static std::array<SpscRing<PtrType>, 128> rings{};
  const int pair_id = state.thread_index() / 2;
  auto& ring = rings[static_cast<std::size_t>(pair_id) & 127U];

  if (state.thread_index() % 2 == 0) {
    std::uint64_t seq = 0;
    for (auto _ : state) {
      for (std::uint32_t s = 0; s < work_spins; ++s) {
        theta::arch::cpu_relax();
      }
      PtrType local = make_fn(++seq);
      std::uint64_t t = ring.tail.load(std::memory_order_relaxed);
      while (t - ring.head.load(std::memory_order_acquire) >=
             SpscRing<PtrType>::kCap) {
        theta::arch::cpu_relax();
      }
      ring.slots[t & (SpscRing<PtrType>::kCap - 1U)] = local;
      ring.tail.store(t + 1ULL, std::memory_order_release);
      benchmark::DoNotOptimize(local);
    }
    while (ring.head.load(std::memory_order_acquire) !=
           ring.tail.load(std::memory_order_relaxed)) {
      theta::arch::cpu_relax();
    }
  } else {
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

void BM_07a_EphemeralCrossCpu_StdSharedPtr(benchmark::State& state) {
  run_ephemeral_cross_cpu_bench<std::shared_ptr<std::uint64_t>>(
      state, [](std::uint64_t v) { return std::make_shared<std::uint64_t>(v); },
      /*work_spins=*/0U);
}
BENCHMARK(BM_07a_EphemeralCrossCpu_StdSharedPtr)->Apply(apply_pair_sweep);

void BM_07b_EphemeralCrossCpu_ThetaSharedPtr(benchmark::State& state) {
  run_ephemeral_cross_cpu_bench<theta::SharedPtr<std::uint64_t>>(
      state,
      [](std::uint64_t v) { return theta::make_shared<std::uint64_t>(v); },
      /*work_spins=*/0U);
}
BENCHMARK(BM_07b_EphemeralCrossCpu_ThetaSharedPtr)->Apply(apply_pair_sweep);

void BM_07c_EphemeralCrossCpu_PortableSharedPtr(benchmark::State& state) {
  run_ephemeral_cross_cpu_bench<theta::portable::SharedPtr<std::uint64_t>>(
      state,
      [](std::uint64_t v) {
        return theta::portable::make_shared<std::uint64_t>(v);
      },
      /*work_spins=*/0U);
}
BENCHMARK(BM_07c_EphemeralCrossCpu_PortableSharedPtr)->Apply(apply_pair_sweep);

// ============================================================================
// 08. Mixed Workload: 98.4% Hot Shared-Cache Reads + 1.6% Ephemeral Cross-CPU
//     Create/Copy/Destroy + Periodic Cache Rotation
//     Models a real RPC/storage server where worker threads read a periodically
//     refreshed shared handle (`63/64` ops) while also passing 1-copy ephemeral
//     per-request handles across CPUs (`1/64` ops).
// ============================================================================
template <typename PtrType>
struct HotVersionRing {
  static constexpr std::size_t kSlots = 65536;
  alignas(64) std::atomic<std::size_t> ver{0};
  std::array<PtrType, kSlots> slots{};
};

template <typename PtrType, typename MakeFn>
void run_mixed_hot_and_ephemeral_bench(benchmark::State& state,
                                       MakeFn make_fn) {
  static std::array<SpscRing<PtrType>, 128> rings{};
  static auto* g_hot = [make_fn]() {
    auto* hr = new HotVersionRing<PtrType>();
    hr->slots[0] = make_fn(1ULL);
    return hr;
  }();

  const int tid = state.thread_index();
  const int pair_id = tid / 2;
  auto& ring = rings[static_cast<std::size_t>(pair_id) & 127U];

  std::uint64_t iter = 0;
  for (auto _ : state) {
    ++iter;
    if (tid == 0 && ((iter & 4095ULL) == 0ULL)) [[unlikely]] {
      const std::size_t cur = g_hot->ver.load(std::memory_order_relaxed);
      const std::size_t nxt = cur + 1U;
      if (nxt < HotVersionRing<PtrType>::kSlots) [[likely]] {
        g_hot->slots[nxt] = make_fn(iter);
        g_hot->ver.store(nxt, std::memory_order_release);
      }
    }

    if ((iter & 63ULL) == 0ULL) [[unlikely]] {
      if (tid % 2 == 0) {
        PtrType local = make_fn(iter);
        std::uint64_t t = ring.tail.load(std::memory_order_relaxed);
        while (t - ring.head.load(std::memory_order_acquire) >=
               SpscRing<PtrType>::kCap) {
          theta::arch::cpu_relax();
        }
        ring.slots[t & (SpscRing<PtrType>::kCap - 1U)] = local;
        ring.tail.store(t + 1ULL, std::memory_order_release);
        benchmark::DoNotOptimize(local);
      } else {
        std::uint64_t h = ring.head.load(std::memory_order_relaxed);
        while (ring.tail.load(std::memory_order_acquire) == h) {
          theta::arch::cpu_relax();
        }
        PtrType item =
            std::move(ring.slots[h & (SpscRing<PtrType>::kCap - 1U)]);
        ring.head.store(h + 1ULL, std::memory_order_release);
        benchmark::DoNotOptimize(item);
      }
    } else {
      const std::size_t v = g_hot->ver.load(std::memory_order_acquire);
      PtrType snap = g_hot->slots[v];
      benchmark::DoNotOptimize(snap);
    }
  }

  if (tid % 2 == 0) {
    while (ring.head.load(std::memory_order_acquire) !=
           ring.tail.load(std::memory_order_relaxed)) {
      theta::arch::cpu_relax();
    }
  }
}

void BM_08a_MixedHotAndEphemeral_StdSharedPtr(benchmark::State& state) {
  run_mixed_hot_and_ephemeral_bench<std::shared_ptr<std::uint64_t>>(
      state,
      [](std::uint64_t v) { return std::make_shared<std::uint64_t>(v); });
}
BENCHMARK(BM_08a_MixedHotAndEphemeral_StdSharedPtr)->Apply(apply_pair_sweep);

void BM_08b_MixedHotAndEphemeral_ThetaSharedPtr(benchmark::State& state) {
  run_mixed_hot_and_ephemeral_bench<theta::SharedPtr<std::uint64_t>>(
      state,
      [](std::uint64_t v) { return theta::make_shared<std::uint64_t>(v); });
}
BENCHMARK(BM_08b_MixedHotAndEphemeral_ThetaSharedPtr)->Apply(apply_pair_sweep);

void BM_08c_MixedHotAndEphemeral_PortableSharedPtr(benchmark::State& state) {
  run_mixed_hot_and_ephemeral_bench<theta::portable::SharedPtr<std::uint64_t>>(
      state, [](std::uint64_t v) {
        return theta::portable::make_shared<std::uint64_t>(v);
      });
}
BENCHMARK(BM_08c_MixedHotAndEphemeral_PortableSharedPtr)
    ->Apply(apply_pair_sweep);

// ============================================================================
// 09. Hot Concurrent `std::atomic<shared_ptr<T>>::load()` (Pure Read Scaling)
// ============================================================================
void BM_09a_AtomicLoadHot_StdSharedPtr(benchmark::State& state) {
  static auto* g_cell = new std::atomic<std::shared_ptr<std::uint64_t>>(
      std::make_shared<std::uint64_t>(77ULL));
  for (auto _ : state) {
    auto snap = g_cell->load(std::memory_order_acquire);
    benchmark::DoNotOptimize(*snap);
  }
}
BENCHMARK(BM_09a_AtomicLoadHot_StdSharedPtr)->Apply(apply_thread_sweep);

void BM_09b_AtomicLoadHot_ThetaSharedPtr(benchmark::State& state) {
  static auto* g_cell = new std::atomic<theta::SharedPtr<std::uint64_t>>(
      theta::make_shared<std::uint64_t>(77ULL));
  for (auto _ : state) {
    auto snap = g_cell->load(std::memory_order_acquire);
    benchmark::DoNotOptimize(*snap);
  }
}
BENCHMARK(BM_09b_AtomicLoadHot_ThetaSharedPtr)->Apply(apply_thread_sweep);

void BM_09c_AtomicLoadHot_PortableSharedPtr(benchmark::State& state) {
  static auto* g_cell =
      new std::atomic<theta::portable::SharedPtr<std::uint64_t>>(
          theta::portable::make_shared<std::uint64_t>(77ULL));
  for (auto _ : state) {
    auto snap = g_cell->load(std::memory_order_acquire);
    benchmark::DoNotOptimize(*snap);
  }
}
BENCHMARK(BM_09c_AtomicLoadHot_PortableSharedPtr)->Apply(apply_thread_sweep);

// ============================================================================
// 10. `std::atomic<shared_ptr<T>>` Live Snapshot Reload (16,383 Reads : 1 Swap)
//     Direct C++20 `std::atomic` counterpart to `BM_05` (`KeepAliveSnapshot`).
// ============================================================================
void BM_10a_AtomicSnapshotReload_StdSharedPtr(benchmark::State& state) {
  static auto* g_cell = new std::atomic<std::shared_ptr<std::uint64_t>>(
      std::make_shared<std::uint64_t>(1ULL));
  std::uint64_t iter = 0;
  for (auto _ : state) {
    if (state.thread_index() == 0 && ((++iter & 16383ULL) == 0ULL))
        [[unlikely]] {
      g_cell->store(std::make_shared<std::uint64_t>(iter),
                    std::memory_order_release);
    } else {
      auto snap = g_cell->load(std::memory_order_acquire);
      benchmark::DoNotOptimize(*snap);
    }
  }
}
BENCHMARK(BM_10a_AtomicSnapshotReload_StdSharedPtr)->Apply(apply_thread_sweep);

void BM_10b_AtomicSnapshotReload_ThetaSharedPtr(benchmark::State& state) {
  static auto* g_cell = new std::atomic<theta::SharedPtr<std::uint64_t>>(
      theta::make_shared<std::uint64_t>(1ULL));
  std::uint64_t iter = 0;
  for (auto _ : state) {
    if (state.thread_index() == 0 && ((++iter & 16383ULL) == 0ULL))
        [[unlikely]] {
      g_cell->store(theta::make_shared<std::uint64_t>(iter),
                    std::memory_order_release);
    } else {
      auto snap = g_cell->load(std::memory_order_acquire);
      benchmark::DoNotOptimize(*snap);
    }
  }
}
BENCHMARK(BM_10b_AtomicSnapshotReload_ThetaSharedPtr)
    ->Apply(apply_thread_sweep);

void BM_10c_AtomicSnapshotReload_PortableSharedPtr(benchmark::State& state) {
  static auto* g_cell =
      new std::atomic<theta::portable::SharedPtr<std::uint64_t>>(
          theta::portable::make_shared<std::uint64_t>(1ULL));
  std::uint64_t iter = 0;
  for (auto _ : state) {
    if (state.thread_index() == 0 && ((++iter & 16383ULL) == 0ULL))
        [[unlikely]] {
      g_cell->store(theta::portable::make_shared<std::uint64_t>(iter),
                    std::memory_order_release);
    } else {
      auto snap = g_cell->load(std::memory_order_acquire);
      benchmark::DoNotOptimize(*snap);
    }
  }
}
BENCHMARK(BM_10c_AtomicSnapshotReload_PortableSharedPtr)
    ->Apply(apply_thread_sweep);

// ============================================================================
// 11. Contended Write-Heavy `compare_exchange_weak` Churn (100% CAS Mutations)
//     Verifies adaptive `atomic_pin_reads` warmup prevents `SYS_membarrier`
//     storms on write-dominated `std::atomic<theta::SharedPtr<T>>` cells.
// ============================================================================
void BM_11a_AtomicCasChurn_StdSharedPtr(benchmark::State& state) {
  static auto* g_cell = new std::atomic<std::shared_ptr<std::uint64_t>>(
      std::make_shared<std::uint64_t>(0ULL));
  for (auto _ : state) {
    auto cur = g_cell->load(std::memory_order_acquire);
    for (;;) {
      auto next = std::make_shared<std::uint64_t>(*cur + 1ULL);
      if (g_cell->compare_exchange_weak(cur, std::move(next),
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        break;
      }
    }
  }
}
BENCHMARK(BM_11a_AtomicCasChurn_StdSharedPtr)->Apply(apply_thread_sweep);

void BM_11b_AtomicCasChurn_ThetaSharedPtr(benchmark::State& state) {
  static auto* g_cell = new std::atomic<theta::SharedPtr<std::uint64_t>>(
      theta::make_shared<std::uint64_t>(0ULL));
  for (auto _ : state) {
    auto cur = g_cell->load(std::memory_order_acquire);
    for (;;) {
      auto next = theta::make_shared<std::uint64_t>(*cur + 1ULL);
      if (g_cell->compare_exchange_weak(cur, std::move(next),
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        break;
      }
    }
  }
}
BENCHMARK(BM_11b_AtomicCasChurn_ThetaSharedPtr)->Apply(apply_thread_sweep);

void BM_11c_AtomicCasChurn_PortableSharedPtr(benchmark::State& state) {
  static auto* g_cell =
      new std::atomic<theta::portable::SharedPtr<std::uint64_t>>(
          theta::portable::make_shared<std::uint64_t>(0ULL));
  for (auto _ : state) {
    auto cur = g_cell->load(std::memory_order_acquire);
    for (;;) {
      auto next = theta::portable::make_shared<std::uint64_t>(*cur + 1ULL);
      if (g_cell->compare_exchange_weak(cur, std::move(next),
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        break;
      }
    }
  }
}
BENCHMARK(BM_11c_AtomicCasChurn_PortableSharedPtr)->Apply(apply_thread_sweep);

}  // namespace

BENCHMARK_MAIN();
