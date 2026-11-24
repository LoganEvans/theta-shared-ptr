// FILENAME: tests/test_hardened.cpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT
//
// 5 kHz `SIGPROF` preemption-storm stress test continuously interrupting
// oversubscribed threads inside `rseq` critical sections (`SharedPtr` +
// `KeepAlive`) and per-CPU atomic shards (`HyperSharedPtr`).

#include <gtest/gtest.h>

#include <atomic>
#include <csignal>
#include <cstdint>
#include <theta/hyper_shared_ptr.hpp>
#include <thread>
#include <vector>

#if THETA_OS_LINUX
#include <sys/time.h>
#endif

namespace theta {
namespace {

std::atomic<std::uint64_t> g_sigprof_ticks{0};

#if THETA_OS_LINUX
extern "C" void sigprof_preempt_handler(int /*sig*/) noexcept {
  g_sigprof_ticks.fetch_add(1ULL, std::memory_order_relaxed);
  std::this_thread::yield();
}
#endif

struct StormPayload {
  static inline std::atomic<int> live{0};
  std::uint64_t cookie{0xA5A55A5AF00DFACEULL};
  int epoch{0};

  explicit StormPayload(int e) noexcept : epoch(e) {
    live.fetch_add(1, std::memory_order_relaxed);
  }

  ~StormPayload() noexcept {
    cookie = 0ULL;
    live.fetch_sub(1, std::memory_order_relaxed);
  }
};

TEST(HardenedPreemptionStormTest, RseqAbortAndKeepAliveUnder5kHzSignals) {
  StormPayload::live.store(0, std::memory_order_seq_cst);
  g_sigprof_ticks.store(0ULL, std::memory_order_relaxed);

#if THETA_OS_LINUX && !THETA_TSAN_ENABLED
  struct sigaction sa{};
  struct sigaction old_sa{};
  sa.sa_handler = &sigprof_preempt_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;
  const bool timer_installed = (::sigaction(SIGPROF, &sa, &old_sa) == 0);

  struct itimerval itv{};
  if (timer_installed) {
    itv.it_interval.tv_sec = 0;
    itv.it_interval.tv_usec = 200;  // 5 kHz preemption interrupts
    itv.it_value = itv.it_interval;
    (void)::setitimer(ITIMER_PROF, &itv, nullptr);
  }
#endif

  {
    KeepAlive<StormPayload> ka(make_shared<StormPayload>(1));
    auto hot_abi = make_shared<StormPayload>(777);
    auto hot_hsp = make_hyper_shared<StormPayload>(888);

    std::atomic<bool> stop{false};
    constexpr int kWorkers = 12;
    std::vector<std::thread> workers;
    workers.reserve(kWorkers);

    for (int t = 0; t < kWorkers; ++t) {
      workers.emplace_back([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
          for (int i = 0; i < 64; ++i) {
            auto s1 = ka.get();
            if (s1) {
              ASSERT_EQ(s1->cookie, 0xA5A55A5AF00DFACEULL);
            }
            auto s2 = hot_abi;
            ASSERT_EQ(s2->epoch, 777);
            auto s3 = hot_hsp;
            ASSERT_EQ(s3->epoch, 888);
          }
        }
      });
    }

    for (int gen = 2; gen <= 120; ++gen) {
      (void)ka.reset(make_shared<StormPayload>(gen));
    }

    stop.store(true, std::memory_order_release);
    for (auto& w : workers) {
      w.join();
    }
  }

#if THETA_OS_LINUX && !THETA_TSAN_ENABLED
  if (timer_installed) {
    struct itimerval stop_itv{};
    (void)::setitimer(ITIMER_PROF, &stop_itv, nullptr);
    (void)::sigaction(SIGPROF, &old_sa, nullptr);
  }
#endif

  EXPECT_EQ(StormPayload::live.load(), 0);
}

}  // namespace
}  // namespace theta
