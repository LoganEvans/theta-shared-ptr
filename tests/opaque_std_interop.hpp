// FILENAME: tests/opaque_std_interop.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

namespace theta::test_opaque {

struct OpaquePayload {
  static inline std::atomic<int> live_count{0};
  static inline std::atomic<int> dtor_count{0};

  std::uint64_t magic{0xC0FFEE42ULL};
  int value{0};

  explicit OpaquePayload(int v) noexcept : value(v) {
    live_count.fetch_add(1, std::memory_order_relaxed);
  }

  ~OpaquePayload() noexcept {
    magic = 0xDEADDEADULL;
    live_count.fetch_sub(1, std::memory_order_relaxed);
    dtor_count.fetch_add(1, std::memory_order_relaxed);
  }
};

// Functions compiled in `opaque_std_interop.cpp` with ZERO Theta headers
int opaque_read_and_copy_churn(const std::shared_ptr<OpaquePayload>& sp,
                               int copies);
std::shared_ptr<OpaquePayload> opaque_clone_by_value(
    std::shared_ptr<OpaquePayload> sp);
std::shared_ptr<OpaquePayload> opaque_make_native_std(int v);
bool opaque_verify_owner_equivalence(const std::shared_ptr<OpaquePayload>& a,
                                     const std::shared_ptr<OpaquePayload>& b,
                                     const std::weak_ptr<OpaquePayload>& w);
int opaque_lock_weak_and_read(const std::weak_ptr<OpaquePayload>& w,
                              int fallback);

}  // namespace theta::test_opaque
