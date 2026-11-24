// FILENAME: include/theta/hyper-shared-ptr/hyper-shared-ptr.h
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <theta/hyper_shared_ptr.hpp>

namespace theta::hsp {

// Backwards-compatible aliases preserving `theta::hsp::HyperSharedPointer<T>`
// and `theta::hsp::KeepAlive<T>` for downstream consumers (`theta-epoch-ptr`).
template <typename T>
using HyperSharedPointer = ::theta::HyperSharedPtr<T>;

template <typename T>
using KeepAlive = ::theta::HyperKeepAlive<T>;

// NOLINTNEXTLINE(readability-identifier-naming)
inline int getCpu() noexcept {
  return static_cast<int>(::theta::hyper_shared_ptr::detail::current_cpu_id());
}

}  // namespace theta::hsp
