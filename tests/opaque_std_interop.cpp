// FILENAME: tests/opaque_std_interop.cpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT
//
// Deliberately includes ONLY standard C++ `<memory>` / `<map>` / `<vector>`
// (never any `theta/*` header) so the compiler emits stock libstdc++ inline
// `_Sp_counted_base` operations against `ThetaSpCountedBase` instances.

#include "opaque_std_interop.hpp"

#include <map>
#include <memory>
#include <vector>

namespace theta::test_opaque {

int opaque_read_and_copy_churn(const std::shared_ptr<OpaquePayload>& sp,
                               int copies) {
  if (!sp) {
    return -1;
  }
  std::vector<std::shared_ptr<OpaquePayload>> vec;
  vec.reserve(static_cast<std::size_t>(copies));
  for (int i = 0; i < copies; ++i) {
    vec.push_back(sp);
  }
  int acc = 0;
  for (const auto& item : vec) {
    if (item->magic == 0xC0FFEE42ULL) {
      acc += item->value;
    }
  }
  return acc / (copies > 0 ? copies : 1);
}

std::shared_ptr<OpaquePayload> opaque_clone_by_value(
    std::shared_ptr<OpaquePayload> sp) {
  std::weak_ptr<OpaquePayload> wp = sp;
  auto locked = wp.lock();
  return locked ? locked : sp;
}

std::shared_ptr<OpaquePayload> opaque_make_native_std(int v) {
  return std::make_shared<OpaquePayload>(v);
}

bool opaque_verify_owner_equivalence(const std::shared_ptr<OpaquePayload>& a,
                                     const std::shared_ptr<OpaquePayload>& b,
                                     const std::weak_ptr<OpaquePayload>& w) {
  std::owner_less<std::shared_ptr<OpaquePayload>> sp_less;
  if (sp_less(a, b) || sp_less(b, a)) {
    return false;
  }
  std::map<std::weak_ptr<OpaquePayload>, int,
           std::owner_less<std::weak_ptr<OpaquePayload>>>
      owner_map;
  owner_map[w] = 99;
  std::weak_ptr<OpaquePayload> w2(a);
  return owner_map.find(w2) != owner_map.end() && owner_map[w2] == 99;
}

int opaque_lock_weak_and_read(const std::weak_ptr<OpaquePayload>& w,
                              int fallback) {
  if (auto sp = w.lock()) {
    return (sp->magic == 0xC0FFEE42ULL) ? sp->value : -999;
  }
  return fallback;
}

}  // namespace theta::test_opaque
