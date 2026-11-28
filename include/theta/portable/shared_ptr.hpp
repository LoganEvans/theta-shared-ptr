// FILENAME: include/theta/portable/shared_ptr.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT
//
// `theta::portable::SharedPtr<T>` / `AtomicSharedPtr<T>` / `KeepAlive<T>`
// ----------------------------------------------------------------------------
// Portable concurrent smart pointer & atomic snapshot suite with zero Linux
// `rseq`, zero `<pthread.h>`, and zero `SYS_membarrier` IPIs:
//   1. `SharedPtr<T>` (`16 B`, `std::shared_ptr<T>` ABI-compatible):
//      - `0` copies: zero-atomic, zero-vcall sole-owner destroy (`hdr64 == 0`).
//      - `1..63` copies: lock-free inline `slot_id` warmup CAS (`0 B` slab).
//      - `>= 64` copies: dedicated per-thread L1d-private slab (`CAS` on
//        `copy`/dtor, harvested via Dekker gate + `atomic_exchange(cell, 0)`).
//   2. `AtomicSharedPtr<T>` (`16 B`, `std::atomic<SharedPtr<T>>`) &
//      `KeepAlive<T>`:
//      - Reader `.load()` / `.get()` enters a 64B-isolated per-thread 2-party
//        Dekker byte gate (`[harvester_busy, owner_busy]`: red-zone
//        `lock orl $0, -8(%rsp)` on x86-64; disjoint-byte `STLRB -> LDARB`
//        `RCsc` with zero `dmb ish` on ARM64), validates the pointer, bumps
//        the single-owner L1d slab cell with a plain non-RMW store, and exits
//        the gate (`0` shared cache-line writes, `0` `CAS` on `.load()`).

#pragma once

#include <atomic>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <theta/arch.hpp>
#include <theta/portable/detail/common.hpp>
#include <theta/shared_ptr.hpp>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace theta::portable {

template <typename T>
class SharedPtr;

template <typename T>
class WeakPtr;

template <typename T>
class EnableSharedFromThis;

template <typename T = void>
struct OwnerLess;

template <typename T>
class KeepAlive;

template <typename T>
using AtomicSharedPtr = std::atomic<SharedPtr<T>>;

template <typename T>
using AtomicWeakPtr = std::atomic<WeakPtr<T>>;

template <typename T>
using shared_ptr = SharedPtr<T>;

template <typename T>
using weak_ptr = WeakPtr<T>;

template <typename T>
using atomic_shared_ptr = std::atomic<SharedPtr<T>>;

template <typename T>
using atomic_weak_ptr = std::atomic<WeakPtr<T>>;

template <typename T>
using enable_shared_from_this = EnableSharedFromThis<T>;

template <typename T = void>
using owner_less = OwnerLess<T>;

namespace detail::shared_ptr {

using ::theta::detail::shared_ptr::AdoptRefTag;
using ::theta::detail::shared_ptr::ForOverwriteTag;
using ::theta::detail::shared_ptr::kSpCompatibleV;
using ::theta::detail::shared_ptr::kSpRawPtrCompatibleV;
using ::theta::detail::shared_ptr::RawStdPtrLayout;
using ::theta::detail::shared_ptr::sp_construct_copy_alloc;
using ::theta::detail::shared_ptr::sp_construct_copy_noalloc;
using ::theta::detail::shared_ptr::sp_construct_val_alloc;
using ::theta::detail::shared_ptr::sp_destroy_alloc_reverse;
using ::theta::detail::shared_ptr::sp_destroy_reverse;
using ::theta::detail::shared_ptr::SpRemoveCvTargetT;

THETA_FORCE_INLINE void sp_add_ref(StdSpCountedBase* pi) noexcept {
  if (pi == nullptr) {
    return;
  }
  if (!is_portable_sp(pi)) [[unlikely]] {
    pi->_M_add_ref_copy();
    return;
  }
  auto* cb = static_cast<PortableSpCountedBase*>(pi);
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) && \
    !THETA_TSAN_ENABLED
  using FastHdrWord [[gnu::may_alias]] = std::uint64_t;
  const std::uint64_t hdr64 = __atomic_load_n(
      reinterpret_cast<const FastHdrWord*>(&cb->fast_), __ATOMIC_ACQUIRE);
  if ((hdr64 & (0xFFULL << 32U)) != 0ULL) [[likely]] {
    const auto slot = static_cast<std::uint32_t>(hdr64);
    if (PortableBackend::try_stripe_inc_slot(slot)) [[likely]] {
      return;
    }
    sp_add_ref_slow(pi);
    return;
  }
  const auto mode = static_cast<std::uint8_t>((hdr64 >> 40U) & 0xFFULL);
  if (mode == kModeUnpromoted) [[likely]] {
    auto w = static_cast<std::uint32_t>(hdr64);
    while (w == 0U || ((w - kWarmupCopyBase) < kWarmupCopyRange &&
                       (w & 0xFFFFU) < 0xFFFEU)) [[likely]] {
      const std::uint32_t next_w =
          (w == 0U) ? kWarmupFirstCopyWord : (w + 0x00010001U);
      if (__atomic_compare_exchange_n(&cb->fast_.slot_id, &w, next_w, true,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED))
          [[likely]] {
        return;
      }
    }
  } else if (mode == kModeAtomicRetired) {
    pi->_M_add_ref_copy();
    return;
  }
#else
  if (__atomic_load_n(&cb->fast_.fast_active, __ATOMIC_ACQUIRE) != 0U)
      [[likely]] {
    const std::uint32_t slot =
        __atomic_load_n(&cb->fast_.slot_id, __ATOMIC_RELAXED);
    if (slot != 0U && slot < kMaxPortableSlots &&
        PortableBackend::try_stripe_inc_slot(slot)) [[likely]] {
      return;
    }
    sp_add_ref_slow(pi);
    return;
  }
  const std::uint8_t mode = __atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE);
  if (mode == kModeUnpromoted) [[likely]] {
    std::uint32_t w = __atomic_load_n(&cb->fast_.slot_id, __ATOMIC_RELAXED);
    while (w == 0U || ((w - kWarmupCopyBase) < kWarmupCopyRange &&
                       (w & 0xFFFFU) < 0xFFFEU)) [[likely]] {
      const std::uint32_t next_w =
          (w == 0U) ? kWarmupFirstCopyWord : (w + 0x00010001U);
      if (__atomic_compare_exchange_n(&cb->fast_.slot_id, &w, next_w, true,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED))
          [[likely]] {
        return;
      }
    }
  } else if (mode == kModeAtomicRetired) {
    pi->_M_add_ref_copy();
    return;
  }
#endif
  sp_add_ref_slow(pi);
}

THETA_FORCE_INLINE void sp_release(StdSpCountedBase* pi) noexcept {
  if (pi == nullptr) {
    return;
  }
  if (!is_portable_sp(pi)) [[unlikely]] {
    pi->_M_release();
    return;
  }
  auto* cb = static_cast<PortableSpCountedBase*>(pi);
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) && \
    !THETA_TSAN_ENABLED
  using FastHdrWord [[gnu::may_alias]] = std::uint64_t;
  const std::uint64_t hdr64 = __atomic_load_n(
      reinterpret_cast<const FastHdrWord*>(&cb->fast_), __ATOMIC_ACQUIRE);
  if ((hdr64 & (0xFFULL << 32U)) != 0ULL) [[likely]] {
    const auto slot = static_cast<std::uint32_t>(hdr64);
    if (PortableBackend::try_stripe_dec_slot(slot)) [[likely]] {
      return;
    }
    sp_release_slow(pi);
    return;
  }
  if (hdr64 == 0ULL) [[likely]] {
    const std::uintptr_t tag = cb->ops_or_size_;
    if ((tag & 1U) != 0U) [[likely]] {
      ::operator delete(static_cast<void*>(cb), tag >> 1U);
    } else {
      reinterpret_cast<const PortableColdOps*>(tag)->dispose_and_destroy(cb);
    }
    return;
  }
  const auto mode = static_cast<std::uint8_t>((hdr64 >> 40U) & 0xFFULL);
  if (mode == kModeUnpromoted) [[likely]] {
    auto w = static_cast<std::uint32_t>(hdr64);
    while ((w & kWarmupTagMask) == kWarmupActiveBit && (w & 0xFFFFU) != 0U)
        [[likely]] {
      if (__atomic_compare_exchange_n(&cb->fast_.slot_id, &w, w - 1U, true,
                                      __ATOMIC_RELEASE, __ATOMIC_RELAXED))
          [[likely]] {
        return;
      }
    }
  } else if (mode == kModeAtomicRetired) {
    pi->_M_release();
    return;
  }
#else
  if (__atomic_load_n(&cb->fast_.fast_active, __ATOMIC_ACQUIRE) != 0U)
      [[likely]] {
    const std::uint32_t slot =
        __atomic_load_n(&cb->fast_.slot_id, __ATOMIC_RELAXED);
    if (slot != 0U && slot < kMaxPortableSlots &&
        PortableBackend::try_stripe_dec_slot(slot)) [[likely]] {
      return;
    }
    sp_release_slow(pi);
    return;
  }
  const std::uint8_t mode = __atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE);
  if (mode == kModeUnpromoted) [[likely]] {
    std::uint32_t w = __atomic_load_n(&cb->fast_.slot_id, __ATOMIC_ACQUIRE);
    if (w == 0U && __atomic_load_n(&cb->fast_.has_weak_or_std,
                                   __ATOMIC_ACQUIRE) == 0U) [[likely]] {
      const std::uintptr_t tag = cb->ops_or_size_;
      if ((tag & 1U) != 0U) [[likely]] {
        ::operator delete(static_cast<void*>(cb), tag >> 1U);
      } else {
        reinterpret_cast<const PortableColdOps*>(tag)->dispose_and_destroy(cb);
      }
      return;
    }
    while ((w & kWarmupTagMask) == kWarmupActiveBit && (w & 0xFFFFU) != 0U)
        [[likely]] {
      if (__atomic_compare_exchange_n(&cb->fast_.slot_id, &w, w - 1U, true,
                                      __ATOMIC_RELEASE, __ATOMIC_RELAXED))
          [[likely]] {
        return;
      }
    }
  } else if (mode == kModeAtomicRetired) {
    pi->_M_release();
    return;
  }
#endif
  sp_release_slow(pi);
}

// ============================================================================
// Control-Block Storage Specializations
// ============================================================================
template <typename T>
struct SpInplaceStorage final {
  PortableSpCountedBase base;
  alignas(T) unsigned char storage[sizeof(T)];

  static constexpr bool kStandardAlign =
      alignof(T) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__;
  static constexpr bool kTrivialFastDelete =
      std::is_trivially_destructible_v<T> && kStandardAlign;

  static void* alloc_raw() {
    if constexpr (kStandardAlign) {
      return ::operator new(sizeof(SpInplaceStorage));
    } else {
      return ::operator new(sizeof(SpInplaceStorage),
                            std::align_val_t{alignof(SpInplaceStorage)});
    }
  }

  static void free_raw(void* p) noexcept {
    if constexpr (kStandardAlign) {
      ::operator delete(p, sizeof(SpInplaceStorage));
    } else {
      ::operator delete(p, sizeof(SpInplaceStorage),
                        std::align_val_t{alignof(SpInplaceStorage)});
    }
  }

  explicit SpInplaceStorage(ForOverwriteTag) : base(ops_or_size_value()) {
    ::new (static_cast<void*>(storage)) T;
  }

  template <typename... Args>
    requires(!(sizeof...(Args) == 1 &&
               (std::is_same_v<std::remove_cvref_t<Args>, ForOverwriteTag> &&
                ...)))
  explicit SpInplaceStorage(Args&&... args) : base(ops_or_size_value()) {
    ::new (static_cast<void*>(storage)) T(std::forward<Args>(args)...);
  }

  template <typename Elem>
    requires(std::is_bounded_array_v<T> &&
             std::is_same_v<SpRemoveCvTargetT<Elem>,
                            SpRemoveCvTargetT<std::remove_extent_t<T>>>)
  SpInplaceStorage(ForOverwriteTag, const Elem& fill_val)
      : base(ops_or_size_value()) {
    using TargetElem = std::remove_extent_t<T>;
    constexpr std::size_t kCount = std::extent_v<T>;
    auto* arr = reinterpret_cast<TargetElem*>(storage);
    std::size_t i = 0U;
    try {
      for (; i < kCount; ++i) {
        sp_construct_copy_noalloc(arr + i, fill_val);
      }
    } catch (...) {
      while (i > 0U) {
        --i;
        sp_destroy_reverse(arr + i);
      }
      throw;
    }
  }

  auto* ptr() noexcept {
    if constexpr (std::is_bounded_array_v<T>) {
      using Elem = std::remove_extent_t<T>;
      return std::launder(reinterpret_cast<Elem*>(storage));
    } else {
      return std::launder(reinterpret_cast<T*>(storage));
    }
  }

  static void dispose_object(SpInplaceStorage* self) noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      if constexpr (std::is_bounded_array_v<T>) {
        using Elem = std::remove_extent_t<T>;
        constexpr std::size_t kCount = std::extent_v<T>;
        auto* arr = std::launder(reinterpret_cast<Elem*>(self->storage));
        for (std::size_t i = kCount; i > 0U; --i) {
          sp_destroy_reverse(arr + i - 1U);
        }
      } else {
        std::destroy_at(self->ptr());
      }
    }
  }

  static void dispose_impl(PortableSpCountedBase* b) noexcept {
    dispose_object(reinterpret_cast<SpInplaceStorage*>(b));
  }

  static void destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpInplaceStorage*>(b);
    self->~SpInplaceStorage();
    free_raw(static_cast<void*>(self));
  }

  static void dispose_and_destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpInplaceStorage*>(b);
    dispose_object(self);
    self->~SpInplaceStorage();
    free_raw(static_cast<void*>(self));
  }

  static inline constexpr PortableColdOps kOps{
      std::is_trivially_destructible_v<T> ? nullptr : &dispose_impl,
      &destroy_impl,
      &dispose_and_destroy_impl,
      nullptr,
  };

  static constexpr std::uintptr_t ops_or_size_value() noexcept {
    if constexpr (kTrivialFastDelete) {
      return (sizeof(SpInplaceStorage) << 1U) | 1ULL;
    } else {
      return reinterpret_cast<std::uintptr_t>(&kOps);
    }
  }
};

template <typename T, typename Alloc, bool kForOverwrite = false>
struct SpAllocInplaceStorage final {
  using Scalar = std::remove_cv_t<std::remove_all_extents_t<T>>;
  using ValueAlloc =
      typename std::allocator_traits<Alloc>::template rebind_alloc<Scalar>;
  PortableSpCountedBase base;
  [[no_unique_address]] ValueAlloc alloc;
  alignas(T) unsigned char storage[sizeof(T)];

  using SelfAlloc = typename std::allocator_traits<
      ValueAlloc>::template rebind_alloc<SpAllocInplaceStorage>;

  SpAllocInplaceStorage(const Alloc& a, ForOverwriteTag)
    requires kForOverwrite
      : base(reinterpret_cast<std::uintptr_t>(&kOps)), alloc(a) {
    ::new (static_cast<void*>(storage)) T;
  }

  template <typename... Args>
    requires(!kForOverwrite &&
             !(sizeof...(Args) == 1 &&
               (std::is_same_v<std::remove_cvref_t<Args>, ForOverwriteTag> &&
                ...)))
  explicit SpAllocInplaceStorage(const Alloc& a, Args&&... args)
      : base(reinterpret_cast<std::uintptr_t>(&kOps)), alloc(a) {
    if constexpr (std::is_bounded_array_v<T>) {
      static_assert(sizeof...(Args) == 0U);
      sp_construct_val_alloc(alloc, reinterpret_cast<T*>(storage));
    } else {
      std::allocator_traits<ValueAlloc>::construct(
          alloc, reinterpret_cast<T*>(storage), std::forward<Args>(args)...);
    }
  }

  template <typename Elem>
    requires(!kForOverwrite && std::is_bounded_array_v<T> &&
             std::is_same_v<SpRemoveCvTargetT<Elem>,
                            SpRemoveCvTargetT<std::remove_extent_t<T>>>)
  SpAllocInplaceStorage(const Alloc& a, ForOverwriteTag, const Elem& fill_val)
      : base(reinterpret_cast<std::uintptr_t>(&kOps)), alloc(a) {
    using TargetElem = std::remove_extent_t<T>;
    constexpr std::size_t kCount = std::extent_v<T>;
    auto* arr = reinterpret_cast<TargetElem*>(storage);
    std::size_t i = 0U;
    try {
      for (; i < kCount; ++i) {
        sp_construct_copy_alloc(alloc, arr + i, fill_val);
      }
    } catch (...) {
      while (i > 0U) {
        --i;
        sp_destroy_alloc_reverse(alloc, arr + i);
      }
      throw;
    }
  }

  auto* ptr() noexcept {
    if constexpr (std::is_bounded_array_v<T>) {
      using Elem = std::remove_extent_t<T>;
      return std::launder(reinterpret_cast<Elem*>(storage));
    } else {
      return std::launder(reinterpret_cast<T*>(storage));
    }
  }

  static void dispose_object(SpAllocInplaceStorage* self) noexcept {
    if constexpr (kForOverwrite) {
      if constexpr (!std::is_trivially_destructible_v<T>) {
        if constexpr (std::is_bounded_array_v<T>) {
          using Elem = std::remove_extent_t<T>;
          constexpr std::size_t kCount = std::extent_v<T>;
          auto* arr = std::launder(reinterpret_cast<Elem*>(self->storage));
          for (std::size_t i = kCount; i > 0U; --i) {
            sp_destroy_reverse(arr + i - 1U);
          }
        } else {
          std::destroy_at(self->ptr());
        }
      }
    } else if constexpr (std::is_bounded_array_v<T>) {
      using Elem = std::remove_extent_t<T>;
      constexpr std::size_t kCount = std::extent_v<T>;
      auto* arr = std::launder(reinterpret_cast<Elem*>(self->storage));
      for (std::size_t i = kCount; i > 0U; --i) {
        sp_destroy_alloc_reverse(self->alloc, arr + i - 1U);
      }
    } else {
      std::allocator_traits<ValueAlloc>::destroy(self->alloc, self->ptr());
    }
  }

  static void dispose_impl(PortableSpCountedBase* b) noexcept {
    dispose_object(reinterpret_cast<SpAllocInplaceStorage*>(b));
  }

  static void destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocInplaceStorage*>(b);
    SelfAlloc sa(self->alloc);
    self->~SpAllocInplaceStorage();
    std::allocator_traits<SelfAlloc>::deallocate(sa, self, 1U);
  }

  static void dispose_and_destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocInplaceStorage*>(b);
    dispose_object(self);
    SelfAlloc sa(self->alloc);
    self->~SpAllocInplaceStorage();
    std::allocator_traits<SelfAlloc>::deallocate(sa, self, 1U);
  }

  static inline constexpr PortableColdOps kOps{
      &dispose_impl,
      &destroy_impl,
      &dispose_and_destroy_impl,
      nullptr,
  };
};

template <typename Elem, typename Alloc, bool kForOverwrite = false>
struct SpUnboundedArrayStorage final {
  using Scalar = std::remove_cv_t<std::remove_all_extents_t<Elem>>;
  using ValueAlloc =
      typename std::allocator_traits<Alloc>::template rebind_alloc<Scalar>;

  static constexpr std::size_t kChunkAlign = alignof(Elem) >
                                                     alignof(std::max_align_t)
                                                 ? alignof(Elem)
                                                 : alignof(std::max_align_t);

  struct alignas(kChunkAlign) RawChunk {
    unsigned char data[kChunkAlign];
  };

  using ChunkAlloc =
      typename std::allocator_traits<Alloc>::template rebind_alloc<RawChunk>;

  PortableSpCountedBase base;
  std::size_t count;
  std::size_t total_chunks;
  [[no_unique_address]] ValueAlloc alloc;

  static constexpr std::size_t header_bytes() noexcept {
    constexpr std::size_t kRaw = sizeof(SpUnboundedArrayStorage);
    constexpr std::size_t kAlign = alignof(Elem);
    return (kRaw + kAlign - 1U) & ~(kAlign - 1U);
  }

  static std::size_t compute_chunks(std::size_t n) {
    if (n > (static_cast<std::size_t>(-1) - header_bytes()) / sizeof(Elem)) {
      throw std::bad_array_new_length{};
    }
    const std::size_t bytes = header_bytes() + n * sizeof(Elem);
    return (bytes + sizeof(RawChunk) - 1U) / sizeof(RawChunk);
  }

  Elem* elems_ptr() noexcept {
    auto* raw = reinterpret_cast<unsigned char*>(this) + header_bytes();
    return std::launder(reinterpret_cast<Elem*>(raw));
  }

  template <typename InitFn>
  static SpUnboundedArrayStorage* create(const Alloc& a, std::size_t n,
                                         InitFn&& init_elem) {
    const std::size_t nchunks = compute_chunks(n);
    ChunkAlloc ca(a);
    RawChunk* mem = std::allocator_traits<ChunkAlloc>::allocate(ca, nchunks);
    auto* self =
        ::new (static_cast<void*>(mem)) SpUnboundedArrayStorage(a, n, nchunks);
    Elem* arr = reinterpret_cast<Elem*>(reinterpret_cast<unsigned char*>(self) +
                                        header_bytes());
    std::size_t i = 0U;
    try {
      for (; i < n; ++i) {
        init_elem(self->alloc, arr + i);
      }
    } catch (...) {
      while (i > 0U) {
        --i;
        if constexpr (kForOverwrite) {
          sp_destroy_reverse(arr + i);
        } else {
          sp_destroy_alloc_reverse(self->alloc, arr + i);
        }
      }
      self->~SpUnboundedArrayStorage();
      std::allocator_traits<ChunkAlloc>::deallocate(ca, mem, nchunks);
      throw;
    }
    return self;
  }

  SpUnboundedArrayStorage(const Alloc& a, std::size_t n,
                          std::size_t nchunks) noexcept
      : base(reinterpret_cast<std::uintptr_t>(&kOps)),
        count(n),
        total_chunks(nchunks),
        alloc(a) {}

  static void dispose_elements(SpUnboundedArrayStorage* self) noexcept {
    Elem* arr = self->elems_ptr();
    if constexpr (kForOverwrite) {
      if constexpr (!std::is_trivially_destructible_v<Elem>) {
        for (std::size_t i = self->count; i > 0U; --i) {
          sp_destroy_reverse(arr + i - 1U);
        }
      }
    } else {
      for (std::size_t i = self->count; i > 0U; --i) {
        sp_destroy_alloc_reverse(self->alloc, arr + i - 1U);
      }
    }
  }

  static void dispose_impl(PortableSpCountedBase* b) noexcept {
    dispose_elements(reinterpret_cast<SpUnboundedArrayStorage*>(b));
  }

  static void destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpUnboundedArrayStorage*>(b);
    ChunkAlloc ca(self->alloc);
    const std::size_t nchunks = self->total_chunks;
    auto* mem = reinterpret_cast<RawChunk*>(self);
    self->~SpUnboundedArrayStorage();
    std::allocator_traits<ChunkAlloc>::deallocate(ca, mem, nchunks);
  }

  static void dispose_and_destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpUnboundedArrayStorage*>(b);
    dispose_elements(self);
    ChunkAlloc ca(self->alloc);
    const std::size_t nchunks = self->total_chunks;
    auto* mem = reinterpret_cast<RawChunk*>(self);
    self->~SpUnboundedArrayStorage();
    std::allocator_traits<ChunkAlloc>::deallocate(ca, mem, nchunks);
  }

  static inline constexpr PortableColdOps kOps{
      &dispose_impl,
      &destroy_impl,
      &dispose_and_destroy_impl,
      nullptr,
  };
};

template <typename Ptr, typename Deleter>
struct SpDeleterStorage final {
  PortableSpCountedBase base;
  Ptr ptr;
  [[no_unique_address]] Deleter deleter;

  static constexpr bool kStandardAlign =
      alignof(Deleter) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__ &&
      alignof(Ptr) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__;

  static void* alloc_raw() {
    if constexpr (kStandardAlign) {
      return ::operator new(sizeof(SpDeleterStorage));
    } else {
      return ::operator new(sizeof(SpDeleterStorage),
                            std::align_val_t{alignof(SpDeleterStorage)});
    }
  }

  static void free_raw(void* p) noexcept {
    if constexpr (kStandardAlign) {
      ::operator delete(p, sizeof(SpDeleterStorage));
    } else {
      ::operator delete(p, sizeof(SpDeleterStorage),
                        std::align_val_t{alignof(SpDeleterStorage)});
    }
  }

  SpDeleterStorage(Ptr p, Deleter d) noexcept(
      std::is_nothrow_move_constructible_v<Deleter>)
      : base(reinterpret_cast<std::uintptr_t>(&kOps)),
        ptr(p),
        deleter(std::move(d)) {}

  static void dispose_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpDeleterStorage*>(b);
    self->deleter(self->ptr);
  }

  static void destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpDeleterStorage*>(b);
    self->~SpDeleterStorage();
    free_raw(static_cast<void*>(self));
  }

  static void dispose_and_destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpDeleterStorage*>(b);
    self->deleter(self->ptr);
    self->~SpDeleterStorage();
    free_raw(static_cast<void*>(self));
  }

  static void* get_deleter_impl(PortableSpCountedBase* b,
                                const std::type_info& ti) noexcept {
#if defined(__cpp_rtti) && __cpp_rtti
    if (ti == typeid(Deleter)) {
      auto* self = reinterpret_cast<SpDeleterStorage*>(b);
      return static_cast<void*>(std::addressof(self->deleter));
    }
#else
    (void)b;
    (void)ti;
#endif
    return nullptr;
  }

  static inline constexpr PortableColdOps kOps{
      &dispose_impl,
      &destroy_impl,
      &dispose_and_destroy_impl,
      &get_deleter_impl,
  };
};

template <typename Ptr, typename Deleter, typename Alloc>
struct SpAllocDeleterStorage final {
  using SelfAlloc = typename std::allocator_traits<
      Alloc>::template rebind_alloc<SpAllocDeleterStorage>;

  PortableSpCountedBase base;
  Ptr ptr;
  [[no_unique_address]] Deleter deleter;
  [[no_unique_address]] SelfAlloc alloc;

  SpAllocDeleterStorage(Ptr p, Deleter d, const Alloc& a) noexcept(
      std::is_nothrow_move_constructible_v<Deleter> &&
      std::is_nothrow_constructible_v<SelfAlloc, const Alloc&>)
      : base(reinterpret_cast<std::uintptr_t>(&kOps)),
        ptr(p),
        deleter(std::move(d)),
        alloc(a) {}

  static void dispose_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocDeleterStorage*>(b);
    self->deleter(self->ptr);
  }

  static void destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocDeleterStorage*>(b);
    SelfAlloc sa(self->alloc);
    self->~SpAllocDeleterStorage();
    std::allocator_traits<SelfAlloc>::deallocate(sa, self, 1U);
  }

  static void dispose_and_destroy_impl(PortableSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocDeleterStorage*>(b);
    self->deleter(self->ptr);
    SelfAlloc sa(self->alloc);
    self->~SpAllocDeleterStorage();
    std::allocator_traits<SelfAlloc>::deallocate(sa, self, 1U);
  }

  static void* get_deleter_impl(PortableSpCountedBase* b,
                                const std::type_info& ti) noexcept {
#if defined(__cpp_rtti) && __cpp_rtti
    if (ti == typeid(Deleter)) {
      auto* self = reinterpret_cast<SpAllocDeleterStorage*>(b);
      return static_cast<void*>(std::addressof(self->deleter));
    }
#else
    (void)b;
    (void)ti;
#endif
    return nullptr;
  }

  static inline constexpr PortableColdOps kOps{
      &dispose_impl,
      &destroy_impl,
      &dispose_and_destroy_impl,
      &get_deleter_impl,
  };
};

struct SpEsftAccess {
  template <typename Base>
  static EnableSharedFromThis<Base>* detect_portable(
      const EnableSharedFromThis<Base>* p) noexcept {
    return const_cast<EnableSharedFromThis<Base>*>(p);
  }

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
  template <typename Base>
  static std::enable_shared_from_this<Base>* detect_std(
      const std::enable_shared_from_this<Base>* p) noexcept {
    return const_cast<std::enable_shared_from_this<Base>*>(p);
  }

  template <typename OwnerT, typename Base, typename RawClean>
  static void init_std_esft(const SharedPtr<OwnerT>& owner,
                            std::enable_shared_from_this<Base>* std_base,
                            RawClean* raw_ptr) noexcept;
#endif

  template <typename OwnerT, typename Ptr>
  static void maybe_enable(const SharedPtr<OwnerT>& owner,
                           Ptr* raw_ptr) noexcept;
};

struct SpAtomicAccess {
  template <typename PtrT>
  static THETA_FORCE_INLINE void raw_parts(const PtrT& p, void*& out_ptr,
                                           StdSpCountedBase*& out_pi) noexcept {
    out_ptr = const_cast<void*>(static_cast<const volatile void*>(p.ptr_));
    out_pi = p.pi_;
  }

  template <typename T>
  static THETA_FORCE_INLINE void take_parts(
      SharedPtr<T>&& p, typename SharedPtr<T>::element_type*& out_ptr,
      StdSpCountedBase*& out_pi) noexcept {
    out_ptr = p.ptr_;
    out_pi = p.pi_;
    p.ptr_ = nullptr;
    p.pi_ = nullptr;
  }

  template <typename T>
  static THETA_FORCE_INLINE SharedPtr<T> make_adopted(
      typename SharedPtr<T>::element_type* ptr, StdSpCountedBase* pi) noexcept {
    return SharedPtr<T>(ptr, pi, AdoptRefTag{});
  }
};

}  // namespace detail::shared_ptr

// ============================================================================
// `theta::portable::SharedPtr<T>`
// ============================================================================
template <typename T>
class SharedPtr {
  template <typename U>
  friend class SharedPtr;
  template <typename U>
  friend class WeakPtr;
  template <typename U>
  friend class KeepAlive;
  template <typename U>
  friend struct OwnerLess;
  friend struct detail::shared_ptr::SpEsftAccess;
  friend struct detail::shared_ptr::SpAtomicAccess;

 public:
  using element_type = std::remove_extent_t<T>;
  using weak_type = WeakPtr<T>;

  constexpr SharedPtr() noexcept = default;
  constexpr SharedPtr(std::nullptr_t) noexcept {}

  template <typename Y>
    requires detail::shared_ptr::kSpRawPtrCompatibleV<Y, T>
  explicit SharedPtr(Y* ptr)
      : SharedPtr(
            ptr,
            std::conditional_t<std::is_array_v<T>, std::default_delete<Y[]>,
                               std::default_delete<Y>>{}) {}

  template <typename Y, typename Deleter>
    requires(detail::shared_ptr::kSpRawPtrCompatibleV<Y, T> &&
             std::is_move_constructible_v<Deleter> &&
             std::is_invocable_v<Deleter&, Y*>)
  SharedPtr(Y* ptr, Deleter deleter) {
    using Storage = detail::shared_ptr::SpDeleterStorage<Y*, Deleter>;
    void* mem = Storage::alloc_raw();
    try {
      auto* s = ::new (mem) Storage(ptr, std::move(deleter));
      ptr_ = static_cast<element_type*>(ptr);
      pi_ = &s->base;
      detail::shared_ptr::SpEsftAccess::maybe_enable(*this, ptr);
    } catch (...) {
      Storage::free_raw(mem);
      deleter(ptr);
      throw;
    }
  }

  template <typename Deleter>
    requires(std::is_move_constructible_v<Deleter> &&
             std::is_invocable_v<Deleter&, std::nullptr_t>)
  SharedPtr(std::nullptr_t, Deleter deleter) {
    using Storage =
        detail::shared_ptr::SpDeleterStorage<std::nullptr_t, Deleter>;
    void* mem = Storage::alloc_raw();
    try {
      auto* s = ::new (mem) Storage(nullptr, std::move(deleter));
      ptr_ = nullptr;
      pi_ = &s->base;
    } catch (...) {
      Storage::free_raw(mem);
      deleter(nullptr);
      throw;
    }
  }

  template <typename Y, typename Deleter, typename Alloc>
    requires(detail::shared_ptr::kSpRawPtrCompatibleV<Y, T> &&
             std::is_move_constructible_v<Deleter> &&
             std::is_invocable_v<Deleter&, Y*>)
  SharedPtr(Y* ptr, Deleter deleter, Alloc alloc) {
    using Storage =
        detail::shared_ptr::SpAllocDeleterStorage<Y*, Deleter, Alloc>;
    using SelfAlloc = typename Storage::SelfAlloc;
    SelfAlloc sa(alloc);
    Storage* mem = std::allocator_traits<SelfAlloc>::allocate(sa, 1U);
    try {
      auto* s = ::new (static_cast<void*>(mem))
          Storage(ptr, std::move(deleter), alloc);
      ptr_ = static_cast<element_type*>(ptr);
      pi_ = &s->base;
      detail::shared_ptr::SpEsftAccess::maybe_enable(*this, ptr);
    } catch (...) {
      std::allocator_traits<SelfAlloc>::deallocate(sa, mem, 1U);
      deleter(ptr);
      throw;
    }
  }

  template <typename Deleter, typename Alloc>
    requires(std::is_move_constructible_v<Deleter> &&
             std::is_invocable_v<Deleter&, std::nullptr_t>)
  SharedPtr(std::nullptr_t, Deleter deleter, Alloc alloc) {
    using Storage = detail::shared_ptr::SpAllocDeleterStorage<std::nullptr_t,
                                                              Deleter, Alloc>;
    using SelfAlloc = typename Storage::SelfAlloc;
    SelfAlloc sa(alloc);
    Storage* mem = std::allocator_traits<SelfAlloc>::allocate(sa, 1U);
    try {
      auto* s = ::new (static_cast<void*>(mem))
          Storage(nullptr, std::move(deleter), alloc);
      ptr_ = nullptr;
      pi_ = &s->base;
    } catch (...) {
      std::allocator_traits<SelfAlloc>::deallocate(sa, mem, 1U);
      deleter(nullptr);
      throw;
    }
  }

  template <typename Y>
  SharedPtr(const SharedPtr<Y>& other, element_type* alias_ptr) noexcept
      : ptr_(alias_ptr), pi_(other.pi_) {
    detail::shared_ptr::sp_add_ref(pi_);
  }

  template <typename Y>
  SharedPtr(SharedPtr<Y>&& other, element_type* alias_ptr) noexcept
      : ptr_(alias_ptr), pi_(other.pi_) {
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
  }

  SharedPtr(const SharedPtr& other) noexcept
      : ptr_(other.ptr_), pi_(other.pi_) {
    detail::shared_ptr::sp_add_ref(pi_);
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  SharedPtr(const SharedPtr<Y>& other) noexcept
      : ptr_(other.ptr_), pi_(other.pi_) {
    detail::shared_ptr::sp_add_ref(pi_);
  }

  SharedPtr(SharedPtr&& other) noexcept : ptr_(other.ptr_), pi_(other.pi_) {
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  SharedPtr(SharedPtr<Y>&& other) noexcept : ptr_(other.ptr_), pi_(other.pi_) {
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  explicit SharedPtr(const WeakPtr<Y>& r);

  template <typename Y, typename Deleter>
    requires(detail::shared_ptr::kSpCompatibleV<Y, T> &&
             std::is_convertible_v<
                 typename std::unique_ptr<Y, Deleter>::pointer, element_type*>)
  SharedPtr(std::unique_ptr<Y, Deleter>&& r) {
    if (r.get() == nullptr) {
      return;
    }
    auto raw = r.get();
    if constexpr (std::is_reference_v<Deleter>) {
      SharedPtr tmp(raw, std::ref(r.get_deleter()));
      r.release();
      swap(tmp);
    } else {
      SharedPtr tmp(raw, std::move(r.get_deleter()));
      r.release();
      swap(tmp);
    }
  }

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  SharedPtr(const std::shared_ptr<Y>& sp) noexcept {
    detail::shared_ptr::RawStdPtrLayout raw{};
    std::memcpy(&raw, static_cast<const void*>(&sp), sizeof(raw));
    ptr_ = sp.get();
    pi_ = raw.pi;
    if (pi_ != nullptr) {
      detail::shared_ptr::sp_add_ref_from_std(pi_);
    }
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  SharedPtr(std::shared_ptr<Y>&& sp) noexcept {
    detail::shared_ptr::RawStdPtrLayout raw{};
    std::memcpy(&raw, static_cast<const void*>(&sp), sizeof(raw));
    ptr_ = sp.get();
    pi_ = raw.pi;
    const detail::shared_ptr::RawStdPtrLayout zero{nullptr, nullptr};
    std::memcpy(static_cast<void*>(&sp), &zero, sizeof(zero));
    if (pi_ != nullptr) {
      detail::shared_ptr::sp_adopt_from_std_move(pi_);
    }
  }
#endif

  ~SharedPtr() noexcept { detail::shared_ptr::sp_release(pi_); }

  SharedPtr& operator=(const SharedPtr& other) noexcept {
    if (this == &other) {
      return *this;
    }
    SharedPtr tmp(other);
    swap(tmp);
    return *this;
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  SharedPtr& operator=(const SharedPtr<Y>& other) noexcept {
    SharedPtr tmp(other);
    swap(tmp);
    return *this;
  }

  SharedPtr& operator=(SharedPtr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    detail::shared_ptr::sp_release(pi_);
    ptr_ = other.ptr_;
    pi_ = other.pi_;
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
    return *this;
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  SharedPtr& operator=(SharedPtr<Y>&& other) noexcept {
    detail::shared_ptr::sp_release(pi_);
    ptr_ = other.ptr_;
    pi_ = other.pi_;
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
    return *this;
  }

  template <typename Y, typename Deleter>
    requires(detail::shared_ptr::kSpCompatibleV<Y, T> &&
             std::is_convertible_v<
                 typename std::unique_ptr<Y, Deleter>::pointer, element_type*>)
  SharedPtr& operator=(std::unique_ptr<Y, Deleter>&& r) {
    SharedPtr tmp(std::move(r));
    swap(tmp);
    return *this;
  }

  SharedPtr& operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }

  void reset() noexcept {
    SharedPtr tmp;
    swap(tmp);
  }

  template <typename Y>
    requires detail::shared_ptr::kSpRawPtrCompatibleV<Y, T>
  void reset(Y* ptr) {
    SharedPtr tmp(ptr);
    swap(tmp);
  }

  template <typename Y, typename Deleter>
    requires(detail::shared_ptr::kSpRawPtrCompatibleV<Y, T> &&
             std::is_move_constructible_v<Deleter> &&
             std::is_invocable_v<Deleter&, Y*>)
  void reset(Y* ptr, Deleter deleter) {
    SharedPtr tmp(ptr, std::move(deleter));
    swap(tmp);
  }

  template <typename Y, typename Deleter, typename Alloc>
    requires(detail::shared_ptr::kSpRawPtrCompatibleV<Y, T> &&
             std::is_move_constructible_v<Deleter> &&
             std::is_invocable_v<Deleter&, Y*>)
  void reset(Y* ptr, Deleter deleter, Alloc alloc) {
    SharedPtr tmp(ptr, std::move(deleter), std::move(alloc));
    swap(tmp);
  }

  void swap(SharedPtr& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(pi_, other.pi_);
  }

  [[nodiscard]] element_type* get() const noexcept { return ptr_; }

  [[nodiscard]] std::add_lvalue_reference_t<T> operator*() const noexcept
    requires(!std::is_void_v<T> && !std::is_array_v<T>)
  {
    return *ptr_;
  }

  [[nodiscard]] T* operator->() const noexcept
    requires(!std::is_array_v<T>)
  {
    return ptr_;
  }

  [[nodiscard]] std::add_lvalue_reference_t<element_type> operator[](
      std::ptrdiff_t idx) const noexcept
    requires std::is_array_v<T>
  {
    return ptr_[idx];
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return ptr_ != nullptr;
  }

  [[nodiscard]] long use_count() const noexcept {
    return detail::shared_ptr::sp_use_count_slow(pi_);
  }

  template <typename U>
  [[nodiscard]] bool owner_before(const SharedPtr<U>& other) const noexcept {
    return std::less<detail::shared_ptr::StdSpCountedBase*>{}(pi_, other.pi_);
  }

  template <typename U>
  [[nodiscard]] bool owner_before(const WeakPtr<U>& other) const noexcept;

  template <typename U>
  [[nodiscard]] bool owner_equal(const SharedPtr<U>& other) const noexcept {
    return pi_ == other.pi_;
  }

  template <typename U>
  [[nodiscard]] bool owner_equal(const WeakPtr<U>& other) const noexcept;

  [[nodiscard]] std::size_t owner_hash() const noexcept {
    return std::hash<detail::shared_ptr::StdSpCountedBase*>{}(pi_);
  }

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
  [[nodiscard]] std::shared_ptr<T> to_std() const& noexcept {
    std::shared_ptr<T> out;
    if (pi_ != nullptr) {
      detail::shared_ptr::sp_note_to_std_copy(pi_);
      const detail::shared_ptr::RawStdPtrLayout raw{
          const_cast<void*>(static_cast<const volatile void*>(ptr_)), pi_};
      std::memcpy(static_cast<void*>(&out), &raw, sizeof(out));
    }
    return out;
  }

  [[nodiscard]] std::shared_ptr<T> to_std() && noexcept {
    std::shared_ptr<T> out;
    if (pi_ != nullptr) {
      detail::shared_ptr::sp_note_to_std_move(pi_);
      const detail::shared_ptr::RawStdPtrLayout raw{
          const_cast<void*>(static_cast<const volatile void*>(ptr_)), pi_};
      std::memcpy(static_cast<void*>(&out), &raw, sizeof(out));
      ptr_ = nullptr;
      pi_ = nullptr;
    }
    return out;
  }

  operator std::shared_ptr<T>() const& noexcept { return to_std(); }
  operator std::shared_ptr<T>() && noexcept {
    return std::move(*this).to_std();
  }
#endif

 private:
  SharedPtr(element_type* ptr, detail::shared_ptr::StdSpCountedBase* pi,
            detail::shared_ptr::AdoptRefTag) noexcept
      : ptr_(ptr), pi_(pi) {}

  template <typename U, typename... Args>
    requires(!std::is_array_v<U>)
  friend SharedPtr<U> make_shared(Args&&... args);

  template <typename U>
    requires std::is_unbounded_array_v<U>
  friend SharedPtr<U> make_shared(std::size_t n);

  template <typename U>
    requires std::is_unbounded_array_v<U>
  friend SharedPtr<U> make_shared(std::size_t n,
                                  const std::remove_extent_t<U>& u);

  template <typename U>
    requires std::is_bounded_array_v<U>
  friend SharedPtr<U> make_shared();

  template <typename U>
    requires std::is_bounded_array_v<U>
  friend SharedPtr<U> make_shared(const std::remove_extent_t<U>& u);

  template <typename U>
    requires(!std::is_unbounded_array_v<U>)
  friend SharedPtr<U> make_shared_for_overwrite();

  template <typename U>
    requires std::is_unbounded_array_v<U>
  friend SharedPtr<U> make_shared_for_overwrite(std::size_t n);

  template <typename U, typename Alloc, typename... Args>
    requires(!std::is_array_v<U>)
  friend SharedPtr<U> allocate_shared(const Alloc& alloc, Args&&... args);

  template <typename U, typename Alloc>
    requires std::is_unbounded_array_v<U>
  friend SharedPtr<U> allocate_shared(const Alloc& alloc, std::size_t n);

  template <typename U, typename Alloc>
    requires std::is_unbounded_array_v<U>
  friend SharedPtr<U> allocate_shared(const Alloc& alloc, std::size_t n,
                                      const std::remove_extent_t<U>& u);

  template <typename U, typename Alloc>
    requires std::is_bounded_array_v<U>
  friend SharedPtr<U> allocate_shared(const Alloc& alloc);

  template <typename U, typename Alloc>
    requires std::is_bounded_array_v<U>
  friend SharedPtr<U> allocate_shared(const Alloc& alloc,
                                      const std::remove_extent_t<U>& u);

  template <typename U, typename Alloc>
    requires(!std::is_unbounded_array_v<U>)
  friend SharedPtr<U> allocate_shared_for_overwrite(const Alloc& alloc);

  template <typename U, typename Alloc>
    requires std::is_unbounded_array_v<U>
  friend SharedPtr<U> allocate_shared_for_overwrite(const Alloc& alloc,
                                                    std::size_t n);

  template <typename Deleter, typename U>
  friend Deleter* get_deleter(const SharedPtr<U>& p) noexcept;

  element_type* ptr_{nullptr};
  detail::shared_ptr::StdSpCountedBase* pi_{nullptr};
};

static_assert(sizeof(SharedPtr<int>) == 2 * sizeof(void*));

// ============================================================================
// Factory Functions
// ============================================================================
template <typename T, typename... Args>
  requires(!std::is_array_v<T>)
[[nodiscard]] SharedPtr<T> make_shared(Args&&... args) {
  using CleanT = std::remove_cv_t<T>;
  using Storage = detail::shared_ptr::SpInplaceStorage<CleanT>;
  void* mem = Storage::alloc_raw();
  try {
    auto* s = ::new (mem) Storage(std::forward<Args>(args)...);
    SharedPtr<T> out(s->ptr(), &s->base, detail::shared_ptr::AdoptRefTag{});
    detail::shared_ptr::SpEsftAccess::maybe_enable(out, s->ptr());
    return out;
  } catch (...) {
    Storage::free_raw(mem);
    throw;
  }
}

template <typename T>
  requires std::is_unbounded_array_v<T>
[[nodiscard]] SharedPtr<T> make_shared(std::size_t n) {
  using CleanElem =
      detail::shared_ptr::SpRemoveCvTargetT<std::remove_extent_t<T>>;
  using Scalar = std::remove_all_extents_t<CleanElem>;
  using Alloc = std::allocator<Scalar>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
  using ValueAlloc = typename Storage::ValueAlloc;
  auto* s = Storage::create(Alloc{}, n, [](ValueAlloc& va, CleanElem* p) {
    detail::shared_ptr::sp_construct_val_alloc(va, p);
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

template <typename T>
  requires std::is_unbounded_array_v<T>
[[nodiscard]] SharedPtr<T> make_shared(std::size_t n,
                                       const std::remove_extent_t<T>& u) {
  using CleanElem =
      detail::shared_ptr::SpRemoveCvTargetT<std::remove_extent_t<T>>;
  using Scalar = std::remove_all_extents_t<CleanElem>;
  using Alloc = std::allocator<Scalar>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
  using ValueAlloc = typename Storage::ValueAlloc;
  auto* s = Storage::create(Alloc{}, n, [&u](ValueAlloc& va, CleanElem* p) {
    detail::shared_ptr::sp_construct_copy_alloc(va, p, u);
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

template <typename T>
  requires std::is_bounded_array_v<T>
[[nodiscard]] SharedPtr<T> make_shared() {
  using CleanT = detail::shared_ptr::SpRemoveCvTargetT<T>;
  using Storage = detail::shared_ptr::SpInplaceStorage<CleanT>;
  void* mem = Storage::alloc_raw();
  try {
    auto* s = ::new (mem) Storage();
    return SharedPtr<T>(s->ptr(), &s->base, detail::shared_ptr::AdoptRefTag{});
  } catch (...) {
    Storage::free_raw(mem);
    throw;
  }
}

template <typename T>
  requires std::is_bounded_array_v<T>
[[nodiscard]] SharedPtr<T> make_shared(const std::remove_extent_t<T>& u) {
  using CleanT = detail::shared_ptr::SpRemoveCvTargetT<T>;
  using Storage = detail::shared_ptr::SpInplaceStorage<CleanT>;
  void* mem = Storage::alloc_raw();
  try {
    auto* s = ::new (mem) Storage(detail::shared_ptr::ForOverwriteTag{}, u);
    return SharedPtr<T>(s->ptr(), &s->base, detail::shared_ptr::AdoptRefTag{});
  } catch (...) {
    Storage::free_raw(mem);
    throw;
  }
}

template <typename T>
  requires(!std::is_unbounded_array_v<T>)
[[nodiscard]] SharedPtr<T> make_shared_for_overwrite() {
  using CleanT = detail::shared_ptr::SpRemoveCvTargetT<T>;
  using Storage = detail::shared_ptr::SpInplaceStorage<CleanT>;
  void* mem = Storage::alloc_raw();
  try {
    auto* s = ::new (mem) Storage(detail::shared_ptr::ForOverwriteTag{});
    SharedPtr<T> out(s->ptr(), &s->base, detail::shared_ptr::AdoptRefTag{});
    if constexpr (!std::is_array_v<T>) {
      detail::shared_ptr::SpEsftAccess::maybe_enable(out, s->ptr());
    }
    return out;
  } catch (...) {
    Storage::free_raw(mem);
    throw;
  }
}

template <typename T>
  requires std::is_unbounded_array_v<T>
[[nodiscard]] SharedPtr<T> make_shared_for_overwrite(std::size_t n) {
  using CleanElem =
      detail::shared_ptr::SpRemoveCvTargetT<std::remove_extent_t<T>>;
  using Scalar = std::remove_all_extents_t<CleanElem>;
  using Alloc = std::allocator<Scalar>;
  using Storage =
      detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc,
                                                  /*kForOverwrite=*/true>;
  using ValueAlloc = typename Storage::ValueAlloc;
  auto* s = Storage::create(Alloc{}, n, [](ValueAlloc&, CleanElem* p) {
    ::new (static_cast<void*>(p)) CleanElem;
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

template <typename T, typename Alloc, typename... Args>
  requires(!std::is_array_v<T>)
[[nodiscard]] SharedPtr<T> allocate_shared(const Alloc& alloc, Args&&... args) {
  using CleanT = std::remove_cv_t<T>;
  using Storage = detail::shared_ptr::SpAllocInplaceStorage<CleanT, Alloc>;
  using SelfAlloc = typename Storage::SelfAlloc;
  SelfAlloc sa(alloc);
  Storage* mem = std::allocator_traits<SelfAlloc>::allocate(sa, 1U);
  try {
    auto* s = ::new (static_cast<void*>(mem))
        Storage(alloc, std::forward<Args>(args)...);
    SharedPtr<T> out(s->ptr(), &s->base, detail::shared_ptr::AdoptRefTag{});
    detail::shared_ptr::SpEsftAccess::maybe_enable(out, s->ptr());
    return out;
  } catch (...) {
    std::allocator_traits<SelfAlloc>::deallocate(sa, mem, 1U);
    throw;
  }
}

template <typename T, typename Alloc>
  requires std::is_unbounded_array_v<T>
[[nodiscard]] SharedPtr<T> allocate_shared(const Alloc& alloc, std::size_t n) {
  using CleanElem =
      detail::shared_ptr::SpRemoveCvTargetT<std::remove_extent_t<T>>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
  using ValueAlloc = typename Storage::ValueAlloc;
  auto* s = Storage::create(alloc, n, [](ValueAlloc& va, CleanElem* p) {
    detail::shared_ptr::sp_construct_val_alloc(va, p);
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

template <typename T, typename Alloc>
  requires std::is_unbounded_array_v<T>
[[nodiscard]] SharedPtr<T> allocate_shared(const Alloc& alloc, std::size_t n,
                                           const std::remove_extent_t<T>& u) {
  using CleanElem =
      detail::shared_ptr::SpRemoveCvTargetT<std::remove_extent_t<T>>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
  using ValueAlloc = typename Storage::ValueAlloc;
  auto* s = Storage::create(alloc, n, [&u](ValueAlloc& va, CleanElem* p) {
    detail::shared_ptr::sp_construct_copy_alloc(va, p, u);
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

template <typename T, typename Alloc>
  requires std::is_bounded_array_v<T>
[[nodiscard]] SharedPtr<T> allocate_shared(const Alloc& alloc) {
  using CleanT = detail::shared_ptr::SpRemoveCvTargetT<T>;
  using Storage = detail::shared_ptr::SpAllocInplaceStorage<CleanT, Alloc>;
  using SelfAlloc = typename Storage::SelfAlloc;
  SelfAlloc sa(alloc);
  Storage* mem = std::allocator_traits<SelfAlloc>::allocate(sa, 1U);
  try {
    auto* s = ::new (static_cast<void*>(mem)) Storage(alloc);
    return SharedPtr<T>(s->ptr(), &s->base, detail::shared_ptr::AdoptRefTag{});
  } catch (...) {
    std::allocator_traits<SelfAlloc>::deallocate(sa, mem, 1U);
    throw;
  }
}

template <typename T, typename Alloc>
  requires std::is_bounded_array_v<T>
[[nodiscard]] SharedPtr<T> allocate_shared(const Alloc& alloc,
                                           const std::remove_extent_t<T>& u) {
  using CleanT = detail::shared_ptr::SpRemoveCvTargetT<T>;
  using Storage = detail::shared_ptr::SpAllocInplaceStorage<CleanT, Alloc>;
  using SelfAlloc = typename Storage::SelfAlloc;
  SelfAlloc sa(alloc);
  Storage* mem = std::allocator_traits<SelfAlloc>::allocate(sa, 1U);
  try {
    auto* s = ::new (static_cast<void*>(mem))
        Storage(alloc, detail::shared_ptr::ForOverwriteTag{}, u);
    return SharedPtr<T>(s->ptr(), &s->base, detail::shared_ptr::AdoptRefTag{});
  } catch (...) {
    std::allocator_traits<SelfAlloc>::deallocate(sa, mem, 1U);
    throw;
  }
}

template <typename T, typename Alloc>
  requires(!std::is_unbounded_array_v<T>)
[[nodiscard]] SharedPtr<T> allocate_shared_for_overwrite(const Alloc& alloc) {
  using CleanT = detail::shared_ptr::SpRemoveCvTargetT<T>;
  using Storage =
      detail::shared_ptr::SpAllocInplaceStorage<CleanT, Alloc,
                                                /*kForOverwrite=*/true>;
  using SelfAlloc = typename Storage::SelfAlloc;
  SelfAlloc sa(alloc);
  Storage* mem = std::allocator_traits<SelfAlloc>::allocate(sa, 1U);
  try {
    auto* s = ::new (static_cast<void*>(mem))
        Storage(alloc, detail::shared_ptr::ForOverwriteTag{});
    SharedPtr<T> out(s->ptr(), &s->base, detail::shared_ptr::AdoptRefTag{});
    if constexpr (!std::is_array_v<T>) {
      detail::shared_ptr::SpEsftAccess::maybe_enable(out, s->ptr());
    }
    return out;
  } catch (...) {
    std::allocator_traits<SelfAlloc>::deallocate(sa, mem, 1U);
    throw;
  }
}

template <typename T, typename Alloc>
  requires std::is_unbounded_array_v<T>
[[nodiscard]] SharedPtr<T> allocate_shared_for_overwrite(const Alloc& alloc,
                                                         std::size_t n) {
  using CleanElem =
      detail::shared_ptr::SpRemoveCvTargetT<std::remove_extent_t<T>>;
  using Storage =
      detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc,
                                                  /*kForOverwrite=*/true>;
  using ValueAlloc = typename Storage::ValueAlloc;
  auto* s = Storage::create(alloc, n, [](ValueAlloc&, CleanElem* p) {
    ::new (static_cast<void*>(p)) CleanElem;
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

// ============================================================================
// Casts, `get_deleter`, Comparisons, Streaming, & Swap
// ============================================================================
template <typename T, typename U>
[[nodiscard]] SharedPtr<T> static_pointer_cast(const SharedPtr<U>& r) noexcept {
  auto* p = static_cast<typename SharedPtr<T>::element_type*>(r.get());
  return SharedPtr<T>(r, p);
}

template <typename T, typename U>
[[nodiscard]] SharedPtr<T> static_pointer_cast(SharedPtr<U>&& r) noexcept {
  auto* p = static_cast<typename SharedPtr<T>::element_type*>(r.get());
  return SharedPtr<T>(std::move(r), p);
}

template <typename T, typename U>
[[nodiscard]] SharedPtr<T> dynamic_pointer_cast(
    const SharedPtr<U>& r) noexcept {
  if (auto* p = dynamic_cast<typename SharedPtr<T>::element_type*>(r.get())) {
    return SharedPtr<T>(r, p);
  }
  return SharedPtr<T>{};
}

template <typename T, typename U>
[[nodiscard]] SharedPtr<T> dynamic_pointer_cast(SharedPtr<U>&& r) noexcept {
  if (auto* p = dynamic_cast<typename SharedPtr<T>::element_type*>(r.get())) {
    return SharedPtr<T>(std::move(r), p);
  }
  return SharedPtr<T>{};
}

template <typename T, typename U>
[[nodiscard]] SharedPtr<T> const_pointer_cast(const SharedPtr<U>& r) noexcept {
  auto* p = const_cast<typename SharedPtr<T>::element_type*>(r.get());
  return SharedPtr<T>(r, p);
}

template <typename T, typename U>
[[nodiscard]] SharedPtr<T> const_pointer_cast(SharedPtr<U>&& r) noexcept {
  auto* p = const_cast<typename SharedPtr<T>::element_type*>(r.get());
  return SharedPtr<T>(std::move(r), p);
}

template <typename T, typename U>
[[nodiscard]] SharedPtr<T> reinterpret_pointer_cast(
    const SharedPtr<U>& r) noexcept {
  auto* p = reinterpret_cast<typename SharedPtr<T>::element_type*>(r.get());
  return SharedPtr<T>(r, p);
}

template <typename T, typename U>
[[nodiscard]] SharedPtr<T> reinterpret_pointer_cast(SharedPtr<U>&& r) noexcept {
  auto* p = reinterpret_cast<typename SharedPtr<T>::element_type*>(r.get());
  return SharedPtr<T>(std::move(r), p);
}

template <typename Deleter, typename T>
[[nodiscard]] Deleter* get_deleter(const SharedPtr<T>& p) noexcept {
#if defined(__cpp_rtti) && __cpp_rtti
  if (p.pi_ == nullptr) {
    return nullptr;
  }
  return static_cast<Deleter*>(p.pi_->_M_get_deleter(typeid(Deleter)));
#else
  (void)p;
  return nullptr;
#endif
}

template <typename T, typename U>
[[nodiscard]] bool operator==(const SharedPtr<T>& lhs,
                              const SharedPtr<U>& rhs) noexcept {
  return lhs.get() == rhs.get();
}

template <typename T>
[[nodiscard]] bool operator==(const SharedPtr<T>& lhs,
                              std::nullptr_t) noexcept {
  return lhs.get() == nullptr;
}

template <typename T, typename U>
[[nodiscard]] std::strong_ordering operator<=>(
    const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept {
  using CommonPtr = std::common_type_t<typename SharedPtr<T>::element_type*,
                                       typename SharedPtr<U>::element_type*>;
  return std::compare_three_way{}(static_cast<CommonPtr>(lhs.get()),
                                  static_cast<CommonPtr>(rhs.get()));
}

template <typename T>
[[nodiscard]] std::strong_ordering operator<=>(const SharedPtr<T>& lhs,
                                               std::nullptr_t) noexcept {
  return std::compare_three_way{}(
      lhs.get(), static_cast<typename SharedPtr<T>::element_type*>(nullptr));
}

template <typename CharT, typename Traits, typename Y>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os, const SharedPtr<Y>& p) {
  return os << p.get();
}

template <typename T>
void swap(SharedPtr<T>& a, SharedPtr<T>& b) noexcept {
  a.swap(b);
}

// ============================================================================
// `theta::portable::WeakPtr<T>`
// ============================================================================
template <typename T>
class WeakPtr {
  template <typename U>
  friend class WeakPtr;
  template <typename U>
  friend class SharedPtr;
  template <typename U>
  friend struct OwnerLess;
  friend struct detail::shared_ptr::SpAtomicAccess;

 public:
  using element_type = std::remove_extent_t<T>;

  constexpr WeakPtr() noexcept = default;

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  WeakPtr(const SharedPtr<Y>& sp) noexcept : ptr_(sp.ptr_), pi_(sp.pi_) {
    if (pi_ != nullptr) {
      if (detail::shared_ptr::is_portable_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::PortableSpCountedBase*>(pi_);
        __atomic_store_n(&cb->fast_.has_weak_or_std, 1U, __ATOMIC_RELEASE);
      }
      pi_->_M_weak_add_ref();
    }
  }

  WeakPtr(const WeakPtr& other) noexcept : ptr_(other.ptr_), pi_(other.pi_) {
    if (pi_ != nullptr) {
      pi_->_M_weak_add_ref();
    }
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  WeakPtr(const WeakPtr<Y>& other) noexcept : ptr_(other.ptr_), pi_(other.pi_) {
    if (pi_ != nullptr) {
      pi_->_M_weak_add_ref();
    }
  }

  WeakPtr(WeakPtr&& other) noexcept : ptr_(other.ptr_), pi_(other.pi_) {
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  WeakPtr(WeakPtr<Y>&& other) noexcept : ptr_(other.ptr_), pi_(other.pi_) {
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
  }

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  WeakPtr(const std::shared_ptr<Y>& sp) noexcept {
    detail::shared_ptr::RawStdPtrLayout raw{};
    std::memcpy(&raw, static_cast<const void*>(&sp), sizeof(raw));
    ptr_ = sp.get();
    pi_ = raw.pi;
    if (pi_ != nullptr) {
      if (detail::shared_ptr::is_portable_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::PortableSpCountedBase*>(pi_);
        __atomic_store_n(&cb->fast_.has_weak_or_std, 1U, __ATOMIC_RELEASE);
      }
      pi_->_M_weak_add_ref();
    }
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  WeakPtr(const std::weak_ptr<Y>& wp) noexcept {
    detail::shared_ptr::RawStdPtrLayout raw{};
    std::memcpy(&raw, static_cast<const void*>(&wp), sizeof(raw));
    ptr_ = static_cast<element_type*>(raw.ptr);
    pi_ = raw.pi;
    if (pi_ != nullptr) {
      if (detail::shared_ptr::is_portable_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::PortableSpCountedBase*>(pi_);
        __atomic_store_n(&cb->fast_.has_weak_or_std, 1U, __ATOMIC_RELEASE);
      }
      pi_->_M_weak_add_ref();
    }
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  WeakPtr(std::weak_ptr<Y>&& wp) noexcept {
    detail::shared_ptr::RawStdPtrLayout raw{};
    std::memcpy(&raw, static_cast<const void*>(&wp), sizeof(raw));
    ptr_ = static_cast<element_type*>(raw.ptr);
    pi_ = raw.pi;
    const detail::shared_ptr::RawStdPtrLayout zero{nullptr, nullptr};
    std::memcpy(static_cast<void*>(&wp), &zero, sizeof(zero));
    if (pi_ != nullptr && detail::shared_ptr::is_portable_sp(pi_)) {
      auto* cb = static_cast<detail::shared_ptr::PortableSpCountedBase*>(pi_);
      __atomic_store_n(&cb->fast_.has_weak_or_std, 1U, __ATOMIC_RELEASE);
    }
  }
#endif

  ~WeakPtr() noexcept {
    if (pi_ != nullptr) {
      pi_->_M_weak_release();
    }
  }

  WeakPtr& operator=(const WeakPtr& other) noexcept {
    if (this == &other) {
      return *this;
    }
    WeakPtr tmp(other);
    swap(tmp);
    return *this;
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  WeakPtr& operator=(const WeakPtr<Y>& other) noexcept {
    WeakPtr tmp(other);
    swap(tmp);
    return *this;
  }

  WeakPtr& operator=(WeakPtr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (pi_ != nullptr) {
      pi_->_M_weak_release();
    }
    ptr_ = other.ptr_;
    pi_ = other.pi_;
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
    return *this;
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  WeakPtr& operator=(WeakPtr<Y>&& other) noexcept {
    if (pi_ != nullptr) {
      pi_->_M_weak_release();
    }
    ptr_ = other.ptr_;
    pi_ = other.pi_;
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
    return *this;
  }

  template <typename Y>
    requires detail::shared_ptr::kSpCompatibleV<Y, T>
  WeakPtr& operator=(const SharedPtr<Y>& sp) noexcept {
    WeakPtr tmp(sp);
    swap(tmp);
    return *this;
  }

  void reset() noexcept {
    WeakPtr tmp;
    swap(tmp);
  }

  void swap(WeakPtr& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(pi_, other.pi_);
  }

  [[nodiscard]] long use_count() const noexcept {
    return detail::shared_ptr::sp_use_count_slow(pi_);
  }

  [[nodiscard]] bool expired() const noexcept { return use_count() == 0L; }

  [[nodiscard]] SharedPtr<T> lock() const noexcept {
    if (pi_ == nullptr) {
      return SharedPtr<T>{};
    }
    if (detail::shared_ptr::sp_weak_lock_slow(pi_)) {
      return SharedPtr<T>(ptr_, pi_, detail::shared_ptr::AdoptRefTag{});
    }
    return SharedPtr<T>{};
  }

  template <typename U>
  [[nodiscard]] bool owner_before(const WeakPtr<U>& other) const noexcept {
    return std::less<detail::shared_ptr::StdSpCountedBase*>{}(pi_, other.pi_);
  }

  template <typename U>
  [[nodiscard]] bool owner_before(const SharedPtr<U>& other) const noexcept {
    return std::less<detail::shared_ptr::StdSpCountedBase*>{}(pi_, other.pi_);
  }

  template <typename U>
  [[nodiscard]] bool owner_equal(const WeakPtr<U>& other) const noexcept {
    return pi_ == other.pi_;
  }

  template <typename U>
  [[nodiscard]] bool owner_equal(const SharedPtr<U>& other) const noexcept {
    return pi_ == other.pi_;
  }

  [[nodiscard]] std::size_t owner_hash() const noexcept {
    return std::hash<detail::shared_ptr::StdSpCountedBase*>{}(pi_);
  }

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
  [[nodiscard]] std::weak_ptr<T> to_std() const& noexcept {
    std::weak_ptr<T> out;
    if (pi_ != nullptr) {
      if (detail::shared_ptr::is_portable_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::PortableSpCountedBase*>(pi_);
        __atomic_store_n(&cb->fast_.has_weak_or_std, 1U, __ATOMIC_RELEASE);
      }
      pi_->_M_weak_add_ref();
      const detail::shared_ptr::RawStdPtrLayout raw{
          const_cast<void*>(static_cast<const volatile void*>(ptr_)), pi_};
      std::memcpy(static_cast<void*>(&out), &raw, sizeof(out));
    }
    return out;
  }

  [[nodiscard]] std::weak_ptr<T> to_std() && noexcept {
    std::weak_ptr<T> out;
    if (pi_ != nullptr) {
      if (detail::shared_ptr::is_portable_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::PortableSpCountedBase*>(pi_);
        __atomic_store_n(&cb->fast_.has_weak_or_std, 1U, __ATOMIC_RELEASE);
      }
      const detail::shared_ptr::RawStdPtrLayout raw{
          const_cast<void*>(static_cast<const volatile void*>(ptr_)), pi_};
      std::memcpy(static_cast<void*>(&out), &raw, sizeof(out));
      ptr_ = nullptr;
      pi_ = nullptr;
    }
    return out;
  }

  operator std::weak_ptr<T>() const& noexcept { return to_std(); }
  operator std::weak_ptr<T>() && noexcept { return std::move(*this).to_std(); }
#endif

 private:
  element_type* ptr_{nullptr};
  detail::shared_ptr::StdSpCountedBase* pi_{nullptr};
};

template <typename T>
void swap(WeakPtr<T>& a, WeakPtr<T>& b) noexcept {
  a.swap(b);
}

template <typename T>
template <typename Y>
  requires detail::shared_ptr::kSpCompatibleV<Y, T>
SharedPtr<T>::SharedPtr(const WeakPtr<Y>& r) {
  if (r.pi_ == nullptr || !detail::shared_ptr::sp_weak_lock_slow(r.pi_)) {
    throw std::bad_weak_ptr{};
  }
  ptr_ = r.ptr_;
  pi_ = r.pi_;
}

template <typename T>
template <typename U>
bool SharedPtr<T>::owner_before(const WeakPtr<U>& other) const noexcept {
  return std::less<detail::shared_ptr::StdSpCountedBase*>{}(pi_, other.pi_);
}

template <typename T>
template <typename U>
bool SharedPtr<T>::owner_equal(const WeakPtr<U>& other) const noexcept {
  return pi_ == other.pi_;
}

// ============================================================================
// `theta::portable::OwnerLess<T>` & `EnableSharedFromThis<T>`
// ============================================================================
template <typename T>
struct OwnerLess<SharedPtr<T>> {
  [[nodiscard]] bool operator()(const SharedPtr<T>& lhs,
                                const SharedPtr<T>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
  [[nodiscard]] bool operator()(const SharedPtr<T>& lhs,
                                const WeakPtr<T>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
  [[nodiscard]] bool operator()(const WeakPtr<T>& lhs,
                                const SharedPtr<T>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
};

template <typename T>
struct OwnerLess<WeakPtr<T>> {
  [[nodiscard]] bool operator()(const WeakPtr<T>& lhs,
                                const WeakPtr<T>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
  [[nodiscard]] bool operator()(const SharedPtr<T>& lhs,
                                const WeakPtr<T>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
  [[nodiscard]] bool operator()(const WeakPtr<T>& lhs,
                                const SharedPtr<T>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
};

template <>
struct OwnerLess<void> {
  using is_transparent = void;

  template <typename T, typename U>
  [[nodiscard]] bool operator()(const SharedPtr<T>& lhs,
                                const SharedPtr<U>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
  template <typename T, typename U>
  [[nodiscard]] bool operator()(const SharedPtr<T>& lhs,
                                const WeakPtr<U>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
  template <typename T, typename U>
  [[nodiscard]] bool operator()(const WeakPtr<T>& lhs,
                                const SharedPtr<U>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
  template <typename T, typename U>
  [[nodiscard]] bool operator()(const WeakPtr<T>& lhs,
                                const WeakPtr<U>& rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
};

template <typename T>
class EnableSharedFromThis {
  friend struct detail::shared_ptr::SpEsftAccess;

 protected:
  constexpr EnableSharedFromThis() noexcept = default;
  EnableSharedFromThis(const EnableSharedFromThis&) noexcept {}
  EnableSharedFromThis& operator=(const EnableSharedFromThis&) noexcept {
    return *this;
  }
  ~EnableSharedFromThis() = default;

 public:
  [[nodiscard]] SharedPtr<T> shared_from_this() {
    return SharedPtr<T>(weak_this_);
  }

  [[nodiscard]] SharedPtr<const T> shared_from_this() const {
    return SharedPtr<const T>(weak_this_);
  }

  [[nodiscard]] WeakPtr<T> weak_from_this() noexcept { return weak_this_; }

  [[nodiscard]] WeakPtr<const T> weak_from_this() const noexcept {
    return weak_this_;
  }

 private:
  mutable WeakPtr<T> weak_this_;
};

namespace detail::shared_ptr {

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
template <typename OwnerT, typename Base, typename RawClean>
void SpEsftAccess::init_std_esft(const SharedPtr<OwnerT>& owner,
                                 std::enable_shared_from_this<Base>* std_base,
                                 RawClean* raw_ptr) noexcept {
  static_assert(sizeof(std::enable_shared_from_this<Base>) ==
                sizeof(RawStdPtrLayout));
  if (owner.pi_ == nullptr) {
    return;
  }
  RawStdPtrLayout cur{};
  std::memcpy(&cur, static_cast<const void*>(std_base), sizeof(cur));
  if (cur.pi != nullptr && sp_use_count_slow(cur.pi) != 0L) {
    return;
  }
  if (cur.pi != nullptr) {
    cur.pi->_M_weak_release();
  }
  if (is_portable_sp(owner.pi_)) {
    auto* cb = static_cast<PortableSpCountedBase*>(owner.pi_);
    __atomic_store_n(&cb->fast_.has_weak_or_std, 1U, __ATOMIC_RELEASE);
  }
  owner.pi_->_M_weak_add_ref();
  using CleanBase = std::remove_cv_t<Base>;
  CleanBase* base_ptr = static_cast<CleanBase*>(raw_ptr);
  const RawStdPtrLayout next{static_cast<void*>(base_ptr), owner.pi_};
  std::memcpy(static_cast<void*>(std_base), &next, sizeof(next));
}
#endif

template <typename OwnerT, typename Ptr>
void SpEsftAccess::maybe_enable(const SharedPtr<OwnerT>& owner,
                                Ptr* raw_ptr) noexcept {
  if (raw_ptr == nullptr) {
    return;
  }
  if constexpr (!std::is_array_v<OwnerT> && !std::is_void_v<Ptr>) {
    using RawClean = std::remove_cv_t<Ptr>;
    if constexpr (requires(RawClean* p) { detect_portable(p); }) {
      auto* base = detect_portable(const_cast<RawClean*>(raw_ptr));
      if (base != nullptr && base->weak_this_.expired()) {
        using BaseElem = typename decltype(base->weak_this_)::element_type;
        base->weak_this_ = SharedPtr<BaseElem>(
            owner,
            const_cast<BaseElem*>(static_cast<const BaseElem*>(raw_ptr)));
      }
    }
#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
    if constexpr (requires(RawClean* p) { detect_std(p); }) {
      auto* std_base = detect_std(const_cast<RawClean*>(raw_ptr));
      if (std_base != nullptr) {
        init_std_esft(owner, std_base, const_cast<RawClean*>(raw_ptr));
      }
    }
#endif
  }
}

}  // namespace detail::shared_ptr

// ============================================================================
// `theta::portable::KeepAlive<T>`: Portable Hazard-Slot + Striped-Slab Cell
// ============================================================================
template <typename T>
class KeepAlive {
 public:
  using element_type = typename SharedPtr<T>::element_type;

  KeepAlive() noexcept = default;
  explicit KeepAlive(element_type* ptr) { reset(ptr); }
  explicit KeepAlive(SharedPtr<T> sp) { reset(std::move(sp)); }
  ~KeepAlive() noexcept { reset(SharedPtr<T>{}); }

  KeepAlive(const KeepAlive&) = delete;
  KeepAlive& operator=(const KeepAlive&) = delete;

  SharedPtr<T> reset(element_type* ptr) {
    if (ptr == nullptr) {
      return reset(SharedPtr<T>{});
    }
    return reset(SharedPtr<T>(ptr));
  }

  SharedPtr<T> reset(SharedPtr<T> next) {
    detail::shared_ptr::PortableSpCountedBase* new_cb = nullptr;
    if (next.pi_ != nullptr) {
      void* raw_managed =
          const_cast<void*>(static_cast<const volatile void*>(next.get()));
      if (!detail::shared_ptr::is_portable_sp(next.pi_)) [[unlikely]] {
        new_cb = wrap_and_pin_cold(next, raw_managed);
      } else {
        auto* cand =
            static_cast<detail::shared_ptr::PortableSpCountedBase*>(next.pi_);
        const auto pin_res =
            detail::shared_ptr::sp_pin_for_keepalive(cand, raw_managed);
        if (pin_res == detail::shared_ptr::SpPinResult::kPinned) [[likely]] {
          new_cb = cand;
        } else if (pin_res ==
                   detail::shared_ptr::SpPinResult::kThrottledStorm) {
          new_cb = detail::shared_ptr::sp_keepalive_unpinned_sentinel();
        } else {
          new_cb = wrap_and_pin_cold(next, raw_managed);
        }
      }
    }

    SharedPtr<T> ret;
    SharedPtr<T> old_anchor;
    detail::shared_ptr::PortableSpCountedBase* old_cb = nullptr;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      old_cb = __atomic_exchange_n(&active_cb_, new_cb, __ATOMIC_SEQ_CST);
      old_anchor = std::move(anchor_);
      anchor_ = std::move(next);
      ret = anchor_;
    }
    detail::shared_ptr::PortableSpCountedBase* const sentinel =
        detail::shared_ptr::sp_keepalive_unpinned_sentinel();
    if (old_cb != nullptr && old_cb != sentinel && old_anchor.pi_ == old_cb)
        [[likely]] {
      old_anchor.ptr_ = nullptr;
      old_anchor.pi_ = nullptr;
      detail::shared_ptr::sp_unpin_and_retire_old(
          old_cb, /*drop_anchor_ref=*/true,
          /*quiesce_readers=*/(old_cb != new_cb));
    } else if (old_cb != nullptr && old_cb != sentinel) {
      detail::shared_ptr::sp_unpin_and_retire_old(
          old_cb, /*drop_anchor_ref=*/false,
          /*quiesce_readers=*/(old_cb != new_cb));
    }
    return ret;
  }

  void store(SharedPtr<T> desired) { (void)reset(std::move(desired)); }

  [[nodiscard]] THETA_FORCE_INLINE SharedPtr<T> get() const noexcept {
    auto* const cb_peek = __atomic_load_n(&active_cb_, __ATOMIC_RELAXED);
    if (cb_peek == nullptr) [[unlikely]] {
      (void)__atomic_load_n(&active_cb_, __ATOMIC_ACQUIRE);
      return SharedPtr<T>{};
    }
    if (cb_peek == detail::shared_ptr::sp_keepalive_unpinned_sentinel())
        [[unlikely]] {
      return get_slow();
    }
    const auto tls = detail::shared_ptr::g_portable_fast_tls;
    if (tls.gate == nullptr) [[unlikely]] {
      return get_slow();
    }
    const bool gate_ok =
        detail::shared_ptr::PortableBackend::enter_owner_gate(tls.gate);
    auto* const cb = __atomic_load_n(&active_cb_, __ATOMIC_SEQ_CST);
    if (!gate_ok || cb != cb_peek) [[unlikely]] {
      detail::shared_ptr::PortableBackend::exit_owner_gate(tls.gate);
      return get_slow();
    }
    const std::uint32_t slot =
        __atomic_load_n(&cb->fast_.slot_id, __ATOMIC_RELAXED);
    std::int64_t* const cell = tls.stripe_base + slot;
    const std::int64_t cur = __atomic_load_n(cell, __ATOMIC_RELAXED);
    if (cur >= 1 && cur < 65) [[likely]] {
      __atomic_store_n(cell, cur + 1, __ATOMIC_RELAXED);
      auto* const managed = static_cast<element_type*>(
          __atomic_load_n(&detail::shared_ptr::g_portable_keepalive_ptrs[slot],
                          __ATOMIC_RELAXED));
      detail::shared_ptr::PortableBackend::exit_owner_gate(tls.gate);
      return SharedPtr<T>(managed, cb, detail::shared_ptr::AdoptRefTag{});
    }
    auto* const managed = static_cast<element_type*>(
        detail::shared_ptr::sp_acquire_pinned_under_hazard_slow(cb, tls.gate));
    return SharedPtr<T>(managed, cb, detail::shared_ptr::AdoptRefTag{});
  }

  [[nodiscard]] THETA_FORCE_INLINE SharedPtr<T> load() const noexcept {
    return get();
  }

 private:
  [[gnu::noinline, gnu::cold]] static detail::shared_ptr::PortableSpCountedBase*
  wrap_and_pin_cold(SharedPtr<T>& next, void* raw_managed) {
    if (!detail::shared_ptr::is_portable_sp(next.pi_)) {
#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
      std::shared_ptr<T> ext = next.to_std();
      element_type* raw_p = ext.get();
      next = SharedPtr<T>(raw_p, [hold = std::move(ext)](
                                     element_type*) mutable { hold.reset(); });
#else
      SharedPtr<T> ext = std::move(next);
      element_type* raw_p = ext.get();
      next = SharedPtr<T>(raw_p, [hold = std::move(ext)](
                                     element_type*) mutable { hold.reset(); });
#endif
    }
    auto* new_cb =
        static_cast<detail::shared_ptr::PortableSpCountedBase*>(next.pi_);
    auto pin_res =
        detail::shared_ptr::sp_pin_for_keepalive(new_cb, raw_managed);
    if (pin_res == detail::shared_ptr::SpPinResult::kAliasConflict) {
      SharedPtr<T> hold_inner = std::move(next);
      element_type* raw_p = hold_inner.get();
      next = SharedPtr<T>(raw_p, [hold = std::move(hold_inner)](
                                     element_type*) mutable { hold.reset(); });
      new_cb =
          static_cast<detail::shared_ptr::PortableSpCountedBase*>(next.pi_);
      pin_res = detail::shared_ptr::sp_pin_for_keepalive(new_cb, raw_managed);
    }
    if (pin_res == detail::shared_ptr::SpPinResult::kPinned) {
      return new_cb;
    }
    return detail::shared_ptr::sp_keepalive_unpinned_sentinel();
  }

  [[gnu::noinline]] SharedPtr<T> get_slow() const noexcept {
    detail::shared_ptr::PortableSpCountedBase* const sentinel =
        detail::shared_ptr::sp_keepalive_unpinned_sentinel();
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      if (!anchor_) {
        return SharedPtr<T>{};
      }
      if (__atomic_load_n(&active_cb_, __ATOMIC_ACQUIRE) != sentinel &&
          detail::shared_ptr::is_portable_sp(anchor_.pi_)) {
        auto* cb = static_cast<detail::shared_ptr::PortableSpCountedBase*>(
            anchor_.pi_);
        detail::shared_ptr::sp_register_keepalive_stripe_slow(cb);
      }
      return anchor_;
    }
  }

  mutable detail::shared_ptr::PortableSpCountedBase* active_cb_{nullptr};
  mutable std::shared_mutex mutex_;
  SharedPtr<T> anchor_;
};

}  // namespace theta::portable

// ============================================================================
// Standard Library Specializations: `std::hash`,
// `std::atomic<theta::portable::SharedPtr<T>>`, and
// `std::atomic<theta::portable::WeakPtr<T>>`
// ============================================================================
namespace std {

template <typename T>
struct hash<::theta::portable::SharedPtr<T>> {
  [[nodiscard]] std::size_t operator()(
      const ::theta::portable::SharedPtr<T>& p) const noexcept {
    return std::hash<typename ::theta::portable::SharedPtr<T>::element_type*>{}(
        p.get());
  }
};

template <typename T>
struct atomic<::theta::portable::SharedPtr<T>> {
  using value_type = ::theta::portable::SharedPtr<T>;
  using element_type = typename value_type::element_type;
  static constexpr bool is_always_lock_free = false;

  static_assert(
      alignof(::theta::portable::detail::shared_ptr::StdSpCountedBase) >= 8U,
      "StdSpCountedBase must be >= 8B aligned for 3-bit rep tags.");

  [[nodiscard]] bool is_lock_free() const noexcept { return false; }

  constexpr atomic() noexcept = default;
  constexpr atomic(std::nullptr_t) noexcept : ptr_(nullptr), rep_(0ULL) {}
  atomic(value_type desired) noexcept {
    element_type* p = nullptr;
    ::theta::portable::detail::shared_ptr::StdSpCountedBase* pi = nullptr;
    ::theta::portable::detail::shared_ptr::SpAtomicAccess::take_parts(
        std::move(desired), p, pi);
    ptr_ = p;
    rep_ = (p == nullptr && pi == nullptr)
               ? 0ULL
               : (reinterpret_cast<std::uintptr_t>(pi) | kRepUnpinnedBit);
  }

  ~atomic() noexcept {
    const std::uintptr_t cur = __atomic_load_n(&rep_, __ATOMIC_RELAXED);
    auto* const pi = reinterpret_cast<
        ::theta::portable::detail::shared_ptr::StdSpCountedBase*>(
        cur & ~kRepFlagsMask);
    if (pi != nullptr) {
      if ((cur & kRepUnpinnedBit) == 0ULL) {
        ::theta::portable::detail::shared_ptr::sp_unpin_and_retire_old(
            static_cast<
                ::theta::portable::detail::shared_ptr::PortableSpCountedBase*>(
                pi),
            /*drop_anchor_ref=*/true, /*quiesce_readers=*/false);
      } else {
        ::theta::portable::detail::shared_ptr::sp_release(pi);
      }
    }
  }

  atomic(const atomic&) = delete;
  void operator=(const atomic&) = delete;

  void operator=(value_type desired) noexcept { store(std::move(desired)); }
  void operator=(std::nullptr_t) noexcept { store(nullptr); }

  [[nodiscard]] THETA_FORCE_INLINE value_type
  load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
    (void)order;
    const std::uintptr_t rep_peek = __atomic_load_n(&rep_, __ATOMIC_RELAXED);
    if (rep_peek == 0ULL) [[unlikely]] {
      (void)__atomic_load_n(&rep_, __ATOMIC_ACQUIRE);
      return value_type{};
    }
    if ((rep_peek & kRepUnpinnedBit) != 0ULL) [[unlikely]] {
      return load_slow();
    }
    const auto tls = ::theta::portable::detail::shared_ptr::g_portable_fast_tls;
    if (tls.gate == nullptr) [[unlikely]] {
      return load_slow();
    }
    const bool gate_ok = ::theta::portable::detail::shared_ptr::
        PortableBackend::enter_owner_gate(tls.gate);
    const std::uintptr_t rep = __atomic_load_n(&rep_, __ATOMIC_SEQ_CST);
    if (!gate_ok || rep != rep_peek) [[unlikely]] {
      ::theta::portable::detail::shared_ptr::PortableBackend::exit_owner_gate(
          tls.gate);
      return load_slow();
    }
    auto* const cb = reinterpret_cast<
        ::theta::portable::detail::shared_ptr::PortableSpCountedBase*>(rep);
    const std::uint32_t slot =
        __atomic_load_n(&cb->fast_.slot_id, __ATOMIC_RELAXED);
    std::int64_t* const cell = tls.stripe_base + slot;
    const std::int64_t cur = __atomic_load_n(cell, __ATOMIC_RELAXED);
    if (cur >= 1 && cur < 65) [[likely]] {
      __atomic_store_n(cell, cur + 1, __ATOMIC_RELAXED);
      auto* const managed = static_cast<element_type*>(
          __atomic_load_n(&::theta::portable::detail::shared_ptr::
                              g_portable_keepalive_ptrs[slot],
                          __ATOMIC_RELAXED));
      ::theta::portable::detail::shared_ptr::PortableBackend::exit_owner_gate(
          tls.gate);
      return ::theta::portable::detail::shared_ptr::SpAtomicAccess::
          make_adopted<T>(managed, cb);
    }
    auto* const managed = static_cast<element_type*>(
        ::theta::portable::detail::shared_ptr::
            sp_acquire_pinned_under_hazard_slow(cb, tls.gate));
    return ::theta::portable::detail::shared_ptr::SpAtomicAccess::make_adopted<
        T>(managed, cb);
  }

  operator value_type() const noexcept { return load(); }

  void store(value_type desired,
             std::memory_order order = std::memory_order_seq_cst) noexcept {
    (void)exchange_or_store<false>(std::move(desired), order);
  }

  value_type exchange(
      value_type desired,
      std::memory_order order = std::memory_order_seq_cst) noexcept {
    return exchange_or_store<true>(std::move(desired), order);
  }

  bool compare_exchange_strong(value_type& expected, value_type desired,
                               std::memory_order success,
                               std::memory_order /*failure*/) noexcept {
    void* exp_p = nullptr;
    ::theta::portable::detail::shared_ptr::StdSpCountedBase* exp_pi = nullptr;
    ::theta::portable::detail::shared_ptr::SpAtomicAccess::raw_parts(
        expected, exp_p, exp_pi);

    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    const std::uintptr_t pre = __atomic_load_n(&rep_, __ATOMIC_RELAXED);
    element_type* const cur_p = ptr_;
    auto* const cur_pi = reinterpret_cast<
        ::theta::portable::detail::shared_ptr::StdSpCountedBase*>(
        pre & ~kRepFlagsMask);
    const void* const cur_raw_p =
        const_cast<const void*>(static_cast<const volatile void*>(cur_p));

    if (cur_raw_p == exp_p && cur_pi == exp_pi) {
      element_type* new_p = nullptr;
      ::theta::portable::detail::shared_ptr::StdSpCountedBase* new_pi = nullptr;
      ::theta::portable::detail::shared_ptr::SpAtomicAccess::take_parts(
          std::move(desired), new_p, new_pi);

      std::uintptr_t next_rep = pre;
      bool was_pinned = false;
      if (new_p != cur_p || new_pi != cur_pi) {
        was_pinned = (cur_pi != nullptr) && ((pre & kRepUnpinnedBit) == 0ULL);
        ptr_ = new_p;
        if (new_p == nullptr && new_pi == nullptr) {
          next_rep = 0ULL;
        } else {
          std::uintptr_t flags = kRepUnpinnedBit;
          if (was_pinned && new_p != nullptr &&
              ::theta::portable::detail::shared_ptr::is_portable_sp(new_pi)) {
            auto* const new_cb = static_cast<
                ::theta::portable::detail::shared_ptr::PortableSpCountedBase*>(
                new_pi);
            const auto pin_res =
                ::theta::portable::detail::shared_ptr::sp_pin_for_keepalive(
                    new_cb, const_cast<void*>(
                                static_cast<const volatile void*>(new_p)));
            if (pin_res == ::theta::portable::detail::shared_ptr::SpPinResult::
                               kPinned &&
                __atomic_load_n(&new_cb->fast_.mode, __ATOMIC_ACQUIRE) ==
                    ::theta::portable::detail::shared_ptr::kModePerCpuActive) {
              flags = 0ULL;
            } else if (pin_res == ::theta::portable::detail::shared_ptr::
                                      SpPinResult::kAliasConflict) {
              flags = kRepUnpinnedBit | kRepNoPinBit;
            }
          }
          next_rep = reinterpret_cast<std::uintptr_t>(new_pi) | flags;
        }
        publish_rep_locked(next_rep, success);
      }

      ::theta::portable::detail::shared_ptr::atomic_ptr_unlock_and_notify(this,
                                                                          true);
      if (was_pinned) [[unlikely]] {
        ::theta::portable::detail::shared_ptr::sp_unpin_and_retire_old(
            static_cast<
                ::theta::portable::detail::shared_ptr::PortableSpCountedBase*>(
                cur_pi),
            /*drop_anchor_ref=*/true, /*quiesce_readers=*/true);
      } else if (new_p != cur_p || new_pi != cur_pi) {
        ::theta::portable::detail::shared_ptr::sp_release(cur_pi);
      } else {
        ::theta::portable::detail::shared_ptr::sp_release(new_pi);
      }
      return true;
    }

    if (cur_pi != nullptr) {
      ::theta::portable::detail::shared_ptr::sp_add_ref(cur_pi);
    }
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock(this);
    expected =
        ::theta::portable::detail::shared_ptr::SpAtomicAccess::make_adopted<T>(
            cur_p, cur_pi);
    return false;
  }

  bool compare_exchange_strong(
      value_type& expected, value_type desired,
      std::memory_order order = std::memory_order_seq_cst) noexcept {
    return compare_exchange_strong(expected, std::move(desired), order, order);
  }

  bool compare_exchange_weak(value_type& expected, value_type desired,
                             std::memory_order success,
                             std::memory_order failure) noexcept {
    return compare_exchange_strong(expected, std::move(desired), success,
                                   failure);
  }

  bool compare_exchange_weak(
      value_type& expected, value_type desired,
      std::memory_order order = std::memory_order_seq_cst) noexcept {
    return compare_exchange_strong(expected, std::move(desired), order, order);
  }

  void wait(
      const value_type& old,
      std::memory_order /*order*/ = std::memory_order_seq_cst) const noexcept {
    void* old_p = nullptr;
    ::theta::portable::detail::shared_ptr::StdSpCountedBase* old_pi = nullptr;
    ::theta::portable::detail::shared_ptr::SpAtomicAccess::raw_parts(old, old_p,
                                                                     old_pi);

    while (true) {
      ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
      const std::uintptr_t pre = __atomic_load_n(&rep_, __ATOMIC_RELAXED);
      const void* const cur_p =
          const_cast<const void*>(static_cast<const volatile void*>(ptr_));
      const auto* const cur_pi = reinterpret_cast<
          const ::theta::portable::detail::shared_ptr::StdSpCountedBase*>(
          pre & ~kRepFlagsMask);
      if (cur_p != old_p || cur_pi != old_pi) {
        ::theta::portable::detail::shared_ptr::atomic_ptr_unlock(this);
        return;
      }
      const std::uint32_t seq =
          ::theta::portable::detail::shared_ptr::atomic_ptr_seq(this);
      ::theta::portable::detail::shared_ptr::atomic_ptr_unlock(this);
      ::theta::portable::detail::shared_ptr::atomic_ptr_wait_seq(this, seq);
    }
  }

  void notify_one() noexcept {
    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock_and_notify(this,
                                                                        false);
  }

  void notify_all() noexcept {
    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock_and_notify(this,
                                                                        true);
  }

 private:
  static constexpr std::uintptr_t kRepUnpinnedBit = 2ULL;
  static constexpr std::uintptr_t kRepNoPinBit = 4ULL;
  static constexpr std::uintptr_t kRepFlagsMask = 7ULL;

  THETA_FORCE_INLINE void publish_rep_locked(
      std::uintptr_t next_rep, std::memory_order /*order*/) const noexcept {
    // Always publish with `SEQ_CST` (`xchg` on x86-64, `swpal`/`stlr+dmb` on
    // ARM64) so Dekker ordering holds against reader `hazard_ptr` `SEQ_CST`.
    (void)__atomic_exchange_n(&rep_, next_rep, __ATOMIC_SEQ_CST);
  }

  [[gnu::noinline]] value_type load_slow() const noexcept {
    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    const std::uintptr_t pre = __atomic_load_n(&rep_, __ATOMIC_RELAXED);
    element_type* const p = ptr_;
    auto* const pi = reinterpret_cast<
        ::theta::portable::detail::shared_ptr::StdSpCountedBase*>(
        pre & ~kRepFlagsMask);
    std::uintptr_t next_rep = pre;
    if (pi != nullptr) {
      ::theta::portable::detail::shared_ptr::sp_add_ref(pi);
      if (::theta::portable::detail::shared_ptr::is_portable_sp(pi)) {
        auto* const cb = static_cast<
            ::theta::portable::detail::shared_ptr::PortableSpCountedBase*>(pi);
        const std::uint8_t m =
            __atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE);
        if ((pre & kRepUnpinnedBit) != 0ULL) {
          if ((pre & kRepNoPinBit) == 0ULL && p != nullptr &&
              m == ::theta::portable::detail::shared_ptr::kModePerCpuActive) {
            bool never_pin = false;
            if (::theta::portable::detail::shared_ptr::sp_try_pin_atomic_fast(
                    cb, const_cast<void*>(static_cast<const volatile void*>(p)),
                    never_pin)) {
              next_rep = reinterpret_cast<std::uintptr_t>(cb);
            } else if (never_pin) {
              next_rep = pre | kRepNoPinBit;
            }
          }
        } else if (m !=
                   ::theta::portable::detail::shared_ptr::kModePerCpuActive) {
          next_rep = pre | kRepUnpinnedBit | kRepNoPinBit;
        } else {
          ::theta::portable::detail::shared_ptr::
              sp_register_keepalive_stripe_slow(cb);
        }
      }
    }
    if (next_rep != pre) [[unlikely]] {
      (void)__atomic_exchange_n(&rep_, next_rep, __ATOMIC_SEQ_CST);
    }
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock(this);
    return ::theta::portable::detail::shared_ptr::SpAtomicAccess::make_adopted<
        T>(p, pi);
  }

  template <bool ReturnOld>
  value_type exchange_or_store(value_type desired,
                               std::memory_order order) noexcept {
    element_type* new_p = nullptr;
    ::theta::portable::detail::shared_ptr::StdSpCountedBase* new_pi = nullptr;
    ::theta::portable::detail::shared_ptr::SpAtomicAccess::take_parts(
        std::move(desired), new_p, new_pi);

    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    const std::uintptr_t pre = __atomic_load_n(&rep_, __ATOMIC_RELAXED);
    element_type* const old_p = ptr_;
    auto* const old_pi = reinterpret_cast<
        ::theta::portable::detail::shared_ptr::StdSpCountedBase*>(
        pre & ~kRepFlagsMask);

    std::uintptr_t next_rep = pre;
    bool was_pinned = false;
    if (new_p != old_p || new_pi != old_pi) {
      was_pinned = (old_pi != nullptr) && ((pre & kRepUnpinnedBit) == 0ULL);
      ptr_ = new_p;
      if (new_p == nullptr && new_pi == nullptr) {
        next_rep = 0ULL;
      } else {
        std::uintptr_t flags = kRepUnpinnedBit;
        if (was_pinned && new_p != nullptr &&
            ::theta::portable::detail::shared_ptr::is_portable_sp(new_pi)) {
          auto* const new_cb = static_cast<
              ::theta::portable::detail::shared_ptr::PortableSpCountedBase*>(
              new_pi);
          const auto pin_res =
              ::theta::portable::detail::shared_ptr::sp_pin_for_keepalive(
                  new_cb,
                  const_cast<void*>(static_cast<const volatile void*>(new_p)));
          if (pin_res ==
                  ::theta::portable::detail::shared_ptr::SpPinResult::kPinned &&
              __atomic_load_n(&new_cb->fast_.mode, __ATOMIC_ACQUIRE) ==
                  ::theta::portable::detail::shared_ptr::kModePerCpuActive) {
            flags = 0ULL;
          } else if (pin_res == ::theta::portable::detail::shared_ptr::
                                    SpPinResult::kAliasConflict) {
            flags = kRepUnpinnedBit | kRepNoPinBit;
          }
        }
        next_rep = reinterpret_cast<std::uintptr_t>(new_pi) | flags;
      }
      publish_rep_locked(next_rep, order);
    }

    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock_and_notify(this,
                                                                        true);

    if constexpr (ReturnOld) {
      if (was_pinned) {
        ::theta::portable::detail::shared_ptr::sp_unpin_and_retire_old(
            static_cast<
                ::theta::portable::detail::shared_ptr::PortableSpCountedBase*>(
                old_pi),
            /*drop_anchor_ref=*/false, /*quiesce_readers=*/true);
      }
      return ::theta::portable::detail::shared_ptr::SpAtomicAccess::
          make_adopted<T>(old_p, old_pi);
    } else {
      if (was_pinned) {
        ::theta::portable::detail::shared_ptr::sp_unpin_and_retire_old(
            static_cast<
                ::theta::portable::detail::shared_ptr::PortableSpCountedBase*>(
                old_pi),
            /*drop_anchor_ref=*/true, /*quiesce_readers=*/true);
      } else {
        ::theta::portable::detail::shared_ptr::sp_release(old_pi);
      }
      return value_type{};
    }
  }

  mutable element_type* ptr_{nullptr};
  mutable std::uintptr_t rep_{0ULL};
};

static_assert(sizeof(std::atomic<::theta::portable::SharedPtr<int>>) ==
              2 * sizeof(void*));

template <typename T>
struct atomic<::theta::portable::WeakPtr<T>> {
  using value_type = ::theta::portable::WeakPtr<T>;
  static constexpr bool is_always_lock_free = false;

  [[nodiscard]] bool is_lock_free() const noexcept { return false; }

  constexpr atomic() noexcept = default;
  atomic(value_type desired) noexcept : val_(std::move(desired)) {}

  atomic(const atomic&) = delete;
  void operator=(const atomic&) = delete;

  void operator=(value_type desired) noexcept { store(std::move(desired)); }

  [[nodiscard]] value_type load(
      std::memory_order /*order*/ = std::memory_order_seq_cst) const noexcept {
    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    value_type out = val_;
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock(this);
    return out;
  }

  operator value_type() const noexcept { return load(); }

  void store(value_type desired,
             std::memory_order /*order*/ = std::memory_order_seq_cst) noexcept {
    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    val_.swap(desired);
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock_and_notify(this,
                                                                        true);
  }

  value_type exchange(
      value_type desired,
      std::memory_order /*order*/ = std::memory_order_seq_cst) noexcept {
    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    val_.swap(desired);
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock_and_notify(this,
                                                                        true);
    return desired;
  }

  bool compare_exchange_strong(value_type& expected, value_type desired,
                               std::memory_order /*success*/,
                               std::memory_order /*failure*/) noexcept {
    value_type old_to_drop;
    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    if (same_ptr_and_owner(val_, expected)) {
      old_to_drop = std::move(val_);
      val_ = std::move(desired);
      ::theta::portable::detail::shared_ptr::atomic_ptr_unlock_and_notify(this,
                                                                          true);
      return true;
    }
    value_type current_copy = val_;
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock(this);
    expected = std::move(current_copy);
    return false;
  }

  bool compare_exchange_strong(
      value_type& expected, value_type desired,
      std::memory_order order = std::memory_order_seq_cst) noexcept {
    return compare_exchange_strong(expected, std::move(desired), order, order);
  }

  bool compare_exchange_weak(value_type& expected, value_type desired,
                             std::memory_order success,
                             std::memory_order failure) noexcept {
    return compare_exchange_strong(expected, std::move(desired), success,
                                   failure);
  }

  bool compare_exchange_weak(
      value_type& expected, value_type desired,
      std::memory_order order = std::memory_order_seq_cst) noexcept {
    return compare_exchange_strong(expected, std::move(desired), order, order);
  }

  void wait(
      const value_type& old,
      std::memory_order /*order*/ = std::memory_order_seq_cst) const noexcept {
    while (true) {
      ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
      const std::uint32_t seq =
          ::theta::portable::detail::shared_ptr::atomic_ptr_seq(this);
      const bool matches = same_ptr_and_owner(val_, old);
      ::theta::portable::detail::shared_ptr::atomic_ptr_unlock(this);
      if (!matches) {
        return;
      }
      ::theta::portable::detail::shared_ptr::atomic_ptr_wait_seq(this, seq);
    }
  }

  void notify_one() noexcept {
    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock_and_notify(this,
                                                                        false);
  }

  void notify_all() noexcept {
    ::theta::portable::detail::shared_ptr::atomic_ptr_lock(this);
    ::theta::portable::detail::shared_ptr::atomic_ptr_unlock_and_notify(this,
                                                                        true);
  }

 private:
  static bool same_ptr_and_owner(const value_type& a,
                                 const value_type& b) noexcept {
    void* ap = nullptr;
    void* bp = nullptr;
    ::theta::portable::detail::shared_ptr::StdSpCountedBase* api = nullptr;
    ::theta::portable::detail::shared_ptr::StdSpCountedBase* bpi = nullptr;
    ::theta::portable::detail::shared_ptr::SpAtomicAccess::raw_parts(a, ap,
                                                                     api);
    ::theta::portable::detail::shared_ptr::SpAtomicAccess::raw_parts(b, bp,
                                                                     bpi);
    return ap == bp && api == bpi;
  }

  mutable value_type val_{};
};

}  // namespace std
