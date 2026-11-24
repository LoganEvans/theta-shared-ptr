// FILENAME: include/theta/shared_ptr.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT
//
// `theta::SharedPtr<T>` / `theta::WeakPtr<T>` / `theta::KeepAlive<T>`
// ----------------------------------------------------------------------------
// 100% binary `std::shared_ptr<T>` and `std::weak_ptr<T>` ABI-compatible
// concurrent smart pointer powered by Linux `rseq` (`0 .. 8,191` CPUs).
//
// Memory & Virtual-Address Footprint Contract:
//   1. Handle layout (`16 B`):
//      `sizeof(theta::SharedPtr<T>) == 16` (`[T*, _Sp_counted_base*]`),
//      untagged and binary-identical to libstdc++ `std::shared_ptr<T>`.
//   2. Heap control block (`32 B` `ThetaSpCountedBase`):
//      `make_shared<T>` allocates `32 B + sizeof(T)` in a single allocation
//      (`+8 B` over `std::make_shared<T>`'s `24 B + sizeof(T)`; `40 B` -> same
//      `48 B` glibc `malloc` bin for 8-byte types).
//   3. Uncopied & ephemeral pointers (`0 .. 7` copies -> `0 B` extra RSS):
//      Sole-owner pointers (`0` copies, incl. cross-CPU `std::move`) stay in
//      `kModeUnpromoted` and destroy with `0` atomic RMWs and `0` virtual
//      calls. Ephemeral shared pointers (`1 .. 7` copies, e.g. cross-CPU RPC
//      or queue handoffs) stay in `kModeWarmup` inline inside `SpFastHeader`,
//      allocating `0` per-CPU slots, faulting `0` slab pages, and firing `0`
//      `SYS_membarrier` IPIs on cross-CPU teardown.
//   4. Promoted (`>= 8` copies) pointers & `KeepAlive<T>` (`MAP_NORESERVE`):
//      - Virtual address reservation (uncommitted, `0 B` physical RAM until
//        faulted): `131,072` slots * `8,192` CPUs * `8 B` = `8 GiB` slab VA
//        (`256 MiB` under ASan/TSan) + `8 MiB` slot-metadata VA + `1 MiB`
//        `KeepAlive` table VA.
//      - Cold-start RSS (first promoted pointer in process): slabs are indexed
//        as `(cpu << 17) + slot_id`, packing `512` consecutive `slot_id`s onto
//        one `4 KiB` page per CPU. The first promoted pointer faults `1 * 4
//        KiB` metadata page + `1 * 4 KiB` slab page per CPU that actually
//        copies it (e.g., `~52 KiB` across 12 CPUs; `0 B` on untouched CPUs
//        `12..8191`).
//      - Marginal RSS (pointers `#2..#511`, and recycled slots):
//        `64 B` (`PerCpuSlotMeta`) + `8 B * active_cpus` per concurrently live
//        promoted control block (e.g., `160 B`/ptr across 12 CPUs, `576 B`/ptr
//        across 64 CPUs). Recycled `slot_id`s reuse already-faulted pages via
//        `initial-exec` thread-local caches.
//      - Graceful saturation: up to `131,072` concurrently promoted objects use
//        `rseq` slabs; excess promotions transparently fall back to atomic
//        `_Sp_counted_base` reference counting (`0` allocation failures).

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
#include <theta/shared_ptr/detail/common.hpp>
#include <type_traits>
#include <typeinfo>
#include <utility>

#if THETA_ARCH_X86_64 && THETA_SHARED_PTR_USE_RSEQ_ASM
#include <theta/shared_ptr/detail/backend_x86_64.hpp>
#elif THETA_ARCH_AARCH64 && THETA_SHARED_PTR_USE_RSEQ_ASM
#include <theta/shared_ptr/detail/backend_aarch64.hpp>
#else
#include <theta/shared_ptr/detail/backend_fallback.hpp>
#endif

namespace theta {

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

// STL-compatible snake_case drop-in aliases
// (`s/std::shared_ptr/theta::shared_ptr/`)
template <typename T>
using shared_ptr = SharedPtr<T>;

template <typename T>
using weak_ptr = WeakPtr<T>;

template <typename T>
using enable_shared_from_this = EnableSharedFromThis<T>;

template <typename T = void>
using owner_less = OwnerLess<T>;

namespace detail::shared_ptr {

struct RawStdPtrLayout {
  void* ptr;
  StdSpCountedBase* pi;
};

struct AdoptRefTag {};
struct ForOverwriteTag {};

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
static_assert(sizeof(std::shared_ptr<int>) == sizeof(RawStdPtrLayout),
              "std::shared_ptr<T> must be 16 bytes (ptr + _Sp_counted_base*).");
static_assert(alignof(std::shared_ptr<int>) == alignof(RawStdPtrLayout),
              "std::shared_ptr<T> alignment must match RawStdPtrLayout.");
static_assert(sizeof(std::weak_ptr<int>) == sizeof(RawStdPtrLayout),
              "std::weak_ptr<T> must be 16 bytes (ptr + _Sp_counted_base*).");
#endif

// C++20 [util.smartptr.shared] pointer-compatibility trait (`Y*` compatible
// with `T*`, including `U[N]` -> `U[]` and cv-qualified array conversions).
template <typename Y, typename T>
struct SpCompatible : std::is_convertible<Y*, T*> {};

template <typename U, std::size_t N>
struct SpCompatible<U[N], U[]> : std::true_type {};

template <typename U, std::size_t N>
struct SpCompatible<U[N], const U[]> : std::true_type {};

template <typename U, std::size_t N>
struct SpCompatible<U[N], volatile U[]> : std::true_type {};

template <typename U, std::size_t N>
struct SpCompatible<U[N], const volatile U[]> : std::true_type {};

template <typename U>
struct SpCompatible<U[], const U[]> : std::true_type {};

template <typename U>
struct SpCompatible<U[], volatile U[]> : std::true_type {};

template <typename U>
struct SpCompatible<U[], const volatile U[]> : std::true_type {};

template <typename Y, typename T>
inline constexpr bool kSpCompatibleV = SpCompatible<Y, T>::value;

template <typename Y, typename T>
struct SpRawPtrCompatible : std::is_convertible<Y*, T*> {};

template <typename Y, typename U>
struct SpRawPtrCompatible<Y, U[]> : SpCompatible<Y[], U[]> {};

template <typename Y, typename U, std::size_t N>
struct SpRawPtrCompatible<Y, U[N]> : SpCompatible<Y[N], U[N]> {};

template <typename Y, typename T>
inline constexpr bool kSpRawPtrCompatibleV = SpRawPtrCompatible<Y, T>::value;

template <typename T>
struct SpRemoveCvTarget {
  using type = std::remove_cv_t<T>;
};

template <typename U, std::size_t N>
struct SpRemoveCvTarget<U[N]> {
  using type = std::remove_cv_t<U>[N];
};

template <typename T>
using SpRemoveCvTargetT = typename SpRemoveCvTarget<T>::type;

THETA_FORCE_INLINE void sp_add_ref(StdSpCountedBase* pi) noexcept {
  if (pi == nullptr) {
    return;
  }
  if (is_theta_sp(pi)) [[likely]] {
    auto* cb = static_cast<ThetaSpCountedBase*>(pi);
    if (__atomic_load_n(&cb->fast_.fast_active, __ATOMIC_RELAXED) != 0U)
        [[likely]] {
      if (ArchBackend::try_rseq_inc(&cb->fast_)) [[likely]] {
        return;
      }
    }
  }
  sp_add_ref_slow(pi);
}

THETA_FORCE_INLINE void sp_release(StdSpCountedBase* pi) noexcept {
  if (pi == nullptr) {
    return;
  }
  if (is_theta_sp(pi)) [[likely]] {
    auto* cb = static_cast<ThetaSpCountedBase*>(pi);
    if (__atomic_load_n(&cb->fast_.fast_active, __ATOMIC_RELAXED) != 0U)
        [[likely]] {
      if (ArchBackend::try_rseq_dec(&cb->fast_)) [[likely]] {
        return;
      }
      sp_release_slow(pi);
      return;
    }
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) && \
    !THETA_TSAN_ENABLED
    std::uint32_t state32 = 0U;
    std::memcpy(&state32, &cb->fast_.fast_active, sizeof(state32));
    std::atomic_signal_fence(std::memory_order_acquire);
    if ((state32 & 0x00FFFFFFU) == 0U) [[likely]] {
      const std::uintptr_t tag = cb->ops_or_size_;
      if ((tag & 1U) != 0U) [[likely]] {
        ::operator delete(static_cast<void*>(cb), tag >> 1U);
      } else {
        reinterpret_cast<const SpColdOps*>(tag)->dispose_and_destroy(cb);
      }
      return;
    }
#else
    if (__atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE) == kModeUnpromoted &&
        __atomic_load_n(&cb->fast_.has_weak_or_std, __ATOMIC_ACQUIRE) == 0U)
        [[likely]] {
      const std::uintptr_t tag = cb->ops_or_size_;
      if ((tag & 1U) != 0U) [[likely]] {
        ::operator delete(static_cast<void*>(cb), tag >> 1U);
      } else {
        reinterpret_cast<const SpColdOps*>(tag)->dispose_and_destroy(cb);
      }
      return;
    }
#endif
  }
  sp_release_slow(pi);
}

// ============================================================================
// Control-Block Storage Specializations (Default/Custom Deleter & Allocators,
// Scalar/Bounded-Array Inplace, and Single-Allocation Unbounded-Array Inplace)
// ============================================================================
template <typename T>
struct SpInplaceStorage final {
  ThetaSpCountedBase base;
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

  // Bounded-array value fill (`make_shared<U[N]>(const U& val)`)
  template <typename Elem>
    requires(std::is_bounded_array_v<T> &&
             std::is_same_v<Elem, std::remove_extent_t<T>>)
  SpInplaceStorage(ForOverwriteTag, const Elem& fill_val)
      : base(ops_or_size_value()) {
    constexpr std::size_t kCount = std::extent_v<T>;
    auto* arr = reinterpret_cast<Elem*>(storage);
    std::size_t i = 0U;
    try {
      for (; i < kCount; ++i) {
        ::new (static_cast<void*>(arr + i)) Elem(fill_val);
      }
    } catch (...) {
      while (i > 0U) {
        --i;
        arr[i].~Elem();
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
    if constexpr (std::is_bounded_array_v<T>) {
      using Elem = std::remove_extent_t<T>;
      if constexpr (!std::is_trivially_destructible_v<Elem>) {
        constexpr std::size_t kCount = std::extent_v<T>;
        auto* arr = std::launder(reinterpret_cast<Elem*>(self->storage));
        for (std::size_t i = kCount; i > 0U; --i) {
          arr[i - 1U].~Elem();
        }
      }
    } else if constexpr (!std::is_trivially_destructible_v<T>) {
      self->ptr()->~T();
    }
  }

  static void dispose_impl(ThetaSpCountedBase* b) noexcept {
    dispose_object(reinterpret_cast<SpInplaceStorage*>(b));
  }

  static void destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpInplaceStorage*>(b);
    self->~SpInplaceStorage();
    free_raw(static_cast<void*>(self));
  }

  static void dispose_and_destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpInplaceStorage*>(b);
    dispose_object(self);
    self->~SpInplaceStorage();
    free_raw(static_cast<void*>(self));
  }

  static inline constexpr SpColdOps kOps{
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

template <typename T, typename Alloc>
struct SpAllocInplaceStorage final {
  using ValueAlloc = typename std::allocator_traits<
      Alloc>::template rebind_alloc<std::remove_cv_t<std::remove_extent_t<T>>>;
  ThetaSpCountedBase base;
  [[no_unique_address]] ValueAlloc alloc;
  alignas(T) unsigned char storage[sizeof(T)];

  using SelfAlloc = typename std::allocator_traits<
      ValueAlloc>::template rebind_alloc<SpAllocInplaceStorage>;

  SpAllocInplaceStorage(const Alloc& a, ForOverwriteTag)
      : base(reinterpret_cast<std::uintptr_t>(&kOps)), alloc(a) {
    ::new (static_cast<void*>(storage)) T;
  }

  template <typename... Args>
    requires(!(sizeof...(Args) == 1 &&
               (std::is_same_v<std::remove_cvref_t<Args>, ForOverwriteTag> &&
                ...)))
  explicit SpAllocInplaceStorage(const Alloc& a, Args&&... args)
      : base(reinterpret_cast<std::uintptr_t>(&kOps)), alloc(a) {
    if constexpr (std::is_bounded_array_v<T>) {
      ::new (static_cast<void*>(storage)) T();
    } else {
      std::allocator_traits<ValueAlloc>::construct(
          alloc, reinterpret_cast<T*>(storage), std::forward<Args>(args)...);
    }
  }

  template <typename Elem>
    requires(std::is_bounded_array_v<T> &&
             std::is_same_v<Elem, std::remove_extent_t<T>>)
  SpAllocInplaceStorage(const Alloc& a, ForOverwriteTag, const Elem& fill_val)
      : base(reinterpret_cast<std::uintptr_t>(&kOps)), alloc(a) {
    constexpr std::size_t kCount = std::extent_v<T>;
    auto* arr = reinterpret_cast<Elem*>(storage);
    std::size_t i = 0U;
    try {
      for (; i < kCount; ++i) {
        std::allocator_traits<ValueAlloc>::construct(alloc, arr + i, fill_val);
      }
    } catch (...) {
      while (i > 0U) {
        --i;
        std::allocator_traits<ValueAlloc>::destroy(alloc, arr + i);
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
    if constexpr (std::is_bounded_array_v<T>) {
      using Elem = std::remove_extent_t<T>;
      constexpr std::size_t kCount = std::extent_v<T>;
      auto* arr = std::launder(reinterpret_cast<Elem*>(self->storage));
      for (std::size_t i = kCount; i > 0U; --i) {
        std::allocator_traits<ValueAlloc>::destroy(self->alloc, arr + i - 1U);
      }
    } else {
      std::allocator_traits<ValueAlloc>::destroy(self->alloc, self->ptr());
    }
  }

  static void dispose_impl(ThetaSpCountedBase* b) noexcept {
    dispose_object(reinterpret_cast<SpAllocInplaceStorage*>(b));
  }

  static void destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocInplaceStorage*>(b);
    SelfAlloc sa(self->alloc);
    self->~SpAllocInplaceStorage();
    std::allocator_traits<SelfAlloc>::deallocate(sa, self, 1U);
  }

  static void dispose_and_destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocInplaceStorage*>(b);
    dispose_object(self);
    SelfAlloc sa(self->alloc);
    self->~SpAllocInplaceStorage();
    std::allocator_traits<SelfAlloc>::deallocate(sa, self, 1U);
  }

  static inline constexpr SpColdOps kOps{
      &dispose_impl,
      &destroy_impl,
      &dispose_and_destroy_impl,
      nullptr,
  };
};

// Single-allocation unbounded array control block (`make_shared<U[]>(n, ...)`
// and `allocate_shared<U[]>(alloc, n, ...)`).
template <typename Elem, typename Alloc>
struct SpUnboundedArrayStorage final {
  using ValueAlloc =
      typename std::allocator_traits<Alloc>::template rebind_alloc<Elem>;

  static constexpr std::size_t kChunkAlign = alignof(Elem) >
                                                     alignof(std::max_align_t)
                                                 ? alignof(Elem)
                                                 : alignof(std::max_align_t);

  struct alignas(kChunkAlign) RawChunk {
    unsigned char data[kChunkAlign];
  };

  using ChunkAlloc =
      typename std::allocator_traits<Alloc>::template rebind_alloc<RawChunk>;

  ThetaSpCountedBase base;
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
        std::allocator_traits<ValueAlloc>::destroy(self->alloc, arr + i);
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
    for (std::size_t i = self->count; i > 0U; --i) {
      std::allocator_traits<ValueAlloc>::destroy(self->alloc, arr + i - 1U);
    }
  }

  static void dispose_impl(ThetaSpCountedBase* b) noexcept {
    dispose_elements(reinterpret_cast<SpUnboundedArrayStorage*>(b));
  }

  static void destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpUnboundedArrayStorage*>(b);
    ChunkAlloc ca(self->alloc);
    const std::size_t nchunks = self->total_chunks;
    auto* mem = reinterpret_cast<RawChunk*>(self);
    self->~SpUnboundedArrayStorage();
    std::allocator_traits<ChunkAlloc>::deallocate(ca, mem, nchunks);
  }

  static void dispose_and_destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpUnboundedArrayStorage*>(b);
    dispose_elements(self);
    ChunkAlloc ca(self->alloc);
    const std::size_t nchunks = self->total_chunks;
    auto* mem = reinterpret_cast<RawChunk*>(self);
    self->~SpUnboundedArrayStorage();
    std::allocator_traits<ChunkAlloc>::deallocate(ca, mem, nchunks);
  }

  static inline constexpr SpColdOps kOps{
      &dispose_impl,
      &destroy_impl,
      &dispose_and_destroy_impl,
      nullptr,
  };
};

template <typename Ptr, typename Deleter>
struct SpDeleterStorage final {
  ThetaSpCountedBase base;
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

  static void dispose_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpDeleterStorage*>(b);
    self->deleter(self->ptr);
  }

  static void destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpDeleterStorage*>(b);
    self->~SpDeleterStorage();
    free_raw(static_cast<void*>(self));
  }

  static void dispose_and_destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpDeleterStorage*>(b);
    self->deleter(self->ptr);
    self->~SpDeleterStorage();
    free_raw(static_cast<void*>(self));
  }

  static void* get_deleter_impl(ThetaSpCountedBase* b,
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

  static inline constexpr SpColdOps kOps{
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

  ThetaSpCountedBase base;
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

  static void dispose_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocDeleterStorage*>(b);
    self->deleter(self->ptr);
  }

  static void destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocDeleterStorage*>(b);
    SelfAlloc sa(self->alloc);
    self->~SpAllocDeleterStorage();
    std::allocator_traits<SelfAlloc>::deallocate(sa, self, 1U);
  }

  static void dispose_and_destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpAllocDeleterStorage*>(b);
    self->deleter(self->ptr);
    SelfAlloc sa(self->alloc);
    self->~SpAllocDeleterStorage();
    std::allocator_traits<SelfAlloc>::deallocate(sa, self, 1U);
  }

  static void* get_deleter_impl(ThetaSpCountedBase* b,
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

  static inline constexpr SpColdOps kOps{
      &dispose_impl,
      &destroy_impl,
      &dispose_and_destroy_impl,
      &get_deleter_impl,
  };
};

struct SpEsftAccess {
  template <typename Base>
  static EnableSharedFromThis<Base>* detect_theta(
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
  static void raw_parts(const PtrT& p, void*& out_ptr,
                        StdSpCountedBase*& out_pi) noexcept {
    out_ptr = const_cast<void*>(static_cast<const volatile void*>(p.ptr_));
    out_pi = p.pi_;
  }
};

}  // namespace detail::shared_ptr

// ============================================================================
// `theta::SharedPtr<T>`: 16-Byte `std::shared_ptr<T>` Binary-ABI-Compatible
// Smart Pointer Powered by `rseq` 8,192-CPU Striped Per-CPU Reference Counting
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

  // Aliasing constructors
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

  // Adoption from `std::unique_ptr<Y, Deleter>&&`
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
  // Zero-allocation binary ABI interop from `std::shared_ptr<Y>`
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
  // Zero-allocation conversion to genuine `std::shared_ptr<T>` sharing the
  // exact same `_Sp_counted_base` control block (`owner_before` & `weak_ptr`
  // preserved across unmodified precompiled DSOs).
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

#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
static_assert(sizeof(SharedPtr<int>) == sizeof(std::shared_ptr<int>));
static_assert(alignof(SharedPtr<int>) == alignof(std::shared_ptr<int>));
#endif

// ============================================================================
// Factory Functions (`make_shared`, `make_shared_for_overwrite`,
// `allocate_shared`, `allocate_shared_for_overwrite`)
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
  using CleanElem = std::remove_cv_t<std::remove_extent_t<T>>;
  using Alloc = std::allocator<CleanElem>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
  auto* s = Storage::create(Alloc{}, n, [](Alloc& a, CleanElem* p) {
    std::allocator_traits<Alloc>::construct(a, p);
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

template <typename T>
  requires std::is_unbounded_array_v<T>
[[nodiscard]] SharedPtr<T> make_shared(std::size_t n,
                                       const std::remove_extent_t<T>& u) {
  using CleanElem = std::remove_cv_t<std::remove_extent_t<T>>;
  using Alloc = std::allocator<CleanElem>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
  auto* s = Storage::create(Alloc{}, n, [&u](Alloc& a, CleanElem* p) {
    std::allocator_traits<Alloc>::construct(a, p, u);
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

template <typename T>
  requires std::is_bounded_array_v<T>
[[nodiscard]] SharedPtr<T> make_shared() {
  using CleanT = std::remove_cv_t<T>;
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
  using CleanElem = std::remove_cv_t<std::remove_extent_t<T>>;
  using CleanT = CleanElem[std::extent_v<T>];
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
  using CleanElem = std::remove_cv_t<std::remove_extent_t<T>>;
  using Alloc = std::allocator<CleanElem>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
  auto* s = Storage::create(Alloc{}, n, [](Alloc&, CleanElem* p) {
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
  using CleanElem = std::remove_cv_t<std::remove_extent_t<T>>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
  using ValueAlloc = typename Storage::ValueAlloc;
  auto* s = Storage::create(alloc, n, [](ValueAlloc& va, CleanElem* p) {
    std::allocator_traits<ValueAlloc>::construct(va, p);
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

template <typename T, typename Alloc>
  requires std::is_unbounded_array_v<T>
[[nodiscard]] SharedPtr<T> allocate_shared(const Alloc& alloc, std::size_t n,
                                           const std::remove_extent_t<T>& u) {
  using CleanElem = std::remove_cv_t<std::remove_extent_t<T>>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
  using ValueAlloc = typename Storage::ValueAlloc;
  auto* s = Storage::create(alloc, n, [&u](ValueAlloc& va, CleanElem* p) {
    std::allocator_traits<ValueAlloc>::construct(va, p, u);
  });
  return SharedPtr<T>(s->elems_ptr(), &s->base,
                      detail::shared_ptr::AdoptRefTag{});
}

template <typename T, typename Alloc>
  requires std::is_bounded_array_v<T>
[[nodiscard]] SharedPtr<T> allocate_shared(const Alloc& alloc) {
  using CleanElem = std::remove_cv_t<std::remove_extent_t<T>>;
  using CleanT = CleanElem[std::extent_v<T>];
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
  using CleanElem = std::remove_cv_t<std::remove_extent_t<T>>;
  using CleanT = CleanElem[std::extent_v<T>];
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
  using Storage = detail::shared_ptr::SpAllocInplaceStorage<CleanT, Alloc>;
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
  using CleanElem = std::remove_cv_t<std::remove_extent_t<T>>;
  using Storage = detail::shared_ptr::SpUnboundedArrayStorage<CleanElem, Alloc>;
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
// `theta::WeakPtr<T>`: 16-Byte `std::weak_ptr<T>` ABI-Compatible Weak Pointer
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
      if (detail::shared_ptr::is_theta_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::ThetaSpCountedBase*>(pi_);
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
      if (detail::shared_ptr::is_theta_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::ThetaSpCountedBase*>(pi_);
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
      if (detail::shared_ptr::is_theta_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::ThetaSpCountedBase*>(pi_);
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
    if (pi_ != nullptr && detail::shared_ptr::is_theta_sp(pi_)) {
      auto* cb = static_cast<detail::shared_ptr::ThetaSpCountedBase*>(pi_);
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
      if (detail::shared_ptr::is_theta_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::ThetaSpCountedBase*>(pi_);
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
      if (detail::shared_ptr::is_theta_sp(pi_)) {
        auto* cb = static_cast<detail::shared_ptr::ThetaSpCountedBase*>(pi_);
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
// `theta::OwnerLess<T>` (`theta::owner_less<T>`)
// ============================================================================
template <typename T>
struct OwnerLess;

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

// ============================================================================
// `theta::EnableSharedFromThis<T>` (`theta::enable_shared_from_this<T>`)
// ============================================================================
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
  if (is_theta_sp(owner.pi_)) {
    auto* cb = static_cast<ThetaSpCountedBase*>(owner.pi_);
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
    if constexpr (requires(RawClean* p) { detect_theta(p); }) {
      auto* base = detect_theta(const_cast<RawClean*>(raw_ptr));
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
// `theta::KeepAlive<T>`: Wait-Free `rseq` Concurrent Snapshot Cell (`~3.6 ns`
// `get()` Across 8,192 CPUs, Zero Reader `shared_mutex` Contention)
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
    detail::shared_ptr::ThetaSpCountedBase* new_cb = nullptr;
    if (next.pi_ != nullptr) {
      void* raw_managed =
          const_cast<void*>(static_cast<const volatile void*>(next.get()));
      if (!detail::shared_ptr::is_theta_sp(next.pi_)) {
#if THETA_SHARED_PTR_HAS_GLIBCXX_ABI
        std::shared_ptr<T> ext = next.to_std();
        element_type* raw_p = ext.get();
        next = SharedPtr<T>(
            raw_p,
            [hold = std::move(ext)](element_type*) mutable { hold.reset(); });
#else
        SharedPtr<T> ext = std::move(next);
        element_type* raw_p = ext.get();
        next = SharedPtr<T>(
            raw_p,
            [hold = std::move(ext)](element_type*) mutable { hold.reset(); });
#endif
      }
      new_cb = static_cast<detail::shared_ptr::ThetaSpCountedBase*>(next.pi_);
      if (!detail::shared_ptr::sp_pin_for_keepalive(new_cb, raw_managed)) {
        SharedPtr<T> hold_inner = std::move(next);
        element_type* raw_p = hold_inner.get();
        next =
            SharedPtr<T>(raw_p, [hold = std::move(hold_inner)](
                                    element_type*) mutable { hold.reset(); });
        new_cb = static_cast<detail::shared_ptr::ThetaSpCountedBase*>(next.pi_);
        (void)detail::shared_ptr::sp_pin_for_keepalive(new_cb, raw_managed);
      }
    }

    SharedPtr<T> ret;
    SharedPtr<T> old_anchor;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      auto* old_cb = __atomic_exchange_n(&active_cb_, new_cb, __ATOMIC_ACQ_REL);
      if (old_cb != nullptr) {
        detail::shared_ptr::sp_keepalive_retire_old(old_cb);
      }
      old_anchor = std::move(anchor_);
      anchor_ = std::move(next);
      ret = anchor_;
    }
    return ret;
  }

  void store(SharedPtr<T> desired) { (void)reset(std::move(desired)); }

  [[nodiscard]] SharedPtr<T> get() const noexcept {
    detail::shared_ptr::ThetaSpCountedBase* cb = nullptr;
    std::uint32_t slot = 0U;
    if (detail::shared_ptr::ArchBackend::try_rseq_keepalive_acquire(
            &active_cb_, cb, slot)) [[likely]] {
      if (cb == nullptr) {
        return SharedPtr<T>{};
      }
      return SharedPtr<T>(static_cast<element_type*>(
                              detail::shared_ptr::g_rseq_keepalive_ptrs[slot]),
                          cb, detail::shared_ptr::AdoptRefTag{});
    }
    return get_slow();
  }

  [[nodiscard]] SharedPtr<T> load() const noexcept { return get(); }

 private:
  [[gnu::noinline]] SharedPtr<T> get_slow() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!anchor_) {
      return SharedPtr<T>{};
    }
    if (detail::shared_ptr::is_theta_sp(anchor_.pi_)) {
      auto* cb =
          static_cast<detail::shared_ptr::ThetaSpCountedBase*>(anchor_.pi_);
      detail::shared_ptr::sp_register_keepalive_cpu_slow(cb);
    }
    return anchor_;
  }

  mutable detail::shared_ptr::ThetaSpCountedBase* active_cb_{nullptr};
  mutable std::shared_mutex mutex_;
  SharedPtr<T> anchor_;
};

}  // namespace theta

// ============================================================================
// Standard Library Specializations: `std::hash`, `std::atomic<SharedPtr<T>>`,
// and `std::atomic<WeakPtr<T>>`
// ============================================================================
namespace std {

template <typename T>
struct hash<::theta::SharedPtr<T>> {
  [[nodiscard]] std::size_t operator()(
      const ::theta::SharedPtr<T>& p) const noexcept {
    return std::hash<typename ::theta::SharedPtr<T>::element_type*>{}(p.get());
  }
};

template <typename T>
struct atomic<::theta::SharedPtr<T>> {
  using value_type = ::theta::SharedPtr<T>;
  static constexpr bool is_always_lock_free = false;

  [[nodiscard]] bool is_lock_free() const noexcept { return false; }

  constexpr atomic() noexcept = default;
  constexpr atomic(std::nullptr_t) noexcept : val_{} {}
  atomic(value_type desired) noexcept : val_(std::move(desired)) {}

  atomic(const atomic&) = delete;
  void operator=(const atomic&) = delete;

  void operator=(value_type desired) noexcept { store(std::move(desired)); }
  void operator=(std::nullptr_t) noexcept { store(nullptr); }

  [[nodiscard]] value_type load(
      std::memory_order /*order*/ = std::memory_order_seq_cst) const noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    value_type out = val_;
    ::theta::detail::shared_ptr::atomic_ptr_unlock(this);
    return out;
  }

  operator value_type() const noexcept { return load(); }

  void store(value_type desired,
             std::memory_order /*order*/ = std::memory_order_seq_cst) noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    val_.swap(desired);
    ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, true);
  }

  value_type exchange(
      value_type desired,
      std::memory_order /*order*/ = std::memory_order_seq_cst) noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    val_.swap(desired);
    ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, true);
    return desired;
  }

  bool compare_exchange_strong(value_type& expected, value_type desired,
                               std::memory_order /*success*/,
                               std::memory_order /*failure*/) noexcept {
    value_type old_to_drop;
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    if (same_ptr_and_owner(val_, expected)) {
      old_to_drop = std::move(val_);
      val_ = std::move(desired);
      ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, true);
      return true;
    }
    value_type current_copy = val_;
    ::theta::detail::shared_ptr::atomic_ptr_unlock(this);
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
      ::theta::detail::shared_ptr::atomic_ptr_lock(this);
      const std::uint32_t seq =
          ::theta::detail::shared_ptr::atomic_ptr_seq(this);
      const bool matches = same_ptr_and_owner(val_, old);
      ::theta::detail::shared_ptr::atomic_ptr_unlock(this);
      if (!matches) {
        return;
      }
      ::theta::detail::shared_ptr::atomic_ptr_wait_seq(this, seq);
    }
  }

  void notify_one() noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, false);
  }

  void notify_all() noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, true);
  }

 private:
  static bool same_ptr_and_owner(const value_type& a,
                                 const value_type& b) noexcept {
    void* ap = nullptr;
    void* bp = nullptr;
    ::theta::detail::shared_ptr::StdSpCountedBase* api = nullptr;
    ::theta::detail::shared_ptr::StdSpCountedBase* bpi = nullptr;
    ::theta::detail::shared_ptr::SpAtomicAccess::raw_parts(a, ap, api);
    ::theta::detail::shared_ptr::SpAtomicAccess::raw_parts(b, bp, bpi);
    return ap == bp && api == bpi;
  }

  mutable value_type val_{};
};

template <typename T>
struct atomic<::theta::WeakPtr<T>> {
  using value_type = ::theta::WeakPtr<T>;
  static constexpr bool is_always_lock_free = false;

  [[nodiscard]] bool is_lock_free() const noexcept { return false; }

  constexpr atomic() noexcept = default;
  atomic(value_type desired) noexcept : val_(std::move(desired)) {}

  atomic(const atomic&) = delete;
  void operator=(const atomic&) = delete;

  void operator=(value_type desired) noexcept { store(std::move(desired)); }

  [[nodiscard]] value_type load(
      std::memory_order /*order*/ = std::memory_order_seq_cst) const noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    value_type out = val_;
    ::theta::detail::shared_ptr::atomic_ptr_unlock(this);
    return out;
  }

  operator value_type() const noexcept { return load(); }

  void store(value_type desired,
             std::memory_order /*order*/ = std::memory_order_seq_cst) noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    val_.swap(desired);
    ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, true);
  }

  value_type exchange(
      value_type desired,
      std::memory_order /*order*/ = std::memory_order_seq_cst) noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    val_.swap(desired);
    ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, true);
    return desired;
  }

  bool compare_exchange_strong(value_type& expected, value_type desired,
                               std::memory_order /*success*/,
                               std::memory_order /*failure*/) noexcept {
    value_type old_to_drop;
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    if (same_ptr_and_owner(val_, expected)) {
      old_to_drop = std::move(val_);
      val_ = std::move(desired);
      ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, true);
      return true;
    }
    value_type current_copy = val_;
    ::theta::detail::shared_ptr::atomic_ptr_unlock(this);
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
      ::theta::detail::shared_ptr::atomic_ptr_lock(this);
      const std::uint32_t seq =
          ::theta::detail::shared_ptr::atomic_ptr_seq(this);
      const bool matches = same_ptr_and_owner(val_, old);
      ::theta::detail::shared_ptr::atomic_ptr_unlock(this);
      if (!matches) {
        return;
      }
      ::theta::detail::shared_ptr::atomic_ptr_wait_seq(this, seq);
    }
  }

  void notify_one() noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, false);
  }

  void notify_all() noexcept {
    ::theta::detail::shared_ptr::atomic_ptr_lock(this);
    ::theta::detail::shared_ptr::atomic_ptr_unlock_and_notify(this, true);
  }

 private:
  static bool same_ptr_and_owner(const value_type& a,
                                 const value_type& b) noexcept {
    void* ap = nullptr;
    void* bp = nullptr;
    ::theta::detail::shared_ptr::StdSpCountedBase* api = nullptr;
    ::theta::detail::shared_ptr::StdSpCountedBase* bpi = nullptr;
    ::theta::detail::shared_ptr::SpAtomicAccess::raw_parts(a, ap, api);
    ::theta::detail::shared_ptr::SpAtomicAccess::raw_parts(b, bp, bpi);
    return ap == bp && api == bpi;
  }

  mutable value_type val_{};
};

}  // namespace std
