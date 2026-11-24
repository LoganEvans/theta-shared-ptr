// FILENAME: include/theta/hyper_shared_ptr.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <theta/arch.hpp>
#include <theta/hyper_shared_ptr/detail/common.hpp>
#include <type_traits>
#include <typeinfo>
#include <utility>

#if THETA_ARCH_X86_64 && THETA_HSP_USE_RSEQ_ASM
#include <theta/hyper_shared_ptr/detail/backend_x86_64.hpp>
#elif THETA_ARCH_AARCH64 && THETA_HSP_USE_RSEQ_ASM
#include <theta/hyper_shared_ptr/detail/backend_aarch64.hpp>
#else
#include <theta/hyper_shared_ptr/detail/backend_fallback.hpp>
#endif

namespace theta {

template <typename T>
class SharedPtr;

template <typename T>
class WeakPtr;

template <typename T>
class KeepAlive;

template <typename T>
class HyperSharedPtr;

template <typename T>
class HyperWeakPtr;

template <typename T>
class HyperKeepAlive;

namespace hyper_shared_ptr::detail {

struct RawStdPtrLayout {
  void* ptr;
  StdSpCountedBase* pi;
};

#if THETA_HSP_HAS_GLIBCXX_ABI
static_assert(sizeof(std::shared_ptr<int>) == sizeof(RawStdPtrLayout),
              "std::shared_ptr<T> must be 16 bytes (ptr + _Sp_counted_base*).");
static_assert(alignof(std::shared_ptr<int>) == alignof(RawStdPtrLayout),
              "std::shared_ptr<T> alignment must match RawStdPtrLayout.");
static_assert(sizeof(std::weak_ptr<int>) == sizeof(RawStdPtrLayout),
              "std::weak_ptr<T> must be 16 bytes (ptr + _Sp_counted_base*).");
#endif

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
    const std::uint8_t mode =
        __atomic_load_n(&cb->fast_.mode, __ATOMIC_ACQUIRE);
    if (mode == kModeAtomicRetired) {
      pi->_M_release();
      return;
    }
    if (mode == kModeUnpromoted &&
        __atomic_load_n(&cb->has_weak_or_std_, __ATOMIC_ACQUIRE) == 0U &&
        pi->_M_get_use_count() == 1L) [[likely]] {
      pi->_M_release();
      return;
    }
    sp_release_slow(pi);
    return;
  }
  pi->_M_release();
}

template <typename T>
struct SpInplaceStorage final {
  ThetaSpCountedBase base;
  alignas(T) unsigned char storage[sizeof(T)];

  template <typename... Args>
  explicit SpInplaceStorage(Args&&... args)
      : base(static_cast<void*>(storage), &dispose_impl, &destroy_impl,
             nullptr) {
    ::new (static_cast<void*>(storage)) T(std::forward<Args>(args)...);
  }

  T* ptr() noexcept { return std::launder(reinterpret_cast<T*>(storage)); }

  static void dispose_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpInplaceStorage*>(b);
    self->ptr()->~T();
  }

  static void destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpInplaceStorage*>(b);
    self->~SpInplaceStorage();
    ::operator delete(static_cast<void*>(self),
                      std::align_val_t{alignof(SpInplaceStorage)});
  }
};

template <typename Ptr, typename Deleter>
struct SpDeleterStorage final {
  ThetaSpCountedBase base;
  Ptr ptr;
  [[no_unique_address]] Deleter deleter;

  SpDeleterStorage(Ptr p, Deleter d) noexcept(
      std::is_nothrow_move_constructible_v<Deleter>)
      : base(const_cast<void*>(static_cast<const volatile void*>(p)),
             &dispose_impl, &destroy_impl, &get_deleter_impl),
        ptr(p),
        deleter(std::move(d)) {}

  static void dispose_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpDeleterStorage*>(b);
    self->deleter(self->ptr);
  }

  static void destroy_impl(ThetaSpCountedBase* b) noexcept {
    auto* self = reinterpret_cast<SpDeleterStorage*>(b);
    self->~SpDeleterStorage();
    ::operator delete(static_cast<void*>(self),
                      std::align_val_t{alignof(SpDeleterStorage)});
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
};

template <typename T>
struct HspInplaceStorage final {
  HyperControlBlock base;
  alignas(T) unsigned char storage[sizeof(T)];

  template <typename... Args>
  explicit HspInplaceStorage(std::uint32_t slot_id, Args&&... args)
      : base(static_cast<void*>(storage), slot_id, &dispose_impl,
             &destroy_impl) {
    ::new (static_cast<void*>(storage)) T(std::forward<Args>(args)...);
  }

  T* ptr() noexcept { return std::launder(reinterpret_cast<T*>(storage)); }

  static void dispose_impl(HyperControlBlock* b) noexcept {
    auto* self = reinterpret_cast<HspInplaceStorage*>(b);
    self->ptr()->~T();
  }

  static void destroy_impl(HyperControlBlock* b) noexcept {
    auto* self = reinterpret_cast<HspInplaceStorage*>(b);
    self->~HspInplaceStorage();
    ::operator delete(static_cast<void*>(self),
                      std::align_val_t{alignof(HspInplaceStorage)});
  }
};

template <typename Ptr, typename Deleter>
struct HspDeleterStorage final {
  HyperControlBlock base;
  Ptr ptr;
  [[no_unique_address]] Deleter deleter;

  HspDeleterStorage(std::uint32_t slot_id, Ptr p, Deleter d) noexcept(
      std::is_nothrow_move_constructible_v<Deleter>)
      : base(const_cast<void*>(static_cast<const volatile void*>(p)), slot_id,
             &dispose_impl, &destroy_impl),
        ptr(p),
        deleter(std::move(d)) {}

  static void dispose_impl(HyperControlBlock* b) noexcept {
    auto* self = reinterpret_cast<HspDeleterStorage*>(b);
    self->deleter(self->ptr);
  }

  static void destroy_impl(HyperControlBlock* b) noexcept {
    auto* self = reinterpret_cast<HspDeleterStorage*>(b);
    self->~HspDeleterStorage();
    ::operator delete(static_cast<void*>(self),
                      std::align_val_t{alignof(HspDeleterStorage)});
  }
};

struct AdoptRefTag {};

}  // namespace hyper_shared_ptr::detail

// ============================================================================
// `theta::SharedPtr<T>`: 16-Byte `std::shared_ptr<T>` Binary-ABI-Compatible
// Smart Pointer Powered by `rseq` 8,192-CPU Striped Per-CPU Reference Counting
// ============================================================================
template <typename T>
class SharedPtr {
  static_assert(
      !std::is_array_v<T>,
      "theta::SharedPtr<T> currently manages non-array object types.");

  template <typename U>
  friend class SharedPtr;
  template <typename U>
  friend class WeakPtr;
  template <typename U>
  friend class KeepAlive;

 public:
  using element_type = T;
  using weak_type = WeakPtr<T>;

  constexpr SharedPtr() noexcept = default;
  constexpr SharedPtr(std::nullptr_t) noexcept {}

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  explicit SharedPtr(Y* ptr) : SharedPtr(ptr, std::default_delete<Y>{}) {}

  template <typename Y, typename Deleter>
    requires std::is_convertible_v<Y*, T*>
  SharedPtr(Y* ptr, Deleter deleter) {
    if (ptr == nullptr) {
      deleter(ptr);
      return;
    }
    using Storage = hyper_shared_ptr::detail::SpDeleterStorage<Y*, Deleter>;
    void* mem =
        ::operator new(sizeof(Storage), std::align_val_t{alignof(Storage)});
    try {
      auto* s = ::new (mem) Storage(ptr, std::move(deleter));
      ptr_ = ptr;
      pi_ = &s->base;
    } catch (...) {
      ::operator delete(mem, std::align_val_t{alignof(Storage)});
      deleter(ptr);
      throw;
    }
  }

  // Aliasing constructor
  template <typename Y>
  SharedPtr(const SharedPtr<Y>& other, element_type* alias_ptr) noexcept
      : ptr_(alias_ptr), pi_(other.pi_) {
    hyper_shared_ptr::detail::sp_add_ref(pi_);
  }

  template <typename Y>
  SharedPtr(SharedPtr<Y>&& other, element_type* alias_ptr) noexcept
      : ptr_(alias_ptr), pi_(other.pi_) {
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
  }

  SharedPtr(const SharedPtr& other) noexcept
      : ptr_(other.ptr_), pi_(other.pi_) {
    hyper_shared_ptr::detail::sp_add_ref(pi_);
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  SharedPtr(const SharedPtr<Y>& other) noexcept
      : ptr_(other.ptr_), pi_(other.pi_) {
    hyper_shared_ptr::detail::sp_add_ref(pi_);
  }

  SharedPtr(SharedPtr&& other) noexcept : ptr_(other.ptr_), pi_(other.pi_) {
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  SharedPtr(SharedPtr<Y>&& other) noexcept : ptr_(other.ptr_), pi_(other.pi_) {
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
  }

#if THETA_HSP_HAS_GLIBCXX_ABI
  // Zero-allocation binary ABI interop from `std::shared_ptr<Y>`
  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  SharedPtr(const std::shared_ptr<Y>& sp) noexcept {
    hyper_shared_ptr::detail::RawStdPtrLayout raw{};
    std::memcpy(&raw, static_cast<const void*>(&sp), sizeof(raw));
    ptr_ = sp.get();
    pi_ = raw.pi;
    if (pi_ != nullptr) {
      hyper_shared_ptr::detail::sp_add_ref_from_std(pi_);
    }
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  SharedPtr(std::shared_ptr<Y>&& sp) noexcept {
    hyper_shared_ptr::detail::RawStdPtrLayout raw{};
    std::memcpy(&raw, static_cast<const void*>(&sp), sizeof(raw));
    ptr_ = sp.get();
    pi_ = raw.pi;
    const hyper_shared_ptr::detail::RawStdPtrLayout zero{nullptr, nullptr};
    std::memcpy(static_cast<void*>(&sp), &zero, sizeof(zero));
    if (pi_ != nullptr) {
      hyper_shared_ptr::detail::sp_adopt_from_std_move(pi_);
    }
  }
#endif

  ~SharedPtr() noexcept { hyper_shared_ptr::detail::sp_release(pi_); }

  SharedPtr& operator=(const SharedPtr& other) noexcept {
    if (this == &other) {
      return *this;
    }
    SharedPtr tmp(other);
    swap(tmp);
    return *this;
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  SharedPtr& operator=(const SharedPtr<Y>& other) noexcept {
    SharedPtr tmp(other);
    swap(tmp);
    return *this;
  }

  SharedPtr& operator=(SharedPtr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    hyper_shared_ptr::detail::sp_release(pi_);
    ptr_ = other.ptr_;
    pi_ = other.pi_;
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
    return *this;
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  SharedPtr& operator=(SharedPtr<Y>&& other) noexcept {
    hyper_shared_ptr::detail::sp_release(pi_);
    ptr_ = other.ptr_;
    pi_ = other.pi_;
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
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
    requires std::is_convertible_v<Y*, T*>
  void reset(Y* ptr) {
    SharedPtr tmp(ptr);
    swap(tmp);
  }

  template <typename Y, typename Deleter>
    requires std::is_convertible_v<Y*, T*>
  void reset(Y* ptr, Deleter deleter) {
    SharedPtr tmp(ptr, std::move(deleter));
    swap(tmp);
  }

  void swap(SharedPtr& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(pi_, other.pi_);
  }

  [[nodiscard]] element_type* get() const noexcept { return ptr_; }

  [[nodiscard]] std::add_lvalue_reference_t<T> operator*() const noexcept
    requires(!std::is_void_v<T>)
  {
    return *ptr_;
  }

  [[nodiscard]] element_type* operator->() const noexcept { return ptr_; }

  [[nodiscard]] explicit operator bool() const noexcept {
    return ptr_ != nullptr;
  }

  [[nodiscard]] long use_count() const noexcept {
    return hyper_shared_ptr::detail::sp_use_count_slow(pi_);
  }

  template <typename U>
  [[nodiscard]] bool owner_before(const SharedPtr<U>& other) const noexcept {
    return std::less<hyper_shared_ptr::detail::StdSpCountedBase*>{}(pi_,
                                                                    other.pi_);
  }

  template <typename U>
  [[nodiscard]] bool owner_before(const WeakPtr<U>& other) const noexcept;

#if THETA_HSP_HAS_GLIBCXX_ABI
  // Zero-allocation conversion to genuine `std::shared_ptr<T>` sharing the
  // exact same `_Sp_counted_base` control block (`owner_before` & `weak_ptr`
  // preserved across unmodified precompiled DSOs).
  [[nodiscard]] std::shared_ptr<T> to_std() const& noexcept {
    std::shared_ptr<T> out;
    if (pi_ != nullptr) {
      hyper_shared_ptr::detail::sp_note_to_std_copy(pi_);
      const hyper_shared_ptr::detail::RawStdPtrLayout raw{
          const_cast<void*>(static_cast<const volatile void*>(ptr_)), pi_};
      std::memcpy(static_cast<void*>(&out), &raw, sizeof(out));
    }
    return out;
  }

  [[nodiscard]] std::shared_ptr<T> to_std() && noexcept {
    std::shared_ptr<T> out;
    if (pi_ != nullptr) {
      hyper_shared_ptr::detail::sp_note_to_std_move(pi_);
      const hyper_shared_ptr::detail::RawStdPtrLayout raw{
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
  SharedPtr(element_type* ptr, hyper_shared_ptr::detail::StdSpCountedBase* pi,
            hyper_shared_ptr::detail::AdoptRefTag) noexcept
      : ptr_(ptr), pi_(pi) {}

  template <typename U, typename... Args>
  friend SharedPtr<U> make_shared(Args&&... args);

  template <typename Deleter, typename U>
  friend Deleter* get_deleter(const SharedPtr<U>& p) noexcept;

  element_type* ptr_{nullptr};
  hyper_shared_ptr::detail::StdSpCountedBase* pi_{nullptr};
};

#if THETA_HSP_HAS_GLIBCXX_ABI
static_assert(sizeof(SharedPtr<int>) == sizeof(std::shared_ptr<int>));
static_assert(alignof(SharedPtr<int>) == alignof(std::shared_ptr<int>));
#endif

template <typename T>
using AbiSharedPtr = SharedPtr<T>;

template <typename T, typename... Args>
[[nodiscard]] SharedPtr<T> make_shared(Args&&... args) {
  using CleanT = std::remove_cv_t<T>;
  using Storage = hyper_shared_ptr::detail::SpInplaceStorage<CleanT>;
  void* mem =
      ::operator new(sizeof(Storage), std::align_val_t{alignof(Storage)});
  try {
    auto* s = ::new (mem) Storage(std::forward<Args>(args)...);
    return SharedPtr<T>(s->ptr(), &s->base,
                        hyper_shared_ptr::detail::AdoptRefTag{});
  } catch (...) {
    ::operator delete(mem, std::align_val_t{alignof(Storage)});
    throw;
  }
}

template <typename T, typename... Args>
[[nodiscard]] SharedPtr<T> make_abi_shared(Args&&... args) {
  return make_shared<T>(std::forward<Args>(args)...);
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

 public:
  using element_type = T;

  constexpr WeakPtr() noexcept = default;

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  WeakPtr(const SharedPtr<Y>& sp) noexcept : ptr_(sp.ptr_), pi_(sp.pi_) {
    if (pi_ != nullptr) {
      if (hyper_shared_ptr::detail::is_theta_sp(pi_)) {
        auto* cb =
            static_cast<hyper_shared_ptr::detail::ThetaSpCountedBase*>(pi_);
        __atomic_store_n(&cb->has_weak_or_std_, 1U, __ATOMIC_RELEASE);
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
    requires std::is_convertible_v<Y*, T*>
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
    requires std::is_convertible_v<Y*, T*>
  WeakPtr(WeakPtr<Y>&& other) noexcept : ptr_(other.ptr_), pi_(other.pi_) {
    other.ptr_ = nullptr;
    other.pi_ = nullptr;
  }

#if THETA_HSP_HAS_GLIBCXX_ABI
  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  WeakPtr(const std::shared_ptr<Y>& sp) noexcept {
    hyper_shared_ptr::detail::RawStdPtrLayout raw{};
    std::memcpy(&raw, static_cast<const void*>(&sp), sizeof(raw));
    ptr_ = sp.get();
    pi_ = raw.pi;
    if (pi_ != nullptr) {
      pi_->_M_weak_add_ref();
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
    requires std::is_convertible_v<Y*, T*>
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
    return hyper_shared_ptr::detail::sp_use_count_slow(pi_);
  }

  [[nodiscard]] bool expired() const noexcept { return use_count() == 0L; }

  [[nodiscard]] SharedPtr<T> lock() const noexcept {
    if (pi_ == nullptr) {
      return SharedPtr<T>{};
    }
    if (hyper_shared_ptr::detail::sp_weak_lock_slow(pi_)) {
      return SharedPtr<T>(ptr_, pi_, hyper_shared_ptr::detail::AdoptRefTag{});
    }
    return SharedPtr<T>{};
  }

  template <typename U>
  [[nodiscard]] bool owner_before(const WeakPtr<U>& other) const noexcept {
    return std::less<hyper_shared_ptr::detail::StdSpCountedBase*>{}(pi_,
                                                                    other.pi_);
  }

  template <typename U>
  [[nodiscard]] bool owner_before(const SharedPtr<U>& other) const noexcept {
    return std::less<hyper_shared_ptr::detail::StdSpCountedBase*>{}(pi_,
                                                                    other.pi_);
  }

#if THETA_HSP_HAS_GLIBCXX_ABI
  [[nodiscard]] std::weak_ptr<T> to_std() const noexcept {
    std::weak_ptr<T> out;
    if (pi_ != nullptr) {
      pi_->_M_weak_add_ref();
      const hyper_shared_ptr::detail::RawStdPtrLayout raw{
          const_cast<void*>(static_cast<const volatile void*>(ptr_)), pi_};
      std::memcpy(static_cast<void*>(&out), &raw, sizeof(out));
    }
    return out;
  }
#endif

 private:
  T* ptr_{nullptr};
  hyper_shared_ptr::detail::StdSpCountedBase* pi_{nullptr};
};

template <typename T>
template <typename U>
bool SharedPtr<T>::owner_before(const WeakPtr<U>& other) const noexcept {
  return std::less<hyper_shared_ptr::detail::StdSpCountedBase*>{}(pi_,
                                                                  other.pi_);
}

// ============================================================================
// `theta::KeepAlive<T>`: Lock-Free `rseq` Concurrent Snapshot Cell (`~2 ns`
// `get()` Across 8,192 CPUs, Zero Reader `shared_mutex` Contention!)
// ============================================================================
template <typename T>
class KeepAlive {
 public:
  KeepAlive() noexcept = default;

  explicit KeepAlive(T* ptr) { reset(ptr); }

  explicit KeepAlive(SharedPtr<T> sp) { reset(std::move(sp)); }

  ~KeepAlive() noexcept { reset(static_cast<T*>(nullptr)); }

  KeepAlive(const KeepAlive&) = delete;
  KeepAlive& operator=(const KeepAlive&) = delete;

  SharedPtr<T> reset(T* ptr) {
    if (ptr == nullptr) {
      return reset(SharedPtr<T>{});
    }
    return reset(SharedPtr<T>(ptr));
  }

  SharedPtr<T> reset(SharedPtr<T> next) {
    hyper_shared_ptr::detail::ThetaSpCountedBase* new_cb = nullptr;
    if (next.pi_ != nullptr) {
      if (!hyper_shared_ptr::detail::is_theta_sp(next.pi_)) {
        // Wrap external `std::shared_ptr` control block inside a Theta CB so
        // `rseq` fast-path offsets are always valid.
        std::shared_ptr<T> ext = next.to_std();
        T* raw_p = ext.get();
        next = SharedPtr<T>(
            raw_p, [hold = std::move(ext)](T*) mutable { hold.reset(); });
      }
      new_cb =
          static_cast<hyper_shared_ptr::detail::ThetaSpCountedBase*>(next.pi_);
      hyper_shared_ptr::detail::sp_pin_for_keepalive(new_cb);
    }

    SharedPtr<T> ret;
    SharedPtr<T> old_anchor;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      auto* old_cb = __atomic_exchange_n(&active_cb_, new_cb, __ATOMIC_ACQ_REL);
      if (old_cb != nullptr) {
        hyper_shared_ptr::detail::sp_keepalive_retire_old(old_cb);
      }
      old_anchor = std::move(anchor_);
      anchor_ = std::move(next);
      ret = anchor_;
    }
    return ret;
  }

  [[nodiscard]] SharedPtr<T> get() const noexcept {
    hyper_shared_ptr::detail::ThetaSpCountedBase* cb = nullptr;
    if (hyper_shared_ptr::detail::ArchBackend::try_rseq_keepalive_acquire(
            &active_cb_, cb)) [[likely]] {
      if (cb == nullptr) {
        return SharedPtr<T>{};
      }
      return SharedPtr<T>(static_cast<T*>(cb->fast_.managed_ptr), cb,
                          hyper_shared_ptr::detail::AdoptRefTag{});
    }
    return get_slow();
  }

 private:
  [[gnu::noinline]] SharedPtr<T> get_slow() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!anchor_) {
      return SharedPtr<T>{};
    }
    if (hyper_shared_ptr::detail::is_theta_sp(anchor_.pi_)) {
      auto* cb = static_cast<hyper_shared_ptr::detail::ThetaSpCountedBase*>(
          anchor_.pi_);
      hyper_shared_ptr::detail::sp_register_keepalive_cpu_slow(cb);
    }
    return anchor_;
  }

  mutable hyper_shared_ptr::detail::ThetaSpCountedBase* active_cb_{nullptr};
  mutable std::shared_mutex mutex_;
  SharedPtr<T> anchor_;
};

// ============================================================================
// `theta::HyperSharedPtr<T>`: Modernized 8,192-CPU Handle-Tagged Atomic-Shard
// Smart Pointer (Direct Evolution of Original `HyperSharedPointer`)
// ============================================================================
template <typename T>
class HyperSharedPtr {
  static_assert(!std::is_array_v<T>,
                "theta::HyperSharedPtr<T> manages non-array object types.");

  template <typename U>
  friend class HyperSharedPtr;
  template <typename U>
  friend class HyperWeakPtr;
  template <typename U>
  friend class HyperKeepAlive;

 public:
  using element_type = T;
  using weak_type = HyperWeakPtr<T>;

  constexpr HyperSharedPtr() noexcept = default;
  constexpr HyperSharedPtr(std::nullptr_t) noexcept {}

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  explicit HyperSharedPtr(Y* ptr)
      : HyperSharedPtr(ptr, std::default_delete<Y>{}) {}

  template <typename Y, typename Deleter>
    requires std::is_convertible_v<Y*, T*>
  HyperSharedPtr(Y* ptr, Deleter deleter) {
    if (ptr == nullptr) {
      deleter(ptr);
      return;
    }
    const std::int32_t cpu =
        hyper_shared_ptr::detail::ArchBackend::fast_cpu_id();
    const std::uint32_t slot =
        hyper_shared_ptr::detail::alloc_hsp_slot_and_init_cpu(cpu);
    using Storage = hyper_shared_ptr::detail::HspDeleterStorage<Y*, Deleter>;
    void* mem =
        ::operator new(sizeof(Storage), std::align_val_t{alignof(Storage)});
    try {
      auto* s = ::new (mem) Storage(slot, ptr, std::move(deleter));
      tagged_ = hyper_shared_ptr::detail::pack_hsp_tag(&s->base, cpu);
      ptr_ = ptr;
    } catch (...) {
      ::operator delete(mem, std::align_val_t{alignof(Storage)});
      deleter(ptr);
      throw;
    }
  }

  template <typename Y>
  HyperSharedPtr(const HyperSharedPtr<Y>& other,
                 element_type* alias_ptr) noexcept {
    auto* cb = hyper_shared_ptr::detail::unpack_hsp_cb(other.tagged_);
    if (cb != nullptr) {
      const std::int32_t cpu =
          hyper_shared_ptr::detail::ArchBackend::fast_cpu_id();
      hyper_shared_ptr::detail::hsp_inc_on_cpu(cb, cpu);
      tagged_ = hyper_shared_ptr::detail::pack_hsp_tag(cb, cpu);
      ptr_ = alias_ptr;
    }
  }

  HyperSharedPtr(const HyperSharedPtr& other) noexcept {
    auto* cb = hyper_shared_ptr::detail::unpack_hsp_cb(other.tagged_);
    if (cb != nullptr) {
      const std::int32_t cpu =
          hyper_shared_ptr::detail::ArchBackend::fast_cpu_id();
      hyper_shared_ptr::detail::hsp_inc_on_cpu(cb, cpu);
      tagged_ = hyper_shared_ptr::detail::pack_hsp_tag(cb, cpu);
      ptr_ = other.ptr_;
    }
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  HyperSharedPtr(const HyperSharedPtr<Y>& other) noexcept {
    auto* cb = hyper_shared_ptr::detail::unpack_hsp_cb(other.tagged_);
    if (cb != nullptr) {
      const std::int32_t cpu =
          hyper_shared_ptr::detail::ArchBackend::fast_cpu_id();
      hyper_shared_ptr::detail::hsp_inc_on_cpu(cb, cpu);
      tagged_ = hyper_shared_ptr::detail::pack_hsp_tag(cb, cpu);
      ptr_ = other.ptr_;
    }
  }

  HyperSharedPtr(HyperSharedPtr&& other) noexcept
      : tagged_(other.tagged_), ptr_(other.ptr_) {
    other.tagged_ = 0U;
    other.ptr_ = nullptr;
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  HyperSharedPtr(HyperSharedPtr<Y>&& other) noexcept
      : tagged_(other.tagged_), ptr_(other.ptr_) {
    other.tagged_ = 0U;
    other.ptr_ = nullptr;
  }

  ~HyperSharedPtr() noexcept { release_internal(); }

  HyperSharedPtr& operator=(const HyperSharedPtr& other) noexcept {
    if (this == &other) {
      return *this;
    }
    HyperSharedPtr tmp(other);
    swap(tmp);
    return *this;
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  HyperSharedPtr& operator=(const HyperSharedPtr<Y>& other) noexcept {
    HyperSharedPtr tmp(other);
    swap(tmp);
    return *this;
  }

  HyperSharedPtr& operator=(HyperSharedPtr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    release_internal();
    tagged_ = other.tagged_;
    ptr_ = other.ptr_;
    other.tagged_ = 0U;
    other.ptr_ = nullptr;
    return *this;
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  HyperSharedPtr& operator=(HyperSharedPtr<Y>&& other) noexcept {
    release_internal();
    tagged_ = other.tagged_;
    ptr_ = other.ptr_;
    other.tagged_ = 0U;
    other.ptr_ = nullptr;
    return *this;
  }

  HyperSharedPtr& operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }

  void reset() noexcept {
    HyperSharedPtr tmp;
    swap(tmp);
  }

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  void reset(Y* ptr) {
    HyperSharedPtr tmp(ptr);
    swap(tmp);
  }

  template <typename Y, typename Deleter>
    requires std::is_convertible_v<Y*, T*>
  void reset(Y* ptr, Deleter deleter) {
    HyperSharedPtr tmp(ptr, std::move(deleter));
    swap(tmp);
  }

  void swap(HyperSharedPtr& other) noexcept {
    std::swap(tagged_, other.tagged_);
    std::swap(ptr_, other.ptr_);
  }

  [[nodiscard]] element_type* get() const noexcept { return ptr_; }

  [[nodiscard]] std::add_lvalue_reference_t<T> operator*() const noexcept
    requires(!std::is_void_v<T>)
  {
    return *ptr_;
  }

  [[nodiscard]] element_type* operator->() const noexcept { return ptr_; }

  [[nodiscard]] explicit operator bool() const noexcept {
    return ptr_ != nullptr;
  }

  [[nodiscard]] std::size_t use_count() const noexcept {
    auto* cb = hyper_shared_ptr::detail::unpack_hsp_cb(tagged_);
    return hyper_shared_ptr::detail::hsp_use_count_slow(cb);
  }

  [[nodiscard]] std::int32_t original_cpu() const noexcept {
    return hyper_shared_ptr::detail::unpack_hsp_cpu(tagged_);
  }

  // Legacy method name retained for `theta-epoch-ptr` compatibility
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] int originalCpu() const noexcept {
    return static_cast<int>(original_cpu());
  }

 private:
  HyperSharedPtr(std::uintptr_t tagged, element_type* ptr,
                 hyper_shared_ptr::detail::AdoptRefTag) noexcept
      : tagged_(tagged), ptr_(ptr) {}

  void release_internal() noexcept {
    if (tagged_ != 0U) {
      auto* cb = hyper_shared_ptr::detail::unpack_hsp_cb(tagged_);
      const std::int32_t cpu =
          hyper_shared_ptr::detail::unpack_hsp_cpu(tagged_);
      tagged_ = 0U;
      ptr_ = nullptr;
      hyper_shared_ptr::detail::hsp_dec_on_origin(cb, cpu);
    }
  }

  template <typename U, typename... Args>
  friend HyperSharedPtr<U> make_hyper_shared(Args&&... args);

  // Testing hook for explicit CPU-ID handle tagging across `[0, 8191]`
  template <typename U>
  friend HyperSharedPtr<U> copy_hyper_on_cpu_for_test(
      const HyperSharedPtr<U>& other, std::int32_t cpu) noexcept;

  std::uintptr_t tagged_{0U};
  element_type* ptr_{nullptr};
};

static_assert(sizeof(HyperSharedPtr<int>) == 2 * sizeof(void*));

template <typename T, typename... Args>
[[nodiscard]] HyperSharedPtr<T> make_hyper_shared(Args&&... args) {
  using CleanT = std::remove_cv_t<T>;
  const std::int32_t cpu = hyper_shared_ptr::detail::ArchBackend::fast_cpu_id();
  const std::uint32_t slot =
      hyper_shared_ptr::detail::alloc_hsp_slot_and_init_cpu(cpu);
  using Storage = hyper_shared_ptr::detail::HspInplaceStorage<CleanT>;
  void* mem =
      ::operator new(sizeof(Storage), std::align_val_t{alignof(Storage)});
  try {
    auto* s = ::new (mem) Storage(slot, std::forward<Args>(args)...);
    const std::uintptr_t tagged =
        hyper_shared_ptr::detail::pack_hsp_tag(&s->base, cpu);
    return HyperSharedPtr<T>(tagged, s->ptr(),
                             hyper_shared_ptr::detail::AdoptRefTag{});
  } catch (...) {
    ::operator delete(mem, std::align_val_t{alignof(Storage)});
    throw;
  }
}

template <typename T>
[[nodiscard]] HyperSharedPtr<T> copy_hyper_on_cpu_for_test(
    const HyperSharedPtr<T>& other, std::int32_t cpu) noexcept {
  auto* cb = hyper_shared_ptr::detail::unpack_hsp_cb(other.tagged_);
  if (cb == nullptr) {
    return HyperSharedPtr<T>{};
  }
  const std::int32_t clamped = (cpu >= 0 && cpu < kHspMaxCpus) ? cpu : 0;
  hyper_shared_ptr::detail::hsp_inc_on_cpu(cb, clamped);
  return HyperSharedPtr<T>(hyper_shared_ptr::detail::pack_hsp_tag(cb, clamped),
                           other.ptr_, hyper_shared_ptr::detail::AdoptRefTag{});
}

template <typename T, typename U>
[[nodiscard]] bool operator==(const HyperSharedPtr<T>& lhs,
                              const HyperSharedPtr<U>& rhs) noexcept {
  return lhs.get() == rhs.get();
}

template <typename T>
[[nodiscard]] bool operator==(const HyperSharedPtr<T>& lhs,
                              std::nullptr_t) noexcept {
  return lhs.get() == nullptr;
}

template <typename T>
void swap(HyperSharedPtr<T>& a, HyperSharedPtr<T>& b) noexcept {
  a.swap(b);
}

// ============================================================================
// `theta::HyperWeakPtr<T>`: Weak Counterpart for `HyperSharedPtr<T>`
// ============================================================================
template <typename T>
class HyperWeakPtr {
  template <typename U>
  friend class HyperWeakPtr;

 public:
  using element_type = T;

  constexpr HyperWeakPtr() noexcept = default;

  template <typename Y>
    requires std::is_convertible_v<Y*, T*>
  HyperWeakPtr(const HyperSharedPtr<Y>& sp) noexcept
      : cb_(hyper_shared_ptr::detail::unpack_hsp_cb(sp.tagged_)),
        ptr_(sp.ptr_) {
    if (cb_ != nullptr) {
      cb_->weak_refs_.fetch_add(1U, std::memory_order_relaxed);
    }
  }

  HyperWeakPtr(const HyperWeakPtr& other) noexcept
      : cb_(other.cb_), ptr_(other.ptr_) {
    if (cb_ != nullptr) {
      cb_->weak_refs_.fetch_add(1U, std::memory_order_relaxed);
    }
  }

  HyperWeakPtr(HyperWeakPtr&& other) noexcept
      : cb_(other.cb_), ptr_(other.ptr_) {
    other.cb_ = nullptr;
    other.ptr_ = nullptr;
  }

  ~HyperWeakPtr() noexcept {
    if (cb_ != nullptr) {
      hyper_shared_ptr::detail::hsp_weak_release_slow(cb_);
    }
  }

  HyperWeakPtr& operator=(const HyperWeakPtr& other) noexcept {
    if (this == &other) {
      return *this;
    }
    HyperWeakPtr tmp(other);
    swap(tmp);
    return *this;
  }

  HyperWeakPtr& operator=(HyperWeakPtr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (cb_ != nullptr) {
      hyper_shared_ptr::detail::hsp_weak_release_slow(cb_);
    }
    cb_ = other.cb_;
    ptr_ = other.ptr_;
    other.cb_ = nullptr;
    other.ptr_ = nullptr;
    return *this;
  }

  void reset() noexcept {
    HyperWeakPtr tmp;
    swap(tmp);
  }

  void swap(HyperWeakPtr& other) noexcept {
    std::swap(cb_, other.cb_);
    std::swap(ptr_, other.ptr_);
  }

  [[nodiscard]] std::size_t use_count() const noexcept {
    return hyper_shared_ptr::detail::hsp_use_count_slow(cb_);
  }

  [[nodiscard]] bool expired() const noexcept {
    return cb_ == nullptr ||
           cb_->active_shards_.load(std::memory_order_acquire) == 0U;
  }

  [[nodiscard]] HyperSharedPtr<T> lock() const noexcept {
    if (cb_ == nullptr) {
      return HyperSharedPtr<T>{};
    }
    const std::int32_t cpu =
        hyper_shared_ptr::detail::ArchBackend::fast_cpu_id();
    if (hyper_shared_ptr::detail::hsp_weak_lock_slow(cb_, cpu)) {
      return HyperSharedPtr<T>(hyper_shared_ptr::detail::pack_hsp_tag(cb_, cpu),
                               ptr_, hyper_shared_ptr::detail::AdoptRefTag{});
    }
    return HyperSharedPtr<T>{};
  }

 private:
  hyper_shared_ptr::detail::HyperControlBlock* cb_{nullptr};
  T* ptr_{nullptr};
};

// ============================================================================
// `theta::HyperKeepAlive<T>`: Pre-Pinned Shard Factory for `HyperSharedPtr<T>`
// ============================================================================
template <typename T>
class HyperKeepAlive {
 public:
  HyperKeepAlive() noexcept = default;

  explicit HyperKeepAlive(T* ptr) { reset(ptr); }

  ~HyperKeepAlive() noexcept { reset(nullptr); }

  HyperKeepAlive(const HyperKeepAlive&) = delete;
  HyperKeepAlive& operator=(const HyperKeepAlive&) = delete;

  HyperSharedPtr<T> reset(T* ptr) {
    const std::int32_t ncpus = hyper_shared_ptr::detail::max_possible_cpus();
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (ptr_) {
      auto* old_cb = hyper_shared_ptr::detail::unpack_hsp_cb(ptr_.tagged_);
      for (std::int32_t c = 0; c < ncpus; ++c) {
        hyper_shared_ptr::detail::hsp_dec_on_origin(old_cb, c);
      }
      ptr_.reset();
    }

    if (ptr != nullptr) {
      ptr_ = HyperSharedPtr<T>(ptr);
      auto* cb = hyper_shared_ptr::detail::unpack_hsp_cb(ptr_.tagged_);
      for (std::int32_t c = 0; c < ncpus; ++c) {
        hyper_shared_ptr::detail::hsp_inc_on_cpu(cb, c);
      }
    }
    return ptr_;
  }

  [[nodiscard]] HyperSharedPtr<T> get() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return ptr_;
  }

 private:
  mutable std::shared_mutex mutex_;
  HyperSharedPtr<T> ptr_;
};

}  // namespace theta
