<!-- FILENAME: README.md -->

# `theta-shared-ptr`

100% binary `std::shared_ptr<T>` / `std::weak_ptr<T>` ABI-compatible concurrent smart pointer (`theta::SharedPtr<T>` / `theta::shared_ptr<T>`) and wait-free snapshot cell (`theta::KeepAlive<T>`), powered by Linux `rseq` (`0 .. 8,191` CPUs) and demand-paged `MAP_NORESERVE` striped per-CPU delta slabs.

## Highlights

- **Zero-Allocation Binary `std::shared_ptr<T>` & `std::weak_ptr<T>` ABI Compatibility**:
  - `sizeof(theta::SharedPtr<T>) == 16` (`[T*, std::_Sp_counted_base<_S_atomic>*]`, untagged pointers).
  - `ThetaSpCountedBase` is a `32 B` non-template `final` subclass of `std::_Sp_counted_base<_S_atomic>` identified in 1 instruction via its canonical Itanium vtable symbol (`_ZTV + 16`), preserving `owner_before`, `std::weak_ptr::lock()`, `std::enable_shared_from_this<T>`, and zero-allocation handoff across unmodified precompiled C++ DSOs.
- **Full C++20/23 Drop-In `std::shared_ptr` Feature Parity (`<theta/shared_ptr.hpp>`)**:
  - `theta::SharedPtr<T>` / `theta::shared_ptr<T>`, `theta::WeakPtr<T>` / `theta::weak_ptr<T>`, `theta::EnableSharedFromThis<T>` / `theta::enable_shared_from_this<T>` (plus automatic wiring of `std::enable_shared_from_this<T>`).
  - Scalar, bounded-array (`T[N]`), and single-allocation unbounded-array (`T[]`) `make_shared`, `make_shared_for_overwrite`, `allocate_shared`, and `allocate_shared_for_overwrite`.
  - `std::unique_ptr` adoption, custom deleters & allocators, `get_deleter`, `static_pointer_cast` / `dynamic_pointer_cast` / `const_pointer_cast` / `reinterpret_pointer_cast` (`const&` and `&&`), `<=>`, `owner_less` / `owner_equal` / `owner_hash`, `std::hash<theta::SharedPtr<T>>`, and `std::atomic<theta::SharedPtr<T>>` / `std::atomic<theta::WeakPtr<T>>`.
- **Wait-Free `rseq` Hot Sharing & `KeepAlive<T>` Across Up to `8,192` CPUs**:
  - Unshared `make_shared<uint64_t>` + `~SharedPtr`: **`10.2 ns`** on x86-64 (`1T`, faster than `std::make_shared`'s `10.9 ns`) and **`14.6 ns`** (`155 insns/op`, matching `std::make_shared` to the instruction) on ARM64 via `0`-indirect-call LSB-tagged sized `delete`.
  - Concurrent `0 -> 1 -> 0` and `1 -> 2 -> 1` hot sharing: **`3.3–5.9 ns`** (`81x–85x` faster than `std::shared_ptr` at `12T` x86-64; `21x–38x` faster at `4T` ARM64).
  - `theta::KeepAlive<T>::get()`: **`3.6–7.2 ns`** (`150x` faster than `std::shared_mutex` + `std::shared_ptr` at `12T` x86-64; `91x` faster at `4T` ARM64).

## Build & Test

Inside `nix develop` (or via `bspot`):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```
