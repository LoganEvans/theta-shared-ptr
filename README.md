<!-- FILENAME: README.md -->

# `theta-shared-ptr`

100% binary `std::shared_ptr<T>` / `std::weak_ptr<T>` ABI-compatible concurrent smart pointer (`theta::SharedPtr<T>` / `theta::shared_ptr<T>`), `16 B` `rseq`-accelerated atomic smart pointer (`theta::AtomicSharedPtr<T>` / `theta::atomic_shared_ptr<T>` = `std::atomic<theta::SharedPtr<T>>`), and wait-free snapshot cell (`theta::KeepAlive<T>`), powered by Linux `rseq` (`0 .. 8,191` CPUs) and demand-paged `MAP_NORESERVE` striped per-CPU delta slabs.

## Highlights

- **Zero-Allocation Binary `std::shared_ptr<T>` & `std::weak_ptr<T>` ABI Compatibility**:
  - `sizeof(theta::SharedPtr<T>) == 16` (`[T*, std::_Sp_counted_base<_S_atomic>*]`, untagged pointers).
  - `ThetaSpCountedBase` is a `32 B` non-template `final` subclass of `std::_Sp_counted_base<_S_atomic>` identified in 1 instruction via its canonical Itanium vtable symbol (`_ZTV + 16`), preserving `owner_before`, `std::weak_ptr::lock()`, `std::enable_shared_from_this<T>`, and zero-allocation handoff across unmodified precompiled C++ DSOs.
- **Full C++20/23 Drop-In `std::shared_ptr` & `std::atomic<std::shared_ptr>` Feature Parity (`<theta/shared_ptr.hpp>`)**:
  - `theta::SharedPtr<T>` / `theta::shared_ptr<T>`, `theta::WeakPtr<T>` / `theta::weak_ptr<T>`, `theta::AtomicSharedPtr<T>` / `theta::atomic_shared_ptr<T>` (`std::atomic<theta::SharedPtr<T>>`), `theta::AtomicWeakPtr<T>` / `theta::atomic_weak_ptr<T>` (`std::atomic<theta::WeakPtr<T>>`), and `theta::EnableSharedFromThis<T>` / `theta::enable_shared_from_this<T>` (plus automatic wiring of `std::enable_shared_from_this<T>`).
  - Scalar, bounded-array (`T[N]`), and single-allocation unbounded-array (`T[]`) `make_shared`, `make_shared_for_overwrite`, `allocate_shared`, and `allocate_shared_for_overwrite`.
  - `std::unique_ptr` adoption, custom deleters & allocators, `get_deleter`, `static_pointer_cast` / `dynamic_pointer_cast` / `const_pointer_cast` / `reinterpret_pointer_cast` (`const&` and `&&`), `<=>`, `owner_less` / `owner_equal` / `owner_hash`, and `std::hash<theta::SharedPtr<T>>`.
- **Wait-Free `rseq` Hot Sharing, `16 B` `AtomicSharedPtr<T>` & `KeepAlive<T>` Across Up to `8,192` CPUs**:
  - Unshared `make_shared<uint64_t>` + `~SharedPtr`: zero atomic RMWs and zero virtual calls on teardown via LSB-tagged sized `delete`.
  - Ephemeral / warm `< 8`-copy objects: lockless inline 32-bit `slot_id` CAS (`0` spinlocks, `0` out-of-line calls, `0` per-CPU slots, `0` `SYS_membarrier` IPIs).
  - Concurrent `>= 8`-copy hot sharing, `theta::AtomicSharedPtr<T>::load()` (`16 B`, full `compare_exchange_strong/weak` + `wait`/`notify_*` — **preferred default** for concurrent config/route/snapshot cells), and `theta::KeepAlive<T>::get()` (`72 B`) execute inside a single CPU-local `rseq` critical section with zero `LOCK` instructions, zero shared cache-line writes, adaptive read-warmup (`0` `SYS_membarrier` IPIs on write/CAS-dominated cells), and automatic `SYS_membarrier` storm throttling + cooldown re-pinning.

## Memory & Virtual-Address Footprint

`theta::SharedPtr<T>` trades a demand-paged per-CPU delta arena for zero-coherence-traffic reference counting on shared objects while keeping uncopied objects completely free of arena overhead:

| Component | `std::shared_ptr<T>` | `theta::SharedPtr<T>` | Notes |
| :--- | :--- | :--- | :--- |
| **Inline handle (`sizeof`)** | `16 B` | `16 B` | Identical `[T*, _Sp_counted_base*]` ABI layout |
| **Atomic handle (`AtomicSharedPtr<T>`)** | `16 B` | `16 B` | Wait-free `rseq` `load()` (`4.5x` smaller than `72 B` `KeepAlive<T>`) + `CAS`/`wait`/`notify_*` |
| **`make_shared<T>` heap block** | `24 B + sizeof(T)` | `32 B + sizeof(T)` | `40 B` for `uint64_t` -> same `48 B` glibc `malloc` chunk |
| **Sole-owner (`0` copies) & warm (`1..7` copies)** | `0 B` extra | **`0 B` extra** | `kModeUnpromoted` (`0` atomics on `0` copies; lockless inline 32-bit CAS on `1..7` copies): `0` slots, `0` slab pages, **`0` `SYS_membarrier` IPIs** |
| **Process virtual address (`VIRT`)** | `0 B` | **`8.01 GiB` `MAP_NORESERVE`** | `131,072` slots × `8,192` CPUs × `8 B` (`256 MiB` under ASan/TSan); **`0 B` physical RAM until faulted** |
| **Cold-start RSS (1st promoted ptr, `>= 8` copies)** | `0 B` | **`4 KiB` + `4 KiB × active_cpus`** | `(cpu << 17) + slot_id` packs **`512` pointers per `4 KiB` CPU page** (`~52 KiB` on 12 CPUs; `0 B` on unused CPUs) |
| **Marginal RSS (promoted ptrs `#2..#511`)** | `0 B` | **`64 B` + `8 B × active_cpus`** | Reuses faulted `4 KiB` pages (`160 B`/ptr on 12 CPUs, `576 B`/ptr on 64 CPUs; `+8 B` if in `KeepAlive<T>`) |
| **Slot recycling & saturation** | N/A | **`131,072` concurrent slots** | Freed `slot_id`s recycle via `initial-exec` TLS (`0` new faults); `> 131,072` live shared objects fall back to atomic `_Sp_counted_base` |

## Build & Test

Inside `nix develop` (or via `bspot`):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```
