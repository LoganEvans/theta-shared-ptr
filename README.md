<!-- FILENAME: README.md -->

# `theta-hyper-shared-ptr` — `8,192`-CPU `rseq` `std::shared_ptr`-ABI & Handle-Tagged Smart Pointers

`theta-hyper-shared-ptr` provides two high-concurrency C++23 reference-counted smart pointer families scaling up to **`8,192` (`MAXSMP`) CPUs**:

1. **`theta::SharedPtr<T>` / `theta::WeakPtr<T>` / `theta::KeepAlive<T>` (`<theta/hyper_shared_ptr.hpp>`)**:
   - **100% Binary `std::shared_ptr<T>` & `std::weak_ptr<T>` ABI Compatible** (`sizeof == 16`, untagged `[T*, std::_Sp_counted_base<_S_atomic>*]` layout, preserving `owner_before` / `std::owner_less` and zero-allocation `.to_std()` / implicit conversions across unmodified precompiled DSOs).
   - **Zero-Atomic `rseq` Striped Per-CPU Fast Path**: Lazy promotion on first copy into demand-paged (`MAP_NORESERVE`) `8,192`-CPU striped per-CPU delta counters (`0` `LOCK` instructions, `0` memory fences, width-matched 8-byte Store-to-Load Forwarding, **`3.3–6.2 ns`** across `1T–12T` x86-64 and **`5.5–6.0 ns` (`~17 cyc`)** across `1T–4T` ARM64).
   - **Wait-Free `rseq` `KeepAlive<T>::get()`**: Acquires both the active pointer snapshot and per-CPU reference count inside one `rseq` critical section (`3.3–6.7 ns` at `12T`, **`91x–180x` faster** than `std::shared_mutex`-guarded `std::shared_ptr`).
2. **`theta::HyperSharedPtr<T>` / `theta::HyperWeakPtr<T>` / `theta::HyperKeepAlive<T>`** (also aliased under legacy `theta::hsp::HyperSharedPointer<T>`):
   - Modernized evolution of the handle-tagged per-CPU atomic shard design: packs `13`-bit `origin_cpu` (`0 .. 8,191`) into `alignas(128)` low `7` bits + upper `6` canonical bits (`58..63`, LA57 & ARM64 LVA safe), backed by striped per-CPU atomic shards and lock-free `active_shards_` counting.
   - Excels on **multi-pair cross-CPU Producer → Consumer queues** (`1.9x–2.1x` faster than `std::shared_ptr`), where `origin_cpu` isolates each producer–consumer pair onto its own producer cache line.

## 3-Way ThinLTO Benchmark Standing (`ns/op` Median)

Measured via `bspot` (`Clang 23 -O3 -flto=thin`) on **Intel Coffee Lake `i7-8086K` (`6C/12T`, `r776`)** and **ARM Neoverse-N1 (`Ampere Altra 4C/4T`, `r777`)**:

| Workload | Host / Threads | `std::shared_ptr` | `theta::HyperSharedPtr` | `theta::SharedPtr` (`ABI` + `rseq`) | Speedup (`SharedPtr` vs `std`) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `01_CreateDestroy` (unshared) | x86 `1T` / `4T` / `12T` | `10.9` / `10.7` / `25.4 ns` | `34.9` / `35.0` / `45.9 ns` | **`10.2` / `10.4` / `20.0 ns`** | **`1.07x` / `1.03x` / `1.27x`** (`440x` vs v0.1 HSP) |
| `01_CreateDestroy` (unshared) | ARM64 `1T` / `4T` | `14.1` / `14.1 ns` | `42.8` / `42.9 ns` | **`14.6` / `14.6 ns`** (`155` insns) | `0.97x` / `0.96x` (exact `155`-insn parity) |
| `02_HotShare_0to1` (`auto p = g;`) | x86 `1T` / `4T` / `12T` | `4.92` / `147.1` / `503.2 ns` | `14.9` / `170.2` / `517.4 ns` | **`3.31` / `3.32` / `5.90 ns`** | **`1.49x` / `44.3x` / `85.3x`** |
| `02_HotShare_0to1` (`auto p = g;`) | ARM64 `1T` / `2T` / `4T` | `3.58` / `48.2` / `231.6 ns` | `16.1` / `48.2` / `250.9 ns` | **`6.21` / `6.19` / `6.11 ns`** | `0.58x` / **`7.8x` / `37.9x`** |
| `03_HotShare_Pinned` (`1->2->1`) | x86 `1T` / `4T` / `12T` | `4.93` / `190.8` / `467.5 ns` | `14.9` / `71.0` / `239.3 ns` | **`3.31` / `3.32` / `5.74 ns`** | **`1.49x` / `57.5x` / `81.5x`** |
| `03_HotShare_Pinned` (`1->2->1`) | ARM64 `1T` / `2T` / `4T` | `3.58` / `92.9` / `124.5 ns` | `16.4` / `16.6` / `49.0 ns` | **`5.85` / `5.68` / `5.87 ns`** | `0.61x` / **`16.4x` / `21.2x`** |
| `04_CrossThread_ProdCons` | x86 `2T` / `4T` / `12T` | `104.8` / `127.4` / `266.2 ns` | `95.6` / `103.1` / **`139.9 ns`** | **`59.4` / `102.2`** / `292.3 ns` | **`1.76x` / `1.25x`** / `0.91x` (`HSP`: **`1.90x`**) |
| `04_CrossThread_ProdCons` | ARM64 `2T` / `4T` | `205.1` / `339.8 ns` | `167.1` / **`171.9 ns`** | **`153.3`** / `204.7 ns` | **`1.34x` / `1.66x`** (`HSP`: **`1.98x`**) |
| `05_KeepAlive_Snapshot` (`99.99%` R) | x86 `1T` / `4T` / `12T` | `15.9` / `328.7` / `1080.0 ns` | `26.6` / `273.9` / `1060.0 ns` | **`3.59` / `4.38` / `7.21 ns`** | **`4.43x` / `75.0x` / `149.8x`** |
| `05_KeepAlive_Snapshot` (`99.99%` R) | ARM64 `1T` / `2T` / `4T` | `19.3` / `396.6` / `825.7 ns` | `32.1` / `129.2` / `348.3 ns` | **`7.25` / `7.71` / `9.09 ns`** | **`2.66x` / `51.4x` / `90.8x`** |
| `06_Table2K_WorkingSet` (`2,048` ptrs) | x86 `1T` / `4T` / `12T` | `4.74` / `22.1` / `38.5 ns` | `13.2` / `32.6` / `59.7 ns` | **`3.43` / `3.51` / `5.86 ns`** | **`1.38x` / `6.30x` / `6.57x`** |
| `06_Table2K_WorkingSet` (`2,048` ptrs) | ARM64 `1T` / `2T` / `4T` | `2.93` / `45.5` / `52.4 ns` | `14.6` / `25.2` / `69.7 ns` | **`5.64` / `5.67` / `5.64 ns`** | `0.52x` / **`8.02x` / `9.29x`** |

## Build, Test & Health Check

```bash
# Format, build, run unit/sanitizer/preemption-storm tests, and check artifacts via bspot
bspot test ~/theta/theta-hyper-shared-ptr --remote local,theta-arm64
bspot test ~/theta/theta-hyper-shared-ptr --remote local,theta-arm64 --asan
bspot test ~/theta/theta-hyper-shared-ptr --remote local,theta-arm64 --tsan
bspot check ~/theta/theta-hyper-shared-ptr
```
