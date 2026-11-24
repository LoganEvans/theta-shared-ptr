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

Measured via `bspot` (`Clang 23 -O3 -flto=thin`) on **Intel Coffee Lake `i7-8086K` (`6C/12T`, `r767`)** and **ARM Neoverse-N1 (`Ampere Altra 4C/4T`, `r768`)**:

| Workload | Host / Threads | `std::shared_ptr` | `theta::HyperSharedPtr` | `theta::SharedPtr` (`ABI` + `rseq`) | Speedup (`SharedPtr` vs `std`) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `01_CreateDestroy` (unshared) | x86 `1T` / `12T` | `10.2` / `24.6 ns` | `33.9` / `46.1 ns` | `18.1` / `35.6 ns` | `0.56x` / `0.69x` (`250x` vs v0.1 HSP) |
| `01_CreateDestroy` (unshared) | ARM64 `1T` / `4T` | `14.1` / `14.1 ns` | `42.8` / `43.4 ns` | `27.3` / `27.4 ns` | `0.52x` / `0.51x` |
| `02_HotShare_0to1` (`auto p = g;`) | x86 `1T` / `4T` / `12T` | `4.91` / `151.9` / `474.9 ns` | `14.8` / `168.8` / `492.6 ns` | **`3.34` / `3.35` / `6.15 ns`** | **`1.47x` / `45.3x` / `77.2x`** |
| `02_HotShare_0to1` (`auto p = g;`) | ARM64 `1T` / `2T` / `4T` | `3.59` / `47.6` / `289.7 ns` | `16.1` / `48.6` / `257.9 ns` | **`5.45` / `5.95` / `5.79 ns`** | `0.66x` / **`8.0x` / `50.0x`** |
| `03_HotShare_Pinned` (`1->2->1`) | x86 `1T` / `4T` / `12T` | `4.92` / `191.3` / `487.9 ns` | `14.9` / `78.0` / `277.9 ns` | **`3.37` / `3.37` / `6.22 ns`** | **`1.46x` / `56.8x` / `78.4x`** |
| `03_HotShare_Pinned` (`1->2->1`) | ARM64 `1T` / `2T` / `4T` | `3.59` / `90.1` / `129.0 ns` | `16.5` / `16.6` / `43.0 ns` | **`6.11` / `5.79` / `5.96 ns`** | `0.59x` / **`15.6x` / `21.7x`** |
| `04_CrossThread_ProdCons` | x86 `2T` / `4T` / `12T` | `104.5` / `125.2` / `289.9 ns` | `93.0` / **`100.3` / `140.6 ns`** | **`83.9`** / `123.4` / `269.8 ns` | **`1.24x` / `1.01x` / `1.07x`** (`HSP`: **`2.06x`**) |
| `04_CrossThread_ProdCons` | ARM64 `2T` / `4T` | `218.8` / `330.4 ns` | **`184.3` / `173.9 ns`** | `194.9` / `331.9 ns` | **`1.12x`** / `1.00x` (`HSP`: **`1.90x`**) |
| `05_KeepAlive_Snapshot` (`99.99%` R) | x86 `1T` / `4T` / `12T` | `15.5` / `334.4` / `1208.7 ns` | `26.0` / `303.3` / `1130.9 ns` | **`3.26` / `3.50` / `6.73 ns`** | **`4.76x` / `95.6x` / `179.6x`** |
| `05_KeepAlive_Snapshot` (`99.99%` R) | ARM64 `1T` / `2T` / `4T` | `19.8` / `403.1` / `776.7 ns` | `32.3` / `141.2` / `359.9 ns` | **`7.01` / `7.49` / `8.49 ns`** | **`2.82x` / `53.8x` / `91.5x`** |
| `06_Table2K_WorkingSet` (`2,048` ptrs) | x86 `4T` / `12T` | `24.8` / `37.7 ns` | `32.2` / `62.0 ns` | **`5.36` / `8.06 ns`** | **`4.62x` / `4.68x`** |
| `06_Table2K_WorkingSet` (`2,048` ptrs) | ARM64 `2T` / `4T` | `46.8` / `53.2 ns` | `25.3` / `69.9 ns` | **`6.03` / `5.95 ns`** | **`7.76x` / `8.93x`** |

## Build, Test & Health Check

```bash
# Format, build, run unit/sanitizer/preemption-storm tests, and check artifacts via bspot
bspot test ~/theta/theta-hyper-shared-ptr --remote local,theta-arm64
bspot test ~/theta/theta-hyper-shared-ptr --remote local,theta-arm64 --asan
bspot test ~/theta/theta-hyper-shared-ptr --remote local,theta-arm64 --tsan
bspot check ~/theta/theta-hyper-shared-ptr
```
