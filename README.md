# ispc_apfrnn

All-pairs fixed-radius nearest neighbor search in 3D. It's a faster drop-in
alternative to [CompactNSearch] and [TreeNSearch] for SPH-style workloads,
built from ISPC kernels and TBB parallelism. The main consumer is learnsph.

[CompactNSearch]: https://github.com/InteractiveComputerGraphics/CompactNSearch
[TreeNSearch]: https://github.com/InteractiveComputerGraphics/TreeNSearch

## Layout

```
include/apfrnn/
  neighbor_search.hpp   NSearch: the public API (the only public header)
src/
  neighbor_search.cpp   set/pair state, caching, run() phases, skin filtering
  plan.{hpp,cpp}        strategy selection: which writer, ranges, capacities
  sweep.{hpp,cpp}       sweep backend for extremely sparse sets
  apfrnn.{hpp,cpp}      grid build, cell ranges, fixed-row/CSR writers
  kernels/              SIMD kernels (ISPC, no raw intrinsics):
    common.isph         structs shared with C++, cell keys, hash lookup
    cells.ispc          candidate cells and point ranges per query cell
    rows.ispc           fixed-capacity row writers
    csr.ispc            exact CSR count and write passes
    sync.ispc           coordinate sync from the caller's xyz
  sycl/                 optional static SYCL GPU backend
support/                synthetic scenes and VTK loader shared by tests/bench
tests/                  doctest suite (brute-force comparisons)
bench/                  Google Benchmark suite (Frame/, Threads/, Api/)
scripts/                benchmark, CI, coverage and profiling helpers
```

## Usage

```cpp
#include <apfrnn/neighbor_search.hpp>

apfrnn::NSearch search(radius);
const int fluid = search.add_point_set(fluid_xyz, fluid_count);
const int boundary =
    search.add_point_set(boundary_xyz, boundary_count, /*is_static=*/true);

search.set_all_searches(false);
search.set_active_search(fluid, fluid);
search.set_active_search(fluid, boundary);

search.run();

for (uint32_t j : search.get_neighbors(fluid, boundary, i)) {
  // j is an index into the boundary set's original order
}
```

- `add_point_set` keeps a pointer to the caller's interleaved `xyz` buffer
  (float or double). The buffer must stay alive. If it reallocates or
  changes size, call `resize_point_set` with the new pointer.
- `run()` re-reads dynamic sets every time. Static sets are cached until they
  are resized or toggled with `set_point_set_static`.
- `set_skin(s)` builds lists with `radius + s` and reuses them while no point
  has moved more than `s / 2` from the positions they were built at.
  `get_neighbors` still returns exact-radius lists.
- `set_memory_budget(bytes)` caps one row allocation (default 3 GiB). When a
  fixed-row table does not fit, the cross writer uses a compact per-row block
  layout before falling back to CSR rows.
- `last_stats()` reports build, grid, same-set and cross-set timings, row
  mode and memory.

### Consuming from CMake

```cmake
add_subdirectory(path/to/ispc_apfrnn ispc_apfrnn EXCLUDE_FROM_ALL)
target_link_libraries(my_target PRIVATE ispc_apfrnn_kernels)
```

The parent project must enable the ISPC language and provide TBB. Tests and
benchmarks are only built when this repository is the top-level project.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ -DAPFRNN_BUILD_BASELINES=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

| Option | Default | |
|---|---|---|
| `APFRNN_BUILD_TESTS` | top-level | doctest suite |
| `APFRNN_BUILD_BENCHMARKS` | top-level | Google Benchmark targets |
| `APFRNN_BUILD_BASELINES` | `OFF` | fetches CompactNSearch/TreeNSearch |
| `APFRNN_ENABLE_SYCL` | `OFF` | static SYCL GPU backend, requires icpx |

## Optional SYCL GPU backend

```sh
source /opt/intel/oneapi/setvars.sh
cmake --preset sycl-release
cmake --build --preset sycl-release
```

```cpp
search.set_search_backend(apfrnn::SearchBackend::GpuExact);
search.set_search_backend(apfrnn::SearchBackend::Auto);
```

`APFRNN_BACKEND=cpu|gpu|auto` preselects the backend. GPU results are
set-equal to CPU results.

### GPU frame times

Earlier SYCL build; its CPU column is not the current CPU table.

<!-- bench:gpu:start -->
| Scene | CPU | GPU | Auto | GPU / CPU |
|---|---:|---:|---:|---:|
| uniform, 1M | 31.9 ms | 19.9 ms | 19.0 ms | 1.60x |
| uniform, 200k | 5.62 ms | 4.67 ms | 4.73 ms | 1.20x |
| SPH-like slab, 200k | 7.16 ms | 5.84 ms | 5.88 ms | 1.23x |
| uniform, 50k | 1.19 ms | 2.03 ms | 1.20 ms | 0.59x |
| surface, 200k | 15.0 ms | 19.0 ms | 15.6 ms | 0.79x |
| clustered, 200k | 24.8 ms | 34.4 ms | 24.8 ms | 0.72x |
| thin pipe, 200k | 107 ms | 104 ms* | 106 ms | 0.97x* |
| learnsph wall | 3.41 ms | 2.25 ms | 3.42 ms | 1.52x |
| learnsph bowl | 3.28 ms | 2.05 ms | 3.29 ms | 1.60x |

\* Falls back to the CPU because the output exceeds `APFRNN_SYCL_MAX_OUTPUT_MB`.
<!-- bench:gpu:end -->

## Performance

2 s x 7 runs, wall time, i7-12700K pinned to 0-15. Static scenes reuse an
unchanged grid/sort and rewrite rows; moving scenes rebuild.

### Synthetic scenes

<!-- bench:headline-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 32 | 4.22 ms | 19.18 ms | 28.97 ms | 4.5x | 6.9x |
| uniform, 1M | 33 | 24.87 ms | 132.62 ms | 305.52 ms | 5.3x | 12.3x |
| uniform, moving | 32 | 6.78 ms | 19.98 ms | 33.78 ms | 2.9x | 5.0x |
| jittered lattice | 29 | 3.73 ms | 9.04 ms | 22.64 ms | 2.4x | 6.1x |
| SPH-like slab | 58 | 5.88 ms | 19.73 ms | 42.16 ms | 3.4x | 7.2x |
| slab, moving | 58 | 8.44 ms | 20.43 ms | 46.05 ms | 2.4x | 5.5x |
| huge sparse domain | 0 | 1.44 ms | 4.85 ms | 36.09 ms | 3.4x | 25.0x |
| thin slab | 243 | 14.66 ms | 25.30 ms | 451.19 ms | 1.7x | 30.8x |
| surface | 270 | 15.24 ms | 23.51 ms | 84.27 ms | 1.5x | 5.5x |
| two-scale clusters | 575 | 20.82 ms | 36.68 ms | 685.90 ms | 1.8x | 33.0x |
| clustered | 632 | 27.03 ms | 42.30 ms | 1106.85 ms | 1.6x | 40.9x |
| thin pipe | 3402 | 100.42 ms | 165.35 ms | 1275.94 ms | 1.6x | 12.7x |
| pathological (10k, one cell) | 9999 | 28.81 ms | 60.40 ms | 346.84 ms | 2.1x | 12.0x |
<!-- bench:headline-timing:end -->

Fixed-row capacity and resident memory:

<!-- bench:headline-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| uniform | 151 | 196 | 138 | 168 |
| uniform, 1M | 757 | 952 | 606 | 813 |
| uniform, moving | 167 | 197 | 135 | 169 |
| jittered lattice | 84 | 207 | 97 | 168 |
| SPH-like slab | 151 | 195 | 160 | 234 |
| slab, moving | 167 | 196 | 160 | 236 |
| huge sparse domain | 5 | 25 | 41 | 201 |
| thin slab | 329 | 393 | 330 | 474 |
| surface | 369 | 417 | 456 | 552 |
| two-scale clusters | 2181 | 1303 | 745 | 874 |
| clustered | 1350 | 1244 | 725 | 962 |
| thin pipe | 3683 | 3728 | 3806 | 5544 |
| pathological (10k, one cell) | 492 | 501 | 1474 | 597 |
<!-- bench:headline-memory:end -->

### Stress scenes

<!-- bench:stress-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|
| clustered 500k | 646 | 73.95 ms | 127.56 ms | 2918.08 ms | 1.7x | 39.5x |
| clustered 1M | 652 | 147.00 ms | 300.50 ms | 6173.50 ms | 2.0x | 42.0x |
| two-scale 1M | 810 | 133.54 ms | 295.28 ms | 6750.99 ms | 2.2x | 50.6x |
| uniform 2M | 33 | 65.19 ms | 340.17 ms | 682.66 ms | 5.2x | 10.5x |
| 8 sets x 2.5k | 4 | 0.79 ms | 1.69 ms | 2.01 ms | 2.1x | 2.5x |
| 8 sets x 12.5k | 4 | 4.76 ms | 9.62 ms | 18.25 ms | 2.0x | 3.8x |
| 16 sets x 1.25k | 2 | 1.74 ms | 1.99 ms | 2.68 ms | 1.1x | 1.5x |
| droplets 200k | 26 | 1.73 ms | 5.00 ms | 11.46 ms | 2.9x | 6.6x |
| droplets 200k, jitter | 25 | 6.28 ms | 5.84 ms | 20.78 ms | 0.9x | 3.3x |
<!-- bench:stress-timing:end -->

<!-- bench:stress-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| clustered 500k | 1334 | 1473 | 1625 | 2433 |
| clustered 1M | 2691 | 2953 | 3323 | 4850 |
| two-scale 1M | 3295 | 3586 | 4041 | 5636 |
| uniform 2M | 492 | 806 | 1200 | 1618 |
| 8 sets x 2.5k | 25 | 52 | 20 | 44 |
| 8 sets x 12.5k | 61 | 212 | 55 | 195 |
| 16 sets x 1.25k | 50 | 77 | 24 | 80 |
| droplets 200k | 105 | 138 | 58 | 76 |
| droplets 200k, jitter | 105 | 146 | 61 | 78 |
<!-- bench:stress-memory:end -->

### Motion

<!-- bench:motion:start -->
| Scene | static | jitter | jitter vs static | maximum row, static -> jitter |
|---|---:|---:|---:|---:|
| uniform 200k | 4.22 ms | 7.83 ms | +86% | 62 -> 60 |
| slab 200k | 5.88 ms | 9.36 ms | +59% | 92 -> 95 |
| clustered 200k | 27.03 ms | 34.18 ms | +26% | 1717 -> 1694 |
| thin pipe 50k | 19.51 ms | 27.94 ms | +43% | 2475 -> 3388 |
| droplets 200k | 1.73 ms | 6.93 ms | +301% | 61 -> 57 |
<!-- bench:motion:end -->

### Saved learnsph frames

<!-- bench:learnsph:start -->
| Frame | Fluid / boundary | ispc_apfrnn | TreeNSearch | CompactNSearch | Peak RSS (ours / Tree / Compact) |
|---|---:|---:|---:|---:|---:|
| ramp_split_lv | 20k / 78k | 0.25 ms | 1.39 ms | 2.73 ms | 32 / 17 / 31 MB |
| ramp_split_hv | 29k / 78k | 0.36 ms | 1.74 ms | 4.72 ms | 32 / 28 / 40 MB |
| wall_01 | 74k / 208k | 0.87 ms | 4.34 ms | 9.62 ms | 70 / 76 / 100 MB |
| funnel | 21k / 535k | 0.28 ms | 5.04 ms | 17.84 ms | 77 / 76 / 105 MB |
| sink | 57k / 13k | 1.35 ms | 2.78 ms | 10.11 ms | 85 / 63 / 80 MB |
| bowl | 111k / 82k | 1.12 ms | 6.12 ms | 14.74 ms | 100 / 42 / 83 MB |
| wall | 107k / 208k | 1.19 ms | 5.22 ms | 19.11 ms | 109 / 59 / 102 MB |
<!-- bench:learnsph:end -->

## Benchmarks

```sh
./scripts/bench.sh build
./scripts/bench_tables.py --write
./build/bench/ispc_apfrnn_bench --benchmark_filter='^Stress/'
./build/bench/ispc_apfrnn_bench --benchmark_filter='^Api/'
```

`scripts/bench_tables.py` runs the matrix, regenerates the tables above and
updates `README.md`. Review the `git diff`: a changed row is the measured
delta for the commit. `--check` verifies the tables without rewriting them.
Pass `--gpu-json <json>` from a SYCL run to regenerate the separate GPU table.

## Development

```sh
./scripts/ci.sh
./scripts/coverage.sh
./scripts/profile.sh build 50000
./scripts/ispc_analyze.sh
```
