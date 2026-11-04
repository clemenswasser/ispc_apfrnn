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
| uniform | 32 | 4.36 ms | 18.80 ms | 28.47 ms | 4.3x | 6.5x |
| uniform, 1M | 33 | 25.29 ms | 122.39 ms | 271.67 ms | 4.8x | 10.7x |
| uniform, moving | 32 | 5.84 ms | 19.12 ms | 31.19 ms | 3.3x | 5.3x |
| jittered lattice | 29 | 3.90 ms | 8.80 ms | 22.33 ms | 2.3x | 5.7x |
| SPH-like slab | 58 | 5.90 ms | 19.35 ms | 40.77 ms | 3.3x | 6.9x |
| slab, moving | 58 | 7.30 ms | 19.76 ms | 43.36 ms | 2.7x | 5.9x |
| huge sparse domain | 0 | 1.29 ms | 4.54 ms | 31.81 ms | 3.5x | 24.6x |
| thin slab | 243 | 14.42 ms | 24.06 ms | 210.58 ms | 1.7x | 14.6x |
| surface | 270 | 13.47 ms | 21.30 ms | 81.54 ms | 1.6x | 6.1x |
| two-scale clusters | 575 | 20.16 ms | 36.09 ms | 649.13 ms | 1.8x | 32.2x |
| clustered | 632 | 26.82 ms | 42.49 ms | 1051.50 ms | 1.6x | 39.2x |
| thin pipe | 3402 | 91.13 ms | 154.85 ms | 1254.53 ms | 1.7x | 13.8x |
| pathological (10k, one cell) | 9999 | 26.92 ms | 66.84 ms | 324.76 ms | 2.5x | 12.1x |
<!-- bench:headline-timing:end -->

Fixed-row capacity and resident memory:

<!-- bench:headline-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| uniform | 103 | 150 | 132 | 168 |
| uniform, 1M | 513 | 723 | 583 | 812 |
| uniform, moving | 118 | 152 | 132 | 169 |
| jittered lattice | 84 | 132 | 89 | 168 |
| SPH-like slab | 127 | 174 | 144 | 235 |
| slab, moving | 143 | 174 | 148 | 236 |
| huge sparse domain | 5 | 26 | 40 | 200 |
| thin slab | 311 | 376 | 322 | 474 |
| surface | 350 | 407 | 424 | 552 |
| two-scale clusters | 480 | 552 | 731 | 874 |
| clustered | 523 | 604 | 680 | 963 |
| thin pipe | 3664 | 3711 | 3782 | 5526 |
| pathological (10k, one cell) | 490 | 499 | 2519 | 597 |
<!-- bench:headline-memory:end -->

### Stress scenes

<!-- bench:stress-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|
| clustered 500k | 646 | 67.64 ms | 118.03 ms | 2710.01 ms | 1.7x | 40.1x |
| clustered 1M | 652 | 134.44 ms | 285.26 ms | 5772.57 ms | 2.1x | 42.9x |
| two-scale 1M | 810 | 122.24 ms | 278.75 ms | 6422.62 ms | 2.3x | 52.5x |
| uniform 2M | 33 | 54.02 ms | 318.41 ms | 640.01 ms | 5.9x | 11.8x |
| 8 sets x 2.5k | 4 | 0.76 ms | 1.66 ms | 1.97 ms | 2.2x | 2.6x |
| 8 sets x 12.5k | 4 | 4.71 ms | 9.45 ms | 18.20 ms | 2.0x | 3.9x |
| 16 sets x 1.25k | 2 | 1.82 ms | 1.98 ms | 2.61 ms | 1.1x | 1.4x |
| droplets 200k | 26 | 1.71 ms | 4.77 ms | 11.41 ms | 2.8x | 6.7x |
| droplets 200k, jitter | 25 | 5.52 ms | 5.80 ms | 20.17 ms | 1.0x | 3.7x |
<!-- bench:stress-timing:end -->

<!-- bench:stress-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| clustered 500k | 1334 | 1473 | 1599 | 2435 |
| clustered 1M | 2691 | 2955 | 3108 | 4859 |
| two-scale 1M | 3295 | 3589 | 3882 | 5635 |
| uniform 2M | 492 | 807 | 1184 | 1618 |
| 8 sets x 2.5k | 25 | 44 | 19 | 45 |
| 8 sets x 12.5k | 61 | 179 | 52 | 195 |
| 16 sets x 1.25k | 50 | 70 | 23 | 80 |
| droplets 200k | 56 | 91 | 44 | 76 |
| droplets 200k, jitter | 56 | 103 | 49 | 78 |
<!-- bench:stress-memory:end -->

### Motion

<!-- bench:motion:start -->
| Scene | static | jitter | jitter vs static | maximum row, static -> jitter |
|---|---:|---:|---:|---:|
| uniform 200k | 4.35 ms | 6.70 ms | +54% | 62 -> 60 |
| slab 200k | 5.92 ms | 7.98 ms | +35% | 92 -> 95 |
| clustered 200k | 26.64 ms | 34.50 ms | +30% | 1717 -> 1689 |
| thin pipe 50k | 16.01 ms | 28.81 ms | +80% | 2475 -> 3432 |
| droplets 200k | 1.71 ms | 5.51 ms | +223% | 61 -> 58 |
<!-- bench:motion:end -->

### Saved learnsph frames

<!-- bench:learnsph:start -->
| Frame | Fluid / boundary | ispc_apfrnn | TreeNSearch | CompactNSearch | Peak RSS (ours / Tree / Compact) |
|---|---:|---:|---:|---:|---:|
| ramp_split_lv | 20k / 78k | 0.26 ms | 1.33 ms | 2.82 ms | 27 / 16 / 31 MB |
| ramp_split_hv | 29k / 78k | 0.38 ms | 1.74 ms | 4.84 ms | 27 / 27 / 40 MB |
| wall_01 | 74k / 208k | 0.80 ms | 4.07 ms | 9.33 ms | 50 / 70 / 100 MB |
| funnel | 21k / 535k | 0.28 ms | 4.85 ms | 17.32 ms | 72 / 75 / 105 MB |
| sink | 57k / 13k | 1.31 ms | 2.65 ms | 10.23 ms | 57 / 49 / 79 MB |
| bowl | 111k / 82k | 0.96 ms | 5.99 ms | 14.79 ms | 58 / 40 / 83 MB |
| wall | 107k / 208k | 0.99 ms | 4.87 ms | 18.07 ms | 68 / 56 / 101 MB |
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
