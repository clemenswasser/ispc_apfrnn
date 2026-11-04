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
  kernel.ispc           SIMD kernels (no raw intrinsics)
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
- `set_memory_budget(bytes)` caps one fixed-row allocation (default 3 GiB).
  Larger requests fall back to CSR rows.
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
| uniform | 32 | 4.46 ms | 20.10 ms | 30.11 ms | 4.5x | 6.8x |
| uniform, 1M | 33 | 26.07 ms | 126.95 ms | 286.15 ms | 4.9x | 11.0x |
| uniform, moving | 32 | 6.49 ms | 19.84 ms | 34.79 ms | 3.1x | 5.4x |
| jittered lattice | 29 | 4.16 ms | 9.31 ms | 22.89 ms | 2.2x | 5.5x |
| SPH-like slab | 58 | 6.15 ms | 20.01 ms | 42.29 ms | 3.3x | 6.9x |
| slab, moving | 58 | 8.25 ms | 20.31 ms | 45.47 ms | 2.5x | 5.5x |
| huge sparse domain | 0 | 1.40 ms | 4.76 ms | 35.46 ms | 3.4x | 25.3x |
| thin slab | 243 | 14.95 ms | 27.42 ms | 229.37 ms | 1.8x | 15.3x |
| surface | 270 | 14.77 ms | 23.33 ms | 83.78 ms | 1.6x | 5.7x |
| two-scale clusters | 575 | 20.68 ms | 38.87 ms | 689.52 ms | 1.9x | 33.3x |
| clustered | 632 | 27.58 ms | 44.31 ms | 1118.14 ms | 1.6x | 40.5x |
| thin pipe | 3402 | 98.61 ms | 158.94 ms | 1274.20 ms | 1.6x | 12.9x |
| pathological (10k, one cell) | 9999 | 28.03 ms | 63.85 ms | 346.63 ms | 2.3x | 12.4x |
<!-- bench:headline-timing:end -->

Fixed-row capacity and resident memory:

<!-- bench:headline-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| uniform | 167 | 223 | 131 | 168 |
| uniform, 1M | 835 | 937 | 575 | 812 |
| uniform, moving | 167 | 194 | 131 | 169 |
| jittered lattice | 100 | 223 | 94 | 168 |
| SPH-like slab | 167 | 192 | 145 | 235 |
| slab, moving | 167 | 193 | 156 | 236 |
| huge sparse domain | 5 | 26 | 43 | 201 |
| thin slab | 362 | 391 | 304 | 474 |
| surface | 387 | 415 | 397 | 552 |
| two-scale clusters | 2242 | 1331 | 706 | 875 |
| clustered | 1412 | 1274 | 684 | 963 |
| thin pipe | 3683 | 3728 | 3861 | 5576 |
| pathological (10k, one cell) | 492 | 501 | 1542 | 596 |
<!-- bench:headline-memory:end -->

### Stress scenes

<!-- bench:stress-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|
| clustered 500k | 646 | 71.77 ms | 122.41 ms | 2891.71 ms | 1.7x | 40.3x |
| clustered 1M | 652 | 145.13 ms | 322.59 ms | 6096.93 ms | 2.2x | 42.0x |
| two-scale 1M | 810 | 132.98 ms | 287.64 ms | 6788.74 ms | 2.2x | 51.0x |
| uniform 2M | 33 | 64.27 ms | 326.73 ms | 675.64 ms | 5.1x | 10.5x |
| 8 sets x 2.5k | 4 | 0.76 ms | 1.72 ms | 2.11 ms | 2.3x | 2.8x |
| 8 sets x 12.5k | 4 | 5.64 ms | 9.71 ms | 19.04 ms | 1.7x | 3.4x |
| 16 sets x 1.25k | 2 | 1.79 ms | 2.06 ms | 2.78 ms | 1.1x | 1.6x |
| droplets 200k | 26 | 1.85 ms | 5.04 ms | 11.77 ms | 2.7x | 6.4x |
| droplets 200k, jitter | 25 | 6.40 ms | 5.96 ms | 21.98 ms | 0.9x | 3.4x |
<!-- bench:stress-timing:end -->

<!-- bench:stress-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| clustered 500k | 1490 | 1497 | 1611 | 2434 |
| clustered 1M | 3002 | 2997 | 3102 | 4858 |
| two-scale 1M | 3847 | 3798 | 4041 | 5635 |
| uniform 2M | 492 | 806 | 1184 | 1618 |
| 8 sets x 2.5k | 25 | 52 | 18 | 44 |
| 8 sets x 12.5k | 125 | 205 | 54 | 194 |
| 16 sets x 1.25k | 50 | 75 | 23 | 80 |
| droplets 200k | 105 | 138 | 56 | 76 |
| droplets 200k, jitter | 105 | 144 | 53 | 79 |
<!-- bench:stress-memory:end -->

### Motion

<!-- bench:motion:start -->
| Scene | static | jitter | jitter vs static | maximum row, static -> jitter |
|---|---:|---:|---:|---:|
| uniform 200k | 4.40 ms | 7.18 ms | +63% | 62 -> 62 |
| slab 200k | 6.18 ms | 8.79 ms | +42% | 92 -> 93 |
| clustered 200k | 27.54 ms | 33.37 ms | +21% | 1717 -> 1684 |
| thin pipe 50k | 16.76 ms | 26.35 ms | +57% | 2475 -> 3454 |
| droplets 200k | 1.83 ms | 6.40 ms | +250% | 61 -> 65 |
<!-- bench:motion:end -->

### Saved learnsph frames

<!-- bench:learnsph:start -->
| Frame | Fluid / boundary | ispc_apfrnn | TreeNSearch | CompactNSearch | Peak RSS (ours / Tree / Compact) |
|---|---:|---:|---:|---:|---:|
| ramp_split_lv | 20k / 78k | 0.24 ms | 1.35 ms | 2.73 ms | 38 / 17 / 31 MB |
| ramp_split_hv | 29k / 78k | 0.37 ms | 1.76 ms | 4.78 ms | 32 / 28 / 40 MB |
| wall_01 | 74k / 208k | 0.87 ms | 4.45 ms | 9.57 ms | 69 / 73 / 100 MB |
| funnel | 21k / 535k | 0.28 ms | 5.18 ms | 18.67 ms | 78 / 76 / 104 MB |
| sink | 57k / 13k | 1.49 ms | 2.72 ms | 10.07 ms | 85 / 59 / 80 MB |
| bowl | 111k / 82k | 1.40 ms | 6.14 ms | 14.62 ms | 126 / 42 / 83 MB |
| wall | 107k / 208k | 1.46 ms | 5.10 ms | 17.85 ms | 139 / 58 / 102 MB |
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
