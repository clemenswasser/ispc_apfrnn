# ispc_apfrnn

All-pairs fixed-radius nearest neighbor search in 3D. It's a faster drop-in
alternative to [CompactNSearch] and [TreeNSearch] for SPH-style workloads,
built from ISPC kernels and TBB parallelism. The main consumer is learnsph.

[CompactNSearch]: https://github.com/InteractiveComputerGraphics/CompactNSearch
[TreeNSearch]: https://github.com/InteractiveComputerGraphics/TreeNSearch

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

## Performance

Time per frame for one search plus one density pass over the neighbor lists,
over the same fixed window of moving frames for every library; i7-12700K,
8 threads, library defaults. Lower is better. "vs Tree (search)" compares the
search alone, without the density pass.

### Simulation replays

Searches captured from SPH simulations, in the order the simulation produced.

<!-- bench:replay:start -->
| Simulation | Moving particles | Avg neighbors | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact | Memory ours / Tree / Compact (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ramp hv, t=0.10 s | 3k | 23 | 0.15 ms | 1.64 ms | n/a | 11.1x | n/a | 64 / 65 / n/a |
| ramp lv, t=0.10 s | 3k | 23 | 0.15 ms | 1.65 ms | n/a | 11.0x | n/a | 64 / 65 / n/a |
| ramp hv, t=0.30 s | 8k | 44 | 0.38 ms | 2.01 ms | 3.19 ms | 5.3x | 8.4x | 72 / 65 / 64 |
| ramp lv, t=0.30 s | 8k | 38 | 0.34 ms | 2.04 ms | 2.77 ms | 5.9x | 8.0x | 63 / 65 / 64 |
| funnel, t=0.20 s | 23k | 45 | 1.09 ms | 6.95 ms | n/a | 6.4x | n/a | 163 / 103 / n/a |
| sink, t=0.10 s | 29k | 64 | 1.06 ms | 2.05 ms | 5.53 ms | 1.9x | 5.2x | 64 / 66 / 72 |
| sink, t=0.50 s | 29k | 60 | 1.05 ms | 2.21 ms | 5.95 ms | 2.1x | 5.7x | 64 / 65 / 70 |
| funnel, t=0.60 s | 73k | 47 | 4.72 ms | 11.68 ms | n/a | 2.5x | n/a | 185 / 124 / n/a |
| wall, t=0.50 s | 74k | 61 | 3.37 ms | 7.69 ms | 25.22 ms | 2.3x | 7.5x | 144 / 104 / 97 |
| droplet, t=0.05 s | 74k | 56 | 2.26 ms | 5.25 ms | 12.17 ms | 2.3x | 5.4x | 63 / 75 / 88 |
| droplet, t=0.40 s | 74k | 57 | 2.44 ms | 6.02 ms | 12.64 ms | 2.5x | 5.2x | 63 / 77 / 88 |
| wall, t=0.10 s | 74k | 53 | 2.11 ms | 6.61 ms | 17.44 ms | 3.1x | 8.3x | 63 / 106 / 99 |
| funnel, t=0.20 s | 106k | 48 | 4.74 ms | 18.64 ms | n/a | 3.9x | n/a | 333 / 195 / n/a |
| dam break, t=0.10 s | 155k | 58 | 5.90 ms | 9.51 ms | 35.61 ms | 1.6x | 6.0x | 167 / 123 / 167 |
| dam break, t=0.40 s | 155k | 58 | 6.53 ms | 11.30 ms | 38.13 ms | 1.7x | 5.8x | 193 / 128 / 163 |
| bowl, t=0.10 s | 170k | 64 | 7.15 ms | 10.29 ms | 32.94 ms | 1.4x | 4.6x | 211 / 133 / 164 |
| bowl, t=0.50 s | 170k | 62 | 6.90 ms | 11.71 ms | 34.40 ms | 1.7x | 5.0x | 182 / 133 / 162 |
| funnel, t=0.40 s | 213k | 49 | 11.82 ms | 28.90 ms | n/a | 2.4x | n/a | 445 / 251 / n/a |
| sink, t=0.10 s | 214k | 63 | 8.39 ms | 11.63 ms | 38.87 ms | 1.4x | 4.6x | 201 / 146 / 197 |
| sink, t=0.40 s | 214k | 56 | 8.98 ms | 12.61 ms | 42.34 ms | 1.4x | 4.7x | 215 / 148 / 188 |
| droplet, t=0.05 s | 578k | 57 | 21.64 ms | 35.21 ms | 104.20 ms | 1.6x | 4.8x | 242 / 298 / 520 |
| droplet, t=0.25 s | 578k | 58 | 22.03 ms | 37.68 ms | 108.88 ms | 1.7x | 4.9x | 241 / 301 / 518 |
| wall, t=0.10 s | 578k | 54 | 19.90 ms | 37.96 ms | 124.99 ms | 1.9x | 6.3x | 397 / 365 / 651 |
| dam break, t=0.10 s | 701k | 50 | 36.55 ms | 65.97 ms | 212.00 ms | 1.8x | 5.8x | 566 / 458 / 642 |
| bowl, t=0.10 s | 1.3M | 38 | 71.68 ms | 141.37 ms | 519.00 ms | 2.0x | 7.2x | 1038 / 794 / 945 |
| wall, t=0.10 s | 1.7M | 56 | 62.44 ms | 103.98 ms | 345.76 ms | 1.7x | 5.5x | 814 / 933 / 1786 |
| dam break, t=0.10 s | 2.4M | 43 | 211.26 ms | 283.61 ms | 986.84 ms | 1.3x | 4.7x | 1955 / 1513 / 2063 |
| bowl, t=0.10 s | 3.8M | 40 | 220.17 ms | 452.94 ms | 1439.93 ms | 2.1x | 6.5x | 2894 / 2285 / 2642 |
<!-- bench:replay:end -->

### Synthetic scenes

<!-- bench:headline-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 32 | 8.02 ms | 17.22 ms | 38.87 ms | 2.1x | 4.8x |
| uniform, 1M | 33 | 44.12 ms | 94.54 ms | 225.39 ms | 2.1x | 5.1x |
| uniform, moving | 32 | 8.36 ms | 19.81 ms | 39.15 ms | 2.4x | 4.7x |
| jittered lattice | 29 | 7.67 ms | 15.17 ms | 37.70 ms | 2.0x | 4.9x |
| SPH-like slab | 58 | 10.22 ms | 20.01 ms | 56.04 ms | 2.0x | 5.5x |
| slab, moving | 58 | 10.28 ms | 20.79 ms | 55.64 ms | 2.0x | 5.4x |
| huge sparse domain | 0 | 4.09 ms | 4.98 ms | 101.63 ms | 1.2x | 24.8x |
| thin slab | 243 | 28.76 ms | 35.68 ms | 385.92 ms | 1.2x | 13.4x |
| surface | 270 | 29.10 ms | 37.64 ms | 130.86 ms | 1.3x | 4.5x |
| two-scale clusters | 575 | 54.93 ms | 59.29 ms | 551.41 ms | 1.1x | 10.0x |
| clustered | 632 | 61.17 ms | 70.88 ms | 934.93 ms | 1.2x | 15.3x |
| thin pipe | 3403 | 301.93 ms | 332.81 ms | 1410.84 ms | 1.1x | 4.7x |
| pathological (10k, one cell) | 9999 | 48.63 ms | 90.05 ms | 424.92 ms | 1.9x | 8.7x |
<!-- bench:headline-timing:end -->

<!-- bench:headline-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| uniform | 92 | 150 | 168 | 175 |
| uniform, 1M | 462 | 790 | 628 | 842 |
| uniform, moving | 80 | 150 | 175 | 175 |
| jittered lattice | 80 | 169 | 151 | 175 |
| SPH-like slab | 129 | 204 | 187 | 234 |
| slab, moving | 105 | 204 | 194 | 233 |
| huge sparse domain | 5 | 150 | 151 | 227 |
| thin slab | 356 | 393 | 344 | 453 |
| surface | 411 | 408 | 440 | 534 |
| two-scale clusters | 665 | 669 | 695 | 803 |
| clustered | 714 | 711 | 693 | 871 |
| thin pipe | 3744 | 3759 | 3865 | 5447 |
| pathological (10k, one cell) | 491 | 551 | 2457 | 604 |
<!-- bench:headline-memory:end -->

### Stress scenes

<!-- bench:stress-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|
| clustered 500k | 646 | 153.45 ms | 184.38 ms | 2369.19 ms | 1.2x | 15.4x |
| clustered 1M | 652 | 311.91 ms | 383.06 ms | 4834.39 ms | 1.2x | 15.5x |
| two-scale 1M | 810 | 342.24 ms | 385.49 ms | 3731.80 ms | 1.1x | 10.9x |
| uniform 2M | 33 | 92.85 ms | 145.39 ms | 451.57 ms | 1.6x | 4.9x |
| 8 sets x 2.5k | 4 | 1.75 ms | 2.74 ms | 3.74 ms | 1.6x | 2.1x |
| 8 sets x 12.5k | 4 | 9.94 ms | 11.07 ms | 23.95 ms | 1.1x | 2.4x |
| 16 sets x 1.25k | 2 | 5.88 ms | 5.38 ms | 6.41 ms | 0.9x | 1.1x |
| droplets 200k | 26 | 6.00 ms | 6.49 ms | 24.19 ms | 1.1x | 4.0x |
| droplets 200k, jitter | 25 | 6.18 ms | 6.32 ms | 29.50 ms | 1.0x | 4.8x |
<!-- bench:stress-timing:end -->

<!-- bench:stress-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| clustered 500k | 1835 | 1760 | 1652 | 2181 |
| clustered 1M | 3698 | 3490 | 3327 | 4381 |
| two-scale 1M | 4704 | 4432 | 4114 | 5064 |
| uniform 2M | 924 | 1255 | 826 | 1626 |
| 8 sets x 2.5k | 29 | 150 | 150 | 150 |
| 8 sets x 12.5k | 82 | 228 | 150 | 202 |
| 16 sets x 1.25k | 59 | 150 | 150 | 150 |
| droplets 200k | 56 | 150 | 150 | 151 |
| droplets 200k, jitter | 56 | 151 | 150 | 150 |
<!-- bench:stress-memory:end -->

### Scaling

<!-- bench:scale:start -->
| Distribution | Points | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 175k | 32 | 7.12 ms | 15.08 ms | 32.24 ms | 2.1x | 4.5x |
| uniform | 500k | 33 | 20.47 ms | 43.46 ms | 104.98 ms | 2.1x | 5.1x |
| uniform | 1M | 33 | 43.27 ms | 93.53 ms | 218.45 ms | 2.2x | 5.0x |
| uniform | 2M | 33 | 96.52 ms | 143.33 ms | 450.13 ms | 1.5x | 4.7x |
| uniform | 4M | 33 | 172.01 ms | 410.29 ms | 946.89 ms | 2.4x | 5.5x |
| uniform | 8M | 33 | 349.43 ms | 826.80 ms | 1916.25 ms | 2.4x | 5.5x |
| sparse | 175k | 0 | 3.55 ms | 4.18 ms | 92.49 ms | 1.2x | 26.1x |
| sparse | 500k | 0 | 10.81 ms | 14.23 ms | 231.57 ms | 1.3x | 21.4x |
| sparse | 1M | 0 | 14.65 ms | 17.76 ms | 567.26 ms | 1.2x | 38.7x |
| sparse | 2M | 0 | 24.65 ms | 41.26 ms | 1218.50 ms | 1.7x | 49.4x |
| sparse | 4M | 0 | 51.23 ms | 110.08 ms | 2519.24 ms | 2.1x | 49.2x |
| sparse | 8M | 0 | 114.33 ms | 152.58 ms | 5170.79 ms | 1.3x | 45.2x |
<!-- bench:scale:end -->

### Growing sets

<!-- bench:growing:start -->
| Scene | ispc_apfrnn | TreeNSearch | vs Tree |
|---|---:|---:|---:|
| uniform 200k -> 400k | 30.26 ms | 51.02 ms | 1.7x |
| slab 200k -> 400k | 49.39 ms | 58.30 ms | 1.2x |
| clustered 200k -> 400k | 87.03 ms | 90.14 ms | 1.0x |
<!-- bench:growing:end -->
