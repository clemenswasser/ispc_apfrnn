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
| Simulation | Moving particles | Avg neighbors | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact | Memory ours / Tree / Compact (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ramp hv, t=0.10 s | 3k | 23 | 0.14 ms | 1.63 ms | n/a | 11.7x | 21.0x | n/a | 64 / 65 / n/a |
| ramp lv, t=0.10 s | 3k | 23 | 0.15 ms | 1.65 ms | n/a | 11.0x | 19.5x | n/a | 65 / 65 / n/a |
| ramp hv, t=0.30 s | 8k | 44 | 0.38 ms | 1.98 ms | 3.25 ms | 5.3x | 7.8x | 8.6x | 82 / 65 / 64 |
| ramp lv, t=0.30 s | 8k | 38 | 0.34 ms | 1.98 ms | 2.63 ms | 5.8x | 8.6x | 7.7x | 63 / 65 / 64 |
| funnel, t=0.20 s | 23k | 45 | 1.06 ms | 6.99 ms | n/a | 6.6x | 8.9x | n/a | 179 / 103 / n/a |
| sink, t=0.10 s | 29k | 64 | 1.08 ms | 2.09 ms | 5.44 ms | 1.9x | 2.6x | 5.1x | 64 / 66 / 72 |
| sink, t=0.50 s | 29k | 60 | 1.13 ms | 2.17 ms | 5.79 ms | 1.9x | 2.5x | 5.1x | 64 / 65 / 70 |
| funnel, t=0.60 s | 73k | 47 | 4.60 ms | 11.67 ms | n/a | 2.5x | 3.6x | n/a | 184 / 125 / n/a |
| wall, t=0.50 s | 74k | 61 | 3.32 ms | 7.53 ms | 25.35 ms | 2.3x | 3.5x | 7.6x | 144 / 104 / 97 |
| droplet, t=0.05 s | 74k | 56 | 2.26 ms | 5.07 ms | 11.55 ms | 2.2x | 3.1x | 5.1x | 63 / 75 / 89 |
| droplet, t=0.40 s | 74k | 57 | 2.42 ms | 6.06 ms | 12.53 ms | 2.5x | 3.6x | 5.2x | 63 / 77 / 88 |
| wall, t=0.10 s | 74k | 53 | 2.06 ms | 6.35 ms | 17.51 ms | 3.1x | 4.3x | 8.5x | 63 / 105 / 99 |
| funnel, t=0.20 s | 106k | 48 | 4.56 ms | 18.47 ms | n/a | 4.1x | 5.3x | n/a | 358 / 191 / n/a |
| dam break, t=0.10 s | 155k | 58 | 5.81 ms | 9.29 ms | 35.19 ms | 1.6x | 2.2x | 6.1x | 167 / 123 / 167 |
| dam break, t=0.40 s | 155k | 58 | 6.51 ms | 11.02 ms | 38.60 ms | 1.7x | 2.4x | 5.9x | 193 / 127 / 163 |
| bowl, t=0.10 s | 170k | 64 | 6.96 ms | 10.46 ms | 33.16 ms | 1.5x | 2.0x | 4.8x | 210 / 133 / 164 |
| bowl, t=0.50 s | 170k | 62 | 6.90 ms | 11.69 ms | 34.04 ms | 1.7x | 2.4x | 4.9x | 205 / 132 / 162 |
| funnel, t=0.40 s | 213k | 49 | 11.71 ms | 28.59 ms | n/a | 2.4x | 3.1x | n/a | 445 / 251 / n/a |
| sink, t=0.10 s | 214k | 63 | 8.46 ms | 11.20 ms | 39.01 ms | 1.3x | 1.6x | 4.6x | 201 / 142 / 198 |
| sink, t=0.40 s | 214k | 56 | 8.90 ms | 12.50 ms | 45.45 ms | 1.4x | 1.9x | 5.1x | 215 / 147 / 188 |
| droplet, t=0.05 s | 578k | 57 | 21.95 ms | 34.94 ms | 105.73 ms | 1.6x | 2.1x | 4.8x | 242 / 290 / 519 |
| droplet, t=0.25 s | 578k | 58 | 22.27 ms | 37.55 ms | 109.72 ms | 1.7x | 2.3x | 4.9x | 241 / 297 / 517 |
| wall, t=0.10 s | 578k | 54 | 19.72 ms | 36.74 ms | 125.39 ms | 1.9x | 2.4x | 6.4x | 397 / 363 / 650 |
| dam break, t=0.10 s | 701k | 50 | 35.93 ms | 66.01 ms | 205.39 ms | 1.8x | 2.8x | 5.7x | 633 / 455 / 642 |
| bowl, t=0.10 s | 1.3M | 38 | 65.48 ms | 132.24 ms | 459.55 ms | 2.0x | 3.6x | 7.0x | 1038 / 783 / 951 |
| wall, t=0.10 s | 1.7M | 56 | 59.21 ms | 102.19 ms | 344.33 ms | 1.7x | 2.2x | 5.8x | 1004 / 949 / 1786 |
| dam break, t=0.10 s | 2.4M | 43 | 190.73 ms | 277.20 ms | 965.59 ms | 1.5x | 2.2x | 5.1x | 2081 / 1489 / 2063 |
| bowl, t=0.10 s | 3.8M | 40 | 218.84 ms | 440.35 ms | 1425.59 ms | 2.0x | 3.8x | 6.5x | 2849 / 2340 / 2642 |
<!-- bench:replay:end -->

### Synthetic scenes

<!-- bench:headline-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 32 | 8.23 ms | 17.57 ms | 39.79 ms | 2.1x | 2.9x | 4.8x |
| uniform, 1M | 33 | 45.79 ms | 97.58 ms | 224.40 ms | 2.1x | 2.7x | 4.9x |
| uniform, moving | 32 | 8.14 ms | 18.62 ms | 39.98 ms | 2.3x | 3.2x | 4.9x |
| jittered lattice | 29 | 7.23 ms | 14.08 ms | 37.81 ms | 1.9x | 2.5x | 5.2x |
| SPH-like slab | 58 | 10.22 ms | 20.15 ms | 56.83 ms | 2.0x | 2.5x | 5.6x |
| slab, moving | 58 | 10.35 ms | 21.55 ms | 56.27 ms | 2.1x | 2.7x | 5.4x |
| huge sparse domain | 0 | 4.08 ms | 5.06 ms | 177.00 ms | 1.2x | 1.3x | 43.3x |
| thin slab | 243 | 29.65 ms | 37.47 ms | 389.27 ms | 1.3x | 1.2x | 13.1x |
| surface | 270 | 29.62 ms | 41.77 ms | 139.57 ms | 1.4x | 1.3x | 4.7x |
| two-scale clusters | 575 | 54.18 ms | 59.29 ms | 557.05 ms | 1.1x | 1.2x | 10.3x |
| clustered | 632 | 62.55 ms | 76.14 ms | 950.96 ms | 1.2x | 1.3x | 15.2x |
| thin pipe | 3403 | 290.44 ms | 334.42 ms | 1437.99 ms | 1.2x | 1.4x | 5.0x |
| pathological (10k, one cell) | 9999 | 49.42 ms | 124.11 ms | 402.39 ms | 2.5x | 5.5x | 8.1x |
<!-- bench:headline-timing:end -->

<!-- bench:headline-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| uniform | 80 | 150 | 150 | 175 |
| uniform, 1M | 645 | 790 | 497 | 842 |
| uniform, moving | 80 | 150 | 152 | 175 |
| jittered lattice | 62 | 150 | 150 | 175 |
| SPH-like slab | 105 | 167 | 155 | 234 |
| slab, moving | 105 | 167 | 170 | 233 |
| huge sparse domain | 5 | 150 | 150 | 353 |
| thin slab | 356 | 392 | 307 | 455 |
| surface | 344 | 406 | 378 | 517 |
| two-scale clusters | 665 | 669 | 630 | 803 |
| clustered | 714 | 711 | 624 | 872 |
| thin pipe | 3695 | 3762 | 3838 | 5457 |
| pathological (10k, one cell) | 491 | 551 | 2457 | 596 |
<!-- bench:headline-memory:end -->

### Stress scenes

<!-- bench:stress-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|
| clustered 500k | 646 | 155.42 ms | 184.84 ms | 2367.12 ms | 1.2x | 1.3x | 15.2x |
| clustered 1M | 652 | 315.80 ms | 388.33 ms | 4812.79 ms | 1.2x | 1.4x | 15.2x |
| two-scale 1M | 810 | 337.11 ms | 392.56 ms | 3678.75 ms | 1.2x | 1.3x | 10.9x |
| uniform 2M | 33 | 85.95 ms | 122.02 ms | 472.53 ms | 1.4x | 1.6x | 5.5x |
| 8 sets x 2.5k | 4 | 1.99 ms | 2.84 ms | 3.61 ms | 1.4x | 2.4x | 1.8x |
| 8 sets x 12.5k | 4 | 10.23 ms | 11.00 ms | 25.38 ms | 1.1x | 1.3x | 2.5x |
| 16 sets x 1.25k | 2 | 5.93 ms | 5.44 ms | 6.78 ms | 0.9x | 1.4x | 1.1x |
| droplets 200k | 26 | 6.21 ms | 6.84 ms | 21.65 ms | 1.1x | 1.2x | 3.5x |
| droplets 200k, jitter | 25 | 6.39 ms | 6.45 ms | 26.74 ms | 1.0x | 1.1x | 4.2x |
<!-- bench:stress-timing:end -->

<!-- bench:stress-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| clustered 500k | 1835 | 1753 | 1629 | 2183 |
| clustered 1M | 3698 | 3488 | 3125 | 4383 |
| two-scale 1M | 4704 | 4426 | 4040 | 5065 |
| uniform 2M | 863 | 1254 | 704 | 1626 |
| 8 sets x 2.5k | 29 | 150 | 150 | 150 |
| 8 sets x 12.5k | 82 | 204 | 150 | 202 |
| 16 sets x 1.25k | 59 | 150 | 150 | 150 |
| droplets 200k | 56 | 150 | 150 | 150 |
| droplets 200k, jitter | 56 | 150 | 150 | 150 |
<!-- bench:stress-memory:end -->

### Scaling

<!-- bench:scale:start -->
| Distribution | Points | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 175k | 32 | 7.26 ms | 15.64 ms | 34.55 ms | 2.2x | 3.0x | 4.8x |
| uniform | 500k | 33 | 21.17 ms | 42.77 ms | 108.11 ms | 2.0x | 2.6x | 5.1x |
| uniform | 1M | 33 | 47.49 ms | 100.09 ms | 229.19 ms | 2.1x | 2.6x | 4.8x |
| uniform | 2M | 33 | 86.55 ms | 121.17 ms | 470.24 ms | 1.4x | 1.6x | 5.4x |
| uniform | 4M | 33 | 180.03 ms | 424.72 ms | 985.10 ms | 2.4x | 3.1x | 5.5x |
| uniform | 8M | 33 | 366.77 ms | 845.25 ms | 1965.95 ms | 2.3x | 2.9x | 5.4x |
| sparse | 175k | 0 | 3.55 ms | 4.27 ms | 149.43 ms | 1.2x | 1.3x | 42.1x |
| sparse | 500k | 0 | 10.97 ms | 13.61 ms | 310.74 ms | 1.2x | 1.3x | 28.3x |
| sparse | 1M | 0 | 15.12 ms | 18.31 ms | 625.85 ms | 1.2x | 1.4x | 41.4x |
| sparse | 2M | 0 | 27.92 ms | 42.50 ms | 1256.35 ms | 1.5x | 1.8x | 45.0x |
| sparse | 4M | 0 | 50.56 ms | 118.60 ms | 2770.13 ms | 2.3x | 3.0x | 54.8x |
| sparse | 8M | 0 | 98.42 ms | 156.24 ms | 5528.23 ms | 1.6x | 2.1x | 56.2x |
<!-- bench:scale:end -->

### Growing sets

<!-- bench:growing:start -->
| Scene | ispc_apfrnn | TreeNSearch | vs Tree | vs Tree (search) |
|---|---:|---:|---:|---:|
| uniform 200k -> 400k | 27.11 ms | 40.74 ms | 1.5x | 1.9x |
| slab 200k -> 400k | 47.16 ms | 47.13 ms | 1.0x | 1.0x |
| clustered 200k -> 400k | 90.53 ms | 96.98 ms | 1.1x | 1.1x |
<!-- bench:growing:end -->
