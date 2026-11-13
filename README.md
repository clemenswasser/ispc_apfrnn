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
- To read many lists of one pair, take `auto pair = search.neighbors(fluid,
  boundary);` once and read `pair[i]`. It skips the per-call pair lookup of
  `get_neighbors` and, like the returned lists, stays valid until the next
  call that changes the search or `run()`.
- For large moving sets, keep your particles in the search's order: every
  few dozen steps call `search.prepare_zsort()`, then
  `search.apply_zsort(set, data, stride)` on the positions (`stride` 3 for
  interleaved xyz) and every other per-point array of the set, then `run()`.
  Point indices change, so earlier lists no longer apply.
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
| ramp hv, t=0.10 s | 3k | 23 | 0.13 ms | 1.59 ms | n/a | 12.5x | 21.8x | n/a | 64 / 65 / n/a |
| ramp lv, t=0.10 s | 3k | 23 | 0.13 ms | 1.57 ms | n/a | 12.1x | 21.1x | n/a | 64 / 65 / n/a |
| ramp hv, t=0.30 s | 8k | 44 | 0.22 ms | 0.94 ms | 3.36 ms | 4.2x | 5.3x | 15.2x | 78 / 66 / 64 |
| ramp lv, t=0.30 s | 8k | 38 | 0.21 ms | 0.89 ms | 2.87 ms | 4.2x | 5.2x | 13.6x | 63 / 66 / 64 |
| funnel, t=0.20 s | 23k | 45 | 0.77 ms | 6.62 ms | n/a | 8.6x | 12.0x | n/a | 152 / 104 / n/a |
| sink, t=0.10 s | 29k | 64 | 0.72 ms | 1.47 ms | 5.23 ms | 2.1x | 2.6x | 7.3x | 65 / 67 / 74 |
| sink, t=0.50 s | 29k | 60 | 0.78 ms | 1.39 ms | 5.42 ms | 1.8x | 2.2x | 7.0x | 65 / 66 / 73 |
| funnel, t=0.60 s | 73k | 47 | 2.84 ms | 10.99 ms | n/a | 3.9x | 5.6x | n/a | 146 / 131 / n/a |
| wall, t=0.50 s | 74k | 61 | 1.98 ms | 4.02 ms | 25.63 ms | 2.0x | 2.7x | 13.0x | 69 / 104 / 111 |
| droplet, t=0.05 s | 74k | 56 | 1.61 ms | 3.30 ms | 11.90 ms | 2.1x | 2.7x | 7.4x | 63 / 78 / 92 |
| droplet, t=0.40 s | 74k | 57 | 1.64 ms | 3.38 ms | 12.07 ms | 2.1x | 2.8x | 7.3x | 63 / 78 / 92 |
| wall, t=0.10 s | 74k | 53 | 1.56 ms | 3.97 ms | 21.16 ms | 2.5x | 3.6x | 13.5x | 63 / 107 / 103 |
| funnel, t=0.20 s | 106k | 48 | 2.88 ms | 17.48 ms | n/a | 6.1x | 8.4x | n/a | 286 / 199 / n/a |
| dam break, t=0.10 s | 155k | 58 | 3.54 ms | 6.64 ms | 37.54 ms | 1.9x | 2.2x | 10.6x | 99 / 123 / 176 |
| dam break, t=0.40 s | 155k | 58 | 4.24 ms | 8.53 ms | 39.61 ms | 2.0x | 2.5x | 9.3x | 100 / 137 / 177 |
| bowl, t=0.10 s | 170k | 64 | 4.26 ms | 7.27 ms | 36.36 ms | 1.7x | 1.9x | 8.5x | 108 / 131 / 175 |
| bowl, t=0.50 s | 170k | 62 | 4.24 ms | 6.91 ms | 36.21 ms | 1.6x | 1.8x | 8.5x | 106 / 129 / 175 |
| funnel, t=0.40 s | 213k | 49 | 7.01 ms | 26.34 ms | n/a | 3.8x | 4.9x | n/a | 333 / 244 / n/a |
| sink, t=0.10 s | 214k | 63 | 5.38 ms | 10.71 ms | 41.14 ms | 2.0x | 2.4x | 7.7x | 117 / 151 / 212 |
| sink, t=0.40 s | 214k | 56 | 5.36 ms | 9.57 ms | 42.09 ms | 1.8x | 2.0x | 7.8x | 112 / 140 / 210 |
| droplet, t=0.05 s | 578k | 57 | 14.58 ms | 23.48 ms | 100.01 ms | 1.6x | 1.7x | 6.9x | 205 / 310 / 540 |
| droplet, t=0.25 s | 578k | 58 | 14.53 ms | 23.27 ms | 101.58 ms | 1.6x | 1.7x | 7.0x | 204 / 314 / 541 |
| wall, t=0.10 s | 578k | 54 | 13.24 ms | 27.74 ms | 144.07 ms | 2.1x | 2.4x | 10.9x | 278 / 409 / 668 |
| dam break, t=0.10 s | 701k | 50 | 17.15 ms | 36.67 ms | 179.27 ms | 2.1x | 2.5x | 10.5x | 287 / 478 / 734 |
| bowl, t=0.10 s | 1.3M | 38 | 26.77 ms | 63.89 ms | 280.77 ms | 2.4x | 2.9x | 10.5x | 398 / 805 / 1163 |
| wall, t=0.10 s | 1.7M | 56 | 40.46 ms | 83.47 ms | 394.40 ms | 2.1x | 2.4x | 9.7x | 711 / 1054 / 1857 |
| dam break, t=0.10 s | 2.4M | 43 | 61.38 ms | 129.83 ms | 570.03 ms | 2.1x | 2.5x | 9.3x | 855 / 1561 / 2366 |
| bowl, t=0.10 s | 3.8M | 40 | 80.26 ms | 186.78 ms | 844.37 ms | 2.3x | 2.8x | 10.5x | 1077 / 2372 / 3255 |
<!-- bench:replay:end -->

### Synthetic scenes

<!-- bench:headline-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 32 | 5.28 ms | 17.26 ms | 35.35 ms | 3.3x | 4.2x | 6.7x |
| uniform, 1M | 33 | 28.08 ms | 92.10 ms | 200.69 ms | 3.3x | 4.1x | 7.1x |
| uniform, moving | 32 | 5.74 ms | 17.60 ms | 35.00 ms | 3.1x | 4.0x | 6.1x |
| jittered lattice | 29 | 4.86 ms | 13.24 ms | 33.61 ms | 2.7x | 3.4x | 6.9x |
| SPH-like slab | 58 | 7.02 ms | 18.31 ms | 56.92 ms | 2.6x | 3.2x | 8.1x |
| slab, moving | 58 | 7.04 ms | 19.68 ms | 54.92 ms | 2.8x | 3.5x | 7.8x |
| huge sparse domain | 0 | 2.16 ms | 4.43 ms | 163.95 ms | 2.1x | 2.1x | 76.0x |
| thin slab | 243 | 23.66 ms | 33.16 ms | 385.48 ms | 1.4x | 1.4x | 16.3x |
| surface | 270 | 24.41 ms | 36.50 ms | 118.35 ms | 1.5x | 1.4x | 4.8x |
| two-scale clusters | 575 | 40.98 ms | 54.69 ms | 544.70 ms | 1.3x | 1.5x | 13.3x |
| clustered | 632 | 45.23 ms | 63.56 ms | 930.31 ms | 1.4x | 1.6x | 20.6x |
| thin pipe | 3403 | 224.76 ms | 289.40 ms | 1246.48 ms | 1.3x | 1.7x | 5.5x |
| pathological (10k, one cell) | 9999 | 31.99 ms | 92.20 ms | 387.03 ms | 2.9x | 5.5x | 12.1x |
<!-- bench:headline-timing:end -->

<!-- bench:headline-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| uniform | 89 | 151 | 151 | 175 |
| uniform, 1M | 313 | 338 | 482 | 842 |
| uniform, moving | 89 | 150 | 152 | 175 |
| jittered lattice | 69 | 151 | 150 | 175 |
| SPH-like slab | 116 | 151 | 152 | 235 |
| slab, moving | 116 | 150 | 174 | 232 |
| huge sparse domain | 5 | 151 | 151 | 353 |
| thin slab | 356 | 392 | 289 | 455 |
| surface | 344 | 406 | 360 | 519 |
| two-scale clusters | 666 | 669 | 582 | 804 |
| clustered | 715 | 711 | 639 | 872 |
| thin pipe | 4143 | 3762 | 3577 | 5442 |
| pathological (10k, one cell) | 502 | 550 | 1481 | 596 |
<!-- bench:headline-memory:end -->

### Stress scenes

<!-- bench:stress-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|
| clustered 500k | 646 | 116.25 ms | 168.37 ms | 2340.44 ms | 1.4x | 1.7x | 20.1x |
| clustered 1M | 652 | 235.60 ms | 343.20 ms | 4758.10 ms | 1.5x | 1.7x | 20.2x |
| two-scale 1M | 810 | 257.96 ms | 343.90 ms | 3647.47 ms | 1.3x | 1.6x | 14.1x |
| uniform 2M | 33 | 53.19 ms | 113.31 ms | 416.77 ms | 2.1x | 2.4x | 7.8x |
| 8 sets x 2.5k | 4 | 1.33 ms | 2.68 ms | 3.29 ms | 2.0x | 2.3x | 2.5x |
| 8 sets x 12.5k | 4 | 4.48 ms | 10.21 ms | 21.82 ms | 2.3x | 2.6x | 4.9x |
| 16 sets x 1.25k | 2 | 2.74 ms | 5.19 ms | 5.99 ms | 1.9x | 2.5x | 2.2x |
| droplets 200k | 26 | 3.87 ms | 6.05 ms | 18.43 ms | 1.6x | 1.7x | 4.8x |
| droplets 200k, jitter | 25 | 4.03 ms | 5.77 ms | 24.33 ms | 1.4x | 1.5x | 6.0x |
<!-- bench:stress-timing:end -->

<!-- bench:stress-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| clustered 500k | 1839 | 1754 | 1486 | 2184 |
| clustered 1M | 3706 | 3498 | 3072 | 4383 |
| two-scale 1M | 4742 | 4432 | 3796 | 5067 |
| uniform 2M | 946 | 605 | 670 | 1626 |
| 8 sets x 2.5k | 8 | 150 | 150 | 150 |
| 8 sets x 12.5k | 44 | 150 | 150 | 202 |
| 16 sets x 1.25k | 9 | 151 | 150 | 150 |
| droplets 200k | 54 | 150 | 150 | 151 |
| droplets 200k, jitter | 67 | 150 | 150 | 151 |
<!-- bench:stress-memory:end -->

### Scaling

<!-- bench:scale:start -->
| Distribution | Points | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 175k | 32 | 4.57 ms | 14.53 ms | 30.17 ms | 3.2x | 4.1x | 6.6x |
| uniform | 500k | 33 | 13.14 ms | 39.45 ms | 93.82 ms | 3.0x | 3.9x | 7.1x |
| uniform | 1M | 33 | 27.88 ms | 91.66 ms | 200.14 ms | 3.3x | 4.2x | 7.2x |
| uniform | 2M | 33 | 53.18 ms | 110.42 ms | 415.73 ms | 2.1x | 2.3x | 7.8x |
| uniform | 4M | 33 | 113.78 ms | 390.67 ms | 882.43 ms | 3.4x | 4.3x | 7.8x |
| uniform | 8M | 33 | 290.08 ms | 780.67 ms | 1785.29 ms | 2.7x | 3.1x | 6.2x |
| sparse | 175k | 0 | 1.88 ms | 3.79 ms | 135.88 ms | 2.0x | 2.1x | 72.3x |
| sparse | 500k | 0 | 5.94 ms | 11.89 ms | 284.16 ms | 2.0x | 2.1x | 47.8x |
| sparse | 1M | 0 | 10.44 ms | 16.25 ms | 546.57 ms | 1.6x | 1.7x | 52.3x |
| sparse | 2M | 0 | 21.33 ms | 38.69 ms | 1123.15 ms | 1.8x | 2.0x | 52.7x |
| sparse | 4M | 0 | 36.44 ms | 102.74 ms | 2453.01 ms | 2.8x | 3.2x | 67.3x |
| sparse | 8M | 0 | 78.75 ms | 141.15 ms | 4888.43 ms | 1.8x | 2.0x | 62.1x |
<!-- bench:scale:end -->

### Growing sets

<!-- bench:growing:start -->
| Scene | ispc_apfrnn | TreeNSearch | vs Tree | vs Tree (search) |
|---|---:|---:|---:|---:|
| uniform 200k -> 400k | 18.04 ms | 37.47 ms | 2.1x | 2.2x |
| slab 200k -> 400k | 33.02 ms | 43.50 ms | 1.3x | 1.3x |
| clustered 200k -> 400k | 70.86 ms | 87.88 ms | 1.2x | 1.3x |
<!-- bench:growing:end -->
