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
| ramp hv, t=0.10 s | 3k | 23 | 0.13 ms | 1.55 ms | n/a | 11.9x | 19.9x | n/a | 64 / 65 / n/a |
| ramp lv, t=0.10 s | 3k | 23 | 0.13 ms | 1.58 ms | n/a | 12.0x | 19.9x | n/a | 64 / 65 / n/a |
| ramp hv, t=0.30 s | 8k | 44 | 0.25 ms | 0.85 ms | 3.36 ms | 3.4x | 4.1x | 13.4x | 75 / 66 / 64 |
| ramp lv, t=0.30 s | 8k | 38 | 0.25 ms | 0.90 ms | 2.89 ms | 3.7x | 4.5x | 11.8x | 63 / 66 / 64 |
| funnel, t=0.20 s | 23k | 45 | 0.92 ms | 6.73 ms | n/a | 7.3x | 9.1x | n/a | 165 / 102 / n/a |
| sink, t=0.10 s | 29k | 64 | 0.85 ms | 1.45 ms | 5.39 ms | 1.7x | 1.9x | 6.3x | 64 / 67 / 74 |
| sink, t=0.50 s | 29k | 60 | 0.91 ms | 1.38 ms | 5.57 ms | 1.5x | 1.7x | 6.1x | 64 / 67 / 73 |
| funnel, t=0.60 s | 73k | 47 | 4.11 ms | 10.87 ms | n/a | 2.6x | 3.5x | n/a | 178 / 124 / n/a |
| wall, t=0.50 s | 74k | 61 | 2.82 ms | 4.09 ms | 25.91 ms | 1.4x | 1.8x | 9.2x | 147 / 105 / 111 |
| droplet, t=0.05 s | 74k | 56 | 1.87 ms | 3.28 ms | 12.00 ms | 1.8x | 2.1x | 6.4x | 63 / 77 / 92 |
| droplet, t=0.40 s | 74k | 57 | 1.87 ms | 3.40 ms | 12.09 ms | 1.8x | 2.2x | 6.5x | 63 / 77 / 92 |
| wall, t=0.10 s | 74k | 53 | 1.68 ms | 3.70 ms | 21.21 ms | 2.2x | 2.8x | 12.6x | 63 / 104 / 103 |
| funnel, t=0.20 s | 106k | 48 | 4.02 ms | 17.59 ms | n/a | 4.4x | 5.4x | n/a | 333 / 185 / n/a |
| dam break, t=0.10 s | 155k | 58 | 4.65 ms | 6.86 ms | 37.59 ms | 1.5x | 1.7x | 8.1x | 193 / 128 / 176 |
| dam break, t=0.40 s | 155k | 58 | 5.14 ms | 8.18 ms | 39.80 ms | 1.6x | 1.9x | 7.7x | 191 / 133 / 177 |
| bowl, t=0.10 s | 170k | 64 | 5.75 ms | 7.14 ms | 36.35 ms | 1.2x | 1.3x | 6.3x | 209 / 130 / 175 |
| bowl, t=0.50 s | 170k | 62 | 5.56 ms | 6.89 ms | 36.19 ms | 1.2x | 1.3x | 6.5x | 204 / 133 / 175 |
| funnel, t=0.40 s | 213k | 49 | 10.78 ms | 26.88 ms | n/a | 2.5x | 2.9x | n/a | 458 / 244 / n/a |
| sink, t=0.10 s | 214k | 63 | 7.16 ms | 11.22 ms | 41.47 ms | 1.6x | 1.8x | 5.8x | 201 / 152 / 212 |
| sink, t=0.40 s | 214k | 56 | 7.74 ms | 9.76 ms | 42.42 ms | 1.3x | 1.4x | 5.5x | 215 / 141 / 210 |
| droplet, t=0.05 s | 578k | 57 | 17.67 ms | 22.71 ms | 101.20 ms | 1.3x | 1.3x | 5.7x | 255 / 300 / 541 |
| droplet, t=0.25 s | 578k | 58 | 17.78 ms | 21.74 ms | 104.49 ms | 1.2x | 1.2x | 5.9x | 254 / 301 / 540 |
| wall, t=0.10 s | 578k | 54 | 16.48 ms | 28.40 ms | 144.59 ms | 1.7x | 1.9x | 8.8x | 419 / 415 / 668 |
| dam break, t=0.10 s | 701k | 50 | 23.54 ms | 35.85 ms | 178.34 ms | 1.5x | 1.7x | 7.6x | 647 / 460 / 734 |
| bowl, t=0.10 s | 1.3M | 38 | 37.21 ms | 64.30 ms | 280.93 ms | 1.7x | 2.1x | 7.6x | 1071 / 762 / 1162 |
| wall, t=0.10 s | 1.7M | 56 | 48.85 ms | 82.50 ms | 397.50 ms | 1.7x | 1.8x | 8.1x | 1112 / 1058 / 1857 |
| dam break, t=0.10 s | 2.4M | 43 | 82.07 ms | 126.65 ms | 568.81 ms | 1.5x | 1.6x | 6.9x | 2231 / 1447 / 2366 |
| bowl, t=0.10 s | 3.8M | 40 | 109.10 ms | 188.96 ms | 854.51 ms | 1.7x | 2.0x | 7.8x | 3021 / 2238 / 3255 |
<!-- bench:replay:end -->

### Synthetic scenes

<!-- bench:headline-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 32 | 7.15 ms | 16.28 ms | 35.78 ms | 2.3x | 2.9x | 5.0x |
| uniform, 1M | 33 | 42.30 ms | 91.74 ms | 200.29 ms | 2.2x | 2.6x | 4.7x |
| uniform, moving | 32 | 7.05 ms | 17.91 ms | 35.10 ms | 2.5x | 3.2x | 5.0x |
| jittered lattice | 29 | 6.32 ms | 13.21 ms | 34.29 ms | 2.1x | 2.5x | 5.4x |
| SPH-like slab | 58 | 9.32 ms | 18.58 ms | 56.41 ms | 2.0x | 2.3x | 6.0x |
| slab, moving | 58 | 9.28 ms | 19.49 ms | 55.50 ms | 2.1x | 2.5x | 6.0x |
| huge sparse domain | 0 | 3.79 ms | 4.56 ms | 163.12 ms | 1.2x | 1.2x | 43.0x |
| thin slab | 243 | 26.91 ms | 33.67 ms | 386.21 ms | 1.3x | 1.2x | 14.4x |
| surface | 270 | 26.48 ms | 36.35 ms | 116.73 ms | 1.4x | 1.3x | 4.4x |
| two-scale clusters | 575 | 47.61 ms | 54.64 ms | 545.34 ms | 1.1x | 1.2x | 11.5x |
| clustered | 632 | 53.21 ms | 63.82 ms | 933.87 ms | 1.2x | 1.2x | 17.5x |
| thin pipe | 3403 | 231.34 ms | 290.51 ms | 1251.10 ms | 1.3x | 1.6x | 5.4x |
| pathological (10k, one cell) | 9999 | 32.80 ms | 107.51 ms | 394.24 ms | 3.3x | 6.3x | 12.0x |
<!-- bench:headline-timing:end -->

<!-- bench:headline-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| uniform | 81 | 150 | 150 | 175 |
| uniform, 1M | 649 | 794 | 488 | 842 |
| uniform, moving | 81 | 150 | 150 | 175 |
| jittered lattice | 63 | 150 | 150 | 175 |
| SPH-like slab | 106 | 167 | 152 | 235 |
| slab, moving | 106 | 167 | 172 | 232 |
| huge sparse domain | 5 | 150 | 150 | 353 |
| thin slab | 356 | 392 | 298 | 455 |
| surface | 344 | 406 | 344 | 519 |
| two-scale clusters | 666 | 669 | 601 | 803 |
| clustered | 715 | 711 | 615 | 872 |
| thin pipe | 3695 | 3762 | 3710 | 5437 |
| pathological (10k, one cell) | 491 | 550 | 1482 | 596 |
<!-- bench:headline-memory:end -->

### Stress scenes

<!-- bench:stress-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|
| clustered 500k | 646 | 135.11 ms | 169.14 ms | 2349.42 ms | 1.3x | 1.3x | 17.4x |
| clustered 1M | 652 | 270.51 ms | 347.25 ms | 4777.70 ms | 1.3x | 1.4x | 17.7x |
| two-scale 1M | 810 | 290.31 ms | 347.02 ms | 3671.30 ms | 1.2x | 1.3x | 12.6x |
| uniform 2M | 33 | 77.32 ms | 111.46 ms | 418.03 ms | 1.4x | 1.5x | 5.4x |
| 8 sets x 2.5k | 4 | 1.61 ms | 2.70 ms | 3.29 ms | 1.7x | 2.5x | 2.0x |
| 8 sets x 12.5k | 4 | 8.92 ms | 10.28 ms | 21.84 ms | 1.2x | 1.3x | 2.4x |
| 16 sets x 1.25k | 2 | 4.67 ms | 5.31 ms | 6.03 ms | 1.1x | 1.5x | 1.3x |
| droplets 200k | 26 | 5.56 ms | 6.12 ms | 18.67 ms | 1.1x | 1.1x | 3.4x |
| droplets 200k, jitter | 25 | 5.74 ms | 5.72 ms | 24.46 ms | 1.0x | 1.0x | 4.3x |
<!-- bench:stress-timing:end -->

<!-- bench:stress-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| clustered 500k | 1837 | 1754 | 1527 | 2183 |
| clustered 1M | 3702 | 3492 | 2913 | 4384 |
| two-scale 1M | 4708 | 4430 | 3701 | 5066 |
| uniform 2M | 870 | 1254 | 685 | 1626 |
| 8 sets x 2.5k | 29 | 150 | 150 | 150 |
| 8 sets x 12.5k | 83 | 210 | 150 | 202 |
| 16 sets x 1.25k | 59 | 150 | 150 | 150 |
| droplets 200k | 56 | 150 | 150 | 150 |
| droplets 200k, jitter | 56 | 150 | 150 | 150 |
<!-- bench:stress-memory:end -->

### Scaling

<!-- bench:scale:start -->
| Distribution | Points | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 175k | 32 | 6.10 ms | 14.56 ms | 30.64 ms | 2.4x | 3.0x | 5.0x |
| uniform | 500k | 33 | 18.82 ms | 39.25 ms | 95.27 ms | 2.1x | 2.6x | 5.1x |
| uniform | 1M | 33 | 42.15 ms | 91.48 ms | 199.89 ms | 2.2x | 2.6x | 4.7x |
| uniform | 2M | 33 | 77.30 ms | 111.53 ms | 419.31 ms | 1.4x | 1.5x | 5.4x |
| uniform | 4M | 33 | 162.19 ms | 391.12 ms | 879.97 ms | 2.4x | 3.0x | 5.4x |
| uniform | 8M | 33 | 335.09 ms | 780.88 ms | 1783.11 ms | 2.3x | 2.6x | 5.3x |
| sparse | 175k | 0 | 3.28 ms | 3.88 ms | 136.52 ms | 1.2x | 1.2x | 41.6x |
| sparse | 500k | 0 | 9.99 ms | 12.26 ms | 283.18 ms | 1.2x | 1.3x | 28.3x |
| sparse | 1M | 0 | 10.82 ms | 16.00 ms | 545.38 ms | 1.5x | 1.6x | 50.4x |
| sparse | 2M | 0 | 22.85 ms | 37.97 ms | 1136.94 ms | 1.7x | 1.8x | 49.7x |
| sparse | 4M | 0 | 38.58 ms | 105.66 ms | 2469.53 ms | 2.7x | 3.2x | 64.0x |
| sparse | 8M | 0 | 80.48 ms | 141.50 ms | 4928.53 ms | 1.8x | 2.0x | 61.2x |
<!-- bench:scale:end -->

### Growing sets

<!-- bench:growing:start -->
| Scene | ispc_apfrnn | TreeNSearch | vs Tree | vs Tree (search) |
|---|---:|---:|---:|---:|
| uniform 200k -> 400k | 23.19 ms | 37.09 ms | 1.6x | 1.8x |
| slab 200k -> 400k | 42.83 ms | 43.18 ms | 1.0x | 1.0x |
| clustered 200k -> 400k | 79.80 ms | 86.73 ms | 1.1x | 1.0x |
<!-- bench:growing:end -->
