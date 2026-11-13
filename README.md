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
| ramp hv, t=0.10 s | 3k | 23 | 0.13 ms | 1.53 ms | n/a | 11.9x | 20.9x | n/a | 64 / 65 / n/a |
| ramp lv, t=0.10 s | 3k | 23 | 0.13 ms | 1.59 ms | n/a | 12.0x | 20.6x | n/a | 65 / 65 / n/a |
| ramp hv, t=0.30 s | 8k | 44 | 0.23 ms | 0.89 ms | 3.35 ms | 4.0x | 5.0x | 14.8x | 76 / 66 / 64 |
| ramp lv, t=0.30 s | 8k | 38 | 0.22 ms | 0.91 ms | 2.91 ms | 4.2x | 5.1x | 13.5x | 63 / 66 / 64 |
| funnel, t=0.20 s | 23k | 45 | 0.79 ms | 6.55 ms | n/a | 8.3x | 11.3x | n/a | 166 / 107 / n/a |
| sink, t=0.10 s | 29k | 64 | 0.73 ms | 1.50 ms | 5.32 ms | 2.1x | 2.6x | 7.3x | 65 / 67 / 74 |
| sink, t=0.50 s | 29k | 60 | 0.79 ms | 1.39 ms | 5.47 ms | 1.8x | 2.1x | 6.9x | 65 / 66 / 74 |
| funnel, t=0.60 s | 73k | 47 | 2.88 ms | 10.82 ms | n/a | 3.8x | 5.5x | n/a | 138 / 130 / n/a |
| wall, t=0.50 s | 74k | 61 | 2.02 ms | 4.08 ms | 25.66 ms | 2.0x | 2.6x | 12.7x | 69 / 105 / 111 |
| droplet, t=0.05 s | 74k | 56 | 1.63 ms | 3.26 ms | 12.03 ms | 2.0x | 2.6x | 7.4x | 63 / 78 / 92 |
| droplet, t=0.40 s | 74k | 57 | 1.67 ms | 3.40 ms | 11.95 ms | 2.0x | 2.6x | 7.2x | 63 / 78 / 92 |
| wall, t=0.10 s | 74k | 53 | 1.39 ms | 3.63 ms | 21.04 ms | 2.6x | 3.6x | 15.1x | 63 / 106 / 103 |
| funnel, t=0.20 s | 106k | 48 | 3.14 ms | 17.59 ms | n/a | 5.6x | 7.7x | n/a | 301 / 196 / n/a |
| dam break, t=0.10 s | 155k | 58 | 3.60 ms | 6.72 ms | 37.32 ms | 1.9x | 2.3x | 10.4x | 101 / 125 / 176 |
| dam break, t=0.40 s | 155k | 58 | 3.92 ms | 7.77 ms | 39.34 ms | 2.0x | 2.4x | 10.0x | 100 / 133 / 177 |
| bowl, t=0.10 s | 170k | 64 | 4.33 ms | 7.28 ms | 36.20 ms | 1.7x | 1.9x | 8.4x | 108 / 129 / 175 |
| bowl, t=0.50 s | 170k | 62 | 4.31 ms | 6.81 ms | 36.20 ms | 1.6x | 1.8x | 8.4x | 107 / 130 / 175 |
| funnel, t=0.40 s | 213k | 49 | 7.46 ms | 26.38 ms | n/a | 3.5x | 4.8x | n/a | 332 / 247 / n/a |
| sink, t=0.10 s | 214k | 63 | 5.51 ms | 11.09 ms | 41.18 ms | 2.0x | 2.5x | 7.5x | 117 / 151 / 212 |
| sink, t=0.40 s | 214k | 56 | 5.68 ms | 9.88 ms | 42.07 ms | 1.7x | 2.0x | 7.4x | 112 / 144 / 210 |
| droplet, t=0.05 s | 578k | 57 | 14.94 ms | 22.59 ms | 98.07 ms | 1.5x | 1.6x | 6.6x | 205 / 302 / 541 |
| droplet, t=0.25 s | 578k | 58 | 15.03 ms | 23.02 ms | 100.01 ms | 1.5x | 1.7x | 6.7x | 204 / 308 / 540 |
| wall, t=0.10 s | 578k | 54 | 13.31 ms | 28.50 ms | 143.45 ms | 2.1x | 2.5x | 10.8x | 278 / 417 / 668 |
| dam break, t=0.10 s | 701k | 50 | 18.39 ms | 36.35 ms | 176.83 ms | 2.0x | 2.3x | 9.6x | 287 / 485 / 734 |
| bowl, t=0.10 s | 1.3M | 38 | 28.18 ms | 63.14 ms | 277.52 ms | 2.2x | 2.7x | 9.8x | 398 / 796 / 1163 |
| wall, t=0.10 s | 1.7M | 56 | 41.30 ms | 82.97 ms | 391.48 ms | 2.0x | 2.3x | 9.5x | 713 / 1086 / 1857 |
| dam break, t=0.10 s | 2.4M | 43 | 64.05 ms | 128.52 ms | 564.10 ms | 2.0x | 2.4x | 8.8x | 855 / 1550 / 2366 |
| bowl, t=0.10 s | 3.8M | 40 | 83.83 ms | 187.08 ms | 844.44 ms | 2.2x | 2.7x | 10.1x | 1077 / 2387 / 3256 |
<!-- bench:replay:end -->

### Synthetic scenes

<!-- bench:headline-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 32 | 5.38 ms | 16.37 ms | 35.51 ms | 3.0x | 4.0x | 6.6x |
| uniform, 1M | 33 | 28.51 ms | 91.96 ms | 199.59 ms | 3.2x | 4.1x | 7.0x |
| uniform, moving | 32 | 5.42 ms | 17.48 ms | 34.49 ms | 3.2x | 4.3x | 6.4x |
| jittered lattice | 29 | 4.98 ms | 13.17 ms | 33.99 ms | 2.6x | 3.4x | 6.8x |
| SPH-like slab | 58 | 7.18 ms | 19.05 ms | 56.09 ms | 2.7x | 3.4x | 7.8x |
| slab, moving | 58 | 7.20 ms | 20.09 ms | 54.84 ms | 2.8x | 3.6x | 7.6x |
| huge sparse domain | 0 | 2.15 ms | 4.63 ms | 161.82 ms | 2.2x | 2.2x | 75.3x |
| thin slab | 243 | 23.97 ms | 32.98 ms | 385.70 ms | 1.4x | 1.4x | 16.1x |
| surface | 270 | 24.73 ms | 36.30 ms | 112.98 ms | 1.5x | 1.4x | 4.6x |
| two-scale clusters | 575 | 43.76 ms | 53.26 ms | 542.86 ms | 1.2x | 1.3x | 12.4x |
| clustered | 632 | 47.13 ms | 65.51 ms | 928.76 ms | 1.4x | 1.5x | 19.7x |
| thin pipe | 3403 | 231.40 ms | 291.45 ms | 1242.74 ms | 1.3x | 1.6x | 5.4x |
| pathological (10k, one cell) | 9999 | 32.33 ms | 71.82 ms | 389.50 ms | 2.2x | 4.1x | 12.0x |
<!-- bench:headline-timing:end -->

<!-- bench:headline-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| uniform | 89 | 151 | 151 | 175 |
| uniform, 1M | 313 | 339 | 482 | 842 |
| uniform, moving | 89 | 151 | 151 | 175 |
| jittered lattice | 69 | 150 | 151 | 175 |
| SPH-like slab | 116 | 151 | 153 | 235 |
| slab, moving | 116 | 151 | 172 | 233 |
| huge sparse domain | 5 | 150 | 150 | 353 |
| thin slab | 356 | 392 | 288 | 455 |
| surface | 344 | 406 | 346 | 523 |
| two-scale clusters | 666 | 669 | 583 | 804 |
| clustered | 715 | 711 | 605 | 872 |
| thin pipe | 4143 | 3762 | 3655 | 5434 |
| pathological (10k, one cell) | 502 | 551 | 1482 | 596 |
<!-- bench:headline-memory:end -->

### Stress scenes

<!-- bench:stress-timing:start -->
| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|
| clustered 500k | 646 | 120.59 ms | 168.94 ms | 2333.93 ms | 1.4x | 1.6x | 19.4x |
| clustered 1M | 652 | 240.99 ms | 346.85 ms | 4755.97 ms | 1.4x | 1.6x | 19.7x |
| two-scale 1M | 810 | 275.52 ms | 351.58 ms | 3651.34 ms | 1.3x | 1.4x | 13.3x |
| uniform 2M | 33 | 54.31 ms | 110.86 ms | 414.70 ms | 2.0x | 2.3x | 7.6x |
| 8 sets x 2.5k | 4 | 1.30 ms | 2.67 ms | 3.28 ms | 2.1x | 2.3x | 2.5x |
| 8 sets x 12.5k | 4 | 4.68 ms | 10.30 ms | 21.87 ms | 2.2x | 2.4x | 4.7x |
| 16 sets x 1.25k | 2 | 2.71 ms | 5.23 ms | 5.94 ms | 1.9x | 2.6x | 2.2x |
| droplets 200k | 26 | 4.43 ms | 6.09 ms | 18.65 ms | 1.4x | 1.5x | 4.2x |
| droplets 200k, jitter | 25 | 4.66 ms | 6.11 ms | 24.19 ms | 1.3x | 1.4x | 5.2x |
<!-- bench:stress-timing:end -->

<!-- bench:stress-memory:start -->
| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| clustered 500k | 1839 | 1753 | 1528 | 2184 |
| clustered 1M | 3706 | 3497 | 2932 | 4383 |
| two-scale 1M | 4742 | 4431 | 3833 | 5067 |
| uniform 2M | 946 | 605 | 680 | 1626 |
| 8 sets x 2.5k | 8 | 151 | 150 | 150 |
| 8 sets x 12.5k | 44 | 151 | 151 | 202 |
| 16 sets x 1.25k | 9 | 150 | 150 | 151 |
| droplets 200k | 54 | 150 | 151 | 151 |
| droplets 200k, jitter | 67 | 150 | 151 | 150 |
<!-- bench:stress-memory:end -->

### Scaling

<!-- bench:scale:start -->
| Distribution | Points | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 175k | 32 | 4.69 ms | 14.51 ms | 30.02 ms | 3.1x | 4.1x | 6.4x |
| uniform | 500k | 33 | 13.39 ms | 40.34 ms | 94.28 ms | 3.0x | 4.0x | 7.0x |
| uniform | 1M | 33 | 28.44 ms | 91.35 ms | 198.76 ms | 3.2x | 4.1x | 7.0x |
| uniform | 2M | 33 | 54.28 ms | 111.76 ms | 416.01 ms | 2.1x | 2.3x | 7.7x |
| uniform | 4M | 33 | 114.25 ms | 387.04 ms | 875.32 ms | 3.4x | 4.4x | 7.7x |
| uniform | 8M | 33 | 287.83 ms | 774.72 ms | 1780.22 ms | 2.7x | 3.1x | 6.2x |
| sparse | 175k | 0 | 1.92 ms | 3.99 ms | 136.03 ms | 2.1x | 2.1x | 70.9x |
| sparse | 500k | 0 | 6.01 ms | 12.20 ms | 284.14 ms | 2.0x | 2.1x | 47.3x |
| sparse | 1M | 0 | 10.43 ms | 15.75 ms | 541.66 ms | 1.5x | 1.6x | 51.9x |
| sparse | 2M | 0 | 21.24 ms | 38.79 ms | 1126.46 ms | 1.8x | 2.0x | 53.0x |
| sparse | 4M | 0 | 36.15 ms | 102.54 ms | 2474.52 ms | 2.8x | 3.2x | 68.5x |
| sparse | 8M | 0 | 77.30 ms | 145.30 ms | 4914.39 ms | 1.9x | 2.1x | 63.6x |
<!-- bench:scale:end -->

### Growing sets

<!-- bench:growing:start -->
| Scene | ispc_apfrnn | TreeNSearch | vs Tree | vs Tree (search) |
|---|---:|---:|---:|---:|
| uniform 200k -> 400k | 19.06 ms | 37.05 ms | 1.9x | 2.2x |
| slab 200k -> 400k | 40.28 ms | 43.45 ms | 1.1x | 1.0x |
| clustered 200k -> 400k | 72.20 ms | 87.11 ms | 1.2x | 1.2x |
<!-- bench:growing:end -->
