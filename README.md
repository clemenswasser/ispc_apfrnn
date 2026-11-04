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
- `last_stats()` breaks down the last run (build, grid, same-set and
  cross-set timings, fixed/CSR mode and memory).

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

When this repository is built on its own, vcpkg is fetched from the pinned
tarball in `CMakeLists.txt`, and `vcpkg.json` lists the dependencies (TBB,
doctest, benchmark).

| Option | Default | |
|---|---|---|
| `APFRNN_BUILD_TESTS` | top-level | doctest suite |
| `APFRNN_BUILD_BENCHMARKS` | top-level | Google Benchmark targets |
| `APFRNN_BUILD_BASELINES` | `OFF` | fetches CompactNSearch/TreeNSearch for comparisons |
| `APFRNN_ENABLE_SYCL` | `OFF` | static SYCL GPU backend, requires icpx |

## Optional SYCL GPU backend

```sh
source /opt/intel/oneapi/setvars.sh
cmake --preset sycl-release
cmake --build --preset sycl-release
```

One CMake tree has one C++ compiler, so enabling SYCL compiles the whole
project with icpx and links the GPU code statically. The default build needs
no oneAPI at all.

```cpp
search.set_search_backend(apfrnn::SearchBackend::GpuExact);  // false if unavailable
search.set_search_backend(apfrnn::SearchBackend::Auto);      // GPU only where it wins
```

The backend can also be preselected with `APFRNN_BACKEND=cpu|gpu|auto`. GPU
results are set-equal to CPU results. Auto picks the GPU only when every
active source set has at least 200k points and the cells hold a moderate
number of points (20–48). High-degree scenes are limited by the PCIe copy of
the int32 neighbor rows, and small frames by launch overhead. If GPU output
would exceed `APFRNN_SYCL_MAX_OUTPUT_MB` (default 512), the search falls back
to the CPU. It also falls back when one cell would hold so many points that a
single GPU work item would scan the whole neighborhood (the pathological
scene); that keeps a long-running kernel from resetting the device.

### GPU frame times

Median frame time on the machine below with an Intel Arc A770 (oneAPI 2026.2,
Level Zero 1.17.39758+11), 5 x 1 s repetitions with the CPU threads pinned to
0-15. These are raw neighbor-list frames; they do not include an SPH consumer.
`Auto` applies the policy above. This table is from the earlier SYCL build; its
CPU column is that build's CPU backend and has not been re-run for the current
CPU table above.

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

\* Falls back to the CPU because the int32 output exceeds
`APFRNN_SYCL_MAX_OUTPUT_MB`.

## Performance

Steady-state frame times, wall clock, i7-12700K pinned to its 8 P-cores (16
threads). The baselines use OpenMP, so the comparison is wall time, not CPU
time. Each benchmark ran for 2 s after two warm-up frames (see
[Methodology](#methodology)). All numbers below were measured on the current
tree at `7445981`.

### Synthetic scenes

Fluid set plus a static boundary and an emitter, frame replay
(`ispc_apfrnn_bench`, the `Frame/...` benchmarks). The speedup shrinks as the
neighbor count per point grows, because writing the rows dominates there.

| Scene (200k unless noted) | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 32 | 5.94 ms | 19.10 ms | 28.97 ms | 3.2x | 4.9x |
| uniform, 1M | 33 | 32.77 ms | 125.78 ms | 278.92 ms | 3.8x | 8.5x |
| uniform, moving | 32 | 6.22 ms | 19.50 ms | 32.27 ms | 3.1x | 5.2x |
| jittered lattice | 29 | 4.72 ms | 9.02 ms | 22.50 ms | 1.9x | 4.8x |
| SPH-like slab | 58 | 7.64 ms | 19.76 ms | 41.42 ms | 2.6x | 5.4x |
| slab, moving | 58 | 7.92 ms | 20.04 ms | 43.96 ms | 2.5x | 5.5x |
| huge sparse domain | 0 | 1.44 ms | 4.61 ms | 32.20 ms | 3.2x | 22.3x |
| thin slab | 243 | 17.07 ms | 26.27 ms | 220.08 ms | 1.5x | 12.9x |
| surface | 270 | 16.20 ms | 21.67 ms | 81.96 ms | 1.3x | 5.1x |
| two-scale clusters | 575 | 26.78 ms | 38.64 ms | 693.24 ms | 1.4x | 25.9x |
| clustered | 632 | 30.79 ms | 42.78 ms | 1111.61 ms | 1.4x | 36.1x |
| thin pipe | 3402 | 105.89 ms | 155.05 ms | 1267.22 ms | 1.5x | 12.0x |
| pathological (10k, one cell) | 9999 | 37.80 ms | 57.77 ms | 347.72 ms | 1.5x | 9.2x |

Fixed-row capacity and resident memory:

| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| uniform | 167 | 193 | 131 | 168 |
| uniform, 1M | 835 | 938 | 575 | 812 |
| uniform, moving | 167 | 194 | 131 | 169 |
| jittered lattice | 100 | 223 | 94 | 168 |
| SPH-like slab | 167 | 192 | 145 | 235 |
| slab, moving | 167 | 193 | 156 | 236 |
| huge sparse domain | 5 | 26 | 43 | 201 |
| thin slab | 362 | 390 | 304 | 474 |
| surface | 387 | 415 | 397 | 552 |
| two-scale clusters | 2242 | 1332 | 706 | 875 |
| clustered | 1412 | 1275 | 684 | 963 |
| thin pipe | 3683 | 3728 | 3861 | 5576 |
| pathological (10k, one cell) | 492 | 501 | 1542 | 596 |

`Rows MB` is the fixed-row/CSR capacity reported by `last_stats()`. `Peak RSS`
is the process high-water mark from a separate one-scene run; rows can exceed
RSS when capacity pages are not all touched. Tree and Compact have no row
counter. At 50k points the tree speedups range from 1.2x to 6.0x and the
Compact speedups from 4.2x to 28x.

### Stress scenes

These are deliberately outside the headline matrix: 500k-2M high-degree sets,
many all-pairs sets, and dense droplets in a huge domain.

| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|
| clustered 500k | 646 | 79.4 ms | 119.0 ms | 2861 ms | 1.5x | 36x |
| clustered 1M | 652 | 159 ms | 288 ms | 6057 ms | 1.8x | 38x |
| two-scale 1M | 810 | 158 ms | 281 ms | 6785 ms | 1.8x | 43x |
| uniform 2M | 33 | 83.0 ms | 323 ms | 655 ms | 3.9x | 7.9x |
| 8 sets x 2.5k | 4 | 0.98 ms | 1.69 ms | 2.05 ms | 1.7x | 2.1x |
| 8 sets x 12.5k | 4 | 6.87 ms | 9.60 ms | 18.7 ms | 1.4x | 2.7x |
| 16 sets x 1.25k | 2 | 2.02 ms | 2.03 ms | 2.72 ms | 1.0x | 1.3x |
| droplets 200k | 26 | 3.70 ms | 5.01 ms | 11.4 ms | 1.4x | 3.1x |
| droplets 200k, jitter | 25 | 6.23 ms | 5.79 ms | 19.9 ms | 0.9x | 3.2x |

| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |
|---|---:|---:|---:|---:|
| clustered 500k | 1490 | 1497 | 1611 | 2434 |
| clustered 1M | 3002 | 2997 | 3102 | 4858 |
| two-scale 1M | 3847 | 3797 | 4041 | 5635 |
| uniform 2M | 492 | 803 | 1184 | 1618 |
| 8 sets x 2.5k | 25 | 53 | 18 | 44 |
| 8 sets x 12.5k | 125 | 214 | 54 | 194 |
| 16 sets x 1.25k | 50 | 76 | 23 | 80 |
| droplets 200k | 105 | 138 | 56 | 76 |
| droplets 200k, jitter | 105 | 146 | 53 | 79 |

Jittered droplets is the one stress scene where TreeNSearch is faster (by
about 8%): the static cache hits disappear when every point moves, so the
hash-grid stencil and range merge run every frame.

### Motion

The timed loop advances the scene before every `backend->run()`, so the
dynamic fluid set is re-read and searched on every frame. `_moving` and
`_fast` translate the whole set rigidly (3% and 35% of the radius per frame),
which leaves relative geometry unchanged and is the easy case. `_jitter`
moves every point independently by up to a quarter radius per axis, changing
cell occupancy and the maximum degree, and is closer to real flow.

| Scene | static | jitter | jitter vs static | maximum row, static -> jitter |
|---|---:|---:|---:|---:|
| uniform 200k | 5.95 ms | 7.05 ms | +19% | 62 -> 62 |
| slab 200k | 7.68 ms | 8.61 ms | +12% | 92 -> 94 |
| clustered 200k | 30.81 ms | 32.96 ms | +7% | 1717 -> 1694 |
| thin pipe 50k | 20.00 ms | 26.86 ms | +34% | 2475 -> 3413 |
| droplets 200k | 3.66 ms | 6.26 ms | +71% | 61 -> 63 |

Jitter costs 7% to 71% more than a static frame. Thin pipe rises because the
neighbour count itself increases. Droplets rises most because the static
frame reuses the unchanged cell and range tables while the jittered frame
rebuilds them; that is the honest cost of motion on the hash-grid path. The
`Frame` numbers exclude the O(n) `scene.advance()`; `Api/Frame` times it inside
the loop and the small-move case is 6.38 ms against 5.96 ms static at 200k.

The warm-up is not hiding motion: it only removes one-time allocation and
capacity fitting. `Cold/` times setup plus the first frame: 11.7 ms uniform
200k, 12.4 ms slab 200k, 98.3 ms clustered 200k and 49.6 ms thin pipe 50k,
against 5.9/7.6/30.8/20.0 ms in steady state.

### Saved learnsph frames

One fluid/boundary pair from real simulations (`ispc_apfrnn_bench` with
`--learnsph_res`, median of 7 runs after a warm-up). Peak RSS is from a
separate one-scene process, as in the synthetic tables.

| Frame | Fluid / boundary | ispc_apfrnn | TreeNSearch | CompactNSearch | Peak RSS (ours / Tree / Compact) |
|---|---:|---:|---:|---:|---:|
| ramp_split_lv | 20k / 78k | 0.69 ms | 1.35 ms | 2.69 ms | 38 / 17 / 31 MB |
| ramp_split_hv | 29k / 78k | 0.55 ms | 1.75 ms | 4.77 ms | 32 / 28 / 40 MB |
| wall_01 | 74k / 208k | 1.16 ms | 4.21 ms | 9.40 ms | 69 / 73 / 100 MB |
| funnel | 21k / 535k | 0.51 ms | 4.99 ms | 17.8 ms | 78 / 76 / 104 MB |
| sink | 57k / 13k | 1.88 ms | 2.72 ms | 10.4 ms | 85 / 59 / 80 MB |
| bowl | 111k / 82k | 2.92 ms | 6.12 ms | 14.8 ms | 134 / 42 / 83 MB |
| wall | 107k / 208k | 3.15 ms | 5.02 ms | 17.6 ms | 161 / 58 / 102 MB |


### Costs and caveats

- **Memory.** Rows are fixed-capacity (the observed maximum degree plus a small
  margin), so they use more memory than exact CSR. Peak RSS is 0.9x to 3.2x
  that of TreeNSearch on the learnsph frames. Thin pipe at 200k needs 3.68 GB
  of rows and 3.73 GB peak RSS. `set_memory_budget` caps this; larger rows fall
  back to CSR.
- **First frame.** The first frame starts from a default row capacity: it
  can overflow, retry with a larger one, and touch freshly allocated memory.
  Thin pipe's first frame takes about 250 ms against 103 ms in steady state;
  from the second frame on, rows reuse the fitted capacity and buffer. All
  tables above are steady state.
- **CPU time.** Tables report wall time because the baselines use OpenMP. On
  ramp_split_lv the current CPU time is 0.69 ms against TreeNSearch's 1.27 ms;
  the earlier CPU-time caveat no longer applies.
- **One machine.** All numbers come from one AVX2 desktop CPU.

### Methodology

The synthetic numbers come from:

```sh
taskset -c 0-15 ./build/bench/ispc_apfrnn_bench \
  --benchmark_filter='^Frame/(apfrnn_cpu|tree|compact)/' \
  --benchmark_min_time=2s --benchmark_repetitions=7 \
  --benchmark_report_aggregates_only=true
```

Every benchmark runs two warm-up frames before the timed frames, so they are
steady state even for short `--benchmark_min_time` values. The second warm-up
frame is validated, so plans that only change after the first frame (fitted
row capacities, small cells) are checked too: the label carries a checksum of
the neighbor pairs clearly within the radius, which must be equal for all
backends of a scene, and `FalsePositives` must be 0. The learnsph tables use the same command with
`--learnsph_res=../learnsph/res` (the default); the saved frames join the
scene list when their files exist.

Memory is measured separately, one scene and backend per process, with the
same warm-up and a 0.5 s timed region. `Peak RSS` is the process high-water
mark; `Rows MB` is the fixed-row/CSR capacity from `last_stats()`.

### How it works

- The grid is a flat cell grid with a dense cell lookup when the bounding box
  fits the memory budget, and a hash otherwise. Uniformly sparse sets use a
  sweep fallback; sparse sets with dense clusters keep the hash grid, because
  an x-window sweep would scan each cluster repeatedly.
- Dense cell domains are counting-sorted by cell; sparse domains use a stable
  radix sort over the cell key. Results are mapped back to the caller's order.
- Neighbor-cell and merged-range tables are reused when the occupied cells and
  their point ranges are unchanged, so stationary or slow sets skip the hash
  stencil and range merge. Moving sets fall back to a rebuild.
- Rows are written in one pass into fixed-capacity rows, with CSR as the
  fallback when a row overflows. Up to eight queries share a target load.
  Row capacity adapts to the observed degree.
- Dense cells are split into point blocks with bounding boxes, and
  high-degree rows are written with non-temporal stores.

### Benchmarks

```sh
./scripts/bench.sh build        # headline comparison against the baselines
./scripts/bench_gpu.sh build-sycl  # CPU vs GPU vs Auto (SYCL build)
./build/bench/ispc_apfrnn_bench --benchmark_filter='^Stress/'
./build/bench/ispc_apfrnn_bench --benchmark_filter='^Api/'
```

`ispc_apfrnn_bench` has five benchmark families:

- `Frame/<backend>/<scene>` runs every backend on the headline scenes.
- `Stress/<backend>/<scene>` extends the matrix to 500k-2M high-degree sets,
  8-16 point sets with every pair active, and dense droplets scattered
  through a huge domain (hash grid with ~26 neighbors per point); these are
  the regimes outside the headline scenes.
- `Threads/<backend>/<scene>/<threads>` is CPU thread scaling (uniform 200k
  across 1, 2, 4, 8 and 16 threads, 1M across 8 and 16, and clustered 200k
  and learnsph ramp_split_lv across 1, 4 and 16).
- `Cold/<backend>/<scene>` times setup plus the first frame on a fresh
  backend (uniform, slab and clustered at 200k, thin pipe at 50k), where row
  capacities are not yet fitted.
- `Api/<scenario>/<scene>` covers the NSearch-API scenarios: static, small and
  large motion frames with stage counters (`Api/Frame/...`), skin reuse
  (`Api/Skin`, `Api/SkinConsume`), consuming neighbors (`Api/Consume`),
  resizing (`Api/Resize`) and setup cost (`Api/Setup`).

`--benchmark_filter` selects by family, backend or scene. The backends are
`apfrnn_cpu`, `apfrnn_gpu` and `apfrnn_auto` (the GPU pair only in a SYCL build
with a GPU), plus `tree` and `compact` in a baseline build. The Frame scenes
are the synthetic distributions uniform, jittered lattice, lattice, clustered,
slab, thin slab, huge domain, surface, thin pipe and two-scale at 50k and
200k, uniform at 1k/10k/1M, clustered at 10k, pathological at 10k, the moving
variants (`_moving`: small rigid motion, `_fast`: large rigid motion,
`_jitter`: non-rigid per-point motion, covering uniform, slab, clustered and
droplets at 200k and thin pipe at 50k), slab with the boundary also querying
(`_boundary_queries`), a 1k probe set queried into 200k/1M points, and any
learnsph frames found under `--learnsph_res` (default `../learnsph/res`). `scripts/bench.sh` accepts `APFRNN_FILTER` and
`APFRNN_CPUS`.

`--memory_budget_mb=N` overrides the library's fixed-row cap. The default is
3 GiB, and raising it keeps large high-degree scenes on the fixed-row writer
instead of the slower CSR fallback.

The Stress scenes are deliberately outside the headline set: they include the
current scaling gaps (large high-degree sets on the CSR fallback and many
all-pairs sets), so improvements can be measured without changing the headline
comparison.

## Development

```sh
./scripts/ci.sh                  # Release + ASan/UBSan tests + coverage floors
./scripts/coverage.sh            # clang source coverage (90% lines / 70% branches)
./scripts/profile.sh build 50000 # samply profile of the NSearch benchmark
./scripts/ispc_analyze.sh        # ISPC asm/IR + llvm-mca per kernel
```

The test suite includes a multi-frame fuzz test (`tests/fuzz_test.cpp`) that
mutates random multi-set scenes between frames and checks every active pair
against a reference on every frame. It also requires its seeds to take, enter
and leave every code path recorded in `RunStats::paths`, so a new path needs
fuzz coverage. `APFRNN_FUZZ_SEEDS`, `APFRNN_FUZZ_FIRST_SEED`,
`APFRNN_FUZZ_SEED` (replay one seed) and `APFRNN_FUZZ_TRACE=1` control it; a
long soak is `APFRNN_FUZZ_SEEDS=6000 ./build/tests/ispc_apfrnn_tests
-tc='multi-frame fuzz*'`.
