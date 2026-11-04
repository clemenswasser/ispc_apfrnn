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
`Auto` applies the policy above.

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
[Methodology](#methodology)).

### Synthetic scenes

Fluid set plus a static boundary and an emitter, frame replay
(`ispc_apfrnn_bench`, the `Frame/...` benchmarks). The speedup shrinks
as the neighbor count per point grows, because writing the rows dominates
there.

| Scene (200k unless noted) | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 32 | 5.99 ms | 19.2 ms | 29.3 ms | 3.2x | 4.9x |
| uniform, 1M | 33 | 32.1 ms | 123 ms | 282 ms | 3.8x | 8.8x |
| uniform, moving | 32 | 6.23 ms | 19.5 ms | 32.0 ms | 3.1x | 5.1x |
| jittered lattice | 29 | 4.64 ms | 8.91 ms | 21.9 ms | 1.9x | 4.7x |
| SPH-like slab | 58 | 7.20 ms | 19.7 ms | 41.7 ms | 2.7x | 5.8x |
| slab, moving | 58 | 7.70 ms | 20.3 ms | 44.4 ms | 2.6x | 5.8x |
| huge sparse domain | 0 | 1.48 ms | 4.58 ms | 32.0 ms | 3.1x | 22x |
| thin slab | 243 | 14.9 ms | 23.9 ms | 225 ms | 1.6x | 15x |
| surface | 270 | 15.3 ms | 21.8 ms | 81.4 ms | 1.4x | 5.3x |
| two-scale clusters | 575 | 26.0 ms | 36.5 ms | 684 ms | 1.4x | 26x |
| clustered | 632 | 31.4 ms | 41.7 ms | 1107 ms | 1.3x | 35x |
| thin pipe | 3400 | 112 ms | 156 ms | 1262 ms | 1.4x | 11x |
| pathological (10k, one cell) | 10000 | 37.6 ms | 54.7 ms | 345 ms | 1.5x | 9.2x |

At 50k points the ratios are similar (1.45x to 5.5x against TreeNSearch).

### Motion

The timed loop advances the scene before every `backend->run()`, so the
dynamic fluid set is re-read and searched on every frame. `_moving` and
`_fast` translate the whole set rigidly (3% and 35% of the radius per frame),
which leaves relative geometry unchanged and is the easy case. `_jitter`
moves every point independently by up to a quarter radius per axis, changing
cell occupancy and the maximum degree, and is closer to real flow.

| Scene | jitter vs static | maximum row, static -> jitter |
|---|---:|---:|
| uniform 200k | +20% | 62 -> 61 |
| slab 200k | +13% | 92 -> 95 |
| clustered 200k | +7% | 1717 -> 1694 |
| thin pipe 50k | +40% | 2475 -> 3429 |

The largest increase is on the high-degree thin pipe, where the neighbour
count itself rises. Rigid translation does not cover this case. The `Frame`
numbers exclude the O(n) `scene.advance()`; `Api/Frame` times it inside the
loop and the small-move case is 6.38 ms against 5.96 ms static at 200k.

The warm-up is not hiding motion: it only removes one-time allocation and
capacity fitting. `Cold/` times setup plus the first frame, 11.6 ms for
uniform 200k and 12.2 ms for slab 200k, against 5.8 ms and 7.6 ms in steady
state.

### Saved learnsph frames

One fluid/boundary pair from real simulations (`ispc_apfrnn_bench` with
`--learnsph_res`, median of 7 runs after a warm-up). Peak RSS is that of the
whole benchmark process.

| Frame | Fluid / boundary | ispc_apfrnn | TreeNSearch | CompactNSearch | Peak RSS (ours / Tree) |
|---|---:|---:|---:|---:|---:|
| ramp_split_lv | 20k / 78k | 0.88 ms | 1.40 ms | 2.69 ms | 32 / 12 MB |
| ramp_split_hv | 29k / 78k | 0.62 ms | 1.80 ms | 4.69 ms | 33 / 19 MB |
| wall_01 | 74k / 208k | 1.28 ms | 4.18 ms | 11.0 ms | 64 / 51 MB |
| funnel | 21k / 535k | 0.57 ms | 4.99 ms | 18.1 ms | 77 / 50 MB |
| sink | 57k / 13k | 1.71 ms | 2.62 ms | 9.86 ms | 70 / 42 MB |
| bowl | 111k / 82k | 3.35 ms | 6.20 ms | 14.7 ms | 114 / 32 MB |
| wall | 107k / 208k | 3.33 ms | 5.03 ms | 17.4 ms | 130 / 40 MB |

### Costs and caveats

- **Memory.** Rows are fixed-capacity (the observed maximum degree plus a small
  margin), so they use more memory than exact CSR. Peak RSS is 1.3x to 3.6x
  that of TreeNSearch on the learnsph frames. Thin pipe at 200k needs about
  3.7 GB of rows. `set_memory_budget` caps this; larger rows fall back to CSR.
- **First frame.** The first frame starts from a default row capacity: it
  can overflow, retry with a larger one, and touch freshly allocated memory.
  Thin pipe's first frame takes about 250 ms against 103 ms in steady state;
  from the second frame on, rows reuse the fitted capacity and buffer. All
  tables above are steady state.
- **CPU time.** Wall time is lower everywhere, but total CPU time is not
  always: on ramp_split_lv we use 12.2 ms of CPU against TreeNSearch's 9.9 ms.
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

### How it works

- The grid is a flat cell grid with a dense cell lookup when the bounding box
  fits the memory budget, and a hash otherwise. Uniformly sparse sets use a
  sweep fallback; sparse sets with dense clusters keep the hash grid, because
  an x-window sweep would scan each cluster repeatedly.
- Points are counting-sorted by cell. Results are mapped back to the caller's
  order.
- Rows are written in one pass into fixed-capacity rows, with CSR as the
  fallback when a row overflows. Row capacity adapts to the observed degree.
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
`_jitter`: non-rigid per-point motion, covering uniform, slab and clustered at
200k and thin pipe at 50k), slab with the boundary also querying
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
