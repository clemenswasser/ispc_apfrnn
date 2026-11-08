#include "api_bench.hpp"

#include "apfrnn/neighbor_search.hpp"
#include "scene.hpp"

#include <benchmark/benchmark.h>
#include <sys/resource.h>
#include <tbb/global_control.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace apfrnn::bench {

namespace {

using apfrnn::NSearch;

struct SearchFixtureData {
  Scene scene;
  std::unique_ptr<NSearch> search;
  int fluid_id = -1;
  int boundary_id = -1;
  int emitter_id = -1;
  Motion motion = Motion::Static;
  int step = 0;

  void build(Distribution distribution, int fluid_points) {
    static std::unique_ptr<tbb::global_control> thread_control;
    static bool thread_control_set = false;
    if (!thread_control_set) {
      thread_control_set = true;
      if (const char *env = std::getenv("APFRNN_TBB_THREADS")) {
        const int threads = std::atoi(env);
        if (threads > 0) {
          thread_control = std::make_unique<tbb::global_control>(
              tbb::global_control::max_allowed_parallelism,
              static_cast<std::size_t>(threads));
        }
      }
    }
    scene = make_scene(distribution, fluid_points);
    search = std::make_unique<NSearch>(scene.radius);
    fluid_id = search->add_point_set(
        scene.fluid_xyz.data(), static_cast<int>(scene.fluid.x.size()));
    boundary_id = search->add_point_set(
        scene.boundary_xyz.data(), static_cast<int>(scene.boundary.x.size()),
        true);
    emitter_id = search->add_point_set(
        scene.emitter_xyz.data(), static_cast<int>(scene.emitter.x.size()));

    search->set_all_searches(false);
    bool same = true;
    bool boundary = true;
    bool emitter = true;
    if (const char *active = std::getenv("APFRNN_BENCH_ACTIVE")) {
      same = std::strcmp(active, "same") == 0;
      boundary = std::strcmp(active, "boundary") == 0;
      emitter = std::strcmp(active, "emitter") == 0;
      if (std::strcmp(active, "all") == 0) {
        same = true;
        boundary = true;
        emitter = true;
      }
    }
    search->set_active_search(fluid_id, fluid_id, same);
    search->set_active_search(fluid_id, boundary_id, boundary);
    search->set_active_search(fluid_id, emitter_id, emitter);
    search->run();
  }

  void advance() {
    if (motion != Motion::Static) {
      apply_motion(scene, motion, step++);
    }
  }
};

int64_t consume_neighbors(const SearchFixtureData &data) {
  int64_t total = 0;
  for (int point = 0; point < static_cast<int>(data.scene.fluid.x.size());
       ++point) {
    total +=
        data.search->get_neighbors(data.fluid_id, data.fluid_id, point).size();
    total += data.search->get_neighbors(data.fluid_id, data.boundary_id,
                                        point)
                 .size();
    total +=
        data.search->get_neighbors(data.fluid_id, data.emitter_id, point)
            .size();
  }
  return total;
}

void attach_counters(benchmark::State &state, const SearchFixtureData &data) {
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(data.scene.fluid.x.size()));
  state.counters["FluidPoints"] =
      static_cast<double>(data.scene.fluid.x.size());
  state.counters["BoundaryPoints"] =
      static_cast<double>(data.scene.boundary.x.size());
  state.counters["EmitterPoints"] =
      static_cast<double>(data.scene.emitter.x.size());

  const auto stats = data.search->last_stats();
  state.counters["BuildMs"] = stats.sync_build_ms;
  state.counters["GridMs"] = stats.grid_ms;
  state.counters["SameBuildMs"] = stats.same_build_ms;
  state.counters["SameMs"] = stats.same_set_ms;
  state.counters["CrossMs"] = stats.cross_set_ms;
  state.counters["NeighborMB"] =
      static_cast<double>(stats.neighbor_bytes) / (1024.0 * 1024.0);
  state.counters["GridMB"] =
      static_cast<double>(stats.grid_bytes) / (1024.0 * 1024.0);
  state.counters["FixedOverflows"] = stats.fixed_overflows;
  state.counters["MaxNeighbors"] = stats.max_neighbors;
  state.counters["MaxDispSq"] = stats.max_displacement_sq;
  state.counters["RebuiltSets"] = stats.rebuilt_sets;
  state.counters["ReusedSets"] = stats.reused_sets;
  state.counters["GridSets"] = stats.grid_sets;
  state.counters["SweepSets"] = stats.sweep_sets;
  state.counters["SameFixed"] = stats.same_fixed_pairs;
  state.counters["SameCSR"] = stats.same_csr_pairs;
  state.counters["CrossFixed"] = stats.cross_fixed_pairs;
  state.counters["CrossCSR"] = stats.cross_csr_pairs;

  const int set_count = data.search->n_sets();
  const auto pair_index = [set_count](int i, int j) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(set_count) +
           static_cast<std::size_t>(j);
  };
  if (stats.pair_ms.size() >=
      static_cast<std::size_t>(set_count * set_count)) {
    state.counters["FluidSelfMs"] =
        stats.pair_ms[pair_index(data.fluid_id, data.fluid_id)];
    state.counters["FluidBoundaryMs"] =
        stats.pair_ms[pair_index(data.fluid_id, data.boundary_id)];
    state.counters["FluidEmitterMs"] =
        stats.pair_ms[pair_index(data.fluid_id, data.emitter_id)];
  }

  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    state.counters["PeakRSSMB"] =
        static_cast<double>(usage.ru_maxrss) / 1024.0;
  }
}

void api_frame(benchmark::State &state, Distribution distribution,
               int fluid_points, Motion motion, bool consume) {
  SearchFixtureData data;
  data.build(distribution, fluid_points);
  data.motion = motion;
  for (auto _ : state) {
    data.advance();
    data.search->run();
    if (consume) {
      const int64_t total = consume_neighbors(data);
      benchmark::DoNotOptimize(&total);
    }
  }
  attach_counters(state, data);
}

void api_resize(benchmark::State &state, Distribution distribution,
                int fluid_points) {
  SearchFixtureData data;
  data.build(distribution, fluid_points);
  bool shrink = true;
  for (auto _ : state) {
    const int count = shrink ? (fluid_points * 99) / 100 : fluid_points;
    shrink = !shrink;
    data.search->resize_point_set(data.fluid_id, data.scene.fluid_xyz.data(),
                                  count);
    data.search->run();
  }
  attach_counters(state, data);
}

void api_setup(benchmark::State &state, Distribution distribution,
               int fluid_points) {
  const Scene scene = make_scene(distribution, fluid_points);
  for (auto _ : state) {
    NSearch search(scene.radius);
    const int fluid = search.add_point_set(
        scene.fluid_xyz.data(), static_cast<int>(scene.fluid.x.size()));
    const int boundary = search.add_point_set(
        scene.boundary_xyz.data(), static_cast<int>(scene.boundary.x.size()),
        true);
    const int emitter = search.add_point_set(
        scene.emitter_xyz.data(), static_cast<int>(scene.emitter.x.size()));
    search.set_all_searches(false);
    search.set_active_search(fluid, fluid, true);
    search.set_active_search(fluid, boundary, true);
    search.set_active_search(fluid, emitter, true);
    search.run();
    benchmark::DoNotOptimize(search.get_neighbors(fluid, fluid, 0));
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(scene.fluid.x.size()));
}

const char *distribution_label(Distribution distribution) {
  switch (distribution) {
  case Distribution::Uniform: return "uniform";
  case Distribution::JitteredLattice: return "jittered";
  case Distribution::Clustered: return "clustered";
  case Distribution::Slab: return "slab";
  case Distribution::Lattice: return "lattice";
  case Distribution::Pathological: return "pathological";
  case Distribution::HugeDomain: return "huge_domain";
  case Distribution::Surface: return "surface";
  case Distribution::ThinPipe: return "thin_pipe";
  case Distribution::TwoScale: return "two_scale";
  case Distribution::SlabThin: return "thin_slab";
  }
  return "unknown";
}

std::string scene_label(Distribution distribution, int points) {
  std::string name = distribution_label(distribution);
  name += "_" + (points % 1000000 == 0
                     ? std::to_string(points / 1000000) + "m"
                     : std::to_string(points / 1000) + "k");
  return name;
}

void register_frame(const char *motion_name, Motion motion,
                    Distribution distribution, int points) {
  benchmark::RegisterBenchmark(
      std::string("Api/Frame/") + motion_name + "/" +
          scene_label(distribution, points),
      api_frame, distribution, points, motion, false)
      ->Unit(benchmark::kMillisecond);
}

} // namespace

void register_api_benchmarks() {
  constexpr Distribution kUniform = Distribution::Uniform;
  constexpr Distribution kJittered = Distribution::JitteredLattice;
  constexpr Distribution kClustered = Distribution::Clustered;
  constexpr Distribution kSlab = Distribution::Slab;
  constexpr Distribution kLattice = Distribution::Lattice;
  constexpr Distribution kPathological = Distribution::Pathological;
  constexpr Distribution kHugeDomain = Distribution::HugeDomain;
  constexpr Distribution kSurface = Distribution::Surface;
  constexpr Distribution kThinPipe = Distribution::ThinPipe;
  constexpr Distribution kTwoScale = Distribution::TwoScale;

  for (const auto &[distribution, points] :
       std::initializer_list<std::pair<Distribution, int>>{
           {kUniform, 50000},      {kUniform, 200000},
           {kUniform, 1000000},    {kJittered, 50000},
           {kJittered, 200000},    {kClustered, 50000},
           {kClustered, 200000},   {kSlab, 50000},
           {kSlab, 200000},        {kLattice, 50000},
           {kPathological, 10000}, {kHugeDomain, 50000},
           {kHugeDomain, 200000},  {kSurface, 50000},
           {kSurface, 200000},     {kThinPipe, 50000},
           {kThinPipe, 200000},    {kTwoScale, 50000},
           {kTwoScale, 200000}}) {
    register_frame("static", Motion::Static, distribution, points);
  }
  for (const auto &[distribution, points] :
       std::initializer_list<std::pair<Distribution, int>>{
           {kUniform, 50000},    {kUniform, 200000},    {kJittered, 50000},
           {kJittered, 200000},  {kSlab, 50000},       {kSlab, 200000}}) {
    register_frame("small_move", Motion::SmallMove, distribution, points);
  }
  for (const auto &[distribution, points] :
       std::initializer_list<std::pair<Distribution, int>>{
           {kUniform, 50000}, {kUniform, 200000}, {kSlab, 50000},
           {kSlab, 200000}}) {
    register_frame("large_move", Motion::LargeMove, distribution, points);
  }

  for (int points : {50000, 200000}) {
    benchmark::RegisterBenchmark(
        "Api/Consume/" + scene_label(kUniform, points), api_frame, kUniform,
        points, Motion::SmallMove, true)
        ->Unit(benchmark::kMillisecond);
  }
  benchmark::RegisterBenchmark("Api/Resize/uniform_200k", api_resize, kUniform,
                               200000)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("Api/Setup/uniform_200k", api_setup, kUniform,
                               200000)
      ->Unit(benchmark::kMillisecond);
}

} // namespace apfrnn::bench
