// Frame-time benchmark of every neighbor-search backend on every scene.
//
// Benchmarks are named Frame/<backend>/<scene> and Threads/<backend>/
// uniform_200k/<threads>, so --benchmark_filter selects by backend or scene.
// Each run times frames after one warm-up frame. That first frame is also
// validated (see validate()): the label carries a checksum of the neighbor
// pairs clearly within the radius, which must be equal for all backends on a
// scene, and FalsePositives must be 0.
//
//   --learnsph_res=DIR  learnsph/res with the saved frames (default
//                       ../learnsph/res); scenes whose files are missing are
//                       skipped.

#include "backends.hpp"
#include "scenes.hpp"

#include <benchmark/benchmark.h>
#include <tbb/global_control.h>
#ifdef APFRNN_BENCH_BASELINES
#include <omp.h>
#endif
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using apfrnn::bench::Backend;
using apfrnn::bench::BackendSpec;
using apfrnn::bench::SceneData;
using apfrnn::bench::SceneSpec;

// The last scene built. Benchmarks are registered scene by scene, so all
// backends of a scene reuse it; each run copies it because frames move it.
const SceneData &cached_scene(const SceneSpec &spec) {
  static std::string name;
  static SceneData data;
  if (name != spec.name) {
    data = spec.make();
    name = spec.name;
  }
  return data;
}

// Limits both thread pools: TBB (apfrnn) and OpenMP (the baselines).
// 0 means all hardware threads.
class ThreadLimit {
public:
  explicit ThreadLimit(int threads) {
    if (threads > 0) {
      tbb_ = std::make_unique<tbb::global_control>(
          tbb::global_control::max_allowed_parallelism,
          static_cast<std::size_t>(threads));
    }
#ifdef APFRNN_BENCH_BASELINES
    omp_set_num_threads(threads > 0 ? threads : omp_get_num_procs());
#endif
  }

private:
  std::unique_ptr<tbb::global_control> tbb_;
};

uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

struct Validation {
  int64_t neighbors = 0;
  int64_t same_set_neighbors = 0;  // of the fluid, for the average degree
  int64_t false_positives = 0;     // pairs clearly farther than the radius
  uint64_t checksum = 0;           // of the pairs clearly within the radius
};

// Backends round the distance test differently, so pairs at almost exactly
// the radius may or may not be listed. Distances are recomputed in double:
// pairs clearly within the radius are hashed (every backend must list all of
// them, so the checksum is backend-independent), pairs clearly outside are
// false positives, and the band in between is ignored.
Validation validate(const Backend &backend, const SceneData &scene) {
  float max_abs = scene.radius;
  for (const std::vector<float> &xyz : scene.xyz) {
    for (float v : xyz) {
      max_abs = std::max(max_abs, std::fabs(v));
    }
  }
  const double r = scene.radius;
  const double band = 64.0 * std::numeric_limits<float>::epsilon() * max_abs;

  Validation result;
  std::vector<uint32_t> row;
  for (std::size_t pair = 0; pair < scene.pairs.size(); ++pair) {
    const auto [query, target] = scene.pairs[pair];
    const float *q_xyz = scene.xyz[static_cast<std::size_t>(query)].data();
    const float *t_xyz = scene.xyz[static_cast<std::size_t>(target)].data();
    for (int point = 0; point < scene.points(query); ++point) {
      backend.neighbors(query, target, point, row);
      const uint64_t key = mix((static_cast<uint64_t>(pair) << 32) |
                               static_cast<uint32_t>(point));
      const float *q = q_xyz + 3 * static_cast<std::size_t>(point);
      for (uint32_t neighbor : row) {
        const float *t = t_xyz + 3 * static_cast<std::size_t>(neighbor);
        const double dx = double(t[0]) - q[0], dy = double(t[1]) - q[1],
                     dz = double(t[2]) - q[2];
        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (d < r - band) {
          result.checksum += mix(key ^ neighbor);
        } else if (d > r + band) {
          ++result.false_positives;
        }
      }
      result.neighbors += static_cast<int64_t>(row.size());
      if (query == 0 && target == 0) {
        result.same_set_neighbors += static_cast<int64_t>(row.size());
      }
    }
  }
  // Pairs clearly within the radius that a backend did not list are not
  // seen here; they change the checksum.
  return result;
}

void attach_rss(benchmark::State &state) {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    state.counters["PeakRSSMB"] = static_cast<double>(usage.ru_maxrss) / 1024.0;
  }
  std::FILE *statm = std::fopen("/proc/self/statm", "r");
  if (statm != nullptr) {
    long total_pages = 0;
    long resident_pages = 0;
    if (std::fscanf(statm, "%ld %ld", &total_pages, &resident_pages) == 2) {
      const double page_mb =
          static_cast<double>(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
      state.counters["RSSMB"] = static_cast<double>(resident_pages) * page_mb;
    }
    std::fclose(statm);
  }
}

void frames(benchmark::State &state, const SceneSpec &spec,
            const BackendSpec &backend_spec, int threads) {
  SceneData scene = cached_scene(spec);
  const ThreadLimit limit(threads);
  std::unique_ptr<Backend> backend = backend_spec.make();
  backend->setup(scene);
  backend->run();  // the warm-up frame, which is the one validated

  // Benchmark functions run several times per benchmark; validate once.
  static std::map<std::string, Validation> validated;
  auto found = validated.find(state.name());
  if (found == validated.end()) {
    found = validated.emplace(state.name(), validate(*backend, scene)).first;
  }
  const Validation &validation = found->second;

  int step = 1;
  for (auto _ : state) {
    scene.advance(step++);
    backend->run();
  }

  const int fluid_points = scene.points(0);
  state.SetItemsProcessed(state.iterations() * fluid_points);
  state.counters["Neighbors"] = static_cast<double>(validation.neighbors);
  state.counters["FalsePositives"] =
      static_cast<double>(validation.false_positives);
  state.counters["AvgNeighbors"] =
      static_cast<double>(validation.same_set_neighbors) /
      std::max(1, fluid_points);
  char label[40];
  std::snprintf(label, sizeof(label), "checksum=%016llx",
                static_cast<unsigned long long>(validation.checksum));
  state.SetLabel(label);
  backend->add_counters(state);
  attach_rss(state);
}

} // namespace

int main(int argc, char **argv) {
  std::string learnsph_res = "../learnsph/res";
  std::vector<char *> args;
  for (int i = 0; i < argc; ++i) {
    constexpr const char *kRes = "--learnsph_res=";
    if (std::strncmp(argv[i], kRes, std::strlen(kRes)) == 0) {
      learnsph_res = argv[i] + std::strlen(kRes);
    } else {
      args.push_back(argv[i]);
    }
  }
  int count = static_cast<int>(args.size());
  benchmark::Initialize(&count, args.data());
  if (benchmark::ReportUnrecognizedArguments(count, args.data())) {
    return 1;
  }

  const std::vector<BackendSpec> backends =
      apfrnn::bench::available_backends();
  std::vector<SceneSpec> scenes = apfrnn::bench::synthetic_scenes();
  for (SceneSpec &scene : apfrnn::bench::learnsph_scenes(learnsph_res)) {
    scenes.push_back(std::move(scene));
  }

  for (const SceneSpec &scene : scenes) {
    for (const BackendSpec &backend : backends) {
      benchmark::RegisterBenchmark(
          "Frame/" + backend.name + "/" + scene.name, frames, scene, backend,
          0)
          ->Unit(benchmark::kMillisecond)
          ->UseRealTime();
    }
  }
  const SceneSpec scaling = apfrnn::bench::synthetic_scene(
      apfrnn::bench::Distribution::Uniform, 200000);
  for (const BackendSpec &backend : backends) {
    if (!backend.uses_cpu_threads) {
      continue;
    }
    for (int threads : {1, 2, 4, 8, 16}) {
      benchmark::RegisterBenchmark("Threads/" + backend.name + "/" +
                                       scaling.name + "/" +
                                       std::to_string(threads),
                                   frames, scaling, backend, threads)
          ->Unit(benchmark::kMillisecond)
          ->UseRealTime();
    }
  }

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
