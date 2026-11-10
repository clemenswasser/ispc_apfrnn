// Frame-time benchmark of every neighbor-search backend on the headline and
// stress scenes, plus the NSearch-API scenarios (Api/...).
//
// Replay/<backend>/<trace> is the headline family: it replays searches captured
// from a simulation, one forward and backward pass per run. Verify/ runs the
// same frames untimed and checks the lists against an independent reference.
// All timed families measure run() plus one density pass over the lists. Scene
// updates are not timed.
//
// Benchmarks are named Frame/<backend>/<scene>, Stress/<backend>/<scene>,
// Threads/<backend>/<scene>/<threads>, Cold/<backend>/<scene> and
// Api/<scenario>/<scene>, so --benchmark_filter selects by family, backend or
// scene. Each run times a fixed window of frames after two warm-up frames
// (see measured_frames()), so every backend and every tool times the same
// frames whatever --benchmark_min_time is; the reported time is per frame.
// The second warm-up frame is validated (see validate()), so plans that change
// after the first frame
// are covered: the label carries a checksum of the neighbor pairs clearly
// within the radius, which must be equal for all backends on a scene, and
// FalsePositives must be 0. Stress holds the regimes outside the headline
// matrix (large high-degree sets, many active pairs, sparse hash-grid
// droplets). Cold times setup plus the first frame on a fresh backend.
//
//   --learnsph_res=DIR       learnsph/res with the saved frames (default
//                            ../learnsph/res); scenes whose files are missing
//                            are skipped.
//   --traces=DIR             captured traces (default traces)
//   --memory_budget_mb=N     cap on one fixed-row allocation (library default
//                            when omitted); useful to compare CSR fallback.

#include "api_bench.hpp"
#include "backends.hpp"
#include "reference.hpp"
#include "scenes.hpp"

#include <benchmark/benchmark.h>
#include <tbb/global_control.h>
#ifdef APFRNN_BENCH_BASELINES
#include <omp.h>
#endif
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using apfrnn::bench::Backend;
using apfrnn::bench::BackendSpec;
using apfrnn::bench::ReferenceSummary;
using apfrnn::bench::SceneData;
using apfrnn::bench::SceneSpec;

// Default worker count for Frame/Stress/Cold runs. `--threads=N` overrides it;
// Threads/ benchmarks always carry their own count. Keeping the default
// explicit makes TBB and OpenMP run the same number of workers instead of
// leaving each pool to size itself from the machine.
int64_t &benchmark_threads();

// The last scene built. Benchmarks are registered scene by scene, so all
// backends of a scene reuse it; each run copies it because frames move it.
const SceneData &cached_scene(const SceneSpec &spec) {
  static std::string name;
  static SceneData data;
  if (name != spec.name) {
    data = spec.make();
    if (spec.zsorted) {
      apfrnn::bench::zsort_scene(data);
    }
    // Every benchmark scene moves every frame. Static benchmarks are not a
    // representation of an SPH simulation and are intentionally impossible.
    if (!data.trace && !data.growing &&
        data.motion == apfrnn::bench::Motion::Static && data.jitter == 0.0f) {
      data.flow = 0.02f;
      float low = std::numeric_limits<float>::max();
      float high = std::numeric_limits<float>::lowest();
      for (std::size_t i = 0; i < data.xyz[0].size(); i += 3) {
        low = std::min(low, data.xyz[0][i]);
        high = std::max(high, data.xyz[0][i]);
      }
      data.flow_wavenumber =
          high > low ? 6.2831853f / (high - low) : 1.0f;
    }
    name = spec.name;
  }
  return data;
}

// Default worker count for Frame/Stress/Cold runs. `--threads=N` overrides it;
// Threads/ benchmarks always carry their own count. Keeping the default
// explicit makes TBB and OpenMP run the same number of workers instead of
// leaving each pool to size itself from the machine.
int64_t &benchmark_threads() {
  static int64_t threads = 8;
  return threads;
}

// Limits both thread pools: TBB (apfrnn) and OpenMP (the baselines).
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

// Every validated backend result is recorded by scene and backend. After the
// run, the harness requires all backends of a scene to agree on the checksum
// and, when the scene is small enough for the independent reference, to agree
// with the reference pair set. A backend that returns nothing no longer
// passes silently. Results are kept per run so a thread-count or family
// variant cannot silently overwrite a disagreement.
std::map<std::string,
         std::vector<std::pair<std::string, Validation>>> &
verified_results() {
  static std::map<std::string,
                  std::vector<std::pair<std::string, Validation>>>
      results;
  return results;
}

std::map<std::string, ReferenceSummary> &reference_cache() {
  static std::map<std::string, ReferenceSummary> references;
  return references;
}

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

int &verify_failures() {
  static int failures = 0;
  return failures;
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
  // Excludes the mapped trace files, which would otherwise dominate replays.
  std::FILE *status = std::fopen("/proc/self/status", "r");
  if (status != nullptr) {
    char line[128];
    long kilobytes = 0;
    while (std::fgets(line, sizeof(line), status) != nullptr) {
      if (std::sscanf(line, "RssAnon: %ld", &kilobytes) == 1) {
        state.counters["AnonMB"] = static_cast<double>(kilobytes) / 1024.0;
        break;
      }
    }
    std::fclose(status);
  }
}

// Switches perf counters on and off around the timed frames. perf runs with
// -D -1 and --control=fifo:CTL,ACK; APFRNN_PERF_CTL and APFRNN_PERF_ACK name the
// same FIFOs. Does nothing outside scripts/perf_evidence.sh.
void perf_region(const char *command) {
  static const char *ctl_path = std::getenv("APFRNN_PERF_CTL");
  static const char *ack_path = std::getenv("APFRNN_PERF_ACK");
  if (ctl_path == nullptr || ack_path == nullptr) {
    return;
  }
  static const int ctl = open(ctl_path, O_WRONLY);
  static const int ack = open(ack_path, O_RDONLY);
  if (ctl < 0 || ack < 0) {
    return;
  }
  if (write(ctl, command, std::strlen(command)) > 0) {
    char reply[8];
    (void)!read(ack, reply, sizeof(reply));
  }
}

void advance_scene(SceneData &scene, Backend &backend, int step) {
  scene.advance(step);
  if (scene.growing) {
    backend.resize(0, scene.xyz[0].data(), scene.points(0));
  }
  for (const int set : scene.resized) {
    backend.resize(set, scene.xyz[static_cast<std::size_t>(set)].data(),
                   scene.points(set));
  }
}

// Frames timed per run. The window depends only on the scene, so every backend
// and every tool (README tables, paired A/B runs) times the same frames: the
// motion and the decay of memory order depend on how many frames ran, and a
// growing set's size depends on it directly. The window holds about 2^29 units
// of work, where a point costs 16 units and a neighbor pair one, rounded down
// to a power of two between 4 and 256 frames. A growing scene runs until it
// reaches its limit and 8 frames beyond. A replay runs whole forward and
// backward passes, enough of them to fill the window.
int measured_frames(const SceneData &scene, int64_t neighbors) {
  int64_t points = 0;
  for (std::size_t set = 0; set < scene.xyz.size(); ++set) {
    points += scene.points(static_cast<int>(set));
  }
  const double work = 16.0 * static_cast<double>(points) +
                      static_cast<double>(std::max<int64_t>(0, neighbors));
  int frames = 4;
  while (frames < 256 && 2.0 * frames * work <= double(1 << 29)) {
    frames *= 2;
  }
  if (scene.trace) {
    const int cycle = scene.trace->cycle();
    return std::max(1, (frames + cycle - 1) / cycle) * cycle;
  }
  if (scene.growing && scene.grow_per_frame > 0) {
    const int remaining = std::max(0, scene.grow_limit - scene.points(0));
    frames = std::max(frames,
                      (remaining + scene.grow_per_frame - 1) /
                              scene.grow_per_frame + 8);
  }
  return frames;
}

void frames(benchmark::State &state, const SceneSpec &spec,
            const BackendSpec &backend_spec, int threads) {
  SceneData scene = cached_scene(spec);
  const int thread_count =
      threads > 0 ? threads : static_cast<int>(benchmark_threads());
  const ThreadLimit limit(thread_count);
  std::unique_ptr<Backend> backend = backend_spec.make();
  backend->setup(scene);
  // Two warm-up frames. Row capacities and some plans adapt after the first
  // frame, so the second one is the one validated.
  backend->run();
  advance_scene(scene, *backend, 1);
  backend->run();

  // Benchmark functions run several times per benchmark; validate once.
  static std::map<std::string, Validation> validated;
  auto found = validated.find(state.name());
  if (found == validated.end()) {
    found = validated.emplace(state.name(), validate(*backend, scene)).first;
  }
  const Validation &validation = found->second;

  auto &references = reference_cache();
  auto reference = references.find(spec.name);
  if (reference == references.end()) {
    reference = references.emplace(spec.name, reference_summary(scene)).first;
  }
  const ReferenceSummary &reference_summary = reference->second;

  auto &results = verified_results();
  results[spec.name].push_back({backend_spec.name, validation});
  if (reference_summary.complete) {
    const bool checksum_matches =
        validation.checksum == reference_summary.checksum;
    const bool count_in_range =
        validation.neighbors >= reference_summary.exact_pairs &&
        validation.neighbors <=
            reference_summary.exact_pairs + reference_summary.band_pairs;
    if (!checksum_matches || !count_in_range) {
      std::fprintf(stderr,
                   "REFERENCE MISMATCH scene=%s backend=%s checksum=%016llx "
                   "ref=%016llx neighbors=%lld exact=%lld band=%lld\n",
                   spec.name.c_str(), backend_spec.name.c_str(),
                   static_cast<unsigned long long>(validation.checksum),
                   static_cast<unsigned long long>(reference_summary.checksum),
                   static_cast<long long>(validation.neighbors),
                   static_cast<long long>(reference_summary.exact_pairs),
                   static_cast<long long>(reference_summary.band_pairs));
      std::exit(2);
    }
  }

  // The exact reference count when there is one, so the window does not
  // depend on the backend; larger scenes use the validated count.
  const int window = measured_frames(
      scene, reference_summary.complete
                 ? reference_summary.exact_pairs + reference_summary.band_pairs
                 : validation.neighbors);
  int step = 2;
  double run_seconds = 0.0;
  double read_seconds = 0.0;
  double read_sum = 0.0;
  for (auto _ : state) {
    double window_seconds = 0.0;
    for (int frame = 0; frame < window; ++frame) {
      advance_scene(scene, *backend, step++);
      perf_region("enable\n");
      const auto start = std::chrono::steady_clock::now();
      backend->run();
      const auto built = std::chrono::steady_clock::now();
      read_sum += backend->read_lists(scene);
      const auto done = std::chrono::steady_clock::now();
      perf_region("disable\n");
      const double run_time =
          std::chrono::duration<double>(built - start).count();
      const double read_time =
          std::chrono::duration<double>(done - built).count();
      run_seconds += run_time;
      read_seconds += read_time;
      window_seconds += run_time + read_time;
    }
    state.SetIterationTime(window_seconds / window);
  }
  benchmark::DoNotOptimize(read_sum);
  const double frame_count =
      static_cast<double>(window) * std::max<double>(1.0, state.iterations());
  state.counters["RunMs"] = 1000.0 * run_seconds / frame_count;
  state.counters["ReadMs"] = 1000.0 * read_seconds / frame_count;
  state.counters["Frames"] = window;

  const int fluid_points = scene.points(0);
  state.SetItemsProcessed(state.iterations() * window * fluid_points);
  state.counters["Points"] = fluid_points;
  // Particles that move: every set that is not static. The replay table
  // reports these and their neighbors over all timed pairs.
  int moving_points = 0;
  for (int set = 0; set < static_cast<int>(scene.xyz.size()); ++set) {
    if (set >= static_cast<int>(scene.is_static.size()) ||
        !scene.is_static[static_cast<std::size_t>(set)]) {
      moving_points += scene.points(set);
    }
  }
  state.counters["MovingPoints"] = moving_points;
  state.counters["CpuThreads"] = thread_count;
  state.counters["Neighbors"] = static_cast<double>(validation.neighbors);
  state.counters["FalsePositives"] =
      static_cast<double>(validation.false_positives);
  state.counters["AvgNeighbors"] =
      static_cast<double>(validation.same_set_neighbors) /
      std::max(1, fluid_points);
  if (reference_summary.complete) {
    state.counters["RefPairs"] =
        static_cast<double>(reference_summary.exact_pairs);
    state.counters["RefBandPairs"] =
        static_cast<double>(reference_summary.band_pairs);
  }
  char label[40];
  std::snprintf(label, sizeof(label), "checksum=%016llx",
                static_cast<unsigned long long>(validation.checksum));
  state.SetLabel(label);
  backend->add_counters(state);
  attach_rss(state);
}

// Runs every frame of a trace and checks the lists after each checked frame
// against the reference. Large scenes check a subset of frames, but every frame
// still runs, so state built up over the trace is what gets checked.
void verify_frames(benchmark::State &state, const SceneSpec &spec,
                   const BackendSpec &backend_spec) {
  SceneData scene = cached_scene(spec);
  const ThreadLimit limit(static_cast<int>(benchmark_threads()));
  std::unique_ptr<Backend> backend = backend_spec.make();
  backend->setup(scene);
  backend->run();
  const int total = scene.trace->cycle() + 1;
  const int stride = std::max(1, scene.trace->max_points() / 100000);
  int checked = 0;
  for (auto _ : state) {
    for (int step = 1; step <= total; ++step) {
      advance_scene(scene, *backend, step);
      backend->run();
      if (step % stride != 0 && step != total) {
        continue;
      }
      const Validation validation = validate(*backend, scene);
      const ReferenceSummary expected = apfrnn::bench::reference_summary(scene);
      ++checked;
      if (validation.false_positives != 0 ||
          (expected.complete &&
           (validation.checksum != expected.checksum ||
            validation.neighbors < expected.exact_pairs ||
            validation.neighbors >
                expected.exact_pairs + expected.band_pairs))) {
        std::fprintf(stderr,
                     "VERIFY FAILURE scene=%s backend=%s step=%d "
                     "false_positives=%lld neighbors=%lld exact=%lld band=%lld\n",
                     spec.name.c_str(), backend_spec.name.c_str(), step,
                     static_cast<long long>(validation.false_positives),
                     static_cast<long long>(validation.neighbors),
                     static_cast<long long>(expected.exact_pairs),
                     static_cast<long long>(expected.band_pairs));
        verify_failures() += 1;
        break;
      }
    }
  }
  state.counters["CheckedFrames"] = checked;
}

// Setup plus the first run() on a fresh backend: row capacities start from
// their defaults and buffers are freshly allocated. Destruction is not timed.
void cold_frame(benchmark::State &state, const SceneSpec &spec,
                const BackendSpec &backend_spec) {
  const SceneData &scene = cached_scene(spec);
  const int thread_count = static_cast<int>(benchmark_threads());
  const ThreadLimit limit(thread_count);
  for (auto _ : state) {
    std::unique_ptr<Backend> backend = backend_spec.make();
    backend->setup(scene);
    backend->run();
    state.PauseTiming();
    backend.reset();
    state.ResumeTiming();
  }
  state.counters["CpuThreads"] = thread_count;
  attach_rss(state);
}

bool verify_all() {
  bool ok = verify_failures() == 0;
  int scenes = 0;
  int results = 0;
  for (const auto &[scene_name, entries] : verified_results()) {
    ++scenes;
    const auto reference = reference_cache().find(scene_name);
    const bool has_reference =
        reference != reference_cache().end() && reference->second.complete;
    uint64_t checksum = 0;
    bool have_checksum = false;
    for (const auto &[backend_name, validation] : entries) {
      ++results;
      if (validation.false_positives != 0) {
        std::fprintf(stderr,
                     "BENCHMARK FAILURE scene=%s backend=%s false_positives=%lld\n",
                     scene_name.c_str(), backend_name.c_str(),
                     static_cast<long long>(validation.false_positives));
        ok = false;
      }
      if (has_reference) {
        const ReferenceSummary &reference_summary = reference->second;
        if (validation.checksum != reference_summary.checksum ||
            validation.neighbors < reference_summary.exact_pairs ||
            validation.neighbors >
                reference_summary.exact_pairs + reference_summary.band_pairs) {
          std::fprintf(stderr,
                       "BENCHMARK FAILURE scene=%s backend=%s reference\n",
                       scene_name.c_str(), backend_name.c_str());
          ok = false;
        }
      }
      if (!have_checksum) {
        checksum = validation.checksum;
        have_checksum = true;
      } else if (validation.checksum != checksum) {
        std::fprintf(stderr,
                     "BENCHMARK FAILURE scene=%s backend=%s checksum disagrees "
                     "with another backend\n",
                     scene_name.c_str(), backend_name.c_str());
        ok = false;
      }
    }
  }
  std::fprintf(stderr,
               "Benchmark verification: %d scenes, %d backend results, %s\n",
               scenes, results, ok ? "all checksums match" : "FAILED");
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  std::string learnsph_res = "../learnsph/res";
  std::string traces = "traces";
  std::vector<char *> args;
  for (int i = 0; i < argc; ++i) {
    constexpr const char *kRes = "--learnsph_res=";
    constexpr const char *kBudget = "--memory_budget_mb=";
    constexpr const char *kThreads = "--threads=";
    constexpr const char *kTraces = "--traces=";
    if (std::strncmp(argv[i], kTraces, std::strlen(kTraces)) == 0) {
      traces = argv[i] + std::strlen(kTraces);
    } else if (std::strncmp(argv[i], kRes, std::strlen(kRes)) == 0) {
      learnsph_res = argv[i] + std::strlen(kRes);
    } else if (std::strncmp(argv[i], kBudget, std::strlen(kBudget)) == 0) {
      apfrnn::bench::benchmark_memory_budget_mb() =
          std::stoll(argv[i] + std::strlen(kBudget));
    } else if (std::strncmp(argv[i], kThreads, std::strlen(kThreads)) == 0) {
      benchmark_threads() = std::stoll(argv[i] + std::strlen(kThreads));

    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      benchmark_threads() = std::stoll(argv[++i]);
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

  auto register_family = [&](const char *family,
                             const std::vector<SceneSpec> &family_scenes,
                             bool require_growing = false) {
    for (const SceneSpec &scene : family_scenes) {
      for (const BackendSpec &backend : backends) {
        if (require_growing && !backend.supports_growing) {
          continue;
        }
        benchmark::RegisterBenchmark(
            std::string(family) + "/" + backend.name + "/" + scene.name, frames,
            scene, backend, 0)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
      }
    }
  };
  // Replay measures whole forward and backward passes (see measured_frames()),
  // so every backend times the same searches.
  for (const SceneSpec &scene : apfrnn::bench::trace_scenes(traces)) {
    const SceneData &data = cached_scene(scene);
    if (!data.trace) {
      continue;
    }
    const bool resizes = data.trace->resizes();
    for (const BackendSpec &backend : backends) {
      if (resizes && !backend.supports_growing) {
        continue;
      }
      benchmark::RegisterBenchmark("Replay/" + backend.name + "/" + scene.name,
                                   frames, scene, backend, 0)
          ->Unit(benchmark::kMillisecond)
          ->UseManualTime()
          ->Iterations(1);
      benchmark::RegisterBenchmark("Verify/" + backend.name + "/" + scene.name,
                                   verify_frames, scene, backend)
          ->Unit(benchmark::kMillisecond)
          ->Iterations(1);
    }
  }
  register_family("Frame", scenes);
  register_family("Stress", apfrnn::bench::stress_scenes());
  register_family("Order", apfrnn::bench::order_scenes());
  register_family("Scale", apfrnn::bench::scale_scenes());
  register_family("Growing", apfrnn::bench::growing_scenes(),
                  /*require_growing=*/true);
  register_family("Dynamic", apfrnn::bench::dynamic_scenes(learnsph_res));
  for (const SceneSpec &scene : apfrnn::bench::cold_scenes()) {
    for (const BackendSpec &backend : backends) {
      benchmark::RegisterBenchmark("Cold/" + backend.name + "/" + scene.name,
                                   cold_frame, scene, backend)
          ->Unit(benchmark::kMillisecond)
          ->UseRealTime();
    }
  }
  // Thread scaling on the CPU backends: the 200k scene across the core counts,
  // 1M where the memory system starts to matter, a high-degree scene and a
  // small learnsph frame (when present), where per-frame fixed costs show up
  // at low thread counts. Scenes are registered in groups so cached_scene()
  // keeps the scene alive between backends.
  std::vector<std::pair<SceneSpec, std::vector<int>>> scaling = {
      {apfrnn::bench::synthetic_scene(apfrnn::bench::Distribution::Uniform,
                                      200000),
       {1, 2, 4, 8, 16}},
      {apfrnn::bench::synthetic_scene(apfrnn::bench::Distribution::Uniform,
                                      1000000),
       {8, 16}},
      {apfrnn::bench::synthetic_scene(apfrnn::bench::Distribution::Clustered,
                                      200000),
       {1, 4, 16}},
  };
  for (SceneSpec &scene : apfrnn::bench::learnsph_scenes(learnsph_res)) {
    if (scene.name == "learnsph_ramp_split_lv") {
      scaling.push_back({std::move(scene), {1, 4, 16}});
    }
  }
  // APFRNN_SCALING_THREADS=1,2,4,...,96 replaces every scene's thread
  // counts, for machines with more cores than the defaults reach.
  if (const char *env = std::getenv("APFRNN_SCALING_THREADS")) {
    std::vector<int> counts;
    for (const char *p = env; *p != '\0';) {
      char *end = nullptr;
      const long value = std::strtol(p, &end, 10);
      if (end == p) {
        break;
      }
      if (value > 0) {
        counts.push_back(static_cast<int>(value));
      }
      p = *end == ',' ? end + 1 : end;
    }
    if (!counts.empty()) {
      for (auto &entry : scaling) {
        entry.second = counts;
      }
    }
  }
  for (const auto &[scaling_scene, thread_counts] : scaling) {
    for (const BackendSpec &backend : backends) {
      if (!backend.uses_cpu_threads) {
        continue;
      }
      for (int threads : thread_counts) {
        benchmark::RegisterBenchmark("Threads/" + backend.name + "/" +
                                         scaling_scene.name + "/" +
                                         std::to_string(threads),
                                     frames, scaling_scene, backend, threads)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
      }
    }
  }

  apfrnn::bench::register_api_benchmarks();

  benchmark::RunSpecifiedBenchmarks();
  const bool verified = verify_all();
  benchmark::Shutdown();
  return verified ? 0 : 1;
}
