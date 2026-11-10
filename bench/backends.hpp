#pragma once

// A backend is set up once per scene, then run() is timed once per frame.
// Add a class and an entry in available_backends() to compare a new one.

#include "scenes.hpp"

#include "apfrnn/neighbor_search.hpp"

#include <benchmark/benchmark.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef APFRNN_BENCH_BASELINES
#include <CompactNSearch.h>
#include <TreeNSearch>
#endif

namespace apfrnn::bench {

// One density-style pass over all lists: each neighbor's position is fetched
// and a kernel evaluated, so list layout and order affect the cost.
// `pair_reader(query, target)` is called once per active pair and returns a
// reader; `reader(point, visit)` calls visit(neighbor) for every neighbor.
// Each backend reads through its library's own per-pair access, as a
// simulation looping over a pair's points would.
template <typename PairReader>
inline double density_pass(const SceneData &scene,
                           const PairReader &pair_reader) {
  const float h2 = scene.radius * scene.radius;
  double total = 0.0;
  for (const auto &[query, target] : scene.pairs) {
    const float *q_xyz = scene.xyz[static_cast<std::size_t>(query)].data();
    const float *t_xyz = scene.xyz[static_cast<std::size_t>(target)].data();
    const auto reader = pair_reader(query, target);
    total += tbb::parallel_reduce(
        tbb::blocked_range<int>(0, scene.points(query), 1024), 0.0,
        [&](const tbb::blocked_range<int> &range, double sum) {
          for (int point = range.begin(); point != range.end(); ++point) {
            const float *q = q_xyz + 3 * static_cast<std::size_t>(point);
            float density = 0.0f;
            reader(point, [&](uint32_t neighbor) {
              const float *t = t_xyz + 3 * static_cast<std::size_t>(neighbor);
              const float dx = t[0] - q[0], dy = t[1] - q[1], dz = t[2] - q[2];
              const float w = std::max(0.0f, h2 - dx * dx - dy * dy - dz * dz);
              density += w * w * w;
            });
            sum += density;
          }
          return sum;
        },
        std::plus<double>());
  }
  return total;
}

class Backend {
public:
  virtual ~Backend() = default;
  // The scene outlives the backend and its buffers move in place between
  // frames.
  virtual void setup(const SceneData &scene) = 0;
  virtual void run() = 0;
  // Pointer/count changes for a growing point set. Called between frames when
  // SceneData::growing is set.
  virtual void resize(int set, const float *xyz, int n) = 0;
  // Neighbors of `point` of set `query` in set `target`, in any order.
  virtual void neighbors(int query, int target, int point,
                         std::vector<uint32_t> &out) const = 0;
  // Reads every neighbor list the way a density pass does and returns the sum.
  virtual double read_lists(const SceneData &scene) const = 0;
  // Backend-specific counters, recorded after the timed frames.
  virtual void add_counters(benchmark::State &) const {}
};

struct BackendSpec {
  std::string name;
  std::function<std::unique_ptr<Backend>()> make;
  bool uses_cpu_threads = true;
  // False for baselines whose repeated-resize path is not correct enough to
  // publish a growing-scene comparison. CompactNSearch currently drops a few
  // pairs after many growth steps (verified against brute force), so it is
  // excluded from Growing/ rather than silently compared.
  bool supports_growing = true;
};

// Set by --memory_budget_mb=N; 0 keeps the library default.
inline int64_t &benchmark_memory_budget_mb() {
  static int64_t mb = 0;
  return mb;
}

class ApfrnnBackend final : public Backend {
public:
  explicit ApfrnnBackend(SearchBackend backend) : backend_(backend) {}

  void setup(const SceneData &scene) override {
    search_ = std::make_unique<NSearch>(scene.radius);
    search_->set_search_backend(backend_);
    if (benchmark_memory_budget_mb() > 0) {
      search_->set_memory_budget(benchmark_memory_budget_mb() << 20);
    }
    for (std::size_t set = 0; set < scene.xyz.size(); ++set) {
      search_->add_point_set(scene.xyz[set].data(),
                             scene.points(static_cast<int>(set)),
                             scene.is_static[set]);
    }
    search_->set_all_searches(false);
    for (const auto &[query, target] : scene.pairs) {
      search_->set_active_search(query, target, true);
    }
  }

  void run() override { search_->run(); }

  void resize(int set, const float *xyz, int n) override {
    search_->resize_point_set(set, xyz, n);
  }

  void neighbors(int query, int target, int point,
                 std::vector<uint32_t> &out) const override {
    const NeighborList list = search_->get_neighbors(query, target, point);
    out.assign(list.begin(), list.end());
  }

  double read_lists(const SceneData &scene) const override {
    return density_pass(scene, [this](int query, int target) {
      return [pair = search_->neighbors(query, target)](int point,
                                                        auto &&visit) {
        for (uint32_t neighbor : pair[point]) {
          visit(neighbor);
        }
      };
    });
  }

  void add_counters(benchmark::State &state) const override {
    const NSearch::RunStats stats = search_->last_stats();
    state.counters["BuildMs"] = stats.sync_build_ms;
    state.counters["GridMs"] = stats.grid_ms;
    state.counters["SameBuildMs"] = stats.same_build_ms;
    state.counters["SameMs"] = stats.same_set_ms;
    state.counters["CrossMs"] = stats.cross_set_ms;
    const int set_count = search_->n_sets();
    for (int query = 0; query < set_count; ++query) {
      for (int target = 0; target < set_count; ++target) {
        const std::size_t index =
            static_cast<std::size_t>(query) * set_count + target;
        if (index < stats.pair_ms.size() && stats.pair_ms[index] > 0.0) {
          char name[32];
          std::snprintf(name, sizeof(name), "PairMs%d_%d", query, target);
          state.counters[name] = stats.pair_ms[index];
        }
      }
    }
    state.counters["NeighborMB"] =
        static_cast<double>(stats.neighbor_bytes) / (1024.0 * 1024.0);
    state.counters["FixedOverflows"] = stats.fixed_overflows;
    state.counters["MaxNeighbors"] = stats.max_neighbors;
    // Path bits are diagnostics, not a stable API.
    for (uint64_t bit = 1; bit != 0; bit <<= 1) {
      if ((stats.paths & bit) != 0) {
        state.counters[std::string("Path") + NSearch::path_name(bit)] = 1;
      }
    }
    // A GPU run that fell back to the CPU would otherwise look like GPU time.
    state.counters["Fallback"] = search_->search_backend() != backend_;
  }

private:
  SearchBackend backend_;
  std::unique_ptr<NSearch> search_;
};

#ifdef APFRNN_BENCH_BASELINES

class TreeBackend final : public Backend {
public:
  void setup(const SceneData &scene) override {
    search_ = std::make_unique<tns::TreeNSearch>();
    search_->set_search_radius(scene.radius);
    for (std::size_t set = 0; set < scene.xyz.size(); ++set) {
      search_->add_point_set(scene.xyz[set].data(),
                             scene.points(static_cast<int>(set)));
    }
    search_->set_all_searches(false);
    for (const auto &[query, target] : scene.pairs) {
      search_->set_active_search(query, target, true);
    }
  }

  void run() override { search_->run(); }

  void resize(int set, const float *xyz, int n) override {
    search_->resize_point_set(set, xyz, n);
  }

  void neighbors(int query, int target, int point,
                 std::vector<uint32_t> &out) const override {
    const tns::NeighborList list =
        search_->get_neighborlist(query, target, point);
    out.resize(static_cast<std::size_t>(list.size()));
    for (int k = 0; k < list.size(); ++k) {
      out[static_cast<std::size_t>(k)] = static_cast<uint32_t>(list[k]);
    }
  }

  double read_lists(const SceneData &scene) const override {
    return density_pass(scene, [this](int query, int target) {
      return [this, query, target](int point, auto &&visit) {
        const tns::NeighborList list =
            search_->get_neighborlist(query, target, point);
        for (int k = 0; k < list.size(); ++k) {
          visit(static_cast<uint32_t>(list[k]));
        }
      };
    });
  }

private:
  std::unique_ptr<tns::TreeNSearch> search_;
};

class CompactBackend final : public Backend {
public:
  void setup(const SceneData &scene) override {
    using CompactNSearch::Real;
    search_ = std::make_unique<CompactNSearch::NeighborhoodSearch>(
        static_cast<Real>(scene.radius));
    for (std::size_t set = 0; set < scene.xyz.size(); ++set) {
      search_->add_point_set(
          reinterpret_cast<const Real *>(scene.xyz[set].data()),
          static_cast<std::size_t>(scene.points(static_cast<int>(set))),
          !scene.is_static[set], true, true);
    }
    search_->set_active(false);
    for (const auto &[query, target] : scene.pairs) {
      search_->set_active(static_cast<unsigned int>(query),
                          static_cast<unsigned int>(target), true);
    }
  }

  void run() override { search_->find_neighbors(); }

  void resize(int set, const float *xyz, int n) override {
    search_->resize_point_set(
        static_cast<unsigned int>(set),
        reinterpret_cast<const CompactNSearch::Real *>(xyz),
        static_cast<std::size_t>(n));
  }

  void neighbors(int query, int target, int point,
                 std::vector<uint32_t> &out) const override {
    const auto &set = search_->point_set(static_cast<unsigned int>(query));
    const auto p = static_cast<unsigned int>(point);
    const auto t = static_cast<unsigned int>(target);
    out.resize(set.n_neighbors(t, p));
    for (unsigned int k = 0; k < out.size(); ++k) {
      out[k] = set.neighbor(t, p, k);
    }
  }

  double read_lists(const SceneData &scene) const override {
    return density_pass(scene, [this](int query, int target) {
      const auto &set = search_->point_set(static_cast<unsigned int>(query));
      return [&set, t = static_cast<unsigned int>(target)](int point,
                                                           auto &&visit) {
        const auto p = static_cast<unsigned int>(point);
        const unsigned int count = set.n_neighbors(t, p);
        for (unsigned int k = 0; k < count; ++k) {
          visit(set.neighbor(t, p, k));
        }
      };
    });
  }

private:
  std::unique_ptr<CompactNSearch::NeighborhoodSearch> search_;
};

#endif

// Backends usable in this build and on this machine: GPU variants only with
// the SYCL backend and a GPU, the baselines only with APFRNN_BUILD_BASELINES.
inline std::vector<BackendSpec> available_backends() {
  auto apfrnn = [](SearchBackend backend) {
    return [backend] { return std::make_unique<ApfrnnBackend>(backend); };
  };
  std::vector<BackendSpec> backends = {
      {"apfrnn_cpu", apfrnn(SearchBackend::CpuExact)}};
  if (NSearch(1.0f).gpu_available()) {
    backends.push_back(
        {"apfrnn_gpu", apfrnn(SearchBackend::GpuExact), false});
    backends.push_back({"apfrnn_auto", apfrnn(SearchBackend::Auto), false});
  }
#ifdef APFRNN_BENCH_BASELINES
  backends.push_back({"tree", [] { return std::make_unique<TreeBackend>(); }});
  backends.push_back(
      {"compact", [] { return std::make_unique<CompactBackend>(); }, true,
       /*supports_growing=*/false});
#endif
  return backends;
}

} // namespace apfrnn::bench
