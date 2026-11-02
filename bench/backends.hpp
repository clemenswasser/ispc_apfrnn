#pragma once

// A backend is set up once per scene, then run() is timed once per frame.
// Add a class and an entry in available_backends() to compare a new one.

#include "scenes.hpp"

#include "apfrnn/neighbor_search.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef APFRNN_BENCH_BASELINES
#include <CompactNSearch.h>
#include <TreeNSearch>
#endif

namespace apfrnn::bench {

class Backend {
public:
  virtual ~Backend() = default;
  // The scene outlives the backend and its buffers move in place between
  // frames.
  virtual void setup(const SceneData &scene) = 0;
  virtual void run() = 0;
  // Neighbors of `point` of set `query` in set `target`, in any order.
  virtual void neighbors(int query, int target, int point,
                         std::vector<uint32_t> &out) const = 0;
  // Backend-specific counters, recorded after the timed frames.
  virtual void add_counters(benchmark::State &) const {}
};

struct BackendSpec {
  std::string name;
  std::function<std::unique_ptr<Backend>()> make;
  bool uses_cpu_threads = true;
};

class ApfrnnBackend final : public Backend {
public:
  explicit ApfrnnBackend(SearchBackend backend) : backend_(backend) {}

  void setup(const SceneData &scene) override {
    search_ = std::make_unique<NSearch>(scene.radius);
    search_->set_search_backend(backend_);
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

  void neighbors(int query, int target, int point,
                 std::vector<uint32_t> &out) const override {
    const NeighborList list = search_->get_neighbors(query, target, point);
    out.assign(list.begin(), list.end());
  }

  void add_counters(benchmark::State &state) const override {
    const NSearch::RunStats stats = search_->last_stats();
    state.counters["BuildMs"] = stats.sync_build_ms;
    state.counters["CrossMs"] = stats.cross_set_ms;
    state.counters["NeighborMB"] =
        static_cast<double>(stats.neighbor_bytes) / (1024.0 * 1024.0);
    state.counters["FixedOverflows"] = stats.fixed_overflows;
    state.counters["MaxNeighbors"] = stats.max_neighbors;
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

  void neighbors(int query, int target, int point,
                 std::vector<uint32_t> &out) const override {
    const tns::NeighborList list =
        search_->get_neighborlist(query, target, point);
    out.resize(static_cast<std::size_t>(list.size()));
    for (int k = 0; k < list.size(); ++k) {
      out[static_cast<std::size_t>(k)] = static_cast<uint32_t>(list[k]);
    }
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
      {"compact", [] { return std::make_unique<CompactBackend>(); }});
#endif
  return backends;
}

} // namespace apfrnn::bench
