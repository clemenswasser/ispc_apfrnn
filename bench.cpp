#include "apfrnn.hpp"
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct PointCloud {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
};

PointCloud make_point_cloud(int num_points, float side, uint32_t seed) {
  PointCloud cloud;
  cloud.x.resize(num_points);
  cloud.y.resize(num_points);
  cloud.z.resize(num_points);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(0.0f, side);
  for (int i = 0; i < num_points; ++i) {
    cloud.x[i] = dist(gen);
    cloud.y[i] = dist(gen);
    cloud.z[i] = dist(gen);
  }

  return cloud;
}

std::size_t
collect_concat_cross_neighbors(const apfrnn::NeighborSearchData &data,
                               int query_size) {
  std::vector<int> original_to_sorted(data.num_points);
  for (int sorted_index = 0; sorted_index < data.num_points; ++sorted_index) {
    original_to_sorted[data.original_index[sorted_index]] = sorted_index;
  }

  std::size_t total_neighbors = 0;
  for (int point_index = 0; point_index < query_size; ++point_index) {
    int sorted_index = original_to_sorted[point_index];
    int begin = data.row_ptr[sorted_index];
    int end = data.row_ptr[sorted_index + 1];
    for (int row_index = begin; row_index < end; ++row_index) {
      if (data.col_idx[row_index] >= query_size) {
        ++total_neighbors;
      }
    }
  }

  return total_neighbors;
}

} // namespace

class ISPCScalingFixture : public benchmark::Fixture {
public:
  int N;
  float R = 2.0f;
  apfrnn::NeighborSearchData neighbor_data;

  void SetUp(const ::benchmark::State &state) override {
    N = state.range(0);

    std::vector<float> X(N), Y(N), Z(N);
    std::mt19937 gen(42);
    float side = std::pow(N, 1.0f / 3.0f) * 2.0f;
    std::uniform_real_distribution<float> dist(0.0f, side);

    for (int i = 0; i < N; ++i) {
      X[i] = dist(gen);
      Y[i] = dist(gen);
      Z[i] = dist(gen);
    }

    neighbor_data = apfrnn::build_neighbor_search_data(X, Y, Z, R);
  }

  void TearDown(const benchmark::State &state) override { neighbor_data = {}; }
};

BENCHMARK_DEFINE_F(ISPCScalingFixture,
                   BM_WriteNeighbors)(benchmark::State &state) {
  for (auto _ : state) {
    apfrnn::write_neighbors_range(neighbor_data, 0, neighbor_data.num_cells);
    benchmark::DoNotOptimize(neighbor_data.col_idx.data());
  }

  state.SetItemsProcessed(state.iterations() * N);
  state.counters["TotalEdges"] = (double)neighbor_data.col_idx.size();
  state.counters["EdgesPerParticle"] = (double)neighbor_data.col_idx.size() / N;
}

BENCHMARK_REGISTER_F(ISPCScalingFixture, BM_WriteNeighbors)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20);

class MixedSetFixture : public benchmark::Fixture {
public:
  static constexpr int target_n = 78437;
  static constexpr int substeps_per_frame = 333;
  float R = 0.02f;
  PointCloud target;
  apfrnn::NeighborSearchData target_data;
  PointCloud query;

  void SetUp(const ::benchmark::State &state) override {
    int query_n = static_cast<int>(state.range(0));
    target = make_point_cloud(target_n, 1.0f, 42);
    query = make_point_cloud(query_n, 1.0f, 1337);
    target_data =
        apfrnn::build_neighbor_search_data(target.x, target.y, target.z, R);
  }

  void TearDown(const ::benchmark::State &) override {
    target = {};
    target_data = {};
    query = {};
  }
};

BENCHMARK_DEFINE_F(MixedSetFixture,
                   BM_ConcatMixedBuildAndWrite)(benchmark::State &state) {
  for (auto _ : state) {
    std::vector<float> combined_x = query.x;
    std::vector<float> combined_y = query.y;
    std::vector<float> combined_z = query.z;
    combined_x.insert(combined_x.end(), target.x.begin(), target.x.end());
    combined_y.insert(combined_y.end(), target.y.begin(), target.y.end());
    combined_z.insert(combined_z.end(), target.z.begin(), target.z.end());

    auto combined = apfrnn::build_neighbor_search_data(combined_x, combined_y,
                                                       combined_z, R);
    apfrnn::write_neighbors_parallel(combined);
    auto edges = collect_concat_cross_neighbors(
        combined, static_cast<int>(query.x.size()));
    benchmark::DoNotOptimize(edges);
  }

  state.SetItemsProcessed(state.iterations() * query.x.size());
}

BENCHMARK_DEFINE_F(MixedSetFixture,
                   BM_NativeCrossBuildAndWrite)(benchmark::State &state) {
  for (auto _ : state) {
    auto cross = apfrnn::build_cross_neighbor_search_data(
        query.x, query.y, query.z, target_data, R);
    apfrnn::write_cross_neighbors_parallel(cross, target_data);
    benchmark::DoNotOptimize(cross.col_idx.data());
  }

  state.SetItemsProcessed(state.iterations() * query.x.size());
}

BENCHMARK_DEFINE_F(MixedSetFixture,
                   BM_ConcatMixedFrameReplay)(benchmark::State &state) {
  for (auto _ : state) {
    std::size_t total_edges = 0;
    for (int substep = 0; substep < substeps_per_frame; ++substep) {
      std::vector<float> combined_x = query.x;
      std::vector<float> combined_y = query.y;
      std::vector<float> combined_z = query.z;
      combined_x.insert(combined_x.end(), target.x.begin(), target.x.end());
      combined_y.insert(combined_y.end(), target.y.begin(), target.y.end());
      combined_z.insert(combined_z.end(), target.z.begin(), target.z.end());

      auto combined = apfrnn::build_neighbor_search_data(combined_x, combined_y,
                                                         combined_z, R);
      apfrnn::write_neighbors_parallel(combined);
      total_edges += collect_concat_cross_neighbors(
          combined, static_cast<int>(query.x.size()));
    }
    benchmark::DoNotOptimize(total_edges);
  }

  state.SetItemsProcessed(state.iterations() * query.x.size() *
                          substeps_per_frame);
}

BENCHMARK_DEFINE_F(MixedSetFixture,
                   BM_NativeCrossFrameReplay)(benchmark::State &state) {
  for (auto _ : state) {
    std::size_t total_edges = 0;
    for (int substep = 0; substep < substeps_per_frame; ++substep) {
      auto cross = apfrnn::build_cross_neighbor_search_data(
          query.x, query.y, query.z, target_data, R);
      apfrnn::write_cross_neighbors_parallel(cross, target_data);
      total_edges += cross.col_idx.size();
    }
    benchmark::DoNotOptimize(total_edges);
  }

  state.SetItemsProcessed(state.iterations() * query.x.size() *
                          substeps_per_frame);
}

BENCHMARK_REGISTER_F(MixedSetFixture, BM_ConcatMixedBuildAndWrite)
    ->Arg(484)
    ->Arg(1936)
    ->Arg(7744);

BENCHMARK_REGISTER_F(MixedSetFixture, BM_NativeCrossBuildAndWrite)
    ->Arg(484)
    ->Arg(1936)
    ->Arg(7744);

BENCHMARK_REGISTER_F(MixedSetFixture, BM_ConcatMixedFrameReplay)->Arg(484);

BENCHMARK_REGISTER_F(MixedSetFixture, BM_NativeCrossFrameReplay)->Arg(484);

BENCHMARK_MAIN();