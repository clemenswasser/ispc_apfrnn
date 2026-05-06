#include "apfrnn_support.hpp"
#include <benchmark/benchmark.h>
#include <cmath>

class ISPCScalingFixture : public benchmark::Fixture {
public:
  int N;
  float R = 2.0f;
  static constexpr int substeps_per_frame = 333;
  apfrnn::support::PointCloud cloud;
  apfrnn::NeighborSearchData neighbor_data;

  void SetUp(const ::benchmark::State &state) override {
    N = state.range(0);

    float side = std::pow(N, 1.0f / 3.0f) * 2.0f;
    cloud = apfrnn::support::make_point_cloud(N, side, 42);
    neighbor_data =
        apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, R);
  }

  void TearDown(const benchmark::State &state) override {
    cloud = {};
    neighbor_data = {};
  }
};

BENCHMARK_DEFINE_F(ISPCScalingFixture,
                   BM_BuildNeighborData)(benchmark::State &state) {
  for (auto _ : state) {
    auto data =
        apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, R);
    benchmark::DoNotOptimize(data.row_ptr.data());
  }

  state.SetItemsProcessed(state.iterations() * N);
}

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

BENCHMARK_DEFINE_F(ISPCScalingFixture,
                   BM_SameSetFrameReplay)(benchmark::State &state) {
  for (auto _ : state) {
    std::size_t total_edges = 0;
    for (int substep = 0; substep < substeps_per_frame; ++substep) {
      auto data =
          apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, R);
      apfrnn::write_neighbors_parallel(data);
      total_edges += data.col_idx.size();
    }
    benchmark::DoNotOptimize(total_edges);
  }

  state.SetItemsProcessed(state.iterations() * N * substeps_per_frame);
}

BENCHMARK_REGISTER_F(ISPCScalingFixture, BM_BuildNeighborData)
    ->Arg(484)
    ->Arg(1936)
    ->Arg(7744)
    ->Arg(16384);

BENCHMARK_REGISTER_F(ISPCScalingFixture, BM_WriteNeighbors)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20);

BENCHMARK_REGISTER_F(ISPCScalingFixture, BM_SameSetFrameReplay)
    ->Arg(484)
    ->Arg(1936)
    ->Arg(7744)
    ->Arg(16384);

class MixedSetFixture : public benchmark::Fixture {
public:
  static constexpr int target_n = 78437;
  static constexpr int substeps_per_frame = 333;
  float R = 0.02f;
  apfrnn::support::PointCloud target;
  apfrnn::NeighborSearchData target_data;
  apfrnn::support::PointCloud query;
  apfrnn::CrossNeighborSearchData cross_data;

  void SetUp(const ::benchmark::State &state) override {
    int query_n = static_cast<int>(state.range(0));
    target = apfrnn::support::make_point_cloud(target_n, 1.0f, 42);
    query = apfrnn::support::make_point_cloud(query_n, 1.0f, 1337);
    target_data =
        apfrnn::build_neighbor_search_data(target.x, target.y, target.z, R);
    cross_data = apfrnn::build_cross_neighbor_search_data(
        query.x, query.y, query.z, target_data, R);
  }

  void TearDown(const ::benchmark::State &) override {
    target = {};
    target_data = {};
    query = {};
    cross_data = {};
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
    auto edges = apfrnn::support::collect_concat_cross_neighbors(
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
                   BM_NativeCrossBuildOnly)(benchmark::State &state) {
  for (auto _ : state) {
    auto cross = apfrnn::build_cross_neighbor_search_data(
        query.x, query.y, query.z, target_data, R);
    benchmark::DoNotOptimize(cross.row_ptr.data());
  }

  state.SetItemsProcessed(state.iterations() * query.x.size());
}

BENCHMARK_DEFINE_F(MixedSetFixture,
                   BM_NativeCrossWriteOnly)(benchmark::State &state) {
  for (auto _ : state) {
    apfrnn::write_cross_neighbors_parallel(cross_data, target_data);
    benchmark::DoNotOptimize(cross_data.col_idx.data());
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
      total_edges += apfrnn::support::collect_concat_cross_neighbors(
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

BENCHMARK_REGISTER_F(MixedSetFixture, BM_NativeCrossBuildOnly)
    ->Arg(484)
    ->Arg(1936)
    ->Arg(7744);

BENCHMARK_REGISTER_F(MixedSetFixture, BM_NativeCrossWriteOnly)
    ->Arg(484)
    ->Arg(1936)
    ->Arg(7744);

BENCHMARK_REGISTER_F(MixedSetFixture, BM_ConcatMixedFrameReplay)->Arg(484);

BENCHMARK_REGISTER_F(MixedSetFixture, BM_NativeCrossFrameReplay)->Arg(484);

BENCHMARK_MAIN();