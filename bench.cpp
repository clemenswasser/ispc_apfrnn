#include "apfrnn.hpp"
#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <vector>

class ISPCScalingFixture : public benchmark::Fixture {
public:
  int N;
  float R = 2.0f;
  uint32_t HASH_SIZE = 1 << 22;
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

    neighbor_data = apfrnn::build_neighbor_search_data(X, Y, Z, R, HASH_SIZE);
  }

  void TearDown(const benchmark::State &state) override { neighbor_data = {}; }
};

BENCHMARK_DEFINE_F(ISPCScalingFixture,
                   BM_WriteNeighbors)(benchmark::State &state) {
  for (auto _ : state) {
    apfrnn::write_neighbors_range(neighbor_data, 0, N);
    benchmark::DoNotOptimize(neighbor_data.col_idx.data());
  }

  state.SetItemsProcessed(state.iterations() * N);
  state.counters["TotalEdges"] = (double)neighbor_data.col_idx.size();
  state.counters["EdgesPerParticle"] = (double)neighbor_data.col_idx.size() / N;
}

BENCHMARK_REGISTER_F(ISPCScalingFixture, BM_WriteNeighbors)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20);

BENCHMARK_MAIN();