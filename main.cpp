#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "apfrnn.hpp"
#include <algorithm>
#include <random>
#include <vector>

#include <doctest/doctest.h>

namespace {

struct PointCloud {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
};

PointCloud make_point_cloud(int num_points) {
  PointCloud cloud{std::vector<float>(num_points),
                   std::vector<float>(num_points),
                   std::vector<float>(num_points)};

  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
  for (int index = 0; index < num_points; ++index) {
    cloud.x[index] = distribution(generator);
    cloud.y[index] = distribution(generator);
    cloud.z[index] = distribution(generator);
  }

  cloud.x[0] = 1e8f;
  cloud.y[0] = -1e8f;
  cloud.z[0] = 5e8f;
  return cloud;
}

std::vector<int> exact_neighbors_for(const PointCloud &cloud, int test_index,
                                     float radius) {
  std::vector<int> neighbors;
  for (int index = 0; index < static_cast<int>(cloud.x.size()); ++index) {
    if (index == test_index) {
      continue;
    }

    float dx = cloud.x[test_index] - cloud.x[index];
    float dy = cloud.y[test_index] - cloud.y[index];
    float dz = cloud.z[test_index] - cloud.z[index];
    if (dx * dx + dy * dy + dz * dz <= radius * radius) {
      neighbors.push_back(index);
    }
  }

  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

int sorted_index_for(const apfrnn::NeighborSearchData &neighbor_data,
                     int original_index) {
  auto sorted_it =
      std::find(neighbor_data.original_index.begin(),
                neighbor_data.original_index.end(), original_index);
  REQUIRE(sorted_it != neighbor_data.original_index.end());
  return static_cast<int>(sorted_it - neighbor_data.original_index.begin());
}

std::vector<int>
ispc_neighbors_for(const apfrnn::NeighborSearchData &neighbor_data,
                   int original_index) {
  std::vector<int> neighbors;
  int sorted_index = sorted_index_for(neighbor_data, original_index);
  for (int index = neighbor_data.row_ptr[sorted_index];
       index < neighbor_data.row_ptr[sorted_index + 1]; ++index) {
    neighbors.push_back(neighbor_data.col_idx[index]);
  }

  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

} // namespace

TEST_CASE(
    "neighbor search matches brute force for a deterministic point cloud") {
  constexpr int num_points = 10000;
  constexpr float radius = 2.0f;
  constexpr int test_index = 42;

  PointCloud cloud = make_point_cloud(num_points);

  auto neighbor_data =
      apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, radius);
  apfrnn::write_neighbors_parallel(neighbor_data);

  CHECK(neighbor_data.num_points == num_points);
  CHECK(neighbor_data.row_ptr.size() == static_cast<size_t>(num_points + 1));

  auto exact_neighbors = exact_neighbors_for(cloud, test_index, radius);
  auto ispc_neighbors = ispc_neighbors_for(neighbor_data, test_index);

  CHECK(exact_neighbors == ispc_neighbors);
  CHECK(neighbor_data.row_ptr.back() ==
        static_cast<int>(neighbor_data.col_idx.size()));
}

TEST_CASE("extreme outlier has no neighbors") {
  constexpr int num_points = 10000;
  constexpr float radius = 2.0f;

  PointCloud cloud = make_point_cloud(num_points);

  auto neighbor_data =
      apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, radius);
  apfrnn::write_neighbors_parallel(neighbor_data);

  CHECK(ispc_neighbors_for(neighbor_data, 0).empty());
}
