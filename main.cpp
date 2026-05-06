#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "apfrnn_support.hpp"

#include <doctest/doctest.h>

TEST_CASE(
    "neighbor search matches brute force for a deterministic point cloud") {
  constexpr int num_points = 10000;
  constexpr float radius = 2.0f;
  constexpr int test_index = 42;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 100.0f, 42);
  apfrnn::support::set_extreme_outlier(cloud);

  auto neighbor_data =
      apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, radius);
  apfrnn::write_neighbors_parallel(neighbor_data);

  CHECK(neighbor_data.num_points == num_points);
  CHECK(neighbor_data.row_ptr.size() == static_cast<size_t>(num_points + 1));

  auto exact_neighbors =
      apfrnn::support::exact_neighbors_for(cloud, test_index, radius);
  auto ispc_neighbors =
      apfrnn::support::ispc_neighbors_for(neighbor_data, test_index);

  CHECK(exact_neighbors == ispc_neighbors);
  CHECK(neighbor_data.row_ptr.back() ==
        static_cast<int>(neighbor_data.col_idx.size()));
}

TEST_CASE("extreme outlier has no neighbors") {
  constexpr int num_points = 10000;
  constexpr float radius = 2.0f;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 100.0f, 42);
  apfrnn::support::set_extreme_outlier(cloud);

  auto neighbor_data =
      apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, radius);
  apfrnn::write_neighbors_parallel(neighbor_data);

  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 0).empty());
}
