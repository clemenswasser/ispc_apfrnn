#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "apfrnn_support.hpp"

#include <doctest/doctest.h>

namespace {

apfrnn::support::PointCloud make_duplicate_point_cloud() {
  apfrnn::support::PointCloud cloud;
  cloud.x = {0.0f, 0.0f, 1.0f, 3.0f};
  cloud.y = {0.0f, 0.0f, 0.0f, 0.0f};
  cloud.z = {0.0f, 0.0f, 0.0f, 0.0f};
  return cloud;
}

} // namespace

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

TEST_CASE(
    "neighbor search matches brute force for every point in a small cloud") {
  constexpr int num_points = 256;
  constexpr float radius = 1.75f;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 20.0f, 7);
  auto neighbor_data =
      apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, radius);
  apfrnn::write_neighbors_parallel(neighbor_data);

  CHECK(apfrnn::support::has_valid_row_ptr(neighbor_data));

  for (int point_index = 0; point_index < num_points; ++point_index) {
    auto exact_neighbors =
        apfrnn::support::exact_neighbors_for(cloud, point_index, radius);
    auto ispc_neighbors =
        apfrnn::support::ispc_neighbors_for(neighbor_data, point_index);
    CHECK(exact_neighbors == ispc_neighbors);
  }
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

TEST_CASE("cross neighbor search matches brute force for every query point") {
  constexpr int query_points = 128;
  constexpr int target_points = 512;
  constexpr float radius = 1.25f;

  auto query = apfrnn::support::make_point_cloud(query_points, 15.0f, 11);
  auto target = apfrnn::support::make_point_cloud(target_points, 15.0f, 19);

  auto target_data =
      apfrnn::build_neighbor_search_data(target.x, target.y, target.z, radius);
  auto cross_data = apfrnn::build_cross_neighbor_search_data(
      query.x, query.y, query.z, target_data, radius);
  apfrnn::write_cross_neighbors_parallel(cross_data, target_data);

  CHECK(apfrnn::support::has_valid_row_ptr(target_data));
  CHECK(apfrnn::support::has_valid_row_ptr(cross_data));

  for (int point_index = 0; point_index < query_points; ++point_index) {
    auto exact_neighbors = apfrnn::support::exact_cross_neighbors_for(
        query, point_index, target, radius);
    auto ispc_neighbors =
        apfrnn::support::ispc_neighbors_for(cross_data, point_index);
    CHECK(exact_neighbors == ispc_neighbors);
  }
}

TEST_CASE("native cross search matches concat baseline exactly") {
  constexpr int query_points = 96;
  constexpr int target_points = 384;
  constexpr float radius = 1.4f;

  auto query = apfrnn::support::make_point_cloud(query_points, 18.0f, 123);
  auto target = apfrnn::support::make_point_cloud(target_points, 18.0f, 456);

  std::vector<float> combined_x = query.x;
  std::vector<float> combined_y = query.y;
  std::vector<float> combined_z = query.z;
  combined_x.insert(combined_x.end(), target.x.begin(), target.x.end());
  combined_y.insert(combined_y.end(), target.y.begin(), target.y.end());
  combined_z.insert(combined_z.end(), target.z.begin(), target.z.end());

  auto concat_data = apfrnn::build_neighbor_search_data(combined_x, combined_y,
                                                        combined_z, radius);
  apfrnn::write_neighbors_parallel(concat_data);

  auto target_data =
      apfrnn::build_neighbor_search_data(target.x, target.y, target.z, radius);
  auto cross_data = apfrnn::build_cross_neighbor_search_data(
      query.x, query.y, query.z, target_data, radius);
  apfrnn::write_cross_neighbors_parallel(cross_data, target_data);

  CHECK(apfrnn::support::has_valid_row_ptr(concat_data));
  CHECK(apfrnn::support::has_valid_row_ptr(cross_data));

  for (int point_index = 0; point_index < query_points; ++point_index) {
    auto exact_cross = apfrnn::support::exact_cross_neighbors_for(
        query, point_index, target, radius);
    auto native_cross =
        apfrnn::support::ispc_neighbors_for(cross_data, point_index);
    CHECK(exact_cross == native_cross);

    auto concat_neighbors =
        apfrnn::support::ispc_neighbors_for(concat_data, point_index);
    std::vector<int> filtered_concat;
    for (int neighbor_index : concat_neighbors) {
      if (neighbor_index >= query_points) {
        filtered_concat.push_back(neighbor_index - query_points);
      }
    }
    CHECK(filtered_concat == native_cross);
  }
}

TEST_CASE("empty same-set and cross-set inputs produce valid empty CSR") {
  constexpr float radius = 1.0f;

  apfrnn::support::PointCloud empty;
  auto empty_data =
      apfrnn::build_neighbor_search_data(empty.x, empty.y, empty.z, radius);
  apfrnn::write_neighbors_parallel(empty_data);

  CHECK(empty_data.num_points == 0);
  CHECK(empty_data.row_ptr == std::vector<int>{0});
  CHECK(empty_data.col_idx.empty());
  CHECK(apfrnn::support::has_valid_row_ptr(empty_data));

  auto target = apfrnn::support::make_point_cloud(32, 4.0f, 99);
  auto target_data =
      apfrnn::build_neighbor_search_data(target.x, target.y, target.z, radius);
  auto empty_cross = apfrnn::build_cross_neighbor_search_data(
      empty.x, empty.y, empty.z, target_data, radius);
  apfrnn::write_cross_neighbors_parallel(empty_cross, target_data);

  CHECK(empty_cross.num_points == 0);
  CHECK(empty_cross.row_ptr == std::vector<int>{0});
  CHECK(empty_cross.col_idx.empty());
  CHECK(apfrnn::support::has_valid_row_ptr(empty_cross));
}

TEST_CASE("duplicate points preserve expected zero-distance neighbors") {
  constexpr float radius = 0.0f;

  auto cloud = make_duplicate_point_cloud();
  auto neighbor_data =
      apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, radius);
  apfrnn::write_neighbors_parallel(neighbor_data);

  CHECK(apfrnn::support::has_valid_row_ptr(neighbor_data));
  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 0) ==
        std::vector<int>{1});
  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 1) ==
        std::vector<int>{0});
  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 2).empty());
  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 3).empty());
}
