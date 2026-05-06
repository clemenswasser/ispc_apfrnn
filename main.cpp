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

apfrnn::support::PointCloud make_threshold_point_cloud() {
  apfrnn::support::PointCloud cloud;
  cloud.x = {0.0f, 1.0f, 2.0001f, -1.0f, 0.0f};
  cloud.y = {0.0f, 0.0f, 0.0f, 0.0f, 2.0f};
  cloud.z = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  return cloud;
}

void translate_cloud(apfrnn::support::PointCloud &cloud, float dx, float dy,
                     float dz) {
  for (std::size_t index = 0; index < cloud.x.size(); ++index) {
    cloud.x[index] += dx;
    cloud.y[index] += dy;
    cloud.z[index] += dz;
  }
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

TEST_CASE(
    "same-set neighbors are symmetric, duplicate-free, and exclude self") {
  constexpr int num_points = 192;
  constexpr float radius = 1.5f;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 12.0f, 31415);
  auto neighbor_data =
      apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, radius);
  apfrnn::write_neighbors_parallel(neighbor_data);

  CHECK(apfrnn::support::has_valid_row_ptr(neighbor_data));

  std::vector<std::vector<int>> all_neighbors(num_points);
  for (int point_index = 0; point_index < num_points; ++point_index) {
    all_neighbors[point_index] =
        apfrnn::support::ispc_neighbors_for(neighbor_data, point_index);
    CHECK(
        apfrnn::support::is_strictly_sorted_unique(all_neighbors[point_index]));
    CHECK(std::find(all_neighbors[point_index].begin(),
                    all_neighbors[point_index].end(),
                    point_index) == all_neighbors[point_index].end());
  }

  for (int point_index = 0; point_index < num_points; ++point_index) {
    for (int neighbor_index : all_neighbors[point_index]) {
      CHECK(std::binary_search(all_neighbors[neighbor_index].begin(),
                               all_neighbors[neighbor_index].end(),
                               point_index));
    }
  }
}

TEST_CASE(
    "distance threshold is inclusive and points beyond radius are excluded") {
  constexpr float radius = 1.0f;

  auto cloud = make_threshold_point_cloud();
  auto neighbor_data =
      apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, radius);
  apfrnn::write_neighbors_parallel(neighbor_data);

  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 0) ==
        std::vector<int>{1, 3});
  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 1) ==
        std::vector<int>{0});
  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 2).empty());
  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 3) ==
        std::vector<int>{0});
  CHECK(apfrnn::support::ispc_neighbors_for(neighbor_data, 4).empty());
}

TEST_CASE(
    "same-set and cross-set results are deterministic across repeated runs") {
  constexpr int query_points = 64;
  constexpr int target_points = 160;
  constexpr float radius = 1.1f;

  auto query = apfrnn::support::make_point_cloud(query_points, 10.0f, 21);
  auto target = apfrnn::support::make_point_cloud(target_points, 10.0f, 84);

  auto first_same =
      apfrnn::build_neighbor_search_data(query.x, query.y, query.z, radius);
  apfrnn::write_neighbors_parallel(first_same);
  auto second_same =
      apfrnn::build_neighbor_search_data(query.x, query.y, query.z, radius);
  apfrnn::write_neighbors_parallel(second_same);

  CHECK(first_same.original_index == second_same.original_index);
  CHECK(first_same.row_ptr == second_same.row_ptr);
  CHECK(first_same.col_idx == second_same.col_idx);

  auto target_data =
      apfrnn::build_neighbor_search_data(target.x, target.y, target.z, radius);
  auto first_cross = apfrnn::build_cross_neighbor_search_data(
      query.x, query.y, query.z, target_data, radius);
  apfrnn::write_cross_neighbors_parallel(first_cross, target_data);
  auto second_cross = apfrnn::build_cross_neighbor_search_data(
      query.x, query.y, query.z, target_data, radius);
  apfrnn::write_cross_neighbors_parallel(second_cross, target_data);

  CHECK(first_cross.original_index == second_cross.original_index);
  CHECK(first_cross.row_ptr == second_cross.row_ptr);
  CHECK(first_cross.col_idx == second_cross.col_idx);
}

TEST_CASE("cross-set writes do not mutate cached target search data") {
  constexpr int query_points = 48;
  constexpr int target_points = 96;
  constexpr float radius = 1.3f;

  auto query = apfrnn::support::make_point_cloud(query_points, 8.0f, 17);
  auto target = apfrnn::support::make_point_cloud(target_points, 8.0f, 18);

  auto target_data =
      apfrnn::build_neighbor_search_data(target.x, target.y, target.z, radius);
  auto target_data_before = target_data;

  auto cross_data = apfrnn::build_cross_neighbor_search_data(
      query.x, query.y, query.z, target_data, radius);
  apfrnn::write_cross_neighbors_parallel(cross_data, target_data);

  CHECK(target_data.original_index == target_data_before.original_index);
  CHECK(target_data.cell_starts == target_data_before.cell_starts);
  CHECK(target_data.cell_ends == target_data_before.cell_ends);
  CHECK(target_data.hash_keys == target_data_before.hash_keys);
  CHECK(target_data.hash_vals == target_data_before.hash_vals);
  CHECK(target_data.row_ptr == target_data_before.row_ptr);
  CHECK(target_data.col_idx == target_data_before.col_idx);
}

TEST_CASE(
    "mixed frame replay remains exact for same-set and cross-set queries") {
  constexpr int fluid_points = 24;
  constexpr int boundary_points = 40;
  constexpr float radius = 0.9f;
  constexpr int replay_steps = 6;

  auto fluid = apfrnn::support::make_point_cloud(fluid_points, 6.0f, 1001);
  auto boundary =
      apfrnn::support::make_point_cloud(boundary_points, 6.0f, 2002);
  auto boundary_data = apfrnn::build_neighbor_search_data(
      boundary.x, boundary.y, boundary.z, radius);

  for (int step = 0; step < replay_steps; ++step) {
    auto same_data =
        apfrnn::build_neighbor_search_data(fluid.x, fluid.y, fluid.z, radius);
    apfrnn::write_neighbors_parallel(same_data);

    auto cross_data = apfrnn::build_cross_neighbor_search_data(
        fluid.x, fluid.y, fluid.z, boundary_data, radius);
    apfrnn::write_cross_neighbors_parallel(cross_data, boundary_data);

    CHECK(apfrnn::support::has_valid_row_ptr(same_data));
    CHECK(apfrnn::support::has_valid_row_ptr(cross_data));

    for (int point_index = 0; point_index < fluid_points; ++point_index) {
      CHECK(apfrnn::support::exact_neighbors_for(fluid, point_index, radius) ==
            apfrnn::support::ispc_neighbors_for(same_data, point_index));
      CHECK(apfrnn::support::exact_cross_neighbors_for(fluid, point_index,
                                                       boundary, radius) ==
            apfrnn::support::ispc_neighbors_for(cross_data, point_index));
    }

    translate_cloud(fluid, 0.05f, -0.02f, 0.03f);
  }
}

TEST_CASE("same-set write_neighbors_range matches write_neighbors_parallel") {
  constexpr int num_points = 320;
  constexpr float radius = 1.35f;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 14.0f, 551);
  auto ranged =
      apfrnn::build_neighbor_search_data(cloud.x, cloud.y, cloud.z, radius);
  auto parallel = ranged;

  apfrnn::write_neighbors_range(ranged, 0, ranged.num_cells);
  apfrnn::write_neighbors_parallel(parallel);

  CHECK(ranged.original_index == parallel.original_index);
  CHECK(ranged.row_ptr == parallel.row_ptr);
  CHECK(ranged.col_idx == parallel.col_idx);
}

TEST_CASE("cross-set write_cross_neighbors_range matches "
          "write_cross_neighbors_parallel") {
  constexpr int query_points = 128;
  constexpr int target_points = 384;
  constexpr float radius = 1.15f;

  auto query = apfrnn::support::make_point_cloud(query_points, 12.0f, 701);
  auto target = apfrnn::support::make_point_cloud(target_points, 12.0f, 702);
  auto target_data =
      apfrnn::build_neighbor_search_data(target.x, target.y, target.z, radius);
  auto ranged = apfrnn::build_cross_neighbor_search_data(
      query.x, query.y, query.z, target_data, radius);
  auto parallel = ranged;

  apfrnn::write_cross_neighbors_range(ranged, target_data, 0, ranged.num_cells);
  apfrnn::write_cross_neighbors_parallel(parallel, target_data);

  CHECK(ranged.original_index == parallel.original_index);
  CHECK(ranged.row_ptr == parallel.row_ptr);
  CHECK(ranged.col_idx == parallel.col_idx);
}

TEST_CASE(
    "same-set results are permutation invariant in original index space") {
  constexpr int num_points = 180;
  constexpr float radius = 1.6f;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 16.0f, 808);
  auto permutation = apfrnn::support::make_permutation(num_points, 809);
  auto inverse_permutation = apfrnn::support::invert_permutation(permutation);
  auto permuted_cloud = apfrnn::support::permute_cloud(cloud, permutation);

  auto permuted_data = apfrnn::build_neighbor_search_data(
      permuted_cloud.x, permuted_cloud.y, permuted_cloud.z, radius);
  apfrnn::write_neighbors_parallel(permuted_data);

  for (int original_point = 0; original_point < num_points; ++original_point) {
    int permuted_point =
        inverse_permutation[static_cast<std::size_t>(original_point)];
    auto permuted_neighbors =
        apfrnn::support::ispc_neighbors_for(permuted_data, permuted_point);
    auto remapped_neighbors =
        apfrnn::support::remap_neighbors(permuted_neighbors, permutation);
    auto exact_neighbors =
        apfrnn::support::exact_neighbors_for(cloud, original_point, radius);
    CHECK(remapped_neighbors == exact_neighbors);
  }
}

TEST_CASE("cross-set results are permutation invariant for query and target") {
  constexpr int query_points = 72;
  constexpr int target_points = 144;
  constexpr float radius = 1.05f;

  auto query = apfrnn::support::make_point_cloud(query_points, 9.0f, 901);
  auto target = apfrnn::support::make_point_cloud(target_points, 9.0f, 902);
  auto query_permutation = apfrnn::support::make_permutation(query_points, 903);
  auto target_permutation =
      apfrnn::support::make_permutation(target_points, 904);
  auto inverse_query_permutation =
      apfrnn::support::invert_permutation(query_permutation);
  auto permuted_query =
      apfrnn::support::permute_cloud(query, query_permutation);
  auto permuted_target =
      apfrnn::support::permute_cloud(target, target_permutation);

  auto permuted_target_data = apfrnn::build_neighbor_search_data(
      permuted_target.x, permuted_target.y, permuted_target.z, radius);
  auto permuted_cross = apfrnn::build_cross_neighbor_search_data(
      permuted_query.x, permuted_query.y, permuted_query.z,
      permuted_target_data, radius);
  apfrnn::write_cross_neighbors_parallel(permuted_cross, permuted_target_data);

  for (int original_point = 0; original_point < query_points;
       ++original_point) {
    int permuted_point =
        inverse_query_permutation[static_cast<std::size_t>(original_point)];
    auto permuted_neighbors =
        apfrnn::support::ispc_neighbors_for(permuted_cross, permuted_point);
    auto remapped_neighbors = apfrnn::support::remap_neighbors(
        permuted_neighbors, target_permutation);
    auto exact_neighbors = apfrnn::support::exact_cross_neighbors_for(
        query, original_point, target, radius);
    CHECK(remapped_neighbors == exact_neighbors);
  }
}

TEST_CASE("same-set neighbors are monotonic with increasing radius") {
  constexpr int num_points = 140;
  constexpr float small_radius = 0.8f;
  constexpr float large_radius = 1.6f;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 11.0f, 10001);
  auto small_data = apfrnn::build_neighbor_search_data(cloud.x, cloud.y,
                                                       cloud.z, small_radius);
  auto large_data = apfrnn::build_neighbor_search_data(cloud.x, cloud.y,
                                                       cloud.z, large_radius);
  apfrnn::write_neighbors_parallel(small_data);
  apfrnn::write_neighbors_parallel(large_data);

  for (int point_index = 0; point_index < num_points; ++point_index) {
    auto small_neighbors =
        apfrnn::support::ispc_neighbors_for(small_data, point_index);
    auto large_neighbors =
        apfrnn::support::ispc_neighbors_for(large_data, point_index);
    for (int neighbor_index : small_neighbors) {
      CHECK(std::binary_search(large_neighbors.begin(), large_neighbors.end(),
                               neighbor_index));
    }
  }
}

TEST_CASE("cross-set neighbors are monotonic with increasing radius") {
  constexpr int query_points = 60;
  constexpr int target_points = 100;
  constexpr float small_radius = 0.7f;
  constexpr float large_radius = 1.4f;

  auto query = apfrnn::support::make_point_cloud(query_points, 7.0f, 11001);
  auto target = apfrnn::support::make_point_cloud(target_points, 7.0f, 11002);
  auto small_target_data = apfrnn::build_neighbor_search_data(
      target.x, target.y, target.z, small_radius);
  auto large_target_data = apfrnn::build_neighbor_search_data(
      target.x, target.y, target.z, large_radius);
  auto small_cross = apfrnn::build_cross_neighbor_search_data(
      query.x, query.y, query.z, small_target_data, small_radius);
  auto large_cross = apfrnn::build_cross_neighbor_search_data(
      query.x, query.y, query.z, large_target_data, large_radius);
  apfrnn::write_cross_neighbors_parallel(small_cross, small_target_data);
  apfrnn::write_cross_neighbors_parallel(large_cross, large_target_data);

  for (int point_index = 0; point_index < query_points; ++point_index) {
    auto small_neighbors =
        apfrnn::support::ispc_neighbors_for(small_cross, point_index);
    auto large_neighbors =
        apfrnn::support::ispc_neighbors_for(large_cross, point_index);
    for (int neighbor_index : small_neighbors) {
      CHECK(std::binary_search(large_neighbors.begin(), large_neighbors.end(),
                               neighbor_index));
    }
  }
}
