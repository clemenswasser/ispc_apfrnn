// Coverage for dimensions the baseline suites exercise but ours did not:
// results must be independent of the TBB worker count, tiny sets around
// thread/parallel thresholds, mixed float/double point sets, and the input
// order (the benchmark z-sorts scenes for TreeNSearch's benefit, so swapping
// the order must not change the neighbor sets).

#include "point_cloud.hpp"
#include "scene.hpp"
#include "apfrnn/neighbor_search.hpp"

#include <doctest/doctest.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace {

using apfrnn::NSearch;
using apfrnn::support::PointCloud;

std::vector<uint32_t> row(const NSearch &search, int query, int target,
                          int point) {
  const apfrnn::NeighborList list =
      search.get_neighbors(query, target, point);
  return std::vector<uint32_t>(list.begin(), list.end());
}

void check_same(const NSearch &search, int set, const PointCloud &cloud,
                float radius) {
  for (int point = 0; point < static_cast<int>(cloud.x.size()); ++point) {
    std::vector<int> expected =
        apfrnn::support::exact_neighbors_for(cloud, point, radius);
    std::vector<uint32_t> actual = row(search, set, set, point);
    std::sort(actual.begin(), actual.end());
    REQUIRE(expected.size() == actual.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
      REQUIRE(static_cast<uint32_t>(expected[index]) == actual[index]);
    }
  }
}

void check_cross(const NSearch &search, int query, int target,
                 const PointCloud &query_cloud,
                 const PointCloud &target_cloud, float radius) {
  for (int point = 0; point < static_cast<int>(query_cloud.x.size()); ++point) {
    std::vector<int> expected = apfrnn::support::exact_cross_neighbors_for(
        query_cloud, point, target_cloud, radius);
    std::vector<uint32_t> actual = row(search, query, target, point);
    std::sort(actual.begin(), actual.end());
    REQUIRE(expected.size() == actual.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
      REQUIRE(static_cast<uint32_t>(expected[index]) == actual[index]);
    }
  }
}

std::vector<float> interleave(const PointCloud &cloud) {
  std::vector<float> xyz(cloud.x.size() * 3);
  for (std::size_t index = 0; index < cloud.x.size(); ++index) {
    xyz[3 * index + 0] = cloud.x[index];
    xyz[3 * index + 1] = cloud.y[index];
    xyz[3 * index + 2] = cloud.z[index];
  }
  return xyz;
}

std::vector<double> interleave_double(const PointCloud &cloud) {
  std::vector<double> xyz(cloud.x.size() * 3);
  for (std::size_t index = 0; index < cloud.x.size(); ++index) {
    xyz[3 * index + 0] = cloud.x[index];
    xyz[3 * index + 1] = cloud.y[index];
    xyz[3 * index + 2] = cloud.z[index];
  }
  return xyz;
}

void run_thread_matrix(int threads) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                              static_cast<std::size_t>(threads));
  constexpr float radius = 2.0f;
  const PointCloud fluid = apfrnn::bench::make_clustered(4000, 16.0f, 11);
  const PointCloud boundary = apfrnn::bench::make_uniform(1500, 16.0f, 12);
  const PointCloud emitter = apfrnn::bench::make_slab(700, 16.0f, 13);
  const std::vector<float> fluid_xyz = interleave(fluid);
  const std::vector<float> boundary_xyz = interleave(boundary);
  const std::vector<float> emitter_xyz = interleave(emitter);

  NSearch search(radius);
  const int fluid_id = search.add_point_set(fluid_xyz.data(),
                                            static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  const int emitter_id = search.add_point_set(
      emitter_xyz.data(), static_cast<int>(emitter.x.size()));
  search.set_all_searches(false);
  for (int query = 0; query < 3; ++query) {
    for (int target = 0; target < 3; ++target) {
      search.set_active_search(query, target, true);
    }
  }
  search.run();

  check_same(search, fluid_id, fluid, radius);
  check_same(search, boundary_id, boundary, radius);
  check_same(search, emitter_id, emitter, radius);
  check_cross(search, fluid_id, boundary_id, fluid, boundary, radius);
  check_cross(search, fluid_id, emitter_id, fluid, emitter, radius);
  check_cross(search, boundary_id, fluid_id, boundary, fluid, radius);
  check_cross(search, boundary_id, emitter_id, boundary, emitter, radius);
}

} // namespace

TEST_CASE("results are independent of the TBB worker count") {
  for (int threads : {1, 2, 3, 4, 7, 8, 9, 12}) {
    CAPTURE(threads);
    run_thread_matrix(threads);
  }
}

TEST_CASE("tiny sets around thread and parallel thresholds stay exact") {
  constexpr float radius = 2.0f;
  for (int threads : {1, 4, 8}) {
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                static_cast<std::size_t>(threads));
    for (int sets = 1; sets <= 3; ++sets) {
      for (int points : {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1000}) {
        CAPTURE(threads);
        CAPTURE(sets);
        CAPTURE(points);
        std::vector<PointCloud> clouds;
        std::vector<std::vector<float>> xyz;
        for (int set = 0; set < sets; ++set) {
          clouds.push_back(apfrnn::bench::make_uniform(
              points, 10.0f, static_cast<uint32_t>(100 + 17 * set)));
          xyz.push_back(interleave(clouds.back()));
        }
        NSearch search(radius);
        for (int set = 0; set < sets; ++set) {
          search.add_point_set(xyz[set].data(), points);
        }
        search.set_all_searches(false);
        for (int query = 0; query < sets; ++query) {
          for (int target = 0; target < sets; ++target) {
            search.set_active_search(query, target, true);
          }
        }
        search.run();
        for (int query = 0; query < sets; ++query) {
          for (int target = 0; target < sets; ++target) {
            if (query == target) {
              check_same(search, query, clouds[query], radius);
            } else {
              check_cross(search, query, target, clouds[query],
                          clouds[target], radius);
            }
          }
        }
      }
    }
  }
}

TEST_CASE("mixed float and double point sets match brute force") {
  constexpr float radius = 1.7f;
  const PointCloud floats = apfrnn::bench::make_uniform(600, 12.0f, 31);
  const PointCloud doubles = apfrnn::bench::make_clustered(900, 12.0f, 32);
  const std::vector<float> float_xyz = interleave(floats);
  const std::vector<double> double_xyz = interleave_double(doubles);

  NSearch search(radius);
  const int float_id = search.add_point_set(
      float_xyz.data(), static_cast<int>(floats.x.size()));
  const int double_id = search.add_point_set(
      double_xyz.data(), static_cast<int>(doubles.x.size()));
  search.set_all_searches(false);
  search.set_active_search(float_id, float_id, true);
  search.set_active_search(float_id, double_id, true);
  search.set_active_search(double_id, float_id, true);
  search.set_active_search(double_id, double_id, true);
  search.run();

  check_same(search, float_id, floats, radius);
  check_same(search, double_id, doubles, radius);
  check_cross(search, float_id, double_id, floats, doubles, radius);
  check_cross(search, double_id, float_id, doubles, floats, radius);
}

TEST_CASE("same-set and cross-set results are input-order invariant") {
  constexpr float radius = 2.0f;
  const PointCloud query = apfrnn::bench::make_uniform(2500, 14.0f, 41);
  const PointCloud target = apfrnn::bench::make_clustered(1800, 14.0f, 42);

  // Locality order, the kind of permutation a z-sort produces.
  std::vector<uint32_t> order(query.x.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    if (query.x[a] != query.x[b]) {
      return query.x[a] < query.x[b];
    }
    return a < b;
  });
  PointCloud permuted_query;
  permuted_query.x.resize(query.x.size());
  permuted_query.y.resize(query.x.size());
  permuted_query.z.resize(query.x.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    const std::size_t source = order[index];
    permuted_query.x[index] = query.x[source];
    permuted_query.y[index] = query.y[source];
    permuted_query.z[index] = query.z[source];
  }

  std::vector<uint32_t> target_order(target.x.size());
  std::iota(target_order.begin(), target_order.end(), 0);
  std::mt19937 generator(7);
  std::shuffle(target_order.begin(), target_order.end(), generator);
  PointCloud permuted_target;
  permuted_target.x.resize(target.x.size());
  permuted_target.y.resize(target.x.size());
  permuted_target.z.resize(target.x.size());
  for (std::size_t index = 0; index < target_order.size(); ++index) {
    const std::size_t source = target_order[index];
    permuted_target.x[index] = target.x[source];
    permuted_target.y[index] = target.y[source];
    permuted_target.z[index] = target.z[source];
  }

  const std::vector<float> query_xyz = interleave(query);
  const std::vector<float> permuted_query_xyz = interleave(permuted_query);
  const std::vector<float> target_xyz = interleave(target);
  const std::vector<float> permuted_target_xyz = interleave(permuted_target);

  NSearch search(radius);
  const int query_id = search.add_point_set(
      query_xyz.data(), static_cast<int>(query.x.size()));
  const int target_id = search.add_point_set(
      target_xyz.data(), static_cast<int>(target.x.size()));
  search.set_all_searches(false);
  search.set_active_search(query_id, query_id, true);
  search.set_active_search(query_id, target_id, true);
  search.set_active_search(target_id, query_id, true);
  search.run();

  NSearch permuted(radius);
  const int permuted_query_id = permuted.add_point_set(
      permuted_query_xyz.data(), static_cast<int>(permuted_query.x.size()));
  const int permuted_target_id = permuted.add_point_set(
      permuted_target_xyz.data(), static_cast<int>(permuted_target.x.size()));
  permuted.set_all_searches(false);
  permuted.set_active_search(permuted_query_id, permuted_query_id, true);
  permuted.set_active_search(permuted_query_id, permuted_target_id, true);
  permuted.set_active_search(permuted_target_id, permuted_query_id, true);
  permuted.run();

  // Same set, query order permuted.
  for (std::size_t index = 0; index < order.size(); ++index) {
    std::vector<int> expected =
        apfrnn::support::exact_neighbors_for(query, order[index], radius);
    std::sort(expected.begin(), expected.end());
    std::vector<uint32_t> actual =
        row(permuted, permuted_query_id, permuted_query_id,
            static_cast<int>(index));
    for (uint32_t &neighbor : actual) {
      neighbor = order[neighbor];
    }
    std::sort(actual.begin(), actual.end());
    REQUIRE(expected.size() == actual.size());
    for (std::size_t k = 0; k < expected.size(); ++k) {
      REQUIRE(static_cast<uint32_t>(expected[k]) == actual[k]);
    }
  }

  // Cross set, target order permuted. Query points are permuted too, so map
  // the original query index back to its permuted slot before reading.
  std::vector<uint32_t> inverse_order(order.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    inverse_order[order[index]] = static_cast<uint32_t>(index);
  }
  for (int point = 0; point < static_cast<int>(query.x.size()); ++point) {
    std::vector<int> expected =
        apfrnn::support::exact_cross_neighbors_for(query, point, target,
                                                   radius);
    std::sort(expected.begin(), expected.end());
    std::vector<uint32_t> actual =
        row(permuted, permuted_query_id, permuted_target_id,
            static_cast<int>(inverse_order[static_cast<std::size_t>(point)]));
    for (uint32_t &neighbor : actual) {
      neighbor = target_order[neighbor];
    }
    std::sort(actual.begin(), actual.end());
    REQUIRE(expected.size() == actual.size());
    for (std::size_t k = 0; k < expected.size(); ++k) {
      REQUIRE(static_cast<uint32_t>(expected[k]) == actual[k]);
    }
  }
}
