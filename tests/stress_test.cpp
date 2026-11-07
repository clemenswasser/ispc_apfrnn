// Deterministic add/remove/replace soak, adapted from TreeNSearch's
// dynamic-emitter stress test. Two point sets are grown, shrunk and rebuilt
// thousands of times and every active pair is checked against brute force.
// The default run is short enough for CI; APFRNN_SOAK_ITERS=10000 is the
// nightly soak.

#include "point_cloud.hpp"
#include "scene.hpp"
#include "apfrnn/neighbor_search.hpp"

#include <doctest/doctest.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using apfrnn::NSearch;
using apfrnn::support::PointCloud;

std::vector<float> interleave(const PointCloud &cloud) {
  std::vector<float> xyz(cloud.x.size() * 3);
  for (std::size_t index = 0; index < cloud.x.size(); ++index) {
    xyz[3 * index + 0] = cloud.x[index];
    xyz[3 * index + 1] = cloud.y[index];
    xyz[3 * index + 2] = cloud.z[index];
  }
  return xyz;
}

PointCloud cloud_from(const std::vector<float> &xyz, int count) {
  PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(count));
  cloud.y.resize(static_cast<std::size_t>(count));
  cloud.z.resize(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    cloud.x[static_cast<std::size_t>(index)] = xyz[3 * index + 0];
    cloud.y[static_cast<std::size_t>(index)] = xyz[3 * index + 1];
    cloud.z[static_cast<std::size_t>(index)] = xyz[3 * index + 2];
  }
  return cloud;
}

void check_pair(const NSearch &search, int query, int target,
                const PointCloud &query_cloud, const PointCloud &target_cloud,
                float radius) {
  for (int point = 0; point < static_cast<int>(query_cloud.x.size());
       ++point) {
    std::vector<int> expected =
        query == target
            ? apfrnn::support::exact_neighbors_for(query_cloud, point, radius)
            : apfrnn::support::exact_cross_neighbors_for(
                  query_cloud, point, target_cloud, radius);
    const apfrnn::NeighborList list =
        search.get_neighbors(query, target, point);
    std::vector<uint32_t> actual(list.begin(), list.end());
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    REQUIRE(expected.size() == actual.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
      REQUIRE(static_cast<uint32_t>(expected[index]) == actual[index]);
    }
  }
}

void run_soak(int iterations, int threads) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                              static_cast<std::size_t>(threads));
  constexpr float radius = 2.0f;
  constexpr int kMaxPoints = 400;
  constexpr float kSide = 18.0f;

  std::mt19937 generator(20221101u + static_cast<uint32_t>(threads));
  std::uniform_real_distribution<float> coordinate(0.0f, kSide);
  std::uniform_int_distribution<int> set_pick(0, 1);
  std::uniform_int_distribution<int> action_pick(0, 2);
  std::uniform_int_distribution<int> amount_pick(1, 20);

  std::vector<std::vector<float>> xyz(2);
  std::vector<int> counts = {0, 0};
  const auto refill = [&](int set, int count) {
    xyz[static_cast<std::size_t>(set)].resize(
        static_cast<std::size_t>(count) * 3);
    for (int index = 0; index < count; ++index) {
      xyz[static_cast<std::size_t>(set)][3 * index + 0] = coordinate(generator);
      xyz[static_cast<std::size_t>(set)][3 * index + 1] = coordinate(generator);
      xyz[static_cast<std::size_t>(set)][3 * index + 2] = coordinate(generator);
    }
    counts[static_cast<std::size_t>(set)] = count;
  };

  NSearch search(radius);
  for (int set = 0; set < 2; ++set) {
    search.add_point_set(xyz[static_cast<std::size_t>(set)].data(), 0);
  }
  search.set_all_searches(true);

  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int set = set_pick(generator);
    const int action = action_pick(generator);
    const int amount = amount_pick(generator);
    const int current = counts[static_cast<std::size_t>(set)];
    int target = current;
    if (action == 0) {
      target = std::min(kMaxPoints, current + amount);
    } else if (action == 1) {
      target = std::max(0, current - amount);
    } else {
      target = std::min(kMaxPoints, amount);
    }
    if (target == current && action != 2) {
      continue;
    }
    refill(set, target);
    search.resize_point_set(set, xyz[static_cast<std::size_t>(set)].data(),
                            target);
    search.run();

    for (int query = 0; query < 2; ++query) {
      for (int target_set = 0; target_set < 2; ++target_set) {
        if (counts[static_cast<std::size_t>(query)] == 0) {
          continue;
        }
        check_pair(search, query, target_set,
                   cloud_from(xyz[static_cast<std::size_t>(query)],
                              counts[static_cast<std::size_t>(query)]),
                   cloud_from(xyz[static_cast<std::size_t>(target_set)],
                              counts[static_cast<std::size_t>(target_set)]),
                   radius);
      }
    }
  }
}

} // namespace

TEST_CASE("adding, removing and replacing points stays exact") {
  const char *value = std::getenv("APFRNN_SOAK_ITERS");
  const int iterations = value != nullptr ? std::atoi(value) : 120;
  for (int threads : {1, 4, 8}) {
    CAPTURE(threads);
    CAPTURE(iterations);
    run_soak(iterations, threads);
  }
}
