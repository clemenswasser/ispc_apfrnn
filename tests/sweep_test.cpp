#include "sweep.hpp"

#include "apfrnn/neighbor_search.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct Cloud {
  std::vector<float> x, y, z;

  apfrnn::Bounds bounds() const {
    apfrnn::Bounds result;
    for (std::size_t i = 0; i < x.size(); ++i) {
      result.add(x[i], y[i], z[i]);
    }
    return result;
  }
};

Cloud uniform_cloud(int num_points, float side, uint32_t seed) {
  Cloud cloud;
  cloud.x.resize(static_cast<std::size_t>(num_points));
  cloud.y.resize(static_cast<std::size_t>(num_points));
  cloud.z.resize(static_cast<std::size_t>(num_points));
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> coordinate(0.0f, side);
  for (int i = 0; i < num_points; ++i) {
    cloud.x[static_cast<std::size_t>(i)] = coordinate(generator);
    cloud.y[static_cast<std::size_t>(i)] = coordinate(generator);
    cloud.z[static_cast<std::size_t>(i)] = coordinate(generator);
  }
  return cloud;
}

std::vector<float> interleave(const Cloud &cloud) {
  std::vector<float> xyz(cloud.x.size() * 3);
  for (std::size_t i = 0; i < cloud.x.size(); ++i) {
    xyz[3 * i + 0] = cloud.x[i];
    xyz[3 * i + 1] = cloud.y[i];
    xyz[3 * i + 2] = cloud.z[i];
  }
  return xyz;
}

Cloud droplet_cloud(int droplets, int points_per_droplet, float side,
                    uint32_t seed) {
  Cloud cloud;
  const int num_points = droplets * points_per_droplet;
  cloud.x.resize(static_cast<std::size_t>(num_points));
  cloud.y.resize(static_cast<std::size_t>(num_points));
  cloud.z.resize(static_cast<std::size_t>(num_points));
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> center(0.0f, side);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  const float droplet_radius = std::cbrt(
      static_cast<float>(points_per_droplet) * 3.0f / (4.0f * 3.14159265f));
  int index = 0;
  for (int droplet = 0; droplet < droplets; ++droplet) {
    const float cx = center(generator);
    const float cy = center(generator);
    const float cz = center(generator);
    for (int k = 0; k < points_per_droplet;) {
      const float dx = unit(generator);
      const float dy = unit(generator);
      const float dz = unit(generator);
      if (dx * dx + dy * dy + dz * dz > 1.0f) {
        continue;
      }
      cloud.x[static_cast<std::size_t>(index)] = cx + droplet_radius * dx;
      cloud.y[static_cast<std::size_t>(index)] = cy + droplet_radius * dy;
      cloud.z[static_cast<std::size_t>(index)] = cz + droplet_radius * dz;
      ++index;
      ++k;
    }
  }
  return cloud;
}

} // namespace

TEST_CASE("sweep decision keeps uniformly sparse sets on the sweep") {
  constexpr float radius = 2.0f;
  const Cloud cloud = uniform_cloud(8192, 20000.0f, 11);
  const apfrnn::Bounds bounds = cloud.bounds();
  REQUIRE(apfrnn::sweep::estimated_cells_per_point(bounds, cloud.x.size(),
                                                   radius) > 10000.0);
  CHECK(apfrnn::sweep::should_use(cloud.x, cloud.y, cloud.z, bounds, radius));
}

TEST_CASE("sweep decision sends clustered sparse sets to the grid") {
  constexpr float radius = 2.0f;
  // Droplets of 128 points, each small enough to fit in one coarse decision
  // cell, so the sample sees many repeated cells.
  const Cloud cloud = droplet_cloud(64, 128, 20000.0f, 12);
  const apfrnn::Bounds bounds = cloud.bounds();
  REQUIRE(apfrnn::sweep::estimated_cells_per_point(bounds, cloud.x.size(),
                                                   radius) > 10000.0);
  CHECK_FALSE(
      apfrnn::sweep::should_use(cloud.x, cloud.y, cloud.z, bounds, radius));
}

TEST_CASE("sweep decision leaves dense sets on the grid") {
  constexpr float radius = 2.0f;
  const Cloud cloud = uniform_cloud(8192, 30.0f, 13);
  const apfrnn::Bounds bounds = cloud.bounds();
  REQUIRE(apfrnn::sweep::estimated_cells_per_point(bounds, cloud.x.size(),
                                                   radius) <= 10000.0);
  CHECK_FALSE(
      apfrnn::sweep::should_use(cloud.x, cloud.y, cloud.z, bounds, radius));
}

TEST_CASE("sweep decision preserves the old choice for small sparse sets") {
  constexpr float radius = 1.0f;
  const Cloud cloud = uniform_cloud(600, 1000.0f, 14);
  const apfrnn::Bounds bounds = cloud.bounds();
  REQUIRE(apfrnn::sweep::estimated_cells_per_point(bounds, cloud.x.size(),
                                                   radius) > 10000.0);
  CHECK(apfrnn::sweep::should_use(cloud.x, cloud.y, cloud.z, bounds, radius));
}

TEST_CASE("sweep decision is re-evaluated when a set moves") {
  constexpr float radius = 2.0f;
  const Cloud clustered = droplet_cloud(64, 128, 20000.0f, 15);
  std::vector<float> xyz = interleave(clustered);

  apfrnn::NSearch search(radius);
  const int set = search.add_point_set(xyz.data(), 8192);
  search.set_all_searches(false);
  search.set_active_search(set, set, true);
  search.run();
  CHECK(search.last_stats().grid_sets == 1);
  CHECK(search.last_stats().sweep_sets == 0);

  // Replace the positions in place with a uniformly sparse set. The cached
  // grid choice must be dropped and the sweep selected instead.
  const Cloud uniform = uniform_cloud(8192, 20000.0f, 16);
  const std::vector<float> moved = interleave(uniform);
  std::copy(moved.begin(), moved.end(), xyz.begin());
  search.run();
  CHECK(search.last_stats().grid_sets == 0);
  CHECK(search.last_stats().sweep_sets == 1);
  CHECK((search.last_stats().paths & apfrnn::NSearch::kPathSameSweep) != 0);
}
