#include "point_cloud.hpp"
#include "scene.hpp"
#include "apfrnn/neighbor_search.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace {

std::vector<float>
interleave(const apfrnn::support::PointCloud &cloud) {
  std::vector<float> xyz(cloud.x.size() * 3);
  for (std::size_t index = 0; index < cloud.x.size(); ++index) {
    xyz[3 * index + 0] = cloud.x[index];
    xyz[3 * index + 1] = cloud.y[index];
    xyz[3 * index + 2] = cloud.z[index];
  }
  return xyz;
}

std::vector<double>
interleave_double(const apfrnn::support::PointCloud &cloud) {
  std::vector<double> xyz(cloud.x.size() * 3);
  for (std::size_t index = 0; index < cloud.x.size(); ++index) {
    xyz[3 * index + 0] = cloud.x[index];
    xyz[3 * index + 1] = cloud.y[index];
    xyz[3 * index + 2] = cloud.z[index];
  }
  return xyz;
}

std::vector<int> sorted_neighbors(apfrnn::NeighborList list) {
  std::vector<int> neighbors(list.begin(), list.end());
  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

void translate(apfrnn::support::PointCloud &cloud, float dx, float dy,
               float dz) {
  for (std::size_t index = 0; index < cloud.x.size(); ++index) {
    cloud.x[index] += dx;
    cloud.y[index] += dy;
    cloud.z[index] += dz;
  }
}

void check_same_set(const apfrnn::NSearch &search, int set,
                    const apfrnn::support::PointCloud &cloud, float radius) {
  for (int point = 0; point < static_cast<int>(cloud.x.size()); ++point) {
    CHECK(sorted_neighbors(search.get_neighbors(set, set, point)) ==
          apfrnn::support::exact_neighbors_for(cloud, point, radius));
  }
}

void check_cross_set(const apfrnn::NSearch &search, int source_set,
                     int target_set,
                     const apfrnn::support::PointCloud &source,
                     const apfrnn::support::PointCloud &target,
                     float radius) {
  for (int point = 0; point < static_cast<int>(source.x.size()); ++point) {
    CHECK(sorted_neighbors(
              search.get_neighbors(source_set, target_set, point)) ==
          apfrnn::support::exact_cross_neighbors_for(source, point, target,
                                                     radius));
  }
}

} // namespace

TEST_CASE("native multi-set search matches brute force") {
  constexpr float radius = 1.5f;

  auto fluid = apfrnn::support::make_point_cloud(64, 10.0f, 101);
  auto boundary = apfrnn::support::make_point_cloud(32, 10.0f, 202);
  auto emitter = apfrnn::support::make_point_cloud(9, 10.0f, 303);

  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);
  auto emitter_xyz = interleave(emitter);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  const int emitter_id = search.add_point_set(
      emitter_xyz.data(), static_cast<int>(emitter.x.size()));

  CHECK(search.n_sets() == 3);
  CHECK(search.get_n_points(fluid_id) == static_cast<int>(fluid.x.size()));

  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.set_active_search(fluid_id, emitter_id, true);
  search.set_active_search(boundary_id, boundary_id, true);

  search.run();

  check_same_set(search, fluid_id, fluid, radius);
  check_same_set(search, boundary_id, boundary, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  check_cross_set(search, fluid_id, emitter_id, fluid, emitter, radius);

  // Inactive directions must stay empty.
  CHECK(search.get_neighbors(boundary_id, fluid_id, 0).empty());
  CHECK(search.get_neighbors(emitter_id, fluid_id, 0).empty());
  CHECK(search.get_neighbors(emitter_id, boundary_id, 0).empty());
}

TEST_CASE("native search updates dynamic sets after move and resize") {
  constexpr float radius = 1.2f;

  auto fluid = apfrnn::support::make_point_cloud(48, 8.0f, 11);
  auto boundary = apfrnn::support::make_point_cloud(24, 8.0f, 12);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);

  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();

  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);

  translate(fluid, 0.07f, -0.03f, 0.05f);
  auto moved_xyz = interleave(fluid);
  search.resize_point_set(fluid_id, moved_xyz.data(),
                          static_cast<int>(fluid.x.size()));
  search.run();

  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

TEST_CASE("native search sees in-place updates after update_point_sets") {
  constexpr float radius = 1.2f;

  auto fluid = apfrnn::support::make_point_cloud(40, 8.0f, 61);
  auto fluid_xyz = interleave(fluid);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.run();
  check_same_set(search, fluid_id, fluid, radius);

  translate(fluid, 0.06f, 0.04f, -0.02f);
  for (std::size_t index = 0; index < fluid.x.size(); ++index) {
    fluid_xyz[3 * index + 0] = fluid.x[index];
    fluid_xyz[3 * index + 1] = fluid.y[index];
    fluid_xyz[3 * index + 2] = fluid.z[index];
  }
  search.update_point_sets();
  search.run();

  check_same_set(search, fluid_id, fluid, radius);
}

TEST_CASE("native search supports adding sets after a run") {
  constexpr float radius = 1.0f;

  auto fluid = apfrnn::support::make_point_cloud(32, 7.0f, 21);
  auto boundary = apfrnn::support::make_point_cloud(16, 7.0f, 22);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.run();
  check_same_set(search, fluid_id, fluid, radius);

  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();

  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

TEST_CASE("native search accepts double input") {
  constexpr float radius = 1.3f;

  auto fluid = apfrnn::support::make_point_cloud(40, 9.0f, 31);
  auto fluid_xyz = interleave_double(fluid);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.run();

  check_same_set(search, fluid_id, fluid, radius);
}

TEST_CASE("native search handles empty point sets") {
  constexpr float radius = 1.0f;

  apfrnn::NSearch search(radius);
  const int empty_id =
      search.add_point_set(static_cast<const float *>(nullptr), 0);
  const int nonempty_id =
      search.add_point_set(static_cast<const float *>(nullptr), 0);
  CHECK(search.get_n_points(empty_id) == 0);
  CHECK(search.get_n_points(nonempty_id) == 0);

  search.set_all_searches(false);
  search.set_active_search(empty_id, empty_id, true);
  search.run();

  CHECK(search.get_neighbors(empty_id, empty_id, 0).empty());
}

TEST_CASE("native search can disable and re-enable a pair") {
  constexpr float radius = 1.1f;

  auto fluid = apfrnn::support::make_point_cloud(32, 8.0f, 41);
  auto boundary = apfrnn::support::make_point_cloud(16, 8.0f, 42);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);

  search.set_active_search(fluid_id, boundary_id, false);
  CHECK(search.get_neighbors(fluid_id, boundary_id, 0).empty());

  search.set_active_search(fluid_id, boundary_id, true);
  search.run();
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

TEST_CASE("native search handles invalid arguments safely") {
  auto fluid = apfrnn::support::make_point_cloud(16, 5.0f, 77);
  auto fluid_xyz = interleave(fluid);

  apfrnn::NSearch search(1.0f);
  CHECK(search.add_point_set(static_cast<const float *>(nullptr), 1) == -1);
  CHECK(search.add_point_set(static_cast<const float *>(nullptr), -1) == -1);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));

  search.resize_point_set(-1, fluid_xyz.data(),
                          static_cast<int>(fluid.x.size()));
  search.resize_point_set(fluid_id, static_cast<const float *>(nullptr), 1);
  search.set_point_set_static(-1, true);
  search.set_active_search(-1, fluid_id, true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);

  CHECK(search.get_n_points(-1) == 0);
  CHECK(search.get_n_points(99) == 0);
  CHECK(search.get_neighbors(-1, fluid_id, 0).empty());
  CHECK(search.get_neighbors(fluid_id, -1, 0).empty());
  CHECK(search.get_neighbors(fluid_id, fluid_id, -1).empty());
  CHECK(search.get_neighbors(fluid_id, fluid_id, 999).empty());
  CHECK(search.get_neighbors(fluid_id, fluid_id, 0).empty()); // not run yet

  search.run();
  check_same_set(search, fluid_id, fluid, 1.0f);
}

TEST_CASE("native search supports double resize and static toggles") {
  constexpr float radius = 1.2f;

  auto fluid = apfrnn::support::make_point_cloud(36, 8.0f, 78);
  auto fluid_xyz = interleave_double(fluid);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.run();
  check_same_set(search, fluid_id, fluid, radius);

  translate(fluid, 0.05f, 0.02f, -0.04f);
  fluid_xyz = interleave_double(fluid);
  search.resize_point_set(fluid_id, fluid_xyz.data(),
                          static_cast<int>(fluid.x.size()));
  search.set_point_set_static(fluid_id, true);
  search.set_point_set_static(fluid_id, false);
  search.run();
  check_same_set(search, fluid_id, fluid, radius);
}

TEST_CASE("native search can shrink to empty and grow again") {
  constexpr float radius = 1.1f;

  auto fluid = apfrnn::support::make_point_cloud(24, 7.0f, 79);
  auto fluid_xyz = interleave(fluid);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.run();
  check_same_set(search, fluid_id, fluid, radius);

  search.resize_point_set(fluid_id, static_cast<const float *>(nullptr), 0);
  search.run();
  CHECK(search.get_n_points(fluid_id) == 0);
  CHECK(search.get_neighbors(fluid_id, fluid_id, 0).empty());

  auto regrown = apfrnn::support::make_point_cloud(12, 7.0f, 80);
  auto regrown_xyz = interleave(regrown);
  search.resize_point_set(fluid_id, regrown_xyz.data(),
                          static_cast<int>(regrown.x.size()));
  search.run();
  check_same_set(search, fluid_id, regrown, radius);
}

TEST_CASE("learnsph-like slab scene remains exact across moving frames") {
  using apfrnn::bench::Distribution;
  using apfrnn::bench::Motion;

  auto scene = apfrnn::bench::make_scene(Distribution::Slab, 160, 909);
  apfrnn::NSearch search(scene.radius);
  const int fluid_id = search.add_point_set(
      scene.fluid_xyz.data(), static_cast<int>(scene.fluid.x.size()));
  const int boundary_id = search.add_point_set(
      scene.boundary_xyz.data(), static_cast<int>(scene.boundary.x.size()),
      true);
  const int emitter_id = search.add_point_set(
      scene.emitter_xyz.data(), static_cast<int>(scene.emitter.x.size()));
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.set_active_search(fluid_id, emitter_id, true);

  for (int frame = 0; frame < 6; ++frame) {
    apfrnn::bench::apply_motion(scene, Motion::SmallMove, frame);
    search.run();

    check_same_set(search, fluid_id, scene.fluid, scene.radius);
    check_cross_set(search, fluid_id, boundary_id, scene.fluid,
                    scene.boundary, scene.radius);
    check_cross_set(search, fluid_id, emitter_id, scene.fluid, scene.emitter,
                    scene.radius);
  }
}

TEST_CASE("native skin reuses and then rebuilds neighbor lists") {
  constexpr float radius = 1.0f;
  constexpr float skin = 0.2f;

  auto fluid = apfrnn::support::make_point_cloud(48, 8.0f, 606);
  auto boundary = apfrnn::support::make_point_cloud(24, 8.0f, 607);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  search.set_skin(skin);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();

  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);

  // Move less than skin/2: the skin lists remain valid and are filtered.
  translate(fluid, 0.04f, -0.02f, 0.03f);
  for (std::size_t index = 0; index < fluid.x.size(); ++index) {
    fluid_xyz[3 * index + 0] = fluid.x[index];
    fluid_xyz[3 * index + 1] = fluid.y[index];
    fluid_xyz[3 * index + 2] = fluid.z[index];
  }
  search.run();
  CHECK(search.last_stats().grid_ms == 0.0);
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);

  // Move farther than the reuse threshold: force a rebuild.
  translate(fluid, 0.5f, 0.4f, -0.3f);
  for (std::size_t index = 0; index < fluid.x.size(); ++index) {
    fluid_xyz[3 * index + 0] = fluid.x[index];
    fluid_xyz[3 * index + 1] = fluid.y[index];
    fluid_xyz[3 * index + 2] = fluid.z[index];
  }
  search.run();
  CHECK(search.last_stats().grid_ms > 0.0);
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

TEST_CASE("native randomized activation matrix matches brute force") {
  std::mt19937 generator(424242);
  std::uniform_real_distribution<float> radius_distribution(0.4f, 1.8f);
  std::uniform_int_distribution<int> count_distribution(0, 24);
  std::uniform_int_distribution<int> set_count_distribution(2, 4);
  std::bernoulli_distribution static_distribution(0.35);
  std::bernoulli_distribution active_distribution(0.55);

  for (int trial = 0; trial < 10; ++trial) {
    const int set_count = set_count_distribution(generator);
    const float radius = radius_distribution(generator);
    std::vector<apfrnn::support::PointCloud> clouds(set_count);
    std::vector<std::vector<float>> xyz(set_count);

    for (int set = 0; set < set_count; ++set) {
      clouds[set] = apfrnn::support::make_point_cloud(
          count_distribution(generator), 8.0f,
          static_cast<uint32_t>(1000 + trial * 17 + set));
      xyz[set] = interleave(clouds[set]);
    }

    apfrnn::NSearch search(radius);
    std::vector<int> ids;
    for (int set = 0; set < set_count; ++set) {
      ids.push_back(search.add_point_set(
          xyz[set].empty() ? static_cast<const float *>(nullptr)
                           : xyz[set].data(),
          static_cast<int>(clouds[set].x.size()),
          static_distribution(generator)));
    }

    std::vector<std::vector<bool>> active(
        set_count, std::vector<bool>(set_count, false));
    search.set_all_searches(false);
    for (int i = 0; i < set_count; ++i) {
      for (int j = 0; j < set_count; ++j) {
        active[i][j] = active_distribution(generator);
        search.set_active_search(ids[i], ids[j], active[i][j]);
      }
    }

    search.run();

    for (int i = 0; i < set_count; ++i) {
      for (int j = 0; j < set_count; ++j) {
        if (!active[i][j]) {
          CHECK(search.get_neighbors(ids[i], ids[j], 0).empty());
          continue;
        }
        if (i == j) {
          check_same_set(search, ids[i], clouds[i], radius);
        } else {
          check_cross_set(search, ids[i], ids[j], clouds[i], clouds[j],
                          radius);
        }
      }
    }
  }
}

TEST_CASE("cross sparse-target scatter matches brute force") {
  constexpr float radius = 2.0f;
  auto fluid = apfrnn::support::make_point_cloud(15000, 20.0f, 1301);
  auto emitter = apfrnn::support::make_point_cloud(400, 20.0f, 1302);
  auto fluid_xyz = interleave(fluid);
  auto emitter_xyz = interleave(emitter);

  apfrnn::NSearch search(radius);
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int emitter_id = search.add_point_set(
      emitter_xyz.data(), static_cast<int>(emitter.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, emitter_id, true);
  search.run();
  CHECK(search.last_stats().cross_fixed_pairs >= 1);
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, emitter_id, fluid, emitter, radius);

  for (std::size_t index = 0; index < fluid.x.size(); ++index) {
    fluid.x[index] += 0.11f;
    fluid.z[index] -= 0.07f;
    fluid_xyz[3 * index + 0] = fluid.x[index];
    fluid_xyz[3 * index + 2] = fluid.z[index];
  }
  search.run();
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, emitter_id, fluid, emitter, radius);
}

TEST_CASE("fused dense cell ranges match brute force") {
  constexpr float radius = 2.0f;
  // 312 points per cell: too dense for the on-the-fly writer, so this takes
  // the fused dense cell-range path used by the active2 writers.
  auto fluid = apfrnn::support::make_point_cloud(20000, 8.0f, 1601);
  auto boundary = apfrnn::support::make_point_cloud(5000, 8.0f, 1602);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();

  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

#ifdef APFRNN_HAVE_SYCL
TEST_CASE("GPU backend rebuilds a static set's grid when it changes") {
  constexpr float radius = 1.0f;
  auto fluid = apfrnn::support::make_point_cloud(20000, 25.0f, 1801);
  auto boundary = apfrnn::support::make_point_cloud(20000, 25.0f, 1802);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  if (!search.set_search_backend(apfrnn::SearchBackend::GpuExact)) {
    return;
  }
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);

  // Same buffer and count, new positions: the cached grid must not be used.
  translate(boundary, 0.5f, 0.0f, 0.0f);
  boundary_xyz = interleave(boundary);
  search.resize_point_set(boundary_id, boundary_xyz.data(),
                          static_cast<int>(boundary.x.size()));
  search.run();
  CHECK(search.search_backend() == apfrnn::SearchBackend::GpuExact);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

TEST_CASE("GPU backend handles repeated deferred runs") {
  constexpr float radius = 1.0f;
  auto fluid = apfrnn::support::make_point_cloud(20000, 25.0f, 1951);
  auto boundary = apfrnn::support::make_point_cloud(5000, 25.0f, 1952);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  if (!search.set_search_backend(apfrnn::SearchBackend::GpuExact)) {
    return;
  }
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();

  // Queue several same-set output copies without reading rows in between;
  // the backend must double-buffer them and the read below must wait.
  for (int frame = 0; frame < 6; ++frame) {
    search.run();
  }
  CHECK(search.search_backend() == apfrnn::SearchBackend::GpuExact);
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

TEST_CASE("GPU backend keeps independent searches separate") {
  constexpr float radius = 1.0f;
  auto first = apfrnn::support::make_point_cloud(20000, 20.0f, 1901);
  auto second = apfrnn::support::make_point_cloud(20000, 20.0f, 1902);
  auto first_xyz = interleave(first);
  auto second_xyz = interleave(second);

  apfrnn::NSearch a(radius);
  apfrnn::NSearch b(radius);
  if (!a.set_search_backend(apfrnn::SearchBackend::GpuExact) ||
      !b.set_search_backend(apfrnn::SearchBackend::GpuExact)) {
    return;
  }
  const int ia = a.add_point_set(first_xyz.data(),
                                 static_cast<int>(first.x.size()));
  const int ib = b.add_point_set(second_xyz.data(),
                                 static_cast<int>(second.x.size()));
  a.run();
  b.run();  // same set index and pair as `a`
  check_same_set(a, ia, first, radius);
  check_same_set(b, ib, second, radius);
}

TEST_CASE("GPU backend handles more than four point sets") {
  constexpr float radius = 1.0f;
  std::vector<apfrnn::support::PointCloud> clouds;
  std::vector<std::vector<float>> xyz;
  for (uint32_t set = 0; set < 5; ++set) {
    clouds.push_back(apfrnn::support::make_point_cloud(4000, 15.0f, 2000 + set));
    xyz.push_back(interleave(clouds.back()));
  }
  apfrnn::NSearch search(radius);
  if (!search.set_search_backend(apfrnn::SearchBackend::GpuExact)) {
    return;
  }
  for (std::size_t set = 0; set < clouds.size(); ++set) {
    search.add_point_set(xyz[set].data(),
                         static_cast<int>(clouds[set].x.size()));
  }
  search.run();
  CHECK(search.search_backend() == apfrnn::SearchBackend::GpuExact);
  for (int set = 0; set < 5; ++set) {
    check_same_set(search, set, clouds[static_cast<std::size_t>(set)], radius);
    check_cross_set(search, set, (set + 1) % 5,
                    clouds[static_cast<std::size_t>(set)],
                    clouds[static_cast<std::size_t>((set + 1) % 5)], radius);
  }
}

TEST_CASE("GPU Exact backend matches CPU Exact and restores CPU rows") {
  constexpr float radius = 2.0f;
  auto fluid = apfrnn::support::make_point_cloud(20000, 28.0f, 1701);
  auto boundary = apfrnn::support::make_point_cloud(5000, 28.0f, 1702);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();
  if (!search.gpu_available()) {
    return;
  }

  const int fluid_count = static_cast<int>(fluid.x.size());
  std::vector<std::vector<uint32_t>> same_ref(fluid_count);
  std::vector<std::vector<uint32_t>> cross_ref(fluid_count);
  for (int point = 0; point < fluid_count; ++point) {
    auto same = search.get_neighbors(fluid_id, fluid_id, point);
    same_ref[point].assign(same.begin(), same.end());
    std::sort(same_ref[point].begin(), same_ref[point].end());
    auto cross = search.get_neighbors(fluid_id, boundary_id, point);
    cross_ref[point].assign(cross.begin(), cross.end());
    std::sort(cross_ref[point].begin(), cross_ref[point].end());
  }

  REQUIRE(search.set_search_backend(apfrnn::SearchBackend::GpuExact));
  CHECK(search.search_backend() == apfrnn::SearchBackend::GpuExact);
  search.run();

  int64_t score_mismatches = 0;
  int64_t false_positives = 0;
  auto check_rows = [&](int target_id, bool same_set,
                        const std::vector<float> &target_xyz,
                        const std::vector<std::vector<uint32_t>> &ref) {
    for (int point = 0; point < fluid_count; ++point) {
      auto got = search.get_neighbors(fluid_id, target_id, point);
      std::vector<uint32_t> sorted(got.begin(), got.end());
      std::sort(sorted.begin(), sorted.end());
      if (sorted != ref[static_cast<std::size_t>(point)]) {
        ++score_mismatches;
      }
      const float qx = fluid.x[static_cast<std::size_t>(point)];
      const float qy = fluid.y[static_cast<std::size_t>(point)];
      const float qz = fluid.z[static_cast<std::size_t>(point)];
      for (uint32_t id : got) {
        const float dx = target_xyz[3 * static_cast<std::size_t>(id) + 0] - qx;
        const float dy = target_xyz[3 * static_cast<std::size_t>(id) + 1] - qy;
        const float dz = target_xyz[3 * static_cast<std::size_t>(id) + 2] - qz;
        if (dx * dx + dy * dy + dz * dz > radius * radius * 1.00001f) {
          ++false_positives;
        }
        if (!std::binary_search(ref[static_cast<std::size_t>(point)].begin(),
                                ref[static_cast<std::size_t>(point)].end(),
                                id)) {
          ++false_positives;
        }
      }
    }
    (void)same_set;
  };
  check_rows(fluid_id, true, fluid_xyz, same_ref);
  check_rows(boundary_id, false, boundary_xyz, cross_ref);
  CHECK(score_mismatches == 0);
  CHECK(false_positives == 0);

  // A second GPU run must return the same sets (neighbor order may differ
  // because the GPU grid scatter order is not specified).
  score_mismatches = 0;
  false_positives = 0;
  search.run();
  check_rows(fluid_id, true, fluid_xyz, same_ref);
  check_rows(boundary_id, false, boundary_xyz, cross_ref);
  CHECK(score_mismatches == 0);
  CHECK(false_positives == 0);

  CHECK(search.set_search_backend(apfrnn::SearchBackend::CpuExact));
  search.run();
  int64_t restored_mismatches = 0;
  for (int point = 0; point < fluid_count; ++point) {
    auto same = search.get_neighbors(fluid_id, fluid_id, point);
    std::vector<uint32_t> sorted(same.begin(), same.end());
    std::sort(sorted.begin(), sorted.end());
    restored_mismatches +=
        (sorted == same_ref[static_cast<std::size_t>(point)]) ? 0 : 1;
    auto cross = search.get_neighbors(fluid_id, boundary_id, point);
    std::vector<uint32_t> csorted(cross.begin(), cross.end());
    std::sort(csorted.begin(), csorted.end());
    restored_mismatches +=
        (csorted == cross_ref[static_cast<std::size_t>(point)]) ? 0 : 1;
  }
  CHECK(restored_mismatches == 0);
}
#endif

TEST_CASE("dynamic target cache reuses and invalidates") {
  constexpr float radius = 2.0f;
  auto fluid = apfrnn::support::make_point_cloud(15000, 20.0f, 1401);
  auto target = apfrnn::support::make_point_cloud(3000, 20.0f, 1402);
  auto fluid_xyz = interleave(fluid);
  auto target_xyz = interleave(target);

  apfrnn::NSearch search(radius);
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int target_id = search.add_point_set(
      target_xyz.data(), static_cast<int>(target.x.size()));
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, target_id, true);
  search.run();
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, target_id, fluid, target, radius);

  // Source moves while an unchanged target reuses its cached grid.
  for (std::size_t index = 0; index < fluid.x.size(); ++index) {
    fluid.x[index] += 0.13f;
    fluid.z[index] -= 0.09f;
    fluid_xyz[3 * index + 0] = fluid.x[index];
    fluid_xyz[3 * index + 2] = fluid.z[index];
  }
  search.run();
  CHECK(search.last_stats().reused_sets >= 1);
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, target_id, fluid, target, radius);

  // A moved target must invalidate the cache and rebuild.
  for (std::size_t index = 0; index < target.x.size(); ++index) {
    target.x[index] += 0.21f;
    target_xyz[3 * index + 0] = target.x[index];
  }
  search.run();
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, target_id, fluid, target, radius);
}

TEST_CASE("cross on-the-fly dense ranges match brute force") {
  constexpr float radius = 2.0f;
  auto fluid = apfrnn::support::make_point_cloud(20000, 28.0f, 1201);
  auto boundary = apfrnn::support::make_point_cloud(5000, 28.0f, 1202);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();
  CHECK(search.last_stats().cross_fixed_pairs >= 1);
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);

  // A second frame moves the source cells while the target stays static.
  for (std::size_t index = 0; index < fluid.x.size(); ++index) {
    fluid.x[index] += 0.37f;
    fluid.y[index] -= 0.21f;
    fluid_xyz[3 * index + 0] = fluid.x[index];
    fluid_xyz[3 * index + 1] = fluid.y[index];
  }
  search.run();
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

TEST_CASE("native search hash fallback matches brute force") {
  constexpr float radius = 1.0f;
  auto cloud = apfrnn::support::make_point_cloud(80, 10.0f, 1234);
  apfrnn::support::set_extreme_outlier(cloud, 0);
  auto xyz = interleave(cloud);

  apfrnn::NSearch search(radius);
  const int id = search.add_point_set(xyz.data(),
                                      static_cast<int>(cloud.x.size()));
  search.set_all_searches(false);
  search.set_active_search(id, id, true);
  search.run();

  check_same_set(search, id, cloud, radius);
}

TEST_CASE("native fixed capacity exact boundary and overflow") {
  constexpr float radius = 100.0f;

  for (int count : {129, 130}) {
    auto cloud = apfrnn::support::make_point_cloud(count, 1.0f, 555);
    auto xyz = interleave(cloud);

    apfrnn::NSearch search(radius);
    const int id =
        search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
    search.set_all_searches(false);
    search.set_active_search(id, id, true);
    search.run();

    check_same_set(search, id, cloud, radius);
  }
}

TEST_CASE("query-major fixed writers match brute force") {
  constexpr int num_points = 160;
  constexpr float radius = 1.4f;
  constexpr int capacity = 128;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 16.0f, 91);
  apfrnn::NeighborSearchData data;
  apfrnn::build_point_grid_inplace(data, cloud.x, cloud.y, cloud.z, radius);
  apfrnn::build_same_set_cell_ranges_inplace(data);

  auto verify_rows = [&](const apfrnn::IntBuffer &values,
                         const apfrnn::IntBuffer &counts, int row_stride) {
    REQUIRE(counts.size() == static_cast<std::size_t>(num_points));
    for (int point = 0; point < num_points; ++point) {
      const int sorted_index = apfrnn::support::sorted_index_for(data, point);
      const int count = counts[static_cast<std::size_t>(sorted_index)];
      REQUIRE(count <= capacity);
      const int *row =
          values.data() + static_cast<std::size_t>(sorted_index) *
                              static_cast<std::size_t>(row_stride);
      std::vector<int> neighbors(row, row + count);
      std::sort(neighbors.begin(), neighbors.end());
      CHECK(neighbors ==
            apfrnn::support::exact_neighbors_for(cloud, point, radius));
    }
  };

  apfrnn::IntBuffer values;
  apfrnn::IntBuffer counts;

  apfrnn::write_neighbors_fixed_query_pair_parallel(data, capacity, values,
                                                    counts);
  verify_rows(values, counts, capacity + 8);

  apfrnn::write_neighbors_fixed_query_active2_parallel(data, capacity, values,
                                                       counts);
  verify_rows(values, counts, capacity + 8);
}

TEST_CASE("native search handles empty cross-set directions") {
  constexpr float radius = 1.0f;
  auto fluid = apfrnn::support::make_point_cloud(16, 5.0f, 777);
  auto fluid_xyz = interleave(fluid);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int empty_id =
      search.add_point_set(static_cast<const float *>(nullptr), 0);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, empty_id, true);
  search.set_active_search(empty_id, fluid_id, true);
  search.set_active_search(empty_id, empty_id, true);
  search.run();

  CHECK(search.get_neighbors(fluid_id, empty_id, 0).empty());
  CHECK(search.get_neighbors(empty_id, fluid_id, 0).empty());
  CHECK(search.get_neighbors(empty_id, empty_id, 0).empty());
}

TEST_CASE("cross search reusing a sorted source matches a fresh cross build") {
  constexpr int query_points = 96;
  constexpr int target_points = 320;
  constexpr float radius = 1.15f;

  auto query = apfrnn::support::make_point_cloud(query_points, 12.0f, 71);
  auto target = apfrnn::support::make_point_cloud(target_points, 12.0f, 72);

  auto query_data = apfrnn::build_neighbor_search_data(
      query.x, query.y, query.z, radius);
  auto target_data = apfrnn::build_neighbor_search_data(
      target.x, target.y, target.z, radius);

  auto fresh = apfrnn::build_cross_neighbor_search_data(
      query.x, query.y, query.z, target_data, radius);
  apfrnn::write_cross_neighbors_parallel(fresh, target_data);

  apfrnn::CrossNeighborSearchData reused;
  apfrnn::build_cross_neighbor_search_data_from_sorted_inplace(
      reused, query_data, target_data, radius);
  apfrnn::write_cross_neighbors_parallel(reused, target_data);

  CHECK(reused.row_ptr == fresh.row_ptr);
  CHECK(reused.col_idx == fresh.col_idx);
}

TEST_CASE("query-major cross fixed writer matches brute force") {
  constexpr int query_points = 96;
  constexpr int target_points = 320;
  constexpr float radius = 1.15f;
  constexpr int capacity = 64;

  auto query = apfrnn::support::make_point_cloud(query_points, 12.0f, 71);
  auto target = apfrnn::support::make_point_cloud(target_points, 12.0f, 72);
  auto query_data = apfrnn::build_neighbor_search_data(
      query.x, query.y, query.z, radius);
  auto target_data = apfrnn::build_neighbor_search_data(
      target.x, target.y, target.z, radius);

  apfrnn::CrossNeighborSearchData cross;
  apfrnn::build_cross_neighbor_search_data_from_sorted_inplace(
      cross, query_data, target_data, radius);

  auto verify_rows = [&](const apfrnn::IntBuffer &values,
                         const apfrnn::IntBuffer &counts, int row_stride) {
    for (int point = 0; point < query_points; ++point) {
      const int sorted_index =
          apfrnn::support::sorted_index_for(query_data, point);
      const int count = counts[static_cast<std::size_t>(sorted_index)];
      REQUIRE(count <= capacity);
      const int *row =
          values.data() + static_cast<std::size_t>(sorted_index) *
                              static_cast<std::size_t>(row_stride);
      std::vector<int> neighbors(row, row + count);
      std::sort(neighbors.begin(), neighbors.end());

      std::vector<int> expected;
      for (int other = 0; other < target_points; ++other) {
        const float dx = target.x[static_cast<std::size_t>(other)] -
                         query.x[static_cast<std::size_t>(point)];
        const float dy = target.y[static_cast<std::size_t>(other)] -
                         query.y[static_cast<std::size_t>(point)];
        const float dz = target.z[static_cast<std::size_t>(other)] -
                         query.z[static_cast<std::size_t>(point)];
        if (dx * dx + dy * dy + dz * dz <= radius * radius) {
          expected.push_back(other);
        }
      }
      std::sort(expected.begin(), expected.end());
      CHECK(neighbors == expected);
    }
  };

  apfrnn::IntBuffer values;
  apfrnn::IntBuffer counts;
  apfrnn::write_cross_neighbors_fixed_query_pair_parallel(
      cross, target_data, capacity, values, counts);
  verify_rows(values, counts, capacity + 8);

  apfrnn::write_cross_neighbors_fixed_query_active2_parallel(
      cross, target_data, capacity, values, counts);
  verify_rows(values, counts, capacity + 8);
}

TEST_CASE("active2 writers stay in bounds when rows overflow") {
  constexpr int num_points = 200;
  constexpr float radius = 100.0f;
  constexpr int capacity = 1;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 1.0f, 4242);
  apfrnn::NeighborSearchData data;
  apfrnn::build_point_grid_inplace(data, cloud.x, cloud.y, cloud.z, radius);
  apfrnn::build_same_set_cell_ranges_inplace(data);

  apfrnn::IntBuffer values;
  apfrnn::IntBuffer counts;
  apfrnn::write_neighbors_fixed_query_active2_parallel(data, capacity, values,
                                                       counts);
  REQUIRE(values.size() ==
          static_cast<std::size_t>(num_points) * (capacity + 8));
  REQUIRE(counts.size() == static_cast<std::size_t>(num_points));
  for (int count : counts) {
    CHECK(count == num_points - 1);
  }

  auto target = apfrnn::support::make_point_cloud(64, 1.0f, 4343);
  auto target_data = apfrnn::build_neighbor_search_data(
      target.x, target.y, target.z, radius);
  apfrnn::CrossNeighborSearchData cross;
  apfrnn::build_cross_neighbor_search_data_from_sorted_inplace(
      cross, data, target_data, radius);
  apfrnn::write_cross_neighbors_fixed_query_active2_parallel(
      cross, target_data, capacity, values, counts);
  REQUIRE(values.size() ==
          static_cast<std::size_t>(num_points) * (capacity + 8));
  for (int count : counts) {
    CHECK(count == 64);
  }
}

TEST_CASE("native search handles isolated points without neighbor rows") {
  constexpr float radius = 0.5f;
  constexpr int fluid_points = 64;
  constexpr int boundary_points = 32;

  apfrnn::support::PointCloud fluid;
  apfrnn::support::PointCloud boundary;
  fluid.x.resize(fluid_points);
  fluid.y.resize(fluid_points);
  fluid.z.resize(fluid_points);
  boundary.x.resize(boundary_points);
  boundary.y.resize(boundary_points);
  boundary.z.resize(boundary_points);
  for (int i = 0; i < fluid_points; ++i) {
    fluid.x[static_cast<std::size_t>(i)] = 10.0f * i;
    fluid.y[static_cast<std::size_t>(i)] = 3.0f * i;
    fluid.z[static_cast<std::size_t>(i)] = -7.0f * i;
  }
  for (int i = 0; i < boundary_points; ++i) {
    boundary.x[static_cast<std::size_t>(i)] = -1000.0f - 10.0f * i;
    boundary.y[static_cast<std::size_t>(i)] = 500.0f + 3.0f * i;
    boundary.z[static_cast<std::size_t>(i)] = 900.0f - 7.0f * i;
  }

  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);
  apfrnn::NSearch search(radius);
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();

  for (int point = 0; point < fluid_points; ++point) {
    CHECK(search.get_neighbors(fluid_id, fluid_id, point).empty());
    CHECK(search.get_neighbors(fluid_id, boundary_id, point).empty());
  }
  CHECK(search.last_stats().max_neighbors == 0);
}

TEST_CASE("native search sparse fallbacks match brute force") {
  auto make_clustered_domain = [](int clusters, int per_cluster, float side,
                                  float spread, uint32_t seed) {
    apfrnn::support::PointCloud cloud;
    cloud.x.resize(static_cast<std::size_t>(clusters * per_cluster));
    cloud.y.resize(static_cast<std::size_t>(clusters * per_cluster));
    cloud.z.resize(static_cast<std::size_t>(clusters * per_cluster));
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> center(0.1f * side, 0.9f * side);
    std::normal_distribution<float> noise(0.0f, spread);
    for (int c = 0; c < clusters; ++c) {
      const float cx = center(generator);
      const float cy = center(generator);
      const float cz = center(generator);
      for (int i = 0; i < per_cluster; ++i) {
        const std::size_t index =
            static_cast<std::size_t>(c * per_cluster + i);
        cloud.x[index] = cx + noise(generator);
        cloud.y[index] = cy + noise(generator);
        cloud.z[index] = cz + noise(generator);
      }
    }
    return cloud;
  };

  // AABB density selects the sweep backend (ratio > 10000).
  {
    constexpr float radius = 1.0f;
    auto fluid = make_clustered_domain(10, 20, 1000.0f, 2.0f, 901);
    auto boundary = make_clustered_domain(5, 20, 1000.0f, 2.0f, 902);
    auto fluid_xyz = interleave(fluid);
    auto boundary_xyz = interleave(boundary);
    apfrnn::NSearch search(radius);
    const int fluid_id = search.add_point_set(
        fluid_xyz.data(), static_cast<int>(fluid.x.size()));
    const int boundary_id = search.add_point_set(
        boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
    search.set_all_searches(false);
    search.set_active_search(fluid_id, fluid_id, true);
    search.set_active_search(fluid_id, boundary_id, true);
    search.run();
    check_same_set(search, fluid_id, fluid, radius);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }

  // The same sparse backend must also filter skin lists correctly.
  {
    constexpr float radius = 1.0f;
    constexpr float skin = 0.25f;
    auto fluid = make_clustered_domain(10, 20, 1000.0f, 2.0f, 905);
    auto boundary = make_clustered_domain(5, 20, 1000.0f, 2.0f, 906);
    auto fluid_xyz = interleave(fluid);
    auto boundary_xyz = interleave(boundary);
    apfrnn::NSearch search(radius);
    search.set_skin(skin);
    const int fluid_id = search.add_point_set(
        fluid_xyz.data(), static_cast<int>(fluid.x.size()));
    const int boundary_id = search.add_point_set(
        boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
    search.set_all_searches(false);
    search.set_active_search(fluid_id, fluid_id, true);
    search.set_active_search(fluid_id, boundary_id, true);
    search.run();
    check_same_set(search, fluid_id, fluid, radius);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);

    translate(fluid, 0.05f, -0.03f, 0.02f);
    for (std::size_t index = 0; index < fluid.x.size(); ++index) {
      fluid_xyz[3 * index + 0] = fluid.x[index];
      fluid_xyz[3 * index + 1] = fluid.y[index];
      fluid_xyz[3 * index + 2] = fluid.z[index];
    }
    search.run();
    check_same_set(search, fluid_id, fluid, radius);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }

  // A grid source cross-searched against a sparse sweep target exercises
  // the unordered-source fallback and the non-empty list fill pass.
  {
    constexpr float radius = 1.0f;
    auto fluid = apfrnn::support::make_point_cloud(20000, 28.0f, 907);
    auto boundary = apfrnn::support::make_point_cloud(400, 28.0f, 908);
    apfrnn::support::set_extreme_outlier(boundary, 0);
    auto fluid_xyz = interleave(fluid);
    auto boundary_xyz = interleave(boundary);
    apfrnn::NSearch search(radius);
    const int fluid_id = search.add_point_set(
        fluid_xyz.data(), static_cast<int>(fluid.x.size()));
    const int boundary_id = search.add_point_set(
        boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
    search.set_all_searches(false);
    search.set_active_search(fluid_id, boundary_id, true);
    search.run();
    CHECK(search.last_stats().grid_sets >= 1);
    CHECK(search.last_stats().sweep_sets >= 1);
    std::size_t cross_neighbors = 0;
    for (int point = 0; point < static_cast<int>(fluid.x.size()); ++point) {
      cross_neighbors += search.get_neighbors(fluid_id, boundary_id, point).size();
    }
    CHECK(cross_neighbors > 0);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }

  // Both sides sorted: the ordered-source fast path must also fill lists.
  {
    constexpr float radius = 1.0f;
    auto fluid = make_clustered_domain(5, 20, 1000.0f, 2.0f, 909);
    auto boundary = fluid;
    apfrnn::support::set_extreme_outlier(boundary, 0);
    auto fluid_xyz = interleave(fluid);
    auto boundary_xyz = interleave(boundary);
    apfrnn::NSearch search(radius);
    const int fluid_id = search.add_point_set(
        fluid_xyz.data(), static_cast<int>(fluid.x.size()));
    const int boundary_id = search.add_point_set(
        boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
    search.set_all_searches(false);
    search.set_active_search(fluid_id, boundary_id, true);
    search.run();
    std::size_t cross_neighbors = 0;
    for (int point = 0; point < static_cast<int>(fluid.x.size()); ++point) {
      cross_neighbors += search.get_neighbors(fluid_id, boundary_id, point).size();
    }
    CHECK(cross_neighbors > 0);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }

  // Intermediate density stays on the flat grid.
  {
    constexpr float radius = 1.0f;
    auto fluid = make_clustered_domain(20, 100, 200.0f, 2.0f, 903);
    auto boundary = make_clustered_domain(6, 100, 200.0f, 2.0f, 904);
    auto fluid_xyz = interleave(fluid);
    auto boundary_xyz = interleave(boundary);
    apfrnn::NSearch search(radius);
    const int fluid_id = search.add_point_set(
        fluid_xyz.data(), static_cast<int>(fluid.x.size()));
    const int boundary_id = search.add_point_set(
        boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
    search.set_all_searches(false);
    search.set_active_search(fluid_id, fluid_id, true);
    search.set_active_search(fluid_id, boundary_id, true);
    search.run();
    check_same_set(search, fluid_id, fluid, radius);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }
}

TEST_CASE("memory budget forces CSR fallback") {
  constexpr float radius = 100.0f;
  auto fluid = apfrnn::support::make_point_cloud(160, 1.0f, 5150);
  auto boundary = apfrnn::support::make_point_cloud(80, 1.0f, 5151);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  search.set_memory_budget(1);
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();

  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  CHECK(search.last_stats().same_csr_pairs == 1);
  CHECK(search.last_stats().cross_csr_pairs == 1);
}

TEST_CASE("block-fixed rows match brute force") {
  constexpr int num_points = 2048;
  constexpr float radius = 2.0f;
  auto cloud = apfrnn::support::make_point_cloud(num_points, 20.0f, 9001);
  // One dense cluster makes the global fixed-row capacity (and its memory)
  // much larger than the per-row block layout.
  for (int i = 0; i < 200; ++i) {
    cloud.x[static_cast<std::size_t>(i)] = 10.0f + 1e-4f * i;
    cloud.y[static_cast<std::size_t>(i)] = 10.0f;
    cloud.z[static_cast<std::size_t>(i)] = 10.0f;
  }
  auto xyz = interleave(cloud);

  apfrnn::NSearch search(radius);
  search.set_memory_budget(1 << 20);
  const int set =
      search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
  search.set_all_searches(false);
  search.set_active_search(set, set, true);
  search.run();
  search.run();  // cached per-row capacities and the no-count path

  CHECK(search.last_stats().same_block_pairs == 1);
  check_same_set(search, set, cloud, radius);

  // A skin invalidates the block path; the exact fallback must still work.
  search.set_skin(0.1f);
  search.run();
  check_same_set(search, set, cloud, radius);
}

TEST_CASE("sparse backends fall back to grid under memory pressure") {
  auto make_clustered_domain = [](int clusters, int per_cluster, float side,
                                  float spread, uint32_t seed) {
    apfrnn::support::PointCloud cloud;
    cloud.x.resize(static_cast<std::size_t>(clusters * per_cluster));
    cloud.y.resize(static_cast<std::size_t>(clusters * per_cluster));
    cloud.z.resize(static_cast<std::size_t>(clusters * per_cluster));
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> center(0.1f * side, 0.9f * side);
    std::normal_distribution<float> noise(0.0f, spread);
    for (int c = 0; c < clusters; ++c) {
      const float cx = center(generator);
      const float cy = center(generator);
      const float cz = center(generator);
      for (int i = 0; i < per_cluster; ++i) {
        const std::size_t index =
            static_cast<std::size_t>(c * per_cluster + i);
        cloud.x[index] = cx + noise(generator);
        cloud.y[index] = cy + noise(generator);
        cloud.z[index] = cz + noise(generator);
      }
    }
    return cloud;
  };

  for (int variant = 0; variant < 2; ++variant) {
    constexpr float radius = 1.0f;
    const int clusters = variant == 0 ? 10 : 20;
    const int per_cluster = variant == 0 ? 20 : 100;
    const float side = variant == 0 ? 1000.0f : 200.0f;
    auto fluid = make_clustered_domain(clusters, per_cluster, side, 2.0f,
                                       1000 + variant);
    auto boundary = make_clustered_domain(5, per_cluster, side, 2.0f,
                                          2000 + variant);
    auto fluid_xyz = interleave(fluid);
    auto boundary_xyz = interleave(boundary);
    apfrnn::NSearch search(radius);
    search.set_memory_budget(1);
    const int fluid_id = search.add_point_set(
        fluid_xyz.data(), static_cast<int>(fluid.x.size()));
    const int boundary_id = search.add_point_set(
        boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
    search.set_all_searches(false);
    search.set_active_search(fluid_id, fluid_id, true);
    search.set_active_search(fluid_id, boundary_id, true);
    search.run();
    check_same_set(search, fluid_id, fluid, radius);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }
}

TEST_CASE("blocked cell writers match brute force") {
  constexpr float radius = 1.0f;
  auto fluid = apfrnn::support::make_point_cloud(2000, 3.0f, 7071);
  auto boundary = apfrnn::support::make_point_cloud(1000, 3.0f, 7072);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();

  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

TEST_CASE("dense clustered blocked writer matches brute force") {
  constexpr float radius = 0.5f;
  auto make_clusters = [](int clusters, int per_cluster, float side,
                          float spread, uint32_t seed) {
    apfrnn::support::PointCloud cloud;
    cloud.x.resize(static_cast<std::size_t>(clusters * per_cluster));
    cloud.y.resize(static_cast<std::size_t>(clusters * per_cluster));
    cloud.z.resize(static_cast<std::size_t>(clusters * per_cluster));
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> center(0.25f * side, 0.75f * side);
    std::normal_distribution<float> noise(0.0f, spread);
    for (int c = 0; c < clusters; ++c) {
      const float cx = center(generator);
      const float cy = center(generator);
      const float cz = center(generator);
      for (int i = 0; i < per_cluster; ++i) {
        const std::size_t index =
            static_cast<std::size_t>(c * per_cluster + i);
        cloud.x[index] = cx + noise(generator);
        cloud.y[index] = cy + noise(generator);
        cloud.z[index] = cz + noise(generator);
      }
    }
    return cloud;
  };

  auto fluid = make_clusters(8, 750, 2.0f, 0.02f, 8081);
  auto boundary = make_clusters(8, 250, 2.0f, 0.02f, 8082);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);
  apfrnn::NSearch search(radius);
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.run();
  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
}

TEST_CASE("native search falls back when fixed rows overflow") {
  constexpr float radius = 100.0f;

  // Every point sees every other point, exceeding both fixed capacities.
  auto fluid = apfrnn::support::make_point_cloud(200, 1.0f, 81);
  auto boundary = apfrnn::support::make_point_cloud(100, 1.0f, 82);
  auto emitter = apfrnn::support::make_point_cloud(50, 1.0f, 83);
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);
  auto emitter_xyz = interleave(emitter);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  const int emitter_id = search.add_point_set(
      emitter_xyz.data(), static_cast<int>(emitter.x.size()));

  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);
  search.set_active_search(fluid_id, emitter_id, true);
  search.run();

  check_same_set(search, fluid_id, fluid, radius);
  check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  check_cross_set(search, fluid_id, emitter_id, fluid, emitter, radius);
}

TEST_CASE("native search radius changes rebuild the grid") {
  constexpr float small_radius = 0.7f;
  constexpr float large_radius = 1.5f;

  auto fluid = apfrnn::support::make_point_cloud(36, 8.0f, 51);
  auto fluid_xyz = interleave(fluid);

  apfrnn::NSearch search(small_radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.run();
  check_same_set(search, fluid_id, fluid, small_radius);

  search.set_radius(large_radius);
  search.run();
  check_same_set(search, fluid_id, fluid, large_radius);
}

TEST_CASE("active2 high-capacity writer path is exercised") {
  constexpr float radius = 1.0f;
  auto cloud = apfrnn::support::make_point_cloud(512, 8.0f, 9091);
  apfrnn::NeighborSearchData data;
  apfrnn::build_neighbor_search_data_inplace(data, cloud.x, cloud.y, cloud.z,
                                             radius);
  apfrnn::IntBuffer fixed;
  apfrnn::IntBuffer counts;
  apfrnn::write_neighbors_fixed_query_active2_parallel(data, 256, fixed, counts);
  int total = 0;
  for (int count : counts) {
    CHECK(count >= 0);
    CHECK(count <= 256);
    total += count;
  }
  CHECK(total >= 0);
}

TEST_CASE("cross blocked writer path is exercised") {
  constexpr float radius = 1.0f;
  auto source = apfrnn::support::make_point_cloud(2000, 2.0f, 9101);
  auto target = apfrnn::support::make_point_cloud(2000, 2.0f, 9102);
  apfrnn::NeighborSearchData source_data;
  apfrnn::NeighborSearchData target_data;
  apfrnn::build_neighbor_search_data_inplace(source_data, source.x, source.y,
                                             source.z, radius);
  apfrnn::build_neighbor_search_data_inplace(target_data, target.x, target.y,
                                             target.z, radius);
  apfrnn::ensure_point_blocks(source_data);
  apfrnn::ensure_point_blocks(target_data);
  apfrnn::CrossNeighborSearchData cross;
  apfrnn::build_cross_cell_ranges_from_sorted_inplace(
      cross, source_data, target_data, radius);
  apfrnn::IntBuffer fixed;
  apfrnn::IntBuffer counts;
  apfrnn::write_cross_neighbors_fixed_query_blocked_parallel(
      cross, target_data, source_data, 512, fixed, counts);
  CHECK(counts.size() == static_cast<std::size_t>(source_data.num_points));
}

TEST_CASE("moving sweep sets rebuild their x order") {
  constexpr float radius = 1.0f;
  // ~1.7M cells per point: far past the sweep threshold. Each point has a
  // partner 0.5 away so every frame has neighbors to find.
  auto make_frame = [](uint32_t seed) {
    auto cloud = apfrnn::support::make_point_cloud(300, 1000.0f, seed);
    for (std::size_t i = 0; i < 300; ++i) {
      cloud.x.push_back(cloud.x[i] + 0.5f);
      cloud.y.push_back(cloud.y[i]);
      cloud.z.push_back(cloud.z[i]);
    }
    return cloud;
  };

  auto fluid = make_frame(5101);
  auto target = make_frame(5102);
  auto fluid_xyz = interleave(fluid);
  auto target_xyz = interleave(target);

  apfrnn::NSearch search(radius);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int target_id = search.add_point_set(
      target_xyz.data(), static_cast<int>(target.x.size()));
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, target_id, true);

  for (uint32_t frame = 0; frame < 3; ++frame) {
    if (frame > 0) {
      // Same buffers, unrelated positions: the previous x order is useless.
      fluid = make_frame(6000 + frame);
      target = make_frame(7000 + frame);
      const auto next_fluid = interleave(fluid);
      const auto next_target = interleave(target);
      std::copy(next_fluid.begin(), next_fluid.end(), fluid_xyz.begin());
      std::copy(next_target.begin(), next_target.end(), target_xyz.begin());
    }
    search.run();
    CHECK(search.last_stats().sweep_sets == 2);
    check_same_set(search, fluid_id, fluid, radius);
    check_cross_set(search, fluid_id, target_id, fluid, target, radius);
  }
}

TEST_CASE("shared dense ranges are dropped when a set leaves the dense grid") {
  // Frame 1 shares the dense range cache across several pairs; then one point
  // per set moves far enough that the bounding box exceeds the dense grid
  // limit (but not far enough for the sweep backend), so every set falls back
  // to the hash grid, whose ranges must not be read through stale shared
  // pointers.
  constexpr float radius = 1.0f;
  constexpr int kSets = 3;
  std::vector<apfrnn::support::PointCloud> clouds;
  std::vector<std::vector<float>> xyz;
  for (int set = 0; set < kSets; ++set) {
    clouds.push_back(apfrnn::support::make_point_cloud(2500, 13.6f, 77 + set));
    xyz.push_back(interleave(clouds.back()));
  }
  apfrnn::NSearch search(radius);
  for (int set = 0; set < kSets; ++set) {
    search.add_point_set(xyz[static_cast<std::size_t>(set)].data(),
                         static_cast<int>(clouds[0].x.size()));
  }
  search.set_all_searches(true);

  auto check_all = [&] {
    for (int query = 0; query < kSets; ++query) {
      for (int target = 0; target < kSets; ++target) {
        const auto &q = clouds[static_cast<std::size_t>(query)];
        const auto &t = clouds[static_cast<std::size_t>(target)];
        if (query == target) {
          check_same_set(search, query, q, radius);
        } else {
          check_cross_set(search, query, target, q, t, radius);
        }
      }
    }
  };

  search.run();
  CHECK(search.last_stats().dense_sets == kSets);
  CHECK((search.last_stats().paths &
         apfrnn::NSearch::kPathRangeCacheShared) != 0);
  check_all();

  for (int set = 0; set < kSets; ++set) {
    auto &cloud = clouds[static_cast<std::size_t>(set)];
    cloud.x[0] = cloud.y[0] = cloud.z[0] = 270.0f;
    // In place: a resize would rebuild the pair state from scratch.
    const std::vector<float> moved =
        interleave(clouds[static_cast<std::size_t>(set)]);
    std::copy(moved.begin(), moved.end(),
              xyz[static_cast<std::size_t>(set)].begin());
  }
  search.run();
  CHECK(search.last_stats().hash_sets == kSets);
  check_all();

  // And back onto the dense grid.
  for (int set = 0; set < kSets; ++set) {
    clouds[static_cast<std::size_t>(set)] =
        apfrnn::support::make_point_cloud(2500, 13.6f, 177 + set);
    const std::vector<float> moved =
        interleave(clouds[static_cast<std::size_t>(set)]);
    std::copy(moved.begin(), moved.end(),
              xyz[static_cast<std::size_t>(set)].begin());
  }
  search.run();
  CHECK(search.last_stats().dense_sets == kSets);
  check_all();
}

TEST_CASE("a sweep set converted to a grid mid-run keeps earlier pairs exact") {
  // The tight cluster plus one far point makes `sparse` a sweep set. Pair
  // (sparse, dense) is built first from the sweep, in the caller's order.
  // Pair (dense, sparse) then cannot fit its sweep rows into the small memory
  // budget and converts `sparse` to a grid, which re-sorts it; the earlier
  // pair (and the same-set pair) must not be read through the new order.
  constexpr float radius = 1.7f;
  apfrnn::support::PointCloud sparse;
  std::mt19937 generator(9);
  std::uniform_real_distribution<float> jitter(-0.01f, 0.01f);
  sparse.x.push_back(510.0f);
  sparse.y.push_back(510.0f);
  sparse.z.push_back(510.0f);
  for (int i = 0; i < 11; ++i) {
    sparse.x.push_back(0.95f + jitter(generator));
    sparse.y.push_back(0.01f + jitter(generator));
    sparse.z.push_back(-0.46f + jitter(generator));
  }
  auto dense = apfrnn::support::make_point_cloud(1500, 10.0f, 4);
  auto sparse_xyz = interleave(sparse);
  auto dense_xyz = interleave(dense);

  apfrnn::NSearch search(radius);
  search.set_memory_budget(1 << 16);
  const int s = search.add_point_set(sparse_xyz.data(),
                                     static_cast<int>(sparse.x.size()));
  const int d = search.add_point_set(dense_xyz.data(),
                                     static_cast<int>(dense.x.size()));
  search.set_all_searches(false);
  search.set_active_search(s, s, true);
  search.set_active_search(s, d, true);
  search.set_active_search(d, s, true);
  search.run();
  CHECK((search.last_stats().paths & apfrnn::NSearch::kPathSweep) != 0);

  check_same_set(search, s, sparse, radius);
  check_cross_set(search, s, d, sparse, dense, radius);
  check_cross_set(search, d, s, dense, sparse, radius);
}

TEST_CASE("cross rows with a skin stay exact after a same-set CSR fallback") {
  // The small budget sends the high-degree same-set pair to the exact-radius
  // CSR fallback, which lowers the set's search radius below the grid's
  // radius + skin cell size. The cross writers that skip the range build
  // (target-major, pair-dense) must still use the grid's cells and the
  // pair's radius.
  constexpr float radius = 0.6f;
  const float side = std::cbrt(1500.0f / 6.0f);
  for (uint32_t seed = 1; seed <= 3; ++seed) {
    for (int64_t budget : {int64_t(1) << 18, int64_t(1) << 20}) {
      CAPTURE(seed);
      CAPTURE(budget);
      auto fluid = apfrnn::bench::make_two_scale(1500, side, seed);
      auto probes = apfrnn::support::make_point_cloud(40, side, seed + 100);
      auto fluid_xyz = interleave(fluid);
      auto probe_xyz = interleave(probes);

      apfrnn::NSearch search(radius);
      search.set_skin(0.3f * radius);
      search.set_memory_budget(budget);
      const int f = search.add_point_set(fluid_xyz.data(),
                                         static_cast<int>(fluid.x.size()));
      const int p = search.add_point_set(probe_xyz.data(),
                                         static_cast<int>(probes.x.size()));
      search.set_all_searches(false);
      search.set_active_search(f, f, true);
      search.set_active_search(f, p, true);
      search.run();

      check_same_set(search, f, fluid, radius);
      check_cross_set(search, f, p, fluid, probes, radius);
    }
  }
}
