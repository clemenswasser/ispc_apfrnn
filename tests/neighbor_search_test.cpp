#include "point_cloud.hpp"
#include "scene.hpp"
#include "apfrnn/neighbor_search.hpp"
#include "plan.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <tuple>
#include <numeric>
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

TEST_CASE("symmetric cross pairs transpose both directions") {
  constexpr float radius = 1.5f;

  auto a = apfrnn::support::make_point_cloud(640, 10.0f, 404);
  auto b = apfrnn::support::make_point_cloud(640, 10.0f, 405);
  auto c = apfrnn::support::make_point_cloud(640, 10.0f, 406);
  auto a_xyz = interleave(a);
  auto b_xyz = interleave(b);
  auto c_xyz = interleave(c);

  apfrnn::NSearch search(radius);
  const int a_id =
      search.add_point_set(a_xyz.data(), static_cast<int>(a.x.size()));
  const int b_id =
      search.add_point_set(b_xyz.data(), static_cast<int>(b.x.size()));
  const int c_id =
      search.add_point_set(c_xyz.data(), static_cast<int>(c.x.size()));
  search.set_all_searches(false);
  for (int query : {a_id, b_id, c_id}) {
    for (int target : {a_id, b_id, c_id}) {
      search.set_active_search(query, target, true);
    }
  }
  search.run();
  check_cross_set(search, a_id, b_id, a, b, radius);
  check_cross_set(search, b_id, a_id, b, a, radius);
  check_cross_set(search, a_id, c_id, a, c, radius);
  check_cross_set(search, c_id, a_id, c, a, radius);
  check_cross_set(search, b_id, c_id, b, c, radius);
  check_cross_set(search, c_id, b_id, c, b, radius);

  // The reverse lists are transposed from the forward ones. A second frame
  // with every set moved must rebuild both directions exactly.
  translate(a, 0.09f, -0.04f, 0.02f);
  translate(b, -0.05f, 0.07f, 0.03f);
  translate(c, 0.03f, 0.05f, -0.06f);
  auto a_moved = interleave(a);
  auto b_moved = interleave(b);
  auto c_moved = interleave(c);
  search.resize_point_set(a_id, a_moved.data(),
                          static_cast<int>(a.x.size()));
  search.resize_point_set(b_id, b_moved.data(),
                          static_cast<int>(b.x.size()));
  search.resize_point_set(c_id, c_moved.data(),
                          static_cast<int>(c.x.size()));
  search.run();
  check_cross_set(search, a_id, b_id, a, b, radius);
  check_cross_set(search, b_id, a_id, b, a, radius);
  check_cross_set(search, a_id, c_id, a, c, radius);
  check_cross_set(search, c_id, a_id, c, a, radius);
  check_cross_set(search, b_id, c_id, b, c, radius);
  check_cross_set(search, c_id, b_id, c, b, radius);
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

TEST_CASE("inline reads follow state changes between moving frames") {
  // get_neighbors reads through per-pair views built by run(). Every call that
  // changes the search state drops them, so reads between such a call and the
  // next run() must answer like the full lookup: out-of-range points and
  // inactive pairs give empty lists, and the next run() is exact again.
  constexpr float radius = 1.0f;
  auto fluid = apfrnn::support::make_point_cloud(3000, 14.0f, 91);
  auto boundary = apfrnn::support::make_point_cloud(800, 14.0f, 92);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  std::vector<float> fluid_xyz = interleave(fluid);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, fluid_id, true);
  search.set_active_search(fluid_id, boundary_id, true);

  for (int frame = 0; frame < 6; ++frame) {
    INFO("frame " << frame);
    translate(fluid, 0.03f, -0.02f, 0.01f);
    if (frame == 2) {
      // Shrink without a run: points past the new size have no rows.
      for (auto *axis : {&fluid.x, &fluid.y, &fluid.z}) {
        axis->resize(2000);
      }
      fluid_xyz = interleave(fluid);
      search.resize_point_set(fluid_id, fluid_xyz.data(),
                              static_cast<int>(fluid.x.size()));
      CHECK(search.get_neighbors(fluid_id, fluid_id, 2500).empty());
      CHECK(search.get_neighbors(fluid_id, boundary_id, 2500).empty());
    } else {
      fluid_xyz = interleave(fluid);
      search.resize_point_set(fluid_id, fluid_xyz.data(),
                              static_cast<int>(fluid.x.size()));
    }
    if (frame == 4) {
      search.set_active_search(fluid_id, boundary_id, false);
      CHECK(search.get_neighbors(fluid_id, boundary_id, 0).empty());
      search.set_active_search(fluid_id, boundary_id, true);
    }
    search.run();
    check_same_set(search, fluid_id, fluid, radius);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }
}

TEST_CASE("pair views match per-point reads across moving frames") {
  // neighbors(i, j)[point] must return the same row as get_neighbors for every
  // point and pair: after each run, in fixed-row and in CSR layouts (a tiny
  // memory budget forces CSR), and through the fallback after a state change.
  constexpr float radius = 1.0f;
  auto fluid = apfrnn::support::make_point_cloud(3000, 14.0f, 93);
  auto boundary = apfrnn::support::make_point_cloud(800, 14.0f, 94);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  std::vector<float> fluid_xyz = interleave(fluid);
  const int fluid_id =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(true);

  auto check_views = [&] {
    for (int query : {fluid_id, boundary_id}) {
      for (int target : {fluid_id, boundary_id}) {
        const apfrnn::PairNeighbors pair = search.neighbors(query, target);
        CHECK(pair.size() == search.get_n_points(query));
        for (int point = -1; point <= search.get_n_points(query); ++point) {
          const apfrnn::NeighborList row = pair[point];
          const apfrnn::NeighborList expected =
              search.get_neighbors(query, target, point);
          REQUIRE(row.size() == expected.size());
          CHECK(row.data == expected.data);
        }
      }
    }
  };

  for (int frame = 0; frame < 6; ++frame) {
    INFO("frame " << frame);
    translate(fluid, 0.03f, -0.02f, 0.01f);
    fluid_xyz = interleave(fluid);
    search.resize_point_set(fluid_id, fluid_xyz.data(),
                            static_cast<int>(fluid.x.size()));
    check_views(); // views dropped by the resize: the fallback answers
    if (frame == 3) {
      search.set_memory_budget(1);
    }
    search.run();
    check_views();
    check_same_set(search, fluid_id, fluid, radius);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }
}

TEST_CASE("zsort reorders the caller's points and stays exact") {
  // A caller that reorders its points with prepare_zsort()/apply_zsort()
  // every few frames: the positions and every other per-point array follow
  // the permutation, a static set is re-read, and the lists of the next
  // run() are exact for the new indices.
  constexpr float radius = 1.0f;
  auto fluid = apfrnn::support::make_point_cloud(4000, 14.0f, 95);
  auto boundary = apfrnn::support::make_point_cloud(900, 14.0f, 96);
  std::vector<float> fluid_xyz = interleave(fluid);
  std::vector<float> boundary_xyz = interleave(boundary);
  // The caller's own per-point data: which generated point each one is.
  std::vector<int> fluid_id(fluid.x.size());
  std::iota(fluid_id.begin(), fluid_id.end(), 0);
  const auto original = fluid;

  apfrnn::NSearch search(radius);
  const int fluid_set =
      search.add_point_set(fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_set = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_set, fluid_set, true);
  search.set_active_search(fluid_set, boundary_set, true);
  search.set_active_search(boundary_set, fluid_set, true);

  // Before any run() there is no grid order: nothing moves.
  search.prepare_zsort();
  search.apply_zsort(fluid_set, fluid_id.data());
  CHECK(fluid_id[17] == 17);

  auto sort_cloud = [&](int set, apfrnn::support::PointCloud &cloud,
                        std::vector<float> &xyz) {
    search.apply_zsort(set, xyz.data(), 3);
    search.apply_zsort(set, cloud.x.data());
    search.apply_zsort(set, cloud.y.data());
    search.apply_zsort(set, cloud.z.data());
  };
  float shift = 0.0f;
  for (int frame = 0; frame < 9; ++frame) {
    INFO("frame " << frame);
    translate(fluid, 0.04f, -0.03f, 0.02f);
    shift += 1.0f;
    fluid_xyz = interleave(fluid);
    search.resize_point_set(fluid_set, fluid_xyz.data(),
                            static_cast<int>(fluid.x.size()));
    search.run();
    check_same_set(search, fluid_set, fluid, radius);
    check_cross_set(search, fluid_set, boundary_set, fluid, boundary, radius);
    check_cross_set(search, boundary_set, fluid_set, boundary, fluid, radius);
    if (frame % 3 == 1) {
      search.prepare_zsort();
      sort_cloud(fluid_set, fluid, fluid_xyz);
      search.apply_zsort(fluid_set, fluid_id.data());
      sort_cloud(boundary_set, boundary, boundary_xyz);
      // The lists are void until the next run(), which must see the new
      // order of both sets.
      search.run();
      check_same_set(search, fluid_set, fluid, radius);
      check_cross_set(search, fluid_set, boundary_set, fluid, boundary,
                      radius);
      check_cross_set(search, boundary_set, fluid_set, boundary, fluid,
                      radius);
      // In grid order now: a second sort has nothing to move.
      const std::vector<int> before = fluid_id;
      search.prepare_zsort();
      search.apply_zsort(fluid_set, fluid_id.data());
      CHECK(fluid_id == before);
      // (the second prepare marked nothing; lists stay valid)
      check_same_set(search, fluid_set, fluid, radius);
    }
  }
  // The id array is still a permutation, and each point carries its data.
  std::vector<int> seen = fluid_id;
  std::sort(seen.begin(), seen.end());
  for (std::size_t i = 0; i < seen.size(); ++i) {
    REQUIRE(seen[i] == static_cast<int>(i));
  }
  for (std::size_t i = 0; i < fluid_id.size(); ++i) {
    const std::size_t id = static_cast<std::size_t>(fluid_id[i]);
    CHECK(fluid.x[i] == doctest::Approx(original.x[id] + 0.04f * shift));
  }
}

TEST_CASE("sparse sets across zero sort by box offsets and stay exact") {
  // Two dense blobs on either side of the origin: too sparse a box for the
  // counting sort, so the cell keys go through the radix sort, and their
  // coordinates span zero, which takes the box-relative packed keys. The
  // blobs drift for several frames so the cells keep changing.
  constexpr float radius = 1.0f;
  auto cloud = apfrnn::support::make_point_cloud(24000, 9.0f, 97);
  for (std::size_t i = 0; i < cloud.x.size(); ++i) {
    const float side = i % 2 == 0 ? -45.0f : 36.0f;
    cloud.x[i] += side;
    cloud.y[i] += side * 0.5f;
    cloud.z[i] -= side;
  }
  std::vector<float> xyz = interleave(cloud);
  apfrnn::NSearch search(radius);
  const int set =
      search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
  search.set_all_searches(false);
  search.set_active_search(set, set, true);
  for (int frame = 0; frame < 4; ++frame) {
    INFO("frame " << frame);
    translate(cloud, 0.37f, -0.21f, 0.29f);
    xyz = interleave(cloud);
    search.resize_point_set(set, xyz.data(), static_cast<int>(cloud.x.size()));
    search.run();
    check_same_set(search, set, cloud, radius);
  }
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
  // A sparse target now writes compact CSR rows (the fixed row layout touched
  // one fresh cache line per entry for the same data).
  CHECK(search.last_stats().cross_csr_pairs >= 1);
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
  verify_rows(values, counts, capacity + apfrnn::row_slack());

  apfrnn::write_neighbors_fixed_query_active2_parallel(data, capacity, values,
                                                       counts);
  verify_rows(values, counts, capacity + apfrnn::row_slack());
}

TEST_CASE("fixed rows placed by original point match brute force") {
  // With a placement, every writer puts point p's row in slot p once the row
  // buffer is reused, while counts stays in sorted order.
  constexpr int num_points = 400;
  constexpr float radius = 1.4f;
  constexpr int capacity = 128;

  auto cloud = apfrnn::support::make_point_cloud(num_points, 16.0f, 93);
  apfrnn::NeighborSearchData data;
  apfrnn::build_point_grid_inplace(data, cloud.x, cloud.y, cloud.z, radius);
  apfrnn::build_same_set_cell_ranges_inplace(data);

  apfrnn::IntBuffer values;
  apfrnn::IntBuffer counts;
  const apfrnn::RowPlacement placement{data.original_index.data()};
  const std::size_t stride =
      static_cast<std::size_t>(capacity + apfrnn::row_slack());

  auto verify_rows = [&] {
    REQUIRE(counts.size() == static_cast<std::size_t>(num_points));
    for (int point = 0; point < num_points; ++point) {
      const int sorted_index = apfrnn::support::sorted_index_for(data, point);
      const int count = counts[static_cast<std::size_t>(sorted_index)];
      REQUIRE(count <= capacity);
      const int *row = values.data() + static_cast<std::size_t>(point) * stride;
      std::vector<int> neighbors(row, row + count);
      std::sort(neighbors.begin(), neighbors.end());
      CHECK(neighbors ==
            apfrnn::support::exact_neighbors_for(cloud, point, radius));
    }
  };

  // A newly allocated buffer is written in query order: the placement is
  // ignored and the writer says so.
  CHECK_FALSE(apfrnn::write_neighbors_fixed_query_pair_parallel(
      data, capacity, values, counts, placement));
  for (int point = 0; point < num_points; ++point) {
    const int sorted_index = apfrnn::support::sorted_index_for(data, point);
    const int *row =
        values.data() + static_cast<std::size_t>(sorted_index) * stride;
    std::vector<int> neighbors(
        row, row + counts[static_cast<std::size_t>(sorted_index)]);
    std::sort(neighbors.begin(), neighbors.end());
    CHECK(neighbors ==
          apfrnn::support::exact_neighbors_for(cloud, point, radius));
  }

  // The buffer is reused from here on, so rows are placed.
  CHECK(apfrnn::write_neighbors_fixed_query_pair_parallel(
      data, capacity, values, counts, placement));
  verify_rows();
  CHECK(apfrnn::write_neighbors_fixed_query_active2_parallel(
      data, capacity, values, counts, placement));
  verify_rows();
  CHECK(apfrnn::write_neighbors_fixed_query_oct_parallel(
      data, capacity, values, counts, placement));
  verify_rows();
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
  verify_rows(values, counts, capacity + apfrnn::row_slack());

  apfrnn::write_cross_neighbors_fixed_query_active2_parallel(
      cross, target_data, capacity, values, counts);
  verify_rows(values, counts, capacity + apfrnn::row_slack());
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
  REQUIRE(values.size() == static_cast<std::size_t>(num_points) *
                               (capacity + apfrnn::row_slack()));
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
  REQUIRE(values.size() == static_cast<std::size_t>(num_points) *
                               (capacity + apfrnn::row_slack()));
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
}

TEST_CASE("cross block rows fit a budget the fixed rows cannot") {
  constexpr float radius = 1.0f;
  // The source has 2-4 points per cell and the target is denser, so the
  // on-the-fly dense pair writer is selected. Its global maximum degree needs
  // a fixed row table larger than the budget; the per-row table does not.
  auto fluid = apfrnn::support::make_point_cloud(4096, 11.7f, 9101);
  auto boundary = apfrnn::support::make_point_cloud(6000, 7.5f, 9102);
  // One dense target cluster makes a few cross rows much longer than the
  // typical row, which is exactly the padding the per-row table removes.
  std::mt19937 cluster_generator(9103);
  std::uniform_real_distribution<float> cluster_offset(-0.4f, 0.4f);
  for (int i = 0; i < 800; ++i) {
    boundary.x.push_back(3.0f + cluster_offset(cluster_generator));
    boundary.y.push_back(3.0f + cluster_offset(cluster_generator));
    boundary.z.push_back(3.0f + cluster_offset(cluster_generator));
  }
  auto fluid_xyz = interleave(fluid);
  auto boundary_xyz = interleave(boundary);

  apfrnn::NSearch search(radius);
  search.set_memory_budget(2 << 20);  // below the grown fixed rows
  const int fluid_id = search.add_point_set(
      fluid_xyz.data(), static_cast<int>(fluid.x.size()));
  const int boundary_id = search.add_point_set(
      boundary_xyz.data(), static_cast<int>(boundary.x.size()), true);
  search.set_all_searches(false);
  search.set_active_search(fluid_id, boundary_id, true);

  for (int frame = 0; frame < 3; ++frame) {
    search.run();
    const auto stats = search.last_stats();
    CHECK(stats.cross_fixed_pairs == 1);
    CHECK(stats.cross_csr_pairs == 0);
    CHECK(stats.neighbor_bytes < (2u << 20));
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }
}

TEST_CASE("packed same-set rows fit a budget the grown fixed rows cannot") {
  // A dense set with one tight cluster: a few rows are far longer than the
  // rest, so rows grown to the maximum exceed the budget, while packed rows
  // sized from the exact counts fit it.
  constexpr float radius = 1.0f;
  auto cloud = apfrnn::support::make_point_cloud(8000, 12.0f, 9201);
  std::mt19937 generator(9202);
  std::uniform_real_distribution<float> offset(-0.3f, 0.3f);
  for (int i = 0; i < 150; ++i) {
    cloud.x.push_back(4.0f + offset(generator));
    cloud.y.push_back(4.0f + offset(generator));
    cloud.z.push_back(4.0f + offset(generator));
  }
  auto xyz = interleave(cloud);
  apfrnn::NSearch search(radius);
  search.set_memory_budget(3 << 20);
  const int id =
      search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
  search.set_all_searches(true);
  // The caller adopts the grid order after the first frame, so the later
  // frames take the dense writer's packed rows.
  for (int frame = 0; frame < 4; ++frame) {
    INFO("frame " << frame);
    if (frame == 1) {
      search.prepare_zsort();
      search.apply_zsort(id, cloud.x.data());
      search.apply_zsort(id, cloud.y.data());
      search.apply_zsort(id, cloud.z.data());
    }
    translate(cloud, 0.05f, 0.02f, -0.03f);
    const auto moved = interleave(cloud);
    std::copy(moved.begin(), moved.end(), xyz.begin());
    search.run();
    const auto stats = search.last_stats();
    CAPTURE(stats.paths);
    if (frame >= 2) {
      CHECK((stats.paths & apfrnn::NSearch::kPathSamePairDense) != 0);
    }
    CHECK(stats.neighbor_bytes < (3u << 20));
    check_same_set(search, id, cloud, radius);
  }
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

TEST_CASE("cross rows stay exact after a same-set CSR fallback") {
  // The small budget sends the high-degree same-set pair to the CSR
  // fallback. The cross writers that skip the range build (target-major,
  // pair-dense) must still use the grid's cells and the pair's radius.
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

TEST_CASE("sparse bounding boxes crossing zero keep z runs contiguous") {
  // One far point makes the bounding box too large for the counting sort, so
  // cells are ordered by the comparison sort. The cloud crosses z = 0, where
  // a raw packed-key order would put the z = -1 cells after the positive
  // ones and break the dense stencils' contiguous z runs (stack overflow of
  // the 9 runs, duplicate neighbors).
  for (const float far_x : {50.0f, 300.0f}) {
    CAPTURE(far_x);
    auto cloud = apfrnn::support::make_point_cloud(600, 6.0f, 21);
    for (float &z : cloud.z) {
      z -= 1.0f;
    }
    cloud.x.push_back(far_x);
    cloud.y.push_back(1.0f);
    cloud.z.push_back(1.0f);
    auto xyz = interleave(cloud);

    apfrnn::NSearch search(1.7f);
    const int id =
        search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
    search.set_all_searches(true);
    search.run();
    CHECK(search.last_stats().dense_sets == 1);
    check_same_set(search, id, cloud, 1.7f);
  }
}

TEST_CASE("block rows with cached capacities rebuild the grid's ranges") {
  // Frame 1 selects block rows and caches their capacities; frame 2 rebuilds
  // the grid (smaller radius, points moved) and reuses the capacities, so
  // the cell writer must build this grid's ranges instead of reading the
  // cleared ones from frame 1.
  int block_frames = 0;
  // Clustered rows have very different lengths, so per-block capacities fit
  // budgets that full fixed rows exceed.
  for (int shift = 16; shift <= 26; ++shift) {
    CAPTURE(shift);
    auto cloud = apfrnn::bench::make_clustered(3000, 14.0f, 17);
    auto xyz = interleave(cloud);
    apfrnn::NSearch search(1.7f);
    search.set_memory_budget(int64_t(1) << shift);
    const int id =
        search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
    search.set_all_searches(true);
    search.run();
    check_same_set(search, id, cloud, 1.7f);

    std::mt19937 generator(5);
    std::uniform_real_distribution<float> jitter(-0.6f, 0.6f);
    for (std::size_t i = 0; i < cloud.x.size(); ++i) {
      cloud.x[i] += jitter(generator);
      cloud.y[i] += jitter(generator);
      cloud.z[i] += jitter(generator);
    }
    const auto moved = interleave(cloud);
    std::copy(moved.begin(), moved.end(), xyz.begin());
    search.set_radius(1.0f);
    search.run();
    if ((search.last_stats().paths & apfrnn::NSearch::kPathSameBlockRows) !=
        0) {
      ++block_frames;
    }
    check_same_set(search, id, cloud, 1.0f);
  }
  CHECK(block_frames > 0);
}

TEST_CASE("block rows stay exact through motion and back to rest") {
  // Clustered rows have very different lengths. Frame 1 counts exactly and
  // writes block rows directly (no padded fixed rows that overflow and
  // regrow). Moving the points rebuilds the grid and re-sorts the rows:
  // frame 2 switches to per-point capacities (one count pass), later moving
  // frames reuse them, and large steps make rows overflow and be rewritten.
  // A stationary frame returns to the static block rows.
  auto cloud = apfrnn::bench::make_clustered(20000, 30.0f, 23);
  auto xyz = interleave(cloud);
  apfrnn::NSearch search(1.7f);
  const int id =
      search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
  search.set_all_searches(true);
  search.run();
  uint64_t paths = search.last_stats().paths;
  CHECK((paths & apfrnn::NSearch::kPathSameBlockRows) != 0);
  CHECK((paths & apfrnn::NSearch::kPathSameGrown) == 0);
  check_same_set(search, id, cloud, 1.7f);

  std::mt19937 generator(9);
  for (const float step : {0.3f, 0.05f, 0.8f, 0.05f}) {
    CAPTURE(step);
    std::uniform_real_distribution<float> jitter(-step, step);
    for (std::size_t i = 0; i < cloud.x.size(); ++i) {
      cloud.x[i] += jitter(generator);
      cloud.y[i] += jitter(generator);
      cloud.z[i] += jitter(generator);
    }
    const auto moved = interleave(cloud);
    std::copy(moved.begin(), moved.end(), xyz.begin());
    search.run();
    paths = search.last_stats().paths;
    CHECK((paths & apfrnn::NSearch::kPathSameBlockRows) != 0);
    check_same_set(search, id, cloud, 1.7f);
  }

  search.run();
  paths = search.last_stats().paths;
  CHECK((paths & apfrnn::NSearch::kPathSameBlockRows) != 0);
  check_same_set(search, id, cloud, 1.7f);
}

TEST_CASE("packed dense rows stay exact through moving frames") {
  // A dense moving set takes the packed same-set writer. The points first
  // jitter, then contract so rows outgrow the fitted capacity and are
  // rewritten at a grown one, then hold still; every frame is checked
  // through both read paths.
  constexpr float radius = 1.0f;
  auto cloud = apfrnn::support::make_point_cloud(20000, 14.0f, 41);
  auto xyz = interleave(cloud);
  apfrnn::NSearch search(radius);
  const int id =
      search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
  search.set_all_searches(true);
  std::mt19937 generator(5);
  bool grown = false;
  for (int frame = 0; frame < 7; ++frame) {
    INFO("frame " << frame);
    if (frame >= 1 && frame <= 3) {
      std::uniform_real_distribution<float> jitter(-0.2f, 0.2f);
      for (std::size_t i = 0; i < cloud.x.size(); ++i) {
        cloud.x[i] += jitter(generator);
        cloud.y[i] += jitter(generator);
        cloud.z[i] += jitter(generator);
      }
    } else if (frame == 4) {
      for (std::size_t i = 0; i < cloud.x.size(); ++i) {
        cloud.x[i] *= 0.8f;
        cloud.y[i] *= 0.8f;
        cloud.z[i] *= 0.8f;
      }
    }
    const auto moved = interleave(cloud);
    std::copy(moved.begin(), moved.end(), xyz.begin());
    search.run();
    const uint64_t paths = search.last_stats().paths;
    CHECK((paths & apfrnn::NSearch::kPathSamePairDense) != 0);
    grown = grown || (paths & apfrnn::NSearch::kPathSameGrown) != 0;
    check_same_set(search, id, cloud, radius);
    const apfrnn::PairNeighbors view = search.neighbors(id, id);
    for (int point = 0; point < view.size(); point += 97) {
      const apfrnn::NeighborList row = view[point];
      std::vector<int> neighbors(row.begin(), row.end());
      std::sort(neighbors.begin(), neighbors.end());
      CHECK(neighbors ==
            apfrnn::support::exact_neighbors_for(cloud, point, radius));
    }
  }
  CHECK(grown);
}

TEST_CASE("packed rows replace block rows once the caller follows the grid") {
  // A dense block with sparse spray around it, like a fluid with a free
  // surface: rows have very different lengths, which makes a moving set in
  // generation order prefer per-point rows. After the caller adopts the
  // grid order with zsort, the dense writer's packed rows take over; every
  // frame stays exact.
  constexpr float radius = 1.0f;
  auto cloud = apfrnn::support::make_point_cloud(24000, 14.0f, 31);
  const auto spray = apfrnn::support::make_point_cloud(10000, 22.0f, 32);
  cloud.x.insert(cloud.x.end(), spray.x.begin(), spray.x.end());
  cloud.y.insert(cloud.y.end(), spray.y.begin(), spray.y.end());
  cloud.z.insert(cloud.z.end(), spray.z.begin(), spray.z.end());
  std::vector<float> xyz = interleave(cloud);
  apfrnn::NSearch search(radius);
  const int id =
      search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
  search.set_all_searches(true);
  std::mt19937 generator(13);
  std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);
  uint64_t before_sort = 0;
  uint64_t after_sort = 0;
  for (int frame = 0; frame < 8; ++frame) {
    INFO("frame " << frame);
    if (frame == 4) {
      search.prepare_zsort();
      search.apply_zsort(id, cloud.x.data());
      search.apply_zsort(id, cloud.y.data());
      search.apply_zsort(id, cloud.z.data());
    }
    for (std::size_t i = 0; i < cloud.x.size(); ++i) {
      cloud.x[i] += jitter(generator);
      cloud.y[i] += jitter(generator);
      cloud.z[i] += jitter(generator);
    }
    xyz = interleave(cloud);
    search.resize_point_set(id, xyz.data(), static_cast<int>(cloud.x.size()));
    search.run();
    (frame < 4 ? before_sort : after_sort) |= search.last_stats().paths;
    check_same_set(search, id, cloud, radius);
  }
  CHECK((before_sort & apfrnn::NSearch::kPathSameBlockRows) != 0);
  CHECK((after_sort & apfrnn::NSearch::kPathSamePairDense) != 0);
  CHECK((after_sort & apfrnn::NSearch::kPathSameBlockRows) == 0);
}

TEST_CASE("dense rows switch between in-place and packed tables") {
  // Tables within the last-level cache are written in place at full stride,
  // larger ones are packed; a set and a cross pair alternate between the
  // two as the limit changes, and every frame stays exact.
  constexpr float radius = 1.0f;
  auto fluid = apfrnn::support::make_point_cloud(12000, 11.0f, 51);
  auto boundary = apfrnn::support::make_point_cloud(5000, 11.0f, 52);
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
  int64_t &limit = apfrnn::plan::packed_rows_min_bytes();
  const int64_t saved = limit;
  translate(fluid, 0.0f, 0.0f, 0.0f);
  for (int frame = 0; frame < 6; ++frame) {
    INFO("frame " << frame);
    limit = frame % 2 == 0 ? INT64_MAX : 0;
    translate(fluid, 0.11f, -0.07f, 0.05f);
    fluid_xyz = interleave(fluid);
    search.resize_point_set(fluid_id, fluid_xyz.data(),
                            static_cast<int>(fluid.x.size()));
    search.run();
    const uint64_t paths = search.last_stats().paths;
    CHECK((paths & apfrnn::NSearch::kPathSamePairDense) != 0);
    check_same_set(search, fluid_id, fluid, radius);
    check_cross_set(search, fluid_id, boundary_id, fluid, boundary, radius);
  }
  limit = saved;
}

TEST_CASE("packed dense writer truncates rows above the capacity") {
  // Rows above the capacity keep their full count and their first
  // `capacity` entries, and no two rows overlap in the output.
  constexpr float radius = 1.0f;
  constexpr int capacity = 8;
  auto cloud = apfrnn::support::make_point_cloud(3000, 6.0f, 17);
  apfrnn::NeighborSearchData data;
  apfrnn::build_point_grid_inplace(data, cloud.x, cloud.y, cloud.z, radius);
  REQUIRE(data.dense_valid);
  apfrnn::IntBuffer values;
  apfrnn::IntBuffer counts;
  apfrnn::IntBuffer row_offset;
  REQUIRE(apfrnn::write_neighbors_packed_dense_parallel(data, capacity, values,
                                                        counts, row_offset));
  std::vector<std::pair<int, int>> spans;
  bool truncated = false;
  for (int point = 0; point < static_cast<int>(cloud.x.size()); ++point) {
    const int sorted_index = apfrnn::support::sorted_index_for(data, point);
    const auto exact = apfrnn::support::exact_neighbors_for(cloud, point, radius);
    const int count = counts[static_cast<std::size_t>(sorted_index)];
    CHECK(count == static_cast<int>(exact.size()));
    const int stored = std::min(count, capacity);
    truncated = truncated || count > capacity;
    const int start = row_offset[static_cast<std::size_t>(sorted_index)];
    spans.emplace_back(start, start + stored);
    for (int k = 0; k < stored; ++k) {
      CHECK(std::binary_search(exact.begin(), exact.end(),
                               values[static_cast<std::size_t>(start + k)]));
    }
  }
  CHECK(truncated);
  std::sort(spans.begin(), spans.end());
  for (std::size_t i = 1; i < spans.size(); ++i) {
    CHECK(spans[i - 1].second <= spans[i].first);
  }
}

TEST_CASE("budget-forced per-point rows stay exact through moving frames") {
  // Above the fixed-row point cap with a budget that only fits per-point
  // rows, the moving block layout must keep its per-point capacities across
  // frames and stay exact.
  constexpr int kPoints = 4050000;
  constexpr float kRadius = 2.0f;
  auto cloud = apfrnn::support::make_point_cloud(kPoints, 159.4f, 31);
  auto xyz = interleave(cloud);
  apfrnn::NSearch search(kRadius);
  search.set_memory_budget(int64_t(3) << 29);
  const int id =
      search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
  search.set_all_searches(true);

  const auto compare = [&](int frame) {
    CAPTURE(frame);
    CHECK((search.last_stats().paths & apfrnn::NSearch::kPathGridDense) != 0);
    CHECK((search.last_stats().paths & apfrnn::NSearch::kPathSameBlockRows) !=
          0);
    CHECK((search.last_stats().paths & apfrnn::NSearch::kPathSameBlocked) == 0);
    for (const int point : {0, kPoints / 2, kPoints - 1}) {
      CAPTURE(point);
      CHECK(sorted_neighbors(search.get_neighbors(id, id, point)) ==
            apfrnn::support::exact_neighbors_for(cloud, point, kRadius));
    }
  };
  search.run();
  compare(0);

  std::mt19937 generator(7);
  std::uniform_real_distribution<float> jitter(-0.1f, 0.1f);
  for (int frame = 1; frame <= 2; ++frame) {
    for (std::size_t i = 0; i < cloud.x.size(); ++i) {
      cloud.x[i] += jitter(generator);
      cloud.y[i] += jitter(generator);
      cloud.z[i] += jitter(generator);
    }
    const auto moved = interleave(cloud);
    std::copy(moved.begin(), moved.end(), xyz.begin());
    search.run();
    compare(frame);
  }
}

TEST_CASE("sparse same-set rows stay exact through moving frames") {
  // Sparse rows are written by original point so the read side can skip the
  // sorted mapping. Every moving frame rebuilds the grid and changes that
  // mapping, so the row index and the reported ids must stay aligned.
  constexpr int kPoints = 5000;
  constexpr float kRadius = 78.0f;
  auto cloud = apfrnn::support::make_point_cloud(kPoints, 2000.0f, 41);
  auto xyz = interleave(cloud);
  apfrnn::NSearch search(kRadius);
  const int id =
      search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
  search.set_all_searches(true);

  bool saw_neighbors = false;
  const auto compare = [&](int frame) {
    CAPTURE(frame);
    CHECK((search.last_stats().paths & apfrnn::NSearch::kPathSameSparseKeys) !=
          0);
    for (int point = 0; point < kPoints; point += 37) {
      CAPTURE(point);
      const apfrnn::NeighborList list = search.get_neighbors(id, id, point);
      saw_neighbors = saw_neighbors || !list.empty();
      CHECK(sorted_neighbors(list) ==
            apfrnn::support::exact_neighbors_for(cloud, point, kRadius));
    }
  };
  search.run();
  compare(0);

  std::mt19937 generator(13);
  std::uniform_real_distribution<float> jitter(-20.0f, 20.0f);
  for (int frame = 1; frame <= 3; ++frame) {
    for (std::size_t i = 0; i < cloud.x.size(); ++i) {
      cloud.x[i] += jitter(generator);
      cloud.y[i] += jitter(generator);
      cloud.z[i] += jitter(generator);
    }
    const auto moved = interleave(cloud);
    std::copy(moved.begin(), moved.end(), xyz.begin());
    search.run();
    compare(frame);
  }
  CHECK(saw_neighbors);
}

TEST_CASE("dense cross rows stay exact when candidates exceed the copy") {
  // A target blob of more points than the dense writer copies per cell sits
  // inside one cell, so source cells next to it read their runs in place
  // while the rest use the copied candidates.
  constexpr int kSourcePoints = 60000;
  constexpr float kRadius = 1.7f;
  auto source = apfrnn::support::make_point_cloud(kSourcePoints, 30.0f, 81);
  auto target = apfrnn::support::make_point_cloud(60000, 30.0f, 82);
  std::mt19937 generator(83);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  for (int k = 0; k < 1500;) {
    const float x = unit(generator);
    const float y = unit(generator);
    const float z = unit(generator);
    if (x * x + y * y + z * z > 1.0f) {
      continue;
    }
    target.x.push_back(9.35f + 0.6f * x);
    target.y.push_back(9.35f + 0.6f * y);
    target.z.push_back(9.35f + 0.6f * z);
    ++k;
  }
  auto source_xyz = interleave(source);
  auto target_xyz = interleave(target);
  apfrnn::NSearch search(kRadius);
  const int source_id =
      search.add_point_set(source_xyz.data(), kSourcePoints);
  const int target_id =
      search.add_point_set(target_xyz.data(), static_cast<int>(target.x.size()),
                           true);
  search.set_all_searches(false);
  search.set_active_search(source_id, target_id, true);
  std::mt19937 jitter_generator(9);
  std::uniform_real_distribution<float> jitter(-0.2f, 0.2f);
  for (int frame = 0; frame < 2; ++frame) {
    CAPTURE(frame);
    if (frame > 0) {
      for (std::size_t i = 0; i < source.x.size(); ++i) {
        source.x[i] += jitter(jitter_generator);
        source.y[i] += jitter(jitter_generator);
        source.z[i] += jitter(jitter_generator);
      }
      source_xyz = interleave(source);
      search.resize_point_set(source_id, source_xyz.data(), kSourcePoints);
    }
    search.run();
    CAPTURE(search.last_stats().paths);
    CHECK((search.last_stats().paths & apfrnn::NSearch::kPathCrossPairDense) !=
          0);
    for (int point = 0; point < kSourcePoints; ++point) {
      const float dx = source.x[point] - 9.35f;
      const float dy = source.y[point] - 9.35f;
      const float dz = source.z[point] - 9.35f;
      if (point % 97 != 0 && dx * dx + dy * dy + dz * dz > 9.0f) {
        continue;
      }
      CAPTURE(point);
      CHECK(sorted_neighbors(search.get_neighbors(source_id, target_id,
                                                  point)) ==
            apfrnn::support::exact_cross_neighbors_for(source, point, target,
                                                       kRadius));
    }
  }
}

TEST_CASE("packed cross rows stay exact across moving frames") {
  // Most source points see none of the dense target blob; a few see many.
  // The dense cross writer packs these very uneven rows, and they must stay
  // exact through both read paths while the source moves.
  constexpr int kSourcePoints = 132000;
  constexpr float kRadius = 1.7f;
  auto source = apfrnn::support::make_point_cloud(kSourcePoints, 46.0f, 71);
  auto target = apfrnn::support::make_point_cloud(1700, 16.0f, 72);
  std::mt19937 generator(73);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  for (int k = 0; k < 20;) {
    const float x = unit(generator);
    const float y = unit(generator);
    const float z = unit(generator);
    if (x * x + y * y + z * z > 1.0f) {
      continue;
    }
    target.x.push_back(9.0f + 1.2f * x);
    target.y.push_back(9.0f + 1.2f * y);
    target.z.push_back(9.0f + 1.2f * z);
    ++k;
  }
  auto source_xyz = interleave(source);
  auto target_xyz = interleave(target);
  apfrnn::NSearch search(kRadius);
  const int source_id =
      search.add_point_set(source_xyz.data(), kSourcePoints);
  const int target_id =
      search.add_point_set(target_xyz.data(), static_cast<int>(target.x.size()),
                           true);
  search.set_all_searches(false);
  search.set_active_search(source_id, target_id, true);

  std::mt19937 jitter_generator(7);
  std::uniform_real_distribution<float> jitter(-0.2f, 0.2f);
  for (int frame = 0; frame < 3; ++frame) {
    CAPTURE(frame);
    if (frame > 0) {
      for (std::size_t i = 0; i < source.x.size(); ++i) {
        source.x[i] += jitter(jitter_generator);
        source.y[i] += jitter(jitter_generator);
        source.z[i] += jitter(jitter_generator);
      }
      source_xyz = interleave(source);
      search.resize_point_set(source_id, source_xyz.data(), kSourcePoints);
    }
    search.run();
    CAPTURE(search.last_stats().paths);
    CHECK((search.last_stats().paths & apfrnn::NSearch::kPathCrossPairDense) !=
          0);
    const apfrnn::PairNeighbors view = search.neighbors(source_id, target_id);
    for (int point = 0; point < kSourcePoints; point += kSourcePoints / 40) {
      CAPTURE(point);
      const auto exact = apfrnn::support::exact_cross_neighbors_for(
          source, point, target, kRadius);
      CHECK(sorted_neighbors(search.get_neighbors(source_id, target_id, point)) ==
            exact);
      CHECK(sorted_neighbors(view[point]) == exact);
    }
  }
}

TEST_CASE("hash-grid stencil ranges are exact across the zero cell layer") {
  // Cell keys store z as an unsigned 21-bit field. The radix sort (16384+
  // points) orders raw keys, so a column's z = -1 cell sorts after its
  // z >= 0 cells, and the stencil runs must not chain them through the
  // sorted order. Dense clusters straddle x = y = z = 0 in a domain far too
  // large for the dense lookup. The packed writer merges the runs from the
  // hash table.
  apfrnn::support::PointCloud cloud;
  std::mt19937 generator(41);
  std::uniform_real_distribution<float> local(-3.0f, 3.0f);
  const float centers[][3] = {{0.0f, 0.0f, 0.0f},
                              {5000.0f, -4000.0f, 0.2f},
                              {-6000.0f, 3000.0f, -0.4f},
                              {0.5f, 7000.0f, 0.9f}};
  for (const auto &c : centers) {
    for (int i = 0; i < 5000; ++i) {
      cloud.x.push_back(c[0] + local(generator));
      cloud.y.push_back(c[1] + local(generator));
      cloud.z.push_back(c[2] + local(generator));
    }
  }
  const auto xyz = interleave(cloud);
  apfrnn::NSearch search(1.0f);
  const int id =
      search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
  search.set_all_searches(true);
  search.run();
  const uint64_t paths = search.last_stats().paths;
  CHECK((paths & apfrnn::NSearch::kPathGridHash) != 0);
  CHECK((paths & apfrnn::NSearch::kPathSamePairDense) != 0);
  check_same_set(search, id, cloud, 1.0f);
}

TEST_CASE("a set that switches from block rows to the sweep reads sweep rows") {
  // Frame 1 writes block rows (per-block row offsets); frame 2 moves one
  // point far enough for the sweep backend, whose rows use the plain fixed
  // layout and must not be read through the old block offsets.
  int block_frames = 0;
  for (int shift = 16; shift <= 26; ++shift) {
    CAPTURE(shift);
    auto cloud = apfrnn::bench::make_clustered(3000, 14.0f, 17);
    auto xyz = interleave(cloud);
    apfrnn::NSearch search(1.7f);
    search.set_memory_budget(int64_t(1) << shift);
    const int id =
        search.add_point_set(xyz.data(), static_cast<int>(cloud.x.size()));
    search.set_all_searches(true);
    search.run();
    if ((search.last_stats().paths & apfrnn::NSearch::kPathSameBlockRows) ==
        0) {
      continue;
    }
    ++block_frames;

    // The sweep's fixed rows need the default budget.
    search.set_memory_budget(int64_t(3) << 30);
    apfrnn::support::set_extreme_outlier(cloud, 0);
    const auto moved = interleave(cloud);
    std::copy(moved.begin(), moved.end(), xyz.begin());
    search.run();
    CHECK((search.last_stats().paths & apfrnn::NSearch::kPathSameSweep) != 0);
    check_same_set(search, id, cloud, 1.7f);
  }
  CHECK(block_frames > 0);
}
