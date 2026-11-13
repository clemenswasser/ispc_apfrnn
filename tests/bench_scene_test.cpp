// Runs benchmark scenes for several moving frames and checks every active pair
// against the grid reference on every frame. Some code paths only trigger at
// benchmark sizes (millions of points, large sparse sets), so this covers them
// with the scenes that reach them.
//
//   APFRNN_BENCH_SCENE_TEST=name[,name...]  run these scenes instead
//   APFRNN_BENCH_SCENE_FRAMES=N             frames per scene (default 3)

#include "grid_reference.hpp"
#include "point_cloud.hpp"
#include "scenes.hpp"
#include "apfrnn/neighbor_search.hpp"

#include <doctest/doctest.h>

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace {

using apfrnn::NSearch;
using apfrnn::bench::SceneData;
using apfrnn::bench::SceneSpec;
using apfrnn::support::PointCloud;

PointCloud as_cloud(const std::vector<float> &xyz, int count) {
  PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(count));
  cloud.y.resize(static_cast<std::size_t>(count));
  cloud.z.resize(static_cast<std::size_t>(count));
  for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i) {
    cloud.x[i] = xyz[3 * i + 0];
    cloud.y[i] = xyz[3 * i + 1];
    cloud.z[i] = xyz[3 * i + 2];
  }
  return cloud;
}

std::vector<SceneSpec> all_scenes() {
  std::vector<SceneSpec> scenes = apfrnn::bench::synthetic_scenes();
  for (auto *family :
       {&apfrnn::bench::stress_scenes, &apfrnn::bench::scale_scenes}) {
    for (SceneSpec &scene : family()) {
      scenes.push_back(std::move(scene));
    }
  }
  return scenes;
}

std::vector<std::string> selected(const char *fallback) {
  const char *env = std::getenv("APFRNN_BENCH_SCENE_TEST");
  std::stringstream list(env != nullptr ? env : fallback);
  std::vector<std::string> names;
  for (std::string name; std::getline(list, name, ',');) {
    names.push_back(name);
  }
  return names;
}

void run_scene(const SceneSpec &spec, int frames, int64_t budget = 0) {
  SceneData scene = spec.make();
  // The benchmark moves every static scene with a small smooth flow. Replay
  // the same motion here so late-frame state (changed cells, multi-point
  // cells, row layouts) is exercised and checked.
  if (!scene.trace && !scene.growing &&
      scene.motion == apfrnn::bench::Motion::Static && scene.jitter == 0.0f &&
      !scene.xyz.empty() && !scene.xyz[0].empty()) {
    scene.flow = 0.02f;
    float low = scene.xyz[0][0];
    float high = low;
    for (std::size_t i = 0; i < scene.xyz[0].size(); i += 3) {
      low = std::min(low, scene.xyz[0][i]);
      high = std::max(high, scene.xyz[0][i]);
    }
    scene.flow_wavenumber =
        high > low ? 6.2831853f / (high - low) : 1.0f;
  }
  NSearch search(scene.radius);
  if (budget > 0) {
    search.set_memory_budget(budget);
  }
  const int sets = static_cast<int>(scene.xyz.size());
  for (int set = 0; set < sets; ++set) {
    search.add_point_set(scene.xyz[static_cast<std::size_t>(set)].data(),
                         scene.points(set),
                         scene.is_static[static_cast<std::size_t>(set)]);
  }
  search.set_all_searches(false);
  for (const auto &[query, target] : scene.pairs) {
    search.set_active_search(query, target, true);
  }
  for (int frame = 0; frame < frames; ++frame) {
    if (frame > 0) {
      scene.advance(frame);
      if (scene.growing) {
        search.resize_point_set(0, scene.xyz[0].data(), scene.points(0));
      }
      for (const int set : scene.resized) {
        search.resize_point_set(
            set, scene.xyz[static_cast<std::size_t>(set)].data(),
            scene.points(set));
      }
    }
    search.run();
    std::vector<PointCloud> clouds;
    for (int set = 0; set < sets; ++set) {
      clouds.push_back(as_cloud(scene.xyz[static_cast<std::size_t>(set)],
                                scene.points(set)));
    }
    for (const auto &[query, target] : scene.pairs) {
      INFO(spec.name << " frame " << frame << " pair " << query << ","
                     << target);
      CHECK(apfrnn::support::check_pair(
                search, query, target,
                clouds[static_cast<std::size_t>(query)],
                clouds[static_cast<std::size_t>(target)], scene.radius) == "");
    }
  }
}

} // namespace

TEST_CASE("large sparse benchmark scenes stay exact over moving frames") {
  const char *frames_env = std::getenv("APFRNN_BENCH_SCENE_FRAMES");
  const int frames = frames_env != nullptr ? std::atoi(frames_env) : 3;
  const std::vector<SceneSpec> scenes = all_scenes();
  for (const std::string &name : selected("huge_domain_2m")) {
    bool found = false;
    for (const SceneSpec &spec : scenes) {
      if (spec.name == name) {
        found = true;
        run_scene(spec, frames);
      }
    }
    INFO(name);
    CHECK(found);
  }
}

TEST_CASE("moving high-degree sets stay exact with a small row budget") {
  // Dense clusters make rows much larger than the budget. The per-point row
  // layout may exceed the budget by the exact output size instead of falling
  // back to CSR, and must stay exact across moving frames.
  SceneSpec spec{"dense_moving_budget", [] {
                   SceneData scene;
                   scene.radius = 2.0f;
                   scene.xyz = {apfrnn::bench::interleave(
                       apfrnn::bench::make_clustered(50000, 20.0f, 23))};
                   scene.is_static = {false};
                   scene.pairs = {{0, 0}};
                   return scene;
                 }};
  run_scene(spec, 3, int64_t{16} << 20);
}

TEST_CASE("growing high-degree sets stay exact") {
  // Appending points every frame resizes the set; the per-point row layout
  // keeps the existing capacities and seeds the new points.
  SceneSpec spec{"growing_dense", [] {
                   SceneData scene;
                   scene.radius = 2.0f;
                   scene.xyz = {apfrnn::bench::interleave(
                       apfrnn::bench::make_clustered(30000, 18.0f, 29))};
                   scene.is_static = {false};
                   scene.pairs = {{0, 0}};
                   scene.growing = true;
                   scene.grow_limit = 40000;
                   scene.grow_per_frame = 1000;
                   scene.grow_extent[0] = 18.0f;
                   scene.grow_extent[1] = 18.0f;
                   scene.grow_extent[2] = 18.0f;
                   return scene;
                 }};
  run_scene(spec, 6);
}

TEST_CASE("growing moderate-degree sets stay exact") {
  // A slab grows every frame, so last frame's counts never match the current
  // size; the per-point layout is chosen from them and seeds the new points.
  run_scene(apfrnn::bench::growing_scene(
                apfrnn::bench::synthetic_scene(
                    apfrnn::bench::Distribution::Slab, 20000),
                26000, 1000),
            6);
}

TEST_CASE("sparse moving sets with multi-point cells stay exact") {
  // A sparse cloud small enough that several cells hold two or three points.
  // Those cells take the direct neighborhood scan around the unique-cell
  // walk, and the flow keeps the cell membership changing.
  SceneSpec spec{"sparse_multi_cell", [] {
                   SceneData scene;
                   scene.radius = 2.0f;
                   scene.xyz = {apfrnn::bench::interleave(
                       apfrnn::support::make_point_cloud(200000, 1000.0f, 17))};
                   scene.is_static = {false};
                   scene.pairs = {{0, 0}};
                   return scene;
                 }};
  run_scene(spec, 6);
}

TEST_CASE("sparse moving sets spanning the origin stay exact") {
  // A sparse set (no dense lookup) whose cells have negative and positive
  // coordinates on every axis. Its writers find cells through the column
  // lookup, whose columns and z ranges cross zero.
  SceneSpec spec{"sparse_origin", [] {
                   SceneData scene;
                   scene.radius = 2.0f;
                   auto cloud = apfrnn::support::make_point_cloud(200000,
                                                                  1000.0f, 31);
                   for (std::size_t i = 0; i < cloud.x.size(); ++i) {
                     cloud.x[i] -= 500.0f;
                     cloud.y[i] -= 500.0f;
                     cloud.z[i] -= 500.0f;
                   }
                   scene.xyz = {apfrnn::bench::interleave(cloud)};
                   scene.is_static = {false};
                   scene.pairs = {{0, 0}};
                   return scene;
                 }};
  run_scene(spec, 4);
}
