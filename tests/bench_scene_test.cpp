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

void run_scene(const SceneSpec &spec, int frames) {
  SceneData scene = spec.make();
  NSearch search(scene.radius);
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
