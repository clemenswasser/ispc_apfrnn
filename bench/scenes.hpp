#pragma once

// Benchmark scenes: point sets, the active search pairs, and how the scene
// moves between frames. Backends only see SceneData.

#include "scene.hpp"
#include "vtk_points.hpp"

#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace apfrnn::bench {

struct SceneData {
  float radius = 0.0f;
  // Interleaved xyz per point set. Backends keep pointers into these buffers,
  // which are moved in place between frames.
  std::vector<std::vector<float>> xyz;
  std::vector<bool> is_static;
  // Active (query set, target set) searches.
  std::vector<std::pair<int, int>> pairs;
  // Set 0 (the fluid) moves by motion_offset() every frame.
  Motion motion = Motion::Static;

  int points(int set) const {
    return static_cast<int>(xyz[static_cast<std::size_t>(set)].size() / 3);
  }

  void advance(int step) {
    if (motion == Motion::Static) {
      return;
    }
    const Offset offset = motion_offset(motion, step, radius);
    std::vector<float> &fluid = xyz[0];
    for (std::size_t i = 0; i + 2 < fluid.size(); i += 3) {
      fluid[i + 0] += offset.dx;
      fluid[i + 1] += offset.dy;
      fluid[i + 2] += offset.dz;
    }
  }
};

struct SceneSpec {
  std::string name;
  std::function<SceneData()> make;
};

inline const char *distribution_name(Distribution distribution) {
  switch (distribution) {
  case Distribution::Uniform: return "uniform";
  case Distribution::JitteredLattice: return "jittered";
  case Distribution::Clustered: return "clustered";
  case Distribution::Slab: return "slab";
  case Distribution::Lattice: return "lattice";
  case Distribution::Pathological: return "pathological";
  case Distribution::HugeDomain: return "huge_domain";
  case Distribution::Surface: return "surface";
  case Distribution::ThinPipe: return "thin_pipe";
  case Distribution::TwoScale: return "two_scale";
  case Distribution::SlabThin: return "thin_slab";
  }
  return "unknown";
}

// A synthetic fluid with a static boundary and a small dynamic emitter; the
// fluid is searched against all three.
inline SceneSpec synthetic_scene(Distribution distribution, int points,
                                 Motion motion = Motion::Static) {
  std::string name = distribution_name(distribution);
  name += "_" + (points % 1000000 == 0 ? std::to_string(points / 1000000) + "m"
                                       : std::to_string(points / 1000) + "k");
  if (motion == Motion::SmallMove) {
    name += "_moving";
  } else if (motion == Motion::LargeMove) {
    name += "_fast";
  }
  return {name, [=] {
            Scene scene = make_scene(distribution, points);
            SceneData data;
            data.radius = scene.radius;
            data.xyz = {std::move(scene.fluid_xyz), std::move(scene.boundary_xyz),
                        std::move(scene.emitter_xyz)};
            data.is_static = {false, true, false};
            data.pairs = {{0, 0}, {0, 1}, {0, 2}};
            data.motion = motion;
            return data;
          }};
}

// A saved learnsph frame: fluid and static boundary, searched fluid-fluid and
// fluid-boundary with the learnsph support radius.
inline SceneSpec vtk_scene(const std::string &name, const std::string &fluid,
                           const std::string &boundary) {
  return {name, [=] {
            SceneData data;
            data.radius = 0.048f;
            data.xyz = {read_vtk_points(fluid).xyz,
                        read_vtk_points(boundary).xyz};
            data.is_static = {false, true};
            data.pairs = {{0, 0}, {0, 1}};
            return data;
          }};
}

// Uniformly distributed point sets with every pair active; tests per-pair
// overhead and grid reuse across many sets.
inline SceneSpec many_set_scene(const std::string &name, int sets,
                                int points_per_set) {
  return {name, [=] {
            SceneData data;
            data.radius = 2.0f;
            const float side =
                std::cbrt(static_cast<float>(sets * points_per_set));
            for (int set = 0; set < sets; ++set) {
              data.xyz.push_back(interleave(apfrnn::support::make_point_cloud(
                  points_per_set, side, 1001 + 17 * set)));
            }
            data.is_static.assign(static_cast<std::size_t>(sets), false);
            for (int query = 0; query < sets; ++query) {
              for (int target = 0; target < sets; ++target) {
                data.pairs.emplace_back(query, target);
              }
            }
            return data;
          }};
}

inline std::vector<SceneSpec> synthetic_scenes() {
  using D = Distribution;
  std::vector<SceneSpec> scenes;
  for (int n : {1000, 10000}) {
    scenes.push_back(synthetic_scene(D::Uniform, n));
  }
  scenes.push_back(synthetic_scene(D::Clustered, 10000));
  for (int n : {50000, 200000}) {
    for (D d : {D::Uniform, D::JitteredLattice, D::Lattice, D::Clustered,
                D::Slab, D::SlabThin, D::HugeDomain, D::Surface,
                D::ThinPipe, D::TwoScale}) {
      scenes.push_back(synthetic_scene(d, n));
    }
    scenes.push_back(synthetic_scene(D::Uniform, n, Motion::SmallMove));
    scenes.push_back(synthetic_scene(D::Slab, n, Motion::SmallMove));
  }
  scenes.push_back(synthetic_scene(D::Uniform, 1000000));
  scenes.push_back(synthetic_scene(D::Pathological, 10000));
  return scenes;
}

// Regimes outside the headline matrix: large high-degree sets where the
// default memory budget forces CSR, larger uniform sets, and many point sets
// with every pair active. Kept separate so a normal run stays focused.
inline std::vector<SceneSpec> stress_scenes() {
  using D = Distribution;
  std::vector<SceneSpec> scenes;
  scenes.push_back(synthetic_scene(D::Clustered, 500000));
  scenes.push_back(synthetic_scene(D::Clustered, 1000000));
  scenes.push_back(synthetic_scene(D::TwoScale, 1000000));
  scenes.push_back(synthetic_scene(D::Uniform, 2000000));
  scenes.push_back(many_set_scene("many_sets_20k", 8, 2500));
  scenes.push_back(many_set_scene("many_sets_100k", 8, 12500));
  scenes.push_back(many_set_scene("many_sets_16x1250", 16, 1250));
  return scenes;
}

inline bool file_exists(const std::string &path) {
  std::FILE *file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  std::fclose(file);
  return true;
}

// The saved learnsph frames under `res` (learnsph/res) that exist. The
// *_nsearch frames are written by learnsph's capture_nsearch_frames test.
inline std::vector<SceneSpec> learnsph_scenes(const std::string &res) {
  struct Frame {
    const char *name, *fluid, *boundary;
  };
  const Frame frames[] = {
      {"ramp_split_lv", "ramp_split_lv/fluid_35.vtk",
       "ramp_split_lv/fluid_boundary.vtk"},
      {"ramp_split_hv", "ramp_split_hv/fluid_46.vtk",
       "ramp_split_hv/fluid_boundary.vtk"},
      {"wall_01", "wall_01/fluid_2.vtk", "wall_01/fluid_boundary.vtk"},
      {"funnel", "nsearch_frames/funnel_nsearch/fluid.vtk",
       "nsearch_frames/funnel_nsearch/boundary.vtk"},
      {"sink", "nsearch_frames/sink_nsearch/fluid.vtk",
       "nsearch_frames/sink_nsearch/boundary.vtk"},
      {"bowl", "nsearch_frames/bowl_nsearch/fluid.vtk",
       "nsearch_frames/bowl_nsearch/boundary.vtk"},
      {"wall", "nsearch_frames/wall_nsearch/fluid.vtk",
       "nsearch_frames/wall_nsearch/boundary.vtk"},
  };
  std::vector<SceneSpec> scenes;
  for (const Frame &frame : frames) {
    const std::string fluid = res + "/" + frame.fluid;
    const std::string boundary = res + "/" + frame.boundary;
    if (file_exists(fluid) && file_exists(boundary)) {
      scenes.push_back(
          vtk_scene(std::string("learnsph_") + frame.name, fluid, boundary));
    }
  }
  return scenes;
}

} // namespace apfrnn::bench
