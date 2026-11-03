#pragma once

// Benchmark scenes: point sets, the active search pairs, and how the scene
// moves between frames. Backends only see SceneData.

#include "scene.hpp"
#include "vtk_points.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
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
  // When > 0, set 0 instead moves non-rigidly: every frame each coordinate is
  // its initial value plus independent noise of up to jitter * radius, so
  // the sorted order changes between frames.
  float jitter = 0.0f;
  std::vector<float> jitter_base;

  int points(int set) const {
    return static_cast<int>(xyz[static_cast<std::size_t>(set)].size() / 3);
  }

  void advance(int step) {
    if (jitter > 0.0f) {
      std::vector<float> &fluid = xyz[0];
      if (jitter_base.empty()) {
        jitter_base = fluid;
      }
      uint32_t state = 0x9e3779b9u * static_cast<uint32_t>(step) | 1u;
      for (std::size_t i = 0; i < fluid.size(); ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const float unit = static_cast<float>(state & 0xffffu) / 32768.0f - 1.0f;
        fluid[i] = jitter_base[i] + jitter * radius * unit;
      }
      return;
    }
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

// The synthetic scene with non-rigid per-point motion of up to a quarter
// radius per axis around the initial positions.
inline SceneSpec jittered_scene(Distribution distribution, int points) {
  SceneSpec spec = synthetic_scene(distribution, points);
  auto make = spec.make;
  return {spec.name + "_jitter", [make] {
            SceneData data = make();
            data.jitter = 0.25f;
            return data;
          }};
}

// The synthetic scene where the static boundary also queries the fluid and
// itself, as for boundary-side SPH terms.
inline SceneSpec boundary_query_scene(Distribution distribution, int points) {
  SceneSpec spec = synthetic_scene(distribution, points);
  auto make = spec.make;
  return {spec.name + "_boundary_queries", [make] {
            SceneData data = make();
            data.pairs.push_back({1, 0});
            data.pairs.push_back({1, 1});
            return data;
          }};
}

// 1k dynamic probe points queried against a large dynamic uniform set; the
// large set is only a target.
inline SceneSpec probe_scene(int points) {
  return {"probe_1k_in_" + std::to_string(points / 1000) + "k", [points] {
            SceneData data;
            data.radius = 2.0f;
            const float side = std::cbrt(static_cast<float>(points));
            data.xyz = {interleave(support::make_point_cloud(points, side, 7)),
                        interleave(support::make_point_cloud(1000, side, 8))};
            data.is_static = {false, false};
            data.pairs = {{1, 0}};
            return data;
          }};
}

// Dense spherical droplets (about one point per unit volume, radius 2, so
// ~26 neighbors) scattered through a cube of side `domain`. The bounding box
// is far too large for the dense cell lookup, so this is the hash grid with
// a real neighbor count; huge_domain only covers the empty (sweep) case.
inline SceneSpec droplet_scene(int droplets, int points_per_droplet,
                               float domain) {
  const int points = droplets * points_per_droplet;
  return {"droplets_" + std::to_string(points / 1000) + "k", [=] {
            SceneData data;
            data.radius = 2.0f;
            std::mt19937 generator(5);
            std::uniform_real_distribution<float> center(0.0f, domain);
            std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
            const float droplet_radius = std::cbrt(
                static_cast<float>(points_per_droplet) * 3.0f /
                (4.0f * 3.14159265f));
            std::vector<float> xyz;
            xyz.reserve(static_cast<std::size_t>(points) * 3);
            for (int droplet = 0; droplet < droplets; ++droplet) {
              const float cx = center(generator);
              const float cy = center(generator);
              const float cz = center(generator);
              for (int k = 0; k < points_per_droplet;) {
                const float x = unit(generator);
                const float y = unit(generator);
                const float z = unit(generator);
                if (x * x + y * y + z * z > 1.0f) {
                  continue;
                }
                xyz.insert(xyz.end(), {cx + droplet_radius * x,
                                       cy + droplet_radius * y,
                                       cz + droplet_radius * z});
                ++k;
              }
            }
            data.xyz = {std::move(xyz)};
            data.is_static = {false};
            data.pairs = {{0, 0}};
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
  for (D d : {D::Uniform, D::Slab}) {
    scenes.push_back(synthetic_scene(d, 200000, Motion::LargeMove));
    scenes.push_back(jittered_scene(d, 200000));
  }
  scenes.push_back(jittered_scene(D::Clustered, 200000));
  scenes.push_back(jittered_scene(D::ThinPipe, 50000));
  scenes.push_back(boundary_query_scene(D::Slab, 200000));
  scenes.push_back(probe_scene(200000));
  scenes.push_back(probe_scene(1000000));
  scenes.push_back(synthetic_scene(D::Uniform, 1000000));
  scenes.push_back(synthetic_scene(D::Pathological, 10000));
  return scenes;
}

// Regimes outside the headline matrix: large high-degree sets where the
// default memory budget forces CSR, larger uniform sets, many point sets
// with every pair active, and dense droplets in a huge (hash grid) domain.
// Kept separate so a normal run stays focused.
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
  scenes.push_back(droplet_scene(400, 500, 20000.0f));
  return scenes;
}

// Scenes for Cold/: setup plus the first frame, where row capacities are not
// yet fitted and buffers are freshly allocated.
inline std::vector<SceneSpec> cold_scenes() {
  using D = Distribution;
  return {synthetic_scene(D::Uniform, 200000), synthetic_scene(D::Slab, 200000),
          synthetic_scene(D::Clustered, 200000),
          synthetic_scene(D::ThinPipe, 50000)};
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
