#pragma once

// Benchmark scenes: point sets, the active search pairs, and how the scene
// moves between frames. Backends only see SceneData.

#include "scene.hpp"
#include "trace.hpp"
#include "vtk_points.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
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
  // When > 0, set 0 follows a smooth divergence-free flow and moves at most
  // flow * radius per frame. Points keep drifting, so memory order decays and
  // particles keep changing cells, as they do in a simulation.
  float flow = 0.0f;
  float flow_wavenumber = 1.0f;
  // When set, every set follows this captured trace instead of the fields above.
  std::shared_ptr<const Trace> trace;
  // Per set: the trace point each current point is, once a backend has
  // reordered the points (see Backend::zsort). Empty: the trace's own order.
  // It also stands in for the other per-point arrays a simulation reorders.
  std::vector<std::vector<int>> trace_index;
  // Sets whose point count or storage changed in the last advance().
  std::vector<int> resized;
  // Fountain-like growth: set 0 gains grow_per_frame points per frame until
  // it reaches grow_limit. The points are sampled uniformly in the axis-
  // aligned box stored in grow_min/grow_extent. The harness calls
  // Backend::resize() after advance() so every backend sees the new pointer.
  bool growing = false;
  int grow_limit = 0;
  int grow_per_frame = 0;
  float grow_min[3] = {0.0f, 0.0f, 0.0f};
  float grow_extent[3] = {1.0f, 1.0f, 1.0f};
  uint32_t grow_seed = 0x9e3779b9u;

  int points(int set) const {
    return static_cast<int>(xyz[static_cast<std::size_t>(set)].size() / 3);
  }

  // The trace point that point `i` of `set` is (i itself when the points
  // keep the trace's order). Checksums use it, so they do not depend on how
  // a backend reordered the points.
  uint32_t trace_id(int set, int i) const {
    const std::size_t s = static_cast<std::size_t>(set);
    return s < trace_index.size() && !trace_index[s].empty()
               ? static_cast<uint32_t>(trace_index[s][static_cast<std::size_t>(i)])
               : static_cast<uint32_t>(i);
  }

  void grow(int step) {
    if (!growing) {
      return;
    }
    const int current = points(0);
    const int target = std::min(grow_limit, current + grow_per_frame);
    if (target <= current) {
      return;
    }
    std::vector<float> &fluid = xyz[0];
    fluid.reserve(static_cast<std::size_t>(target) * 3);
    uint32_t state =
        grow_seed ^ (0x9e3779b9u * static_cast<uint32_t>(step + 1));
    const auto unit = [&]() {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      return static_cast<float>(state & 0xffffffu) / 16777216.0f;
    };
    for (int index = current; index < target; ++index) {
      fluid.push_back(grow_min[0] + unit() * grow_extent[0]);
      fluid.push_back(grow_min[1] + unit() * grow_extent[1]);
      fluid.push_back(grow_min[2] + unit() * grow_extent[2]);
    }
  }

  void replay(int step) {
    resized.clear();
    for (std::size_t set = 0; set < xyz.size(); ++set) {
      const Trace::Points points = trace->points(static_cast<int>(set), step);
      std::vector<float> &target = xyz[set];
      const std::size_t values = static_cast<std::size_t>(points.count) * 3;
      if (values != target.size()) {
        resized.push_back(static_cast<int>(set));
      }
      if (is_static[set] && values == target.size()) {
        continue;
      }
      const float *before = target.data();
      target.resize(values);
      if (target.data() != before &&
          (resized.empty() || resized.back() != static_cast<int>(set))) {
        resized.push_back(static_cast<int>(set));
      }
      const std::vector<int> *order =
          set < trace_index.size() && !trace_index[set].empty() &&
                  trace_index[set].size() ==
                      static_cast<std::size_t>(points.count)
              ? &trace_index[set]
              : nullptr;
      if (order != nullptr) {
        for (std::size_t i = 0; i < order->size(); ++i) {
          std::memcpy(target.data() + 3 * i,
                      points.xyz + 3 * static_cast<std::size_t>((*order)[i]),
                      3 * sizeof(float));
        }
      } else if (values != 0) {
        std::memcpy(target.data(), points.xyz, values * sizeof(float));
      }
    }
  }

  void advance_flow() {
    std::vector<float> &fluid = xyz[0];
    const float amplitude = flow * radius;
    const float k = flow_wavenumber;
    for (std::size_t i = 0; i + 2 < fluid.size(); i += 3) {
      const float x = fluid[i + 0];
      const float y = fluid[i + 1];
      const float z = fluid[i + 2];
      fluid[i + 0] = x + amplitude * std::sin(k * y);
      fluid[i + 1] = y + amplitude * std::sin(k * z);
      fluid[i + 2] = z + amplitude * std::sin(k * x);
    }
  }

  void advance(int step) {
    if (trace) {
      replay(step);
      return;
    }
    if (flow > 0.0f) {
      advance_flow();
      return;
    }
    if (growing) {
      grow(step);
    }
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
  // TreeNSearch is designed for almost z-sorted particle data. Benchmarks
  // sort by Morton code by default so every backend sees the input order an
  // SPH simulation would normally maintain. Unsorted/ scenes keep raw input
  // order as the adversarial case.
  bool zsorted = true;
};

// Spreads the low 21 bits of `value` so that three of them can be interleaved
// into a 63-bit Morton code.
inline uint64_t spread_morton_bits(uint64_t value) {
  value &= 0x1fffffULL;
  value = (value | (value << 32)) & 0x1f00000000ffffULL;
  value = (value | (value << 16)) & 0x1f0000ff0000ffULL;
  value = (value | (value << 8)) & 0x100f00f00f00f00fULL;
  value = (value | (value << 4)) & 0x10c30c30c30c30c3ULL;
  value = (value | (value << 2)) & 0x1249249249249249ULL;
  return value;
}

inline uint64_t morton3(uint32_t x, uint32_t y, uint32_t z) {
  return spread_morton_bits(x) | (spread_morton_bits(y) << 1) |
         (spread_morton_bits(z) << 2);
}

// Reorders one interleaved xyz buffer along the Morton curve of a uniform
// grid with cell size `cell` and origin (ox, oy, oz). The permutation is
// stable for points that share a cell, so equal inputs keep a deterministic
// order.
inline void zsort_xyz(std::vector<float> &xyz, float cell, float ox, float oy,
                      float oz) {
  const std::size_t count = xyz.size() / 3;
  if (count < 2) {
    return;
  }
  std::vector<std::pair<uint64_t, uint32_t>> keys(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto cell_index = [&](std::size_t axis) {
      const float value = xyz[3 * index + axis];
      const double coordinate =
          std::floor((static_cast<double>(value) - ox) / cell);
      const double clamped =
          std::clamp(coordinate, 0.0, 2097151.0); // 21 bits per axis
      return static_cast<uint32_t>(clamped);
    };
    keys[index] = {morton3(cell_index(0), cell_index(1), cell_index(2)),
                   static_cast<uint32_t>(index)};
  }
  std::sort(keys.begin(), keys.end());
  std::vector<float> sorted(xyz.size());
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t source = keys[index].second;
    sorted[3 * index + 0] = xyz[3 * source + 0];
    sorted[3 * index + 1] = xyz[3 * source + 1];
    sorted[3 * index + 2] = xyz[3 * source + 2];
  }
  xyz.swap(sorted);
}

inline void zsort_scene(SceneData &scene) {
  if (scene.radius <= 0.0f || scene.xyz.empty()) {
    return;
  }
  float ox = std::numeric_limits<float>::infinity();
  float oy = ox;
  float oz = ox;
  for (const std::vector<float> &xyz : scene.xyz) {
    for (std::size_t index = 0; index + 2 < xyz.size(); index += 3) {
      ox = std::min(ox, xyz[index + 0]);
      oy = std::min(oy, xyz[index + 1]);
      oz = std::min(oz, xyz[index + 2]);
    }
  }
  // The paper's calibrated cell size is 1.5x the support radius. The exact
  // value only affects locality, not which points are neighbors.
  const float cell = std::max(1.5f * scene.radius, 1.0e-6f);
  for (std::vector<float> &xyz : scene.xyz) {
    zsort_xyz(xyz, cell, ox, oy, oz);
  }
}

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
                               float domain, bool jitter = false) {
  const int points = droplets * points_per_droplet;
  std::string name = "droplets_" + std::to_string(points / 1000) + "k";
  if (jitter) {
    name += "_jitter";
  }
  return {name, [=] {
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
            if (jitter) {
              data.jitter = 0.25f;
            }
            return data;
          }};
}

// A saved learnsph frame: fluid and static boundary, searched fluid-fluid and
// fluid-boundary with the learnsph support radius.
inline SceneSpec vtk_scene(const std::string &name, const std::string &fluid,
                           const std::string &boundary,
                           bool static_boundary = true) {
  std::string scene_name = name;
  if (!static_boundary) {
    scene_name += "_dynamic_boundary";
  }
  return {scene_name, [=] {
            SceneData data;
            data.radius = 0.048f;
            data.xyz = {read_vtk_points(fluid).xyz,
                        read_vtk_points(boundary).xyz};
            data.is_static = {false, static_boundary};
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

// One scene per captured trace, named after its file. Playback keeps the order
// the simulation produced, which decays over time as it does in a real run.
inline std::vector<SceneSpec> trace_scenes(const std::string &directory) {
  std::vector<SceneSpec> scenes;
  for (const std::filesystem::path &path : trace_files(directory)) {
    std::string name = path.stem().string();
    for (char &c : name) {
      if (c == '.') {
        c = 'p';
      }
    }
    SceneSpec spec{name, [file = path.string()] {
                     auto trace = std::make_shared<const Trace>(file);
                     SceneData data;
                     if (!trace->valid()) {
                       return data;
                     }
                     const Trace::Header &header = trace->header();
                     data.radius = header.radius;
                     data.xyz.resize(static_cast<std::size_t>(header.sets));
                     data.is_static.resize(static_cast<std::size_t>(header.sets));
                     for (int set = 0; set < header.sets; ++set) {
                       data.is_static[static_cast<std::size_t>(set)] =
                           header.is_static[set] != 0;
                       const Trace::Points points = trace->points(set, 0);
                       if (points.count > 0) {
                         data.xyz[static_cast<std::size_t>(set)].assign(
                             points.xyz, points.xyz + 3 * points.count);
                       }
                     }
                     // A static query set has no moving points to search for.
                     // learnsph leaves such a pair active although it only reads
                     // those lists during setup.
                     for (int pair = 0; pair < header.pair_count; ++pair) {
                       const int query = header.pairs[pair][0];
                       if (header.is_static[query] == 0) {
                         data.pairs.push_back({query, header.pairs[pair][1]});
                       }
                     }
                     data.trace = std::move(trace);
                     return data;
                   }};
    spec.zsorted = false;
    scenes.push_back(std::move(spec));
  }
  return scenes;
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
  scenes.push_back(droplet_scene(400, 500, 20000.0f, true));
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
inline std::vector<SceneSpec> learnsph_scenes(const std::string &res,
                                              bool static_boundary = true) {
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
      scenes.push_back(vtk_scene(
          std::string("learnsph_") + frame.name, fluid, boundary,
          static_boundary));
    }
  }
  return scenes;
}

// The same scene with the input order left as generated. It shows how much
// the almost-z-sorted assumption matters to each backend.
inline SceneSpec unsorted_scene(SceneSpec spec) {
  spec.name += "_unsorted";
  spec.zsorted = false;
  return spec;
}

// The same scene with every point set marked dynamic. This removes the
// static-target cache advantage, so the timing measures neighbor search
// instead of end-to-end caching.
inline SceneSpec all_dynamic_scene(SceneSpec spec) {
  spec.name += "_dynamic_boundary";
  auto make = spec.make;
  spec.make = [make] {
    SceneData data = make();
    data.is_static.assign(data.is_static.size(), false);
    return data;
  };
  return spec;
}

// Order-sensitivity pairs: each scene sorted and unsorted. Only a handful of
// scenes so the table stays readable.
inline std::vector<SceneSpec> order_scenes() {
  using D = Distribution;
  std::vector<SceneSpec> scenes;
  for (const SceneSpec &spec :
       {synthetic_scene(D::Uniform, 200000),
        synthetic_scene(D::Slab, 200000),
        synthetic_scene(D::Clustered, 200000)}) {
    scenes.push_back(spec);
    scenes.push_back(unsorted_scene(spec));
  }
  return scenes;
}

// Static-boundary vs all-dynamic pairs. The static variant is the real
// learnsph workload; the dynamic variant is the algorithm-only comparison.
inline std::vector<SceneSpec> dynamic_scenes(const std::string &res) {
  using D = Distribution;
  std::vector<SceneSpec> scenes;
  for (const SceneSpec &spec :
       {synthetic_scene(D::Uniform, 200000),
        synthetic_scene(D::Slab, 200000),
        synthetic_scene(D::Clustered, 200000)}) {
    scenes.push_back(spec);
    scenes.push_back(all_dynamic_scene(spec));
  }
  for (const SceneSpec &spec : learnsph_scenes(res, /*static_boundary=*/true)) {
    scenes.push_back(spec);
    scenes.push_back(all_dynamic_scene(spec));
  }
  return scenes;
}

// Fountain-like growth: add points to set 0 every frame until `limit`. The
// growth box is the initial bounding box of the fluid, so density rises just
// as it does in the Fountain scene.
inline SceneSpec growing_scene(SceneSpec spec, int limit, int per_frame) {
  spec.name += "_growing";
  auto make = spec.make;
  spec.make = [=] {
    SceneData data = make();
    if (data.xyz.empty() || data.points(0) <= 0 || limit <= data.points(0)) {
      return data;
    }
    data.growing = true;
    data.grow_limit = limit;
    data.grow_per_frame = per_frame;
    data.motion = Motion::Static;
    data.jitter = 0.0f;
    float low[3] = {data.xyz[0][0], data.xyz[0][1], data.xyz[0][2]};
    float high[3] = {low[0], low[1], low[2]};
    for (std::size_t index = 0; index + 2 < data.xyz[0].size(); index += 3) {
      for (int axis = 0; axis < 3; ++axis) {
        low[axis] = std::min(low[axis], data.xyz[0][index + axis]);
        high[axis] = std::max(high[axis], data.xyz[0][index + axis]);
      }
    }
    for (int axis = 0; axis < 3; ++axis) {
      data.grow_min[axis] = low[axis];
      data.grow_extent[axis] = std::max(high[axis] - low[axis], 1.0e-3f);
    }
    return data;
  };
  return spec;
}

// Resize-under-load scenes. The point count doubles over roughly 40 frames,
// which exercises the same regrow path a fountain reaches as emitters fill
// the domain.
inline std::vector<SceneSpec> growing_scenes() {
  using D = Distribution;
  std::vector<SceneSpec> scenes;
  for (const SceneSpec &spec :
       {synthetic_scene(D::Uniform, 200000),
        synthetic_scene(D::Slab, 200000),
        synthetic_scene(D::Clustered, 200000)}) {
    scenes.push_back(growing_scene(spec, 400000, 5000));
  }
  return scenes;
}

// Dense vs sparse scaling, the axis the TreeNSearch paper uses from 175k to
// 27M. The sparse points are separated by far more than the radius, so every
// point has zero neighbors; dense is a uniform SPH-like cloud. 27M is opt-in
// because the fixed-row tables need more than the default memory budget.
inline std::vector<SceneSpec> scale_scenes() {
  using D = Distribution;
  std::vector<SceneSpec> scenes;
  for (int points : {175000, 500000, 1000000, 2000000, 4000000, 8000000}) {
    scenes.push_back(synthetic_scene(D::Uniform, points));
    scenes.push_back(synthetic_scene(D::HugeDomain, points));
  }
  if (const char *big = std::getenv("APFRNN_BENCH_BIG")) {
    if (std::atoi(big) != 0) {
      scenes.push_back(synthetic_scene(D::Uniform, 27000000));
      scenes.push_back(synthetic_scene(D::HugeDomain, 27000000));
    }
  }
  return scenes;
}

} // namespace apfrnn::bench
