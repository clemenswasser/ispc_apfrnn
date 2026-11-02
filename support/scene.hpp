#pragma once

#include "point_cloud.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace apfrnn::bench {

enum class Distribution {
  Uniform,
  JitteredLattice,
  Clustered,
  Slab,
  Lattice,
  Pathological,
  HugeDomain,
  Surface,
  ThinPipe,
  TwoScale,
  SlabThin,
};

enum class Motion {
  Static,
  SmallMove,
  LargeMove,
};

struct Scene {
  support::PointCloud fluid;
  support::PointCloud boundary;
  support::PointCloud emitter;
  std::vector<float> fluid_xyz;
  std::vector<float> boundary_xyz;
  std::vector<float> emitter_xyz;
  float radius = 2.0f;
  float side = 1.0f;
};

inline std::vector<float> interleave(const support::PointCloud &cloud) {
  std::vector<float> xyz(cloud.x.size() * 3);
  for (std::size_t index = 0; index < cloud.x.size(); ++index) {
    xyz[3 * index + 0] = cloud.x[index];
    xyz[3 * index + 1] = cloud.y[index];
    xyz[3 * index + 2] = cloud.z[index];
  }
  return xyz;
}

inline void sync_interleaved(const support::PointCloud &cloud,
                             std::vector<float> &xyz) {
  for (std::size_t index = 0; index < cloud.x.size(); ++index) {
    xyz[3 * index + 0] = cloud.x[index];
    xyz[3 * index + 1] = cloud.y[index];
    xyz[3 * index + 2] = cloud.z[index];
  }
}

inline support::PointCloud make_uniform(int n, float side, uint32_t seed) {
  return support::make_point_cloud(n, side, seed);
}

inline support::PointCloud make_jittered_lattice(int n, float side,
                                                 uint32_t seed,
                                                 bool jitter) {
  support::PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(n));
  cloud.y.resize(static_cast<std::size_t>(n));
  cloud.z.resize(static_cast<std::size_t>(n));

  const int per_axis =
      std::max(1, static_cast<int>(std::ceil(std::cbrt(
                      static_cast<float>(n)))));
  const float spacing = side / static_cast<float>(per_axis);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> noise(-0.18f * spacing,
                                              0.18f * spacing);

  int index = 0;
  for (int x = 0; x < per_axis && index < n; ++x) {
    for (int y = 0; y < per_axis && index < n; ++y) {
      for (int z = 0; z < per_axis && index < n; ++z) {
        const float jx = jitter ? noise(generator) : 0.0f;
        const float jy = jitter ? noise(generator) : 0.0f;
        const float jz = jitter ? noise(generator) : 0.0f;
        cloud.x[static_cast<std::size_t>(index)] =
            (static_cast<float>(x) + 0.5f) * spacing + jx;
        cloud.y[static_cast<std::size_t>(index)] =
            (static_cast<float>(y) + 0.5f) * spacing + jy;
        cloud.z[static_cast<std::size_t>(index)] =
            (static_cast<float>(z) + 0.5f) * spacing + jz;
        ++index;
      }
    }
  }
  return cloud;
}

inline support::PointCloud make_clustered(int n, float side, uint32_t seed) {
  support::PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(n));
  cloud.y.resize(static_cast<std::size_t>(n));
  cloud.z.resize(static_cast<std::size_t>(n));

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> center_distribution(0.1f * side,
                                                            0.9f * side);
  std::normal_distribution<float> spread(0.0f, 0.055f * side);

  constexpr int kClusters = 8;
  float centers[kClusters][3];
  for (int cluster = 0; cluster < kClusters; ++cluster) {
    centers[cluster][0] = center_distribution(generator);
    centers[cluster][1] = center_distribution(generator);
    centers[cluster][2] = center_distribution(generator);
  }

  for (int index = 0; index < n; ++index) {
    const int cluster = index % kClusters;
    cloud.x[static_cast<std::size_t>(index)] =
        std::clamp(centers[cluster][0] + spread(generator), 0.0f, side);
    cloud.y[static_cast<std::size_t>(index)] =
        std::clamp(centers[cluster][1] + spread(generator), 0.0f, side);
    cloud.z[static_cast<std::size_t>(index)] =
        std::clamp(centers[cluster][2] + spread(generator), 0.0f, side);
  }
  return cloud;
}

inline support::PointCloud make_slab(int n, float side, uint32_t seed) {
  support::PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(n));
  cloud.y.resize(static_cast<std::size_t>(n));
  cloud.z.resize(static_cast<std::size_t>(n));

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> x_distribution(0.0f, 0.55f * side);
  std::uniform_real_distribution<float> yz_distribution(0.0f, side);
  for (int index = 0; index < n; ++index) {
    cloud.x[static_cast<std::size_t>(index)] = x_distribution(generator);
    cloud.y[static_cast<std::size_t>(index)] = yz_distribution(generator);
    cloud.z[static_cast<std::size_t>(index)] = yz_distribution(generator);
  }
  return cloud;
}

inline support::PointCloud make_slab_thin(int n, float side, uint32_t seed) {
  support::PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(n));
  cloud.y.resize(static_cast<std::size_t>(n));
  cloud.z.resize(static_cast<std::size_t>(n));

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> x_distribution(0.0f, 0.12f * side);
  std::uniform_real_distribution<float> yz_distribution(0.0f, side);
  for (int index = 0; index < n; ++index) {
    cloud.x[static_cast<std::size_t>(index)] = x_distribution(generator);
    cloud.y[static_cast<std::size_t>(index)] = yz_distribution(generator);
    cloud.z[static_cast<std::size_t>(index)] = yz_distribution(generator);
  }
  return cloud;
}

inline support::PointCloud make_pathological(int n, uint32_t seed) {
  support::PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(n));
  cloud.y.resize(static_cast<std::size_t>(n));
  cloud.z.resize(static_cast<std::size_t>(n));

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(0.0f, 0.02f);
  for (int index = 0; index < n; ++index) {
    cloud.x[static_cast<std::size_t>(index)] = distribution(generator);
    cloud.y[static_cast<std::size_t>(index)] = distribution(generator);
    cloud.z[static_cast<std::size_t>(index)] = distribution(generator);
  }
  return cloud;
}

inline support::PointCloud make_surface(int n, float side, uint32_t seed,
                                        bool jitter) {
  support::PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(n));
  cloud.y.resize(static_cast<std::size_t>(n));
  cloud.z.resize(static_cast<std::size_t>(n));

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> noise(-0.01f, 0.01f);
  const float radius = 0.45f * side;
  const float center = 0.5f * side;
  constexpr float kGoldenAngle = 2.39996322972865332f;
  for (int index = 0; index < n; ++index) {
    const float t = (static_cast<float>(index) + 0.5f) /
                    static_cast<float>(n);
    const float y = 1.0f - 2.0f * t;
    const float ring = std::sqrt(std::max(0.0f, 1.0f - y * y));
    const float theta = kGoldenAngle * static_cast<float>(index);
    const float jitter_amount = jitter ? noise(generator) * side : 0.0f;
    const float r = radius + jitter_amount;
    cloud.x[static_cast<std::size_t>(index)] = center + r * ring * std::cos(theta);
    cloud.y[static_cast<std::size_t>(index)] = center + r * y;
    cloud.z[static_cast<std::size_t>(index)] = center + r * ring * std::sin(theta);
  }
  return cloud;
}

inline support::PointCloud make_thin_pipe(int n, float side, uint32_t seed) {
  support::PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(n));
  cloud.y.resize(static_cast<std::size_t>(n));
  cloud.z.resize(static_cast<std::size_t>(n));

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> along(0.0f, side);
  std::uniform_real_distribution<float> angle(0.0f, 2.0f * 3.14159265358979f);
  const float radius = 0.035f * side;
  for (int index = 0; index < n; ++index) {
    const float theta = angle(generator);
    cloud.x[static_cast<std::size_t>(index)] = along(generator);
    cloud.y[static_cast<std::size_t>(index)] =
        0.5f * side + radius * std::cos(theta);
    cloud.z[static_cast<std::size_t>(index)] =
        0.5f * side + radius * std::sin(theta);
  }
  return cloud;
}

inline support::PointCloud make_two_scale(int n, float side, uint32_t seed) {
  support::PointCloud cloud;
  cloud.x.resize(static_cast<std::size_t>(n));
  cloud.y.resize(static_cast<std::size_t>(n));
  cloud.z.resize(static_cast<std::size_t>(n));

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> center(0.1f * side, 0.9f * side);
  std::normal_distribution<float> broad(0.0f, 0.08f * side);
  std::normal_distribution<float> tight(0.0f, 0.012f * side);
  constexpr int kBroadClusters = 6;
  constexpr int kTightClusters = 64;
  float broad_centers[kBroadClusters][3];
  float tight_centers[kTightClusters][3];
  for (int c = 0; c < kBroadClusters; ++c) {
    for (int axis = 0; axis < 3; ++axis) {
      broad_centers[c][axis] = center(generator);
    }
  }
  for (int c = 0; c < kTightClusters; ++c) {
    const int parent = c % kBroadClusters;
    for (int axis = 0; axis < 3; ++axis) {
      tight_centers[c][axis] =
          broad_centers[parent][axis] + broad(generator);
    }
  }

  for (int index = 0; index < n; ++index) {
    float px;
    float py;
    float pz;
    if ((index % 10) < 7) {
      const int c = index % kBroadClusters;
      px = broad_centers[c][0] + broad(generator);
      py = broad_centers[c][1] + broad(generator);
      pz = broad_centers[c][2] + broad(generator);
    } else {
      const int c = index % kTightClusters;
      px = tight_centers[c][0] + tight(generator);
      py = tight_centers[c][1] + tight(generator);
      pz = tight_centers[c][2] + tight(generator);
    }
    cloud.x[static_cast<std::size_t>(index)] = std::clamp(px, 0.0f, side);
    cloud.y[static_cast<std::size_t>(index)] = std::clamp(py, 0.0f, side);
    cloud.z[static_cast<std::size_t>(index)] = std::clamp(pz, 0.0f, side);
  }
  return cloud;
}

inline Scene make_scene(Distribution distribution, int fluid_points,
                        uint32_t seed = 1001) {
  Scene scene;
  scene.side = std::cbrt(static_cast<float>(fluid_points));
  const int boundary_points = std::max(1, fluid_points / 4);
  const int emitter_points = std::max(1, fluid_points / 40);

  switch (distribution) {
  case Distribution::Uniform:
    scene.fluid = make_uniform(fluid_points, scene.side, seed);
    scene.boundary =
        make_uniform(boundary_points, scene.side, seed + 1000);
    scene.emitter = make_uniform(emitter_points, scene.side, seed + 2000);
    scene.radius = 2.0f;
    break;

  case Distribution::JitteredLattice:
    scene.fluid =
        make_jittered_lattice(fluid_points, scene.side, seed, true);
    scene.boundary =
        make_uniform(boundary_points, scene.side, seed + 1000);
    scene.emitter = make_uniform(emitter_points, scene.side, seed + 2000);
    scene.radius = 2.0f;
    break;

  case Distribution::Lattice:
    scene.fluid =
        make_jittered_lattice(fluid_points, scene.side, seed, false);
    scene.boundary =
        make_jittered_lattice(boundary_points, scene.side, seed + 1000, false);
    scene.emitter =
        make_jittered_lattice(emitter_points, scene.side, seed + 2000, false);
    scene.radius = 2.0f;
    break;

  case Distribution::Clustered:
    scene.fluid = make_clustered(fluid_points, scene.side, seed);
    scene.boundary = make_uniform(boundary_points, scene.side, seed + 1000);
    scene.emitter = make_clustered(emitter_points, scene.side, seed + 2000);
    scene.radius = 2.0f;
    break;

  case Distribution::Slab:
    scene.fluid = make_slab(fluid_points, scene.side, seed);
    scene.boundary = make_uniform(boundary_points, scene.side, seed + 1000);
    scene.emitter = make_slab(emitter_points, scene.side, seed + 2000);
    scene.radius = 2.0f;
    break;

  case Distribution::SlabThin:
    scene.fluid = make_slab_thin(fluid_points, scene.side, seed);
    scene.boundary = make_uniform(boundary_points, scene.side, seed + 1000);
    scene.emitter = make_slab_thin(emitter_points, scene.side, seed + 2000);
    scene.radius = 2.0f;
    break;

  case Distribution::Pathological:
    scene.fluid = make_pathological(fluid_points, seed);
    scene.boundary = make_pathological(boundary_points, seed + 1000);
    scene.emitter = make_pathological(emitter_points, seed + 2000);
    scene.radius = 10.0f;
    break;

  case Distribution::HugeDomain:
    // Large enough to exceed the dense-grid/counting-sort budget while still
    // fitting both the 21-bit cell key range and TreeNSearch's per-dimension
    // cell extent limit.
    scene.side = 6.0e4f;
    scene.fluid = make_uniform(fluid_points, scene.side, seed);
    scene.boundary =
        make_uniform(boundary_points, scene.side, seed + 1000);
    scene.emitter = make_uniform(emitter_points, scene.side, seed + 2000);
    scene.radius = 2.0f;
    break;

  case Distribution::Surface:
    scene.fluid = make_surface(fluid_points, scene.side, seed, true);
    scene.boundary =
        make_surface(boundary_points, scene.side, seed + 1000, true);
    scene.emitter = make_surface(emitter_points, scene.side, seed + 2000, true);
    scene.radius = 2.0f;
    break;

  case Distribution::ThinPipe:
    scene.fluid = make_thin_pipe(fluid_points, scene.side, seed);
    scene.boundary = make_thin_pipe(boundary_points, scene.side, seed + 1000);
    scene.emitter = make_thin_pipe(emitter_points, scene.side, seed + 2000);
    scene.radius = 2.0f;
    break;

  case Distribution::TwoScale:
    scene.fluid = make_two_scale(fluid_points, scene.side, seed);
    scene.boundary = make_uniform(boundary_points, scene.side, seed + 1000);
    scene.emitter = make_two_scale(emitter_points, scene.side, seed + 2000);
    scene.radius = 2.0f;
    break;
  }

  scene.fluid_xyz = interleave(scene.fluid);
  scene.boundary_xyz = interleave(scene.boundary);
  scene.emitter_xyz = interleave(scene.emitter);
  return scene;
}

// Translation applied to the moving point set in frame `step`: the whole set
// drifts along a slow Lissajous path, by 3% (small) or 35% (large) of the
// radius per frame.
struct Offset {
  float dx = 0.0f, dy = 0.0f, dz = 0.0f;
};

inline Offset motion_offset(Motion motion, int step, float radius) {
  if (motion == Motion::Static) {
    return {};
  }
  const float amplitude =
      (motion == Motion::SmallMove ? 0.03f : 0.35f) * radius;
  const float phase = static_cast<float>(step);
  return {amplitude * std::sin(0.17f * phase),
          amplitude * std::cos(0.11f * phase),
          amplitude * std::sin(0.07f * phase)};
}

inline void apply_motion(Scene &scene, Motion motion, int step) {
  if (motion == Motion::Static) {
    return;
  }
  const Offset offset = motion_offset(motion, step, scene.radius);
  for (std::size_t index = 0; index < scene.fluid.x.size(); ++index) {
    scene.fluid.x[index] += offset.dx;
    scene.fluid.y[index] += offset.dy;
    scene.fluid.z[index] += offset.dz;
  }
  sync_interleaved(scene.fluid, scene.fluid_xyz);
}

} // namespace apfrnn::bench
