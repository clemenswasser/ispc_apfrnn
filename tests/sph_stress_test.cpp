// Multi-frame stress test shaped like an SPH solver driving the search: a
// large fluid set in double precision updated in place every frame, a static
// boundary set, splashes that throw particles far outside the domain, removal
// of particles that left the domain (the array shrinks in place), reallocation
// and emitter growth. Every active pair is checked against the grid reference
// on every frame.
//
//   APFRNN_SPH_STRESS_FLUID=N     fluid particles (default 120000)
//   APFRNN_SPH_STRESS_FRAMES=N    frames (default 24)
//   APFRNN_SPH_STRESS_SEED=S      seed (default 1)

#include "grid_reference.hpp"
#include "point_cloud.hpp"
#include "apfrnn/neighbor_search.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using apfrnn::NSearch;
using apfrnn::support::check_pair;
using apfrnn::support::PointCloud;

int env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  return value != nullptr ? std::atoi(value) : fallback;
}

PointCloud as_cloud(const std::vector<double> &xyz) {
  PointCloud cloud;
  const std::size_t n = xyz.size() / 3;
  cloud.x.resize(n);
  cloud.y.resize(n);
  cloud.z.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    cloud.x[i] = static_cast<float>(xyz[3 * i + 0]);
    cloud.y[i] = static_cast<float>(xyz[3 * i + 1]);
    cloud.z[i] = static_cast<float>(xyz[3 * i + 2]);
  }
  return cloud;
}

class SphStress {
public:
  explicit SphStress(uint32_t seed) : rng_(seed) {}

  void run(int fluid_target, int frames) {
    constexpr double kRadius = 0.02;
    constexpr double kSpacing = kRadius / 2.0;
    radius_ = static_cast<float>(kRadius);

    // Fluid block above a floor, next to a wall, as in a dam break.
    const int side = std::max(8, static_cast<int>(std::cbrt(fluid_target)));
    std::uniform_real_distribution<double> jitter(-0.05 * kSpacing,
                                                  0.05 * kSpacing);
    for (int i = 0; i < side; ++i) {
      for (int j = 0; j < side; ++j) {
        for (int k = 0; k < side; ++k) {
          push_fluid(0.1 + i * kSpacing + jitter(rng_),
                     0.1 + j * kSpacing + jitter(rng_),
                     0.1 + k * kSpacing + jitter(rng_), 0.0, 0.0, 0.0);
        }
      }
    }
    extent_ = 0.2 + 2.5 * side * kSpacing;
    for (double x = 0.0; x <= extent_; x += kSpacing) {
      for (double y = 0.0; y <= extent_; y += kSpacing) {
        for (int layer = 0; layer < 2; ++layer) {
          push_boundary(x, y, -layer * kSpacing);           // floor
          push_boundary(extent_ + layer * kSpacing, x, y);  // wall
        }
      }
    }

    NSearch search(radius_);
    const int fluid = search.add_point_set(fluid_xyz_.data(), fluid_count());
    const int boundary = search.add_point_set(
        boundary_xyz_.data(), static_cast<int>(boundary_xyz_.size() / 3),
        true);
    search.set_active_search(fluid, fluid);
    search.set_active_search(fluid, boundary);
    search.set_active_search(boundary, boundary);
    const PointCloud boundary_cloud = as_cloud(boundary_xyz_);
    int registered = fluid_count();

    for (int frame = 0; frame < frames; ++frame) {
      if (frame > 0) {
        step(frame);
        // Like a solver, clip only once per output frame of several
        // searches, so particles thrown out stay in the search for a while.
        const bool clipped = frame % 4 == 0 && clip();
        if (clipped || fluid_count() != registered || frame % 7 == 3) {
          if (frame % 2 == 0) {
            // A grown or copied array moves the caller's storage.
            std::vector<double>(fluid_xyz_).swap(fluid_xyz_);
          }
          search.resize_point_set(fluid, fluid_xyz_.data(), fluid_count());
          registered = fluid_count();
        }
      }
      search.run();
      const PointCloud fluid_cloud = as_cloud(fluid_xyz_);
      INFO("frame " << frame << " fluid " << fluid_count());
      CHECK(check_pair(search, fluid, fluid, fluid_cloud, fluid_cloud,
                       radius_) == "");
      CHECK(check_pair(search, fluid, boundary, fluid_cloud, boundary_cloud,
                       radius_) == "");
      if (frame == 0 || frame + 1 == frames) {
        CHECK(check_pair(search, boundary, boundary, boundary_cloud,
                         boundary_cloud, radius_) == "");
      }
    }
  }

private:
  int fluid_count() const { return static_cast<int>(fluid_xyz_.size() / 3); }

  void push_fluid(double x, double y, double z, double vx, double vy,
                  double vz) {
    fluid_xyz_.insert(fluid_xyz_.end(), {x, y, z});
    velocity_.insert(velocity_.end(), {vx, vy, vz});
  }

  void push_boundary(double x, double y, double z) {
    boundary_xyz_.insert(boundary_xyz_.end(), {x, y, z});
  }

  // Falls under gravity, bounces off the floor and wall with random lateral
  // velocity, and throws a few particles far away, some beyond any dense grid.
  void step(int frame) {
    constexpr double kDt = 0.004;
    const double r = radius_;
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    for (int i = 0; i < fluid_count(); ++i) {
      double *p = &fluid_xyz_[3 * static_cast<std::size_t>(i)];
      double *v = &velocity_[3 * static_cast<std::size_t>(i)];
      v[2] -= 9.81 * kDt;
      if (p[2] < r && v[2] < 0.0) {
        v[2] = -0.5 * v[2];
        v[0] += 0.3 * unit(rng_);
        v[1] += 0.3 * unit(rng_);
      }
      if (p[0] > extent_ - r && v[0] > 0.0) {
        v[0] = -0.5 * v[0];
      }
      const double roll = chance(rng_);
      if (roll < 2e-5) {
        // Splash leaving the domain far enough to defeat a dense grid.
        for (int axis = 0; axis < 3; ++axis) {
          p[axis] += 4e4 * r * unit(rng_);
        }
      } else if (roll < 4e-4) {
        for (int axis = 0; axis < 3; ++axis) {
          v[axis] += 20.0 * r / kDt * unit(rng_);
        }
      }
      for (int axis = 0; axis < 3; ++axis) {
        p[axis] += v[axis] * kDt;
      }
    }
    // Emitter: a new layer of fluid appears above the block.
    if (frame % 5 == 0) {
      const double spacing = r / 2.0;
      const int side = 40;
      for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
          push_fluid(0.15 + i * spacing, 0.15 + j * spacing, extent_ * 0.8,
                     0.0, 0.0, -1.0);
        }
      }
    }
  }

  // Removes particles outside the domain, keeping the order of the rest, the
  // way a solver erases them from its arrays. Returns whether any were removed.
  bool clip() {
    const double lo = -0.5;
    const double hi = extent_ + 0.5;
    std::size_t out = 0;
    const std::size_t n = fluid_xyz_.size() / 3;
    for (std::size_t i = 0; i < n; ++i) {
      const double *p = &fluid_xyz_[3 * i];
      if (p[0] < lo || p[0] > hi || p[1] < lo || p[1] > hi || p[2] < lo ||
          p[2] > hi) {
        continue;
      }
      std::copy(p, p + 3, &fluid_xyz_[3 * out]);
      std::copy(&velocity_[3 * i], &velocity_[3 * i] + 3, &velocity_[3 * out]);
      ++out;
    }
    const bool removed = out != n;
    fluid_xyz_.resize(3 * out);
    velocity_.resize(3 * out);
    return removed;
  }

  std::mt19937 rng_;
  float radius_ = 0.0f;
  double extent_ = 0.0;
  std::vector<double> fluid_xyz_;
  std::vector<double> velocity_;
  std::vector<double> boundary_xyz_;
};

} // namespace

TEST_CASE("an SPH-shaped multi-frame run stays exact") {
  SphStress stress(
      static_cast<uint32_t>(env_int("APFRNN_SPH_STRESS_SEED", 1)));
  stress.run(env_int("APFRNN_SPH_STRESS_FLUID", 120000),
             env_int("APFRNN_SPH_STRESS_FRAMES", 24));
}
