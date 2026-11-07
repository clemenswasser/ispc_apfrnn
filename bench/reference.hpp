#pragma once

// Independent reference checksum for benchmark scenes.
//
// The benchmark harness must not trust any backend to define "correct". This
// builds a hashed uniform grid from the scene itself and enumerates every pair
// whose distance is clearly inside the radius. The resulting checksum and pair
// count are compared against each backend's reported lists. Pairs inside the
// float rounding band are counted separately because backends may legitimately
// disagree about them.

#include "scenes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace apfrnn::bench {

struct ReferenceSummary {
  bool complete = false;
  int64_t exact_pairs = 0; // pairs clearly inside the radius
  int64_t band_pairs = 0;  // pairs within the float rounding band
  uint64_t checksum = 0;   // of exact_pairs, backend-independent
};

namespace reference_detail {

inline uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

struct Cell {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;
  bool operator==(const Cell &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct CellHash {
  std::size_t operator()(const Cell &cell) const {
    uint64_t h = static_cast<uint64_t>(cell.x) * 0x9e3779b97f4a7c15ULL;
    h ^= static_cast<uint64_t>(cell.y) * 0xc2b2ae3d27d4eb4fULL + (h << 6);
    h ^= static_cast<uint64_t>(cell.z) * 0x165667b19e3779f9ULL + (h >> 2);
    return static_cast<std::size_t>(h);
  }
};

class Grid {
public:
  Grid(const float *xyz, int count, float radius) {
    float ox = std::numeric_limits<float>::infinity();
    float oy = ox;
    float oz = ox;
    if (count <= 0) {
      origin_[0] = 0.0f;
      origin_[1] = 0.0f;
      origin_[2] = 0.0f;
      radius_ = radius;
      return;
    }
    for (int index = 0; index < count; ++index) {
      ox = std::min(ox, xyz[3 * index + 0]);
      oy = std::min(oy, xyz[3 * index + 1]);
      oz = std::min(oz, xyz[3 * index + 2]);
    }
    origin_[0] = ox;
    origin_[1] = oy;
    origin_[2] = oz;
    radius_ = radius;
    for (int index = 0; index < count; ++index) {
      cells_[cell_of(xyz + 3 * index)].push_back(index);
    }
  }

  const std::vector<int> *find_cell(int64_t x, int64_t y, int64_t z) const {
    const auto found = cells_.find(Cell{x, y, z});
    return found == cells_.end() ? nullptr : &found->second;
  }

  Cell cell_of(const float *point) const {
    return {static_cast<int64_t>(
                std::floor((static_cast<double>(point[0]) - origin_[0]) /
                           radius_)),
            static_cast<int64_t>(
                std::floor((static_cast<double>(point[1]) - origin_[1]) /
                           radius_)),
            static_cast<int64_t>(
                std::floor((static_cast<double>(point[2]) - origin_[2]) /
                           radius_))};
  }

private:
  std::unordered_map<Cell, std::vector<int>, CellHash> cells_;
  float origin_[3] = {0.0f, 0.0f, 0.0f};
  float radius_ = 1.0f;
};

} // namespace reference_detail

// Returns complete=false when the scene is larger than max_points; the caller
// then falls back to cross-backend consensus only.
inline ReferenceSummary reference_summary(const SceneData &scene,
                                          int max_points = 1500000) {
  ReferenceSummary summary;
  int total_points = 0;
  for (const std::vector<float> &xyz : scene.xyz) {
    total_points += static_cast<int>(xyz.size() / 3);
  }
  if (total_points > max_points || scene.radius <= 0.0f) {
    return summary;
  }

  float max_abs = scene.radius;
  for (const std::vector<float> &xyz : scene.xyz) {
    for (float value : xyz) {
      max_abs = std::max(max_abs, std::fabs(value));
    }
  }
  const double radius = scene.radius;
  const double band =
      64.0 * std::numeric_limits<float>::epsilon() * max_abs;

  // One grid per target set, built lazily because a set may be used by
  // several active pairs.
  std::vector<std::unique_ptr<reference_detail::Grid>> grids(scene.xyz.size());

  for (std::size_t pair_index = 0; pair_index < scene.pairs.size();
       ++pair_index) {
    const auto [query, target] = scene.pairs[pair_index];
    const std::vector<float> &q_xyz = scene.xyz[query];
    const int q_count = scene.points(query);
    if (grids[static_cast<std::size_t>(target)] == nullptr) {
      grids[static_cast<std::size_t>(target)] =
          std::make_unique<reference_detail::Grid>(
              scene.xyz[static_cast<std::size_t>(target)].data(),
              scene.points(target), scene.radius);
    }
    const auto &grid = *grids[static_cast<std::size_t>(target)];
    const float *t_xyz = scene.xyz[static_cast<std::size_t>(target)].data();

    for (int point = 0; point < q_count; ++point) {
      const float *q = q_xyz.data() + 3 * static_cast<std::size_t>(point);
      const uint64_t key =
          reference_detail::mix((static_cast<uint64_t>(pair_index) << 32) |
                                static_cast<uint32_t>(point));
      const reference_detail::Cell center = grid.cell_of(q);
      for (int64_t dx = -1; dx <= 1; ++dx) {
        for (int64_t dy = -1; dy <= 1; ++dy) {
          for (int64_t dz = -1; dz <= 1; ++dz) {
            const auto found =
                grid.find_cell(center.x + dx, center.y + dy, center.z + dz);
            if (found == nullptr) {
              continue;
            }
            for (int neighbor : *found) {
              if (query == target && neighbor == point) {
                continue;
              }
              const float *t = t_xyz + 3 * static_cast<std::size_t>(neighbor);
              const double ex = double(t[0]) - q[0];
              const double ey = double(t[1]) - q[1];
              const double ez = double(t[2]) - q[2];
              const double distance = std::sqrt(ex * ex + ey * ey + ez * ez);
              if (distance < radius - band) {
                summary.checksum +=
                    reference_detail::mix(key ^ static_cast<uint32_t>(neighbor));
                ++summary.exact_pairs;
              } else if (distance <= radius + band) {
                ++summary.band_pairs;
              }
            }
          }
        }
      }
    }
  }
  summary.complete = true;
  return summary;
}

} // namespace apfrnn::bench
