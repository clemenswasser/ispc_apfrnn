#pragma once

#include "point_cloud.hpp"
#include "apfrnn/neighbor_search.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace apfrnn::support {

// Candidate neighbors from a hashed grid of radius-sized cells, decided with
// the float formula of the brute-force helpers. Pairs within float rounding
// of the radius may be reported either way, so they are returned as `maybe`.
class GridReference {
public:
  GridReference(const PointCloud &target, float radius) : target_(target), r_(radius) {
    for (int i = 0; i < static_cast<int>(target.x.size()); ++i) {
      cells_[key(cell(target.x[i]), cell(target.y[i]), cell(target.z[i]))]
          .push_back(i);
    }
  }

  void query(float qx, float qy, float qz, int skip, std::vector<int> &must,
             std::vector<int> &maybe) const {
    must.clear();
    maybe.clear();
    const int64_t cx = cell(qx), cy = cell(qy), cz = cell(qz);
    const float r2 = r_ * r_;
    for (int64_t dx = -1; dx <= 1; ++dx) {
      for (int64_t dy = -1; dy <= 1; ++dy) {
        for (int64_t dz = -1; dz <= 1; ++dz) {
          const auto found = cells_.find(key(cx + dx, cy + dy, cz + dz));
          if (found == cells_.end()) {
            continue;
          }
          for (int i : found->second) {
            if (i == skip) {
              continue;
            }
            const float tx = target_.x[i], ty = target_.y[i], tz = target_.z[i];
            const float ex = qx - tx, ey = qy - ty, ez = qz - tz;
            const float d2 = ex * ex + ey * ey + ez * ez;
            const float scale = std::max(
                {std::fabs(qx), std::fabs(qy), std::fabs(qz), std::fabs(tx),
                 std::fabs(ty), std::fabs(tz), r_});
            const float band = 64.0f * FLT_EPSILON * scale * scale;
            if (d2 <= r2 - band) {
              must.push_back(i);
            } else if (d2 <= r2 + band) {
              maybe.push_back(i);
            }
          }
        }
      }
    }
  }

private:
  int64_t cell(float v) const {
    return static_cast<int64_t>(std::floor(static_cast<double>(v) / r_));
  }
  struct Key {
    int64_t x, y, z;
    bool operator==(const Key &other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key &k) const {
      uint64_t h = static_cast<uint64_t>(k.x) * 0x9e3779b97f4a7c15ULL;
      h ^= static_cast<uint64_t>(k.y) * 0xc2b2ae3d27d4eb4fULL + (h << 6);
      h ^= static_cast<uint64_t>(k.z) * 0x165667b19e3779f9ULL + (h >> 2);
      return static_cast<std::size_t>(h);
    }
  };
  static Key key(int64_t x, int64_t y, int64_t z) { return {x, y, z}; }

  const PointCloud &target_;
  float r_;
  std::unordered_map<Key, std::vector<int>, KeyHash> cells_;
};

// Checks every list of pair (query, target) against the reference. Returns an
// empty string when all lists match, or a description of the first mismatch.
inline std::string check_pair(const NSearch &search, int query, int target,
                              const PointCloud &query_cloud,
                              const PointCloud &target_cloud, float radius) {
  const GridReference reference(target_cloud, radius);
  std::vector<int> must;
  std::vector<int> maybe;
  std::vector<int> got;
  const int n = static_cast<int>(query_cloud.x.size());
  if (search.get_n_points(query) != n) {
    return "point count " + std::to_string(search.get_n_points(query)) +
           " instead of " + std::to_string(n);
  }
  for (int i = 0; i < n; ++i) {
    reference.query(query_cloud.x[i], query_cloud.y[i], query_cloud.z[i],
                    query == target ? i : -1, must, maybe);
    const NeighborList list = search.get_neighbors(query, target, i);
    got.assign(list.begin(), list.end());
    std::sort(got.begin(), got.end());
    std::sort(must.begin(), must.end());
    std::sort(maybe.begin(), maybe.end());
    std::vector<int> extra;
    std::set_difference(got.begin(), got.end(), must.begin(), must.end(),
                        std::back_inserter(extra));
    const bool has_must =
        std::includes(got.begin(), got.end(), must.begin(), must.end());
    const bool extra_ok =
        std::includes(maybe.begin(), maybe.end(), extra.begin(), extra.end());
    if (!has_must || !extra_ok) {
      std::ostringstream out;
      out << "point " << i << ": got " << got.size() << ", expected "
          << must.size() << " (+" << maybe.size() << " on the radius)";
      return out.str();
    }
  }
  return {};
}

} // namespace apfrnn::support
