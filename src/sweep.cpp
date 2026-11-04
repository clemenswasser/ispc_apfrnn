#include "sweep.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

namespace apfrnn::sweep {

namespace {

constexpr double kMinCellsPerPoint = 10000.0;
constexpr int kMaxRowCapacity = 65536;

// A huge bounding box only makes the sweep attractive when its points are
// spread through it. Dense clusters still fill every x-window, so sample
// coarse cells and look for repeated occupancy. Small sets skip the sample:
// their search cost is negligible either way.
constexpr int kClusterMinPoints = 4096;
constexpr int kClusterMinSamples = 2048;
constexpr int kClusterMaxSamples = 32768;
constexpr float kClusterCellScale = 8.0f;
constexpr int kClusterRunThreshold = 4;
constexpr double kClusterUniqueRatio = 0.4;

uint64_t cluster_cell_key(float x, float y, float z, float cell) {
  const double inverse = 1.0 / static_cast<double>(cell);
  const int64_t ix =
      static_cast<int64_t>(std::floor(static_cast<double>(x) * inverse));
  const int64_t iy =
      static_cast<int64_t>(std::floor(static_cast<double>(y) * inverse));
  const int64_t iz =
      static_cast<int64_t>(std::floor(static_cast<double>(z) * inverse));
  uint64_t hash = static_cast<uint64_t>(ix) * 0x9e3779b97f4a7c15ull;
  hash ^= static_cast<uint64_t>(iy) * 0xc2b2ae3d27d4eb4full;
  hash ^= static_cast<uint64_t>(iz) * 0x165667b19e3779f9ull;
  hash ^= hash >> 33;
  hash *= 0xff51afd7ed558ccdull;
  hash ^= hash >> 33;
  return hash;
}

// Visits the query points [begin, end) of the query's visiting order. For each
// query, calls visit(other) for every target point within radius and then
// done(row), where row is the query's output row.
template <typename Visit, typename Done>
void scan(const Cloud &query, const IntBuffer &query_rows, const Cloud &target,
          bool same_set, float radius, int begin, int end, Visit &&visit,
          Done &&done) {
  const float radius_sq = radius * radius;
  const std::vector<float> &target_x = target.index->x;
  const IntBuffer &target_order = target.index->order;
  const int target_count = static_cast<int>(target_x.size());

  auto scan_from = [&](int window_begin, int point, float qx) {
    const float qy = query.y[static_cast<std::size_t>(point)];
    const float qz = query.z[static_cast<std::size_t>(point)];
    for (int j = window_begin; j < target_count; ++j) {
      if (target_x[static_cast<std::size_t>(j)] > qx + radius) {
        break;
      }
      const int other = target_order[static_cast<std::size_t>(j)];
      if (same_set && other == point) {
        continue;
      }
      const float dx = target.x[static_cast<std::size_t>(other)] - qx;
      const float dy = target.y[static_cast<std::size_t>(other)] - qy;
      const float dz = target.z[static_cast<std::size_t>(other)] - qz;
      if (dx * dx + dy * dy + dz * dz <= radius_sq) {
        visit(other);
      }
    }
    done(query_rows[static_cast<std::size_t>(point)]);
  };

  if (query.index != nullptr && query.index->valid) {
    // Queries are visited in x order, so the window start only moves
    // forward. One binary search per chunk replaces one per point.
    const std::vector<float> &query_x = query.index->x;
    int window_begin = static_cast<int>(
        std::lower_bound(target_x.begin(), target_x.end(),
                         query_x[static_cast<std::size_t>(begin)] - radius) -
        target_x.begin());
    for (int k = begin; k < end; ++k) {
      const float qx = query_x[static_cast<std::size_t>(k)];
      while (window_begin < target_count &&
             target_x[static_cast<std::size_t>(window_begin)] < qx - radius) {
        ++window_begin;
      }
      scan_from(window_begin,
                query.index->order[static_cast<std::size_t>(k)], qx);
    }
    return;
  }

  for (int point = begin; point < end; ++point) {
    const float qx = query.x[static_cast<std::size_t>(point)];
    const int window_begin = static_cast<int>(
        std::lower_bound(target_x.begin(), target_x.end(), qx - radius) -
        target_x.begin());
    scan_from(window_begin, point, qx);
  }
}

} // namespace

void Index::clear() {
  x.clear();
  order.clear();
  valid = false;
}

double estimated_cells_per_point(const Bounds &bounds, std::size_t num_points,
                                 float radius) {
  const double r = static_cast<double>(radius);
  if (num_points == 0 || bounds.empty() || !(r > 0.0)) {
    return 0.0;
  }
  const double extent_x = static_cast<double>(bounds.max_x - bounds.min_x) + r;
  const double extent_y = static_cast<double>(bounds.max_y - bounds.min_y) + r;
  const double extent_z = static_cast<double>(bounds.max_z - bounds.min_z) + r;
  const double cells = (extent_x * extent_y * extent_z) / (r * r * r);
  return cells / static_cast<double>(num_points);
}

bool should_use(const std::vector<float> &x, const std::vector<float> &y,
                const std::vector<float> &z, const Bounds &bounds,
                float radius) {
  const std::size_t num_points = x.size();
  if (estimated_cells_per_point(bounds, num_points, radius) <=
      kMinCellsPerPoint) {
    return false;
  }
  if (num_points < static_cast<std::size_t>(kClusterMinPoints) ||
      y.size() != num_points || z.size() != num_points || !(radius > 0.0f)) {
    // Preserve the old choice for small or malformed sets.
    return true;
  }

  const std::size_t wanted =
      std::max(static_cast<std::size_t>(kClusterMinSamples), num_points / 64);
  const std::size_t samples =
      std::min(num_points,
               std::min(wanted, static_cast<std::size_t>(kClusterMaxSamples)));
  std::size_t capacity = 1;
  while (capacity < samples * 2) {
    capacity <<= 1;
  }
  std::vector<uint64_t> keys(capacity, 0);
  std::vector<int> counts(capacity, 0);
  const std::size_t mask = capacity - 1;
  const float cell = kClusterCellScale * radius;
  int max_run = 0;
  int unique = 0;
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const std::size_t index = static_cast<std::size_t>(
        static_cast<unsigned long long>(sample) * num_points / samples);
    uint64_t key = cluster_cell_key(x[index], y[index], z[index], cell);
    if (key == 0) {
      key = 1; // zero marks an empty slot
    }
    std::size_t slot = static_cast<std::size_t>(key) & mask;
    while (keys[slot] != 0 && keys[slot] != key) {
      slot = (slot + 1) & mask;
    }
    if (keys[slot] == 0) {
      keys[slot] = key;
      ++unique;
    }
    ++counts[slot];
    max_run = std::max(max_run, counts[slot]);
  }

  // A few points per coarse cell (pairs, triplets) still leave the sweep
  // cheap. Dense clusters put many samples into the same cell.
  const double unique_ratio =
      static_cast<double>(unique) / static_cast<double>(samples);
  return max_run < kClusterRunThreshold && unique_ratio >= kClusterUniqueRatio;
}

void ensure_index(Index &index, const std::vector<float> &x) {
  if (index.valid) {
    return;
  }
  const int num_points = static_cast<int>(x.size());
  index.order.resize(static_cast<std::size_t>(num_points));
  std::iota(index.order.begin(), index.order.end(), 0);
  tbb::parallel_sort(index.order.begin(), index.order.end(),
                     [&x](int a, int b) {
                       const float xa = x[static_cast<std::size_t>(a)];
                       const float xb = x[static_cast<std::size_t>(b)];
                       return xa < xb || (xa == xb && a < b);
                     });
  index.x.resize(static_cast<std::size_t>(num_points));
  for (int i = 0; i < num_points; ++i) {
    index.x[static_cast<std::size_t>(i)] =
        x[static_cast<std::size_t>(index.order[static_cast<std::size_t>(i)])];
  }
  index.valid = true;
}

bool write_fixed_rows(const Cloud &query, const IntBuffer &query_rows,
                      const Cloud &target, bool same_set, float radius,
                      int64_t memory_budget, IntBuffer &neighbors,
                      IntBuffer &counts, int &row_stride) {
  const int num_points = static_cast<int>(query.x.size());

  IntBuffer row_counts(static_cast<std::size_t>(num_points));
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, 256),
      [&](const tbb::blocked_range<int> &range) {
        int count = 0;
        scan(
            query, query_rows, target, same_set, radius, range.begin(),
            range.end(), [&](int) { ++count; },
            [&](int row) {
              row_counts[static_cast<std::size_t>(row)] = count;
              count = 0;
            });
      });
  int max_count = 0;
  for (int count : row_counts) {
    max_count = std::max(max_count, count);
  }

  const int stride = std::max(1, max_count);
  const int64_t bytes = static_cast<int64_t>(num_points) *
                        static_cast<int64_t>(stride) *
                        static_cast<int64_t>(sizeof(int));
  if (max_count > kMaxRowCapacity || bytes > memory_budget) {
    return false;
  }

  counts = std::move(row_counts);
  neighbors.resize(static_cast<std::size_t>(num_points) *
                   static_cast<std::size_t>(stride));
  row_stride = stride;
  if (max_count == 0) {
    return true;
  }
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, 128),
      [&](const tbb::blocked_range<int> &range) {
        IntBuffer local;
        scan(
            query, query_rows, target, same_set, radius, range.begin(),
            range.end(), [&](int other) { local.push_back(other); },
            [&](int row) {
              if (!local.empty()) {
                std::memcpy(neighbors.data() +
                                static_cast<std::size_t>(row) *
                                    static_cast<std::size_t>(stride),
                            local.data(), local.size() * sizeof(int));
              }
              local.clear();
            });
      });
  return true;
}

} // namespace apfrnn::sweep
