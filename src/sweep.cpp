#include "sweep.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <numeric>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

namespace apfrnn::sweep {

namespace {

constexpr double kMinCellsPerPoint = 10000.0;
// A sweep scans an x-window of `2*r` for every query. It is only attractive
// while that window is short; uniformly sparse random clouds eventually put
// hundreds of points in the window and the grid/hash path wins. Keep the
// threshold well above the paper's 200k-scale sparse case, where the sweep is
// genuinely faster, and below the 2M+ regime where it degrades.
constexpr double kMaxSweepWindow = 96.0;
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
  const std::vector<float> &target_y = target.index->y;
  const std::vector<float> &target_z = target.index->z;
  const IntBuffer &target_order = target.index->order;
  const int target_count = static_cast<int>(target_x.size());

  auto scan_from = [&](int window_begin, int point, float qx, float qy,
                       float qz) {
    for (int j = window_begin; j < target_count; ++j) {
      const float tx = target_x[static_cast<std::size_t>(j)];
      if (tx > qx + radius) {
        break;
      }
      const float dx = tx - qx;
      const float dy = target_y[static_cast<std::size_t>(j)] - qy;
      const float dz = target_z[static_cast<std::size_t>(j)] - qz;
      if (dx * dx + dy * dy + dz * dz <= radius_sq) {
        const int other = target_order[static_cast<std::size_t>(j)];
        if (same_set && other == point) {
          continue;
        }
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
                query.index->order[static_cast<std::size_t>(k)], qx,
                query.index->y[static_cast<std::size_t>(k)],
                query.index->z[static_cast<std::size_t>(k)]);
    }
    return;
  }

  for (int point = begin; point < end; ++point) {
    const float qx = query.x[static_cast<std::size_t>(point)];
    const int window_begin = static_cast<int>(
        std::lower_bound(target_x.begin(), target_x.end(), qx - radius) -
        target_x.begin());
    scan_from(window_begin, point, qx,
              query.y[static_cast<std::size_t>(point)],
              query.z[static_cast<std::size_t>(point)]);
  }
}

} // namespace

void Index::clear() {
  x.clear();
  y.clear();
  z.clear();
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

double estimated_window(const Bounds &bounds, std::size_t num_points,
                        float radius) {
  const double r = static_cast<double>(radius);
  if (num_points == 0 || bounds.empty() || !(r > 0.0)) {
    return 0.0;
  }
  const double extent_x = static_cast<double>(bounds.max_x - bounds.min_x) + r;
  return static_cast<double>(num_points) * 2.0 * r / extent_x;
}

bool should_use(const std::vector<float> &x, const std::vector<float> &y,
                const std::vector<float> &z, const Bounds &bounds,
                float radius) {
  const std::size_t num_points = x.size();
  // Large sparse sets: the sparse grid now has an incremental order merge,
  // a cursor stencil, and a log-time cell-range cross, so a fresh x-sort is
  // no longer the cheaper option.
  if (num_points >= 1000000) {
    return false;
  }
  if (estimated_cells_per_point(bounds, num_points, radius) <=
      kMinCellsPerPoint) {
    return false;
  }
  const double extent_x =
      static_cast<double>(bounds.max_x - bounds.min_x) + radius;
  if (extent_x > 0.0 &&
      static_cast<double>(num_points) * 2.0 * radius / extent_x >
          kMaxSweepWindow) {
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

void adopt_sorted_order(Index &index, const std::vector<float> &x,
                        const std::vector<float> &y,
                        const std::vector<float> &z, const IntBuffer &order,
                        const std::vector<uint64_t> &cell_keys) {
  const std::size_t count = x.size();
  if (y.size() != count || z.size() != count || order.size() != count ||
      cell_keys.size() != count) {
    index.clear();
    return;
  }
  index.x = x;
  index.y = y;
  index.z = z;
  index.order = order;
  index.valid = true;
  if (count < 2) {
    return;
  }

  // Signed cell-x of a packed key. The grid order is sorted by (cell x,
  // cell y, cell z), so equal cell-x runs are contiguous.
  auto cell_x = [](uint64_t key) {
    int value = static_cast<int>((key >> 42) & 0x1FFFFFULL);
    if ((value & 0x100000) != 0) {
      value |= ~0x1FFFFF;
    }
    return value;
  };
  std::vector<int> slab;
  slab.reserve(static_cast<std::size_t>(count) / 64 + 2);
  slab.push_back(0);
  int previous = cell_x(cell_keys[0]);
  for (std::size_t i = 1; i < count; ++i) {
    const int current = cell_x(cell_keys[i]);
    if (current != previous) {
      slab.push_back(static_cast<int>(i));
      previous = current;
    }
  }
  slab.push_back(static_cast<int>(count));

  struct SlabPoint {
    float x;
    float y;
    float z;
    int index;
  };
  const int num_slabs = static_cast<int>(slab.size()) - 1;
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_slabs, 8),
      [&](const tbb::blocked_range<int> &range) {
        std::vector<SlabPoint> points;
        for (int s = range.begin(); s < range.end(); ++s) {
          const int begin = slab[static_cast<std::size_t>(s)];
          const int end = slab[static_cast<std::size_t>(s) + 1];
          const std::size_t length = static_cast<std::size_t>(end - begin);
          if (length < 2) {
            continue;
          }
          points.resize(length);
          for (std::size_t i = 0; i < length; ++i) {
            const std::size_t at = static_cast<std::size_t>(begin) + i;
            points[i] = {index.x[at], index.y[at], index.z[at],
                         index.order[at]};
          }
          std::sort(points.begin(), points.end(),
                    [](const SlabPoint &a, const SlabPoint &b) {
                      if (a.x != b.x) {
                        return a.x < b.x;
                      }
                      return a.index < b.index;
                    });
          for (std::size_t i = 0; i < length; ++i) {
            const std::size_t at = static_cast<std::size_t>(begin) + i;
            index.x[at] = points[i].x;
            index.y[at] = points[i].y;
            index.z[at] = points[i].z;
            index.order[at] = points[i].index;
          }
        }
      });
}

void ensure_index(Index &index, const std::vector<float> &x,
                  const std::vector<float> &y, const std::vector<float> &z) {
  if (index.valid) {
    return;
  }
  const int num_points = static_cast<int>(x.size());
  if (y.size() != x.size() || z.size() != x.size()) {
    index.clear();
    return;
  }
  index.order.resize(static_cast<std::size_t>(num_points));
  std::iota(index.order.begin(), index.order.end(), 0);
  tbb::parallel_sort(index.order.begin(), index.order.end(),
                     [&x](int a, int b) {
                       const float xa = x[static_cast<std::size_t>(a)];
                       const float xb = x[static_cast<std::size_t>(b)];
                       return xa < xb || (xa == xb && a < b);
                     });
  index.x.resize(static_cast<std::size_t>(num_points));
  index.y.resize(static_cast<std::size_t>(num_points));
  index.z.resize(static_cast<std::size_t>(num_points));
  for (int i = 0; i < num_points; ++i) {
    const int source = index.order[static_cast<std::size_t>(i)];
    index.x[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(source)];
    index.y[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(source)];
    index.z[static_cast<std::size_t>(i)] = z[static_cast<std::size_t>(source)];
  }
  index.valid = true;
}

bool write_fixed_rows(const Cloud &query, const IntBuffer &query_rows,
                      const Cloud &target, bool same_set, float radius,
                      int64_t memory_budget, IntBuffer &neighbors,
                      IntBuffer &counts, int &row_stride) {
  const int num_points = static_cast<int>(query.x.size());

  double expected_window = 0.0;
  if (target.index != nullptr && target.index->valid &&
      !target.index->x.empty()) {
    const float extent = target.index->x.back() - target.index->x.front();
    if (extent > 0.0f) {
      expected_window = static_cast<double>(target.index->x.size()) *
                        2.0 * radius / extent;
    }
  }
  if (expected_window >= 32.0 && query.index != nullptr &&
      query.index->valid && target.index != nullptr && target.index->valid &&
      query.index->order.size() == static_cast<std::size_t>(num_points) &&
      query_rows.size() == static_cast<std::size_t>(num_points)) {
    const int target_points = static_cast<int>(target.index->order.size());
    IntBuffer lo(static_cast<std::size_t>(num_points));
    IntBuffer hi(static_cast<std::size_t>(num_points));
    const auto target_x = target.index->x.begin();
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_points, 16384),
        [&](const tbb::blocked_range<int> &range) {
          const int begin = range.begin();
          const int end = range.end();
          int lower = static_cast<int>(
              std::lower_bound(target_x, target_x + target_points,
                               query.index->x[static_cast<std::size_t>(begin)] -
                                   radius) -
              target_x);
          int upper = lower;
          for (int k = begin; k < end; ++k) {
            const float qx = query.index->x[static_cast<std::size_t>(k)];
            while (lower < target_points &&
                   target.index->x[static_cast<std::size_t>(lower)] <
                       qx - radius) {
              ++lower;
            }
            upper = std::max(upper, lower);
            while (upper < target_points &&
                   target.index->x[static_cast<std::size_t>(upper)] <=
                       qx + radius) {
              ++upper;
            }
            lo[static_cast<std::size_t>(k)] = lower;
            hi[static_cast<std::size_t>(k)] = upper;
          }
        });
    counts.resize(static_cast<std::size_t>(num_points));
    IntBuffer row_query(static_cast<std::size_t>(num_points));
    const float radius_sq = radius * radius;
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_points, 256),
        [&](const tbb::blocked_range<int> &range) {
          for (int k = range.begin(); k < range.end(); ++k) {
            const int original =
                query.index->order[static_cast<std::size_t>(k)];
            const float qx = query.index->x[static_cast<std::size_t>(k)];
            const float qy = query.index->y[static_cast<std::size_t>(k)];
            const float qz = query.index->z[static_cast<std::size_t>(k)];
            int count = 0;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
            for (int j = lo[static_cast<std::size_t>(k)];
                 j < hi[static_cast<std::size_t>(k)]; ++j) {
              const float dx =
                  target.index->x[static_cast<std::size_t>(j)] - qx;
              const float dy =
                  target.index->y[static_cast<std::size_t>(j)] - qy;
              const float dz =
                  target.index->z[static_cast<std::size_t>(j)] - qz;
              if (dx * dx + dy * dy + dz * dz <= radius_sq) {
                ++count;
              }
            }
            if (same_set && count > 0) {
              --count;  // the query point is always in its own target set
            }
            const int row = query_rows[static_cast<std::size_t>(original)];
            counts[static_cast<std::size_t>(row)] = count;
            row_query[static_cast<std::size_t>(row)] = k;
          }
        });

    int max_count = 0;
    for (int count : counts) {
      max_count = std::max(max_count, count);
    }
    const int stride = std::max(1, max_count);
    const int64_t bytes = static_cast<int64_t>(num_points) *
                          static_cast<int64_t>(stride) *
                          static_cast<int64_t>(sizeof(int));
    if (max_count > kMaxRowCapacity || bytes > memory_budget) {
      return false;
    }
    neighbors.resize(static_cast<std::size_t>(num_points) *
                     static_cast<std::size_t>(stride));
    row_stride = stride;
    if (max_count == 0) {
      return true;
    }
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_points, 4096),
        [&](const tbb::blocked_range<int> &range) {
          for (int row = range.begin(); row < range.end(); ++row) {
            const int count = counts[static_cast<std::size_t>(row)];
            if (count == 0) {
              continue;
            }
            const int k = row_query[static_cast<std::size_t>(row)];
            const int original =
                query.index->order[static_cast<std::size_t>(k)];
            const float qx = query.index->x[static_cast<std::size_t>(k)];
            const float qy = query.index->y[static_cast<std::size_t>(k)];
            const float qz = query.index->z[static_cast<std::size_t>(k)];
            int *out = neighbors.data() +
                       static_cast<std::size_t>(row) *
                           static_cast<std::size_t>(stride);
            int slot = 0;
            for (int j = lo[static_cast<std::size_t>(k)];
                 j < hi[static_cast<std::size_t>(k)]; ++j) {
              const float dx =
                  target.index->x[static_cast<std::size_t>(j)] - qx;
              const float dy =
                  target.index->y[static_cast<std::size_t>(j)] - qy;
              const float dz =
                  target.index->z[static_cast<std::size_t>(j)] - qz;
              if (dx * dx + dy * dy + dz * dz <= radius_sq) {
                const int other =
                    target.index->order[static_cast<std::size_t>(j)];
                if (same_set && other == original) {
                  continue;
                }
                out[slot++] = other;
              }
            }
          }
        });
    return true;
  }

  // Scalar fallback for callers without x-sorted indexes.
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
