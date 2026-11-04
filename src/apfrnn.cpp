#include "apfrnn.hpp"

#include "kernel_ispc.h"

#include <algorithm>
#include <cassert>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_arena.h>

namespace apfrnn {

inline int dense_cache_position(const NeighborSearchData &data, uint64_t key);

const int *neighbor_permutation_table() {
  struct NeighborPermutationTable {
    alignas(32) int values[256 * 8];
    NeighborPermutationTable() {
      for (int mask = 0; mask < 256; ++mask) {
        int count = 0;
        for (int bit = 0; bit < 8; ++bit) {
          if ((mask & (1 << bit)) != 0) {
            values[mask * 8 + count++] = bit;
          }
        }
        while (count < 8) {
          values[mask * 8 + count++] = 0;
        }
      }
    }
  };
  static const NeighborPermutationTable table;
  return table.values;
}

namespace {

constexpr int kGrain = 64;

// Small grids otherwise collapse to one or two TBB tasks.
inline int cell_grain(int count) {
  return std::max(1, std::min(kGrain, count / 32));
}

template <typename Function>
inline void parallel_cells(int begin, int end, int grain, Function function) {
  tbb::parallel_for(tbb::blocked_range<int>(begin, end, grain),
                    [&](const tbb::blocked_range<int> &range) {
                      function(range.begin(), range.end());
                    });
}

inline int max_row_count(const IntBuffer &counts) {
  int max_count = 0;
  for (int count : counts) {
    max_count = std::max(max_count, count);
  }
  return max_count;
}

// Sizes the fixed-row buffers. The writers rewrite every row, so the old
// contents never need to survive: growing drops the old buffer first instead
// of letting resize() copy it, and shrinking reuses the already-faulted pages
// unless the rows fell below half the buffer.
inline void prepare_fixed_rows(IntBuffer &values, IntBuffer &counts,
                               std::size_t num_points, int row_stride) {
  const std::size_t required = num_points * static_cast<std::size_t>(row_stride);
  counts.resize(num_points);
  if (required > values.capacity() || 2 * required < values.capacity()) {
    IntBuffer().swap(values);
  }
  values.resize(required);
}

// Sizes the block-row buffers. The layout has already been computed in
// row_offset/row_capacity, so only the total slot count is needed.
inline void prepare_block_rows(IntBuffer &values, IntBuffer &counts,
                               std::size_t num_points,
                               const IntBuffer &row_offset,
                               const IntBuffer &row_capacity) {
  counts.resize(num_points);
  std::size_t required = 0;
  if (num_points > 0) {
    required = static_cast<std::size_t>(
        row_offset[num_points - 1] + row_capacity[num_points - 1] + 8);
  }
#ifndef NDEBUG
  // finish_row() uses non-temporal vector stores, which require 32-byte
  // alignment: every row start and stride must be a multiple of eight ints.
  for (std::size_t i = 0; i < num_points; ++i) {
    assert(row_offset[i] % 8 == 0);
    assert(row_capacity[i] % 8 == 0);
  }
#endif
  // Block rows exist to remove padding, so drop a larger fixed-row buffer even
  // when it is not quite twice the new size.
  if (required > values.capacity() || 4 * required < 3 * values.capacity()) {
    IntBuffer().swap(values);
  }
  values.resize(required);
}

inline uint64_t make_key(int cx, int cy, int cz) {
  uint64_t ux = static_cast<uint64_t>(static_cast<uint32_t>(cx)) & 0x1FFFFFULL;
  uint64_t uy = static_cast<uint64_t>(static_cast<uint32_t>(cy)) & 0x1FFFFFULL;
  uint64_t uz = static_cast<uint64_t>(static_cast<uint32_t>(cz)) & 0x1FFFFFULL;
  return (ux << 42) | (uy << 21) | uz;
}

inline int floor_to_cell(float value) {
  // Cell coordinates are packed into 21-bit fields by make_key(). Clamp
  // non-finite values and values outside that representable range instead of
  // invoking undefined behavior in the float-to-int conversion. Clamping only
  // increases candidate work; the exact distance test still decides neighbors.
  constexpr int kCellMin = -0x100000;
  constexpr int kCellMax = 0xFFFFF;
  if (!std::isfinite(value)) {
    return value < 0.0f ? kCellMin : kCellMax;
  }
  if (value >= static_cast<float>(kCellMax)) {
    return kCellMax;
  }
  if (value <= static_cast<float>(kCellMin)) {
    return kCellMin;
  }
  int cell = static_cast<int>(value);
  return cell - static_cast<int>(cell > value);
}

inline int unpack_coord(uint64_t key, int shift) {
  int coord = static_cast<int>((key >> shift) & 0x1FFFFFULL);
  if ((coord & 0x100000) != 0) {
    coord |= ~0x1FFFFF;
  }
  return coord;
}

constexpr int kRadixBits = 8;
constexpr int kRadixBuckets = 1 << kRadixBits;
constexpr int kRadixChunks = 16;
constexpr int kRadixMinPoints = 16384;

// Stable LSD radix sort by the 64-bit cell key. The input is in original
// index order, so equal keys keep that order and the comparator's index
// tie-break is preserved. For the sparse droplets domain the comparison sort
// spends ~42% of the branch misses; this sort moves that work into a
// branch-free histogram/scatter.
void radix_sort_point_keys(std::vector<SortPointKey> &keys,
                           std::vector<SortPointKey> &scratch) {
  const int num_points = static_cast<int>(keys.size());
  if (num_points < kRadixMinPoints) {
    tbb::parallel_sort(keys.begin(), keys.end());
    return;
  }

  scratch.resize(keys.size());
  std::vector<int> histogram(static_cast<std::size_t>(kRadixChunks) *
                             kRadixBuckets);
  std::vector<int> offsets(static_cast<std::size_t>(kRadixChunks) *
                           kRadixBuckets);
  SortPointKey *src = keys.data();
  SortPointKey *dst = scratch.data();
  for (int pass = 0; pass < 8; ++pass) {
    std::fill(histogram.begin(), histogram.end(), 0);
    tbb::parallel_for(0, kRadixChunks, [&](int chunk) {
      const int begin = static_cast<int>(
          static_cast<long long>(chunk) * num_points / kRadixChunks);
      const int end = static_cast<int>(
          static_cast<long long>(chunk + 1) * num_points / kRadixChunks);
      int *row = histogram.data() +
                 static_cast<std::size_t>(chunk) * kRadixBuckets;
      for (int i = begin; i < end; ++i) {
        const int digit =
            static_cast<int>((src[i].key >> (pass * kRadixBits)) &
                             (kRadixBuckets - 1));
        ++row[digit];
      }
    });

    int total = 0;
    for (int bucket = 0; bucket < kRadixBuckets; ++bucket) {
      int running = total;
      for (int chunk = 0; chunk < kRadixChunks; ++chunk) {
        const std::size_t slot =
            static_cast<std::size_t>(chunk) * kRadixBuckets + bucket;
        offsets[slot] = running;
        running += histogram[slot];
      }
      total = running;
    }

    tbb::parallel_for(0, kRadixChunks, [&](int chunk) {
      const int begin = static_cast<int>(
          static_cast<long long>(chunk) * num_points / kRadixChunks);
      const int end = static_cast<int>(
          static_cast<long long>(chunk + 1) * num_points / kRadixChunks);
      int *row = offsets.data() +
                 static_cast<std::size_t>(chunk) * kRadixBuckets;
      for (int i = begin; i < end; ++i) {
        const SortPointKey value = src[i];
        const int digit =
            static_cast<int>((value.key >> (pass * kRadixBits)) &
                             (kRadixBuckets - 1));
        dst[row[digit]++] = value;
      }
    });
    std::swap(src, dst);
  }
  if (src != keys.data()) {
    std::copy(src, src + num_points, keys.data());
  }
}

inline void sort_point_set(
    const std::vector<float> &x, const std::vector<float> &y,
    const std::vector<float> &z, float inv_radius, std::vector<float> &sorted_x,
    std::vector<float> &sorted_y, std::vector<float> &sorted_z,
    IntBuffer &original_index, IntBuffer &cell_starts, IntBuffer &cell_ends,
    std::vector<SortPointKey> &point_keys,
    std::vector<SortPointKey> &point_keys_alt,
    std::vector<uint64_t> &unique_keys) {
  int const num_points = static_cast<int>(x.size());
  point_keys.resize(num_points);

  struct CellBounds {
    int min_cx = INT_MAX;
    int min_cy = INT_MAX;
    int min_cz = INT_MAX;
    int max_cx = INT_MIN;
    int max_cy = INT_MIN;
    int max_cz = INT_MIN;
  };
  const CellBounds bounds = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, num_points, 2048), CellBounds{},
      [&](const tbb::blocked_range<int> &range, CellBounds acc) {
        for (int i = range.begin(); i < range.end(); ++i) {
          const int cx = floor_to_cell(x[static_cast<std::size_t>(i)] *
                                       inv_radius);
          const int cy = floor_to_cell(y[static_cast<std::size_t>(i)] *
                                       inv_radius);
          const int cz = floor_to_cell(z[static_cast<std::size_t>(i)] *
                                       inv_radius);
          point_keys[static_cast<std::size_t>(i)].key =
              make_key(cx, cy, cz);
          point_keys[static_cast<std::size_t>(i)].index = i;
          acc.min_cx = std::min(acc.min_cx, cx);
          acc.min_cy = std::min(acc.min_cy, cy);
          acc.min_cz = std::min(acc.min_cz, cz);
          acc.max_cx = std::max(acc.max_cx, cx);
          acc.max_cy = std::max(acc.max_cy, cy);
          acc.max_cz = std::max(acc.max_cz, cz);
        }
        return acc;
      },
      [](CellBounds a, const CellBounds &b) {
        a.min_cx = std::min(a.min_cx, b.min_cx);
        a.min_cy = std::min(a.min_cy, b.min_cy);
        a.min_cz = std::min(a.min_cz, b.min_cz);
        a.max_cx = std::max(a.max_cx, b.max_cx);
        a.max_cy = std::max(a.max_cy, b.max_cy);
        a.max_cz = std::max(a.max_cz, b.max_cz);
        return a;
      });

  // For bounded domains, a dense counting sort by cell is O(N + cells) and
  // avoids the comparison sort. Fall back to parallel_sort for large/sparse
  // domains.
  constexpr int64_t kCountingCellLimit = 2 * 1024 * 1024;
  const int min_cx = bounds.min_cx;
  const int min_cy = bounds.min_cy;
  const int min_cz = bounds.min_cz;
  const int max_cx = bounds.max_cx;
  const int max_cy = bounds.max_cy;
  const int max_cz = bounds.max_cz;

  const int64_t nx = static_cast<int64_t>(max_cx) - min_cx + 1;
  const int64_t ny = static_cast<int64_t>(max_cy) - min_cy + 1;
  const int64_t nz = static_cast<int64_t>(max_cz) - min_cz + 1;
  const int64_t total_cells = nx * ny * nz;
  // Dense cell domains use a parallel counting sort: a histogram, a prefix
  // sum, a compact cell enumeration, and an atomic scatter. Sparse domains
  // still use parallel_sort, which has better work scaling there.
  const bool dense_domain =
      nx > 0 && ny > 0 && nz > 0 && nx <= kCountingCellLimit &&
      ny <= kCountingCellLimit && nz <= kCountingCellLimit &&
      nx * ny <= kCountingCellLimit && total_cells <= kCountingCellLimit &&
      total_cells <= 4LL * num_points;
  if (dense_domain) {
    constexpr int64_t kCellGrain = 4096;
    std::vector<std::atomic<int>> counts(
        static_cast<std::size_t>(total_cells));
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, total_cells, kCellGrain),
        [&](const tbb::blocked_range<int64_t> &range) {
          for (int64_t cell = range.begin(); cell < range.end(); ++cell) {
            counts[static_cast<std::size_t>(cell)].store(
                0, std::memory_order_relaxed);
          }
        });

    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_points, 2048),
        [&](const tbb::blocked_range<int> &range) {
          for (int i = range.begin(); i < range.end(); ++i) {
            const uint64_t key = point_keys[static_cast<std::size_t>(i)].key;
            const int lx = unpack_coord(key, 42) - min_cx;
            const int ly = unpack_coord(key, 21) - min_cy;
            const int lz = unpack_coord(key, 0) - min_cz;
            const int64_t cell =
                (static_cast<int64_t>(lx) * ny + ly) * nz + lz;
            counts[static_cast<std::size_t>(cell)].fetch_add(
                1, std::memory_order_relaxed);
          }
        });

    std::vector<int> offsets(static_cast<std::size_t>(total_cells));
    tbb::parallel_scan(
        tbb::blocked_range<int64_t>(0, total_cells, kCellGrain), 0,
        [&](const tbb::blocked_range<int64_t> &range, int sum,
            bool is_final_scan) {
          int temp = sum;
          for (int64_t cell = range.begin(); cell < range.end(); ++cell) {
            if (is_final_scan) {
              offsets[static_cast<std::size_t>(cell)] = temp;
            }
            temp += counts[static_cast<std::size_t>(cell)].load(
                std::memory_order_relaxed);
          }
          return temp;
        },
        [](int left, int right) { return left + right; });

    const int num_nonzero = tbb::parallel_reduce(
        tbb::blocked_range<int64_t>(0, total_cells, kCellGrain), 0,
        [&](const tbb::blocked_range<int64_t> &range, int sum) {
          for (int64_t cell = range.begin(); cell < range.end(); ++cell) {
            if (counts[static_cast<std::size_t>(cell)].load(
                    std::memory_order_relaxed) > 0) {
              ++sum;
            }
          }
          return sum;
        },
        [](int left, int right) { return left + right; });

    cell_starts.resize(static_cast<std::size_t>(num_nonzero));
    cell_ends.resize(static_cast<std::size_t>(num_nonzero));
    unique_keys.resize(static_cast<std::size_t>(num_nonzero));

    tbb::parallel_scan(
        tbb::blocked_range<int64_t>(0, total_cells, kCellGrain), 0,
        [&](const tbb::blocked_range<int64_t> &range, int sum,
            bool is_final_scan) {
          int temp = sum;
          int64_t remaining = range.begin();
          int64_t lz = remaining % nz;
          remaining /= nz;
          int64_t ly = remaining % ny;
          int64_t lx = remaining / ny;
          for (int64_t cell = range.begin(); cell < range.end(); ++cell) {
            const int count = counts[static_cast<std::size_t>(cell)].load(
                std::memory_order_relaxed);
            if (count > 0) {
              if (is_final_scan) {
                const int start = offsets[static_cast<std::size_t>(cell)];
                cell_starts[static_cast<std::size_t>(temp)] = start;
                cell_ends[static_cast<std::size_t>(temp)] = start + count;
                unique_keys[static_cast<std::size_t>(temp)] =
                    make_key(min_cx + static_cast<int>(lx),
                             min_cy + static_cast<int>(ly),
                             min_cz + static_cast<int>(lz));
              }
              ++temp;
            }
            ++lz;
            if (lz == nz) {
              lz = 0;
              ++ly;
              if (ly == ny) {
                ly = 0;
                ++lx;
              }
            }
          }
          return temp;
        },
        [](int left, int right) { return left + right; });

    sorted_x.resize(num_points);
    sorted_y.resize(num_points);
    sorted_z.resize(num_points);
    original_index.resize(num_points);

    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_points, 2048),
        [&](const tbb::blocked_range<int> &range) {
          for (int i = range.begin(); i < range.end(); ++i) {
            const uint64_t key = point_keys[static_cast<std::size_t>(i)].key;
            const int lx = unpack_coord(key, 42) - min_cx;
            const int ly = unpack_coord(key, 21) - min_cy;
            const int lz = unpack_coord(key, 0) - min_cz;
            const int64_t cell =
                (static_cast<int64_t>(lx) * ny + ly) * nz + lz;
            const int position = __atomic_fetch_add(
                &offsets[static_cast<std::size_t>(cell)], 1,
                __ATOMIC_RELAXED);
            sorted_x[static_cast<std::size_t>(position)] =
                x[static_cast<std::size_t>(i)];
            sorted_y[static_cast<std::size_t>(position)] =
                y[static_cast<std::size_t>(i)];
            sorted_z[static_cast<std::size_t>(position)] =
                z[static_cast<std::size_t>(i)];
            original_index[static_cast<std::size_t>(position)] = i;
          }
        });

    // The atomic scatter orders points inside a cell by thread timing. Sort
    // each cell by original index so the packed order (and any block layout
    // derived from it) is deterministic across frames.
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_nonzero, 64),
        [&](const tbb::blocked_range<int> &range) {
          for (int cell = range.begin(); cell < range.end(); ++cell) {
            const int begin = cell_starts[static_cast<std::size_t>(cell)];
            const int end = cell_ends[static_cast<std::size_t>(cell)];
            const int count = end - begin;
            if (count < 2) {
              continue;
            }
            if (count <= 32) {
              for (int i = begin + 1; i < end; ++i) {
                int j = i;
                while (j > begin &&
                       original_index[static_cast<std::size_t>(j - 1)] >
                           original_index[static_cast<std::size_t>(j)]) {
                  std::swap(sorted_x[static_cast<std::size_t>(j - 1)],
                            sorted_x[static_cast<std::size_t>(j)]);
                  std::swap(sorted_y[static_cast<std::size_t>(j - 1)],
                            sorted_y[static_cast<std::size_t>(j)]);
                  std::swap(sorted_z[static_cast<std::size_t>(j - 1)],
                            sorted_z[static_cast<std::size_t>(j)]);
                  std::swap(original_index[static_cast<std::size_t>(j - 1)],
                            original_index[static_cast<std::size_t>(j)]);
                  --j;
                }
              }
            } else {
              struct CellPoint {
                int index;
                float x;
                float y;
                float z;
              };
              std::vector<CellPoint> points(static_cast<std::size_t>(count));
              for (int i = 0; i < count; ++i) {
                const std::size_t src = static_cast<std::size_t>(begin + i);
                points[static_cast<std::size_t>(i)] = {
                    original_index[src], sorted_x[src], sorted_y[src],
                    sorted_z[src]};
              }
              std::sort(points.begin(), points.end(),
                        [](const CellPoint &a, const CellPoint &b) {
                          return a.index < b.index;
                        });
              for (int i = 0; i < count; ++i) {
                const std::size_t dst = static_cast<std::size_t>(begin + i);
                sorted_x[dst] = points[static_cast<std::size_t>(i)].x;
                sorted_y[dst] = points[static_cast<std::size_t>(i)].y;
                sorted_z[dst] = points[static_cast<std::size_t>(i)].z;
                original_index[dst] =
                    points[static_cast<std::size_t>(i)].index;
              }
            }
          }
        });
    return;
  }

  radix_sort_point_keys(point_keys, point_keys_alt);

  sorted_x.resize(num_points);
  sorted_y.resize(num_points);
  sorted_z.resize(num_points);
  original_index.resize(num_points);

  tbb::parallel_for(0, num_points, [&](int i) {
    int idx = point_keys[i].index;
    sorted_x[i] = x[idx];
    sorted_y[i] = y[idx];
    sorted_z[i] = z[idx];
    original_index[i] = idx;
  });

  cell_starts.clear();
  cell_ends.clear();
  unique_keys.clear();
  cell_starts.reserve(num_points);
  cell_ends.reserve(num_points);
  unique_keys.reserve(num_points);
  cell_starts.push_back(0);
  unique_keys.push_back(point_keys[0].key);

  for (int i = 1; i < num_points; ++i) {
    if (point_keys[i].key != point_keys[i - 1].key) {
      cell_ends.push_back(i);
      cell_starts.push_back(i);
      unique_keys.push_back(point_keys[i].key);
    }
  }
  cell_ends.push_back(num_points);
}

inline void build_hash_table(const std::vector<uint64_t> &unique_keys,
                             std::vector<uint64_t> &ht_keys, IntBuffer &ht_vals,
                             uint64_t empty_key, uint32_t &hash_mask) {
  int hash_capacity = 1;
  while (hash_capacity < static_cast<int>(unique_keys.size()) * 2) {
    hash_capacity <<= 1;
  }

  hash_mask = static_cast<uint32_t>(hash_capacity - 1);
  ht_keys.assign(hash_capacity, empty_key);
  ht_vals.assign(hash_capacity, -1);

  for (int cell = 0; cell < static_cast<int>(unique_keys.size()); ++cell) {
    uint64_t key = unique_keys[cell];
    uint64_t h64 = (key ^ (key >> 33)) * 0xff51afd7ed558ccdULL;
    h64 ^= h64 >> 33;
    uint32_t h = static_cast<uint32_t>(h64) & hash_mask;

    while (ht_keys[h] != empty_key) {
      h = (h + 1) & hash_mask;
    }
    ht_keys[h] = key;
    ht_vals[h] = cell;
  }
}

inline void build_dense_cell_lookup(NeighborSearchData &data) {
  constexpr int64_t kDenseCellLimit = 16 * 1024 * 1024;

  data.dense_valid = false;
  data.dense_cells.clear();
  data.dense_nx = 0;
  data.dense_ny = 0;
  data.dense_nz = 0;

  if (data.num_cells == 0) {
    return;
  }

  int min_x = INT_MAX;
  int min_y = INT_MAX;
  int min_z = INT_MAX;
  int max_x = INT_MIN;
  int max_y = INT_MIN;
  int max_z = INT_MIN;
  for (uint64_t key : data.scratch_unique_keys) {
    const int cx = unpack_coord(key, 42);
    const int cy = unpack_coord(key, 21);
    const int cz = unpack_coord(key, 0);
    min_x = std::min(min_x, cx);
    min_y = std::min(min_y, cy);
    min_z = std::min(min_z, cz);
    max_x = std::max(max_x, cx);
    max_y = std::max(max_y, cy);
    max_z = std::max(max_z, cz);
  }

  const int64_t nx = static_cast<int64_t>(max_x) - min_x + 1;
  const int64_t ny = static_cast<int64_t>(max_y) - min_y + 1;
  const int64_t nz = static_cast<int64_t>(max_z) - min_z + 1;
  if (nx <= 0 || ny <= 0 || nz <= 0 || nx > kDenseCellLimit ||
      ny > kDenseCellLimit || nz > kDenseCellLimit ||
      nx * ny > kDenseCellLimit || nx * ny * nz > kDenseCellLimit) {
    return;
  }

  const int64_t total_cells = nx * ny * nz;
  data.dense_cells.assign(static_cast<std::size_t>(total_cells), -1);
  data.dense_nx = static_cast<int>(nx);
  data.dense_ny = static_cast<int>(ny);
  data.dense_nz = static_cast<int>(nz);
  data.dense_min_x = min_x;
  data.dense_min_y = min_y;
  data.dense_min_z = min_z;

  for (int cell = 0; cell < data.num_cells; ++cell) {
    const uint64_t key =
        data.scratch_unique_keys[static_cast<std::size_t>(cell)];
    const int lx = unpack_coord(key, 42) - min_x;
    const int ly = unpack_coord(key, 21) - min_y;
    const int lz = unpack_coord(key, 0) - min_z;
    const int64_t index =
        (static_cast<int64_t>(lz) * data.dense_ny + ly) * data.dense_nx + lx;
    data.dense_cells[static_cast<std::size_t>(index)] = cell;
  }
  data.dense_valid = true;
}

inline const std::vector<float> &
cross_sorted_x(const CrossNeighborSearchData &data) {
  return data.source != nullptr ? data.source->sorted_x : data.sorted_x;
}
inline const std::vector<float> &
cross_sorted_y(const CrossNeighborSearchData &data) {
  return data.source != nullptr ? data.source->sorted_y : data.sorted_y;
}
inline const std::vector<float> &
cross_sorted_z(const CrossNeighborSearchData &data) {
  return data.source != nullptr ? data.source->sorted_z : data.sorted_z;
}
inline const IntBuffer &
cross_cell_starts(const CrossNeighborSearchData &data) {
  return data.source != nullptr ? data.source->cell_starts : data.cell_starts;
}
inline const IntBuffer &
cross_cell_ends(const CrossNeighborSearchData &data) {
  return data.source != nullptr ? data.source->cell_ends : data.cell_ends;
}

// Views of the grid data as the ISPC kernel parameter structs.

ispc::Points points_of(const NeighborSearchData &data) {
  return {data.sorted_x.data(), data.sorted_y.data(), data.sorted_z.data(),
          data.original_index.data()};
}

// A cross query's coordinates (shared with its source set when it has one).
// Query rows are indexed by sorted position, so no original index is needed.
ispc::Points query_points_of(const CrossNeighborSearchData &data) {
  return {cross_sorted_x(data).data(), cross_sorted_y(data).data(),
          cross_sorted_z(data).data(), nullptr};
}

ispc::Cells cells_of(const NeighborSearchData &data) {
  return {data.cell_starts.data(), data.cell_ends.data(),
          data.scratch_unique_keys.data()};
}

ispc::Cells query_cells_of(const CrossNeighborSearchData &data) {
  return {cross_cell_starts(data).data(), cross_cell_ends(data).data(),
          nullptr};
}

// Only the cell keys, for kernels that enumerate a cell's neighbors.
ispc::Cells key_cells(const std::vector<uint64_t> &keys) {
  return {nullptr, nullptr, keys.data()};
}

ispc::DenseGrid dense_of(const NeighborSearchData &data) {
  return {data.dense_cells.data(), data.dense_nx,    data.dense_ny,
          data.dense_nz,           data.dense_min_x, data.dense_min_y,
          data.dense_min_z};
}

ispc::HashGrid hash_of(const NeighborSearchData &data) {
  return {data.hash_keys.data(), data.hash_vals.data(), data.empty_key,
          data.hash_mask};
}

// Ranges written into the data's own buffers; drops any shared cache ranges
// a previous build pointed at.
template <typename Data> inline void use_own_ranges(Data &data) {
  data.shared_starts = nullptr;
  data.shared_ends = nullptr;
}

// Debug builds check that the range tables the kernels will read through raw
// pointers cover every cell: stale (cleared or shared) ranges are the bug
// class the STL's own bounds checks cannot see.
template <typename Data> void check_ranges(const Data &data) {
#ifndef NDEBUG
  const IntBuffer &starts = data.shared_starts != nullptr
                                ? *data.shared_starts
                                : data.cell_neighbor_starts;
  const IntBuffer &ends =
      data.shared_ends != nullptr ? *data.shared_ends : data.cell_neighbor_ends;
  const std::size_t cells = static_cast<std::size_t>(data.num_cells);
  assert(data.cell_neighbor_offset.size() >= cells);
  assert(data.cell_num_neighbors.size() >= cells);
  assert(starts.size() == ends.size());
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const int offset = data.cell_neighbor_offset[cell];
    const int count = data.cell_num_neighbors[cell];
    assert(offset >= 0 && count >= 0);
    assert(static_cast<std::size_t>(offset) + static_cast<std::size_t>(count) <=
           starts.size());
    for (int k = offset; k < offset + count; ++k) {
      assert(starts[static_cast<std::size_t>(k)] <=
             ends[static_cast<std::size_t>(k)]);
    }
  }
#else
  (void)data;
#endif
}

template <typename Data> ispc::CellRanges ranges_of(const Data &data) {
  check_ranges(data);
  const int *starts = data.shared_starts != nullptr
                          ? data.shared_starts->data()
                          : data.cell_neighbor_starts.data();
  const int *ends = data.shared_ends != nullptr
                        ? data.shared_ends->data()
                        : data.cell_neighbor_ends.data();
  return {data.cell_neighbor_offset.data(), data.cell_num_neighbors.data(),
          starts, ends};
}

template <typename Data>
ispc::NeighborCells neighbor_cells_of(const Data &data, int stride) {
  return {data.scratch_cached_neighbor_cells.data(),
          data.scratch_cached_neighbor_counts.data(), stride};
}

ispc::Blocks blocks_of(const NeighborSearchData &data) {
  return {data.cell_block_start.data(),
          data.cell_block_count.data(),
          data.block_begin.data(),
          data.block_end.data(),
          data.block_cell.data(),
          {data.block_min_x.data(), data.block_min_y.data(),
           data.block_min_z.data()},
          {data.block_max_x.data(), data.block_max_y.data(),
           data.block_max_z.data()},
          {data.cell_min_x.data(), data.cell_min_y.data(),
           data.cell_min_z.data()},
          {data.cell_max_x.data(), data.cell_max_y.data(),
           data.cell_max_z.data()}};
}

// `scratch` is one staging row for the _nt writers, nullptr otherwise.
ispc::FixedRows fixed_rows(int capacity, int *scratch, IntBuffer &counts,
                           IntBuffer &neighbors) {
  return {capacity, neighbor_permutation_table(), scratch, counts.data(),
          neighbors.data(), nullptr, nullptr};
}

// Per-query block layout: row q starts at neighbors[row_offset[q]] and has
// row_capacity[q] slots.
ispc::FixedRows block_rows(int capacity, int *scratch, IntBuffer &counts,
                           IntBuffer &neighbors, const IntBuffer &row_offset,
                           const IntBuffer &row_capacity) {
  return {capacity, neighbor_permutation_table(), scratch, counts.data(),
          neighbors.data(), row_offset.data(), row_capacity.data()};
}

// Runs write(begin, end, rows) over chunks of [0, num_items). With
// allow_staging, rows of capacity >= 256 are staged in a per-thread scratch
// row and copied out with streaming stores, which avoids write-allocate
// traffic on large outputs.
template <typename Write>
void write_rows_parallel(int num_items, int grain, int capacity,
                         bool allow_staging, IntBuffer &counts,
                         IntBuffer &neighbors, Write write) {
  const int stride = capacity + 8;
  const bool staged = allow_staging && capacity >= 256 && capacity % 8 == 0;
  const int max_threads =
      staged ? tbb::this_task_arena::max_concurrency() : 0;
  IntBuffer scratch;
  if (staged) {
    scratch.resize(static_cast<std::size_t>(max_threads) * stride);
  }
  parallel_cells(0, num_items, grain, [&](int begin, int end) {
    int *scratch_row = nullptr;
    if (staged) {
      int tid = tbb::this_task_arena::current_thread_index();
      if (tid < 0 || tid >= max_threads) {
        tid = 0;
      }
      scratch_row = scratch.data() + static_cast<std::size_t>(tid) * stride;
    }
    const ispc::FixedRows rows =
        fixed_rows(capacity, scratch_row, counts, neighbors);
    write(begin, end, rows);
  });
}

// Block-fixed rows: every query has its own row offset and capacity, so the
// padding is limited to each block's maximum instead of the global maximum.
template <typename Write>
void write_block_rows_parallel(int num_items, int grain, int scratch_capacity,
                               IntBuffer &counts, IntBuffer &neighbors,
                               const IntBuffer &row_offset,
                               const IntBuffer &row_capacity, Write write) {
  const int stride = scratch_capacity + 8;
  int max_threads = tbb::this_task_arena::max_concurrency();
  if (max_threads < 1) {
    max_threads = 1;
  }
  IntBuffer scratch;
  scratch.resize(static_cast<std::size_t>(max_threads) * stride);
  parallel_cells(0, num_items, grain, [&](int begin, int end) {
    int tid = tbb::this_task_arena::current_thread_index();
    if (tid < 0 || tid >= max_threads) {
      tid = 0;
    }
    int *scratch_row = scratch.data() + static_cast<std::size_t>(tid) * stride;
    const ispc::FixedRows rows =
        block_rows(scratch_capacity, scratch_row, counts, neighbors,
                   row_offset, row_capacity);
    write(begin, end, rows);
  });
}

// Paired CSR writer: two rows are staged per thread, so each target is
// loaded once for both queries.
template <typename Write>
void write_csr_rows_pair_parallel(int num_items, int grain, int capacity,
                                  Write write) {
  const int stride = capacity + 8;
  int max_threads = tbb::this_task_arena::max_concurrency();
  if (max_threads < 1) {
    max_threads = 1;
  }
  IntBuffer scratch;
  scratch.resize(static_cast<std::size_t>(max_threads) * 2 * stride);
  parallel_cells(0, num_items, grain, [&](int begin, int end) {
    int tid = tbb::this_task_arena::current_thread_index();
    if (tid < 0 || tid >= max_threads) {
      tid = 0;
    }
    int *scratch_row =
        scratch.data() + static_cast<std::size_t>(tid) * 2 * stride;
    write(begin, end, scratch_row);
  });
}

// Blocked writers run over source blocks; small block counts use a finer
// grain so all threads get work.
int block_grain(int block_count) { return block_count < 256 ? 8 : 64; }


inline void
collect_neighbor_cells_for_target(int num_cells,
                                  const std::vector<uint64_t> &source_keys,
                                  const NeighborSearchData &target,
                                  IntBuffer &scratch_cells,
                                  IntBuffer &scratch_counts, int stencil,
                                  int neighbor_stride) {
  if (num_cells == 0) {
    return;
  }
  if (target.num_cells == 0) {
    std::fill(scratch_counts.begin(), scratch_counts.begin() + num_cells, 0);
    return;
  }

  if (target.dense_valid) {
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_cells, cell_grain(num_cells)),
        [&](const tbb::blocked_range<int> &range) {
          const ispc::Cells queries = key_cells(source_keys);
          const ispc::DenseGrid grid = dense_of(target);
          ispc::collect_neighbor_cells_dense(range.begin(), range.end(),
              &queries, &grid, stencil, neighbor_stride, scratch_cells.data(),
              scratch_counts.data());
        });
    return;
  }

  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_cells, cell_grain(num_cells)),
      [&](const tbb::blocked_range<int> &range) {
        const ispc::Cells queries = key_cells(source_keys);
        const ispc::HashGrid hash = hash_of(target);
        ispc::collect_neighbor_cells(range.begin(), range.end(), &queries,
            &hash, stencil, neighbor_stride, scratch_cells.data(),
            scratch_counts.data());
      });
}

template <typename Data>
inline void build_dense_stencil9_ranges_impl(
    Data &data, const std::vector<uint64_t> &query_keys,
    const NeighborSearchData &target) {
  use_own_ranges(data);
  const int num_cells = data.num_cells;
  data.cell_num_neighbors.assign(static_cast<std::size_t>(num_cells), 9);
  data.cell_neighbor_offset.resize(static_cast<std::size_t>(num_cells));
  for (int cell = 0; cell < num_cells; ++cell) {
    data.cell_neighbor_offset[static_cast<std::size_t>(cell)] = cell * 9;
  }
  data.cell_neighbor_starts.resize(static_cast<std::size_t>(num_cells) * 9);
  data.cell_neighbor_ends.resize(static_cast<std::size_t>(num_cells) * 9);
  if (num_cells == 0) {
    return;
  }
  parallel_cells(0, num_cells, cell_grain(num_cells),
                 [&](int begin, int end) {
                   const ispc::Cells queries = key_cells(query_keys);
                   const ispc::DenseGrid grid = dense_of(target);
                   const ispc::Cells targets = cells_of(target);
                   ispc::dense_stencil9_ranges(
                       begin, end, &queries, &grid, &targets,
                       data.cell_neighbor_starts.data(),
                       data.cell_neighbor_ends.data());
                 });
}

inline void build_merged_dense_cell_ranges(
    int num_cells, const std::vector<uint64_t> &unique_keys,
    const NeighborSearchData &target, IntBuffer &cell_num_neighbors,
    IntBuffer &cell_neighbor_offset, IntBuffer &cell_neighbor_starts,
    IntBuffer &cell_neighbor_ends) {
  cell_num_neighbors.resize(num_cells);
  cell_neighbor_offset.resize(num_cells);

  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_cells, cell_grain(num_cells)),
      [&](const tbb::blocked_range<int> &range) {
        const ispc::Cells queries = key_cells(unique_keys);
        const ispc::DenseGrid grid = dense_of(target);
        const ispc::Cells targets = cells_of(target);
        ispc::merge_dense_neighbor_ranges(range.begin(), range.end(),
            &queries, &grid, &targets, cell_num_neighbors.data(), nullptr,
            nullptr, nullptr);
      });

  int total_cell_neighbors = tbb::parallel_scan(
      tbb::blocked_range<int>(0, num_cells), 0,
      [&](const tbb::blocked_range<int> &range, int sum, bool is_final_scan) {
        int temp = sum;
        for (int i = range.begin(); i < range.end(); ++i) {
          if (is_final_scan) {
            cell_neighbor_offset[i] = temp;
          }
          temp += cell_num_neighbors[i];
        }
        return temp;
      },
      [](int left, int right) { return left + right; });

  cell_neighbor_starts.resize(total_cell_neighbors);
  cell_neighbor_ends.resize(total_cell_neighbors);

  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_cells, cell_grain(num_cells)),
      [&](const tbb::blocked_range<int> &range) {
        const ispc::Cells queries = key_cells(unique_keys);
        const ispc::DenseGrid grid = dense_of(target);
        const ispc::Cells targets = cells_of(target);
        ispc::merge_dense_neighbor_ranges(range.begin(), range.end(),
            &queries, &grid, &targets, nullptr, cell_neighbor_offset.data(),
            cell_neighbor_starts.data(), cell_neighbor_ends.data());
      });
}

inline void build_cross_cell_ranges_from_sorted_impl(
    CrossNeighborSearchData &data,
    const std::vector<uint64_t> &source_unique_keys,
    const NeighborSearchData &target_data, float radius) {
  use_own_ranges(data);
  const int num_cells = data.num_cells;
  data.radius = radius;

  // The blocked cross writer consumes the per-source-block candidate arrays
  // built below, so keep the unfused path when the source uses blocks.
  if (target_data.dense_valid &&
      (data.source == nullptr ||
       (!data.source->blocks_valid && data.source->cell_stencil == 1))) {
    if (target_data.dense_cache_valid) {
      data.cell_neighbor_offset.resize(static_cast<std::size_t>(num_cells));
      data.cell_num_neighbors.resize(static_cast<std::size_t>(num_cells));
      data.shared_starts = &target_data.dense_cache_starts;
      data.shared_ends = &target_data.dense_cache_ends;
      for (int cell = 0; cell < num_cells; ++cell) {
        const int target_cell = dense_cache_position(
            target_data, source_unique_keys[static_cast<std::size_t>(cell)]);
        if (target_cell >= 0) {
          data.cell_neighbor_offset[static_cast<std::size_t>(cell)] =
              target_data
                  .dense_cache_offsets[static_cast<std::size_t>(target_cell)];
          data.cell_num_neighbors[static_cast<std::size_t>(cell)] =
              target_data
                  .dense_cache_counts[static_cast<std::size_t>(target_cell)];
        } else {
          data.cell_neighbor_offset[static_cast<std::size_t>(cell)] = 0;
          data.cell_num_neighbors[static_cast<std::size_t>(cell)] = 0;
        }
      }
      return;
    }
    build_dense_stencil9_ranges_impl(data, source_unique_keys, target_data);
    return;
  }

  const int stencil = data.source != nullptr ? data.source->cell_stencil : 1;
  const int neighbor_stride =
      data.source != nullptr ? data.source->max_neighbor_cells : 27;
  data.scratch_cached_neighbor_cells.resize(
      static_cast<std::size_t>(num_cells) * neighbor_stride);
  data.scratch_cached_neighbor_counts.resize(num_cells);
  data.cell_num_neighbors.resize(num_cells);
  data.cell_neighbor_offset.resize(num_cells);

  collect_neighbor_cells_for_target(
      num_cells, source_unique_keys, target_data,
      data.scratch_cached_neighbor_cells,
      data.scratch_cached_neighbor_counts, stencil, neighbor_stride);

  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_cells, cell_grain(num_cells)),
      [&](const tbb::blocked_range<int> &range) {
        const ispc::Cells targets = cells_of(target_data);
        const ispc::NeighborCells ncells = neighbor_cells_of(data, neighbor_stride);
        ispc::merge_neighbor_ranges(range.begin(), range.end(), &targets,
            &ncells, data.cell_num_neighbors.data(), nullptr, nullptr,
            nullptr);
      });

  int total_cell_neighbors = tbb::parallel_scan(
      tbb::blocked_range<int>(0, num_cells), 0,
      [&](const tbb::blocked_range<int> &range, int sum, bool is_final_scan) {
        int temp = sum;
        for (int i = range.begin(); i < range.end(); ++i) {
          if (is_final_scan) {
            data.cell_neighbor_offset[i] = temp;
          }
          temp += data.cell_num_neighbors[i];
        }
        return temp;
      },
      [](int left, int right) { return left + right; });

  data.cell_neighbor_starts.resize(total_cell_neighbors);
  data.cell_neighbor_ends.resize(total_cell_neighbors);

  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_cells, cell_grain(num_cells)),
      [&](const tbb::blocked_range<int> &range) {
        const ispc::Cells targets = cells_of(target_data);
        const ispc::NeighborCells ncells = neighbor_cells_of(data, neighbor_stride);
        ispc::merge_neighbor_ranges(range.begin(), range.end(), &targets,
            &ncells, nullptr, data.cell_neighbor_offset.data(),
            data.cell_neighbor_starts.data(), data.cell_neighbor_ends.data());
      });
}

inline void build_cross_neighbor_lists_from_sorted(
    CrossNeighborSearchData &data,
    const std::vector<uint64_t> &source_unique_keys,
    const NeighborSearchData &target_data, float radius) {
  build_cross_cell_ranges_from_sorted_impl(
      data, source_unique_keys, target_data, radius);


  data.counts.resize(data.num_points);
  parallel_cells(
      0, data.num_cells, cell_grain(data.num_cells),
      [&](int begin, int end) {
        const ispc::Cells queries = query_cells_of(data);
        const ispc::CellRanges ranges = ranges_of(data);
        const ispc::Points qpts = query_points_of(data);
        const ispc::Points tpts = points_of(target_data);
        ispc::csr_count_rows(begin, end, &queries, &ranges, &qpts, &tpts,
                            false, data.radius * data.radius,
                            data.counts.data());
      });

  data.row_ptr.resize(data.num_points + 1);
  int total_neighbors = tbb::parallel_scan(
      tbb::blocked_range<int>(0, data.num_points), 0,
      [&](const tbb::blocked_range<int> &range, int sum, bool is_final_scan) {
        int temp = sum;
        for (int i = range.begin(); i < range.end(); ++i) {
          if (is_final_scan) {
            data.row_ptr[i] = temp;
          }
          temp += data.counts[i];
        }
        return temp;
      },
      [](int left, int right) { return left + right; });
  data.row_ptr[data.num_points] = total_neighbors;
  data.col_idx.resize(total_neighbors);
}


} // namespace

namespace {

template <typename T, typename Kernel>
SyncResult sync_points_impl(const T *xyz, int n, bool comparable,
                            std::vector<float> &x, std::vector<float> &y,
                            std::vector<float> &z, Kernel kernel) {
  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, n, 16384), SyncResult{},
      [&](const tbb::blocked_range<int> &range, SyncResult acc) {
        float out[7];
        kernel(xyz, range.begin(), range.end(), comparable, x.data(),
               y.data(), z.data(), out);
        acc.max_distance_sq = std::max(acc.max_distance_sq, out[0]);
        acc.bounds.add(out[1], out[2], out[3]);
        acc.bounds.add(out[4], out[5], out[6]);
        return acc;
      },
      [](SyncResult a, const SyncResult &b) {
        a.max_distance_sq = std::max(a.max_distance_sq, b.max_distance_sq);
        a.bounds.add(b.bounds);
        return a;
      });
}

} // namespace

SyncResult sync_points(const float *xyz, int n, bool comparable,
                       std::vector<float> &x, std::vector<float> &y,
                       std::vector<float> &z) {
  return sync_points_impl(xyz, n, comparable, x, y, z,
                          ispc::sync_points_float);
}

SyncResult sync_points(const double *xyz, int n, bool comparable,
                       std::vector<float> &x, std::vector<float> &y,
                       std::vector<float> &z) {
  return sync_points_impl(xyz, n, comparable, x, y, z,
                          ispc::sync_points_double);
}

static void build_point_blocks_inplace(NeighborSearchData &data) {
  constexpr int kBlockThreshold = 64;
  data.block_begin.clear();
  data.block_end.clear();
  data.block_cell.clear();
  data.block_min_x.clear();
  data.block_min_y.clear();
  data.block_min_z.clear();
  data.block_max_x.clear();
  data.block_max_y.clear();
  data.block_max_z.clear();
  data.cell_min_x.clear();
  data.cell_min_y.clear();
  data.cell_min_z.clear();
  data.cell_max_x.clear();
  data.cell_max_y.clear();
  data.cell_max_z.clear();
  data.cell_block_start.clear();
  data.cell_block_count.clear();
  data.blocks_valid = false;
  data.max_cell_points = 0;

  const int num_cells = data.num_cells;
  const int num_points = data.num_points;
  if (num_cells < 8 || num_points == 0) {
    return;
  }

  int max_cell_points = 0;
  for (int cell = 0; cell < num_cells; ++cell) {
    max_cell_points = std::max(
        max_cell_points,
        data.cell_ends[static_cast<std::size_t>(cell)] -
            data.cell_starts[static_cast<std::size_t>(cell)]);
  }
  data.max_cell_points = max_cell_points;
  if (max_cell_points <= kBlockThreshold) {
    return;
  }
  const int block_size =
      (num_points >= 100000 && max_cell_points > 512) ? 64 : 256;

  data.cell_block_start.resize(static_cast<std::size_t>(num_cells) + 1);
  data.cell_block_count.resize(static_cast<std::size_t>(num_cells));
  data.cell_min_x.resize(static_cast<std::size_t>(num_cells));
  data.cell_min_y.resize(static_cast<std::size_t>(num_cells));
  data.cell_min_z.resize(static_cast<std::size_t>(num_cells));
  data.cell_max_x.resize(static_cast<std::size_t>(num_cells));
  data.cell_max_y.resize(static_cast<std::size_t>(num_cells));
  data.cell_max_z.resize(static_cast<std::size_t>(num_cells));
  const std::size_t estimated_blocks =
      static_cast<std::size_t>(num_points / block_size + num_cells);
  data.block_begin.reserve(estimated_blocks);
  data.block_end.reserve(estimated_blocks);
  data.block_cell.reserve(estimated_blocks);
  data.block_min_x.reserve(estimated_blocks);
  data.block_min_y.reserve(estimated_blocks);
  data.block_min_z.reserve(estimated_blocks);
  data.block_max_x.reserve(estimated_blocks);
  data.block_max_y.reserve(estimated_blocks);
  data.block_max_z.reserve(estimated_blocks);

  std::vector<int> order;
  std::vector<uint32_t> codes;
  std::vector<float> temp_x;
  std::vector<float> temp_y;
  std::vector<float> temp_z;
  std::vector<int> temp_index;
  constexpr uint32_t kLocalMortonMax = (1U << 10) - 1U;

  for (int cell = 0; cell < num_cells; ++cell) {
    const std::size_t cell_index = static_cast<std::size_t>(cell);
    const int begin = data.cell_starts[cell_index];
    const int end = data.cell_ends[cell_index];
    const int count = end - begin;
    data.cell_block_start[cell_index] = data.block_begin.size();

    float cell_min_x = data.sorted_x[static_cast<std::size_t>(begin)];
    float cell_min_y = data.sorted_y[static_cast<std::size_t>(begin)];
    float cell_min_z = data.sorted_z[static_cast<std::size_t>(begin)];
    float cell_max_x = cell_min_x;
    float cell_max_y = cell_min_y;
    float cell_max_z = cell_min_z;
    for (int i = begin + 1; i < end; ++i) {
      cell_min_x = std::min(cell_min_x,
                            data.sorted_x[static_cast<std::size_t>(i)]);
      cell_min_y = std::min(cell_min_y,
                            data.sorted_y[static_cast<std::size_t>(i)]);
      cell_min_z = std::min(cell_min_z,
                            data.sorted_z[static_cast<std::size_t>(i)]);
      cell_max_x = std::max(cell_max_x,
                            data.sorted_x[static_cast<std::size_t>(i)]);
      cell_max_y = std::max(cell_max_y,
                            data.sorted_y[static_cast<std::size_t>(i)]);
      cell_max_z = std::max(cell_max_z,
                            data.sorted_z[static_cast<std::size_t>(i)]);
    }
    data.cell_min_x[cell_index] = cell_min_x;
    data.cell_min_y[cell_index] = cell_min_y;
    data.cell_min_z[cell_index] = cell_min_z;
    data.cell_max_x[cell_index] = cell_max_x;
    data.cell_max_y[cell_index] = cell_max_y;
    data.cell_max_z[cell_index] = cell_max_z;

    if (count > block_size) {
      float min_x = data.sorted_x[static_cast<std::size_t>(begin)];
      float min_y = data.sorted_y[static_cast<std::size_t>(begin)];
      float min_z = data.sorted_z[static_cast<std::size_t>(begin)];
      float max_x = min_x;
      float max_y = min_y;
      float max_z = min_z;
      for (int i = begin + 1; i < end; ++i) {
        min_x = std::min(min_x, data.sorted_x[static_cast<std::size_t>(i)]);
        min_y = std::min(min_y, data.sorted_y[static_cast<std::size_t>(i)]);
        min_z = std::min(min_z, data.sorted_z[static_cast<std::size_t>(i)]);
        max_x = std::max(max_x, data.sorted_x[static_cast<std::size_t>(i)]);
        max_y = std::max(max_y, data.sorted_y[static_cast<std::size_t>(i)]);
        max_z = std::max(max_z, data.sorted_z[static_cast<std::size_t>(i)]);
      }
      const float extent_x = max_x - min_x;
      const float extent_y = max_y - min_y;
      const float extent_z = max_z - min_z;
      const float inv_x = extent_x > 0.0f ? kLocalMortonMax / extent_x : 0.0f;
      const float inv_y = extent_y > 0.0f ? kLocalMortonMax / extent_y : 0.0f;
      const float inv_z = extent_z > 0.0f ? kLocalMortonMax / extent_z : 0.0f;

      order.resize(static_cast<std::size_t>(count));
      codes.resize(static_cast<std::size_t>(count));
      std::iota(order.begin(), order.end(), 0);
      for (int local = 0; local < count; ++local) {
        const std::size_t source = static_cast<std::size_t>(begin + local);
        const uint32_t qx = static_cast<uint32_t>(std::min(
            static_cast<float>(kLocalMortonMax),
            std::max(0.0f, (data.sorted_x[source] - min_x) * inv_x)));
        const uint32_t qy = static_cast<uint32_t>(std::min(
            static_cast<float>(kLocalMortonMax),
            std::max(0.0f, (data.sorted_y[source] - min_y) * inv_y)));
        const uint32_t qz = static_cast<uint32_t>(std::min(
            static_cast<float>(kLocalMortonMax),
            std::max(0.0f, (data.sorted_z[source] - min_z) * inv_z)));
        codes[static_cast<std::size_t>(local)] = (qx << 20) | (qy << 10) | qz;
      }
      std::sort(order.begin(), order.end(),
                [&codes](int a, int b) {
                  return codes[static_cast<std::size_t>(a)] <
                         codes[static_cast<std::size_t>(b)];
                });

      temp_x.resize(static_cast<std::size_t>(count));
      temp_y.resize(static_cast<std::size_t>(count));
      temp_z.resize(static_cast<std::size_t>(count));
      temp_index.resize(static_cast<std::size_t>(count));
      for (int local = 0; local < count; ++local) {
        const std::size_t source = static_cast<std::size_t>(
            begin + order[static_cast<std::size_t>(local)]);
        temp_x[static_cast<std::size_t>(local)] = data.sorted_x[source];
        temp_y[static_cast<std::size_t>(local)] = data.sorted_y[source];
        temp_z[static_cast<std::size_t>(local)] = data.sorted_z[source];
        temp_index[static_cast<std::size_t>(local)] =
            data.original_index[source];
      }
      for (int local = 0; local < count; ++local) {
        const std::size_t destination =
            static_cast<std::size_t>(begin + local);
        data.sorted_x[destination] = temp_x[static_cast<std::size_t>(local)];
        data.sorted_y[destination] = temp_y[static_cast<std::size_t>(local)];
        data.sorted_z[destination] = temp_z[static_cast<std::size_t>(local)];
        data.original_index[destination] =
            temp_index[static_cast<std::size_t>(local)];
      }
    }

    for (int block_begin = begin; block_begin < end;
         block_begin += block_size) {
      const int block_end = std::min(end, block_begin + block_size);
      float min_x = data.sorted_x[static_cast<std::size_t>(block_begin)];
      float min_y = data.sorted_y[static_cast<std::size_t>(block_begin)];
      float min_z = data.sorted_z[static_cast<std::size_t>(block_begin)];
      float max_x = min_x;
      float max_y = min_y;
      float max_z = min_z;
      for (int i = block_begin + 1; i < block_end; ++i) {
        min_x = std::min(min_x, data.sorted_x[static_cast<std::size_t>(i)]);
        min_y = std::min(min_y, data.sorted_y[static_cast<std::size_t>(i)]);
        min_z = std::min(min_z, data.sorted_z[static_cast<std::size_t>(i)]);
        max_x = std::max(max_x, data.sorted_x[static_cast<std::size_t>(i)]);
        max_y = std::max(max_y, data.sorted_y[static_cast<std::size_t>(i)]);
        max_z = std::max(max_z, data.sorted_z[static_cast<std::size_t>(i)]);
      }
      data.block_begin.push_back(block_begin);
      data.block_end.push_back(block_end);
      data.block_cell.push_back(cell);
      data.block_min_x.push_back(min_x);
      data.block_min_y.push_back(min_y);
      data.block_min_z.push_back(min_z);
      data.block_max_x.push_back(max_x);
      data.block_max_y.push_back(max_y);
      data.block_max_z.push_back(max_z);
    }
    data.cell_block_count[cell_index] =
        data.block_begin.size() - data.cell_block_start[cell_index];
  }
  data.cell_block_start[static_cast<std::size_t>(num_cells)] =
      data.block_begin.size();
  data.blocks_valid = true;

}


inline void build_dense_range_cache(NeighborSearchData &data);

void build_point_grid_inplace(NeighborSearchData &data,
                              const std::vector<float> &x,
                              const std::vector<float> &y,
                              const std::vector<float> &z, float radius,
                              float cell_radius) {
  data.num_points = static_cast<int>(x.size());
  data.radius = radius;
  data.cell_radius = cell_radius > 0.0f ? cell_radius : radius;
  // A zero (or invalid) radius has no meaningful ratio; one ring of cells
  // still finds the coincident points.
  const float ratio =
      data.cell_radius > 0.0f ? radius / data.cell_radius : 1.0f;
  data.cell_stencil =
      std::max(1, static_cast<int>(std::ceil(ratio - 1.0e-6f)));
  const int stencil_width = 2 * data.cell_stencil + 1;
  data.max_neighbor_cells = stencil_width * stencil_width * stencil_width;

  // Blocks and their cell bounds describe the previous sorted order.
  data.blocks_valid = false;
  data.block_begin.clear();
  data.block_end.clear();
  data.block_cell.clear();
  data.cell_block_start.clear();
  data.cell_block_count.clear();
  data.cell_min_x.clear();
  data.cell_min_y.clear();
  data.cell_min_z.clear();
  data.cell_max_x.clear();
  data.cell_max_y.clear();
  data.cell_max_z.clear();
  // Invalidate any shared range pointers from a previous grid build. The
  // same-set range buffers are kept until the new cell ranges are known, so
  // an unchanged grid can reuse them.
  use_own_ranges(data);
  data.counts.clear();
  data.col_idx.clear();
  data.row_ptr.clear();
  data.row_ptr.push_back(0);

  if (data.num_points == 0) {
    data.num_cells = 0;
    data.sorted_x.clear();
    data.sorted_y.clear();
    data.sorted_z.clear();
    data.original_index.clear();
    data.cell_starts.clear();
    data.cell_ends.clear();
    data.hash_keys.clear();
    data.hash_vals.clear();
    data.hash_mask = 0;
    data.scratch_unique_keys.clear();
    data.dense_cells.clear();
    data.dense_valid = false;
    data.dense_nx = 0;
    data.dense_ny = 0;
    data.dense_nz = 0;
    data.block_begin.clear();
    data.block_end.clear();
    data.block_cell.clear();
    data.block_min_x.clear();
    data.block_min_y.clear();
    data.block_min_z.clear();
    data.block_max_x.clear();
    data.block_max_y.clear();
    data.block_max_z.clear();
    data.cell_min_x.clear();
    data.cell_min_y.clear();
    data.cell_min_z.clear();
    data.cell_max_x.clear();
    data.cell_max_y.clear();
    data.cell_max_z.clear();
    data.cell_block_start.clear();
    data.cell_block_count.clear();
    data.blocks_valid = false;
    data.max_cell_points = 0;
    data.cell_neighbor_offset.clear();
    data.cell_num_neighbors.clear();
    data.cell_neighbor_starts.clear();
    data.cell_neighbor_ends.clear();
    data.cached_range_valid = false;
    data.dense_stencil_keys.clear();
    data.dense_stencil_cell_starts.clear();
    data.dense_stencil_cell_ends.clear();
    data.dense_stencil_total = -1;
    data.dense_stencil_cache_valid = false;
    return;
  }

  float inv_radius = 1.0f / data.cell_radius;
  sort_point_set(x, y, z, inv_radius, data.sorted_x, data.sorted_y,
                 data.sorted_z, data.original_index, data.cell_starts,
                 data.cell_ends, data.scratch_point_keys,
                 data.scratch_point_keys_alt, data.scratch_unique_keys);
  auto const &unique_keys = data.scratch_unique_keys;

  const std::size_t num_cells_size = data.cell_starts.size();
  const bool keep_ranges =
      data.cached_range_valid &&
      data.cached_range_stencil == data.cell_stencil &&
      data.cached_range_stride == data.max_neighbor_cells &&
      data.cached_range_cell_starts == data.cell_starts &&
      data.cached_range_cell_ends == data.cell_ends &&
      data.cell_neighbor_offset.size() == num_cells_size &&
      data.cell_num_neighbors.size() == num_cells_size &&
      data.cell_neighbor_starts.size() ==
          static_cast<std::size_t>(data.cached_range_total) &&
      data.cell_neighbor_ends.size() ==
          static_cast<std::size_t>(data.cached_range_total);
  const bool keep_dense_stencil =
      data.dense_stencil_cache_valid &&
      data.dense_stencil_keys == unique_keys &&
      data.dense_stencil_cell_starts == data.cell_starts &&
      data.dense_stencil_cell_ends == data.cell_ends &&
      data.cell_neighbor_offset.size() == num_cells_size &&
      data.cell_num_neighbors.size() == num_cells_size &&
      data.cell_neighbor_starts.size() ==
          static_cast<std::size_t>(data.dense_stencil_total) &&
      data.cell_neighbor_ends.size() ==
          static_cast<std::size_t>(data.dense_stencil_total);
  if (!keep_ranges && !keep_dense_stencil) {
    data.cell_neighbor_offset.clear();
    data.cell_num_neighbors.clear();
    data.cell_neighbor_starts.clear();
    data.cell_neighbor_ends.clear();
  }
  if (!keep_ranges) {
    data.cached_range_valid = false;
  }
  if (!keep_dense_stencil) {
    data.dense_stencil_cache_valid = false;
  }

  int num_cells = static_cast<int>(data.cell_starts.size());
  data.num_cells = num_cells;
  build_dense_cell_lookup(data);
  if (!data.dense_valid) {
    build_hash_table(unique_keys, data.hash_keys, data.hash_vals,
                     data.empty_key, data.hash_mask);
  } else {
    data.hash_keys.clear();
    data.hash_vals.clear();
    data.hash_mask = 0;
  }
  int max_cell_points = 0;
  for (int cell = 0; cell < num_cells; ++cell) {
    max_cell_points = std::max(
        max_cell_points,
        data.cell_ends[static_cast<std::size_t>(cell)] -
            data.cell_starts[static_cast<std::size_t>(cell)]);
  }
  data.max_cell_points = max_cell_points;
  // Blocks reorder points inside dense cells, so build them before the range
  // cache and neighbor rows are derived from the sorted order.
  if (max_cell_points > 32) {
    build_point_blocks_inplace(data);
  }
  build_dense_range_cache(data);
}

void ensure_point_blocks(NeighborSearchData &data) {
  if (data.blocks_valid || data.max_cell_points <= 32) {
    return;
  }
  build_point_blocks_inplace(data);
}

static void build_same_set_candidate_cells_inplace(NeighborSearchData &data) {
  if (data.num_points == 0) {
    data.scratch_cached_neighbor_cells.clear();
    data.scratch_cached_neighbor_counts.clear();
    data.cached_neighbor_keys.clear();
    data.cached_neighbor_valid = false;
    return;
  }

  const int num_cells = data.num_cells;
  const std::size_t cell_slots =
      static_cast<std::size_t>(num_cells) * data.max_neighbor_cells;
  const bool reuse =
      data.cached_neighbor_valid &&
      data.cached_neighbor_stencil == data.cell_stencil &&
      data.cached_neighbor_stride == data.max_neighbor_cells &&
      data.scratch_cached_neighbor_cells.size() == cell_slots &&
      data.scratch_cached_neighbor_counts.size() ==
          static_cast<std::size_t>(num_cells) &&
      data.cached_neighbor_keys == data.scratch_unique_keys;
  if (reuse) {
    return;
  }

  data.scratch_cached_neighbor_cells.resize(cell_slots);
  data.scratch_cached_neighbor_counts.resize(num_cells);
  collect_neighbor_cells_for_target(
      num_cells, data.scratch_unique_keys, data,
      data.scratch_cached_neighbor_cells, data.scratch_cached_neighbor_counts,
      data.cell_stencil, data.max_neighbor_cells);
  data.cached_neighbor_keys = data.scratch_unique_keys;
  data.cached_neighbor_stencil = data.cell_stencil;
  data.cached_neighbor_stride = data.max_neighbor_cells;
  data.cached_neighbor_valid = true;
  ++data.neighbor_generation;
}

inline int dense_cache_position(const NeighborSearchData &data,
                                uint64_t key) {
  if (!data.dense_cache_valid) {
    return -1;
  }
  const int cx0 = data.dense_min_x - 1;
  const int cy0 = data.dense_min_y - 1;
  const int cz0 = data.dense_min_z - 1;
  const int nx = data.dense_nx + 2;
  const int ny = data.dense_ny + 2;
  const int nz = data.dense_nz + 2;
  const int lx = unpack_coord(key, 42) - cx0;
  const int ly = unpack_coord(key, 21) - cy0;
  const int lz = unpack_coord(key, 0) - cz0;
  if (lx < 0 || lx >= nx || ly < 0 || ly >= ny || lz < 0 || lz >= nz) {
    return -1;
  }
  return (lz * ny + ly) * nx + lx;
}

inline void build_dense_range_cache(NeighborSearchData &data) {
  data.dense_cache_valid = false;
  data.dense_cache_offsets.clear();
  data.dense_cache_counts.clear();
  data.dense_cache_starts.clear();
  data.dense_cache_ends.clear();
  if (!data.dense_valid || data.dense_nx <= 0 || data.dense_ny <= 0 ||
      data.dense_nz <= 0) {
    return;
  }
  // One extra ring lets a source cell just outside the target AABB see its
  // target neighbors.
  const int nx = data.dense_nx + 2;
  const int ny = data.dense_ny + 2;
  const int nz = data.dense_nz + 2;
  const int cx0 = data.dense_min_x - 1;
  const int cy0 = data.dense_min_y - 1;
  const int cz0 = data.dense_min_z - 1;
  const int64_t padded = static_cast<int64_t>(nx) * ny * nz;
  // Sparse boxes (a few clusters in a large domain) would pay for every
  // empty cell; there the direct per-pair ranges are cheaper.
  if (padded > static_cast<int64_t>(std::max(0, data.range_cache_uses)) *
                   data.num_cells) {
    return;
  }
  const int total = static_cast<int>(padded);
  std::vector<uint64_t> keys(static_cast<std::size_t>(total));
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        keys[static_cast<std::size_t>((z * ny + y) * nx + x)] =
            make_key(cx0 + x, cy0 + y, cz0 + z);
      }
    }
  }
  data.dense_cache_offsets.resize(static_cast<std::size_t>(total));
  data.dense_cache_counts.assign(static_cast<std::size_t>(total), 9);
  data.dense_cache_starts.resize(static_cast<std::size_t>(total) * 9);
  data.dense_cache_ends.resize(static_cast<std::size_t>(total) * 9);
  for (int cell = 0; cell < total; ++cell) {
    data.dense_cache_offsets[static_cast<std::size_t>(cell)] = cell * 9;
  }
  tbb::parallel_for(
      tbb::blocked_range<int>(0, total, cell_grain(total)),
      [&](const tbb::blocked_range<int> &range) {
        const ispc::Cells queries = key_cells(keys);
        const ispc::DenseGrid grid = dense_of(data);
        const ispc::Cells targets = cells_of(data);
        ispc::dense_stencil9_ranges(
            range.begin(), range.end(), &queries, &grid, &targets,
            data.dense_cache_starts.data(), data.dense_cache_ends.data());
      });
  data.dense_cache_valid = true;
}

inline int find_cell_index_cpp(const NeighborSearchData &data,
                               uint64_t key) {
  if (data.dense_valid) {
    const int lx = unpack_coord(key, 42) - data.dense_min_x;
    const int ly = unpack_coord(key, 21) - data.dense_min_y;
    const int lz = unpack_coord(key, 0) - data.dense_min_z;
    if (lx < 0 || lx >= data.dense_nx || ly < 0 || ly >= data.dense_ny ||
        lz < 0 || lz >= data.dense_nz) {
      return -1;
    }
    const int64_t index =
        (static_cast<int64_t>(lz) * data.dense_ny + ly) * data.dense_nx + lx;
    return data.dense_cells[static_cast<std::size_t>(index)];
  }

  uint64_t h64 = (key ^ (key >> 33)) * 0xff51afd7ed558ccdULL;
  h64 ^= h64 >> 33;
  uint32_t h = static_cast<uint32_t>(h64) & data.hash_mask;
  while (true) {
    const uint64_t ht_key = data.hash_keys[h];
    if (ht_key == key) {
      return data.hash_vals[h];
    }
    if (ht_key == data.empty_key) {
      return -1;
    }
    h = (h + 1) & data.hash_mask;
  }
}

void build_same_set_cell_ranges_inplace(NeighborSearchData &data) {
  use_own_ranges(data);
  data.dense_stencil_cache_valid = false;
  if (data.num_points == 0) {
    data.cell_neighbor_offset.clear();
    data.cell_num_neighbors.clear();
    data.cell_neighbor_starts.clear();
    data.cell_neighbor_ends.clear();
    data.cached_range_valid = false;
    return;
  }

  build_same_set_candidate_cells_inplace(data);
  const int num_cells = data.num_cells;
  const bool reuse_ranges =
      data.cached_range_valid &&
      data.cached_range_generation == data.neighbor_generation &&
      data.cached_range_stencil == data.cell_stencil &&
      data.cached_range_stride == data.max_neighbor_cells &&
      data.cached_range_cell_starts == data.cell_starts &&
      data.cached_range_cell_ends == data.cell_ends &&
      data.cell_num_neighbors.size() == static_cast<std::size_t>(num_cells) &&
      data.cell_neighbor_offset.size() == static_cast<std::size_t>(num_cells) &&
      data.cell_neighbor_starts.size() ==
          static_cast<std::size_t>(data.cached_range_total) &&
      data.cell_neighbor_ends.size() ==
          static_cast<std::size_t>(data.cached_range_total);
  if (reuse_ranges) {
    return;
  }

  data.cell_num_neighbors.resize(num_cells);
  data.cell_neighbor_offset.resize(num_cells);

  tbb::parallel_for(tbb::blocked_range<int>(0, num_cells, cell_grain(num_cells)),
                    [&](const tbb::blocked_range<int> &range) {
                      const ispc::Cells targets = cells_of(data);
                      const ispc::NeighborCells ncells =
                          neighbor_cells_of(data, data.max_neighbor_cells);
                      ispc::merge_neighbor_ranges(range.begin(), range.end(), &targets,
                          &ncells, data.cell_num_neighbors.data(), nullptr, nullptr,
                          nullptr);
                    });

  int total_cell_neighbors = tbb::parallel_scan(
      tbb::blocked_range<int>(0, num_cells), 0,
      [&](const tbb::blocked_range<int> &range, int sum, bool is_final_scan) {
        int temp = sum;
        for (int i = range.begin(); i < range.end(); ++i) {
          if (is_final_scan) {
            data.cell_neighbor_offset[i] = temp;
          }
          temp += data.cell_num_neighbors[i];
        }
        return temp;
      },
      [](int left, int right) { return left + right; });

  data.cell_neighbor_starts.resize(total_cell_neighbors);
  data.cell_neighbor_ends.resize(total_cell_neighbors);

  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_cells, cell_grain(num_cells)),
      [&](const tbb::blocked_range<int> &range) {
        const ispc::Cells targets = cells_of(data);
        const ispc::NeighborCells ncells =
            neighbor_cells_of(data, data.max_neighbor_cells);
        ispc::merge_neighbor_ranges(range.begin(), range.end(), &targets,
            &ncells, nullptr, data.cell_neighbor_offset.data(),
            data.cell_neighbor_starts.data(), data.cell_neighbor_ends.data());
      });

  data.cached_range_cell_starts = data.cell_starts;
  data.cached_range_cell_ends = data.cell_ends;
  data.cached_range_stencil = data.cell_stencil;
  data.cached_range_stride = data.max_neighbor_cells;
  data.cached_range_total = total_cell_neighbors;
  data.cached_range_generation = data.neighbor_generation;
  data.cached_range_valid = true;
}


void build_same_set_cell_ranges_dense_fused_inplace(
    NeighborSearchData &data) {
  use_own_ranges(data);
  data.cached_range_valid = false;
  if (data.num_points == 0) {
    data.cell_neighbor_offset.clear();
    data.cell_num_neighbors.clear();
    data.cell_neighbor_starts.clear();
    data.cell_neighbor_ends.clear();
    data.dense_stencil_cache_valid = false;
    return;
  }
  if (!data.dense_valid || data.cell_stencil > 1) {
    build_same_set_cell_ranges_inplace(data);
    return;
  }
  const int num_cells = data.num_cells;
  if (!data.dense_cache_valid && data.dense_stencil_cache_valid &&
      data.dense_stencil_keys == data.scratch_unique_keys &&
      data.dense_stencil_cell_starts == data.cell_starts &&
      data.dense_stencil_cell_ends == data.cell_ends &&
      data.cell_neighbor_offset.size() ==
          static_cast<std::size_t>(num_cells) &&
      data.cell_num_neighbors.size() ==
          static_cast<std::size_t>(num_cells) &&
      data.cell_neighbor_starts.size() ==
          static_cast<std::size_t>(data.dense_stencil_total) &&
      data.cell_neighbor_ends.size() ==
          static_cast<std::size_t>(data.dense_stencil_total)) {
    return;
  }
  if (!data.dense_cache_valid) {
    build_dense_stencil9_ranges_impl(data, data.scratch_unique_keys, data);
    data.dense_stencil_keys = data.scratch_unique_keys;
    data.dense_stencil_cell_starts = data.cell_starts;
    data.dense_stencil_cell_ends = data.cell_ends;
    data.dense_stencil_total =
        static_cast<int>(data.cell_neighbor_starts.size());
    data.dense_stencil_cache_valid = true;
    return;
  }
  data.dense_stencil_cache_valid = false;
  data.cell_neighbor_offset.resize(static_cast<std::size_t>(num_cells));
  data.cell_num_neighbors.resize(static_cast<std::size_t>(num_cells));
  data.shared_starts = &data.dense_cache_starts;
  data.shared_ends = &data.dense_cache_ends;
  for (int cell = 0; cell < num_cells; ++cell) {
    const int target_cell = dense_cache_position(
        data, data.scratch_unique_keys[static_cast<std::size_t>(cell)]);
    if (target_cell >= 0) {
      data.cell_neighbor_offset[static_cast<std::size_t>(cell)] =
          data.dense_cache_offsets[static_cast<std::size_t>(target_cell)];
      data.cell_num_neighbors[static_cast<std::size_t>(cell)] =
          data.dense_cache_counts[static_cast<std::size_t>(target_cell)];
    } else {
      data.cell_neighbor_offset[static_cast<std::size_t>(cell)] = 0;
      data.cell_num_neighbors[static_cast<std::size_t>(cell)] = 0;
    }
  }
}

static void count_same_set_neighbors_parallel(NeighborSearchData &data,
                                       IntBuffer &counts) {
  counts.resize(static_cast<std::size_t>(data.num_points));
  if (data.num_points == 0) {
    return;
  }

  parallel_cells(
      0, data.num_cells, cell_grain(data.num_cells),
      [&](int begin, int end) {
        const ispc::Cells cells = cells_of(data);
        const ispc::CellRanges ranges = ranges_of(data);
        const ispc::Points pts = points_of(data);
        ispc::csr_count_rows(begin, end, &cells, &ranges, &pts, &pts, true,
                            data.radius * data.radius, counts.data());
      });
}

void count_same_set_neighbors_inplace(NeighborSearchData &data) {
  if (data.dense_valid && data.cell_stencil == 1) {
    build_same_set_cell_ranges_dense_fused_inplace(data);
  } else {
    build_same_set_cell_ranges_inplace(data);
  }
  count_same_set_neighbors_parallel(data, data.counts);
}

void build_same_set_neighbor_lists_inplace(NeighborSearchData &data,
                                           float radius) {
  data.radius = radius;
  if (data.num_points == 0) {
    data.cell_neighbor_offset.clear();
    data.cell_num_neighbors.clear();
    data.cell_neighbor_starts.clear();
    data.cell_neighbor_ends.clear();
    data.counts.clear();
    data.col_idx.clear();
    data.row_ptr.clear();
    data.row_ptr.push_back(0);
    return;
  }

  if (data.dense_valid && data.cell_stencil == 1) {
    build_same_set_cell_ranges_dense_fused_inplace(data);
  } else {
    build_same_set_cell_ranges_inplace(data);
  }
  count_same_set_neighbors_parallel(data, data.counts);

  data.row_ptr.resize(data.num_points + 1);
  int total_neighbors = tbb::parallel_scan(
      tbb::blocked_range<int>(0, data.num_points), 0,
      [&](const tbb::blocked_range<int> &range, int sum, bool is_final_scan) {
        int temp = sum;
        for (int i = range.begin(); i < range.end(); ++i) {
          if (is_final_scan) {
            data.row_ptr[i] = temp;
          }
          temp += data.counts[i];
        }
        return temp;
      },
      [](int left, int right) { return left + right; });
  data.row_ptr[data.num_points] = total_neighbors;
  data.col_idx.resize(total_neighbors);
}

void write_neighbors_fixed_query_active2_parallel(NeighborSearchData &data,
                                                  int capacity,
                                                  IntBuffer &fixed_neighbors,
                                                  IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(data.num_points), capacity + 8);
  if (data.num_points == 0) {
    return;
  }
  const ispc::Cells cells = cells_of(data);
  const ispc::CellRanges ranges = ranges_of(data);
  const ispc::Points pts = points_of(data);
  const float r2 = data.radius * data.radius;
  write_rows_parallel(
      data.num_cells, cell_grain(data.num_cells), capacity, true, counts,
      fixed_neighbors, [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows(begin, end, &cells, &ranges, &pts, &pts, true,
                               r2, &rows);
      });
}




void write_neighbors_fixed_query_pair_parallel(NeighborSearchData &data,
                                               int capacity,
                                               IntBuffer &fixed_neighbors,
                                               IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(data.num_points), capacity + 8);
  if (data.num_points == 0) {
    return;
  }
  const ispc::Cells cells = cells_of(data);
  const ispc::CellRanges ranges = ranges_of(data);
  const ispc::Points pts = points_of(data);
  const float r2 = data.radius * data.radius;
  write_rows_parallel(
      data.num_cells, cell_grain(data.num_cells), capacity, false, counts,
      fixed_neighbors, [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_pair(begin, end, &cells, &ranges, &pts, &pts,
                                    true, r2, &rows);
      });
}

void write_neighbors_fixed_query_oct_parallel(NeighborSearchData &data,
                                              int capacity,
                                              IntBuffer &fixed_neighbors,
                                              IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(data.num_points), capacity + 8);
  if (data.num_points == 0) {
    return;
  }
  const ispc::Cells cells = cells_of(data);
  const ispc::CellRanges ranges = ranges_of(data);
  const ispc::Points pts = points_of(data);
  const float r2 = data.radius * data.radius;
  write_rows_parallel(
      data.num_cells, cell_grain(data.num_cells), capacity, false, counts,
      fixed_neighbors, [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_oct(begin, end, &cells, &ranges, &pts, &pts,
                                   true, r2, &rows);
      });
}

void write_neighbors_fixed_query_blocked_parallel(
    NeighborSearchData &data, int capacity, IntBuffer &fixed_neighbors,
    IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(data.num_points), capacity + 8);
  if (data.num_points == 0) {
    return;
  }
  const ispc::Blocks blocks = blocks_of(data);
  const ispc::NeighborCells ncells =
      neighbor_cells_of(data, data.max_neighbor_cells);
  const ispc::Points pts = points_of(data);
  const float r2 = data.radius * data.radius;
  const int block_count = static_cast<int>(data.block_begin.size());
  write_rows_parallel(
      block_count, block_grain(block_count), capacity, true, counts,
      fixed_neighbors, [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_blocked(begin, end, &blocks, &ncells, &blocks,
                                       &pts, &pts, true, r2, &rows);
      });
}


void build_neighbor_search_data_inplace(NeighborSearchData &data,
                                        const std::vector<float> &x,
                                        const std::vector<float> &y,
                                        const std::vector<float> &z,
                                        float radius) {
  build_point_grid_inplace(data, x, y, z, radius);
  build_same_set_neighbor_lists_inplace(data, radius);
}

NeighborSearchData build_neighbor_search_data(const std::vector<float> &x,
                                              const std::vector<float> &y,
                                              const std::vector<float> &z,
                                              float radius) {
  NeighborSearchData data;
  build_neighbor_search_data_inplace(data, x, y, z, radius);
  return data;
}

static void build_cross_neighbor_search_data_inplace(
    CrossNeighborSearchData &data, const std::vector<float> &query_x,
    const std::vector<float> &query_y, const std::vector<float> &query_z,
    const NeighborSearchData &target_data, float radius) {
  use_own_ranges(data);
  data.num_points = static_cast<int>(query_x.size());
  data.radius = radius;
  data.source = nullptr;

  if (data.num_points == 0) {
    data.num_cells = 0;
    data.source = nullptr;
    data.sorted_x.clear();
    data.sorted_y.clear();
    data.sorted_z.clear();
    data.original_index.clear();
    data.cell_starts.clear();
    data.cell_ends.clear();
    data.cell_neighbor_offset.clear();
    data.cell_num_neighbors.clear();
    data.cell_neighbor_starts.clear();
    data.cell_neighbor_ends.clear();
    data.counts.clear();
    data.col_idx.clear();
    data.row_ptr.clear();
    data.row_ptr.push_back(0);
    return;
  }

  float inv_radius = 1.0f / radius;
  sort_point_set(query_x, query_y, query_z, inv_radius, data.sorted_x,
                 data.sorted_y, data.sorted_z, data.original_index,
                 data.cell_starts, data.cell_ends, data.scratch_point_keys,
                 data.scratch_point_keys_alt, data.scratch_unique_keys);
  auto const &unique_keys = data.scratch_unique_keys;

  data.num_cells = static_cast<int>(data.cell_starts.size());
  build_cross_neighbor_lists_from_sorted(data, unique_keys, target_data,
                                         radius);
}

void build_cross_cell_ranges_from_sorted_inplace(
    CrossNeighborSearchData &data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, float radius) {
  use_own_ranges(data);
  data.num_points = source_data.num_points;
  data.radius = radius;

  if (data.num_points == 0) {
    data.num_cells = 0;
    data.source = nullptr;
    data.sorted_x.clear();
    data.sorted_y.clear();
    data.sorted_z.clear();
    data.original_index.clear();
    data.cell_starts.clear();
    data.cell_ends.clear();
    data.cell_neighbor_offset.clear();
    data.cell_num_neighbors.clear();
    data.cell_neighbor_starts.clear();
    data.cell_neighbor_ends.clear();
    return;
  }

  data.num_cells = source_data.num_cells;
  data.source = &source_data;
  data.sorted_x.clear();
  data.sorted_y.clear();
  data.sorted_z.clear();
  data.original_index.clear();
  data.cell_starts.clear();
  data.cell_ends.clear();

  build_cross_cell_ranges_from_sorted_impl(
      data, source_data.scratch_unique_keys, target_data, radius);
}

void build_cross_neighbor_search_data_from_sorted_inplace(
    CrossNeighborSearchData &data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, float radius) {
  use_own_ranges(data);
  data.num_points = source_data.num_points;
  data.radius = radius;

  if (data.num_points == 0) {
    data.num_cells = 0;
    data.source = nullptr;
    data.sorted_x.clear();
    data.sorted_y.clear();
    data.sorted_z.clear();
    data.original_index.clear();
    data.cell_starts.clear();
    data.cell_ends.clear();
    data.cell_neighbor_offset.clear();
    data.cell_num_neighbors.clear();
    data.cell_neighbor_starts.clear();
    data.cell_neighbor_ends.clear();
    data.counts.clear();
    data.col_idx.clear();
    data.row_ptr.clear();
    data.row_ptr.push_back(0);
    return;
  }

  data.num_cells = source_data.num_cells;
  data.source = &source_data;
  data.sorted_x.clear();
  data.sorted_y.clear();
  data.sorted_z.clear();
  data.original_index.clear();
  data.cell_starts.clear();
  data.cell_ends.clear();

  build_cross_neighbor_lists_from_sorted(
      data, source_data.scratch_unique_keys, target_data, radius);
}

CrossNeighborSearchData build_cross_neighbor_search_data(
    const std::vector<float> &query_x, const std::vector<float> &query_y,
    const std::vector<float> &query_z, const NeighborSearchData &target_data,
    float radius) {
  CrossNeighborSearchData data;
  build_cross_neighbor_search_data_inplace(data, query_x, query_y, query_z,
                                           target_data, radius);
  return data;
}

void write_neighbors_range(NeighborSearchData &data, int start_cell,
                           int end_cell) {
  const ispc::Cells cells = cells_of(data);
  const ispc::CellRanges ranges = ranges_of(data);
  const ispc::Points pts = points_of(data);
  IntBuffer scratch;
  scratch.resize(static_cast<std::size_t>(max_row_count(data.counts) + 8));
  ispc::csr_rows(start_cell, end_cell, &cells, &ranges, &pts, &pts, true,
                 data.radius * data.radius, data.row_ptr.data(),
                 data.col_idx.data(), scratch.data(),
                 neighbor_permutation_table());
}

void write_same_set_block_rows_parallel(
    NeighborSearchData &data, int capacity, IntBuffer &counts,
    IntBuffer &neighbors, const IntBuffer &row_offset,
    const IntBuffer &row_capacity) {
  if (data.num_points == 0) {
    return;
  }
  const ispc::Cells cells = cells_of(data);
  const ispc::Points pts = points_of(data);
  const float r2 = data.radius * data.radius;
  if (data.max_cell_points > 32) {
    ensure_point_blocks(data);
    build_same_set_candidate_cells_inplace(data);
    if (data.blocks_valid) {
      const ispc::Blocks blocks = blocks_of(data);
      const ispc::NeighborCells ncells =
          neighbor_cells_of(data, data.max_neighbor_cells);
      const int block_count = static_cast<int>(data.block_begin.size());
      write_block_rows_parallel(
          block_count, block_grain(block_count), capacity, counts, neighbors,
          row_offset, row_capacity,
          [&](int begin, int end, const ispc::FixedRows &rows) {
            ispc::write_fixed_rows_blocked(begin, end, &blocks, &ncells,
                                           &blocks, &pts, &pts, true, r2,
                                           &rows);
          });
      return;
    }
  }
  // With cached block capacities nothing has built this grid's ranges yet
  // (every grid build clears them), and the cleared buffers still hold the
  // previous grid's ranges.
  if (data.cell_num_neighbors.size() !=
      static_cast<std::size_t>(data.num_cells)) {
    if (data.dense_valid && data.cell_stencil == 1) {
      build_same_set_cell_ranges_dense_fused_inplace(data);
    } else {
      build_same_set_cell_ranges_inplace(data);
    }
  }
  const ispc::CellRanges current_ranges = ranges_of(data);
  write_block_rows_parallel(
      data.num_cells, cell_grain(data.num_cells), capacity, counts, neighbors,
      row_offset, row_capacity,
      [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_pair(begin, end, &cells, &current_ranges, &pts,
                                    &pts, true, r2, &rows);
      });
}

void write_neighbors_parallel(NeighborSearchData &data) {
  if (data.num_points == 0) {
    return;
  }

  const ispc::Cells cells = cells_of(data);
  const ispc::CellRanges ranges = ranges_of(data);
  const ispc::Points pts = points_of(data);
  const float r2 = data.radius * data.radius;
  const int capacity = max_row_count(data.counts);
  write_csr_rows_pair_parallel(
      data.num_cells, cell_grain(data.num_cells), capacity,
      [&](int begin, int end, int *scratch) {
        ispc::csr_rows_pair(begin, end, &cells, &ranges, &pts, &pts, true, r2,
                            data.row_ptr.data(), data.col_idx.data(), scratch,
                            capacity + 8, neighbor_permutation_table());
      });
}

void write_cross_neighbors_range(CrossNeighborSearchData &query_data,
                                 const NeighborSearchData &target_data,
                                 int start_cell, int end_cell) {
  const ispc::Cells queries = query_cells_of(query_data);
  const ispc::CellRanges ranges = ranges_of(query_data);
  const ispc::Points qpts = query_points_of(query_data);
  const ispc::Points tpts = points_of(target_data);
  IntBuffer scratch;
  scratch.resize(
      static_cast<std::size_t>(max_row_count(query_data.counts) + 8));
  ispc::csr_rows(start_cell, end_cell, &queries, &ranges, &qpts, &tpts, false,
                 query_data.radius * query_data.radius,
                 query_data.row_ptr.data(), query_data.col_idx.data(),
                 scratch.data(), neighbor_permutation_table());
}

void write_cross_neighbors_parallel(CrossNeighborSearchData &query_data,
                                    const NeighborSearchData &target_data) {
  if (query_data.num_points == 0) {
    return;
  }

  const ispc::Cells queries = query_cells_of(query_data);
  const ispc::CellRanges ranges = ranges_of(query_data);
  const ispc::Points qpts = query_points_of(query_data);
  const ispc::Points tpts = points_of(target_data);
  const float r2 = query_data.radius * query_data.radius;
  const int capacity = max_row_count(query_data.counts);
  write_csr_rows_pair_parallel(
      query_data.num_cells, cell_grain(query_data.num_cells), capacity,
      [&](int begin, int end, int *scratch) {
        ispc::csr_rows_pair(begin, end, &queries, &ranges, &qpts, &tpts, false,
                            r2, query_data.row_ptr.data(),
                            query_data.col_idx.data(), scratch, capacity + 8,
                            neighbor_permutation_table());
      });
}

void write_cross_neighbors_fixed_query_active2_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(query_data.num_points),
                     capacity + 8);
  if (query_data.num_points == 0) {
    return;
  }
  const ispc::Cells queries = query_cells_of(query_data);
  const ispc::CellRanges ranges = ranges_of(query_data);
  const ispc::Points qpts = query_points_of(query_data);
  const ispc::Points tpts = points_of(target_data);
  const float r2 = query_data.radius * query_data.radius;
  write_rows_parallel(
      query_data.num_cells, cell_grain(query_data.num_cells), capacity, true,
      counts, fixed_neighbors,
      [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows(begin, end, &queries, &ranges, &qpts, &tpts,
                               false, r2, &rows);
      });
}



void write_cross_neighbors_block_rows_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts,
    const IntBuffer &row_offset, const IntBuffer &row_capacity) {
  if (query_data.num_points == 0) {
    return;
  }
  prepare_block_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(query_data.num_points),
                     row_offset, row_capacity);
  const ispc::Cells queries = query_cells_of(query_data);
  const ispc::CellRanges ranges = ranges_of(query_data);
  const ispc::Points qpts = query_points_of(query_data);
  const ispc::Points tpts = points_of(target_data);
  const float r2 = query_data.radius * query_data.radius;
  write_block_rows_parallel(
      query_data.num_cells, cell_grain(query_data.num_cells), capacity, counts,
      fixed_neighbors, row_offset, row_capacity,
      [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows(begin, end, &queries, &ranges, &qpts, &tpts,
                               false, r2, &rows);
      });
}

void write_cross_neighbors_block_rows_pair_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts,
    const IntBuffer &row_offset, const IntBuffer &row_capacity) {
  if (query_data.num_points == 0) {
    return;
  }
  prepare_block_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(query_data.num_points),
                     row_offset, row_capacity);
  const ispc::Cells queries = query_cells_of(query_data);
  const ispc::CellRanges ranges = ranges_of(query_data);
  const ispc::Points qpts = query_points_of(query_data);
  const ispc::Points tpts = points_of(target_data);
  const float r2 = query_data.radius * query_data.radius;
  write_block_rows_parallel(
      query_data.num_cells, cell_grain(query_data.num_cells), capacity, counts,
      fixed_neighbors, row_offset, row_capacity,
      [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_pair(begin, end, &queries, &ranges, &qpts,
                                    &tpts, false, r2, &rows);
      });
}

void write_cross_neighbors_block_rows_pair_dense_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, int capacity,
    IntBuffer &fixed_neighbors, IntBuffer &counts,
    const IntBuffer &row_offset, const IntBuffer &row_capacity) {
  if (source_data.num_points == 0) {
    return;
  }
  prepare_block_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(source_data.num_points),
                     row_offset, row_capacity);
  const bool sparse_target =
      target_data.num_cells > 0 && target_data.num_points > 0 &&
      target_data.max_cell_points > 0 && target_data.max_cell_points <= 64 &&
      static_cast<int64_t>(target_data.num_points) <
          4 * static_cast<int64_t>(target_data.num_cells);
  const int group = sparse_target ? 4 : 2;
  const ispc::Cells queries = cells_of(source_data);
  const ispc::DenseGrid grid = dense_of(target_data);
  const ispc::Cells targets = cells_of(target_data);
  const ispc::Points qpts = points_of(source_data);
  const ispc::Points tpts = points_of(target_data);
  const float r2 = query_data.radius * query_data.radius;
  write_block_rows_parallel(
      source_data.num_cells, cell_grain(source_data.num_cells), capacity, counts,
      fixed_neighbors, row_offset, row_capacity,
      [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_dense(begin, end, group, &queries, &grid,
                                     &targets, &qpts, &tpts, false, r2, &rows);
      });
}

void write_cross_neighbors_fixed_query_pair_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(query_data.num_points),
                     capacity + 8);
  if (query_data.num_points == 0) {
    return;
  }
  const ispc::Cells queries = query_cells_of(query_data);
  const ispc::CellRanges ranges = ranges_of(query_data);
  const ispc::Points qpts = query_points_of(query_data);
  const ispc::Points tpts = points_of(target_data);
  const float r2 = query_data.radius * query_data.radius;
  write_rows_parallel(
      query_data.num_cells, cell_grain(query_data.num_cells), capacity, false,
      counts, fixed_neighbors,
      [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_pair(begin, end, &queries, &ranges, &qpts,
                                    &tpts, false, r2, &rows);
      });
}

void write_cross_neighbors_target_major_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, int capacity,
    IntBuffer &fixed_neighbors, IntBuffer &counts) {
  const int num_source = source_data.num_points;
  const int num_target = target_data.num_points;
  const int row_stride = capacity + 8;
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(num_source), row_stride);
  if (num_source == 0 || num_target == 0) {
    return;
  }
  std::fill(counts.begin(), counts.end(), 0);
  if (!source_data.dense_valid || source_data.cell_stencil != 1) {
    return;
  }
  // Cells come from the source grid's own cell size; the search radius is
  // the pair's (source_data.radius is the set's last same-set radius, which
  // the exact-radius CSR fallback lowers below the grid's build radius).
  const float inv_radius = 1.0f / source_data.cell_radius;
  const float r2 = query_data.radius * query_data.radius;
  const int* dense_cells = source_data.dense_cells.data();
  const int nx = source_data.dense_nx;
  const int ny = source_data.dense_ny;
  const int nz = source_data.dense_nz;
  const int min_x = source_data.dense_min_x;
  const int min_y = source_data.dense_min_y;
  const int min_z = source_data.dense_min_z;
  const int* cell_starts = source_data.cell_starts.data();
  const int* cell_ends = source_data.cell_ends.data();
  const float* sx = source_data.sorted_x.data();
  const float* sy = source_data.sorted_y.data();
  const float* sz = source_data.sorted_z.data();
  const float* tx = target_data.sorted_x.data();
  const float* ty = target_data.sorted_y.data();
  const float* tz = target_data.sorted_z.data();
  const int* target_original = target_data.original_index.data();
  int* out = fixed_neighbors.data();
  int* row_counts = counts.data();

  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_target, 64),
      [&](const tbb::blocked_range<int> &range) {
        for (int t = range.begin(); t < range.end(); ++t) {
          const float qx = tx[static_cast<std::size_t>(t)];
          const float qy = ty[static_cast<std::size_t>(t)];
          const float qz = tz[static_cast<std::size_t>(t)];
          const int cx = floor_to_cell(qx * inv_radius);
          const int cy = floor_to_cell(qy * inv_radius);
          const int cz = floor_to_cell(qz * inv_radius);
          const int neighbor = target_original[static_cast<std::size_t>(t)];
          for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
              for (int dz = -1; dz <= 1; ++dz) {
                const int lx = cx + dx - min_x;
                const int ly = cy + dy - min_y;
                const int lz = cz + dz - min_z;
                if (lx < 0 || lx >= nx || ly < 0 || ly >= ny || lz < 0 ||
                    lz >= nz) {
                  continue;
                }
                const int cell = dense_cells[static_cast<std::size_t>(
                    (lz * ny + ly) * nx + lx)];
                if (cell < 0) {
                  continue;
                }
                const int begin = cell_starts[static_cast<std::size_t>(cell)];
                const int end = cell_ends[static_cast<std::size_t>(cell)];
                for (int j = begin; j < end; ++j) {
                  const std::size_t ji = static_cast<std::size_t>(j);
                  const float ddx = sx[ji] - qx;
                  const float ddy = sy[ji] - qy;
                  const float ddz = sz[ji] - qz;
                  if (ddx * ddx + ddy * ddy + ddz * ddz <= r2) {
                    const int position = __atomic_fetch_add(
                        &row_counts[j], 1, __ATOMIC_RELAXED);
                    if (position < capacity) {
                      out[ji * static_cast<std::size_t>(row_stride) +
                          static_cast<std::size_t>(position)] = neighbor;
                    }
                  }
                }
              }
            }
          }
        }
      });

}

void write_neighbors_fixed_query_pair_dense_parallel(
    NeighborSearchData &data, int capacity, IntBuffer &fixed_neighbors,
    IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(data.num_points), capacity + 8);
  if (data.num_points == 0) {
    return;
  }
  const ispc::Cells cells = cells_of(data);
  const ispc::DenseGrid grid = dense_of(data);
  const ispc::Points pts = points_of(data);
  const float r2 = data.radius * data.radius;
  write_rows_parallel(
      data.num_cells, cell_grain(data.num_cells), capacity, false, counts,
      fixed_neighbors, [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_dense(begin, end, 4, &cells, &grid, &cells,
                                     &pts, &pts, true, r2, &rows);
      });
}


void write_cross_neighbors_fixed_query_pair_dense_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, int capacity,
    IntBuffer &fixed_neighbors, IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(source_data.num_points),
                     capacity + 8);
  if (source_data.num_points == 0) {
    return;
  }
  // Sparse targets (under 4 points per cell) leave enough work per target
  // load to share it between four queries; denser ones use pairs.
  const bool sparse_target =
      target_data.num_cells > 0 && target_data.num_points > 0 &&
      target_data.max_cell_points > 0 && target_data.max_cell_points <= 64 &&
      static_cast<int64_t>(target_data.num_points) <
          4 * static_cast<int64_t>(target_data.num_cells);
  const int group = sparse_target ? 4 : 2;
  const ispc::Cells queries = cells_of(source_data);
  const ispc::DenseGrid grid = dense_of(target_data);
  const ispc::Cells targets = cells_of(target_data);
  const ispc::Points qpts = points_of(source_data);
  const ispc::Points tpts = points_of(target_data);
  const float r2 = query_data.radius * query_data.radius;
  write_rows_parallel(
      source_data.num_cells, cell_grain(source_data.num_cells), capacity,
      false, counts, fixed_neighbors,
      [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_dense(begin, end, group, &queries, &grid,
                                     &targets, &qpts, &tpts, false, r2, &rows);
      });
}

void write_cross_neighbors_fixed_query_blocked_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    const NeighborSearchData &source_data, int capacity,
    IntBuffer &fixed_neighbors, IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(query_data.num_points),
                     capacity + 8);
  if (query_data.num_points == 0) {
    return;
  }
  const ispc::Blocks src_blocks = blocks_of(source_data);
  const ispc::NeighborCells ncells =
      neighbor_cells_of(query_data, source_data.max_neighbor_cells);
  const ispc::Blocks tgt_blocks = blocks_of(target_data);
  const ispc::Points qpts = query_points_of(query_data);
  const ispc::Points tpts = points_of(target_data);
  const float r2 = query_data.radius * query_data.radius;
  const int block_count = static_cast<int>(source_data.block_begin.size());
  write_rows_parallel(
      block_count, block_grain(block_count), capacity, true, counts,
      fixed_neighbors, [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_blocked(begin, end, &src_blocks, &ncells,
                                       &tgt_blocks, &qpts, &tpts, false, r2,
                                       &rows);
      });
}


} // namespace apfrnn