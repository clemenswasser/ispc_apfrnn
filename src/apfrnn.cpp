#include "apfrnn.hpp"

#include "kernels_ispc.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
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

int gang_width() {
  static const int width = ispc::gang_width();
  return width;
}

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
    required =
        static_cast<std::size_t>(row_offset[num_points - 1] +
                                 row_capacity[num_points - 1] + row_slack());
  }
#ifndef NDEBUG
  // finish_row() uses aligned non-temporal stores of a whole gang: every row
  // must start on a gang boundary. Capacities are rounded to row_align()
  // (at least 8), and the stride adds row_slack() (one gang).
  for (std::size_t i = 0; i < num_points; ++i) {
    assert(row_offset[i] % gang_width() == 0);
    assert(row_capacity[i] % row_align() == 0);
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
bool radix_sort_point_keys(std::vector<SortPointKey> &keys,
                           std::vector<SortPointKey> &scratch) {
  const int num_points = static_cast<int>(keys.size());
  if (num_points < kRadixMinPoints || num_points >= 8000000) {
    // Very large sparse domains: comparison sort is faster than the
    // multi-pass radix histogram on the 8M path (measured 347 vs 375 ms).
    // It compares sign-flipped keys, so the result is signed-coordinate
    // order, not raw key order.
    tbb::parallel_sort(keys.begin(), keys.end());
    return true;
  }

  scratch.resize(keys.size());
  std::vector<int> histogram(static_cast<std::size_t>(kRadixChunks) *
                             kRadixBuckets);
  std::vector<int> offsets(static_cast<std::size_t>(kRadixChunks) *
                           kRadixBuckets);
  SortPointKey *src = keys.data();
  SortPointKey *dst = scratch.data();
  // A pass whose digit is the same in every key leaves the order unchanged:
  // skip it. The digits that vary are those where the OR and AND of all keys
  // differ, which one reduction finds.
  struct Bits {
    uint64_t any = 0;
    uint64_t all = ~uint64_t{0};
  };
  const Bits bits = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, num_points, 1 << 14), Bits{},
      [&](const tbb::blocked_range<int> &range, Bits acc) {
        for (int i = range.begin(); i < range.end(); ++i) {
          acc.any |= src[i].key;
          acc.all &= src[i].key;
        }
        return acc;
      },
      [](Bits a, const Bits &b) {
        a.any |= b.any;
        a.all &= b.all;
        return a;
      });
  const uint64_t varying_bits = bits.any ^ bits.all;
  for (int pass = 0; pass < 8; ++pass) {
    if (((varying_bits >> (pass * kRadixBits)) & (kRadixBuckets - 1)) == 0) {
      continue;
    }
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
  return false;
}

struct CellBounds {
  int min_cx = INT_MAX;
  int min_cy = INT_MAX;
  int min_cz = INT_MAX;
  int max_cx = INT_MIN;
  int max_cy = INT_MIN;
  int max_cz = INT_MIN;
};

inline void sort_point_set(
    const std::vector<float> &x, const std::vector<float> &y,
    const std::vector<float> &z, float inv_radius, std::vector<float> &sorted_x,
    std::vector<float> &sorted_y, std::vector<float> &sorted_z,
    IntBuffer &original_index, IntBuffer &cell_starts, IntBuffer &cell_ends,
    std::vector<SortPointKey> &point_keys,
    std::vector<SortPointKey> &point_keys_alt,
    std::vector<uint64_t> &unique_keys, bool &signed_order,
    CellBounds *bounds_out) {
  int const num_points = static_cast<int>(x.size());
  point_keys.resize(num_points);
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
  if (bounds_out != nullptr) {
    *bounds_out = bounds;
  }

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

    signed_order = true;
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

  signed_order = radix_sort_point_keys(point_keys, point_keys_alt);

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

// Chunked stable compaction: writes the emitting indices in ascending order.
template <typename Pred, typename Emit>
int parallel_compact(int count, int grain, Pred &&pred, Emit &&emit) {
  const int chunks = (count + grain - 1) / grain;
  std::vector<int> offsets(static_cast<std::size_t>(chunks) + 1, 0);
  tbb::parallel_for(0, chunks, [&](int chunk) {
    const int begin = chunk * grain;
    const int end = std::min(count, begin + grain);
    int local = 0;
    for (int i = begin; i < end; ++i) {
      local += pred(i) ? 1 : 0;
    }
    offsets[static_cast<std::size_t>(chunk)] = local;
  });
  int total = 0;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int value = offsets[static_cast<std::size_t>(chunk)];
    offsets[static_cast<std::size_t>(chunk)] = total;
    total += value;
  }
  offsets[static_cast<std::size_t>(chunks)] = total;
  tbb::parallel_for(0, chunks, [&](int chunk) {
    const int begin = chunk * grain;
    const int end = std::min(count, begin + grain);
    int write = offsets[static_cast<std::size_t>(chunk)];
    for (int i = begin; i < end; ++i) {
      if (pred(i)) {
        emit(write++, i);
      }
    }
  });
  return total;
}

// Reuses the previous cell order and merges only the points whose cell key
// changed. The unchanged stream is the previous sorted order with changed
// points removed; both streams are sorted with the previous frame's
// comparator, and a parallel merge path splits the output into one range per
// worker. Returns false when a full sort is needed (first build, radius or
// point count change, or too many changed cells).
bool try_incremental_sort_point_set(NeighborSearchData &data,
                                    const std::vector<float> &x,
                                    const std::vector<float> &y,
                                    const std::vector<float> &z,
                                    float inv_radius) {
  const int num_points = static_cast<int>(x.size());
  if (!data.incremental_valid || num_points == 0 ||
      data.incremental_points != num_points ||
      data.incremental_inv_radius != inv_radius) {
    return false;
  }
  // The merge wins when the grid is sparse: few cells change and the full
  // sort dominates. Dense SPH grids change a large fraction of cells and the
  // merge plus change detection loses to the radix/comparison sort, so keep
  // those sets on the full path.
  if (data.max_cell_points > 1) {
    return false;
  }
  auto &current = data.scratch_point_keys;
  current.resize(static_cast<std::size_t>(num_points));
  auto &changed = data.incremental_changed;
  changed.resize(static_cast<std::size_t>(num_points));
  uint64_t *previous_keys = data.incremental_keys.data();
  struct KeyPass {
    CellBounds bounds;
    int changed = 0;
  };
  const KeyPass pass = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, num_points, 2048), KeyPass{},
      [&](const tbb::blocked_range<int> &range, KeyPass acc) {
        for (int i = range.begin(); i < range.end(); ++i) {
          const std::size_t at = static_cast<std::size_t>(i);
          const int cx = floor_to_cell(x[at] * inv_radius);
          const int cy = floor_to_cell(y[at] * inv_radius);
          const int cz = floor_to_cell(z[at] * inv_radius);
          const uint64_t key = make_key(cx, cy, cz);
          current[at] = SortPointKey{key, i};
          const bool differs = key != previous_keys[at];
          changed[at] = differs ? 1 : 0;
          acc.changed += differs ? 1 : 0;
          previous_keys[at] = key;
          acc.bounds.min_cx = std::min(acc.bounds.min_cx, cx);
          acc.bounds.min_cy = std::min(acc.bounds.min_cy, cy);
          acc.bounds.min_cz = std::min(acc.bounds.min_cz, cz);
          acc.bounds.max_cx = std::max(acc.bounds.max_cx, cx);
          acc.bounds.max_cy = std::max(acc.bounds.max_cy, cy);
          acc.bounds.max_cz = std::max(acc.bounds.max_cz, cz);
        }
        return acc;
      },
      [](KeyPass a, const KeyPass &b) {
        a.changed += b.changed;
        a.bounds.min_cx = std::min(a.bounds.min_cx, b.bounds.min_cx);
        a.bounds.min_cy = std::min(a.bounds.min_cy, b.bounds.min_cy);
        a.bounds.min_cz = std::min(a.bounds.min_cz, b.bounds.min_cz);
        a.bounds.max_cx = std::max(a.bounds.max_cx, b.bounds.max_cx);
        a.bounds.max_cy = std::max(a.bounds.max_cy, b.bounds.max_cy);
        a.bounds.max_cz = std::max(a.bounds.max_cz, b.bounds.max_cz);
        return a;
      });
  const int changed_count = pass.changed;
  data.cell_coord_min_x = pass.bounds.min_cx;
  data.cell_coord_max_x = pass.bounds.max_cx;
  data.cell_coord_min_y = pass.bounds.min_cy;
  data.cell_coord_max_y = pass.bounds.max_cy;
  data.cell_coord_min_z = pass.bounds.min_cz;
  data.cell_coord_max_z = pass.bounds.max_cz;
  data.cell_coord_valid = true;

  if (changed_count == 0) {
    // Same cells, new coordinates: refresh the packed coordinate arrays and
    // keep the order and cell ranges.
    const int *order = data.original_index.data();
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_points, 2048),
        [&](const tbb::blocked_range<int> &range) {
          for (int i = range.begin(); i < range.end(); ++i) {
            const std::size_t at = static_cast<std::size_t>(i);
            const std::size_t idx = static_cast<std::size_t>(order[at]);
            data.sorted_x[at] = x[idx];
            data.sorted_y[at] = y[idx];
            data.sorted_z[at] = z[idx];
          }
        });
    return true;
  }
  if (2 * changed_count > num_points) {
    return false;
  }

  const bool signed_order = data.cell_keys_signed_order;
  const uint64_t sign_bits = SortPointKey::kSignBits;
  auto ordered_key = [&](uint64_t key) {
    return signed_order ? (key ^ sign_bits) : key;
  };
  auto less = [&](const SortPointKey &a, const SortPointKey &b) {
    const uint64_t ka = ordered_key(a.key);
    const uint64_t kb = ordered_key(b.key);
    return ka != kb ? ka < kb : a.index < b.index;
  };

  auto &changed_keys = data.incremental_changed_keys;
  changed_keys.resize(static_cast<std::size_t>(changed_count));
  parallel_compact(
      num_points, 16384, [&](int i) { return changed[static_cast<std::size_t>(i)] != 0; },
      [&](int write, int i) {
        changed_keys[static_cast<std::size_t>(write)] =
            current[static_cast<std::size_t>(i)];
      });
  tbb::parallel_sort(changed_keys.begin(), changed_keys.end(), less);

  const int unchanged_count = num_points - changed_count;
  auto &unchanged = data.incremental_unchanged_keys;
  unchanged.resize(static_cast<std::size_t>(unchanged_count));
  const int *previous_order = data.original_index.data();
  const uint64_t *previous_sorted = data.incremental_keys_are_unique
                                        ? data.scratch_unique_keys.data()
                                        : data.incremental_sorted_keys.data();
  parallel_compact(
      num_points, 16384,
      [&](int i) {
        return changed[static_cast<std::size_t>(
                   previous_order[i])] == 0;
      },
      [&](int write, int i) {
        const int idx = previous_order[i];
        unchanged[static_cast<std::size_t>(write)] =
            SortPointKey{previous_sorted[static_cast<std::size_t>(i)], idx};
      });

  data.incremental_x.resize(static_cast<std::size_t>(num_points));
  data.incremental_y.resize(static_cast<std::size_t>(num_points));
  data.incremental_z.resize(static_cast<std::size_t>(num_points));
  data.incremental_order.resize(static_cast<std::size_t>(num_points));
  data.incremental_merge_keys.resize(static_cast<std::size_t>(num_points));
  float *out_x = data.incremental_x.data();
  float *out_y = data.incremental_y.data();
  float *out_z = data.incremental_z.data();
  int *out_order = data.incremental_order.data();
  uint64_t *out_keys = data.incremental_merge_keys.data();

  const int num_a = unchanged_count;
  const int num_b = changed_count;
  const SortPointKey *a = unchanged.data();
  const SortPointKey *b = changed_keys.data();
  const int workers =
      std::max(1, static_cast<int>(tbb::this_task_arena::max_concurrency()));
  std::vector<int> split_a(static_cast<std::size_t>(workers) + 1, 0);
  std::vector<int> split_b(static_cast<std::size_t>(workers) + 1, 0);
  for (int worker = 0; worker <= workers; ++worker) {
    const int rank = static_cast<int>(
        static_cast<long long>(worker) * (num_a + num_b) / workers);
    int lo = std::max(0, rank - num_b);
    int hi = std::min(rank, num_a);
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      const int other = rank - mid;
      if (less(a[mid], b[other - 1])) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    split_a[static_cast<std::size_t>(worker)] = lo;
    split_b[static_cast<std::size_t>(worker)] = rank - lo;
  }
  tbb::parallel_for(0, workers, [&](int worker) {
    int ia = split_a[static_cast<std::size_t>(worker)];
    int ib = split_b[static_cast<std::size_t>(worker)];
    const int end_a = split_a[static_cast<std::size_t>(worker) + 1];
    const int end_b = split_b[static_cast<std::size_t>(worker) + 1];
    int write = ia + ib;
    while (ia < end_a && ib < end_b) {
      const bool take_a = less(a[ia], b[ib]);
      const SortPointKey &source = take_a ? a[ia] : b[ib];
      if (take_a) {
        ++ia;
      } else {
        ++ib;
      }
      const int idx = source.index;
      out_keys[write] = source.key;
      out_x[write] = x[static_cast<std::size_t>(idx)];
      out_y[write] = y[static_cast<std::size_t>(idx)];
      out_z[write] = z[static_cast<std::size_t>(idx)];
      out_order[write] = idx;
      ++write;
    }
    for (; ia < end_a; ++ia) {
      const int idx = a[ia].index;
      out_keys[write] = a[ia].key;
      out_x[write] = x[static_cast<std::size_t>(idx)];
      out_y[write] = y[static_cast<std::size_t>(idx)];
      out_z[write] = z[static_cast<std::size_t>(idx)];
      out_order[write] = idx;
      ++write;
    }
    for (; ib < end_b; ++ib) {
      const int idx = b[ib].index;
      out_keys[write] = b[ib].key;
      out_x[write] = x[static_cast<std::size_t>(idx)];
      out_y[write] = y[static_cast<std::size_t>(idx)];
      out_z[write] = z[static_cast<std::size_t>(idx)];
      out_order[write] = idx;
      ++write;
    }
  });

  // Cell ranges from the merged key order. Position 0 and every key change
  // starts a cell; chunks count locally, a short prefix assigns cell ids.
  constexpr int kRangeGrain = 16384;
  const int chunks = (num_points + kRangeGrain - 1) / kRangeGrain;
  // One point per cell is the common sparse case: starts/ends are the index
  // sequence and the merged key buffer doubles as the unique-key array (and
  // as next frame's per-position order).
  std::vector<uint8_t> chunk_duplicate(static_cast<std::size_t>(chunks), 0);
  tbb::parallel_for(0, chunks, [&](int chunk) {
    const int begin = chunk * kRangeGrain;
    const int end = std::min(num_points, begin + kRangeGrain);
    uint8_t duplicate = 0;
    for (int i = begin + 1; i < end; ++i) {
      duplicate |= out_keys[i] == out_keys[i - 1] ? 1 : 0;
    }
    if (begin > 0) {
      duplicate |= out_keys[begin] == out_keys[begin - 1] ? 1 : 0;
    }
    chunk_duplicate[static_cast<std::size_t>(chunk)] = duplicate;
  });
  bool all_unique = true;
  for (uint8_t duplicate : chunk_duplicate) {
    all_unique &= duplicate == 0;
  }
  if (all_unique) {
    data.incremental_starts.resize(static_cast<std::size_t>(num_points));
    data.incremental_ends.resize(static_cast<std::size_t>(num_points));
    int *starts = data.incremental_starts.data();
    int *ends = data.incremental_ends.data();
    tbb::parallel_for(0, chunks, [&](int chunk) {
      const int begin = chunk * kRangeGrain;
      const int end = std::min(num_points, begin + kRangeGrain);
      for (int i = begin; i < end; ++i) {
        starts[i] = i;
        ends[i] = i + 1;
      }
    });
    data.sorted_x.swap(data.incremental_x);
    data.sorted_y.swap(data.incremental_y);
    data.sorted_z.swap(data.incremental_z);
    data.original_index.swap(data.incremental_order);
    data.cell_starts.swap(data.incremental_starts);
    data.cell_ends.swap(data.incremental_ends);
    data.scratch_unique_keys.swap(data.incremental_merge_keys);
    data.incremental_keys_are_unique = true;
    data.cell_keys_signed_order = signed_order;
    return true;
  }
  data.incremental_keys_are_unique = false;
  std::vector<int> chunk_cells(static_cast<std::size_t>(chunks) + 1, 0);
  tbb::parallel_for(0, chunks, [&](int chunk) {
    const int begin = chunk * kRangeGrain;
    const int end = std::min(num_points, begin + kRangeGrain);
    int cells = 0;
    for (int i = begin; i < end; ++i) {
      if (i == 0 || out_keys[i] != out_keys[i - 1]) {
        ++cells;
      }
    }
    chunk_cells[static_cast<std::size_t>(chunk)] = cells;
  });
  int num_cells = 0;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int cells = chunk_cells[static_cast<std::size_t>(chunk)];
    chunk_cells[static_cast<std::size_t>(chunk)] = num_cells;
    num_cells += cells;
  }
  data.incremental_starts.resize(static_cast<std::size_t>(num_cells));
  data.incremental_ends.resize(static_cast<std::size_t>(num_cells));
  data.incremental_unique_keys.resize(static_cast<std::size_t>(num_cells));
  int *starts = data.incremental_starts.data();
  int *ends = data.incremental_ends.data();
  uint64_t *unique = data.incremental_unique_keys.data();
  tbb::parallel_for(0, chunks, [&](int chunk) {
    const int begin = chunk * kRangeGrain;
    const int end = std::min(num_points, begin + kRangeGrain);
    int cell = chunk_cells[static_cast<std::size_t>(chunk)];
    for (int i = begin; i < end; ++i) {
      if (i == 0 || out_keys[i] != out_keys[i - 1]) {
        if (cell > 0) {
          ends[cell - 1] = i;
        }
        starts[cell] = i;
        unique[cell] = out_keys[i];
        ++cell;
      }
    }
  });
  ends[num_cells - 1] = num_points;

  data.sorted_x.swap(data.incremental_x);
  data.sorted_y.swap(data.incremental_y);
  data.sorted_z.swap(data.incremental_z);
  data.original_index.swap(data.incremental_order);
  data.cell_starts.swap(data.incremental_starts);
  data.cell_ends.swap(data.incremental_ends);
  data.scratch_unique_keys.swap(data.incremental_unique_keys);
  data.incremental_sorted_keys.swap(data.incremental_merge_keys);
  data.cell_keys_signed_order = signed_order;
  return true;
}

// Captures the state the incremental merge needs after a full sort.
void reset_incremental_state(NeighborSearchData &data,
                             const std::vector<float> &x,
                             const std::vector<float> &y,
                             const std::vector<float> &z, float inv_radius) {
  const int num_points = static_cast<int>(x.size());
  data.incremental_keys.resize(static_cast<std::size_t>(num_points));
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, 2048),
      [&](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
          const std::size_t at = static_cast<std::size_t>(i);
          data.incremental_keys[at] =
              make_key(floor_to_cell(x[at] * inv_radius),
                       floor_to_cell(y[at] * inv_radius),
                       floor_to_cell(z[at] * inv_radius));
        }
      });
  data.incremental_sorted_keys.resize(static_cast<std::size_t>(num_points));
  const int *order = data.original_index.data();
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, 2048),
      [&](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
          const std::size_t at = static_cast<std::size_t>(i);
          data.incremental_sorted_keys[at] =
              data.incremental_keys[static_cast<std::size_t>(order[at])];
        }
      });
  data.incremental_keys_are_unique = false;
  data.incremental_points = num_points;
  data.incremental_inv_radius = inv_radius;
  data.incremental_valid = true;
}

inline void build_hash_table(const std::vector<uint64_t> &unique_keys,
                             std::vector<uint64_t> &ht_keys, IntBuffer &ht_vals,
                             uint64_t empty_key, uint32_t &hash_mask,
                             std::vector<uint8_t> &control) {
  int hash_capacity = 1;
  while (hash_capacity < static_cast<int>(unique_keys.size()) * 2) {
    hash_capacity <<= 1;
  }

  hash_mask = static_cast<uint32_t>(hash_capacity - 1);
  ht_keys.assign(hash_capacity, empty_key);
  ht_vals.assign(hash_capacity, -1);
  control.assign(hash_capacity, 0x80);

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
    control[h] = static_cast<uint8_t>(h64 >> 56) & 0x7f;
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
  if (data.cell_coord_valid) {
    min_x = data.cell_coord_min_x;
    max_x = data.cell_coord_max_x;
    min_y = data.cell_coord_min_y;
    max_y = data.cell_coord_max_y;
    min_z = data.cell_coord_min_z;
    max_z = data.cell_coord_max_z;
  } else {
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
    data.cell_coord_min_x = min_x;
    data.cell_coord_max_x = max_x;
    data.cell_coord_min_y = min_y;
    data.cell_coord_max_y = max_y;
    data.cell_coord_min_z = min_z;
    data.cell_coord_max_z = max_z;
    data.cell_coord_valid = true;
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
          data.hash_mask, data.hash_control.data()};
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
  const int stride = capacity + row_slack();
  const bool staged =
      allow_staging && capacity >= 256 && capacity % row_align() == 0;
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
  const int stride = scratch_capacity + row_slack();
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
  const int stride = capacity + row_slack();
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

  ensure_hash_table(const_cast<NeighborSearchData &>(target));
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

// One-pass 9-probe stencil runs for a hash-grid target. This is the cross-set
// analogue of build_hash_stencil_ranges(): the source's sorted cell keys are
// queried against the target's cell array/hash, and the merged point runs are
// written directly. It removes the 27-probe collect_neighbor_cells buffer and
// the following count/scan/merge passes for the common stencil==1 case.
inline void build_cross_hash_stencil_ranges_impl(
    CrossNeighborSearchData &data,
    const std::vector<uint64_t> &source_unique_keys,
    const NeighborSearchData &target_data) {
  ensure_hash_table(const_cast<NeighborSearchData &>(target_data));
  use_own_ranges(data);
  const int num_cells = data.num_cells;
  const int slots = ispc::hash_stencil_runs_per_cell_cross();
  const std::size_t total = static_cast<std::size_t>(num_cells) * slots;
  data.cell_num_neighbors.resize(static_cast<std::size_t>(num_cells));
  data.cell_neighbor_offset.resize(static_cast<std::size_t>(num_cells));
  data.cell_neighbor_starts.resize(total);
  data.cell_neighbor_ends.resize(total);
  if (num_cells == 0) {
    return;
  }
  parallel_cells(0, num_cells, cell_grain(num_cells),
                 [&](int begin, int end) {
                   const ispc::Cells targets = cells_of(target_data);
                   const ispc::HashGrid hash = hash_of(target_data);
                   ispc::hash_stencil_ranges_cross(
                       begin, end, source_unique_keys.data(), &hash, &targets,
                       target_data.num_cells,
                       data.cell_neighbor_offset.data(),
                       data.cell_num_neighbors.data(),
                       data.cell_neighbor_starts.data(),
                       data.cell_neighbor_ends.data());
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
  if (!target_data.dense_valid && stencil == 1 &&
      target_data.cell_stencil == 1 && target_data.num_points > 0 &&
      target_data.num_cells > 0 && !target_data.hash_keys.empty()) {
    build_cross_hash_stencil_ranges_impl(data, source_unique_keys, target_data);
    return;
  }
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

void ensure_hash_table(NeighborSearchData &data) {
  if (data.dense_valid || data.hash_valid || data.num_cells == 0) {
    return;
  }
  build_hash_table(data.scratch_unique_keys, data.hash_keys, data.hash_vals,
                   data.empty_key, data.hash_mask, data.hash_control);
  data.hash_valid = true;
}

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

  const int num_cells = data.num_cells;
  const int num_points = data.num_points;
  int max_cell_points = 0;
  for (int cell = 0; cell < num_cells; ++cell) {
    max_cell_points = std::max(
        max_cell_points,
        data.cell_ends[static_cast<std::size_t>(cell)] -
            data.cell_starts[static_cast<std::size_t>(cell)]);
  }
  data.max_cell_points = max_cell_points;
  if (num_cells < 8 || num_points == 0 ||
      max_cell_points <= kBlockThreshold) {
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

  // The cell-x slab index describes the previous sorted order.
  data.cell_x_valid = false;
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
    data.hash_control.clear();
    data.hash_mask = 0;
    data.hash_valid = false;
    data.cell_keys_signed_order = false;
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
  const bool incremental =
      try_incremental_sort_point_set(data, x, y, z, inv_radius);
  if (!incremental) {
    CellBounds bounds;
    sort_point_set(x, y, z, inv_radius, data.sorted_x, data.sorted_y,
                   data.sorted_z, data.original_index, data.cell_starts,
                   data.cell_ends, data.scratch_point_keys,
                   data.scratch_point_keys_alt, data.scratch_unique_keys,
                   data.cell_keys_signed_order, &bounds);
    data.cell_coord_min_x = bounds.min_cx;
    data.cell_coord_max_x = bounds.max_cx;
    data.cell_coord_min_y = bounds.min_cy;
    data.cell_coord_max_y = bounds.max_cy;
    data.cell_coord_min_z = bounds.min_cz;
    data.cell_coord_max_z = bounds.max_cz;
    data.cell_coord_valid = true;
  }
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
  int max_cell_points = 0;
  for (int cell = 0; cell < num_cells; ++cell) {
    max_cell_points = std::max(
        max_cell_points,
        data.cell_ends[static_cast<std::size_t>(cell)] -
            data.cell_starts[static_cast<std::size_t>(cell)]);
  }
  data.max_cell_points = max_cell_points;
  if (!incremental) {
    // The merge only targets sparse grids; dense sets keep the full-sort
    // path and must not pay to capture merge state they will never use.
    if (max_cell_points <= 1) {
      reset_incremental_state(data, x, y, z, inv_radius);
    } else {
      data.incremental_valid = false;
    }
  }
  if (!data.dense_valid && max_cell_points > 1) {
    build_hash_table(unique_keys, data.hash_keys, data.hash_vals,
                     data.empty_key, data.hash_mask, data.hash_control);
    data.hash_valid = true;
  } else {
    data.hash_keys.clear();
    data.hash_vals.clear();
    data.hash_control.clear();
    data.hash_mask = 0;
    data.hash_valid = false;
  }
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

// Hash grid with the 27-cell stencil: one pass of hash_stencil_ranges writes
// every cell's merged runs into fixed slots, instead of collecting the 27
// neighbor cells and merging them in a count pass, a scan and a write pass.
static void build_hash_stencil_ranges(NeighborSearchData &data) {
  ensure_hash_table(data);
  const int num_cells = data.num_cells;
  const bool reuse =
      data.cached_range_valid && data.cached_range_stencil == -1 &&
      data.hash_range_keys == data.scratch_unique_keys &&
      data.cached_range_cell_starts == data.cell_starts &&
      data.cached_range_cell_ends == data.cell_ends &&
      data.cell_num_neighbors.size() == static_cast<std::size_t>(num_cells);
  if (reuse) {
    return;
  }
  const int slots = ispc::hash_stencil_runs_per_cell();
  const std::size_t total = static_cast<std::size_t>(num_cells) * slots;
  data.cell_num_neighbors.resize(static_cast<std::size_t>(num_cells));
  data.cell_neighbor_offset.resize(static_cast<std::size_t>(num_cells));
  data.cell_neighbor_starts.resize(total);
  data.cell_neighbor_ends.resize(total);
  parallel_cells(0, num_cells, cell_grain(num_cells), [&](int begin, int end) {
    const ispc::Cells cells = cells_of(data);
    const ispc::HashGrid hash = hash_of(data);
    ispc::hash_stencil_ranges(
        begin, end, &cells, &hash, &cells, num_cells,
        data.cell_neighbor_offset.data(), data.cell_num_neighbors.data(),
        data.cell_neighbor_starts.data(), data.cell_neighbor_ends.data());
  });
  data.hash_range_keys = data.scratch_unique_keys;
  data.cached_range_cell_starts = data.cell_starts;
  data.cached_range_cell_ends = data.cell_ends;
  // -1 marks these ranges; the merged path below never reuses them.
  data.cached_range_stencil = -1;
  data.cached_range_total = static_cast<int>(total);
  data.cached_range_valid = true;
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
  if (!data.dense_valid && data.cell_stencil == 1) {
    build_hash_stencil_ranges(data);
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
                     static_cast<std::size_t>(data.num_points),
                     capacity + row_slack());
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
                     static_cast<std::size_t>(data.num_points),
                     capacity + row_slack());
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
                     static_cast<std::size_t>(data.num_points),
                     capacity + row_slack());
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

// Same-set rows for a grid where every cell holds at most one point. The
// packed cell keys are lexicographic in (x, y, z), so the 13 forward cells of
// the 3x3x3 stencil can be enumerated by merging the sorted key array with
// itself at a constant key offset. Each cursor only moves forward as the query
// cell advances; chunk starts seed it with one binary search. No per-cell hash
// probe and no per-cell range table is needed. The two rows of every pair are
// appended symmetrically with atomic increments, which is cheapest for the
// sparse sets this path targets (their rows are almost always empty).
bool write_sparse_same_rows(NeighborSearchData &data, int capacity,
                            IntBuffer &fixed_neighbors, IntBuffer &counts) {
  const int stride = capacity;
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(data.num_points), stride);
  if (data.num_points == 0) {
    return true;
  }
  std::fill(counts.begin(), counts.end(), 0);
  const int num_cells = data.num_cells;
  const uint64_t *keys = data.scratch_unique_keys.data();
  const int *cell_starts = data.cell_starts.data();
  const float *sx = data.sorted_x.data();
  const float *sy = data.sorted_y.data();
  const float *sz = data.sorted_z.data();
  const int *original = data.original_index.data();
  const float r2 = data.radius * data.radius;
  int *row_counts = counts.data();
  int *row_values = fixed_neighbors.data();
  // The counting and comparison sorts order cells by signed coordinates; the
  // radix sort orders the raw packed keys. Flip the sign bit of every field
  // for the former so lower_bound sees the order the array was built with.
  const bool signed_order = data.cell_keys_signed_order;
  auto order_key = [&](uint64_t key) {
    return signed_order ? (key ^ SortPointKey::kSignBits) : key;
  };

  auto lower_bound_key = [&](uint64_t want) {
    return static_cast<int>(
        std::lower_bound(keys, keys + num_cells, want,
                         [&](uint64_t value, uint64_t target) {
                           return order_key(value) < target;
                         }) -
        keys);
  };
  auto append = [&](int row, int value) {
    const int slot =
        __atomic_fetch_add(&row_counts[row], 1, __ATOMIC_RELAXED);
    if (slot < capacity) {
      row_values[static_cast<std::size_t>(row) * stride + slot] = value;
    }
  };
  auto test_cell = [&](int query, int other_cell) {
    const int other = cell_starts[other_cell];
    if (other == query) {
      return;
    }
    const float dx = sx[other] - sx[query];
    const float dy = sy[other] - sy[query];
    const float dz = sz[other] - sz[query];
    if (dx * dx + dy * dy + dz * dz <= r2) {
      append(query, original[other]);
      append(other, original[query]);
    }
  };

  constexpr int kGrain = 1 << 16;
  struct Group {
    int dx;
    int dy;
  };
  // (0,0,+1) is the immediately following key. Each remaining direction is
  // one cursor; its three z offsets are adjacent keys around the cursor.
  static constexpr Group kGroups[4] = {
      {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_cells, kGrain),
      [&](const tbb::blocked_range<int> &range) {
        const int begin = range.begin();
        const int end = range.end();
        int cursor[4];
        uint64_t last[4];
        for (int k = 0; k < 4; ++k) {
          const uint64_t key = keys[begin];
          last[k] = order_key(make_key(unpack_coord(key, 42) + kGroups[k].dx,
                                       unpack_coord(key, 21) + kGroups[k].dy,
                                       unpack_coord(key, 0)));
          cursor[k] = lower_bound_key(last[k]);
        }
        for (int cell = begin; cell < end; ++cell) {
          const int query = cell_starts[cell];
          const uint64_t key = keys[cell];
          const int cx = unpack_coord(key, 42);
          const int cy = unpack_coord(key, 21);
          const int cz = unpack_coord(key, 0);
          // Same (x, y), next z: the next occupied position when the key is
          // exactly one greater.
          if (signed_order || (cz >= 0 && cz < 0xFFFFF)) {
            const uint64_t want =
                order_key(make_key(cx, cy, cz + 1));
            const int next = cell + 1;
            if (next < num_cells && order_key(keys[next]) == want) {
              test_cell(query, next);
            }
          } else if (cz < 0) {
            const uint64_t want =
                order_key(make_key(cx, cy, cz + 1));
            const int found = lower_bound_key(want);
            if (found < num_cells && order_key(keys[found]) == want) {
              test_cell(query, found);
            }
          }
          for (int k = 0; k < 4; ++k) {
            const uint64_t want = order_key(
                make_key(cx + kGroups[k].dx, cy + kGroups[k].dy, cz));
            int &position = cursor[k];
            if (want < last[k]) {
              position = lower_bound_key(want);
            } else {
              while (position < num_cells &&
                     order_key(keys[position]) < want) {
                ++position;
              }
            }
            last[k] = want;
            const uint64_t key_neg = order_key(
                make_key(cx + kGroups[k].dx, cy + kGroups[k].dy, cz - 1));
            const uint64_t key_pos = order_key(
                make_key(cx + kGroups[k].dx, cy + kGroups[k].dy, cz + 1));
            const bool adjacent = (signed_order ||
                                   (cz > 0 && cz < 0xFFFFF)) &&
                                  key_neg == want - 1 && key_pos == want + 1;
            if (adjacent) {
              if (position < num_cells &&
                  order_key(keys[position]) == want) {
                test_cell(query, position);
                if (position > 0 &&
                    order_key(keys[position - 1]) == want - 1) {
                  test_cell(query, position - 1);
                }
                if (position + 1 < num_cells &&
                    order_key(keys[position + 1]) == want + 1) {
                  test_cell(query, position + 1);
                }
              } else {
                if (position > 0 &&
                    order_key(keys[position - 1]) == want - 1) {
                  test_cell(query, position - 1);
                }
                if (position < num_cells &&
                    order_key(keys[position]) == want + 1) {
                  test_cell(query, position);
                }
              }
            } else {
              for (int dz = -1; dz <= 1; ++dz) {
                const int z = cz + dz;
                if (z < -0x100000 || z > 0xFFFFF) {
                  continue;
                }
                const uint64_t exact = order_key(
                    make_key(cx + kGroups[k].dx, cy + kGroups[k].dy, z));
                const int found = lower_bound_key(exact);
                if (found < num_cells &&
                    order_key(keys[found]) == exact) {
                  test_cell(query, found);
                }
              }
            }
          }
        }
      });
  return max_row_count(counts) <= capacity;
}

// Builds the cell-x slab index and the cell y/z bounds. The cell keys must be
// in lexicographic (x, y, z) order with non-negative coordinates; callers gate
// on that, so raw key order is also coordinate order.
void ensure_cell_x_index(NeighborSearchData &data) {
  if (data.cell_x_valid) {
    return;
  }
  const int num_cells = data.num_cells;
  if (num_cells <= 0) {
    data.cell_x_start.clear();
    return;
  }
  const uint64_t *keys = data.scratch_unique_keys.data();
  const int min_x = unpack_coord(keys[0], 42);
  const int max_x = unpack_coord(keys[num_cells - 1], 42);
  const int span = max_x - min_x + 1;
  if (span <= 0 || span > 4 * num_cells + 65536) {
    return;
  }
  data.cell_x_min = min_x;
  data.cell_x_start.resize(static_cast<std::size_t>(span) + 1);
  int cell = 0;
  for (int i = 0; i < span; ++i) {
    const int x = min_x + i;
    while (cell < num_cells && unpack_coord(keys[cell], 42) < x) {
      ++cell;
    }
    data.cell_x_start[static_cast<std::size_t>(i)] = cell;
  }
  data.cell_x_start[static_cast<std::size_t>(span)] = num_cells;
  int min_y = INT_MAX;
  int max_y = INT_MIN;
  int min_z = INT_MAX;
  int max_z = INT_MIN;
  for (int i = 0; i < num_cells; ++i) {
    const int y = unpack_coord(keys[i], 21);
    const int z = unpack_coord(keys[i], 0);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
  }
  data.cell_range_min_y = min_y;
  data.cell_range_max_y = max_y;
  data.cell_range_min_z = min_z;
  data.cell_range_max_z = max_z;
  data.cell_x_valid = true;
}

bool write_cross_rows_cell_range(NeighborSearchData &source,
                                 NeighborSearchData &target, float radius,
                                 int capacity, IntBuffer &fixed_neighbors,
                                 IntBuffer &counts) {
  ensure_cell_x_index(target);
  if (!target.cell_x_valid || target.num_cells <= 0) {
    return false;
  }
  const int num_points = source.num_points;
  const int stride = capacity;
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(num_points), stride);
  if (num_points == 0) {
    return true;
  }
  const float r2 = radius * radius;
  const float inv_cell =
      target.cell_radius > 0.0f ? 1.0f / target.cell_radius : 1.0f;
  const int stencil =
      std::max(1, static_cast<int>(std::ceil(radius * inv_cell - 1.0e-6f)));
  const int num_cells = target.num_cells;
  const uint64_t *keys = target.scratch_unique_keys.data();
  const int *cell_starts = target.cell_starts.data();
  const int *cell_ends = target.cell_ends.data();
  const float *tx = target.sorted_x.data();
  const float *ty = target.sorted_y.data();
  const float *tz = target.sorted_z.data();
  const int *target_index = target.original_index.data();
  const float *qx = source.sorted_x.data();
  const float *qy = source.sorted_y.data();
  const float *qz = source.sorted_z.data();
  const int *cell_x_start = target.cell_x_start.data();
  const int x_min = target.cell_x_min;
  const int x_span = static_cast<int>(target.cell_x_start.size()) - 1;
  const int y_min = target.cell_range_min_y;
  const int y_max = target.cell_range_max_y;
  const int z_min = target.cell_range_min_z;
  const int z_max = target.cell_range_max_z;
  // A one-point-per-cell source already carries its cell key. When the target
  // uses the same cell edge, reuse the key instead of recomputing the cell
  // coordinates from the float position for every query.
  const bool source_cell_keys =
      source.max_cell_points <= 1 && source.num_cells == num_points &&
      source.scratch_unique_keys.size() ==
          static_cast<std::size_t>(num_points) &&
      source.cell_radius == target.cell_radius;
  const uint64_t *query_keys =
      source_cell_keys ? source.scratch_unique_keys.data() : nullptr;
  int *row_counts = counts.data();
  int *row_values = fixed_neighbors.data();

  auto lower = [&](int begin, int end, uint64_t want) {
    return static_cast<int>(std::lower_bound(keys + begin, keys + end, want) -
                            keys);
  };

  constexpr int kGrain = 2048;
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, kGrain),
      [&](const tbb::blocked_range<int> &range) {
        for (int point = range.begin(); point < range.end(); ++point) {
          const float px = qx[point];
          const float py = qy[point];
          const float pz = qz[point];
          int cx;
          int cy;
          int cz;
          if (query_keys != nullptr) {
            const uint64_t key =
                query_keys[static_cast<std::size_t>(point)];
            cx = unpack_coord(key, 42);
            cy = unpack_coord(key, 21);
            cz = unpack_coord(key, 0);
          } else {
            cx = floor_to_cell(px * inv_cell);
            cy = floor_to_cell(py * inv_cell);
            cz = floor_to_cell(pz * inv_cell);
          }
          int count = 0;
          int x0 = std::max(cx - stencil, x_min);
          int x1 = std::min(cx + stencil, x_min + x_span - 1);
          for (int tcx = x0; tcx <= x1; ++tcx) {
            const int xb = cell_x_start[tcx - x_min];
            const int xe = cell_x_start[tcx - x_min + 1];
            if (xb >= xe) {
              continue;
            }
            const int y0 = std::max(cy - stencil, y_min);
            const int y1 = std::min(cy + stencil, y_max);
            if (y0 > y1) {
              continue;
            }
            const uint64_t y_low = make_key(tcx, y0, 0);
            const int yb = lower(xb, xe, y_low);
            const int ye = y1 == 0xFFFFF
                               ? xe
                               : lower(xb, xe, make_key(tcx, y1 + 1, 0));
            for (int cell = yb; cell < ye;) {
              const int y = unpack_coord(keys[cell], 21);
              if (y > y1) {
                break;
              }
              const int y_end =
                  y == 0xFFFFF
                      ? ye
                      : lower(cell, ye, make_key(tcx, y + 1, 0));
              const int z0 = std::max(cz - stencil, z_min);
              const int z1 = std::min(cz + stencil, z_max);
              if (z0 <= z1) {
                const int zb = lower(cell, y_end, make_key(tcx, y, z0));
                const int ze =
                    z1 == 0xFFFFF
                        ? y_end
                        : lower(zb, y_end, make_key(tcx, y, z1 + 1));
                for (int c = zb; c < ze; ++c) {
                  const int begin = cell_starts[c];
                  const int end = cell_ends[c];
                  for (int p = begin; p < end; ++p) {
                    const float dx = tx[p] - px;
                    const float dy = ty[p] - py;
                    const float dz = tz[p] - pz;
                    if (dx * dx + dy * dy + dz * dz <= r2) {
                      if (count < capacity) {
                        row_values[static_cast<std::size_t>(point) * stride +
                                   count] = target_index[p];
                      }
                      ++count;
                    }
                  }
                }
              }
              cell = y_end;
            }
          }
          row_counts[point] = count;
        }
      });
  return max_row_count(counts) <= capacity;
}

void write_neighbors_fixed_query_blocked_parallel(
    NeighborSearchData &data, int capacity, IntBuffer &fixed_neighbors,
    IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(data.num_points),
                     capacity + row_slack());
  if (data.num_points == 0) {
    return;
  }
  // The hash-grid range builder does not collect candidate cells, so build
  // (or reuse) the ones this writer reads.
  build_same_set_candidate_cells_inplace(data);
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
  bool signed_order = false;
  sort_point_set(query_x, query_y, query_z, inv_radius, data.sorted_x,
                 data.sorted_y, data.sorted_z, data.original_index,
                 data.cell_starts, data.cell_ends, data.scratch_point_keys,
                 data.scratch_point_keys_alt, data.scratch_unique_keys,
                 signed_order, nullptr);
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
  scratch.resize(
      static_cast<std::size_t>(max_row_count(data.counts) + row_slack()));
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

void write_same_set_listed_rows(NeighborSearchData &data,
                                const IntBuffer &queries, IntBuffer &counts,
                                IntBuffer &neighbors,
                                const IntBuffer &row_offset,
                                const IntBuffer &row_capacity) {
  const int num_queries = static_cast<int>(queries.size());
  if (num_queries == 0) {
    return;
  }
  // The blocked writer reads candidate cells, not ranges; every grid build
  // clears the ranges, so a size mismatch means they are not this grid's.
  if (data.cell_num_neighbors.size() !=
      static_cast<std::size_t>(data.num_cells)) {
    if (data.dense_valid && data.cell_stencil == 1) {
      build_same_set_cell_ranges_dense_fused_inplace(data);
    } else {
      build_same_set_cell_ranges_inplace(data);
    }
  }
  IntBuffer query_cells(static_cast<std::size_t>(num_queries));
  tbb::parallel_for(0, num_queries, [&](int k) {
    const int query = queries[static_cast<std::size_t>(k)];
    query_cells[static_cast<std::size_t>(k)] =
        static_cast<int>(std::upper_bound(data.cell_starts.begin(),
                                          data.cell_starts.end(), query) -
                         data.cell_starts.begin()) -
        1;
  });
  const ispc::CellRanges ranges = ranges_of(data);
  const ispc::Points pts = points_of(data);
  const ispc::FixedRows rows =
      block_rows(0, nullptr, counts, neighbors, row_offset, row_capacity);
  const float r2 = data.radius * data.radius;
  tbb::parallel_for(tbb::blocked_range<int>(0, num_queries, 16),
                    [&](const tbb::blocked_range<int> &range) {
                      ispc::write_listed_rows(
                          range.begin(), range.end(), queries.data(),
                          query_cells.data(), &ranges, &pts, r2, &rows);
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
                            capacity + row_slack(),
                            neighbor_permutation_table());
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
      static_cast<std::size_t>(max_row_count(query_data.counts) + row_slack()));
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
        ispc::csr_rows_pair(
            begin, end, &queries, &ranges, &qpts, &tpts, false, r2,
            query_data.row_ptr.data(), query_data.col_idx.data(), scratch,
            capacity + row_slack(), neighbor_permutation_table());
      });
}

void write_cross_neighbors_fixed_query_active2_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts) {
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(query_data.num_points),
                     capacity + row_slack());
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
                     capacity + row_slack());
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
  const int row_stride = capacity + row_slack();
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
                     static_cast<std::size_t>(data.num_points),
                     capacity + row_slack());
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
                     capacity + row_slack());
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
                     capacity + row_slack());
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