#include "apfrnn.hpp"

#include "kernels_ispc.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
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
#if defined(__linux__)
#include <sys/mman.h>
#endif

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
// The row buffers are hundreds of MB and are streamed in point order; with
// 4 KB pages their TLB footprint dwarfs the TLB. Transparent huge pages are
// madvise-only on this host, so hint the large allocations.
inline void hint_huge_pages(void *data, std::size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  constexpr std::uintptr_t kPage = std::uintptr_t{2} << 20;
  if (data == nullptr || bytes < (std::size_t{16} << 20)) {
    return;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
  const std::uintptr_t aligned = (begin + kPage - 1) & ~(kPage - 1);
  const std::uintptr_t end = (begin + bytes) & ~(kPage - 1);
  if (end > aligned) {
    madvise(reinterpret_cast<void *>(aligned),
            static_cast<std::size_t>(end - aligned), MADV_HUGEPAGE);
  }
#endif
}

inline void prepare_fixed_rows(IntBuffer &values, IntBuffer &counts,
                               std::size_t num_points, int row_stride) {
  const std::size_t required = num_points * static_cast<std::size_t>(row_stride);
  counts.resize(num_points);
  if (required > values.capacity() || 2 * required < values.capacity()) {
    IntBuffer().swap(values);
  }
  values.resize(required);
  hint_huge_pages(values.data(),
                  values.size() * static_cast<std::size_t>(sizeof(int)));
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
  Bounds world;
};

inline bool sort_point_set(
    const std::vector<float> &x, const std::vector<float> &y,
    const std::vector<float> &z, float inv_radius, std::vector<float> &sorted_x,
    std::vector<float> &sorted_y, std::vector<float> &sorted_z,
    IntBuffer &original_index, IntBuffer &cell_starts, IntBuffer &cell_ends,
    std::vector<SortPointKey> &point_keys,
    std::vector<SortPointKey> &point_keys_alt,
    std::vector<uint64_t> &dense_keys, std::vector<uint64_t> &unique_keys,
    bool &signed_order, CellBounds *bounds_out, const float *interleaved,
    IntBuffer *inverse_out) {
  int const num_points = static_cast<int>(x.size());
  const bool new_dense_keys =
      dense_keys.capacity() < static_cast<std::size_t>(num_points);
  dense_keys.resize(num_points);
  if (new_dense_keys) {
    hint_huge_pages(dense_keys.data(),
                    dense_keys.size() * sizeof(uint64_t));
  }
  const CellBounds bounds = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, num_points, 2048), CellBounds{},
      [&](const tbb::blocked_range<int> &range, CellBounds acc) {
        for (int i = range.begin(); i < range.end(); ++i) {
          const std::size_t at = static_cast<std::size_t>(i);
          float gx;
          float gy;
          float gz;
          if (interleaved != nullptr) {
            const float *q = interleaved + 3 * at;
            gx = q[0];
            gy = q[1];
            gz = q[2];
          } else {
            gx = x[at];
            gy = y[at];
            gz = z[at];
          }
          const int cx = floor_to_cell(gx * inv_radius);
          const int cy = floor_to_cell(gy * inv_radius);
          const int cz = floor_to_cell(gz * inv_radius);
          dense_keys[at] = make_key(cx, cy, cz);
          acc.min_cx = std::min(acc.min_cx, cx);
          acc.min_cy = std::min(acc.min_cy, cy);
          acc.min_cz = std::min(acc.min_cz, cz);
          acc.max_cx = std::max(acc.max_cx, cx);
          acc.max_cy = std::max(acc.max_cy, cy);
          acc.max_cz = std::max(acc.max_cz, cz);
          acc.world.add(gx, gy, gz);
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
        a.world.add(b.world);
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
    if (inverse_out != nullptr) {
      inverse_out->resize(static_cast<std::size_t>(num_points));
    }
    int *inverse = inverse_out != nullptr ? inverse_out->data() : nullptr;
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
            const uint64_t key = dense_keys[static_cast<std::size_t>(i)];
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

    const bool new_sorted =
        sorted_x.capacity() < static_cast<std::size_t>(num_points);
    sorted_x.resize(num_points);
    sorted_y.resize(num_points);
    sorted_z.resize(num_points);
    original_index.resize(num_points);
    if (new_sorted) {
      hint_huge_pages(sorted_x.data(),
                      sorted_x.size() * sizeof(float));
      hint_huge_pages(sorted_y.data(),
                      sorted_y.size() * sizeof(float));
      hint_huge_pages(sorted_z.data(),
                      sorted_z.size() * sizeof(float));
      hint_huge_pages(original_index.data(),
                      original_index.size() * sizeof(int));
    }

    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_points, 2048),
        [&](const tbb::blocked_range<int> &range) {
          for (int i = range.begin(); i < range.end(); ++i) {
            const uint64_t key = dense_keys[static_cast<std::size_t>(i)];
            const int lx = unpack_coord(key, 42) - min_cx;
            const int ly = unpack_coord(key, 21) - min_cy;
            const int lz = unpack_coord(key, 0) - min_cz;
            const int64_t cell =
                (static_cast<int64_t>(lx) * ny + ly) * nz + lz;
            const int position = __atomic_fetch_add(
                &offsets[static_cast<std::size_t>(cell)], 1,
                __ATOMIC_RELAXED);
            if (interleaved != nullptr) {
              const float *q = interleaved + 3 * static_cast<std::size_t>(i);
              sorted_x[static_cast<std::size_t>(position)] = q[0];
              sorted_y[static_cast<std::size_t>(position)] = q[1];
              sorted_z[static_cast<std::size_t>(position)] = q[2];
            } else {
              sorted_x[static_cast<std::size_t>(position)] =
                  x[static_cast<std::size_t>(i)];
              sorted_y[static_cast<std::size_t>(position)] =
                  y[static_cast<std::size_t>(i)];
              sorted_z[static_cast<std::size_t>(position)] =
                  z[static_cast<std::size_t>(i)];
            }
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
              if (inverse != nullptr) {
                inverse[static_cast<std::size_t>(
                    original_index[static_cast<std::size_t>(begin)])] = begin;
              }
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
            if (inverse != nullptr) {
              for (int i = begin; i < end; ++i) {
                inverse[static_cast<std::size_t>(
                    original_index[static_cast<std::size_t>(i)])] = i;
              }
            }
          }
        });
    return inverse != nullptr;
  }

  // The full-sort paths need the original index next to each key.
  point_keys.resize(num_points);
  tbb::parallel_for(0, num_points, [&](int i) {
    point_keys[static_cast<std::size_t>(i)] =
        SortPointKey{dense_keys[static_cast<std::size_t>(i)], i};
  });
  signed_order = radix_sort_point_keys(point_keys, point_keys_alt);

  sorted_x.resize(num_points);
  sorted_y.resize(num_points);
  sorted_z.resize(num_points);
  original_index.resize(num_points);

  tbb::parallel_for(0, num_points, [&](int i) {
    const int idx = point_keys[i].index;
    if (interleaved != nullptr) {
      const float *q = interleaved + 3 * static_cast<std::size_t>(idx);
      sorted_x[i] = q[0];
      sorted_y[i] = q[1];
      sorted_z[i] = q[2];
    } else {
      sorted_x[i] = x[static_cast<std::size_t>(idx)];
      sorted_y[i] = y[static_cast<std::size_t>(idx)];
      sorted_z[i] = z[static_cast<std::size_t>(idx)];
    }
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
  return false;
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
// Size-adaptive sort for the changed cell keys. When the changed list is
// small relative to its x-cell span, a counting sort by the x field followed
// by a per-bucket comparison sort beats the general comparison sort: the
// buckets hold only a handful of keys. At larger changed counts (the 8M
// sparse case) the buckets grow and the histogram/scatter passes stop paying,
// so the caller keeps the parallel comparison sort (the measured 8M
// behaviour). The per-bucket sort uses the same (key, index) order as the
// comparison path, so the result is identical and deterministic.
template <typename Less>
bool bucket_sort_changed_keys(std::vector<SortPointKey> &keys,
                              std::vector<SortPointKey> &scratch,
                              bool signed_order, int min_x, int max_x,
                              Less &&less) {
  const int num_points = static_cast<int>(keys.size());
  const int span = max_x - min_x + 1;
  constexpr int kMaxSpan = 1 << 18;
  constexpr int kMinPoints = 65536;
  // Up to 4096 points per x-bucket: the per-bucket comparison sorts are then
  // cache-resident and compare far fewer times than one global sort. The old
  // 32-point guard sent every sparse 1M-8M changed-key sort to the global
  // parallel_sort (measured 121.2 -> 114.0 ms at sparse 8M with 4096).
  constexpr long long kMaxPointsPerBucket = 4096;
  if (num_points < kMinPoints || span <= 0 || span > kMaxSpan ||
      static_cast<long long>(num_points) >
          static_cast<long long>(span) * kMaxPointsPerBucket) {
    return false;
  }
  const uint64_t mask = signed_order ? SortPointKey::kSignBits : 0;
  const int offset = signed_order ? 0x100000 : 0;
  // Keys outside [min_x, max_x] are the masked negative-x cells: raw-key
  // order places them after every non-negative x, which is exactly the last
  // bucket, and the per-bucket comparison sorts them behind that bucket's
  // single non-negative key. Clamping is therefore order-correct.
  auto bucket_of = [&](uint64_t key) {
    const int bucket =
        static_cast<int>(((key ^ mask) >> 42) & 0x1FFFFFULL) - offset - min_x;
    return std::min(span - 1, std::max(0, bucket));
  };
  constexpr int kChunks = 8;
  std::vector<int> histogram(static_cast<std::size_t>(kChunks) * span, 0);
  std::vector<int> offsets(static_cast<std::size_t>(kChunks) * span, 0);
  std::vector<int> bucket_start(static_cast<std::size_t>(span) + 1, 0);
  tbb::parallel_for(0, kChunks, [&](int chunk) {
    const int begin = static_cast<int>(
        static_cast<long long>(chunk) * num_points / kChunks);
    const int end = static_cast<int>(
        static_cast<long long>(chunk + 1) * num_points / kChunks);
    int *row = histogram.data() + static_cast<std::size_t>(chunk) * span;
    for (int i = begin; i < end; ++i) {
      ++row[bucket_of(keys[static_cast<std::size_t>(i)].key)];
    }
  });
  int total = 0;
  for (int bucket = 0; bucket < span; ++bucket) {
    bucket_start[static_cast<std::size_t>(bucket)] = total;
    int running = total;
    for (int chunk = 0; chunk < kChunks; ++chunk) {
      const std::size_t slot =
          static_cast<std::size_t>(chunk) * span + bucket;
      offsets[slot] = running;
      running += histogram[slot];
    }
    total = running;
  }
  bucket_start[static_cast<std::size_t>(span)] = total;
  scratch.resize(keys.size());
  tbb::parallel_for(0, kChunks, [&](int chunk) {
    const int begin = static_cast<int>(
        static_cast<long long>(chunk) * num_points / kChunks);
    const int end = static_cast<int>(
        static_cast<long long>(chunk + 1) * num_points / kChunks);
    int *row = offsets.data() + static_cast<std::size_t>(chunk) * span;
    for (int i = begin; i < end; ++i) {
      const SortPointKey value = keys[static_cast<std::size_t>(i)];
      scratch[static_cast<std::size_t>(row[bucket_of(value.key)]++)] = value;
    }
  });
  tbb::parallel_for(
      tbb::blocked_range<int>(0, span, 64),
      [&](const tbb::blocked_range<int> &range) {
        for (int bucket = range.begin(); bucket < range.end(); ++bucket) {
          const int begin = bucket_start[static_cast<std::size_t>(bucket)];
          const int end = bucket_start[static_cast<std::size_t>(bucket) + 1];
          if (end - begin > 1) {
            std::sort(scratch.begin() + begin, scratch.begin() + end, less);
          }
        }
      });
  keys.swap(scratch);
  return true;
}

bool try_incremental_sort_point_set(NeighborSearchData &data,
                                    const std::vector<float> &x,
                                    const std::vector<float> &y,
                                    const std::vector<float> &z,
                                    float inv_radius,
                                    const float *interleaved,
                                    Bounds *world_out) {
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
          float gx;
          float gy;
          float gz;
          if (interleaved != nullptr) {
            const float *q = interleaved + 3 * at;
            gx = q[0];
            gy = q[1];
            gz = q[2];
          } else {
            gx = x[at];
            gy = y[at];
            gz = z[at];
          }
          const int cx = floor_to_cell(gx * inv_radius);
          const int cy = floor_to_cell(gy * inv_radius);
          const int cz = floor_to_cell(gz * inv_radius);
          const uint64_t key = make_key(cx, cy, cz);
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
          acc.bounds.world.add(gx, gy, gz);
        }
        return acc;
      },
      [](KeyPass a, const KeyPass &b) {
        a.changed += b.changed;
        a.bounds.world.add(b.bounds.world);
        a.bounds.min_cx = std::min(a.bounds.min_cx, b.bounds.min_cx);
        a.bounds.min_cy = std::min(a.bounds.min_cy, b.bounds.min_cy);
        a.bounds.min_cz = std::min(a.bounds.min_cz, b.bounds.min_cz);
        a.bounds.max_cx = std::max(a.bounds.max_cx, b.bounds.max_cx);
        a.bounds.max_cy = std::max(a.bounds.max_cy, b.bounds.max_cy);
        a.bounds.max_cz = std::max(a.bounds.max_cz, b.bounds.max_cz);
        return a;
      });
  const int changed_count = pass.changed;
  if (world_out != nullptr) {
    *world_out = pass.bounds.world;
  }
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
            SortPointKey{data.incremental_keys[static_cast<std::size_t>(i)], i};
      });
  if (!bucket_sort_changed_keys(changed_keys, data.scratch_point_keys_alt,
                                signed_order, data.cell_coord_min_x,
                                data.cell_coord_max_x, less)) {
    tbb::parallel_sort(changed_keys.begin(), changed_keys.end(), less);
  }

  const int unchanged_count = num_points - changed_count;
  auto &unchanged_positions = data.incremental_unchanged_positions;
  unchanged_positions.resize(static_cast<std::size_t>(unchanged_count));
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
        unchanged_positions[static_cast<std::size_t>(write)] = i;
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
  const int *unchanged_at = unchanged_positions.data();
  const SortPointKey *b = changed_keys.data();
  // The unchanged stream is a subsequence of the previous order; read its
  // keys and indices from the already-sorted arrays instead of materializing
  // 16-byte pairs.
  auto a_less = [&](int ia, const SortPointKey &other) {
    const int position = unchanged_at[ia];
    const uint64_t ka = ordered_key(previous_sorted[position]);
    const uint64_t kb = ordered_key(other.key);
    return ka != kb ? ka < kb : previous_order[position] < other.index;
  };
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
      if (a_less(mid, b[other - 1])) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    split_a[static_cast<std::size_t>(worker)] = lo;
    split_b[static_cast<std::size_t>(worker)] = rank - lo;
  }
  std::vector<uint8_t> worker_duplicate(static_cast<std::size_t>(workers), 0);
  tbb::parallel_for(0, workers, [&](int worker) {
    int ia = split_a[static_cast<std::size_t>(worker)];
    int ib = split_b[static_cast<std::size_t>(worker)];
    const int end_a = split_a[static_cast<std::size_t>(worker) + 1];
    const int end_b = split_b[static_cast<std::size_t>(worker) + 1];
    int write = ia + ib;
    const int worker_start = write;
    uint8_t duplicate = 0;
    while (ia < end_a && ib < end_b) {
      const bool take_a = a_less(ia, b[ib]);
      int idx;
      uint64_t key;
      if (take_a) {
        const int position = unchanged_at[ia++];
        key = previous_sorted[position];
        idx = previous_order[position];
      } else {
        key = b[ib].key;
        idx = b[ib++].index;
      }
      out_keys[write] = key;
      if (write > worker_start && key == out_keys[write - 1]) {
        duplicate = 1;
      }
      out_order[write] = idx;
      ++write;
    }
    for (; ia < end_a; ++ia) {
      const int position = unchanged_at[ia];
      const int idx = previous_order[position];
      const uint64_t key = previous_sorted[position];
      out_keys[write] = key;
      if (write > worker_start && key == out_keys[write - 1]) {
        duplicate = 1;
      }
      out_order[write] = idx;
      ++write;
    }
    for (; ib < end_b; ++ib) {
      const int idx = b[ib].index;
      const uint64_t key = b[ib].key;
      out_keys[write] = key;
      if (write > worker_start && key == out_keys[write - 1]) {
        duplicate = 1;
      }
      out_order[write] = idx;
      ++write;
    }
    worker_duplicate[static_cast<std::size_t>(worker)] = duplicate;
  });

  // Independent random reads, eight or more in flight per thread; inside
  // the branchy merge each gather sat on the merge's critical path.
  {
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_points, 4096),
        [&](const tbb::blocked_range<int> &range) {
          for (int i = range.begin(); i < range.end(); ++i) {
            const std::size_t idx = static_cast<std::size_t>(
                out_order[static_cast<std::size_t>(i)]);
            const std::size_t at = static_cast<std::size_t>(i);
            if (interleaved != nullptr) {
              const float *q = interleaved + 3 * idx;
              out_x[at] = q[0];
              out_y[at] = q[1];
              out_z[at] = q[2];
            } else {
              out_x[at] = x[idx];
              out_y[at] = y[idx];
              out_z[at] = z[idx];
            }
          }
        });
  }

  // Cell ranges from the merged key order. Position 0 and every key change
  // starts a cell; chunks count locally, a short prefix assigns cell ids.
  constexpr int kRangeGrain = 16384;
  const int chunks = (num_points + kRangeGrain - 1) / kRangeGrain;
  // One point per cell is the common sparse case: starts/ends are the index
  // sequence and the merged key buffer doubles as the unique-key array (and
  // as next frame's per-position order).
  // Duplicates were tracked inside the merge workers; only the worker
  // boundaries remain to check.
  bool all_unique = true;
  for (uint8_t duplicate : worker_duplicate) {
    all_unique &= duplicate == 0;
  }
  if (all_unique) {
    for (int worker = 1; worker < workers; ++worker) {
      const int at = split_a[static_cast<std::size_t>(worker)] +
                     split_b[static_cast<std::size_t>(worker)];
      if (at > 0 && at < num_points && out_keys[at] == out_keys[at - 1]) {
        all_unique = false;
        break;
      }
    }
  }
  if (all_unique) {
    data.incremental_starts.clear();
    data.incremental_ends.clear();
    data.sorted_x.swap(data.incremental_x);
    data.sorted_y.swap(data.incremental_y);
    data.sorted_z.swap(data.incremental_z);
    data.original_index.swap(data.incremental_order);
    data.cell_starts.swap(data.incremental_starts);
    data.cell_ends.swap(data.incremental_ends);
    data.scratch_unique_keys.swap(data.incremental_merge_keys);
    data.incremental_keys_are_unique = true;
    data.cell_ranges_implicit = true;
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
                             const std::vector<float> &z, float inv_radius,
                             const float *interleaved) {
  const int num_points = static_cast<int>(x.size());
  data.incremental_keys.resize(static_cast<std::size_t>(num_points));
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, 2048),
      [&](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
          const std::size_t at = static_cast<std::size_t>(i);
          float gx;
          float gy;
          float gz;
          if (interleaved != nullptr) {
            const float *q = interleaved + 3 * at;
            gx = q[0];
            gy = q[1];
            gz = q[2];
          } else {
            gx = x[at];
            gy = y[at];
            gz = z[at];
          }
          data.incremental_keys[at] =
              make_key(floor_to_cell(gx * inv_radius),
                       floor_to_cell(gy * inv_radius),
                       floor_to_cell(gz * inv_radius));
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
  if (data.cell_ranges_implicit) {
    ensure_cell_ranges(data);
  }
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

  ensure_cell_ranges(const_cast<NeighborSearchData &>(target));
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
  ensure_cell_ranges(const_cast<NeighborSearchData &>(target));
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
  ensure_cell_ranges(const_cast<NeighborSearchData &>(target_data));
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
  ensure_cell_ranges(const_cast<NeighborSearchData &>(target_data));
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

bool build_point_grid_inplace(NeighborSearchData &data,
                              const std::vector<float> &x,
                              const std::vector<float> &y,
                              const std::vector<float> &z, float radius,
                              float cell_radius, const float *interleaved,
                              Bounds *world_bounds, IntBuffer *inverse_out) {
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

  // The cell-x slab index and occupancy describe the previous sorted order.
  data.cell_x_valid = false;
  data.occupancy_valid = false;
  data.cell_ranges_implicit = false;
  data.ordered_x_valid = false;
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
    return false;
  }

  float inv_radius = 1.0f / data.cell_radius;
  const bool incremental = try_incremental_sort_point_set(
      data, x, y, z, inv_radius, interleaved, world_bounds);
  bool inverse_written = false;
  if (!incremental) {
    CellBounds bounds;
    inverse_written = sort_point_set(
        x, y, z, inv_radius, data.sorted_x, data.sorted_y, data.sorted_z,
        data.original_index, data.cell_starts, data.cell_ends,
        data.scratch_point_keys, data.scratch_point_keys_alt, data.dense_keys,
        data.scratch_unique_keys, data.cell_keys_signed_order, &bounds,
        interleaved, inverse_out);
    if (world_bounds != nullptr) {
      *world_bounds = bounds.world;
    }
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
      !data.cell_ranges_implicit && data.cached_range_valid &&
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
      !data.cell_ranges_implicit && data.dense_stencil_cache_valid &&
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

  int num_cells = data.cell_ranges_implicit
                      ? data.num_points
                      : static_cast<int>(data.cell_starts.size());
  data.num_cells = num_cells;
  build_dense_cell_lookup(data);
  int max_cell_points = 1;
  if (!data.cell_ranges_implicit) {
    max_cell_points = 0;
    for (int cell = 0; cell < num_cells; ++cell) {
      max_cell_points = std::max(
          max_cell_points,
          data.cell_ends[static_cast<std::size_t>(cell)] -
              data.cell_starts[static_cast<std::size_t>(cell)]);
    }
  }
  data.max_cell_points = max_cell_points;
  // Dense blocks reorder points inside cells; the inverse written during the
  // in-cell sort would be stale, so let the caller rebuild it in that case.
  if (inverse_written && max_cell_points > 32) {
    inverse_written = false;
  }
  if (!incremental) {
    // The merge only targets sparse grids; dense sets keep the full-sort
    // path and must not pay to capture merge state they will never use.
    if (max_cell_points <= 1) {
      reset_incremental_state(data, x, y, z, inv_radius, interleaved);
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
  return inverse_written;
}

void ensure_point_blocks(NeighborSearchData &data) {
  if (data.blocks_valid || data.max_cell_points <= 32) {
    return;
  }
  build_point_blocks_inplace(data);
}

static void build_same_set_candidate_cells_inplace(NeighborSearchData &data) {
  ensure_cell_ranges(data);
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
  ensure_cell_ranges(data);
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
  ensure_cell_ranges(data);
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
  ensure_cell_ranges(data);
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
  ensure_cell_ranges(data);
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
                            IntBuffer &fixed_neighbors, IntBuffer &counts,
                            std::vector<int> &touched_rows,
                            bool &touched_overflow, bool reuse_rows,
                            int &max_count_out) {
  const int stride = capacity;
  const std::size_t counts_before = counts.size();
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(data.num_points), stride);
  // Counts use a default-init allocator. Zero the whole buffer when it was
  // reallocated or another writer touched it; otherwise only the rows this
  // writer incremented last time need clearing.
  if (!reuse_rows || counts_before != counts.size() || touched_overflow) {
    std::fill(counts.begin(), counts.end(), 0);
    touched_rows.clear();
    touched_overflow = false;
  } else {
    for (int row : touched_rows) {
      counts[static_cast<std::size_t>(row)] = 0;
    }
    touched_rows.clear();
  }
  max_count_out = 0;
  if (data.num_points == 0) {
    return true;
  }
  const int num_cells = data.num_cells;
  const bool implicit_cells = data.cell_ranges_implicit;
  const uint64_t *keys = data.scratch_unique_keys.data();
  const int *cell_starts = implicit_cells ? nullptr
                                          : data.cell_starts.data();
  auto cell_point = [&](int cell) {
    return implicit_cells ? cell : cell_starts[cell];
  };
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
  constexpr std::size_t kTouchedCap = std::size_t{1} << 20;
  std::mutex touched_mutex;
  auto append = [&](int row, int value) {
    const int slot =
        __atomic_fetch_add(&row_counts[row], 1, __ATOMIC_RELAXED);
    if (slot == 0) {
      std::lock_guard<std::mutex> lock(touched_mutex);
      if (touched_rows.size() < kTouchedCap) {
        touched_rows.push_back(row);
      } else {
        touched_overflow = true;
      }
    }
    if (slot < capacity) {
      row_values[static_cast<std::size_t>(row) * stride + slot] = value;
    }
  };
  auto test_cell = [&](int query, int other_cell) {
    const int other = cell_point(other_cell);
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
        auto advance_and_test = [&](int query, int k, uint64_t want) {
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
          if (position < num_cells && order_key(keys[position]) == want) {
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
        };
        for (int cell = begin; cell < end; ++cell) {
          const int query = cell_point(cell);
          const uint64_t key = keys[cell];
          const int cx = unpack_coord(key, 42);
          const int cy = unpack_coord(key, 21);
          const int cz = unpack_coord(key, 0);
          if (cx > 0 && cx < 0xFFFFE && cy > 0 && cy < 0xFFFFE &&
              cz >= 0 && cz < 0xFFFFE) {
            const uint64_t ordered =
                signed_order ? (key | SortPointKey::kSignBits) : key;
            const int next = cell + 1;
            const uint64_t direct = ordered + 1;
            if (next < num_cells && order_key(keys[next]) == direct) {
              test_cell(query, next);
            }
            advance_and_test(query, 0, ordered + (uint64_t{1} << 21));
            advance_and_test(
                query, 1,
                ordered + (uint64_t{1} << 42) - (uint64_t{1} << 21));
            advance_and_test(query, 2, ordered + (uint64_t{1} << 42));
            advance_and_test(
                query, 3,
                ordered + (uint64_t{1} << 42) + (uint64_t{1} << 21));
            continue;
          }
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
  // Only rows with at least one neighbor are non-zero, so the max comes from
  // the touched list; a full scan is the fallback when the list overflowed.
  if (touched_overflow) {
    max_count_out = max_row_count(counts);
  } else {
    for (int row : touched_rows) {
      max_count_out = std::max(max_count_out, counts[row]);
    }
  }
  return max_count_out <= capacity;
}

// Builds the cell-x slab index and the cell y/z bounds. The cell keys must be
// in lexicographic (x, y, z) order with non-negative coordinates; callers gate
// on that, so raw key order is also coordinate order.
void ensure_cell_ranges(NeighborSearchData &data) {
  if (!data.cell_ranges_implicit) {
    return;
  }
  const int count = data.num_points;
  data.cell_starts.resize(static_cast<std::size_t>(count));
  data.cell_ends.resize(static_cast<std::size_t>(count));
  int *starts = data.cell_starts.data();
  int *ends = data.cell_ends.data();
  tbb::parallel_for(tbb::blocked_range<int>(0, count, 16384),
                    [&](const tbb::blocked_range<int> &range) {
                      for (int i = range.begin(); i < range.end(); ++i) {
                        starts[i] = i;
                        ends[i] = i + 1;
                      }
                    });
  data.cell_ranges_implicit = false;
}

// Builds the x-slab index in order-key space. Raw keys are XORed with the
// sign mask when the sort used signed-coordinate ordering, so negative
// coordinates remain supported: the ordered field is (coord + 2^20) for the
// 21-bit packed field, which is monotonic in the real coordinate.
void ensure_ordered_cell_x(NeighborSearchData &data) {
  if (data.ordered_x_valid) {
    return;
  }
  const int num_cells = data.num_cells;
  if (num_cells <= 0 || data.scratch_unique_keys.size() !=
                            static_cast<std::size_t>(num_cells)) {
    data.ordered_x_start.clear();
    return;
  }
  const uint64_t *keys = data.scratch_unique_keys.data();
  const uint64_t mask =
      data.cell_keys_signed_order ? SortPointKey::kSignBits : 0;
  // Order-space fields are unsigned 21-bit offsets (real coordinate plus
  // 2^20 when signed ordering is used), so mask the field instead of
  // sign-extending it.
  auto coord = [&](uint64_t key, int shift) {
    return static_cast<int>(((key ^ mask) >> shift) & 0x1FFFFFULL);
  };
  const int min_x = coord(keys[0], 42);
  const int max_x = coord(keys[num_cells - 1], 42);
  const int span = max_x - min_x + 1;
  if (span <= 0 || span > 4 * num_cells + 65536) {
    data.ordered_x_start.clear();
    return;
  }
  data.ordered_x_start.resize(static_cast<std::size_t>(span) + 1);
  int *slab = data.ordered_x_start.data();
  slab[span] = num_cells;
  auto slab_x_less = [&](uint64_t key, uint64_t want) {
    return (key ^ mask) < want;
  };
  constexpr int kChunks = 32;
  if (span < 4096 || num_cells < 65536) {
    int cell = 0;
    for (int i = 0; i < span; ++i) {
      const int x = min_x + i;
      while (cell < num_cells && coord(keys[cell], 42) < x) {
        ++cell;
      }
      slab[i] = cell;
    }
  } else {
    // Parallel fill: the cell scan is monotonic, so each chunk starts at the
    // lower-bound position for its first slab and only scans its own cells.
    std::vector<int> chunk_start(static_cast<std::size_t>(kChunks) + 1, 0);
    for (int chunk = 1; chunk <= kChunks; ++chunk) {
      const int i = static_cast<int>(static_cast<long long>(chunk) * span /
                                     kChunks);
      const uint64_t want = make_key(min_x + i, 0, 0);
      chunk_start[static_cast<std::size_t>(chunk)] =
          static_cast<int>(std::lower_bound(keys, keys + num_cells, want,
                                            slab_x_less) -
                           keys);
    }
    tbb::parallel_for(0, kChunks, [&](int chunk) {
      const int i0 = static_cast<int>(static_cast<long long>(chunk) * span /
                                      kChunks);
      const int i1 = static_cast<int>(static_cast<long long>(chunk + 1) *
                                      span / kChunks);
      int cell = chunk_start[static_cast<std::size_t>(chunk)];
      const int end = chunk_start[static_cast<std::size_t>(chunk) + 1];
      for (int i = i0; i < i1; ++i) {
        const int x = min_x + i;
        while (cell < end && coord(keys[cell], 42) < x) {
          ++cell;
        }
        slab[i] = cell;
      }
    });
  }
  // Y/Z windows are only clamps; the full field range is exact for the
  // lower-bound scans, so no extra pass over the keys is needed.
  data.ordered_x_min = min_x;
  data.ordered_y_min = 0;
  data.ordered_y_max = 0x1FFFFF;
  data.ordered_z_min = 0;
  data.ordered_z_max = 0x1FFFFF;
  data.ordered_x_valid = true;
}

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

// Builds a coarse occupancy bitmap for a target set. Each bit covers a cube
// of edge >= 2*radius, so a query only needs the 2x2x2 blocks its sphere can
// touch. The bitmap is capped near 16 MiB; wide sparse domains are exactly
// where it pays off.
void ensure_occupancy(NeighborSearchData &data, const Bounds *known) {
  if (data.occupancy_valid) {
    return;
  }
  const int num_points = data.num_points;
  if (num_points <= 0 || !(data.cell_radius > 0.0f) ||
      data.sorted_x.size() != static_cast<std::size_t>(num_points)) {
    return;
  }
  const float *x = data.sorted_x.data();
  const float *y = data.sorted_y.data();
  const float *z = data.sorted_z.data();
  struct Box {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    float max_z = std::numeric_limits<float>::lowest();
  };
  Box box;
  if (known != nullptr && !known->empty()) {
    // The caller already tracks the set's AABB (sync_source); reuse it
    // instead of another full pass over the coordinates.
    box.min_x = known->min_x;
    box.min_y = known->min_y;
    box.min_z = known->min_z;
    box.max_x = known->max_x;
    box.max_y = known->max_y;
    box.max_z = known->max_z;
  } else {
    box = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, num_points, 65536), Box{},
        [&](const tbb::blocked_range<int> &range, Box acc) {
          for (int i = range.begin(); i < range.end(); ++i) {
            const std::size_t at = static_cast<std::size_t>(i);
            acc.min_x = std::min(acc.min_x, x[at]);
            acc.min_y = std::min(acc.min_y, y[at]);
            acc.min_z = std::min(acc.min_z, z[at]);
            acc.max_x = std::max(acc.max_x, x[at]);
            acc.max_y = std::max(acc.max_y, y[at]);
            acc.max_z = std::max(acc.max_z, z[at]);
          }
          return acc;
        },
        [](Box a, const Box &b) {
          a.min_x = std::min(a.min_x, b.min_x);
          a.min_y = std::min(a.min_y, b.min_y);
          a.min_z = std::min(a.min_z, b.min_z);
          a.max_x = std::max(a.max_x, b.max_x);
          a.max_y = std::max(a.max_y, b.max_y);
          a.max_z = std::max(a.max_z, b.max_z);
          return a;
        });
  }
  const float extent_x = box.max_x - box.min_x;
  const float extent_y = box.max_y - box.min_y;
  const float extent_z = box.max_z - box.min_z;
  const float longest = std::max(extent_x, std::max(extent_y, extent_z));
  // Size the bitmap to the target (~64 blocks per point, capped at 16 MiB),
  // not to a fixed 500^3 grid. That keeps the expected occupancy of a block
  // near 1.5%, so a query's radius-2 sphere is culled unless a target point
  // really is nearby (the coarser 8-bits-per-point grid passed ~12% of
  // queries and rescanned target cells for almost all of them).
  constexpr std::size_t kMaxBits = std::size_t{1} << 27;
  const std::size_t target_bits = std::min(
      kMaxBits, std::max<std::size_t>(1024, 64 * static_cast<std::size_t>(num_points)));
  int axis = std::max(
      1, static_cast<int>(std::cbrt(static_cast<double>(target_bits))));
  float block = std::max(2.0f * data.cell_radius, longest / axis);
  if (!(block > 0.0f) || !std::isfinite(block)) {
    return;
  }
  auto axis_blocks = [&](float extent) {
    return std::max(1, static_cast<int>(extent / block) + 1);
  };
  int nx = axis_blocks(extent_x);
  int ny = axis_blocks(extent_y);
  int nz = axis_blocks(extent_z);
  std::size_t total = static_cast<std::size_t>(nx) * ny * nz;
  // Long/thin boxes can exceed the cap even with the derived edge.
  while (total > kMaxBits) {
    block *= 1.25f;
    nx = axis_blocks(extent_x);
    ny = axis_blocks(extent_y);
    nz = axis_blocks(extent_z);
    total = static_cast<std::size_t>(nx) * ny * nz;
  }
  if (total == 0) {
    return;
  }
  data.occupancy_block = block;
  data.occupancy_inv_block = 1.0f / block;
  data.occupancy_min_x = box.min_x;
  data.occupancy_min_y = box.min_y;
  data.occupancy_min_z = box.min_z;
  data.occupancy_nx = nx;
  data.occupancy_ny = ny;
  data.occupancy_nz = nz;
  data.occupancy_bits.assign((total + 7) / 8, 0);
  const float inv_block = 1.0f / block;
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, 16384),
      [&](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
          const std::size_t at = static_cast<std::size_t>(i);
          const int bx = std::min(
              nx - 1,
              std::max(0, static_cast<int>((x[at] - box.min_x) * inv_block)));
          const int by = std::min(
              ny - 1,
              std::max(0, static_cast<int>((y[at] - box.min_y) * inv_block)));
          const int bz = std::min(
              nz - 1,
              std::max(0, static_cast<int>((z[at] - box.min_z) * inv_block)));
          const std::size_t index =
              (static_cast<std::size_t>(bx) * ny + by) * nz + bz;
          __atomic_fetch_or(&data.occupancy_bits[index >> 3],
                            static_cast<uint8_t>(1u << (index & 7)),
                            __ATOMIC_RELAXED);
        }
      });
  data.occupancy_valid = true;
}

inline // Cell-aligned coarse occupancy built straight from the sorted unique cell
// keys. The block edge is a power-of-two number of cells, so block ids are
// monotonic in the sorted key order; partitioning the fill by bitmap byte
// range lets every worker write its own bytes with plain ORs, avoiding the
// 8M-atomic float pass over all points.
void ensure_occupancy_cells(NeighborSearchData &data) {
  if (data.occupancy_valid) {
    return;
  }
  const int num_cells = data.num_cells;
  if (num_cells <= 0 || !(data.cell_radius > 0.0f) ||
      data.scratch_unique_keys.size() !=
          static_cast<std::size_t>(num_cells)) {
    return;
  }
  const uint64_t *keys = data.scratch_unique_keys.data();
  const uint64_t mask =
      data.cell_keys_signed_order ? SortPointKey::kSignBits : 0;
  const int field_offset = mask != 0 ? 0x100000 : 0;
  auto real_coord = [&](uint64_t key, int shift) {
    return static_cast<int>(((key ^ mask) >> shift) & 0x1FFFFFULL) -
           field_offset;
  };
  int min_cx;
  int max_cx;
  int min_cy;
  int max_cy;
  int min_cz;
  int max_cz;
  if (data.cell_coord_valid) {
    min_cx = data.cell_coord_min_x;
    max_cx = data.cell_coord_max_x;
    min_cy = data.cell_coord_min_y;
    max_cy = data.cell_coord_max_y;
    min_cz = data.cell_coord_min_z;
    max_cz = data.cell_coord_max_z;
  } else {
    struct CellBox {
      int min_x = INT_MAX;
      int max_x = INT_MIN;
      int min_y = INT_MAX;
      int max_y = INT_MIN;
      int min_z = INT_MAX;
      int max_z = INT_MIN;
    };
    const CellBox box = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, num_cells, 16384), CellBox{},
        [&](const tbb::blocked_range<int> &range, CellBox acc) {
          for (int i = range.begin(); i < range.end(); ++i) {
            const int x = real_coord(keys[i], 42);
            const int y = real_coord(keys[i], 21);
            const int z = real_coord(keys[i], 0);
            acc.min_x = std::min(acc.min_x, x);
            acc.max_x = std::max(acc.max_x, x);
            acc.min_y = std::min(acc.min_y, y);
            acc.max_y = std::max(acc.max_y, y);
            acc.min_z = std::min(acc.min_z, z);
            acc.max_z = std::max(acc.max_z, z);
          }
          return acc;
        },
        [](CellBox a, const CellBox &b) {
          a.min_x = std::min(a.min_x, b.min_x);
          a.max_x = std::max(a.max_x, b.max_x);
          a.min_y = std::min(a.min_y, b.min_y);
          a.max_y = std::max(a.max_y, b.max_y);
          a.min_z = std::min(a.min_z, b.min_z);
          a.max_z = std::max(a.max_z, b.max_z);
          return a;
        });
    min_cx = box.min_x;
    max_cx = box.max_x;
    min_cy = box.min_y;
    max_cy = box.max_y;
    min_cz = box.min_z;
    max_cz = box.max_z;
  }
  if (max_cx < min_cx || max_cy < min_cy || max_cz < min_cz) {
    return;
  }
  constexpr int64_t kMaxBits = int64_t{1} << 27;
  const int64_t budget = std::min<int64_t>(
      kMaxBits, std::max<int64_t>(1024, int64_t{512} * num_cells));
  int shift = 1;
  int64_t nx = 0;
  int64_t ny = 0;
  int64_t nz = 0;
  for (; shift <= 30; ++shift) {
    const int block_cells = 1 << shift;
    const int origin_x = min_cx & ~(block_cells - 1);
    const int origin_y = min_cy & ~(block_cells - 1);
    const int origin_z = min_cz & ~(block_cells - 1);
    nx = ((max_cx - origin_x) >> shift) + 1;
    ny = ((max_cy - origin_y) >> shift) + 1;
    nz = ((max_cz - origin_z) >> shift) + 1;
    if (nx * ny * nz <= budget) {
      break;
    }
  }
  if (shift > 30) {
    return;
  }
  const int block_cells = 1 << shift;
  const int origin_x = min_cx & ~(block_cells - 1);
  const int origin_y = min_cy & ~(block_cells - 1);
  const int origin_z = min_cz & ~(block_cells - 1);
  const int64_t blocks = nx * ny * nz;
  if (blocks <= 0) {
    return;
  }
  const int64_t num_bytes = (blocks + 7) / 8;
  data.occupancy_bits.assign(static_cast<std::size_t>(num_bytes), 0);
  data.occupancy_block = data.cell_radius * block_cells;
  data.occupancy_inv_block = 1.0f / data.occupancy_block;
  data.occupancy_min_x = static_cast<float>(origin_x) * data.cell_radius;
  data.occupancy_min_y = static_cast<float>(origin_y) * data.cell_radius;
  data.occupancy_min_z = static_cast<float>(origin_z) * data.cell_radius;
  data.occupancy_nx = static_cast<int>(nx);
  data.occupancy_ny = static_cast<int>(ny);
  data.occupancy_nz = static_cast<int>(nz);
  auto key_of_block = [&](int64_t bid) {
    const int bz = static_cast<int>(bid % nz);
    const int64_t rest = bid / nz;
    const int by = static_cast<int>(rest % ny);
    const int bx = static_cast<int>(rest / ny);
    return make_key(origin_x + (bx << shift) + field_offset,
                    origin_y + (by << shift) + field_offset,
                    origin_z + (bz << shift) + field_offset);
  };
  auto ordered_less = [mask](uint64_t a, uint64_t b) {
    return (a ^ mask) < b;
  };
  const int chunks = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(
                               4 * tbb::this_task_arena::max_concurrency(),
                               num_bytes)));
  tbb::parallel_for(0, chunks, [&](int chunk) {
    const int64_t b0 = num_bytes * chunk / chunks;
    const int64_t b1 = num_bytes * (chunk + 1) / chunks;
    if (b0 >= b1) {
      return;
    }
    const int64_t bid0 = b0 * 8;
    const int64_t bid1 = std::min<int64_t>(blocks, b1 * 8);
    if (bid0 >= blocks) {
      return;
    }
    const uint64_t want_lo = key_of_block(bid0);
    const int lo = static_cast<int>(
        std::lower_bound(keys, keys + num_cells, want_lo, ordered_less) -
        keys);
    int hi = num_cells;
    if (bid1 < blocks) {
      const uint64_t want_hi = key_of_block(bid1);
      hi = static_cast<int>(
          std::lower_bound(keys, keys + num_cells, want_hi, ordered_less) -
          keys);
    }
    uint8_t *bits = data.occupancy_bits.data();
    for (int i = lo; i < hi; ++i) {
      int bx = (real_coord(keys[i], 42) - origin_x) >> shift;
      int by = (real_coord(keys[i], 21) - origin_y) >> shift;
      int bz = (real_coord(keys[i], 0) - origin_z) >> shift;
      bx = std::min<int>(static_cast<int>(nx) - 1, std::max(0, bx));
      by = std::min<int>(static_cast<int>(ny) - 1, std::max(0, by));
      bz = std::min<int>(static_cast<int>(nz) - 1, std::max(0, bz));
      const int64_t bid =
          (static_cast<int64_t>(bx) * ny + by) * nz + bz;
      bits[bid >> 3] =
          static_cast<uint8_t>(bits[bid >> 3] | (1u << (bid & 7)));
    }
  });
  data.occupancy_valid = true;
}

bool occupancy_may_contain(const NeighborSearchData &data, float qx,
                                  float qy, float qz, float radius) {
  if (!data.occupancy_valid) {
    return true;
  }
  if (2.0f * radius > data.occupancy_block) {
    return true;
  }
  const float upper_x =
      data.occupancy_min_x + data.occupancy_nx * data.occupancy_block;
  const float upper_y =
      data.occupancy_min_y + data.occupancy_ny * data.occupancy_block;
  const float upper_z =
      data.occupancy_min_z + data.occupancy_nz * data.occupancy_block;
  if (qx + radius < data.occupancy_min_x ||
      qy + radius < data.occupancy_min_y ||
      qz + radius < data.occupancy_min_z || qx - radius >= upper_x ||
      qy - radius >= upper_y || qz - radius >= upper_z) {
    return false;
  }
  const float inv_block = data.occupancy_inv_block;
  auto block_of = [&](float q, float min_value, int n) {
    int block = static_cast<int>((q - min_value) * inv_block);
    return std::min(n - 1, std::max(0, block));
  };
  const int center_x = block_of(qx, data.occupancy_min_x, data.occupancy_nx);
  const int center_y = block_of(qy, data.occupancy_min_y, data.occupancy_ny);
  const int center_z = block_of(qz, data.occupancy_min_z, data.occupancy_nz);
  const float center_x0 =
      data.occupancy_min_x + center_x * data.occupancy_block;
  const float center_y0 =
      data.occupancy_min_y + center_y * data.occupancy_block;
  const float center_z0 =
      data.occupancy_min_z + center_z * data.occupancy_block;
  if (qx - radius >= center_x0 &&
      qx + radius < center_x0 + data.occupancy_block &&
      qy - radius >= center_y0 &&
      qy + radius < center_y0 + data.occupancy_block &&
      qz - radius >= center_z0 &&
      qz + radius < center_z0 + data.occupancy_block) {
    const std::size_t index =
        (static_cast<std::size_t>(center_x) * data.occupancy_ny + center_y) *
            data.occupancy_nz +
        center_z;
    return ((data.occupancy_bits[index >> 3] >> (index & 7)) & 1u) != 0;
  }
  auto range = [&](float q, float min_value, int n, int &first, int &last) {
    first = static_cast<int>((q - radius - min_value) * inv_block);
    last = static_cast<int>((q + radius - min_value) * inv_block);
    first = std::min(n - 1, std::max(0, first));
    last = std::min(n - 1, std::max(0, last));
  };
  int x0, x1, y0, y1, z0, z1;
  range(qx, data.occupancy_min_x, data.occupancy_nx, x0, x1);
  range(qy, data.occupancy_min_y, data.occupancy_ny, y0, y1);
  range(qz, data.occupancy_min_z, data.occupancy_nz, z0, z1);
  for (int bz = z0; bz <= z1; ++bz) {
    for (int by = y0; by <= y1; ++by) {
      for (int bx = x0; bx <= x1; ++bx) {
        const std::size_t index =
            (static_cast<std::size_t>(bx) * data.occupancy_ny + by) *
                data.occupancy_nz +
            bz;
        if ((data.occupancy_bits[index >> 3] >> (index & 7)) & 1u) {
          return true;
        }
      }
    }
  }
  return false;
}

bool write_cross_rows_cell_range(NeighborSearchData &source,
                                 NeighborSearchData &target, float radius,
                                 int capacity, IntBuffer &fixed_neighbors,
                                 IntBuffer &counts,
                                 std::vector<int> &touched_rows,
                                 bool &touched_overflow,
                                 int &max_count_out) {
  ensure_cell_x_index(target);
  ensure_cell_ranges(target);
  if (!target.cell_x_valid || target.num_cells <= 0) {
    return false;
  }
  const int num_points = source.num_points;
  const int stride = capacity;
  const std::size_t counts_before = counts.size();
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(num_points), stride);
  // The counts buffer uses a default-init allocator. Zero it when it is
  // (re)allocated; otherwise clear only the rows written by the previous
  // build and leave the untouched (already zero) rows alone.
  if (counts_before != counts.size() || touched_overflow) {
    std::fill(counts.begin(), counts.end(), 0);
    touched_rows.clear();
    touched_overflow = false;
  } else {
    for (int row : touched_rows) {
      counts[static_cast<std::size_t>(row)] = 0;
    }
    touched_rows.clear();
  }
  max_count_out = 0;
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
  constexpr std::size_t kTouchedCap = std::size_t{1} << 20;
  std::mutex touched_mutex;
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, kGrain),
      [&](const tbb::blocked_range<int> &range) {
        std::vector<int> task_touched;
        task_touched.reserve(16);
        int task_max = 0;
        for (int point = range.begin(); point < range.end(); ++point) {
          const float px = qx[point];
          const float py = qy[point];
          const float pz = qz[point];
          if (target.occupancy_valid &&
              !occupancy_may_contain(target, px, py, pz, radius)) {
            continue;
          }
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
          if (count != 0) {
            row_counts[point] = count;
            task_touched.push_back(point);
            if (count > task_max) {
              task_max = count;
            }
          }
        }
        if (!task_touched.empty() || task_max > 0) {
          std::lock_guard<std::mutex> lock(touched_mutex);
          if (touched_rows.size() + task_touched.size() <= kTouchedCap) {
            touched_rows.insert(touched_rows.end(), task_touched.begin(),
                                task_touched.end());
          } else {
            touched_overflow = true;
          }
          max_count_out = std::max(max_count_out, task_max);
        }
      });
  return max_count_out <= capacity;
}

bool write_cross_rows_target_major(NeighborSearchData &source,
                                   NeighborSearchData &target, float radius,
                                   int capacity, IntBuffer &fixed_neighbors,
                                   IntBuffer &counts,
                                   std::vector<int> &touched_rows,
                                   bool &touched_overflow,
                                   int &max_count_out) {
  const int num_source = source.num_points;
  const int num_target = target.num_points;
  const int stride = capacity;
  const std::size_t counts_before = counts.size();
  // Rows must exist even when one side is empty: the query API reads the
  // fixed counts buffer for every source point.
  prepare_fixed_rows(fixed_neighbors, counts,
                     static_cast<std::size_t>(num_source), stride);
  // The counts buffer uses a default-init allocator (resize does not zero).
  // Zero it when it is (re)allocated; otherwise only the rows written by the
  // previous build need clearing. A capacity retry or a buffer that another
  // writer filled sets `touched_overflow` and forces a full clear.
  if (counts_before != counts.size() || touched_overflow) {
    std::fill(counts.begin(), counts.end(), 0);
    touched_rows.clear();
    touched_overflow = false;
  } else {
    for (int row : touched_rows) {
      counts[static_cast<std::size_t>(row)] = 0;
    }
    touched_rows.clear();
  }
  max_count_out = 0;
  if (num_source == 0 || num_target == 0) {
    return true;
  }
  ensure_ordered_cell_x(source);
  ensure_occupancy_cells(source);
  if (!source.ordered_x_valid || source.num_cells <= 0 ||
      source.scratch_unique_keys.size() !=
          static_cast<std::size_t>(source.num_cells)) {
    return false;
  }
  const float r2 = radius * radius;
  const float inv_cell =
      source.cell_radius > 0.0f ? 1.0f / source.cell_radius : 1.0f;
  const int stencil =
      std::max(1, static_cast<int>(std::ceil(radius * inv_cell - 1.0e-6f)));
  const int num_cells = source.num_cells;
  const bool implicit_cells = source.cell_ranges_implicit;
  const uint64_t *keys = source.scratch_unique_keys.data();
  const int *cell_starts =
      implicit_cells ? nullptr : source.cell_starts.data();
  const int *cell_ends = implicit_cells ? nullptr : source.cell_ends.data();
  const float *sx = source.sorted_x.data();
  const float *sy = source.sorted_y.data();
  const float *sz = source.sorted_z.data();
  const int *cell_x_start = source.ordered_x_start.data();
  const int x_min = source.ordered_x_min;
  const int x_span = static_cast<int>(source.ordered_x_start.size()) - 1;
  const int y_min = source.ordered_y_min;
  const int y_max = source.ordered_y_max;
  const int z_min = source.ordered_z_min;
  const int z_max = source.ordered_z_max;
  const float *tx = target.sorted_x.data();
  const float *ty = target.sorted_y.data();
  const float *tz = target.sorted_z.data();
  const int *target_index = target.original_index.data();
  // Work in order-key space: the raw key array is sorted by (key ^ mask), so
  // lower_bound must compare transformed values. Positive-coordinate grids
  // have mask == 0 and ordered fields equal raw fields.
  const uint64_t mask =
      source.cell_keys_signed_order ? SortPointKey::kSignBits : 0;
  const int field_offset = mask != 0 ? 0x100000 : 0;
  constexpr int kMaxField = 0x1FFFFF;
  auto ordered_coord = [&](int coord) { return coord + field_offset; };
  auto lower = [&](int begin, int end, uint64_t want) {
    return static_cast<int>(
        std::lower_bound(keys + begin, keys + end, want,
                         [mask](uint64_t a, uint64_t b) {
                           return (a ^ mask) < b;
                         }) -
        keys);
  };
  int *row_counts = counts.data();
  int *row_values = fixed_neighbors.data();
  constexpr int kGrain = 2048;
  constexpr std::size_t kTouchedCap = std::size_t{1} << 20;
  std::mutex touched_mutex;
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_target, kGrain),
      [&](const tbb::blocked_range<int> &range) {
        std::vector<int> task_touched;
        task_touched.reserve(16);
        int task_max = 0;
        for (int point = range.begin(); point < range.end(); ++point) {
          const float px = tx[point];
          const float py = ty[point];
          const float pz = tz[point];
          if (source.occupancy_valid &&
              !occupancy_may_contain(source, px, py, pz, radius)) {
            continue;
          }
          const int ox = ordered_coord(floor_to_cell(px * inv_cell));
          const int oy = ordered_coord(floor_to_cell(py * inv_cell));
          const int oz = ordered_coord(floor_to_cell(pz * inv_cell));
          const int neighbor = target_index[point];
          const int x0 = std::max(ox - stencil, x_min);
          const int x1 = std::min(ox + stencil, x_min + x_span - 1);
          for (int tcx = x0; tcx <= x1; ++tcx) {
            const int xb = cell_x_start[tcx - x_min];
            const int xe = cell_x_start[tcx - x_min + 1];
            if (xb >= xe) {
              continue;
            }
            const int y0 = std::max(oy - stencil, y_min);
            const int y1 = std::min(oy + stencil, y_max);
            if (y0 > y1) {
              continue;
            }
            const int yb = lower(xb, xe, make_key(tcx, y0, 0));
            const int ye =
                y1 == kMaxField ? xe : lower(xb, xe, make_key(tcx, y1 + 1, 0));
            for (int cell = yb; cell < ye;) {
              const int y =
                  static_cast<int>(((keys[cell] ^ mask) >> 21) & 0x1FFFFFULL);
              if (y > y1) {
                break;
              }
              const int y_end =
                  y == kMaxField ? ye
                                 : lower(cell, ye, make_key(tcx, y + 1, 0));
              const int z0 = std::max(oz - stencil, z_min);
              const int z1 = std::min(oz + stencil, z_max);
              if (z0 <= z1) {
                const int zb = lower(cell, y_end, make_key(tcx, y, z0));
                const int ze =
                    z1 == kMaxField
                        ? y_end
                        : lower(zb, y_end, make_key(tcx, y, z1 + 1));
                for (int c = zb; c < ze; ++c) {
                  const int begin = implicit_cells ? c : cell_starts[c];
                  const int end = implicit_cells ? c + 1 : cell_ends[c];
                  for (int p = begin; p < end; ++p) {
                    const std::size_t pi = static_cast<std::size_t>(p);
                    const float dx = sx[pi] - px;
                    const float dy = sy[pi] - py;
                    const float dz = sz[pi] - pz;
                    if (dx * dx + dy * dy + dz * dz <= r2) {
                      // Rows are indexed by the source's sorted position, the
                      // same layout the query API reads.
                      const int position = __atomic_fetch_add(
                          &row_counts[p], 1, __ATOMIC_RELAXED);
                      if (position == 0) {
                        task_touched.push_back(p);
                      }
                      if (position >= task_max) {
                        task_max = position + 1;
                      }
                      if (position < stride) {
                        row_values[pi * stride + position] = neighbor;
                      }
                    }
                  }
                }
              }
              cell = y_end;
            }
          }
        }
        if (!task_touched.empty() || task_max > 0) {
          std::lock_guard<std::mutex> lock(touched_mutex);
          if (touched_rows.size() + task_touched.size() <= kTouchedCap) {
            touched_rows.insert(touched_rows.end(), task_touched.begin(),
                                task_touched.end());
          } else {
            touched_overflow = true;
          }
          max_count_out = std::max(max_count_out, task_max);
        }
      });
  return max_count_out <= capacity;
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
                 data.scratch_point_keys_alt, data.dense_keys,
                 data.scratch_unique_keys, signed_order, nullptr, nullptr,
                 nullptr);
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
  ensure_cell_ranges(data);
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
  // Dense cells hold ~7 points, so an eight-query group covers a whole cell
  // per candidate pass instead of splitting it into two quads.
  constexpr int kSameGroup = 8;
  write_rows_parallel(
      data.num_cells, cell_grain(data.num_cells), capacity, false, counts,
      fixed_neighbors, [&](int begin, int end, const ispc::FixedRows &rows) {
        ispc::write_fixed_rows_dense(begin, end, kSameGroup, &cells, &grid,
                                     &cells, &pts, &pts, true, r2, &rows);
      });
}


void write_cross_neighbors_target_major_csr_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data) {
  const int num_source = source_data.num_points;
  const int num_target = target_data.num_points;
  if (num_source <= 0 || num_target <= 0 || !source_data.dense_valid ||
      source_data.cell_stencil != 1) {
    return;
  }
  // Cells come from the source grid's own cell size; the search radius is the
  // pair's.
  const float inv_radius = 1.0f / source_data.cell_radius;
  const float r2 = query_data.radius * query_data.radius;
  const ispc::Points targets = points_of(target_data);
  const ispc::DenseGrid grid = dense_of(source_data);
  const ispc::Cells cells = cells_of(source_data);
  const ispc::Points candidates = points_of(source_data);
  IntBuffer &counts = query_data.counts;
  IntBuffer &row_ptr = query_data.row_ptr;
  IntBuffer &col_idx = query_data.col_idx;
  auto zero_counts = [&]() {
    constexpr int kGrain = 1 << 16;
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_source, kGrain),
        [&](const tbb::blocked_range<int> &range) {
          std::fill(counts.begin() + range.begin(),
                    counts.begin() + range.end(), 0);
        });
  };
  counts.resize(static_cast<std::size_t>(num_source));
  row_ptr.resize(static_cast<std::size_t>(num_source) + 1);
  zero_counts();
  constexpr int kTargetBlock = 64;
  // A target's hit count changes slowly between frames, so the previous
  // frame's counts size compact recorded-hit slots: pass 0 records the hit
  // rows there and the fill pass walks them instead of rescanning every
  // candidate. A target that outgrew its slot is rescanned in the fill pass.
  IntBuffer &slots = query_data.csr_slots;
  IntBuffer &slot_offset = query_data.csr_slot_offset;
  IntBuffer &slot_capacity = query_data.csr_slot_capacity;
  IntBuffer &slot_counts = query_data.csr_slot_counts;
  IntBuffer &hit_totals = query_data.csr_hit_totals;
  IntBuffer &prev_hits = query_data.csr_prev_hits;
  {
    const bool have_prev =
        query_data.csr_slots_valid &&
        prev_hits.size() == static_cast<std::size_t>(num_target);
    slot_offset.resize(static_cast<std::size_t>(num_target));
    slot_capacity.resize(static_cast<std::size_t>(num_target));
    slot_counts.resize(static_cast<std::size_t>(num_target));
    hit_totals.resize(static_cast<std::size_t>(num_target));
    // Offsets are serial over targets (tens of thousands of entries).
    int64_t total_slots = 0;
    for (int t = 0; t < num_target; ++t) {
      slot_offset[static_cast<std::size_t>(t)] =
          static_cast<int>(total_slots);
      // First frame: a guess that covers the emitter's ~33 hits per target;
      // targets above it simply rescan in the fill pass.
      const int capacity =
          have_prev ? prev_hits[static_cast<std::size_t>(t)] + 8 : 64;
      slot_capacity[static_cast<std::size_t>(t)] = capacity;
      total_slots += capacity;
    }
    slots.resize(static_cast<std::size_t>(total_slots));
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_target, kTargetBlock),
        [&](const tbb::blocked_range<int> &range) {
          ispc::target_major_csr_record(
              range.begin(), range.end(), &targets, &grid, &cells,
              &candidates, inv_radius, r2, counts.data(), slot_offset.data(),
              slot_capacity.data(), slot_counts.data(), hit_totals.data(),
              slots.data(), neighbor_permutation_table());
        });
  }
  const int total = tbb::parallel_scan(
      tbb::blocked_range<int>(0, num_source), 0,
      [&](const tbb::blocked_range<int> &range, int sum, bool final_scan) {
        int running = sum;
        for (int i = range.begin(); i < range.end(); ++i) {
          if (final_scan) {
            row_ptr[i] = running;
          }
          running += counts[i];
        }
        return running;
      },
      [](int left, int right) { return left + right; });
  row_ptr[num_source] = total;
  col_idx.resize(static_cast<std::size_t>(total));
  // Pass 1: reuse counts as the per-row fill cursor; it ends at the exact row
  // counts, which the pair keeps for its statistics and skin filtering.
  zero_counts();
  {
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_target, kTargetBlock),
        [&](const tbb::blocked_range<int> &range) {
          ispc::target_major_csr_fill(
              range.begin(), range.end(), &targets, &grid, &cells,
              &candidates, inv_radius, r2, counts.data(), row_ptr.data(),
              col_idx.data(), slot_offset.data(), slots.data(),
              slot_counts.data());
        });
    // The next frame's slot capacities come from this frame's exact counts.
    prev_hits.resize(static_cast<std::size_t>(num_target));
    std::memcpy(prev_hits.data(), hit_totals.data(),
                static_cast<std::size_t>(num_target) * sizeof(int));
    query_data.csr_slots_valid = true;
  }
  query_data.num_points = num_source;
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
  // Sparse targets leave enough work per candidate load to share it between
  // eight queries (one cell); denser targets keep pairs.
  const int group = sparse_target ? 8 : 2;
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