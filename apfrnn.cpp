#include "apfrnn.hpp"

#include "kernel_ispc.h"

#include <cmath>
#include <cstdint>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>

namespace apfrnn {

namespace {

struct PointKey {
  uint64_t key;
  int index;

  bool operator<(const PointKey &other) const { return key < other.key; }
};

inline uint64_t make_key(int cx, int cy, int cz) {
  uint64_t ux = static_cast<uint64_t>(static_cast<uint32_t>(cx)) & 0x1FFFFFULL;
  uint64_t uy = static_cast<uint64_t>(static_cast<uint32_t>(cy)) & 0x1FFFFFULL;
  uint64_t uz = static_cast<uint64_t>(static_cast<uint32_t>(cz)) & 0x1FFFFFULL;
  return (ux << 42) | (uy << 21) | uz;
}

inline int unpack_coord(uint64_t key, int shift) {
  int coord =
      static_cast<int>(static_cast<uint32_t>((key >> shift) & 0x1FFFFF));
  if (coord & 0x100000) {
    coord |= 0xFFE00000;
  }
  return coord;
}

inline int find_cell_index(const std::vector<uint64_t> &ht_keys,
                           const std::vector<int> &ht_vals, uint64_t empty_key,
                           uint32_t hash_mask, uint64_t key) {
  uint64_t h64 = (key ^ (key >> 33)) * 0xff51afd7ed558ccdULL;
  h64 ^= h64 >> 33;
  uint32_t h = static_cast<uint32_t>(h64) & hash_mask;

  while (true) {
    uint64_t ht_key = ht_keys[h];
    if (ht_key == key) {
      return ht_vals[h];
    }
    if (ht_key == empty_key) {
      return -1;
    }
    h = (h + 1) & hash_mask;
  }
}

inline int collect_neighbor_cells(const std::vector<uint64_t> &ht_keys,
                                  const std::vector<int> &ht_vals,
                                  uint64_t empty_key, uint32_t hash_mask,
                                  int cx, int cy, int cz,
                                  int neighbor_cells[27]) {
  int count = 0;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int neighbor_cell =
            find_cell_index(ht_keys, ht_vals, empty_key, hash_mask,
                            make_key(cx + dx, cy + dy, cz + dz));
        if (neighbor_cell != -1) {
          neighbor_cells[count++] = neighbor_cell;
        }
      }
    }
  }

  for (int i = 1; i < count; ++i) {
    int cell = neighbor_cells[i];
    int j = i - 1;
    while (j >= 0 && neighbor_cells[j] > cell) {
      neighbor_cells[j + 1] = neighbor_cells[j];
      --j;
    }
    neighbor_cells[j + 1] = cell;
  }

  return count;
}

inline int count_merged_neighbor_ranges(const NeighborSearchData &data,
                                        const int neighbor_cells[27],
                                        int count) {
  int merged = 0;
  int current_end = -1;

  for (int i = 0; i < count; ++i) {
    int neighbor_cell = neighbor_cells[i];
    int neighbor_start = data.cell_starts[neighbor_cell];
    int neighbor_end = data.cell_ends[neighbor_cell];

    if (current_end == neighbor_start) {
      current_end = neighbor_end;
      continue;
    }

    if (current_end != -1) {
      ++merged;
    }
    current_end = neighbor_end;
  }

  if (current_end != -1) {
    ++merged;
  }

  return merged;
}

inline void write_merged_neighbor_ranges(const NeighborSearchData &data,
                                         const int neighbor_cells[27],
                                         int count, int offset,
                                         int *neighbor_starts,
                                         int *neighbor_ends) {
  int current_start = -1;
  int current_end = -1;

  for (int i = 0; i < count; ++i) {
    int neighbor_cell = neighbor_cells[i];
    int neighbor_start = data.cell_starts[neighbor_cell];
    int neighbor_end = data.cell_ends[neighbor_cell];

    if (current_end == neighbor_start) {
      current_end = neighbor_end;
      continue;
    }

    if (current_start != -1) {
      neighbor_starts[offset] = current_start;
      neighbor_ends[offset] = current_end;
      ++offset;
    }

    current_start = neighbor_start;
    current_end = neighbor_end;
  }

  if (current_start != -1) {
    neighbor_starts[offset] = current_start;
    neighbor_ends[offset] = current_end;
  }
}

} // namespace

NeighborSearchData build_neighbor_search_data(const std::vector<float> &x,
                                              const std::vector<float> &y,
                                              const std::vector<float> &z,
                                              float radius) {
  NeighborSearchData data;
  data.num_points = static_cast<int>(x.size());
  data.radius = radius;
  float inv_radius = 1.0f / radius;

  if (data.num_points == 0) {
    data.row_ptr.push_back(0);
    return data;
  }

  std::vector<PointKey> point_keys(data.num_points);

  tbb::parallel_for(0, data.num_points, [&](int i) {
    int cx = static_cast<int>(std::floor(x[i] * inv_radius));
    int cy = static_cast<int>(std::floor(y[i] * inv_radius));
    int cz = static_cast<int>(std::floor(z[i] * inv_radius));
    point_keys[i].key = make_key(cx, cy, cz);
    point_keys[i].index = i;
  });

  tbb::parallel_sort(point_keys.begin(), point_keys.end());

  data.sorted_x.resize(data.num_points);
  data.sorted_y.resize(data.num_points);
  data.sorted_z.resize(data.num_points);
  data.original_index.resize(data.num_points);
  std::vector<uint64_t> sorted_keys(data.num_points);

  tbb::parallel_for(0, data.num_points, [&](int i) {
    int idx = point_keys[i].index;
    data.sorted_x[i] = x[idx];
    data.sorted_y[i] = y[idx];
    data.sorted_z[i] = z[idx];
    data.original_index[i] = idx;
    sorted_keys[i] = point_keys[i].key;
  });

  std::vector<uint64_t> unique_keys;
  unique_keys.reserve(data.num_points);
  data.cell_starts.push_back(0);
  unique_keys.push_back(sorted_keys[0]);

  for (int i = 1; i < data.num_points; ++i) {
    if (sorted_keys[i] != sorted_keys[i - 1]) {
      data.cell_ends.push_back(i);
      data.cell_starts.push_back(i);
      unique_keys.push_back(sorted_keys[i]);
    }
  }
  data.cell_ends.push_back(data.num_points);

  int num_cells = static_cast<int>(data.cell_starts.size());
  data.num_cells = num_cells;
  int hash_capacity = 1;
  while (hash_capacity < num_cells * 2) {
    hash_capacity <<= 1;
  }

  uint64_t empty_key = ~0ULL;
  uint32_t hash_mask = static_cast<uint32_t>(hash_capacity - 1);
  std::vector<uint64_t> ht_keys(hash_capacity, empty_key);
  std::vector<int> ht_vals(hash_capacity, -1);

  for (int cell = 0; cell < num_cells; ++cell) {
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

  data.cell_num_neighbors.resize(num_cells);
  data.cell_neighbor_offset.resize(num_cells);

  tbb::parallel_for(0, num_cells, [&](int cell) {
    uint64_t key = unique_keys[cell];
    int cx = unpack_coord(key, 42);
    int cy = unpack_coord(key, 21);
    int cz = unpack_coord(key, 0);

    int neighbor_cells[27];
    int neighbor_count = collect_neighbor_cells(
        ht_keys, ht_vals, empty_key, hash_mask, cx, cy, cz, neighbor_cells);
    data.cell_num_neighbors[cell] =
        count_merged_neighbor_ranges(data, neighbor_cells, neighbor_count);
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

  tbb::parallel_for(0, num_cells, [&](int cell) {
    uint64_t key = unique_keys[cell];
    int cx = unpack_coord(key, 42);
    int cy = unpack_coord(key, 21);
    int cz = unpack_coord(key, 0);
    int offset = data.cell_neighbor_offset[cell];

    int neighbor_cells[27];
    int neighbor_count = collect_neighbor_cells(
        ht_keys, ht_vals, empty_key, hash_mask, cx, cy, cz, neighbor_cells);
    write_merged_neighbor_ranges(data, neighbor_cells, neighbor_count, offset,
                                 data.cell_neighbor_starts.data(),
                                 data.cell_neighbor_ends.data());
  });

  data.counts.assign(data.num_points, 0);
  tbb::parallel_for(
      tbb::blocked_range<int>(0, data.num_cells, 64),
      [&](const tbb::blocked_range<int> &range) {
        ispc::count_neighbors_ispc(
            range.begin(), range.end(), data.cell_starts.data(),
            data.cell_ends.data(), data.cell_neighbor_offset.data(),
            data.cell_num_neighbors.data(), data.cell_neighbor_starts.data(),
            data.cell_neighbor_ends.data(), data.radius * data.radius,
            data.sorted_x.data(), data.sorted_y.data(), data.sorted_z.data(),
            data.original_index.data(), data.counts.data());
      });

  data.row_ptr.assign(data.num_points + 1, 0);
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
  data.col_idx.assign(total_neighbors, 0);

  return data;
}

void write_neighbors_range(NeighborSearchData &data, int start_cell,
                           int end_cell) {
  ispc::write_neighbors_ispc(
      start_cell, end_cell, data.cell_starts.data(), data.cell_ends.data(),
      data.cell_neighbor_offset.data(), data.cell_num_neighbors.data(),
      data.cell_neighbor_starts.data(), data.cell_neighbor_ends.data(),
      data.radius * data.radius, data.sorted_x.data(), data.sorted_y.data(),
      data.sorted_z.data(), data.original_index.data(), data.row_ptr.data(),
      data.col_idx.data());
}

void write_neighbors_parallel(NeighborSearchData &data) {
  if (data.num_points == 0) {
    return;
  }

  tbb::parallel_for(tbb::blocked_range<int>(0, data.num_cells, 64),
                    [&](const tbb::blocked_range<int> &range) {
                      write_neighbors_range(data, range.begin(), range.end());
                    });
}

} // namespace apfrnn