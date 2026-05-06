#include "apfrnn.hpp"

#include "kernel_ispc.h"

#include <cmath>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>

namespace apfrnn {

namespace {

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
    return data;
  }

  std::vector<uint64_t> keys(data.num_points);
  std::vector<int> sort_idx(data.num_points);

  tbb::parallel_for(0, data.num_points, [&](int i) {
    int cx = static_cast<int>(std::floor(x[i] * inv_radius));
    int cy = static_cast<int>(std::floor(y[i] * inv_radius));
    int cz = static_cast<int>(std::floor(z[i] * inv_radius));
    keys[i] = make_key(cx, cy, cz);
    sort_idx[i] = i;
  });

  tbb::parallel_sort(
      sort_idx.begin(), sort_idx.end(),
      [&](int left, int right) { return keys[left] < keys[right]; });

  data.sorted_x.resize(data.num_points);
  data.sorted_y.resize(data.num_points);
  data.sorted_z.resize(data.num_points);
  data.original_index.resize(data.num_points);
  std::vector<uint64_t> sorted_keys(data.num_points);

  tbb::parallel_for(0, data.num_points, [&](int i) {
    int idx = sort_idx[i];
    data.sorted_x[i] = x[idx];
    data.sorted_y[i] = y[idx];
    data.sorted_z[i] = z[idx];
    data.original_index[i] = idx;
    sorted_keys[i] = keys[idx];
  });

  std::vector<uint64_t> unique_keys;
  unique_keys.reserve(data.num_points);
  data.cell_starts.push_back(0);
  unique_keys.push_back(sorted_keys[0]);
  data.point_cell_idx.resize(data.num_points);
  data.point_cell_idx[0] = 0;

  int current_cell = 0;
  for (int i = 1; i < data.num_points; ++i) {
    if (sorted_keys[i] != sorted_keys[i - 1]) {
      data.cell_ends.push_back(i);
      data.cell_starts.push_back(i);
      unique_keys.push_back(sorted_keys[i]);
      ++current_cell;
    }
    data.point_cell_idx[i] = current_cell;
  }
  data.cell_ends.push_back(data.num_points);

  int num_cells = static_cast<int>(data.cell_starts.size());
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

    int neighbors = 0;
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (find_cell_index(ht_keys, ht_vals, empty_key, hash_mask,
                              make_key(cx + dx, cy + dy, cz + dz)) != -1) {
            ++neighbors;
          }
        }
      }
    }
    data.cell_num_neighbors[cell] = neighbors;
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

    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          int neighbor_cell =
              find_cell_index(ht_keys, ht_vals, empty_key, hash_mask,
                              make_key(cx + dx, cy + dy, cz + dz));
          if (neighbor_cell != -1) {
            data.cell_neighbor_starts[offset] = data.cell_starts[neighbor_cell];
            data.cell_neighbor_ends[offset] = data.cell_ends[neighbor_cell];
            ++offset;
          }
        }
      }
    }
  });

  data.counts.assign(data.num_points, 0);
  tbb::parallel_for(
      tbb::blocked_range<int>(0, data.num_points, 1024),
      [&](const tbb::blocked_range<int> &range) {
        ispc::count_neighbors_ispc(
            range.begin(), range.end(), data.point_cell_idx.data(),
            data.cell_neighbor_offset.data(), data.cell_num_neighbors.data(),
            data.cell_neighbor_starts.data(), data.cell_neighbor_ends.data(),
            data.radius * data.radius, data.sorted_x.data(),
            data.sorted_y.data(), data.sorted_z.data(),
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

void write_neighbors_range(NeighborSearchData &data, int start_query,
                           int end_query) {
  ispc::write_neighbors_ispc(
      start_query, end_query, data.point_cell_idx.data(),
      data.cell_neighbor_offset.data(), data.cell_num_neighbors.data(),
      data.cell_neighbor_starts.data(), data.cell_neighbor_ends.data(),
      data.radius * data.radius, data.sorted_x.data(), data.sorted_y.data(),
      data.sorted_z.data(), data.original_index.data(), data.row_ptr.data(),
      data.col_idx.data());
}

void write_neighbors_parallel(NeighborSearchData &data) {
  tbb::parallel_for(tbb::blocked_range<int>(0, data.num_points, 1024),
                    [&](const tbb::blocked_range<int> &range) {
                      write_neighbors_range(data, range.begin(), range.end());
                    });
}

} // namespace apfrnn