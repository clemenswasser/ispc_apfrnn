#include "apfrnn.hpp"

#include "kernel_ispc.h"

#include <cstdint>
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

inline int floor_to_cell(float value) {
  int cell = static_cast<int>(value);
  return cell - static_cast<int>(cell > value);
}

inline void sort_point_set(
    const std::vector<float> &x, const std::vector<float> &y,
    const std::vector<float> &z, float inv_radius, std::vector<float> &sorted_x,
    std::vector<float> &sorted_y, std::vector<float> &sorted_z,
    IntBuffer &original_index, IntBuffer &cell_starts, IntBuffer &cell_ends,
    std::vector<SortPointKey> &point_keys, std::vector<uint64_t> &unique_keys) {
  int const num_points = static_cast<int>(x.size());
  point_keys.resize(num_points);

  tbb::parallel_for(0, num_points, [&](int i) {
    int cx = floor_to_cell(x[i] * inv_radius);
    int cy = floor_to_cell(y[i] * inv_radius);
    int cz = floor_to_cell(z[i] * inv_radius);
    point_keys[i].key = make_key(cx, cy, cz);
    point_keys[i].index = i;
  });

  tbb::parallel_sort(point_keys.begin(), point_keys.end());

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

} // namespace

void build_neighbor_search_data_inplace(NeighborSearchData &data,
                                        const std::vector<float> &x,
                                        const std::vector<float> &y,
                                        const std::vector<float> &z,
                                        float radius) {
  data.num_points = static_cast<int>(x.size());
  data.radius = radius;
  float inv_radius = 1.0f / radius;

  if (data.num_points == 0) {
    data.num_cells = 0;
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
    data.hash_keys.clear();
    data.hash_vals.clear();
    data.hash_mask = 0;
    data.row_ptr.clear();
    data.row_ptr.push_back(0);
    return;
  }

  sort_point_set(x, y, z, inv_radius, data.sorted_x, data.sorted_y,
                 data.sorted_z, data.original_index, data.cell_starts,
                 data.cell_ends, data.scratch_point_keys,
                 data.scratch_unique_keys);
  auto const &unique_keys = data.scratch_unique_keys;

  int num_cells = static_cast<int>(data.cell_starts.size());
  data.num_cells = num_cells;
  build_hash_table(unique_keys, data.hash_keys, data.hash_vals, data.empty_key,
                   data.hash_mask);

  data.scratch_cached_neighbor_cells.resize(
      static_cast<std::size_t>(num_cells) * 27);
  data.scratch_cached_neighbor_counts.resize(num_cells);
  data.cell_num_neighbors.resize(num_cells);
  data.cell_neighbor_offset.resize(num_cells);

  tbb::parallel_for(tbb::blocked_range<int>(0, num_cells, 64),
                    [&](const tbb::blocked_range<int> &range) {
                      ispc::collect_neighbor_cells_ispc(
                          range.begin(), range.end(), unique_keys.data(),
                          data.hash_keys.data(), data.hash_vals.data(),
                          data.empty_key, data.hash_mask,
                          data.scratch_cached_neighbor_cells.data(),
                          data.scratch_cached_neighbor_counts.data());
                    });

  tbb::parallel_for(tbb::blocked_range<int>(0, num_cells, 64),
                    [&](const tbb::blocked_range<int> &range) {
                      ispc::count_merged_neighbor_ranges_ispc(
                          range.begin(), range.end(), data.cell_starts.data(),
                          data.cell_ends.data(),
                          data.scratch_cached_neighbor_cells.data(),
                          data.scratch_cached_neighbor_counts.data(),
                          data.cell_num_neighbors.data());
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
      tbb::blocked_range<int>(0, num_cells, 64),
      [&](const tbb::blocked_range<int> &range) {
        ispc::write_merged_neighbor_ranges_ispc(
            range.begin(), range.end(), data.cell_starts.data(),
            data.cell_ends.data(), data.scratch_cached_neighbor_cells.data(),
            data.scratch_cached_neighbor_counts.data(),
            data.cell_neighbor_offset.data(), data.cell_neighbor_starts.data(),
            data.cell_neighbor_ends.data());
      });

  data.counts.resize(data.num_points);
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

NeighborSearchData build_neighbor_search_data(const std::vector<float> &x,
                                              const std::vector<float> &y,
                                              const std::vector<float> &z,
                                              float radius) {
  NeighborSearchData data;
  build_neighbor_search_data_inplace(data, x, y, z, radius);
  return data;
}

void build_cross_neighbor_search_data_inplace(
    CrossNeighborSearchData &data, const std::vector<float> &query_x,
    const std::vector<float> &query_y, const std::vector<float> &query_z,
    const NeighborSearchData &target_data, float radius) {
  data.num_points = static_cast<int>(query_x.size());
  data.radius = radius;

  if (data.num_points == 0) {
    data.num_cells = 0;
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
                 data.scratch_unique_keys);
  auto const &unique_keys = data.scratch_unique_keys;

  data.num_cells = static_cast<int>(data.cell_starts.size());
  data.scratch_cached_neighbor_cells.resize(
      static_cast<std::size_t>(data.num_cells) * 27);
  data.scratch_cached_neighbor_counts.resize(data.num_cells);
  data.cell_num_neighbors.resize(data.num_cells);
  data.cell_neighbor_offset.resize(data.num_cells);

  tbb::parallel_for(tbb::blocked_range<int>(0, data.num_cells, 64),
                    [&](const tbb::blocked_range<int> &range) {
                      ispc::collect_neighbor_cells_ispc(
                          range.begin(), range.end(), unique_keys.data(),
                          target_data.hash_keys.data(),
                          target_data.hash_vals.data(), target_data.empty_key,
                          target_data.hash_mask,
                          data.scratch_cached_neighbor_cells.data(),
                          data.scratch_cached_neighbor_counts.data());
                    });

  tbb::parallel_for(tbb::blocked_range<int>(0, data.num_cells, 64),
                    [&](const tbb::blocked_range<int> &range) {
                      ispc::count_merged_neighbor_ranges_ispc(
                          range.begin(), range.end(),
                          target_data.cell_starts.data(),
                          target_data.cell_ends.data(),
                          data.scratch_cached_neighbor_cells.data(),
                          data.scratch_cached_neighbor_counts.data(),
                          data.cell_num_neighbors.data());
                    });

  int total_cell_neighbors = tbb::parallel_scan(
      tbb::blocked_range<int>(0, data.num_cells), 0,
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
      tbb::blocked_range<int>(0, data.num_cells, 64),
      [&](const tbb::blocked_range<int> &range) {
        ispc::write_merged_neighbor_ranges_ispc(
            range.begin(), range.end(), target_data.cell_starts.data(),
            target_data.cell_ends.data(),
            data.scratch_cached_neighbor_cells.data(),
            data.scratch_cached_neighbor_counts.data(),
            data.cell_neighbor_offset.data(), data.cell_neighbor_starts.data(),
            data.cell_neighbor_ends.data());
      });

  data.counts.resize(data.num_points);
  tbb::parallel_for(
      tbb::blocked_range<int>(0, data.num_cells, 64),
      [&](const tbb::blocked_range<int> &range) {
        ispc::count_cross_neighbors_ispc(
            range.begin(), range.end(), data.cell_starts.data(),
            data.cell_ends.data(), data.cell_neighbor_offset.data(),
            data.cell_num_neighbors.data(), data.cell_neighbor_starts.data(),
            data.cell_neighbor_ends.data(), data.radius * data.radius,
            data.sorted_x.data(), data.sorted_y.data(), data.sorted_z.data(),
            target_data.sorted_x.data(), target_data.sorted_y.data(),
            target_data.sorted_z.data(), data.counts.data());
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

void write_cross_neighbors_range(CrossNeighborSearchData &query_data,
                                 const NeighborSearchData &target_data,
                                 int start_cell, int end_cell) {
  ispc::write_cross_neighbors_ispc(
      start_cell, end_cell, query_data.cell_starts.data(),
      query_data.cell_ends.data(), query_data.cell_neighbor_offset.data(),
      query_data.cell_num_neighbors.data(),
      query_data.cell_neighbor_starts.data(),
      query_data.cell_neighbor_ends.data(),
      query_data.radius * query_data.radius, query_data.sorted_x.data(),
      query_data.sorted_y.data(), query_data.sorted_z.data(),
      target_data.sorted_x.data(), target_data.sorted_y.data(),
      target_data.sorted_z.data(), target_data.original_index.data(),
      query_data.row_ptr.data(), query_data.col_idx.data());
}

void write_cross_neighbors_parallel(CrossNeighborSearchData &query_data,
                                    const NeighborSearchData &target_data) {
  if (query_data.num_points == 0) {
    return;
  }

  tbb::parallel_for(tbb::blocked_range<int>(0, query_data.num_cells, 64),
                    [&](const tbb::blocked_range<int> &range) {
                      write_cross_neighbors_range(query_data, target_data,
                                                  range.begin(), range.end());
                    });
}

} // namespace apfrnn