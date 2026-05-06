#pragma once

#include <cstdint>
#include <vector>

namespace apfrnn {

struct NeighborSearchData {
  int num_points = 0;
  int num_cells = 0;
  float radius = 0.0f;
  std::vector<float> sorted_x;
  std::vector<float> sorted_y;
  std::vector<float> sorted_z;
  std::vector<int> original_index;
  std::vector<int> cell_starts;
  std::vector<int> cell_ends;
  std::vector<int> cell_neighbor_offset;
  std::vector<int> cell_num_neighbors;
  std::vector<int> cell_neighbor_starts;
  std::vector<int> cell_neighbor_ends;
  std::vector<int> counts;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<uint64_t> hash_keys;
  std::vector<int> hash_vals;
  uint64_t empty_key = ~0ULL;
  uint32_t hash_mask = 0;
};

struct CrossNeighborSearchData {
  int num_points = 0;
  int num_cells = 0;
  float radius = 0.0f;
  std::vector<float> sorted_x;
  std::vector<float> sorted_y;
  std::vector<float> sorted_z;
  std::vector<int> original_index;
  std::vector<int> cell_starts;
  std::vector<int> cell_ends;
  std::vector<int> cell_neighbor_offset;
  std::vector<int> cell_num_neighbors;
  std::vector<int> cell_neighbor_starts;
  std::vector<int> cell_neighbor_ends;
  std::vector<int> counts;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
};

NeighborSearchData build_neighbor_search_data(const std::vector<float> &x,
                                              const std::vector<float> &y,
                                              const std::vector<float> &z,
                                              float radius);

CrossNeighborSearchData build_cross_neighbor_search_data(
    const std::vector<float> &query_x, const std::vector<float> &query_y,
    const std::vector<float> &query_z, const NeighborSearchData &target_data,
    float radius);

void write_neighbors_range(NeighborSearchData &data, int start_cell,
                           int end_cell);

void write_neighbors_parallel(NeighborSearchData &data);

void write_cross_neighbors_range(CrossNeighborSearchData &query_data,
                                 const NeighborSearchData &target_data,
                                 int start_cell, int end_cell);

void write_cross_neighbors_parallel(CrossNeighborSearchData &query_data,
                                    const NeighborSearchData &target_data);

} // namespace apfrnn