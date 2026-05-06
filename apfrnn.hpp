#pragma once

#include <vector>

namespace apfrnn {

struct NeighborSearchData {
  int num_points = 0;
  float radius = 0.0f;
  std::vector<float> sorted_x;
  std::vector<float> sorted_y;
  std::vector<float> sorted_z;
  std::vector<int> original_index;
  std::vector<int> point_cell_idx;
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

void write_neighbors_range(NeighborSearchData &data, int start_query,
                           int end_query);

void write_neighbors_parallel(NeighborSearchData &data);

} // namespace apfrnn