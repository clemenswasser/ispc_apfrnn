#pragma once

#include <cstdint>
#include <vector>

namespace apfrnn {

struct NeighborSearchData {
  int num_points = 0;
  float radius = 0.0f;
  uint32_t hash_size = 0;
  uint32_t hash_mask = 0;
  std::vector<float> sorted_x;
  std::vector<float> sorted_y;
  std::vector<float> sorted_z;
  std::vector<int> original_index;
  std::vector<int> bin_start;
  std::vector<int> bin_end;
  std::vector<int> counts;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
};

uint32_t spatial_hash(float x, float y, float z, float radius,
                      uint32_t hash_mask);

NeighborSearchData build_neighbor_search_data(const std::vector<float> &x,
                                              const std::vector<float> &y,
                                              const std::vector<float> &z,
                                              float radius,
                                              uint32_t hash_size = 1u << 22);

void write_neighbors_range(NeighborSearchData &data, int start_query,
                           int end_query);

void write_neighbors_parallel(NeighborSearchData &data);

} // namespace apfrnn