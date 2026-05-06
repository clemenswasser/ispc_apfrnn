#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace apfrnn {

template <typename T> struct DefaultInitAllocator : std::allocator<T> {
  using value_type = T;

  DefaultInitAllocator() = default;

  template <typename U>
  DefaultInitAllocator(const DefaultInitAllocator<U> &) noexcept {}

  template <typename U> struct rebind {
    using other = DefaultInitAllocator<U>;
  };

  template <typename U, typename... Args>
  void construct(U *ptr, Args &&...args) {
    if constexpr (sizeof...(Args) == 0 &&
                  std::is_trivially_default_constructible_v<U>) {
      ::new (static_cast<void *>(ptr)) U;
    } else {
      ::new (static_cast<void *>(ptr)) U(std::forward<Args>(args)...);
    }
  }
};

using IntBuffer = std::vector<int, DefaultInitAllocator<int>>;

struct NeighborSearchData {
  int num_points = 0;
  int num_cells = 0;
  float radius = 0.0f;
  std::vector<float> sorted_x;
  std::vector<float> sorted_y;
  std::vector<float> sorted_z;
  IntBuffer original_index;
  IntBuffer cell_starts;
  IntBuffer cell_ends;
  IntBuffer cell_neighbor_offset;
  IntBuffer cell_num_neighbors;
  IntBuffer cell_neighbor_starts;
  IntBuffer cell_neighbor_ends;
  IntBuffer counts;
  IntBuffer row_ptr;
  IntBuffer col_idx;
  std::vector<uint64_t> hash_keys;
  IntBuffer hash_vals;
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
  IntBuffer original_index;
  IntBuffer cell_starts;
  IntBuffer cell_ends;
  IntBuffer cell_neighbor_offset;
  IntBuffer cell_num_neighbors;
  IntBuffer cell_neighbor_starts;
  IntBuffer cell_neighbor_ends;
  IntBuffer counts;
  IntBuffer row_ptr;
  IntBuffer col_idx;
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