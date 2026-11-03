#pragma once

// Internal grid and neighbor-row building blocks used by NSearch. Not part of
// the public API; tests and benchmarks include it directly.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace apfrnn {

template <typename T, std::size_t Alignment = 32>
struct DefaultInitAllocator {
  using value_type = T;
  using is_always_equal = std::true_type;

  DefaultInitAllocator() = default;

  template <typename U>
  DefaultInitAllocator(const DefaultInitAllocator<U, Alignment> &) noexcept {}

  template <typename U> struct rebind {
    using other = DefaultInitAllocator<U, Alignment>;
  };

  T *allocate(std::size_t count) {
    return static_cast<T *>(::operator new(
        count * sizeof(T), std::align_val_t(Alignment)));
  }

  void deallocate(T *ptr, std::size_t) noexcept {
    ::operator delete(ptr, std::align_val_t(Alignment));
  }

  template <typename U, typename... Args>
  void construct(U *ptr, Args &&...args) {
    if constexpr (sizeof...(Args) == 0 &&
                  std::is_trivially_default_constructible_v<U>) {
      ::new (static_cast<void *>(ptr)) U;
    } else {
      ::new (static_cast<void *>(ptr)) U(std::forward<Args>(args)...);
    }
  }

  template <typename U>
  bool operator==(const DefaultInitAllocator<U, Alignment> &) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const DefaultInitAllocator<U, Alignment> &) const noexcept {
    return false;
  }
};

using IntBuffer = std::vector<int, DefaultInitAllocator<int>>;

// Compaction lookup table shared by the row writers. Internal, but exposed so
// the CSR regression can call the same ISPC writer as the library.
const int *neighbor_permutation_table();

// Axis-aligned bounding box of a point set. Empty when min > max.
struct Bounds {
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float min_z = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  float max_z = -std::numeric_limits<float>::infinity();

  bool empty() const { return min_x > max_x; }
  void add(float x, float y, float z) {
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    min_z = std::min(min_z, z);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
    max_z = std::max(max_z, z);
  }
  void add(const Bounds &other) {
    min_x = std::min(min_x, other.min_x);
    min_y = std::min(min_y, other.min_y);
    min_z = std::min(min_z, other.min_z);
    max_x = std::max(max_x, other.max_x);
    max_y = std::max(max_y, other.max_y);
    max_z = std::max(max_z, other.max_z);
  }
};

struct SyncResult {
  float max_distance_sq = 0.0f;
  Bounds bounds;
};

// Copies n interleaved xyz points into x/y/z (sized n). In the same pass it
// measures the largest squared displacement against the previous x/y/z (only
// when `comparable`) and the bounding box. Parallel over TBB chunks, each
// vectorized by an ISPC kernel. Double input is converted to float.
SyncResult sync_points(const float *xyz, int n, bool comparable,
                       std::vector<float> &x, std::vector<float> &y,
                       std::vector<float> &z);
SyncResult sync_points(const double *xyz, int n, bool comparable,
                       std::vector<float> &x, std::vector<float> &y,
                       std::vector<float> &z);

struct SortPointKey {
  uint64_t key;
  int index;

  bool operator<(const SortPointKey &other) const {
    if (key != other.key) {
      return key < other.key;
    }
    return index < other.index;
  }
};

struct NeighborSearchData {
  int num_points = 0;
  int num_cells = 0;
  float radius = 0.0f;
  float cell_radius = 0.0f;
  int cell_stencil = 1;
  int max_neighbor_cells = 27;
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
  // Optional shared ranges (the per-target dense cache).
  const IntBuffer *shared_starts = nullptr;
  const IntBuffer *shared_ends = nullptr;
  IntBuffer counts;
  IntBuffer row_ptr;
  IntBuffer col_idx;
  std::vector<uint64_t> hash_keys;
  IntBuffer hash_vals;
  std::vector<SortPointKey> scratch_point_keys;
  std::vector<uint64_t> scratch_unique_keys;
  IntBuffer scratch_cached_neighbor_cells;
  IntBuffer scratch_cached_neighbor_counts;
  IntBuffer block_begin;
  IntBuffer block_end;
  IntBuffer block_cell;
  std::vector<float> block_min_x;
  std::vector<float> block_min_y;
  std::vector<float> block_min_z;
  std::vector<float> block_max_x;
  std::vector<float> block_max_y;
  std::vector<float> block_max_z;
  std::vector<float> cell_min_x;
  std::vector<float> cell_min_y;
  std::vector<float> cell_min_z;
  std::vector<float> cell_max_x;
  std::vector<float> cell_max_y;
  std::vector<float> cell_max_z;
  IntBuffer cell_block_start;
  IntBuffer cell_block_count;
  bool blocks_valid = false;
  int max_cell_points = 0;
  uint64_t empty_key = ~0ULL;
  uint32_t hash_mask = 0;

  // Compact dense cell lookup. Valid only when the AABB cell count is below
  // the configured memory limit; otherwise the hash table above is used.
  IntBuffer dense_cells;
  int dense_nx = 0;
  int dense_ny = 0;
  int dense_nz = 0;
  int dense_min_x = 0;
  int dense_min_y = 0;
  int dense_min_z = 0;
  bool dense_valid = false;

  // Nine z-merged target ranges for every dense grid cell. Shared by every
  // source set that uses this set as a target.
  bool dense_cache_valid = false;
  IntBuffer dense_cache_offsets;
  IntBuffer dense_cache_counts;
  IntBuffer dense_cache_starts;
  IntBuffer dense_cache_ends;
};

struct CrossNeighborSearchData {
  int num_points = 0;
  int num_cells = 0;
  float radius = 0.0f;
  const NeighborSearchData *source = nullptr;
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
  const IntBuffer *shared_starts = nullptr;
  const IntBuffer *shared_ends = nullptr;
  IntBuffer counts;
  IntBuffer row_ptr;
  IntBuffer col_idx;
  std::vector<SortPointKey> scratch_point_keys;
  std::vector<uint64_t> scratch_unique_keys;
  IntBuffer scratch_cached_neighbor_cells;
  IntBuffer scratch_cached_neighbor_counts;
};

// Builds blocks lazily for a cross-set target when the source is dense.
void ensure_point_blocks(NeighborSearchData &data);

// Builds only the cell grid, sorted order, unique cell keys, and hash table.
// Same-set cell ranges and neighbor rows are left empty. This is enough for a
// point set that is only used as a cross-search target.
void build_point_grid_inplace(NeighborSearchData &data,
                              const std::vector<float> &x,
                              const std::vector<float> &y,
                              const std::vector<float> &z, float radius,
                              float cell_radius = 0.0f);

// Builds the same-set neighbor-cell ranges using the grid produced by
// build_point_grid_inplace().
void build_same_set_cell_ranges_inplace(NeighborSearchData &data);

// Dense-grid variant that fuses cell collection and range merging.
void build_same_set_cell_ranges_dense_fused_inplace(NeighborSearchData &data);

// Builds only the same-set cell ranges and per-query counts.
void count_same_set_neighbors_inplace(NeighborSearchData &data);

// Writes same-set rows using per-query block offsets/capacities.
void write_same_set_block_rows_parallel(NeighborSearchData &data, int capacity,
                                        IntBuffer &counts,
                                        IntBuffer &neighbors,
                                        const IntBuffer &row_offset,
                                        const IntBuffer &row_capacity);

// Builds the same-set cell ranges, counts, row_ptr, and col_idx using the grid
// produced by build_point_grid_inplace().
void build_same_set_neighbor_lists_inplace(NeighborSearchData &data,
                                           float radius);

// Query-major ISPC writer using packed_store_active2. Rows have capacity + 8
// slots so a full vector store can overshoot the requested capacity.
void write_neighbors_fixed_query_active2_parallel(NeighborSearchData &data,
                                                  int capacity,
                                                  IntBuffer &fixed_neighbors,
                                                  IntBuffer &counts);

void write_neighbors_fixed_query_pair_parallel(NeighborSearchData &data,
                                               int capacity,
                                               IntBuffer &fixed_neighbors,
                                               IntBuffer &counts);

// Block-AABB filtered query-major writer for dense cells.
void write_neighbors_fixed_query_blocked_parallel(NeighborSearchData &data,
                                                  int capacity,
                                                  IntBuffer &fixed_neighbors,
                                                  IntBuffer &counts);


void build_neighbor_search_data_inplace(NeighborSearchData &data,
                                        const std::vector<float> &x,
                                        const std::vector<float> &y,
                                        const std::vector<float> &z,
                                        float radius);

NeighborSearchData build_neighbor_search_data(const std::vector<float> &x,
                                              const std::vector<float> &y,
                                              const std::vector<float> &z,
                                              float radius);

CrossNeighborSearchData build_cross_neighbor_search_data(
    const std::vector<float> &query_x, const std::vector<float> &query_y,
    const std::vector<float> &query_z, const NeighborSearchData &target_data,
    float radius);

// Builds cross-set neighbor-cell ranges while reusing the already sorted
// source point set.
void build_cross_cell_ranges_from_sorted_inplace(
    CrossNeighborSearchData &data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, float radius);

// Builds cross-set neighbor lists while reusing the already sorted source
// point set. source_data and target_data must have been built with the same
// radius. This avoids re-sorting the source set for every target set.
void build_cross_neighbor_search_data_from_sorted_inplace(
    CrossNeighborSearchData &data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, float radius);

// Query-major cross writer using packed_store_active2. Rows have capacity + 8
// slots so a full vector store can overshoot the requested capacity.
void write_cross_neighbors_fixed_query_active2_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts);

void write_cross_neighbors_fixed_query_pair_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts);

// Sparse-target cross writer: iterate target points, scan the dense source
// cells around each, and scatter source-row entries. Used when the target has
// far fewer points than cells, where a source-major scan wastes vector lanes.
void write_cross_neighbors_target_major_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, int capacity,
    IntBuffer &fixed_neighbors, IntBuffer &counts);

// Same-set writer that builds the merged cell runs on the fly from the
// set's own dense cell map, skipping the per-frame global range build.
void write_neighbors_fixed_query_pair_dense_parallel(
    NeighborSearchData &data, int capacity,
    IntBuffer &fixed_neighbors, IntBuffer &counts);

// Cross writer that builds the merged target-cell runs on the fly from the
// target's dense cell map, skipping the per-frame global range build.
void write_cross_neighbors_fixed_query_pair_dense_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, int capacity,
    IntBuffer &fixed_neighbors, IntBuffer &counts);

// Cross-set block-AABB filtered writer. target_data must have blocks.
void write_cross_neighbors_fixed_query_blocked_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    const NeighborSearchData &source_data, int capacity,
    IntBuffer &fixed_neighbors, IntBuffer &counts);


void write_neighbors_range(NeighborSearchData &data, int start_cell,
                           int end_cell);

void write_neighbors_parallel(NeighborSearchData &data);

void write_cross_neighbors_range(CrossNeighborSearchData &query_data,
                                 const NeighborSearchData &target_data,
                                 int start_cell, int end_cell);

void write_cross_neighbors_parallel(CrossNeighborSearchData &query_data,
                                    const NeighborSearchData &target_data);

} // namespace apfrnn