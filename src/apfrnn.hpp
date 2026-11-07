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

// 64 bytes: a whole 16-lane gang of ints, so the kernels' aligned streaming
// stores (vmovnt* of one gang) are legal on AVX-512.
template <typename T, std::size_t Alignment = 64> struct DefaultInitAllocator {
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

// Compaction lookup table shared by the row writers (read by 8-wide kernel
// builds only). Internal, but exposed so the CSR regression can call the same
// ISPC writer as the library.
const int *neighbor_permutation_table();

// Lanes of the ISPC gang the kernels run with (programCount of the target
// ISPC built or dispatched to). Nothing on the host may hard-code it.
int gang_width();

// Spare slots after every fixed row and staging row: the compacting writers
// store a whole gang at the current row end.
inline int row_slack() { return gang_width(); }

// Fixed-row capacities, and so row strides (capacity + row_slack()), are
// multiples of this, which keeps every row start gang-aligned for the
// kernels' aligned streaming stores. At least 8, the AVX2 layout.
inline int row_align() { return std::max(8, gang_width()); }

inline int round_up_to_row_align(int n) {
  const int align = row_align();
  return (n + align - 1) / align * align;
}

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

  // Cell keys pack each coordinate as a 21-bit two's-complement field, so
  // raw key order puts negative coordinates after positive ones. Flipping
  // each field's sign bit compares in signed (x, y, z) order, the order of
  // the counting sort: adjacent z cells stay contiguous across z = 0, which
  // the dense stencil kernels rely on.
  static constexpr uint64_t kSignBits =
      (uint64_t{1} << 62) | (uint64_t{1} << 41) | (uint64_t{1} << 20);

  bool operator<(const SortPointKey &other) const {
    if (key != other.key) {
      return (key ^ kSignBits) < (other.key ^ kSignBits);
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
  std::vector<uint8_t> hash_control;
  bool hash_valid = false;
  std::vector<SortPointKey> scratch_point_keys;
  std::vector<SortPointKey> scratch_point_keys_alt;
  std::vector<uint64_t> scratch_unique_keys;
  // True when cell_starts/cell_ends/scratch_unique_keys are ordered by signed
  // cell coordinates (the comparison/counting sorts); false for the radix
  // sort, which orders by the raw packed key.
  bool cell_keys_signed_order = false;
  // Incremental cell-order state for moving sets. `incremental_keys` is the
  // previous cell key per original index, `incremental_sorted_keys` the same
  // keys in the previous sorted order. The remaining vectors are scratch used
  // by the changed/unchanged merge.
  std::vector<uint64_t> incremental_keys;
  std::vector<uint64_t> incremental_sorted_keys;
  std::vector<uint8_t> incremental_changed;
  std::vector<SortPointKey> incremental_changed_keys;
  std::vector<SortPointKey> incremental_unchanged_keys;
  std::vector<float> incremental_x;
  std::vector<float> incremental_y;
  std::vector<float> incremental_z;
  IntBuffer incremental_order;
  IntBuffer incremental_starts;
  IntBuffer incremental_ends;
  std::vector<uint64_t> incremental_unique_keys;
  std::vector<uint64_t> incremental_merge_keys;
  // Cell-x slab index for the log-time cross range search: position of the
  // first cell with x = cell_x_min + i.
  IntBuffer cell_x_start;
  int cell_x_min = 0;
  int cell_range_min_y = 0;
  int cell_range_max_y = 0;
  int cell_range_min_z = 0;
  int cell_range_max_z = 0;
  bool cell_x_valid = false;
  // Occupied-cell coordinate bounds from the last key pass; avoids rescanning
  // every unique key in build_dense_cell_lookup.
  int cell_coord_min_x = 0;
  int cell_coord_max_x = 0;
  int cell_coord_min_y = 0;
  int cell_coord_max_y = 0;
  int cell_coord_min_z = 0;
  int cell_coord_max_z = 0;
  bool cell_coord_valid = false;
  // Coarse occupancy bitmap of the set's points, used to prove that a cross
  // query's search can be skipped. Built lazily and invalidated by a grid
  // rebuild.
  std::vector<uint8_t> occupancy_bits;
  float occupancy_block = 0.0f;
  float occupancy_min_x = 0.0f;
  float occupancy_min_y = 0.0f;
  float occupancy_min_z = 0.0f;
  int occupancy_nx = 0;
  int occupancy_ny = 0;
  int occupancy_nz = 0;
  bool occupancy_valid = false;
  // True when scratch_unique_keys is also the per-sorted-position key array
  // (one point per cell), so the next merge can use it directly.
  bool incremental_keys_are_unique = false;
  int incremental_points = 0;
  float incremental_inv_radius = 0.0f;
  bool incremental_valid = false;
  IntBuffer scratch_cached_neighbor_cells;
  IntBuffer scratch_cached_neighbor_counts;
  std::vector<uint64_t> cached_neighbor_keys;
  int cached_neighbor_stencil = -1;
  int cached_neighbor_stride = 0;
  bool cached_neighbor_valid = false;
  uint64_t neighbor_generation = 0;
  IntBuffer cached_range_cell_starts;
  IntBuffer cached_range_cell_ends;
  int cached_range_stencil = -1;
  int cached_range_stride = 0;
  int cached_range_total = -1;
  uint64_t cached_range_generation = 0;
  bool cached_range_valid = false;
  // Cell keys the hash-grid stencil ranges were built from (see
  // build_hash_stencil_ranges); cached_range_* holds the rest of the check.
  std::vector<uint64_t> hash_range_keys;
  std::vector<uint64_t> dense_stencil_keys;
  IntBuffer dense_stencil_cell_starts;
  IntBuffer dense_stencil_cell_ends;
  int dense_stencil_total = -1;
  bool dense_stencil_cache_valid = false;
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
  // Active pairs that read this set's ranges as their target in a run. The
  // cache costs one stencil per cell of the padded bounding box, the direct
  // ranges one per occupied cell and pair, so it is only built when
  // padded cells <= range_cache_uses * occupied cells.
  int range_cache_uses = 1;
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
  std::vector<SortPointKey> scratch_point_keys_alt;
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

// Rewrites only the same-set rows of `queries` (sorted point indices) after
// write_same_set_block_rows_parallel(), e.g. rows that overflowed and were
// given new space in row_offset/row_capacity.
void write_same_set_listed_rows(NeighborSearchData &data,
                                const IntBuffer &queries, IntBuffer &counts,
                                IntBuffer &neighbors,
                                const IntBuffer &row_offset,
                                const IntBuffer &row_capacity);

// Builds the same-set cell ranges, counts, row_ptr, and col_idx using the grid
// produced by build_point_grid_inplace().
void build_same_set_neighbor_lists_inplace(NeighborSearchData &data,
                                           float radius);

// Query-major ISPC writer using packed_store_active2. Rows have capacity +
// row_slack() slots so a full-gang store can overshoot the capacity.
void write_neighbors_fixed_query_active2_parallel(NeighborSearchData &data,
                                                  int capacity,
                                                  IntBuffer &fixed_neighbors,
                                                  IntBuffer &counts);

void write_neighbors_fixed_query_pair_parallel(NeighborSearchData &data,
                                               int capacity,
                                               IntBuffer &fixed_neighbors,
                                               IntBuffer &counts);

void write_neighbors_fixed_query_oct_parallel(NeighborSearchData &data,
                                              int capacity,
                                              IntBuffer &fixed_neighbors,
                                              IntBuffer &counts);

// Same-set rows for a grid with at most one point per cell. Enumerates the
// 3x3x3 cell stencil by merging the sorted cell keys at constant key offsets,
// avoiding per-cell hash probes and per-cell range tables. Returns false when
// a row exceeded `capacity` (the caller retries with a larger capacity).
bool write_sparse_same_rows(NeighborSearchData &data, int capacity,
                            IntBuffer &fixed_neighbors, IntBuffer &counts);

// Builds the hashed cell lookup if the grid does not have a dense one and the
// hash has not been built yet. Sparse grids defer it: the merged-key same-set
// path and the sweep cross path never read it.
void ensure_hash_table(NeighborSearchData &data);

// Exact cross-set rows for a sparse target with non-negative cell
// coordinates. The target's sorted cell keys are searched as a log-time range
// index (x slab -> y range -> z range) instead of scanning an x-window or
// probing a hash. Returns false when a row exceeded `capacity`.
void ensure_cell_x_index(NeighborSearchData &data);
void ensure_occupancy(NeighborSearchData &data);

bool write_cross_rows_cell_range(NeighborSearchData &source,
                                 NeighborSearchData &target, float radius,
                                 int capacity, IntBuffer &fixed_neighbors,
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

// Query-major cross writer using packed_store_active2. Rows have capacity +
// row_slack() slots so a full-gang store can overshoot the capacity.
void write_cross_neighbors_fixed_query_active2_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts);

void write_cross_neighbors_fixed_query_pair_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts);

// Query-major cross writer with per-query block offsets/capacities. `capacity`
// is the largest block capacity (the staging row size).
void write_cross_neighbors_block_rows_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts,
    const IntBuffer &row_offset, const IntBuffer &row_capacity);

// Paired variants of the block-row writer, matching the fixed-row pair and
// on-the-fly dense-pair kernels.
void write_cross_neighbors_block_rows_pair_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &target_data,
    int capacity, IntBuffer &fixed_neighbors, IntBuffer &counts,
    const IntBuffer &row_offset, const IntBuffer &row_capacity);

void write_cross_neighbors_block_rows_pair_dense_parallel(
    CrossNeighborSearchData &query_data, const NeighborSearchData &source_data,
    const NeighborSearchData &target_data, int capacity,
    IntBuffer &fixed_neighbors, IntBuffer &counts,
    const IntBuffer &row_offset, const IntBuffer &row_capacity);

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