#pragma once

// Strategy selection for the CPU grid backend. Each function only inspects
// grid statistics and the capacity that fit last frame; NSearch executes the
// returned plan. Keeping the heuristics here makes them testable in isolation.

#include "apfrnn.hpp"

#include <cstdint>

namespace apfrnn::plan {

// How the same-set candidate cell ranges are built before writing rows.
enum class SameRanges {
  None,        // the writer merges the 27-cell runs on the fly
  DenseFused,  // build_same_set_cell_ranges_dense_fused_inplace
  Merged,      // build_same_set_cell_ranges_inplace
};

enum class SameWriter {
  PairDense,  // write_neighbors_fixed_query_pair_dense_parallel
  Blocked,    // write_neighbors_fixed_query_blocked_parallel
  QueryPair,  // write_neighbors_fixed_query_pair_parallel
  Active2,    // write_neighbors_fixed_query_active2_parallel
};

struct SamePlan {
  bool fixed_rows = false;  // false: go straight to CSR
  int first_capacity = 0;
  SameRanges ranges = SameRanges::Merged;
  SameWriter writer = SameWriter::Active2;
};

enum class CrossWriter {
  TargetMajor,  // write_cross_neighbors_target_major_parallel
  PairDense,    // write_cross_neighbors_fixed_query_pair_dense_parallel
  QueryPair,    // write_cross_neighbors_fixed_query_pair_parallel
  Blocked,      // write_cross_neighbors_fixed_query_blocked_parallel
  Active2,      // write_cross_neighbors_fixed_query_active2_parallel
};

struct CrossPlan {
  bool fixed_rows = false;  // false: go straight to CSR
  int first_capacity = 0;
  bool build_ranges = true;  // build_cross_cell_ranges_from_sorted_inplace
  CrossWriter writer = CrossWriter::Active2;
};

// Half-radius cells pay off for sets whose rows are much longer than the
// densest cell. Only used without a skin.
bool wants_small_cells(const NeighborSearchData &data, int last_capacity);

SamePlan same_set(const NeighborSearchData &data, int last_capacity,
                  int64_t memory_budget);

// Dense sources against non-trivial targets use target point blocks. Must be
// applied before cross_set(), which looks at target.blocks_valid.
bool cross_wants_target_blocks(const NeighborSearchData &source,
                               const NeighborSearchData &target);

CrossPlan cross_set(const NeighborSearchData &source, int source_last_capacity,
                    const NeighborSearchData &target, int pair_last_capacity,
                    int64_t memory_budget);

// Capacity for the retry after a fixed-row overflow, or 0 when the rows would
// exceed the limit or the memory budget.
int same_grown_capacity(int max_count, int num_points, int64_t memory_budget);
int cross_grown_capacity(int max_count, int num_points, int64_t memory_budget);

// Capacity to start from next frame, given the observed maximum degree.
int same_fitted_capacity(int max_count, int capacity);
int cross_fitted_capacity(int max_count, int capacity);

// True when every point is alone in its cell with no occupied neighbor cell,
// so all rows are empty. Requires same-set cell ranges.
bool same_set_isolated(const NeighborSearchData &data);

// True when no source cell has a candidate target cell. Requires cross ranges.
bool cross_has_no_candidates(const CrossNeighborSearchData &cross);

} // namespace apfrnn::plan
