#include "plan.hpp"

#include <algorithm>
#include <cstddef>

namespace apfrnn::plan {

namespace {

constexpr int kSameInitialCapacity = 128;
constexpr int kSameMaxCapacity = 32768;
constexpr int kCrossInitialCapacity = 32;
constexpr int kCrossMaxCapacity = 4096;
constexpr int kFixedMaxPoints = 1000000;

int64_t row_bytes(int num_points, int capacity) {
  return static_cast<int64_t>(num_points) *
         static_cast<int64_t>(capacity + 8) *
         static_cast<int64_t>(sizeof(int));
}

// 4-20 points per cell with short rows: the regime where the pair writers,
// which merge cell runs per query cell, beat the generic active2 writer.
bool moderate_density(const NeighborSearchData &data, int last_capacity) {
  return data.cell_stencil == 1 && data.max_cell_points < 256 &&
         last_capacity < 256 && data.num_points >= 4 * data.num_cells &&
         data.num_points < 20 * data.num_cells;
}

int grown_capacity(int max_count, int num_points, int max_capacity,
                   int64_t memory_budget) {
  const int grown = (max_count + 63) & ~63;
  if (max_count > max_capacity ||
      row_bytes(num_points, grown) > memory_budget) {
    return 0;
  }
  return grown;
}

} // namespace

bool wants_small_cells(const NeighborSearchData &data, int last_capacity) {
  return !data.blocks_valid && data.max_cell_points > 0 &&
         data.max_cell_points <= 512 && last_capacity >= 256 &&
         last_capacity > 4 * data.max_cell_points;
}

SamePlan same_set(const NeighborSearchData &data, int last_capacity,
                  int64_t memory_budget) {
  SamePlan plan;
  plan.fixed_rows = data.num_points <= kFixedMaxPoints;

  int first = last_capacity > 0 ? last_capacity : kSameInitialCapacity;
  first = std::min(first, kSameMaxCapacity);
  if (row_bytes(data.num_points, first) > memory_budget) {
    first = kSameInitialCapacity;
  }
  plan.first_capacity = first;

  const bool on_the_fly = data.dense_valid && !data.blocks_valid &&
                          data.max_cell_points > 0 && first < 256 &&
                          moderate_density(data, last_capacity);
  if (on_the_fly) {
    plan.ranges = SameRanges::None;
    plan.writer = SameWriter::PairDense;
    return plan;
  }

  plan.ranges = (!data.blocks_valid && data.dense_valid &&
                 data.cell_stencil == 1)
                    ? SameRanges::DenseFused
                    : SameRanges::Merged;
  if (data.blocks_valid && data.cell_stencil == 1) {
    plan.writer = SameWriter::Blocked;
  } else if (moderate_density(data, last_capacity)) {
    plan.writer = SameWriter::QueryOct;
  } else if (data.num_points <= 1024) {
    plan.writer = SameWriter::Active2;
  } else {
    plan.writer = SameWriter::QueryOct;
  }
  return plan;
}

bool cross_wants_target_blocks(const NeighborSearchData &source,
                               const NeighborSearchData &target) {
  return source.max_cell_points > 512 && target.max_cell_points > 32;
}

CrossPlan cross_set(const NeighborSearchData &source, int source_last_capacity,
                    const NeighborSearchData &target, int pair_last_capacity,
                    int64_t memory_budget) {
  CrossPlan plan;
  plan.fixed_rows = source.num_points <= kFixedMaxPoints;

  int first =
      pair_last_capacity > 0 ? pair_last_capacity : kCrossInitialCapacity;
  first = std::clamp(first, kCrossInitialCapacity, kCrossMaxCapacity);
  if (row_bytes(source.num_points, first) > memory_budget) {
    first = kCrossInitialCapacity;
  }
  plan.first_capacity = first;

  const bool moderate = moderate_density(source, source_last_capacity);
  const bool sparse_target =
      target.num_points > 0 && static_cast<int64_t>(target.num_points) <
                                   2 * static_cast<int64_t>(target.num_cells);
  if (moderate && target.dense_valid && source.dense_valid && sparse_target) {
    plan.build_ranges = false;
    plan.writer = CrossWriter::TargetMajor;
  } else if (moderate && target.dense_valid) {
    plan.build_ranges = false;
    plan.writer = CrossWriter::PairDense;
  } else if (moderate) {
    plan.writer = CrossWriter::QueryPair;
  } else if (target.blocks_valid && source.blocks_valid &&
             source.cell_stencil == 1) {
    plan.writer = CrossWriter::Blocked;
  } else {
    plan.writer = CrossWriter::Active2;
  }
  return plan;
}

int64_t block_output_budget(int64_t exact_neighbors, int num_points) {
  return exact_neighbors * static_cast<int64_t>(sizeof(int)) +
         static_cast<int64_t>(num_points) * 32 +
         (static_cast<int64_t>(64) << 20);
}

int same_grown_capacity(int max_count, int num_points, int64_t memory_budget) {
  return grown_capacity(max_count, num_points, kSameMaxCapacity, memory_budget);
}

int cross_grown_capacity(int max_count, int num_points,
                         int64_t memory_budget) {
  return grown_capacity(max_count, num_points, kCrossMaxCapacity,
                        memory_budget);
}

int same_fitted_capacity(int max_count, int capacity) {
  // Low-degree rows already have rounding slack and dominate memory, so keep
  // them exact. Higher-degree rows keep a margin: a moving set whose maximum
  // degree fluctuates would otherwise regrow and rewrite every row on
  // alternate frames.
  const int margin =
      max_count <= 64 ? 0 : std::max(8, max_count / 16);
  return std::min(capacity, std::max(32, (max_count + margin + 7) & ~7));
}

int cross_fitted_capacity(int max_count, int capacity) {
  const int margin = std::max(8, max_count / 16);
  return std::min(capacity, std::max(32, (max_count + margin + 7) & ~7));
}

bool same_set_isolated(const NeighborSearchData &data) {
  // Ranges may live in a target's shared range cache instead of the data's
  // own buffers.
  const IntBuffer &starts = data.shared_starts != nullptr
                                ? *data.shared_starts
                                : data.cell_neighbor_starts;
  const IntBuffer &ends = data.shared_ends != nullptr ? *data.shared_ends
                                                      : data.cell_neighbor_ends;
  for (int cell = 0; cell < data.num_cells; ++cell) {
    const std::size_t c = static_cast<std::size_t>(cell);
    const int start = data.cell_starts[c];
    const int end = data.cell_ends[c];
    const std::size_t offset =
        static_cast<std::size_t>(data.cell_neighbor_offset[c]);
    if (end - start != 1 || data.cell_num_neighbors[c] != 1 ||
        starts[offset] != start || ends[offset] != end) {
      return false;
    }
  }
  return true;
}

bool cross_has_no_candidates(const CrossNeighborSearchData &cross) {
  for (int cell = 0; cell < cross.num_cells; ++cell) {
    if (cross.cell_num_neighbors[static_cast<std::size_t>(cell)] != 0) {
      return false;
    }
  }
  return true;
}

} // namespace apfrnn::plan
