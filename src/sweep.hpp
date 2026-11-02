#pragma once

// Sweep backend for extremely sparse point sets, where a cell grid would be
// mostly empty. Points are sorted by x once; each query scans the window
// [x - r, x + r] of the target order.

#include "apfrnn.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace apfrnn::sweep {

// Point indices sorted by x, plus the sorted x values.
struct Index {
  std::vector<float> x;
  IntBuffer order;
  bool valid = false;

  void clear();
};

// A point set as seen by the sweep: original-order coordinates and an
// optional x-sorted index. Queries without a valid index are visited in
// original order with one binary search each.
struct Cloud {
  const std::vector<float> &x;
  const std::vector<float> &y;
  const std::vector<float> &z;
  const Index *index;
};

// Grid cells per point when the bounding box is covered with cells of edge
// `radius`. Large values mean a grid would be almost entirely empty.
double estimated_cells_per_point(const Bounds &bounds, std::size_t num_points,
                                 float radius);

bool should_use(const Bounds &bounds, std::size_t num_points, float radius);

void ensure_index(Index &index, const std::vector<float> &x);

// Writes fixed-stride neighbor rows for every query point. Row
// `query_rows[point]` holds the target indices within `radius`; with
// `same_set` the query itself is skipped. The stride is the maximum count.
// Returns false and leaves the outputs untouched when the rows would exceed
// the capacity limit or `memory_budget` bytes.
bool write_fixed_rows(const Cloud &query, const IntBuffer &query_rows,
                      const Cloud &target, bool same_set, float radius,
                      int64_t memory_budget, IntBuffer &neighbors,
                      IntBuffer &counts, int &row_stride);

} // namespace apfrnn::sweep
