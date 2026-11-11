// Fixed-capacity neighbor rows: the same-set and cross-set row writers.

#include "common.isph"

// Nothing below may assume a particular gang width: the kernels are built
// for whatever target ISPC is given (4-, 8- and 16-wide are tested, see
// scripts/ci.sh). The host side sizes row slack and alignment from
// gang_width().
export uniform int gang_width() { return programCount; }

// Neighbor-row writers. Every writer handles both same-set and cross-set
// searches: with `same_set`, queries and targets are the same point set and
// a query never lists itself.
//
// Fixed rows: row q starts at rows.neighbors + q * (capacity + programCount).
// Writers
// count every neighbor but stop storing once a row is past capacity, so the
// caller can detect the overflow and retry with the exact maximum. When
// rows.scratch is set, rows are staged there and copied out with streaming
// stores, which avoids write-allocate traffic on large outputs.

static inline uniform int *uniform row_of(const uniform FixedRows &rows,
                                          uniform int query) {
  if (rows.row_offset != NULL) {
    return rows.neighbors + (uniform int64)rows.row_offset[query];
  }
  const uniform int slot = rows.row_slot != NULL ? rows.row_slot[query] : query;
  return rows.neighbors + (uniform int64)slot * (rows.capacity + programCount);
}

// Appends the active lanes of `value` at `out` (see store_compacted; the
// row's slack absorbs the full-gang store). Returns how far `out` advances
// (0 once the row is past capacity).
static inline uniform int append_active(uniform int *uniform out,
                                        uniform int &count, varying bool active,
                                        varying int value,
                                        const uniform FixedRows &rows,
                                        uniform int capacity) {
  if (count > capacity) {
    count += reduce_add(select(active, 1, 0));
    return 0;
  }
  const uniform int added = store_compacted(out, active, value, rows.perm);
  count += added;
  return added;
}

static inline uniform int *uniform row_output(const uniform FixedRows &rows,
                                              uniform int query) {
  return rows.scratch != NULL ? rows.scratch : row_of(rows, query);
}

// Stores the count and, when staged, streams the row to its destination.
// Row starts are gang-aligned (the host rounds capacities to row_align()), so
// the aligned streaming stores are legal.
static inline void finish_row(const uniform FixedRows &rows,
                              uniform int query, uniform int count) {
  if (rows.scratch != NULL) {
    const uniform int capacity =
        rows.row_capacity != NULL ? rows.row_capacity[query] : rows.capacity;
    const uniform int n = min(count, capacity);
    const uniform int vector_end = n & ~(programCount - 1);
    uniform int *uniform dst = row_of(rows, query);
    for (uniform int i = 0; i < vector_end; i += programCount) {
      streaming_store(dst + i, rows.scratch[i + programIndex]);
    }
    for (uniform int i = vector_end; i < n; ++i) {
      dst[i] = rows.scratch[i];
    }
  }
  rows.counts[query] = count;
}

static inline varying bool is_neighbor(const uniform Points &tpts, int j,
                                       uniform float<3> q, uniform float r2,
                                       uniform bool same_set,
                                       uniform int query) {
  return length_sq(point_at(tpts, j) - q) <= r2 && (!same_set || j != query);
}

// Writes the rows of queries [q_start, q_end) against candidate runs
// [run_start[k], run_end[k]), one query at a time.
static inline void rows_single(uniform int q_start, uniform int q_end,
                               const uniform int *uniform run_start,
                               const uniform int *uniform run_end,
                               uniform int num_runs, const uniform Points &qpts,
                               const uniform Points &tpts, uniform bool same_set,
                               uniform float r2, const uniform FixedRows &rows) {
  for (uniform int query = q_start; query < q_end; ++query) {
    const uniform float<3> q = point_at(qpts, query);
    uniform int *uniform out = row_output(rows, query);
    const uniform int capacity =
        rows.row_capacity != NULL ? rows.row_capacity[query] : rows.capacity;
    uniform int count = 0;
    for (uniform int run = 0; run < num_runs; ++run) {
      foreach (j = run_start[run]... run_end[run]) {
        out += append_active(
            out, count, is_neighbor(tpts, j, q, r2, same_set, query),
            tpts.index[j], rows, capacity);
      }
    }
    finish_row(rows, query, count);
  }
}

static inline varying bool within_radius(varying float<3> p,
                                         uniform float<3> q, uniform float r2) {
  return length_sq(p - q) <= r2;
}

// As rows_single, but two queries share each target load. An odd last query
// is paired with itself and its duplicate is skipped.
static inline void rows_pair(uniform int q_start, uniform int q_end,
                             const uniform int *uniform run_start,
                             const uniform int *uniform run_end,
                             uniform int num_runs, const uniform Points &qpts,
                             const uniform Points &tpts, uniform bool same_set,
                             uniform float r2, const uniform FixedRows &rows) {
  for (uniform int q0 = q_start; q0 < q_end; q0 += 2) {
    const uniform bool v1 = q0 + 1 < q_end;
    const uniform int q1 = v1 ? q0 + 1 : q0;
    const uniform float<3> p0 = point_at(qpts, q0), p1 = point_at(qpts, q1);
    uniform int *uniform o0 = row_of(rows, q0);
    uniform int *uniform o1 = row_of(rows, q1);
    const uniform int cap0 =
        rows.row_capacity != NULL ? rows.row_capacity[q0] : rows.capacity;
    const uniform int cap1 =
        rows.row_capacity != NULL ? rows.row_capacity[q1] : rows.capacity;
    uniform int c0 = 0, c1 = 0;
    for (uniform int run = 0; run < num_runs; ++run) {
      foreach (j = run_start[run]... run_end[run]) {
        const float<3> pj = point_at(tpts, j);
        const int value = tpts.index[j];
        const bool a0 = within_radius(pj, p0, r2) &&
                        (!same_set || j != q0);
        const bool a1 = within_radius(pj, p1, r2) &&
                        (!same_set || j != q1);
        o0 += append_active(o0, c0, a0, value, rows, cap0);
        if (v1) {
          o1 += append_active(o1, c1, a1, value, rows, cap1);
        }
      }
    }
    rows.counts[q0] = c0;
    if (v1) {
      rows.counts[q1] = c1;
    }
  }
}

// As rows_pair, with four queries per target load.
static inline void rows_quad(uniform int q_start, uniform int q_end,
                             const uniform int *uniform run_start,
                             const uniform int *uniform run_end,
                             uniform int num_runs, const uniform Points &qpts,
                             const uniform Points &tpts, uniform bool same_set,
                             uniform float r2, const uniform FixedRows &rows) {
  for (uniform int q0 = q_start; q0 < q_end; q0 += 4) {
    const uniform bool v1 = q0 + 1 < q_end;
    const uniform bool v2 = q0 + 2 < q_end;
    const uniform bool v3 = q0 + 3 < q_end;
    const uniform int q1 = v1 ? q0 + 1 : q0;
    const uniform int q2 = v2 ? q0 + 2 : q0;
    const uniform int q3 = v3 ? q0 + 3 : q0;
    const uniform float<3> p0 = point_at(qpts, q0), p1 = point_at(qpts, q1);
    const uniform float<3> p2 = point_at(qpts, q2), p3 = point_at(qpts, q3);
    uniform int *uniform o0 = row_of(rows, q0);
    uniform int *uniform o1 = row_of(rows, q1);
    uniform int *uniform o2 = row_of(rows, q2);
    uniform int *uniform o3 = row_of(rows, q3);
    const uniform int cap0 =
        rows.row_capacity != NULL ? rows.row_capacity[q0] : rows.capacity;
    const uniform int cap1 =
        rows.row_capacity != NULL ? rows.row_capacity[q1] : rows.capacity;
    const uniform int cap2 =
        rows.row_capacity != NULL ? rows.row_capacity[q2] : rows.capacity;
    const uniform int cap3 =
        rows.row_capacity != NULL ? rows.row_capacity[q3] : rows.capacity;
    uniform int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (uniform int run = 0; run < num_runs; ++run) {
      foreach (j = run_start[run]... run_end[run]) {
        const float<3> pj = point_at(tpts, j);
        const int value = tpts.index[j];
        const bool a0 = within_radius(pj, p0, r2) &&
                        (!same_set || j != q0);
        const bool a1 = within_radius(pj, p1, r2) &&
                        (!same_set || j != q1);
        const bool a2 = within_radius(pj, p2, r2) &&
                        (!same_set || j != q2);
        const bool a3 = within_radius(pj, p3, r2) &&
                        (!same_set || j != q3);
        o0 += append_active(o0, c0, a0, value, rows, cap0);
        if (v1) {
          o1 += append_active(o1, c1, a1, value, rows, cap1);
        }
        if (v2) {
          o2 += append_active(o2, c2, a2, value, rows, cap2);
        }
        if (v3) {
          o3 += append_active(o3, c3, a3, value, rows, cap3);
        }
      }
    }
    rows.counts[q0] = c0;
    if (v1) {
      rows.counts[q1] = c1;
    }
    if (v2) {
      rows.counts[q2] = c2;
    }
    if (v3) {
      rows.counts[q3] = c3;
    }
  }
}

// As rows_quad, with eight queries per target load.
static inline void rows_oct(uniform int q_start, uniform int q_end,
                            const uniform int *uniform run_start,
                            const uniform int *uniform run_end,
                            uniform int num_runs, const uniform Points &qpts,
                            const uniform Points &tpts, uniform bool same_set,
                            uniform float r2, const uniform FixedRows &rows) {
  for (uniform int q0 = q_start; q0 < q_end; q0 += 8) {
    const uniform bool v1 = q0 + 1 < q_end;
    const uniform bool v2 = q0 + 2 < q_end;
    const uniform bool v3 = q0 + 3 < q_end;
    const uniform bool v4 = q0 + 4 < q_end;
    const uniform bool v5 = q0 + 5 < q_end;
    const uniform bool v6 = q0 + 6 < q_end;
    const uniform bool v7 = q0 + 7 < q_end;
    const uniform int q1 = v1 ? q0 + 1 : q0;
    const uniform int q2 = v2 ? q0 + 2 : q0;
    const uniform int q3 = v3 ? q0 + 3 : q0;
    const uniform int q4 = v4 ? q0 + 4 : q0;
    const uniform int q5 = v5 ? q0 + 5 : q0;
    const uniform int q6 = v6 ? q0 + 6 : q0;
    const uniform int q7 = v7 ? q0 + 7 : q0;
    const uniform float<3> p0 = point_at(qpts, q0), p1 = point_at(qpts, q1);
    const uniform float<3> p2 = point_at(qpts, q2), p3 = point_at(qpts, q3);
    const uniform float<3> p4 = point_at(qpts, q4), p5 = point_at(qpts, q5);
    const uniform float<3> p6 = point_at(qpts, q6), p7 = point_at(qpts, q7);
    uniform int *uniform o0 = row_of(rows, q0);
    uniform int *uniform o1 = row_of(rows, q1);
    uniform int *uniform o2 = row_of(rows, q2);
    uniform int *uniform o3 = row_of(rows, q3);
    uniform int *uniform o4 = row_of(rows, q4);
    uniform int *uniform o5 = row_of(rows, q5);
    uniform int *uniform o6 = row_of(rows, q6);
    uniform int *uniform o7 = row_of(rows, q7);
    const uniform int cap0 =
        rows.row_capacity != NULL ? rows.row_capacity[q0] : rows.capacity;
    const uniform int cap1 =
        rows.row_capacity != NULL ? rows.row_capacity[q1] : rows.capacity;
    const uniform int cap2 =
        rows.row_capacity != NULL ? rows.row_capacity[q2] : rows.capacity;
    const uniform int cap3 =
        rows.row_capacity != NULL ? rows.row_capacity[q3] : rows.capacity;
    const uniform int cap4 =
        rows.row_capacity != NULL ? rows.row_capacity[q4] : rows.capacity;
    const uniform int cap5 =
        rows.row_capacity != NULL ? rows.row_capacity[q5] : rows.capacity;
    const uniform int cap6 =
        rows.row_capacity != NULL ? rows.row_capacity[q6] : rows.capacity;
    const uniform int cap7 =
        rows.row_capacity != NULL ? rows.row_capacity[q7] : rows.capacity;
    uniform int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    uniform int c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    for (uniform int run = 0; run < num_runs; ++run) {
      foreach (j = run_start[run]... run_end[run]) {
        const float<3> pj = point_at(tpts, j);
        const int value = tpts.index[j];
        const bool a0 = within_radius(pj, p0, r2) && (!same_set || j != q0);
        const bool a1 = within_radius(pj, p1, r2) && (!same_set || j != q1);
        const bool a2 = within_radius(pj, p2, r2) && (!same_set || j != q2);
        const bool a3 = within_radius(pj, p3, r2) && (!same_set || j != q3);
        const bool a4 = within_radius(pj, p4, r2) && (!same_set || j != q4);
        const bool a5 = within_radius(pj, p5, r2) && (!same_set || j != q5);
        const bool a6 = within_radius(pj, p6, r2) && (!same_set || j != q6);
        const bool a7 = within_radius(pj, p7, r2) && (!same_set || j != q7);
        o0 += append_active(o0, c0, a0, value, rows, cap0);
        if (v1) o1 += append_active(o1, c1, a1, value, rows, cap1);
        if (v2) o2 += append_active(o2, c2, a2, value, rows, cap2);
        if (v3) o3 += append_active(o3, c3, a3, value, rows, cap3);
        if (v4) o4 += append_active(o4, c4, a4, value, rows, cap4);
        if (v5) o5 += append_active(o5, c5, a5, value, rows, cap5);
        if (v6) o6 += append_active(o6, c6, a6, value, rows, cap6);
        if (v7) o7 += append_active(o7, c7, a7, value, rows, cap7);
      }
    }
    rows.counts[q0] = c0;
    if (v1) rows.counts[q1] = c1;
    if (v2) rows.counts[q2] = c2;
    if (v3) rows.counts[q3] = c3;
    if (v4) rows.counts[q4] = c4;
    if (v5) rows.counts[q5] = c5;
    if (v6) rows.counts[q6] = c6;
    if (v7) rows.counts[q7] = c7;
  }
}

// Packs cell coordinates exactly like the host make_key().
static inline uniform uint64 make_key_uniform(uniform int cx, uniform int cy,
                                              uniform int cz) {
  uniform uint64 ux = ((uniform uint64)((uniform uint32)cx)) & 0x1FFFFFULL;
  uniform uint64 uy = ((uniform uint64)((uniform uint32)cy)) & 0x1FFFFFULL;
  uniform uint64 uz = ((uniform uint64)((uniform uint32)cz)) & 0x1FFFFFULL;
  return (ux << 42) | (uy << 21) | uz;
}

// Host floor_to_cell(): clamp non-finite and out-of-range coordinates instead
// of invoking undefined behavior in the float-to-int conversion. Clamping only
// adds candidate work; the exact distance test still decides neighbors.
static inline uniform int floor_to_cell_uniform(uniform float value) {
  const uniform int kCellMin = -0x100000;
  const uniform int kCellMax = 0xFFFFF;
  if (value != value) {
    return value < 0.0f ? kCellMin : kCellMax;
  }
  if (value >= (uniform float)kCellMax) {
    return kCellMax;
  }
  if (value <= (uniform float)kCellMin) {
    return kCellMin;
  }
  return (uniform int)floor(value);
}

// Most runs dense_runs() can return: one per neighbor cell.
#define DENSE_MAX_RUNS 27

// Collects the 27 neighbor cells of cell (cx, cy, cz) as runs of contiguous
// target points. Adjacent cells along z merge into one run when the cells are
// sorted by signed coordinate, giving 9 runs. Radix-sorted keys order a
// negative z after every positive one, so a column crossing z = 0 splits and
// up to DENSE_MAX_RUNS runs are returned.
static inline uniform int dense_runs(const uniform DenseGrid &grid,
                                     const uniform Cells &targets,
                                     uniform uint64 key,
                                     uniform int run_start[],
                                     uniform int run_end[]) {
  const uniform int cx = unpack_coord_uniform(key, 42);
  const uniform int cy = unpack_coord_uniform(key, 21);
  const uniform int cz = unpack_coord_uniform(key, 0);
  uniform int num_runs = 0;
  for (uniform int dx = -1; dx <= 1; ++dx) {
    for (uniform int dy = -1; dy <= 1; ++dy) {
      uniform int current_start = -1;
      uniform int current_end = -1;
      for (uniform int dz = -1; dz <= 1; ++dz) {
        const uniform int lx = cx + dx - grid.min_x;
        const uniform int ly = cy + dy - grid.min_y;
        const uniform int lz = cz + dz - grid.min_z;
        if (lx < 0 || lx >= grid.nx || ly < 0 || ly >= grid.ny || lz < 0 ||
            lz >= grid.nz) {
          continue;
        }
        const uniform int neighbor =
            grid.cells[(lz * grid.ny + ly) * grid.nx + lx];
        if (neighbor == -1) {
          continue;
        }
        const uniform int start = targets.starts[neighbor];
        const uniform int end = targets.ends[neighbor];
        if (current_end == start) {
          current_end = end;
        } else {
          if (current_start != -1) {
            run_start[num_runs] = current_start;
            run_end[num_runs] = current_end;
            ++num_runs;
          }
          current_start = start;
          current_end = end;
        }
      }
      if (current_start != -1) {
        run_start[num_runs] = current_start;
        run_end[num_runs] = current_end;
        ++num_runs;
      }
    }
  }
  return num_runs;
}

// Rows of listed queries against a dense target. Each query's cell index is
// given; used for the rows above a hybrid layout's base capacity.
export void write_listed_rows_dense(
    uniform int begin, uniform int end, uniform const int queries[],
    uniform const int query_cells[],
    const uniform Cells *uniform cells_in,
    const uniform DenseGrid *uniform grid_in,
    const uniform Cells *uniform targets_in,
    const uniform Points *uniform qpts_in,
    const uniform Points *uniform tpts_in, uniform bool same_set,
    uniform float r2, const uniform FixedRows *uniform rows_in) {
  const uniform Cells cells = *cells_in;
  const uniform DenseGrid grid = *grid_in;
  const uniform Cells targets = *targets_in;
  const uniform Points qpts = *qpts_in;
  const uniform Points tpts = *tpts_in;
  const uniform FixedRows rows = *rows_in;
  uniform int run_start[DENSE_MAX_RUNS];
  uniform int run_end[DENSE_MAX_RUNS];
  for (uniform int k = begin; k < end; ++k) {
    const uniform int query = queries[k];
    const uniform int cell = query_cells[k];
    const uniform int num_runs =
        dense_runs(grid, targets, cells.keys[cell], run_start, run_end);
    if (same_set) {
      rows_single(query, query + 1, run_start, run_end, num_runs, qpts, tpts,
                  true, r2, rows);
    } else {
      rows_single(query, query + 1, run_start, run_end, num_runs, qpts, tpts,
                  false, r2, rows);
    }
  }
  if (rows.scratch != NULL) {
    memory_barrier();
  }
}

// Candidate runs from precomputed per-cell ranges; one query at a time.
// Handles any row length, with optional staging (rows.scratch).
export void write_fixed_rows(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform CellRanges *uniform ranges_in,
    const uniform Points *uniform qpts_in,
    const uniform Points *uniform tpts_in, uniform bool same_set,
    uniform float r2, const uniform FixedRows *uniform rows_in) {
  const uniform Cells queries = *queries_in;
  const uniform CellRanges ranges = *ranges_in;
  const uniform Points qpts = *qpts_in;
  const uniform Points tpts = *tpts_in;
  const uniform FixedRows rows = *rows_in;
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    const uniform int first = ranges.offset[cell];
    // Literal same_set: each branch is specialized at compile time.
    if (same_set) {
      rows_single(queries.starts[cell], queries.ends[cell],
                  ranges.starts + first, ranges.ends + first,
                  ranges.count[cell], qpts, tpts, true, r2, rows);
    } else {
      rows_single(queries.starts[cell], queries.ends[cell],
                  ranges.starts + first, ranges.ends + first,
                  ranges.count[cell], qpts, tpts, false, r2, rows);
    }
  }
  if (rows.scratch != NULL) {
    memory_barrier();
  }
}

// Rewrites the same-set rows of listed queries [begin, end): queries[k] in
// cell query_cells[k], against that cell's precomputed ranges. Used for the
// few rows of a moving set that overflowed their capacity and were moved to
// new space (row_offset/row_capacity), so the other rows are not rewritten.
export void write_listed_rows(
    uniform int begin, uniform int end, uniform const int queries[],
    uniform const int query_cells[],
    const uniform CellRanges *uniform ranges_in,
    const uniform Points *uniform pts_in, uniform float r2,
    const uniform FixedRows *uniform rows_in) {
  const uniform CellRanges ranges = *ranges_in;
  const uniform Points pts = *pts_in;
  const uniform FixedRows rows = *rows_in;
  for (uniform int k = begin; k < end; ++k) {
    const uniform int query = queries[k];
    const uniform int cell = query_cells[k];
    const uniform int first = ranges.offset[cell];
    rows_single(query, query + 1, ranges.starts + first, ranges.ends + first,
                ranges.count[cell], pts, pts, true, r2, rows);
  }
}

// Candidate runs from precomputed per-cell ranges; two queries share each
// target load. For short rows; no staging.
export void write_fixed_rows_pair(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform CellRanges *uniform ranges_in,
    const uniform Points *uniform qpts_in,
    const uniform Points *uniform tpts_in, uniform bool same_set,
    uniform float r2, const uniform FixedRows *uniform rows_in) {
  const uniform Cells queries = *queries_in;
  const uniform CellRanges ranges = *ranges_in;
  const uniform Points qpts = *qpts_in;
  const uniform Points tpts = *tpts_in;
  const uniform FixedRows rows = *rows_in;
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    const uniform int first = ranges.offset[cell];
    if (same_set) {
      rows_pair(queries.starts[cell], queries.ends[cell],
                ranges.starts + first, ranges.ends + first,
                ranges.count[cell], qpts, tpts, true, r2, rows);
    } else {
      rows_pair(queries.starts[cell], queries.ends[cell],
                ranges.starts + first, ranges.ends + first,
                ranges.count[cell], qpts, tpts, false, r2, rows);
    }
  }
}

// Candidate runs from precomputed per-cell ranges; eight queries share each
// target load.
export void write_fixed_rows_oct(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform CellRanges *uniform ranges_in,
    const uniform Points *uniform qpts_in,
    const uniform Points *uniform tpts_in, uniform bool same_set,
    uniform float r2, const uniform FixedRows *uniform rows_in) {
  const uniform Cells queries = *queries_in;
  const uniform CellRanges ranges = *ranges_in;
  const uniform Points qpts = *qpts_in;
  const uniform Points tpts = *tpts_in;
  const uniform FixedRows rows = *rows_in;
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    const uniform int first = ranges.offset[cell];
    if (same_set) {
      rows_oct(queries.starts[cell], queries.ends[cell],
               ranges.starts + first, ranges.ends + first,
               ranges.count[cell], qpts, tpts, true, r2, rows);
    } else {
      rows_oct(queries.starts[cell], queries.ends[cell],
               ranges.starts + first, ranges.ends + first,
               ranges.count[cell], qpts, tpts, false, r2, rows);
    }
  }
}

// Candidate runs merged on the fly from the target's dense grid; `group`
// (2 or 4) queries share each target load. For short rows; no staging. Each
// branch below passes literals so the helpers are specialized.
export void write_fixed_rows_dense(
    uniform int start_cell, uniform int end_cell, uniform int group,
    const uniform Cells *uniform queries_in,
    const uniform DenseGrid *uniform grid_in,
    const uniform Cells *uniform targets_in,
    const uniform Points *uniform qpts_in,
    const uniform Points *uniform tpts_in, uniform bool same_set,
    uniform float r2, const uniform FixedRows *uniform rows_in) {
  const uniform Cells queries = *queries_in;
  const uniform DenseGrid grid = *grid_in;
  const uniform Cells targets = *targets_in;
  const uniform Points qpts = *qpts_in;
  const uniform Points tpts = *tpts_in;
  const uniform FixedRows rows = *rows_in;
  uniform int run_start[DENSE_MAX_RUNS];
  uniform int run_end[DENSE_MAX_RUNS];
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    const uniform int num_runs =
        dense_runs(grid, targets, queries.keys[cell], run_start, run_end);
    const uniform int q_start = queries.starts[cell];
    const uniform int q_end = queries.ends[cell];
    if (group == 8 && same_set) {
      rows_oct(q_start, q_end, run_start, run_end, num_runs, qpts, tpts,
               true, r2, rows);
    } else if (group == 8) {
      rows_oct(q_start, q_end, run_start, run_end, num_runs, qpts, tpts,
               false, r2, rows);
    } else if (group == 4 && same_set) {
      rows_quad(q_start, q_end, run_start, run_end, num_runs, qpts, tpts,
                true, r2, rows);
    } else if (group == 4) {
      rows_quad(q_start, q_end, run_start, run_end, num_runs, qpts, tpts,
                false, r2, rows);
    } else if (same_set) {
      rows_pair(q_start, q_end, run_start, run_end, num_runs, qpts, tpts,
                true, r2, rows);
    } else {
      rows_pair(q_start, q_end, run_start, run_end, num_runs, qpts, tpts,
                false, r2, rows);
    }
  }
}

static inline uniform float axis_gap(uniform float q, uniform float lo,
                                     uniform float hi) {
  return q < lo ? lo - q : (q > hi ? q - hi : 0.0f);
}

// Two-pass target-major CSR writer for a dense source grid and a sparse
// target set. Pass 0 counts the hits of every source row; pass 1 appends the
// target id at row_ptr[row] + cursor[row]. The entries of a 2M-point emitter
// pair then occupy a few MB instead of touching one fresh 64 B line per entry
// (the fixed-row layout's RFO floor).
// Pass 0 of the recorded-hit CSR writer: count every source row's hits and
// record the hit candidate rows in this target's compact slot when they fit.
// An overflowed target is marked -1 and rescanned by the fill pass.
export void target_major_csr_record(
    uniform int begin, uniform int end,
    const uniform Points *uniform tpts_in,
    const uniform DenseGrid *uniform grid_in,
    const uniform Cells *uniform src_in,
    const uniform Points *uniform spts_in, uniform float inv_radius,
    uniform float r2, uniform int *uniform counts,
    uniform const int slot_offset[], uniform const int slot_capacity[],
    uniform int *uniform slot_counts, uniform int *uniform hit_totals,
    uniform int *uniform slots, uniform const int *uniform perm_in) {
  const uniform Points tpts = *tpts_in;
  const uniform DenseGrid grid = *grid_in;
  const uniform Cells src = *src_in;
  const uniform Points spts = *spts_in;
  const uniform int *uniform perm = perm_in;
  uniform int run_start[DENSE_MAX_RUNS];
  uniform int run_end[DENSE_MAX_RUNS];
  uniform int cached_cx = 0x7FFFFFFF;
  uniform int cached_cy = 0;
  uniform int cached_cz = 0;
  uniform int num_runs = 0;
  for (uniform int t = begin; t < end; ++t) {
    const uniform float<3> q = point_at(tpts, t);
    const uniform int cx = floor_to_cell_uniform(q.x * inv_radius);
    const uniform int cy = floor_to_cell_uniform(q.y * inv_radius);
    const uniform int cz = floor_to_cell_uniform(q.z * inv_radius);
    if (cx != cached_cx || cy != cached_cy || cz != cached_cz) {
      num_runs = dense_runs(grid, src, make_key_uniform(cx, cy, cz),
                            run_start, run_end);
      cached_cx = cx;
      cached_cy = cy;
      cached_cz = cz;
    }
    const uniform int slot_base = slot_offset[t];
    const uniform int slot_end = slot_base + slot_capacity[t];
    uniform int cursor = slot_base;
    uniform int total = 0;
    uniform bool overflow = false;
    for (uniform int run = 0; run < num_runs; ++run) {
      uniform int base = run_start[run];
      foreach (j = run_start[run] ... run_end[run]) {
        const float<3> pj = point_at(spts, j);
        const varying bool hit = length_sq(pj - q) <= r2;
        uniform int mask = packmask(hit);
        total += popcnt(mask);
        if (!overflow) {
          if (cursor + programCount <= slot_end) {
            cursor += store_compacted(slots + cursor, hit, j, perm);
          } else {
            overflow = true;
          }
        }
        while (mask != 0) {
          const uniform int lane = count_trailing_zeros(mask);
          atomic_add_global(&counts[base + lane], 1);
          mask &= mask - 1;
        }
        base += programCount;
      }
    }
    hit_totals[t] = total;
    slot_counts[t] = overflow ? -1 : (cursor - slot_base);
  }
}

// Fill pass: append each target id from its recorded slot, or rescan the
// target when its slot overflowed.
export void target_major_csr_fill(
    uniform int begin, uniform int end,
    const uniform Points *uniform tpts_in,
    const uniform DenseGrid *uniform grid_in,
    const uniform Cells *uniform src_in,
    const uniform Points *uniform spts_in, uniform float inv_radius,
    uniform float r2, uniform int *uniform counts,
    uniform const int row_ptr[], uniform int col_idx[],
    uniform const int slot_offset[], uniform const int slots[],
    uniform const int slot_counts[]) {
  const uniform Points tpts = *tpts_in;
  const uniform DenseGrid grid = *grid_in;
  const uniform Cells src = *src_in;
  const uniform Points spts = *spts_in;
  uniform int run_start[DENSE_MAX_RUNS];
  uniform int run_end[DENSE_MAX_RUNS];
  uniform int cached_cx = 0x7FFFFFFF;
  uniform int cached_cy = 0;
  uniform int cached_cz = 0;
  uniform int num_runs = 0;
  for (uniform int t = begin; t < end; ++t) {
    const uniform int value = tpts.index[t];
    const uniform int recorded = slot_counts[t];
    if (recorded >= 0) {
      const uniform int base = slot_offset[t];
      for (uniform int k = 0; k < recorded; ++k) {
        const uniform int jj = slots[base + k];
        const uniform int position = atomic_add_global(&counts[jj], 1);
        col_idx[row_ptr[jj] + position] = value;
      }
      continue;
    }
    const uniform float<3> q = point_at(tpts, t);
    const uniform int cx = floor_to_cell_uniform(q.x * inv_radius);
    const uniform int cy = floor_to_cell_uniform(q.y * inv_radius);
    const uniform int cz = floor_to_cell_uniform(q.z * inv_radius);
    if (cx != cached_cx || cy != cached_cy || cz != cached_cz) {
      num_runs = dense_runs(grid, src, make_key_uniform(cx, cy, cz),
                            run_start, run_end);
      cached_cx = cx;
      cached_cy = cy;
      cached_cz = cz;
    }
    for (uniform int run = 0; run < num_runs; ++run) {
      uniform int base = run_start[run];
      foreach (j = run_start[run] ... run_end[run]) {
        const float<3> pj = point_at(spts, j);
        const varying bool hit = length_sq(pj - q) <= r2;
        uniform int mask = packmask(hit);
        while (mask != 0) {
          const uniform int lane = count_trailing_zeros(mask);
          const uniform int jj = base + lane;
          const uniform int position = atomic_add_global(&counts[jj], 1);
          col_idx[row_ptr[jj] + position] = value;
          mask &= mask - 1;
        }
        base += programCount;
      }
    }
  }
}

// Squared distance from q to the box [lo, hi]; 0 inside.
static inline uniform float box_distance_sq(uniform float<3> q,
                                            uniform float<3> lo,
                                            uniform float<3> hi) {
  const uniform float<3> gap = {axis_gap(q.x, lo.x, hi.x),
                                axis_gap(q.y, lo.y, hi.y),
                                axis_gap(q.z, lo.z, hi.z)};
  return length_sq(gap);
}

// Rows of the queries in one source block (see write_fixed_rows_blocked).
static inline void rows_blocked(uniform int source_block,
                                const uniform Blocks &src,
                                const uniform NeighborCells &ncells,
                                const uniform Blocks &tgt,
                                const uniform Points &qpts,
                                const uniform Points &tpts, uniform bool same_set,
                                uniform float r2, const uniform FixedRows &rows) {
  const uniform int cell = src.cell[source_block];
  const uniform int num_neighbors = ncells.counts[cell];
  for (uniform int query = src.begin[source_block];
       query < src.end[source_block]; ++query) {
    const uniform float<3> q = point_at(qpts, query);
    uniform int *uniform out = row_output(rows, query);
    const uniform int capacity =
        rows.row_capacity != NULL ? rows.row_capacity[query] : rows.capacity;
    uniform int count = 0;
    for (uniform int n = 0; n < num_neighbors; ++n) {
      const uniform int c = ncells.cells[cell * ncells.stride + n];
      if (box_distance_sq(q, coords_at(tgt.cell_lo, c),
                          coords_at(tgt.cell_hi, c)) > r2) {
        continue;
      }
      const uniform int block_end = tgt.cell_first[c] + tgt.cell_count[c];
      for (uniform int b = tgt.cell_first[c]; b < block_end; ++b) {
        if (box_distance_sq(q, coords_at(tgt.lo, b), coords_at(tgt.hi, b)) >
            r2) {
          continue;
        }
        foreach (j = tgt.begin[b]... tgt.end[b]) {
          out += append_active(
              out, count,
              is_neighbor(tpts, j, q, r2, same_set, query),
              tpts.index[j], rows, capacity);
        }
      }
    }
    finish_row(rows, query, count);
  }
}

// For dense cells: queries are visited per source block, and target cells
// and their blocks are skipped when their bounding box is out of range.
// Every occupied cell has at least one block. Optional staging.
export void write_fixed_rows_blocked(
    uniform int start_block, uniform int end_block,
    const uniform Blocks *uniform src_blocks_in,
    const uniform NeighborCells *uniform ncells_in,
    const uniform Blocks *uniform tgt_blocks_in,
    const uniform Points *uniform qpts_in,
    const uniform Points *uniform tpts_in, uniform bool same_set,
    uniform float r2, const uniform FixedRows *uniform rows_in) {
  const uniform Blocks src = *src_blocks_in;
  const uniform NeighborCells ncells = *ncells_in;
  const uniform Blocks tgt = *tgt_blocks_in;
  const uniform Points qpts = *qpts_in;
  const uniform Points tpts = *tpts_in;
  const uniform FixedRows rows = *rows_in;
  for (uniform int source_block = start_block; source_block < end_block;
       ++source_block) {
    if (same_set) {
      rows_blocked(source_block, src, ncells, tgt, qpts, tpts, true, r2, rows);
    } else {
      rows_blocked(source_block, src, ncells, tgt, qpts, tpts, false, r2,
                   rows);
    }
  }
  if (rows.scratch != NULL) {
    memory_barrier();
  }
}
