// Parameter structs shared with C++ (the generated header defines them in
// namespace ispc). Kernels take them by pointer and copy them into locals
// on entry, so the fields stay in registers instead of being reloaded after
// every store through an output pointer.

// Sorted point coordinates and their original (caller) indices.
struct Points {
  const uniform float *uniform x;
  const uniform float *uniform y;
  const uniform float *uniform z;
  const uniform int *uniform index;
};

// Occupied cells: points [starts[c], ends[c]) of the sorted order, and the
// packed cell coordinate keys[c].
struct Cells {
  const uniform int *uniform starts;
  const uniform int *uniform ends;
  const uniform uint64 *uniform keys;
};

// Dense cell lookup: cells[(z * ny + y) * nx + x] is a cell index or -1,
// with x/y/z relative to (min_x, min_y, min_z).
struct DenseGrid {
  const uniform int *uniform cells;
  uniform int nx, ny, nz;
  uniform int min_x, min_y, min_z;
};

// Open-addressing hash from cell key to cell index.
struct HashGrid {
  const uniform uint64 *uniform keys;
  const uniform int *uniform vals;
  uniform uint64 empty_key;
  uniform uint32 mask;
};

// Candidate point ranges of each query cell:
// [starts[offset[c] + k], ends[offset[c] + k]) for k < count[c].
struct CellRanges {
  const uniform int *uniform offset;
  const uniform int *uniform count;
  const uniform int *uniform starts;
  const uniform int *uniform ends;
};

// Candidate neighbor cells of each cell: cells[c * stride + k], k < counts[c].
struct NeighborCells {
  const uniform int *uniform cells;
  const uniform int *uniform counts;
  uniform int stride;
};

// x/y/z coordinate arrays.
struct Coords {
  const uniform float *uniform x;
  const uniform float *uniform y;
  const uniform float *uniform z;
};

// Point blocks of dense cells with their bounding boxes [lo, hi], plus
// per-cell bounding boxes [cell_lo, cell_hi]. Cell c owns blocks
// [cell_first[c], + cell_count[c]).
struct Blocks {
  const uniform int *uniform cell_first;
  const uniform int *uniform cell_count;
  const uniform int *uniform begin;
  const uniform int *uniform end;
  const uniform int *uniform cell;
  Coords lo, hi;
  Coords cell_lo, cell_hi;
};

// Fixed-capacity output rows: row q starts at neighbors + q * (capacity + 8);
// the 8 slack slots absorb full-vector compaction stores. perm is the
// compaction lookup table, scratch one row of per-thread staging (only used
// by the _nt writers).
struct FixedRows {
  uniform int capacity;
  const uniform int *uniform perm;
  uniform int *uniform scratch;
  uniform int *uniform counts;
  uniform int *uniform neighbors;
  // Optional per-query block layout. When set, row q starts at
  // neighbors[row_offset[q]] and has row_capacity[q] slots.
  const uniform int *uniform row_offset;
  const uniform int *uniform row_capacity;
};

inline varying uint64 make_key(varying int cx, varying int cy,
                                    varying int cz) {
  varying uint64 ux = ((varying uint64)((varying uint32)cx)) & 0x1FFFFFULL;
  varying uint64 uy = ((varying uint64)((varying uint32)cy)) & 0x1FFFFFULL;
  varying uint64 uz = ((varying uint64)((varying uint32)cz)) & 0x1FFFFFULL;
  return (ux << 42) | (uy << 21) | uz;
}

inline uniform int unpack_coord_uniform(uniform uint64 key,
                                             uniform int shift) {
  uniform int coord = (uniform int)((key >> shift) & 0x1FFFFFULL);
  if ((coord & 0x100000) != 0) {
    coord -= 0x200000;
  }
  return coord;
}

inline varying int unpack_coord(varying uint64 key,
                                     uniform int coord_shift) {
  varying int coord =
      (varying int)((varying uint32)((key >> coord_shift) & 0x1FFFFFULL));
  varying bool is_negative = (coord & 0x100000) != 0;
  coord = select(is_negative, (varying int)(coord | 0xFFE00000), coord);
  return coord;
}

inline varying int find_cell_index(const uniform HashGrid &hash,
                                   varying uint64 key) {
  varying uint64 h64 = (key ^ (key >> 33)) * 0xff51afd7ed558ccdULL;
  h64 ^= h64 >> 33;
  varying uint32 h = (varying uint32)h64 & hash.mask;

  while (true) {
    varying uint64 ht_key = hash.keys[h];
    if (ht_key == key) {
      return hash.vals[h];
    }
    if (ht_key == hash.empty_key) {
      return -1;
    }
    h = (h + 1) & hash.mask;
  }
}

export void collect_neighbor_cells(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform HashGrid *uniform hash_in, uniform int stencil,
    uniform int stride, uniform int out_cells[], uniform int out_counts[]) {
  const uniform Cells queries = *queries_in;
  const uniform HashGrid hash = *hash_in;
  foreach (cell = start_cell... end_cell) {
    varying uint64 key = queries.keys[cell];
    varying int cx = unpack_coord(key, 42);
    varying int cy = unpack_coord(key, 21);
    varying int cz = unpack_coord(key, 0);
    varying int count = 0;

    for (uniform int dx = -stencil; dx <= stencil; ++dx) {
      for (uniform int dy = -stencil; dy <= stencil; ++dy) {
        for (uniform int dz = -stencil; dz <= stencil; ++dz) {
          varying uint64 neighbor_key =
              make_key(cx + dx, cy + dy, cz + dz);
          varying int neighbor_cell = find_cell_index(hash, neighbor_key);

          varying bool is_present = neighbor_cell != -1;
          if (is_present) {
            varying int write_index = cell * stride + count;
            out_cells[write_index] = neighbor_cell;
          }
          count += select(is_present, 1, 0);
        }
      }
    }

    out_counts[cell] = count;
  }
}

export void collect_neighbor_cells_dense(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform DenseGrid *uniform grid_in, uniform int stencil,
    uniform int stride, uniform int out_cells[], uniform int out_counts[]) {
  const uniform Cells queries = *queries_in;
  const uniform DenseGrid grid = *grid_in;
  foreach (cell = start_cell... end_cell) {
    varying uint64 key = queries.keys[cell];
    varying int cx = unpack_coord(key, 42);
    varying int cy = unpack_coord(key, 21);
    varying int cz = unpack_coord(key, 0);
    varying int count = 0;

    for (uniform int dx = -stencil; dx <= stencil; ++dx) {
      for (uniform int dy = -stencil; dy <= stencil; ++dy) {
        for (uniform int dz = -stencil; dz <= stencil; ++dz) {
          varying int lx = cx + dx - grid.min_x;
          varying int ly = cy + dy - grid.min_y;
          varying int lz = cz + dz - grid.min_z;
          varying bool in_bounds =
              (lx >= 0) && (lx < grid.nx) && (ly >= 0) && (ly < grid.ny) &&
              (lz >= 0) && (lz < grid.nz);

          if (in_bounds) {
            varying int dense_index = (lz * grid.ny + ly) * grid.nx + lx;
            varying int neighbor_cell = grid.cells[dense_index];
            if (neighbor_cell != -1) {
              varying int write_index = cell * stride + count;
              out_cells[write_index] = neighbor_cell;
              count += 1;
            }
          }
        }
      }
    }

    out_counts[cell] = count;
  }
}

// Candidate range merging. Each query cell's candidate cells are merged into
// runs of contiguous target points. Every kernel has a count mode (offsets ==
// NULL: writes the number of runs to out_counts) and a write mode (writes the
// runs to out_starts/out_ends from offsets[cell] on).

// Appends the point range [start, end) of the next candidate cell: it extends
// the current run when contiguous, else closes the current run.
static inline void merge_range(varying int start, varying int end,
                               varying int &run_start, varying int &run_end,
                               varying int &runs, varying int &slot,
                               uniform bool write, uniform int out_starts[],
                               uniform int out_ends[]) {
  if (run_end == start) {
    run_end = end;
    return;
  }
  if (run_start != -1) {
    if (write) {
      out_starts[slot] = run_start;
      out_ends[slot] = run_end;
      ++slot;
    }
    ++runs;
  }
  run_start = start;
  run_end = end;
}

static inline void close_runs(varying int cell, varying int run_start,
                              varying int run_end, varying int runs,
                              varying int slot, uniform bool write,
                              uniform int out_counts[], uniform int out_starts[],
                              uniform int out_ends[]) {
  if (run_start != -1) {
    if (write) {
      out_starts[slot] = run_start;
      out_ends[slot] = run_end;
    }
    ++runs;
  }
  if (!write) {
    out_counts[cell] = runs;
  }
}

// Candidate cells come from the cached neighbor-cell lists.
export void merge_neighbor_ranges(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform targets_in,
    const uniform NeighborCells *uniform ncells_in, uniform int out_counts[],
    uniform const int offsets[], uniform int out_starts[],
    uniform int out_ends[]) {
  const uniform Cells targets = *targets_in;
  const uniform NeighborCells ncells = *ncells_in;
  const uniform bool write = offsets != NULL;
  foreach (cell = start_cell... end_cell) {
    const varying int count = ncells.counts[cell];
    varying int run_start = -1, run_end = -1, runs = 0;
    varying int slot = write ? offsets[cell] : 0;
    for (uniform int i = 0; i < ncells.stride; ++i) {
      if (i < count) {
        const varying int neighbor = ncells.cells[cell * ncells.stride + i];
        merge_range(targets.starts[neighbor], targets.ends[neighbor],
                    run_start, run_end, runs, slot, write, out_starts,
                    out_ends);
      }
    }
    close_runs(cell, run_start, run_end, runs, slot, write, out_counts,
               out_starts, out_ends);
  }
}

// Candidate cells are the 27 neighbors of each query cell in a dense grid.
export void merge_dense_neighbor_ranges(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform DenseGrid *uniform grid_in,
    const uniform Cells *uniform targets_in, uniform int out_counts[],
    uniform const int offsets[], uniform int out_starts[],
    uniform int out_ends[]) {
  const uniform Cells queries = *queries_in;
  const uniform DenseGrid grid = *grid_in;
  const uniform Cells targets = *targets_in;
  const uniform bool write = offsets != NULL;
  foreach (cell = start_cell... end_cell) {
    const varying uint64 key = queries.keys[cell];
    const varying int cx = unpack_coord(key, 42);
    const varying int cy = unpack_coord(key, 21);
    const varying int cz = unpack_coord(key, 0);
    varying int run_start = -1, run_end = -1, runs = 0;
    varying int slot = write ? offsets[cell] : 0;
    for (uniform int dx = -1; dx <= 1; ++dx) {
      for (uniform int dy = -1; dy <= 1; ++dy) {
        for (uniform int dz = -1; dz <= 1; ++dz) {
          const varying int lx = cx + dx - grid.min_x;
          const varying int ly = cy + dy - grid.min_y;
          const varying int lz = cz + dz - grid.min_z;
          if (lx >= 0 && lx < grid.nx && ly >= 0 && ly < grid.ny && lz >= 0 &&
              lz < grid.nz) {
            const varying int neighbor =
                grid.cells[(lz * grid.ny + ly) * grid.nx + lx];
            if (neighbor != -1) {
              merge_range(targets.starts[neighbor], targets.ends[neighbor],
                          run_start, run_end, runs, slot, write, out_starts,
                          out_ends);
            }
          }
        }
      }
    }
    close_runs(cell, run_start, run_end, runs, slot, write, out_counts,
               out_starts, out_ends);
  }
}

// Fixed 9-run dense stencil: each query cell gets one merged z-run per
// (dx,dy) pair. One pass, 27 dense lookups, no count/scan; the writer sees
// nine runs instead of 27 individual cells or the branchy merged lists.
export void dense_stencil9_ranges(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform DenseGrid *uniform grid_in,
    const uniform Cells *uniform targets_in, uniform int out_starts[],
    uniform int out_ends[]) {
  const uniform Cells queries = *queries_in;
  const uniform DenseGrid grid = *grid_in;
  const uniform Cells targets = *targets_in;
  foreach (cell = start_cell... end_cell) {
    const varying uint64 key = queries.keys[cell];
    const varying int cx = unpack_coord(key, 42);
    const varying int cy = unpack_coord(key, 21);
    const varying int cz = unpack_coord(key, 0);
    const varying int base = cell * 9;
    varying int slot = 0;
    for (uniform int dx = -1; dx <= 1; ++dx) {
      for (uniform int dy = -1; dy <= 1; ++dy) {
        varying int run_start = -1;
        varying int run_end = -1;
        for (uniform int dz = -1; dz <= 1; ++dz) {
          const varying int lx = cx + dx - grid.min_x;
          const varying int ly = cy + dy - grid.min_y;
          const varying int lz = cz + dz - grid.min_z;
          const varying bool inside =
              lx >= 0 && lx < grid.nx && ly >= 0 && ly < grid.ny &&
              lz >= 0 && lz < grid.nz;
          const varying int safe_lx = inside ? lx : 0;
          const varying int safe_ly = inside ? ly : 0;
          const varying int safe_lz = inside ? lz : 0;
          const varying int neighbor =
              inside
                  ? grid.cells[(safe_lz * grid.ny + safe_ly) * grid.nx +
                               safe_lx]
                  : -1;
          const varying bool valid = neighbor >= 0;
          const varying int safe_neighbor = valid ? neighbor : 0;
          const varying int start =
              valid ? targets.starts[safe_neighbor] : run_start;
          const varying int end = valid ? targets.ends[safe_neighbor] : run_end;
          run_start = run_start < 0 ? start : min(run_start, start);
          run_end = run_end < 0 ? end : max(run_end, end);
        }
        out_starts[base + slot] = run_start < 0 ? 0 : run_start;
        out_ends[base + slot] = run_start < 0 ? 0 : run_end;
        ++slot;
      }
    }
  }
}

// Neighbor-row writers. Every writer handles both same-set and cross-set
// searches: with `same_set`, queries and targets are the same point set and
// a query never lists itself.
//
// Fixed rows: row q starts at rows.neighbors + q * (capacity + 8). Writers
// count every neighbor but stop storing once a row is past capacity, so the
// caller can detect the overflow and retry with the exact maximum. When
// rows.scratch is set, rows are staged there and copied out with streaming
// stores, which avoids write-allocate traffic on large outputs.

// Points as short vectors. The arrays stay SoA; these only gather one
// point's x, y and z.
static inline uniform float<3> point_at(const uniform Points &pts,
                                        uniform int i) {
  const uniform float<3> p = {pts.x[i], pts.y[i], pts.z[i]};
  return p;
}
static inline varying float<3> point_at(const uniform Points &pts,
                                        varying int i) {
  const varying float<3> p = {pts.x[i], pts.y[i], pts.z[i]};
  return p;
}
static inline uniform float<3> coords_at(const uniform Coords &c,
                                         uniform int i) {
  const uniform float<3> p = {c.x[i], c.y[i], c.z[i]};
  return p;
}
static inline uniform float length_sq(uniform float<3> d) {
  return d.x * d.x + d.y * d.y + d.z * d.z;
}
static inline varying float length_sq(varying float<3> d) {
  return d.x * d.x + d.y * d.y + d.z * d.z;
}

static inline uniform int *uniform row_of(const uniform FixedRows &rows,
                                          uniform int query) {
  if (rows.row_offset != NULL) {
    return rows.neighbors + (uniform int64)rows.row_offset[query];
  }
  return rows.neighbors + (uniform int64)query * (rows.capacity + 8);
}

// Appends the active lanes of `value`: compacts them with the lookup table
// and stores a full vector at `out`, which the 8 slack slots absorb. Returns
// how far `out` advances (0 once the row is past capacity).
static inline uniform int append_active(uniform int *uniform out,
                                        uniform int &count, varying bool active,
                                        varying int value,
                                        const uniform FixedRows &rows,
                                        uniform int capacity) {
  if (count > capacity) {
    count += reduce_add(select(active, 1, 0));
    return 0;
  }
  const uniform int mask = packmask(active);
  out[programIndex] = shuffle(value, rows.perm[mask * 8 + programIndex]);
  const uniform int added = popcnt(mask);
  count += added;
  return added;
}

static inline uniform int *uniform row_output(const uniform FixedRows &rows,
                                              uniform int query) {
  return rows.scratch != NULL ? rows.scratch : row_of(rows, query);
}

// Stores the count and, when staged, streams the row to its destination.
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
    const uniform int cap0 = rows.row_capacity != NULL ? rows.row_capacity[q0] : rows.capacity;
    const uniform int cap1 = rows.row_capacity != NULL ? rows.row_capacity[q1] : rows.capacity;
    const uniform int cap2 = rows.row_capacity != NULL ? rows.row_capacity[q2] : rows.capacity;
    const uniform int cap3 = rows.row_capacity != NULL ? rows.row_capacity[q3] : rows.capacity;
    const uniform int cap4 = rows.row_capacity != NULL ? rows.row_capacity[q4] : rows.capacity;
    const uniform int cap5 = rows.row_capacity != NULL ? rows.row_capacity[q5] : rows.capacity;
    const uniform int cap6 = rows.row_capacity != NULL ? rows.row_capacity[q6] : rows.capacity;
    const uniform int cap7 = rows.row_capacity != NULL ? rows.row_capacity[q7] : rows.capacity;
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

// Collects the 27 neighbor cells of cell (cx, cy, cz) as at most 9 runs of
// contiguous target points (adjacent cells along z are contiguous).
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
  uniform int run_start[9];
  uniform int run_end[9];
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    const uniform int num_runs =
        dense_runs(grid, targets, queries.keys[cell], run_start, run_end);
    const uniform int q_start = queries.starts[cell];
    const uniform int q_end = queries.ends[cell];
    if (group == 4 && same_set) {
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

// CSR count pass: queries are vectorized and targets are visited one at a
// time. This is faster than the target-vectorized write loop when runs are
// short, and it is the predicate the row_ptr layout is built from.
export void csr_count_rows(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform CellRanges *uniform ranges_in,
    const uniform Points *uniform qpts_in,
    const uniform Points *uniform tpts_in, uniform bool same_set,
    uniform float r2, uniform int counts[]) {
  const uniform Cells queries = *queries_in;
  const uniform CellRanges ranges = *ranges_in;
  const uniform Points qpts = *qpts_in;
  const uniform Points tpts = *tpts_in;
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    const uniform int first = ranges.offset[cell];
    const uniform int num_runs = ranges.count[cell];
    foreach (i = queries.starts[cell]... queries.ends[cell]) {
      const float<3> q = point_at(qpts, i);
      int count = 0;
      for (uniform int run = 0; run < num_runs; ++run) {
        const uniform int run_end = ranges.ends[first + run];
        for (uniform int j = ranges.starts[first + run]; j < run_end; ++j) {
          const bool hit =
              length_sq(point_at(tpts, j) - q) <= r2 && (!same_set || i != j);
          if (hit) {
            ++count;
          }
        }
      }
      counts[i] = count;
    }
  }
}

// Copies a staged row to its exact CSR slot. col_idx is 32-byte aligned, so
// align the row start with a short scalar prefix and stream the rest.
static inline void copy_csr_row(uniform const int row_ptr[],
                                uniform int col_idx[],
                                uniform const int scratch[], uniform int query,
                                uniform int count) {
  const uniform int row_start = row_ptr[query];
  const uniform int row_len = row_ptr[query + 1] - row_start;
  const uniform int n = min(count, row_len);
  uniform int *uniform row = col_idx + (int64)row_start;
  const uniform int prefix =
      (programCount - (row_start & (programCount - 1))) & (programCount - 1);
  const uniform int prefix_end = min(prefix, n);
  for (uniform int k = 0; k < prefix_end; ++k) {
    row[k] = scratch[k];
  }
  uniform int k = prefix_end;
  for (; k + programCount <= n; k += programCount) {
    streaming_store(row + k, scratch[k + programIndex]);
  }
  for (; k < n; ++k) {
    row[k] = scratch[k];
  }
}

// CSR write pass: writes each query's neighbors from col_idx[row_ptr[q]] on.
// Queries are visited one at a time and targets are vectorized, so the active
// targets can be compacted with packmask/shuffle instead of a per-lane
// scatter. Each row is staged in scratch (capacity + 8 ints) and copied out
// with vector stores; the row base is formed in 64-bit because a CSR offset
// past 2^29 entries would otherwise sign-extend a 32-bit byte offset.
export void csr_rows(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform CellRanges *uniform ranges_in,
    const uniform Points *uniform qpts_in,
    const uniform Points *uniform tpts_in, uniform bool same_set,
    uniform float r2, uniform const int row_ptr[], uniform int col_idx[],
    uniform int scratch[], uniform const int perm[]) {
  const uniform Cells queries = *queries_in;
  const uniform CellRanges ranges = *ranges_in;
  const uniform Points qpts = *qpts_in;
  const uniform Points tpts = *tpts_in;
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    const uniform int first = ranges.offset[cell];
    const uniform int num_runs = ranges.count[cell];
    for (uniform int query = queries.starts[cell];
         query < queries.ends[cell]; ++query) {
      const uniform float<3> q = point_at(qpts, query);
      uniform int count = 0;
      uniform int *uniform out = scratch;
      for (uniform int run = 0; run < num_runs; ++run) {
        const uniform int run_end = ranges.ends[first + run];
        foreach (j = ranges.starts[first + run]... run_end) {
          // Compute the predicate once: the count and write passes must agree
          // exactly, or the write can run past the counted row.
          const bool hit =
              length_sq(point_at(tpts, j) - q) <= r2 && (!same_set || j != query);
          const uniform int mask = packmask(hit);
          out[programIndex] =
              shuffle(tpts.index[j], perm[mask * 8 + programIndex]);
          const uniform int added = popcnt(mask);
          out += added;
          count += added;
        }
      }
      copy_csr_row(row_ptr, col_idx, scratch, query, count);
    }
  }
  memory_barrier();
}

// Paired CSR write pass: two queries share each target load. Each row is
// staged in its own scratch row; an odd last query is paired with itself and
// the duplicate row is skipped.
export void csr_rows_pair(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform CellRanges *uniform ranges_in,
    const uniform Points *uniform qpts_in,
    const uniform Points *uniform tpts_in, uniform bool same_set,
    uniform float r2, uniform const int row_ptr[], uniform int col_idx[],
    uniform int scratch[], uniform int scratch_stride,
    uniform const int perm[]) {
  const uniform Cells queries = *queries_in;
  const uniform CellRanges ranges = *ranges_in;
  const uniform Points qpts = *qpts_in;
  const uniform Points tpts = *tpts_in;
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    const uniform int first = ranges.offset[cell];
    const uniform int num_runs = ranges.count[cell];
    for (uniform int q0 = queries.starts[cell]; q0 < queries.ends[cell];
         q0 += 2) {
      const uniform bool has_q1 = q0 + 1 < queries.ends[cell];
      const uniform int q1 = has_q1 ? q0 + 1 : q0;
      const uniform float<3> p0 = point_at(qpts, q0);
      const uniform float<3> p1 = point_at(qpts, q1);
      uniform int c0 = 0;
      uniform int c1 = 0;
      uniform int *uniform o0 = scratch;
      uniform int *uniform o1 = scratch + scratch_stride;
      for (uniform int run = 0; run < num_runs; ++run) {
        const uniform int run_end = ranges.ends[first + run];
        foreach (j = ranges.starts[first + run]... run_end) {
          const varying float<3> pj = point_at(tpts, j);
          const varying int value = tpts.index[j];
          const bool hit0 =
              length_sq(pj - p0) <= r2 && (!same_set || j != q0);
          const uniform int mask0 = packmask(hit0);
          o0[programIndex] =
              shuffle(value, perm[mask0 * 8 + programIndex]);
          const uniform int added0 = popcnt(mask0);
          o0 += added0;
          c0 += added0;
          if (has_q1) {
            const bool hit1 =
                length_sq(pj - p1) <= r2 && (!same_set || j != q1);
            const uniform int mask1 = packmask(hit1);
            o1[programIndex] =
                shuffle(value, perm[mask1 * 8 + programIndex]);
            const uniform int added1 = popcnt(mask1);
            o1 += added1;
            c1 += added1;
          }
        }
      }
      copy_csr_row(row_ptr, col_idx, scratch, q0, c0);
      if (has_q1) {
        copy_csr_row(row_ptr, col_idx, scratch + scratch_stride, q1, c1);
      }
    }
  }
  memory_barrier();
}

// Coordinate sync: interleaved caller xyz -> SoA x/y/z for points
// [begin, end), plus the largest squared displacement against the previous
// x/y/z (only when `comparable`) and the bounding box.
// out = {max_d2, min_x, min_y, min_z, max_x, max_y, max_z}.

struct SyncStats {
  float max_d2;
  float<3> lo, hi;
};

static inline float<3> min3(float<3> a, float<3> b) {
  const float<3> m = {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
  return m;
}
static inline float<3> max3(float<3> a, float<3> b) {
  const float<3> m = {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
  return m;
}

static inline void init_sync_stats(varying SyncStats &s) {
  const float inf = floatbits(0x7f800000);
  s.max_d2 = 0.0f;
  s.lo = inf;
  s.hi = -inf;
}

static inline void sync_point(float<3> p, int i, uniform bool comparable,
                              uniform float x[], uniform float y[],
                              uniform float z[], varying SyncStats &s) {
  if (comparable) {
    const float<3> old = {x[i], y[i], z[i]};
    s.max_d2 = max(s.max_d2, length_sq(p - old));
  }
  x[i] = p.x;
  y[i] = p.y;
  z[i] = p.z;
  s.lo = min3(s.lo, p);
  s.hi = max3(s.hi, p);
}

static inline void store_sync_stats(const varying SyncStats &s,
                                    uniform float out[7]) {
  out[0] = reduce_max(s.max_d2);
  out[1] = reduce_min(s.lo.x);
  out[2] = reduce_min(s.lo.y);
  out[3] = reduce_min(s.lo.z);
  out[4] = reduce_max(s.hi.x);
  out[5] = reduce_max(s.hi.y);
  out[6] = reduce_max(s.hi.z);
}

export void sync_points_float(uniform const float xyz[],
                                   uniform int begin, uniform int end,
                                   uniform bool comparable, uniform float x[],
                                   uniform float y[], uniform float z[],
                                   uniform float out[7]) {
  varying SyncStats s;
  init_sync_stats(s);
  uniform int base = begin;
  for (; base + programCount <= end; base += programCount) {
    float<3> p;
    // aos_to_soa3 only reads, but its signature takes a non-const pointer.
    aos_to_soa3((uniform float *uniform)&xyz[3 * base], &p.x, &p.y, &p.z);
    sync_point(p, base + programIndex, comparable, x, y, z, s);
  }
  // Tail of fewer than programCount points: gathers are fine here.
  foreach (i = base... end) {
#pragma ignore warning(perf)
    const float<3> p = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
    sync_point(p, i, comparable, x, y, z, s);
  }
  store_sync_stats(s, out);
}

export void sync_points_double(uniform const double xyz[],
                                    uniform int begin, uniform int end,
                                    uniform bool comparable, uniform float x[],
                                    uniform float y[], uniform float z[],
                                    uniform float out[7]) {
  varying SyncStats s;
  init_sync_stats(s);
  uniform int base = begin;
  for (; base + programCount <= end; base += programCount) {
    double<3> d;
    aos_to_soa3((uniform double *uniform)&xyz[3 * base], &d.x, &d.y, &d.z);
    const float<3> p = {(float)d.x, (float)d.y, (float)d.z};
    sync_point(p, base + programIndex, comparable, x, y, z, s);
  }
  foreach (i = base... end) {
#pragma ignore warning(perf)
    const double<3> d = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
    const float<3> p = {(float)d.x, (float)d.y, (float)d.z};
    sync_point(p, i, comparable, x, y, z, s);
  }
  store_sync_stats(s, out);
}
