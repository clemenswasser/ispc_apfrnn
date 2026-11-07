// Candidate cells and point ranges of each query cell: the 27-cell
// neighbor lists and their merged runs, and the one-pass stencil runs of the
// dense and hash grids.

#include "common.isph"

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

// Appends point run [start, end) to a cell's runs, extending the open run
// when they touch; the open run is flushed to out_* when they do not.
static inline void append_run(varying int start, varying int end,
                              varying int base, varying int &runs,
                              varying int &run_start, varying int &run_end,
                              uniform int out_starts[], uniform int out_ends[]) {
  if (start == run_end) {
    run_end = end;
    return;
  }
  if (run_start >= 0) {
    out_starts[base + runs] = run_start;
    out_ends[base + runs] = run_end;
    ++runs;
  }
  run_start = start;
  run_end = end;
}

// Stencil runs on the hash grid, in one pass. Cells are sorted by key (x,
// then y, then z), so the occupied cells of one (x, y) column with z in
// [cz - 1, cz + 1] are adjacent in the cell arrays and their points form one
// run. The first such cell takes a hash probe; each next z is the array
// successor if its key matches (no probe). Only the step from z = -1 to
// z = 0 needs a probe: keys hold z as an unsigned 21-bit field, so -1 sorts
// after every z >= 0 of its column. That case can split a column into two
// runs, so each cell has HASH_STENCIL_RUNS slots at cell * HASH_STENCIL_RUNS;
// out_counts gets the runs used. Touching runs (end == start) are merged,
// also across columns. 9 probes per cell in the common case instead of the
// 27 of collect_neighbor_cells, and no count/scan/merge passes.
#define HASH_STENCIL_RUNS 18
#define HASH_STENCIL_RUNS_CROSS 32
export uniform int hash_stencil_runs_per_cell() { return HASH_STENCIL_RUNS; }
export uniform int hash_stencil_runs_per_cell_cross() {
  return HASH_STENCIL_RUNS_CROSS;
}

export void hash_stencil_ranges(
    uniform int start_cell, uniform int end_cell,
    const uniform Cells *uniform queries_in,
    const uniform HashGrid *uniform hash_in,
    const uniform Cells *uniform targets_in, uniform int num_target_cells,
    uniform int out_offsets[], uniform int out_counts[],
    uniform int out_starts[], uniform int out_ends[]) {
  const uniform Cells queries = *queries_in;
  const uniform HashGrid hash = *hash_in;
  const uniform Cells targets = *targets_in;
  foreach (cell = start_cell... end_cell) {
    const varying uint64 key = queries.keys[cell];
    const varying int cx = unpack_coord(key, 42);
    const varying int cy = unpack_coord(key, 21);
    const varying int cz = unpack_coord(key, 0);
    const varying int base = cell * HASH_STENCIL_RUNS;
    varying int runs = 0;
    varying int run_start = -1;
    varying int run_end = -1;
    for (uniform int dx = -1; dx <= 1; ++dx) {
      for (uniform int dy = -1; dy <= 1; ++dy) {
        // Consecutive cells [first, last] of this column; their points are
        // [starts[first], ends[last]), so only the run's ends are gathered.
        varying int first = -1;
        varying int last = -1;
        for (uniform int dz = -1; dz <= 1; ++dz) {
          const varying int z = cz + dz;
          const varying uint64 want = make_key(cx + dx, cy + dy, z);
          varying int found = -1;
          if (last >= 0 && z != 0) {
            const varying int next = last + 1;
            if (next < num_target_cells && targets.keys[next] == want) {
              found = next;
            }
          } else {
            found = find_cell_index(hash, want);
          }
          if (found >= 0) {
            if (first >= 0 && found != last + 1) {
              // Only across z = -1 -> 0 in raw key order: not adjacent.
              append_run(targets.starts[first], targets.ends[last], base,
                         runs, run_start, run_end, out_starts, out_ends);
              first = found;
            } else if (first < 0) {
              first = found;
            }
            last = found;
          }
        }
        if (first >= 0) {
          append_run(targets.starts[first], targets.ends[last], base, runs,
                     run_start, run_end, out_starts, out_ends);
        }
      }
    }
    if (run_start >= 0) {
      out_starts[base + runs] = run_start;
      out_ends[base + runs] = run_end;
      ++runs;
    }
    out_offsets[cell] = base;
    out_counts[cell] = runs;
  }
}

// Cross-set variant of hash_stencil_ranges: query keys come from the source
// grid and target cells/hash from a different point set. Same 9-probe,
// one-pass run construction; avoids collect_neighbor_cells' 27 probes plus the
// count/scan/merge passes for hash-grid cross pairs.
export void hash_stencil_ranges_cross(
    uniform int start_cell, uniform int end_cell,
    const uniform uint64 *uniform query_keys,
    const uniform HashGrid *uniform hash_in,
    const uniform Cells *uniform targets_in, uniform int num_target_cells,
    uniform int out_offsets[], uniform int out_counts[],
    uniform int out_starts[], uniform int out_ends[]) {
  const uniform HashGrid hash = *hash_in;
  const uniform Cells targets = *targets_in;
  foreach (cell = start_cell... end_cell) {
    const varying uint64 key = query_keys[cell];
    const varying int cx = unpack_coord(key, 42);
    const varying int cy = unpack_coord(key, 21);
    const varying int cz = unpack_coord(key, 0);
    const varying int base = cell * HASH_STENCIL_RUNS_CROSS;
    varying int runs = 0;
    varying int run_start = -1;
    varying int run_end = -1;
    for (uniform int dx = -1; dx <= 1; ++dx) {
      for (uniform int dy = -1; dy <= 1; ++dy) {
        varying int first = -1;
        varying int last = -1;
        for (uniform int dz = -1; dz <= 1; ++dz) {
          const varying int z = cz + dz;
          const varying uint64 want = make_key(cx + dx, cy + dy, z);
          varying int found = -1;
          if (last >= 0 && z != 0) {
            const varying int next = last + 1;
            if (next < num_target_cells && targets.keys[next] == want) {
              found = next;
            } else {
              // The array successor is not always the wanted z (e.g. z=-1
              // sorts after every non-negative z), so fall back to the hash.
              found = find_cell_index(hash, want);
            }
          } else {
            found = find_cell_index(hash, want);
          }
          if (found >= 0) {
            if (first >= 0 && found != last + 1) {
              append_run(targets.starts[first], targets.ends[last], base,
                         runs, run_start, run_end, out_starts, out_ends);
              first = found;
            } else if (first < 0) {
              first = found;
            }
            last = found;
          }
        }
        if (first >= 0) {
          append_run(targets.starts[first], targets.ends[last], base, runs,
                     run_start, run_end, out_starts, out_ends);
        }
      }
    }
    if (run_start >= 0) {
      out_starts[base + runs] = run_start;
      out_ends[base + runs] = run_end;
      ++runs;
    }
    out_offsets[cell] = base;
    out_counts[cell] = runs;
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
