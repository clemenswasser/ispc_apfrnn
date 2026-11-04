// Exact CSR rows: the count pass that sizes row_ptr and the write passes.

#include "common.isph"

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

// Copies a staged row to its exact CSR slot. col_idx is 64-byte aligned (at
// least one gang of ints up to 16 lanes), so align the row start to the gang
// with a short scalar prefix and stream the rest.
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
// scatter. Each row is staged in scratch (capacity + programCount ints) and
// copied out with vector stores; the row base is formed in 64-bit because a
// CSR offset past 2^29 entries would otherwise sign-extend a 32-bit byte
// offset.
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
          const uniform int added =
              store_compacted(out, hit, tpts.index[j], perm);
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
          const uniform int added0 = store_compacted(o0, hit0, value, perm);
          o0 += added0;
          c0 += added0;
          if (has_q1) {
            const bool hit1 =
                length_sq(pj - p1) <= r2 && (!same_set || j != q1);
            const uniform int added1 =
                store_compacted(o1, hit1, value, perm);
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
