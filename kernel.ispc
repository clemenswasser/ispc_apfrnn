inline void neighbor_search_core(
    uniform int start_cell, uniform int end_cell,
    uniform const int cell_starts[], uniform const int cell_ends[],
    uniform const int cell_neighbor_offset[],
    uniform const int cell_num_neighbors[],
    uniform const int cell_neighbor_starts[],
    uniform const int cell_neighbor_ends[], uniform float R2,
    uniform const float X[], uniform const float Y[], uniform const float Z[],
    uniform const int orig_idx[], uniform int counts[],
    uniform const int row_ptr[], uniform int col_idx[], uniform bool is_write) {
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    uniform int q_start = cell_starts[cell];
    uniform int q_end = cell_ends[cell];
    uniform int neighbor_offset = cell_neighbor_offset[cell];
    uniform int num_neighbors = cell_num_neighbors[cell];

    foreach (i = q_start... q_end) {
      float qx = X[i], qy = Y[i], qz = Z[i];
      int local_count = 0;
      int write_offset = 0;
      if (is_write) {
        write_offset = row_ptr[i];
      }

      for (uniform int n = 0; n < num_neighbors; ++n) {
        uniform int n_start = cell_neighbor_starts[neighbor_offset + n];
        uniform int n_end = cell_neighbor_ends[neighbor_offset + n];

        for (uniform int j = n_start; j < n_end; ++j) {
          uniform float xj = X[j];
          uniform float yj = Y[j];
          uniform float zj = Z[j];
          float dx = xj - qx;
          float dy = yj - qy;
          float dz = zj - qz;
          float dist2 = dx * dx + dy * dy + dz * dz;

          if ((i != j) && (dist2 <= R2)) {
            if (is_write) {
              uniform int o_idx = orig_idx[j];
              col_idx[write_offset++] = o_idx;
            } else {
              ++local_count;
            }
          }
        }
      }

      if (!is_write) {
        counts[i] = local_count;
      }
    }
  }
}

export void count_neighbors_ispc(
    uniform int start_cell, uniform int end_cell,
    uniform const int cell_starts[], uniform const int cell_ends[],
    uniform const int cell_neighbor_offset[],
    uniform const int cell_num_neighbors[],
    uniform const int cell_neighbor_starts[],
    uniform const int cell_neighbor_ends[], uniform float R2,
    uniform const float X[], uniform const float Y[], uniform const float Z[],
    uniform const int orig_idx[], uniform int counts[]) {
  neighbor_search_core(start_cell, end_cell, cell_starts, cell_ends,
                       cell_neighbor_offset, cell_num_neighbors,
                       cell_neighbor_starts, cell_neighbor_ends, R2, X, Y, Z,
                       orig_idx, counts, NULL, NULL, false);
}

export void
write_neighbors_ispc(uniform int start_cell, uniform int end_cell,
                     uniform const int cell_starts[],
                     uniform const int cell_ends[],
                     uniform const int cell_neighbor_offset[],
                     uniform const int cell_num_neighbors[],
                     uniform const int cell_neighbor_starts[],
                     uniform const int cell_neighbor_ends[], uniform float R2,
                     uniform const float X[], uniform const float Y[],
                     uniform const float Z[], uniform const int orig_idx[],
                     uniform const int row_ptr[], uniform int col_idx[]) {
  neighbor_search_core(start_cell, end_cell, cell_starts, cell_ends,
                       cell_neighbor_offset, cell_num_neighbors,
                       cell_neighbor_starts, cell_neighbor_ends, R2, X, Y, Z,
                       orig_idx, NULL, row_ptr, col_idx, true);
}
