inline void
neighbor_search_core(uniform int start_query, uniform int end_query,
                     uniform const int point_cell_idx[],
                     uniform const int cell_neighbor_offset[],
                     uniform const int cell_num_neighbors[],
                     uniform const int cell_neighbor_starts[],
                     uniform const int cell_neighbor_ends[], uniform float R2,
                     uniform const float X[], uniform const float Y[],
                     uniform const float Z[], uniform const int orig_idx[],
                     uniform int counts[], uniform const int row_ptr[],
                     uniform int col_idx[], uniform bool is_write) {
  for (uniform int i_base = start_query; i_base < end_query;
       i_base += programCount) {
    uniform int limit = min(i_base + programCount, end_query);
    uniform int c_first = point_cell_idx[i_base];
    uniform int c_last = point_cell_idx[limit - 1];

    if (c_first == c_last) {
      uniform int cell = c_first;
      uniform int neighbor_offset = cell_neighbor_offset[cell];
      uniform int num_neighbors = cell_num_neighbors[cell];

      foreach (i = i_base... limit) {
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
            float dx = X[j] - qx;
            float dy = Y[j] - qy;
            float dz = Z[j] - qz;
            float dist2 = dx * dx + dy * dy + dz * dz;

            if ((i != j) && (dist2 <= R2)) {
              if (is_write) {
                col_idx[write_offset++] = orig_idx[j];
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
    } else {
      foreach (i = i_base... limit) {
        int cell = point_cell_idx[i];
        int neighbor_offset = cell_neighbor_offset[cell];
        int num_neighbors = cell_num_neighbors[cell];
        float qx = X[i], qy = Y[i], qz = Z[i];
        int local_count = 0;
        int write_offset = 0;
        if (is_write) {
          write_offset = row_ptr[i];
        }

        for (int n = 0; n < num_neighbors; ++n) {
          int n_start = cell_neighbor_starts[neighbor_offset + n];
          int n_end = cell_neighbor_ends[neighbor_offset + n];

          for (int j = n_start; j < n_end; ++j) {
            float dx = X[j] - qx;
            float dy = Y[j] - qy;
            float dz = Z[j] - qz;
            float dist2 = dx * dx + dy * dy + dz * dz;

            if ((i != j) && (dist2 <= R2)) {
              if (is_write) {
                col_idx[write_offset++] = orig_idx[j];
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
}

export void count_neighbors_ispc(uniform int start_query, uniform int end_query,
                                 uniform const int point_cell_idx[],
                                 uniform const int cell_neighbor_offset[],
                                 uniform const int cell_num_neighbors[],
                                 uniform const int cell_neighbor_starts[],
                                 uniform const int cell_neighbor_ends[],
                                 uniform float R2, uniform const float X[],
                                 uniform const float Y[],
                                 uniform const float Z[],
                                 uniform const int orig_idx[],
                                 uniform int counts[]) {
  neighbor_search_core(start_query, end_query, point_cell_idx,
                       cell_neighbor_offset, cell_num_neighbors,
                       cell_neighbor_starts, cell_neighbor_ends, R2, X, Y, Z,
                       orig_idx, counts, NULL, NULL, false);
}

export void
write_neighbors_ispc(uniform int start_query, uniform int end_query,
                     uniform const int point_cell_idx[],
                     uniform const int cell_neighbor_offset[],
                     uniform const int cell_num_neighbors[],
                     uniform const int cell_neighbor_starts[],
                     uniform const int cell_neighbor_ends[], uniform float R2,
                     uniform const float X[], uniform const float Y[],
                     uniform const float Z[], uniform const int orig_idx[],
                     uniform const int row_ptr[], uniform int col_idx[]) {
  neighbor_search_core(start_query, end_query, point_cell_idx,
                       cell_neighbor_offset, cell_num_neighbors,
                       cell_neighbor_starts, cell_neighbor_ends, R2, X, Y, Z,
                       orig_idx, NULL, row_ptr, col_idx, true);
}
