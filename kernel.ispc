inline varying uint64 make_key_ispc(varying int cx, varying int cy,
                                    varying int cz) {
  varying uint64 ux = ((varying uint64)((varying uint32)cx)) & 0x1FFFFFULL;
  varying uint64 uy = ((varying uint64)((varying uint32)cy)) & 0x1FFFFFULL;
  varying uint64 uz = ((varying uint64)((varying uint32)cz)) & 0x1FFFFFULL;
  return (ux << 42) | (uy << 21) | uz;
}

inline varying int unpack_coord_ispc(varying uint64 key,
                                     uniform int coord_shift) {
  varying int coord =
      (varying int)((varying uint32)((key >> coord_shift) & 0x1FFFFFULL));
  varying bool is_negative = (coord & 0x100000) != 0;
  coord = select(is_negative, (varying int)(coord | 0xFFE00000), coord);
  return coord;
}

inline varying int find_cell_index_ispc(uniform const uint64 ht_keys[],
                                        uniform const int ht_vals[],
                                        uniform uint64 empty_key,
                                        uniform uint32 hash_mask,
                                        varying uint64 key) {
  varying uint64 h64 = (key ^ (key >> 33)) * 0xff51afd7ed558ccdULL;
  h64 ^= h64 >> 33;
  varying uint32 h = (varying uint32)h64 & hash_mask;

  while (true) {
    varying uint64 ht_key = ht_keys[h];
    if (ht_key == key) {
      return ht_vals[h];
    }
    if (ht_key == empty_key) {
      return -1;
    }
    h = (h + 1) & hash_mask;
  }
}

export void collect_neighbor_cells_ispc(
    uniform int start_cell, uniform int end_cell,
    uniform const uint64 unique_keys[], uniform const uint64 ht_keys[],
    uniform const int ht_vals[], uniform uint64 empty_key,
    uniform uint32 hash_mask, uniform int scratch_cached_neighbor_cells[],
    uniform int scratch_cached_neighbor_counts[]) {
  foreach (cell = start_cell... end_cell) {
    varying uint64 key = unique_keys[cell];
    varying int cx = unpack_coord_ispc(key, 42);
    varying int cy = unpack_coord_ispc(key, 21);
    varying int cz = unpack_coord_ispc(key, 0);
    varying int count = 0;

    for (uniform int dx = -1; dx <= 1; ++dx) {
      for (uniform int dy = -1; dy <= 1; ++dy) {
        for (uniform int dz = -1; dz <= 1; ++dz) {
          varying uint64 neighbor_key =
              make_key_ispc(cx + dx, cy + dy, cz + dz);
          varying int neighbor_cell = find_cell_index_ispc(
              ht_keys, ht_vals, empty_key, hash_mask, neighbor_key);

          varying bool is_present = neighbor_cell != -1;
          if (is_present) {
            varying int write_index = cell * 27 + count;
            scratch_cached_neighbor_cells[write_index] = neighbor_cell;
          }
          count += select(is_present, 1, 0);
        }
      }
    }

    scratch_cached_neighbor_counts[cell] = count;
  }
}

export void
count_merged_neighbor_ranges_ispc(uniform int start_cell, uniform int end_cell,
                                  uniform const int cell_starts[],
                                  uniform const int cell_ends[],
                                  uniform const int scratch_neighbor_cells[],
                                  uniform const int scratch_neighbor_counts[],
                                  uniform int cell_num_neighbors[]) {
  foreach (cell = start_cell... end_cell) {
    varying int count = scratch_neighbor_counts[cell];
    varying int merged = 0;
    varying int current_end = -1;

    for (uniform int i = 0; i < 27; ++i) {
      if (i < count) {
        varying int neighbor_idx = scratch_neighbor_cells[cell * 27 + i];
        varying int neighbor_start = cell_starts[neighbor_idx];
        varying int neighbor_end = cell_ends[neighbor_idx];

        if (current_end == neighbor_start) {
          current_end = neighbor_end;
        } else {
          merged += select(current_end != -1, 1, 0);
          current_end = neighbor_end;
        }
      }
    }

    merged += select(current_end != -1, 1, 0);
    cell_num_neighbors[cell] = merged;
  }
}

export void write_merged_neighbor_ranges_ispc(
    uniform int start_cell, uniform int end_cell,
    uniform const int cell_starts[], uniform const int cell_ends[],
    uniform const int scratch_neighbor_cells[],
    uniform const int scratch_neighbor_counts[],
    uniform const int cell_neighbor_offsets[],
    uniform int global_neighbor_starts[], uniform int global_neighbor_ends[]) {
  foreach (cell = start_cell... end_cell) {
    varying int count = scratch_neighbor_counts[cell];
    varying int offset = cell_neighbor_offsets[cell];
    varying int current_start = -1;
    varying int current_end = -1;

    for (uniform int i = 0; i < 27; ++i) {
      if (i < count) {
        varying int neighbor_idx = scratch_neighbor_cells[cell * 27 + i];
        varying int neighbor_start = cell_starts[neighbor_idx];
        varying int neighbor_end = cell_ends[neighbor_idx];

        if (current_end == neighbor_start) {
          current_end = neighbor_end;
        } else {
          if (current_start != -1) {
            global_neighbor_starts[offset] = current_start;
            global_neighbor_ends[offset] = current_end;
            ++offset;
          }

          current_start = neighbor_start;
          current_end = neighbor_end;
        }
      }
    }

    if (current_start != -1) {
      global_neighbor_starts[offset] = current_start;
      global_neighbor_ends[offset] = current_end;
    }
  }
}

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

#pragma unroll 4
        for (uniform int j = n_start; j < n_end; ++j) {
          uniform float xj = X[j];
          uniform float yj = Y[j];
          uniform float zj = Z[j];
          float dx = xj - qx;
          float dy = yj - qy;
          float dz = zj - qz;
          float dist2 = dx * dx + dy * dy + dz * dz;

          if (dist2 <= R2) {
            if (is_write) {
              if (i != j) {
                uniform int o_idx = orig_idx[j];
                col_idx[write_offset++] = o_idx;
              }
            } else {
              ++local_count;
            }
          }
        }
      }

      if (!is_write) {
        counts[i] = local_count - 1;
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

inline void cross_neighbor_search_core(
    uniform int start_cell, uniform int end_cell,
    uniform const int query_cell_starts[], uniform const int query_cell_ends[],
    uniform const int query_cell_neighbor_offset[],
    uniform const int query_cell_num_neighbors[],
    uniform const int query_cell_neighbor_starts[],
    uniform const int query_cell_neighbor_ends[], uniform float R2,
    uniform const float query_x[], uniform const float query_y[],
    uniform const float query_z[], uniform const float target_x[],
    uniform const float target_y[], uniform const float target_z[],
    uniform const int target_orig_idx[], uniform int counts[],
    uniform const int row_ptr[], uniform int col_idx[], uniform bool is_write) {
  for (uniform int cell = start_cell; cell < end_cell; ++cell) {
    uniform int q_start = query_cell_starts[cell];
    uniform int q_end = query_cell_ends[cell];
    uniform int neighbor_offset = query_cell_neighbor_offset[cell];
    uniform int num_neighbors = query_cell_num_neighbors[cell];

    foreach (i = q_start... q_end) {
      float qx = query_x[i], qy = query_y[i], qz = query_z[i];
      int local_count = 0;
      int write_offset = 0;
      if (is_write) {
        write_offset = row_ptr[i];
      }

      for (uniform int n = 0; n < num_neighbors; ++n) {
        uniform int n_start = query_cell_neighbor_starts[neighbor_offset + n];
        uniform int n_end = query_cell_neighbor_ends[neighbor_offset + n];

#pragma unroll 4
        for (uniform int j = n_start; j < n_end; ++j) {
          uniform float xj = target_x[j];
          uniform float yj = target_y[j];
          uniform float zj = target_z[j];
          float dx = xj - qx;
          float dy = yj - qy;
          float dz = zj - qz;
          float dist2 = dx * dx + dy * dy + dz * dz;

          if (dist2 <= R2) {
            if (is_write) {
              col_idx[write_offset++] = target_orig_idx[j];
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

export void count_cross_neighbors_ispc(
    uniform int start_cell, uniform int end_cell,
    uniform const int query_cell_starts[], uniform const int query_cell_ends[],
    uniform const int query_cell_neighbor_offset[],
    uniform const int query_cell_num_neighbors[],
    uniform const int query_cell_neighbor_starts[],
    uniform const int query_cell_neighbor_ends[], uniform float R2,
    uniform const float query_x[], uniform const float query_y[],
    uniform const float query_z[], uniform const float target_x[],
    uniform const float target_y[], uniform const float target_z[],
    uniform int counts[]) {
  cross_neighbor_search_core(
      start_cell, end_cell, query_cell_starts, query_cell_ends,
      query_cell_neighbor_offset, query_cell_num_neighbors,
      query_cell_neighbor_starts, query_cell_neighbor_ends, R2, query_x,
      query_y, query_z, target_x, target_y, target_z, NULL, counts, NULL, NULL,
      false);
}

export void write_cross_neighbors_ispc(
    uniform int start_cell, uniform int end_cell,
    uniform const int query_cell_starts[], uniform const int query_cell_ends[],
    uniform const int query_cell_neighbor_offset[],
    uniform const int query_cell_num_neighbors[],
    uniform const int query_cell_neighbor_starts[],
    uniform const int query_cell_neighbor_ends[], uniform float R2,
    uniform const float query_x[], uniform const float query_y[],
    uniform const float query_z[], uniform const float target_x[],
    uniform const float target_y[], uniform const float target_z[],
    uniform const int target_orig_idx[], uniform const int row_ptr[],
    uniform int col_idx[]) {
  cross_neighbor_search_core(
      start_cell, end_cell, query_cell_starts, query_cell_ends,
      query_cell_neighbor_offset, query_cell_num_neighbors,
      query_cell_neighbor_starts, query_cell_neighbor_ends, R2, query_x,
      query_y, query_z, target_x, target_y, target_z, target_orig_idx, NULL,
      row_ptr, col_idx, true);
}
