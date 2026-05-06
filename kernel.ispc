inline uint32 hash_coords(uint32 ux, uint32 uy, uint32 uz, uint32 hash_mask) {
  return (ux ^ uy ^ uz) & hash_mask;
}

inline void neighbor_search_core(
    uniform int start_query, uniform int end_query, uniform float R,
    uniform float R2, uniform const float X[], uniform const float Y[],
    uniform const float Z[], uniform const int orig_idx[],
    uniform const int bin_start[], uniform const int bin_end[],
    uniform uint32 hash_mask, uniform int counts[], uniform const int row_ptr[],
    uniform int col_idx[], uniform bool is_write_mode) {
  uniform float invR = 1.0f / R;
  const uint32 K1 = 73856093;
  const uint32 K2 = 19349663;
  const uint32 K3 = 83492791;

  foreach (i = start_query... end_query) {
    float qx = X[i], qy = Y[i], qz = Z[i];
    int cx = (int)(qx * invR);
    int cy = (int)(qy * invR);
    int cz = (int)(qz * invR);
    uint32 hx_start = (uint32)(cx - 1) * K1;
    uint32 hy_start = (uint32)(cy - 1) * K2;
    uint32 hz_start = (uint32)(cz - 1) * K3;

    int local_count = 0;
    int write_offset = 0;
    cif (is_write_mode) {
      write_offset = row_ptr[i];
    }

    uint32 hz = hz_start;
    for (uniform int dz = 0; dz < 3; dz++) {
      uint32 hy = hy_start;
      for (uniform int dy = 0; dy < 3; dy++) {
        uint32 hx = hx_start;
        uint32 hyz = hy ^ hz;
        for (uniform int dx = 0; dx < 3; dx++) {
          uint32 h = (hx ^ hyz) & hash_mask;
          int b_start = bin_start[h];
          int b_end = bin_end[h];

          for (int j = b_start; j < b_end; ++j) {
            float dx_p = X[j] - qx;
            float dy_p = Y[j] - qy;
            float dz_p = Z[j] - qz;
            float dist2 = dx_p * dx_p + dy_p * dy_p + dz_p * dz_p;

            if ((i != j) && (dist2 <= R2)) {
              cif(is_write_mode) { col_idx[write_offset++] = orig_idx[j]; }
              else {
                ++local_count;
              }
            }
          }
          hx += K1;
        }
        hy += K2;
      }
      hz += K3;
    }

    cif(!is_write_mode) { counts[i] = local_count; }
  }
}

export void
count_neighbors_ispc(uniform int start_query, uniform int end_query,
                     uniform float R, uniform float R2, uniform const float X[],
                     uniform const float Y[], uniform const float Z[],
                     uniform const int orig_idx[],
                     uniform const int bin_start[], uniform const int bin_end[],
                     uniform uint32 hash_mask, uniform int counts[]) {
  neighbor_search_core(start_query, end_query, R, R2, X, Y, Z, orig_idx,
                       bin_start, bin_end, hash_mask, counts, NULL, NULL,
                       false);
}

export void
write_neighbors_ispc(uniform int start_query, uniform int end_query,
                     uniform float R, uniform float R2, uniform const float X[],
                     uniform const float Y[], uniform const float Z[],
                     uniform const int orig_idx[],
                     uniform const int bin_start[], uniform const int bin_end[],
                     uniform uint32 hash_mask, uniform const int row_ptr[],
                     uniform int col_idx[]) {
  neighbor_search_core(start_query, end_query, R, R2, X, Y, Z, orig_idx,
                       bin_start, bin_end, hash_mask, NULL, row_ptr, col_idx,
                       true);
}
