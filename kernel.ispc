
// neighbors.ispc

// Bitwise-safe spatial hash for massive coordinate ranges
inline uniform uint32 spatial_hash(uniform int cx, uniform int cy, uniform int cz, uniform uint32 hash_mask) {
    uniform uint32 ux = (uint32)cx;
    uniform uint32 uy = (uint32)cy;
    uniform uint32 uz = (uint32)cz;
    return ((ux * 73856093) ^ (uy * 19349663) ^ (uz * 83492791)) & hash_mask;
}

// PASS 1: Count the exact number of valid neighbors
export void count_neighbors_ispc(
    uniform int start_query, uniform int end_query,
    uniform float R, uniform float R2,
    uniform const float X[], uniform const float Y[], uniform const float Z[],
    uniform const int orig_idx[],
    uniform const int bin_start[], uniform const int bin_end[],
    uniform uint32 hash_mask,
    uniform int counts[]
) {
    // TBB parallelizes over this outer loop. 
    // i is `uniform`, meaning one query particle per gang (no divergence!).
    for (uniform int i = start_query; i < end_query; ++i) {
        uniform float qx = X[i], qy = Y[i], qz = Z[i];
        uniform int cx = (int)floor(qx / R);
        uniform int cy = (int)floor(qy / R);
        uniform int cz = (int)floor(qz / R);

        uniform int local_count = 0;

        // Iterate over 27 neighborhood cells
        for (uniform int dz = -1; dz <= 1; dz++) {
            for (uniform int dy = -1; dy <= 1; dy++) {
                for (uniform int dx = -1; dx <= 1; dx++) {
                    
                    uniform uint32 h = spatial_hash(cx+dx, cy+dy, cz+dz, hash_mask);
                    uniform int b_start = bin_start[h];
                    uniform int b_end   = bin_end[h];

                    // SIMD Parallelization: Process candidates in active vector lanes
                    varying int v_count = 0;
                    foreach (j = b_start ... b_end) {
                        float dx_p = X[j] - qx;
                        float dy_p = Y[j] - qy;
                        float dz_p = Z[j] - qz;
                        float dist2 = dx_p*dx_p + dy_p*dy_p + dz_p*dz_p;
                        
                        if ((i != j) && (dist2 <= R2)) {
                            v_count += 1;
                        }
                    }
                    // Horizontally reduce the active lanes into the scalar counter
                    local_count += reduce_add(v_count);
                }
            }
        }
        // Write exactly to the original un-sorted index position!
        counts[orig_idx[i]] = local_count;
    }
}

// PASS 2: Write neighbor indices to CSR array
export void write_neighbors_ispc(
    uniform int start_query, uniform int end_query,
    uniform float R, uniform float R2,
    uniform const float X[], uniform const float Y[], uniform const float Z[],
    uniform const int orig_idx[],
    uniform const int bin_start[], uniform const int bin_end[],
    uniform uint32 hash_mask,
    uniform const int row_ptr[],
    uniform int col_idx[]
) {
    for (uniform int i = start_query; i < end_query; ++i) {
        uniform float qx = X[i], qy = Y[i], qz = Z[i];
        uniform int q_orig = orig_idx[i];
        
        uniform int cx = (int)floor(qx / R);
        uniform int cy = (int)floor(qy / R);
        uniform int cz = (int)floor(qz / R);

        // Fetch exact starting offset for this particle based on prefix-sum
        uniform int write_offset = row_ptr[q_orig];

        for (uniform int dz = -1; dz <= 1; dz++) {
            for (uniform int dy = -1; dy <= 1; dy++) {
                for (uniform int dx = -1; dx <= 1; dx++) {
                    
                    uniform uint32 h = spatial_hash(cx+dx, cy+dy, cz+dz, hash_mask);
                    uniform int b_start = bin_start[h];
                    uniform int b_end   = bin_end[h];

                    foreach (j = b_start ... b_end) {
                        float dx_p = X[j] - qx;
                        float dy_p = Y[j] - qy;
                        float dz_p = Z[j] - qz;
                        float dist2 = dx_p*dx_p + dy_p*dy_p + dz_p*dz_p;
                        cif ((i != j) && (dist2 <= R2)) {
                            write_offset += packed_store_active(&col_idx[write_offset], orig_idx[j]);
                        }
                    }
                }
            }
        }
    }
}
