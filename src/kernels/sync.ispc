// Coordinate sync from the caller's interleaved xyz.

#include "common.isph"

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
