// SYCL GPU backend. Compiled into the static apfrnn_sycl library only when
// APFRNN_ENABLE_SYCL=ON; the CPU library calls it through apfrnn_sycl.h.

#include "apfrnn_sycl.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct alignas(16) AosPoint {
  float x;
  float y;
  float z;
  int32_t id;
};

struct DeviceGrid {
  int n = 0;
  float radius = 0.0f;
  float inv_radius = 0.0f;
  float origin_x = 0.0f, origin_y = 0.0f, origin_z = 0.0f;
  int nx = 0, ny = 0, nz = 0;
  std::size_t cell_count = 0;
  float *x = nullptr;
  float *y = nullptr;
  float *z = nullptr;
  int *original = nullptr;
  AosPoint *aos = nullptr;
  int *starts = nullptr;
  int *counts = nullptr;
  int max_cell = 0;
};

inline int cell_coord(float v, float origin, float inv_radius, int count) {
  const int c = static_cast<int>(std::floor((v - origin) * inv_radius));
  return (c < 0 || c >= count) ? -1 : c;
}

// Order-preserving float-to-unsigned mapping for atomic min/max.
inline unsigned float_order_key(float v) {
  const unsigned u = sycl::bit_cast<unsigned>(v);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

inline float float_from_order_key(unsigned k) {
  const unsigned u = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
  return sycl::bit_cast<float>(u);
}

int64_t max_output_bytes() {
  const char *env = std::getenv("APFRNN_SYCL_MAX_OUTPUT_MB");
  const long long mb = env ? std::atoll(env) : 512;
  return std::max<long long>(16, mb) * 1024LL * 1024LL;
}

bool output_within_limit(int64_t total) {
  if (total < 0) {
    return false;
  }
  if (total > max_output_bytes() / static_cast<int64_t>(sizeof(int32_t))) {
    std::fprintf(stderr,
                 "SYCL neighbor output is %lld bytes (limit %lld MB); "
                 "falling back to CPU\n",
                 static_cast<long long>(total) *
                     static_cast<long long>(sizeof(int32_t)),
                 static_cast<long long>(max_output_bytes() / (1024LL * 1024LL)));
    return false;
  }
  return true;
}

// The grouped-query ESIMD kernel wins on the 200k Auto scenes but its extra
// AoS grid layout costs more than it saves at 1M, where the exact-neighbor
// frame is dominated by the output copy. Keep it for grids up to 500k and
// use the cell-major SIMT kernel above that.
bool group16_enabled(int n) { return n <= 500000; }

sycl::queue *gpu_queue() {
  static sycl::queue *queue = nullptr;
  static bool attempted = false;
  if (!attempted) {
    attempted = true;
    try {
      queue = new sycl::queue(
          sycl::gpu_selector_v,
          sycl::property_list{sycl::property::queue::in_order{}});
    } catch (...) {
      queue = nullptr;
    }
  }
  return queue;
}

// Separate in-order queue for deferred same-set output copies. It shares the
// compute queue's context so the event can order the copy after the write.
sycl::queue *gpu_copy_queue() {
  static sycl::queue *queue = nullptr;
  static bool attempted = false;
  if (!attempted) {
    attempted = true;
    sycl::queue *compute = gpu_queue();
    if (compute != nullptr) {
      try {
        queue = new sycl::queue(
            compute->get_context(), compute->get_device(),
            sycl::property_list{sycl::property::queue::in_order{}});
      } catch (...) {
        queue = nullptr;
      }
    }
  }
  return queue;
}

template <typename T>
void ensure_device(sycl::queue &q, T *&ptr, std::size_t &capacity,
                   std::size_t needed) {
  if (capacity >= needed && ptr != nullptr) {
    return;
  }
  if (ptr) {
    sycl::free(ptr, q);
  }
  ptr = sycl::malloc_device<T>(needed, q);
  capacity = needed;
}

template <typename T>
void ensure_host(sycl::queue &q, T *&ptr, std::size_t &capacity,
                 std::size_t needed) {
  if (capacity >= needed && ptr != nullptr) {
    return;
  }
  if (ptr) {
    sycl::free(ptr, q);
  }
  ptr = sycl::malloc_host<T>(needed, q);
  capacity = needed;
}

struct SlotBuffers {
  int32_t *rows = nullptr;
  std::size_t rows_cap = 0;
  int64_t *offsets = nullptr;
  std::size_t offsets_cap = 0;
};

// A grid together with what it was built from.
struct CachedGrid {
  int32_t id = -1;
  uint64_t version = 0;
  int n = 0;
  float radius = 0.0f;
  DeviceGrid grid;

  bool matches(const ApfrnnSyclPoints &points, float r) const {
    return id == points.id && version == points.version && n == points.n &&
           radius == r;
  }
};

}  // namespace

struct ApfrnnSyclContext {
  // Device scratch reused by every search.
  int *dcounts = nullptr;
  std::size_t dcounts_cap = 0;
  int64_t *doffsets = nullptr;
  std::size_t doffsets_cap = 0;
  int32_t *dout = nullptr;
  std::size_t dout_cap = 0;
  // Output of a same-set search whose host copy may still be in flight.
  int32_t *dout_async = nullptr;
  std::size_t dout_async_cap = 0;
  bool async_pending = false;
  int async_slot = -1;
  std::vector<int32_t> counts;
  std::vector<SlotBuffers> slots;
  // One grid per static set, and the last dynamic same-set grid, which the
  // following cross search of that set reuses as its query grid.
  std::vector<CachedGrid> static_grids;
  CachedGrid last_query;
};

namespace {

void free_grid(sycl::queue &q, DeviceGrid &g) {
  if (g.x) sycl::free(g.x, q);
  if (g.y) sycl::free(g.y, q);
  if (g.z) sycl::free(g.z, q);
  if (g.original) sycl::free(g.original, q);
  if (g.aos) sycl::free(g.aos, q);
  if (g.starts) sycl::free(g.starts, q);
  if (g.counts) sycl::free(g.counts, q);
  g.x = g.y = g.z = nullptr;
  g.original = g.starts = g.counts = nullptr;
  g.aos = nullptr;
}

DeviceGrid build_grid_gpu(sycl::queue &q, const float *xyz, int n,
                          float radius) {
  DeviceGrid g;
  g.n = n;
  g.radius = radius;
  g.inv_radius = 1.0f / radius;
  if (n == 0) {
    return g;
  }
  float *dxyz = sycl::malloc_device<float>(static_cast<std::size_t>(n) * 3, q);
  q.memcpy(dxyz, xyz, static_cast<std::size_t>(n) * 3 * sizeof(float));

  // GPU bbox reduction. The host scan over 1M points cost ~2.2 ms/frame;
  // ordered-key atomics keep the min/max exact and order-independent.
  float min_x, min_y, min_z, max_x, max_y, max_z;
  unsigned *dbbox = sycl::malloc_device<unsigned>(6, q);
  const unsigned bbox_init[6] = {0xFFFFFFFFu, 0u, 0xFFFFFFFFu,
                                 0u,         0xFFFFFFFFu, 0u};
  q.memcpy(dbbox, bbox_init, 6 * sizeof(unsigned));
  constexpr int kBBoxWG = 256;
  const int bbox_groups =
      std::min(256, (n + kBBoxWG - 1) / kBBoxWG);
  const int bbox_threads = bbox_groups * kBBoxWG;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<unsigned, 1> lbox(
        sycl::range<1>(static_cast<std::size_t>(6 * kBBoxWG)), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(bbox_threads)),
                          sycl::range<1>(kBBoxWG)),
        [=](sycl::nd_item<1> item) {
          unsigned mnx = 0xFFFFFFFFu, mxx = 0u;
          unsigned mny = 0xFFFFFFFFu, mxy = 0u;
          unsigned mnz = 0xFFFFFFFFu, mxz = 0u;
          for (int i = static_cast<int>(item.get_global_id(0)); i < n;
               i += bbox_threads) {
            const unsigned kx = float_order_key(dxyz[3 * i + 0]);
            const unsigned ky = float_order_key(dxyz[3 * i + 1]);
            const unsigned kz = float_order_key(dxyz[3 * i + 2]);
            mnx = kx < mnx ? kx : mnx;
            mxx = kx > mxx ? kx : mxx;
            mny = ky < mny ? ky : mny;
            mxy = ky > mxy ? ky : mxy;
            mnz = kz < mnz ? kz : mnz;
            mxz = kz > mxz ? kz : mxz;
          }
          const int lid = static_cast<int>(item.get_local_id(0));
          lbox[0 * kBBoxWG + lid] = mnx;
          lbox[1 * kBBoxWG + lid] = mxx;
          lbox[2 * kBBoxWG + lid] = mny;
          lbox[3 * kBBoxWG + lid] = mxy;
          lbox[4 * kBBoxWG + lid] = mnz;
          lbox[5 * kBBoxWG + lid] = mxz;
          item.barrier(sycl::access::fence_space::local_space);
          for (int stride = kBBoxWG / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              unsigned *p0 = &lbox[0 * kBBoxWG + lid];
              unsigned *p1 = &lbox[1 * kBBoxWG + lid];
              unsigned *p2 = &lbox[2 * kBBoxWG + lid];
              unsigned *p3 = &lbox[3 * kBBoxWG + lid];
              unsigned *p4 = &lbox[4 * kBBoxWG + lid];
              unsigned *p5 = &lbox[5 * kBBoxWG + lid];
              const unsigned q0 = lbox[0 * kBBoxWG + lid + stride];
              const unsigned q1 = lbox[1 * kBBoxWG + lid + stride];
              const unsigned q2 = lbox[2 * kBBoxWG + lid + stride];
              const unsigned q3 = lbox[3 * kBBoxWG + lid + stride];
              const unsigned q4 = lbox[4 * kBBoxWG + lid + stride];
              const unsigned q5 = lbox[5 * kBBoxWG + lid + stride];
              *p0 = *p0 < q0 ? *p0 : q0;
              *p1 = *p1 > q1 ? *p1 : q1;
              *p2 = *p2 < q2 ? *p2 : q2;
              *p3 = *p3 > q3 ? *p3 : q3;
              *p4 = *p4 < q4 ? *p4 : q4;
              *p5 = *p5 > q5 ? *p5 : q5;
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (lid == 0) {
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                a0(dbbox[0]);
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                a1(dbbox[1]);
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                a2(dbbox[2]);
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                a3(dbbox[3]);
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                a4(dbbox[4]);
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                a5(dbbox[5]);
            a0.fetch_min(lbox[0 * kBBoxWG]);
            a1.fetch_max(lbox[1 * kBBoxWG]);
            a2.fetch_min(lbox[2 * kBBoxWG]);
            a3.fetch_max(lbox[3 * kBBoxWG]);
            a4.fetch_min(lbox[4 * kBBoxWG]);
            a5.fetch_max(lbox[5 * kBBoxWG]);
          }
        });
  });
  unsigned bbox_keys[6] = {0u, 0u, 0u, 0u, 0u, 0u};
  q.memcpy(bbox_keys, dbbox, 6 * sizeof(unsigned)).wait();
  sycl::free(dbbox, q);
  min_x = float_from_order_key(bbox_keys[0]);
  max_x = float_from_order_key(bbox_keys[1]);
  min_y = float_from_order_key(bbox_keys[2]);
  max_y = float_from_order_key(bbox_keys[3]);
  min_z = float_from_order_key(bbox_keys[4]);
  max_z = float_from_order_key(bbox_keys[5]);
  g.origin_x = min_x - radius;
  g.origin_y = min_y - radius;
  g.origin_z = min_z - radius;
  g.nx = static_cast<int>(
             std::floor((max_x - g.origin_x) * g.inv_radius)) + 2;
  g.ny = static_cast<int>(
             std::floor((max_y - g.origin_y) * g.inv_radius)) + 2;
  g.nz = static_cast<int>(
             std::floor((max_z - g.origin_z) * g.inv_radius)) + 2;
  g.cell_count = static_cast<std::size_t>(g.nx) * g.ny * g.nz;
  if (g.cell_count > 200u * 1000u * 1000u) {
    sycl::free(dxyz, q);
    return g;
  }

  int *dcell = sycl::malloc_device<int>(n, q);
  int *dcounts = sycl::malloc_device<int>(g.cell_count, q);
  int *dcursor = sycl::malloc_device<int>(g.cell_count, q);
  // ESIMD block loads read in 16-element chunks. The last cell of an array
  // may overread up to 15 elements; keep that read inside the allocation and
  // mask the lanes out in the kernel.
  constexpr int kVecPad = 16;
  g.x = sycl::malloc_device<float>(static_cast<std::size_t>(n) + kVecPad, q);
  g.y = sycl::malloc_device<float>(static_cast<std::size_t>(n) + kVecPad, q);
  g.z = sycl::malloc_device<float>(static_cast<std::size_t>(n) + kVecPad, q);
  g.original = sycl::malloc_device<int>(static_cast<std::size_t>(n) + kVecPad, q);
  const bool with_aos = group16_enabled(n);
  if (with_aos) {
    g.aos =
        sycl::malloc_device<AosPoint>(static_cast<std::size_t>(n) + kVecPad, q);
  }
  g.starts = sycl::malloc_device<int>(g.cell_count, q);
  g.counts = dcounts;
  q.parallel_for(sycl::range<1>(g.cell_count),
                 [=](sycl::id<1> id) { dcounts[id[0]] = 0; });
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(n)),
                 [=](sycl::id<1> id) {
                   const int i = static_cast<int>(id[0]);
                   const int cx = cell_coord(dxyz[3 * i + 0], g.origin_x,
                                             g.inv_radius, g.nx);
                   const int cy = cell_coord(dxyz[3 * i + 1], g.origin_y,
                                             g.inv_radius, g.ny);
                   const int cz = cell_coord(dxyz[3 * i + 2], g.origin_z,
                                             g.inv_radius, g.nz);
                   const int c = (cz * g.ny + cy) * g.nx + cx;
                   dcell[i] = c;
                   sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>
                       counter(dcounts[c]);
                   counter.fetch_add(1);
                 });
  std::vector<int> counts(g.cell_count);
  q.memcpy(counts.data(), dcounts, g.cell_count * sizeof(int)).wait();
  std::vector<int> starts(g.cell_count);
  int sum = 0;
  for (std::size_t c = 0; c < g.cell_count; ++c) {
    starts[c] = sum;
    sum += counts[c];
    g.max_cell = std::max(g.max_cell, counts[c]);
  }
  q.memcpy(g.starts, starts.data(), g.cell_count * sizeof(int));
  q.memcpy(dcursor, starts.data(), g.cell_count * sizeof(int));
  if (with_aos) {
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(n)),
                   [=](sycl::id<1> id) {
                     const int i = static_cast<int>(id[0]);
                     const int c = dcell[i];
                     sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device>
                         cursor(dcursor[c]);
                     const int dst = cursor.fetch_add(1);
                     g.x[dst] = dxyz[3 * i + 0];
                     g.y[dst] = dxyz[3 * i + 1];
                     g.z[dst] = dxyz[3 * i + 2];
                     g.original[dst] = i;
                     g.aos[dst] = AosPoint{dxyz[3 * i + 0], dxyz[3 * i + 1],
                                           dxyz[3 * i + 2],
                                           static_cast<int32_t>(i)};
                   });
  } else {
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(n)),
                   [=](sycl::id<1> id) {
                     const int i = static_cast<int>(id[0]);
                     const int c = dcell[i];
                     sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device>
                         cursor(dcursor[c]);
                     const int dst = cursor.fetch_add(1);
                     g.x[dst] = dxyz[3 * i + 0];
                     g.y[dst] = dxyz[3 * i + 1];
                     g.z[dst] = dxyz[3 * i + 2];
                     g.original[dst] = i;
                   });
  }
  q.wait();
  sycl::free(dxyz, q);
  sycl::free(dcell, q);
  sycl::free(dcursor, q);
  return g;
}

SlotBuffers &slot_buffers(ApfrnnSyclContext &ctx, int slot) {
  if (static_cast<std::size_t>(slot) >= ctx.slots.size()) {
    ctx.slots.resize(static_cast<std::size_t>(slot) + 1);
  }
  return ctx.slots[static_cast<std::size_t>(slot)];
}

// Turns the per-query counts in ctx.dcounts into row offsets (in the slot's
// host buffer and on the device) and sizes the device output. Returns the
// total, or -1 when the output would exceed the limit.
int64_t prepare_rows(sycl::queue &q, ApfrnnSyclContext &ctx, int nq, int slot,
                     bool async_output, int32_t *&dout, int64_t *&doffsets) {
  ctx.counts.resize(static_cast<std::size_t>(nq));
  q.memcpy(ctx.counts.data(), ctx.dcounts,
           static_cast<std::size_t>(nq) * sizeof(int))
      .wait();
  SlotBuffers &buffers = slot_buffers(ctx, slot);
  ensure_host<int64_t>(q, buffers.offsets, buffers.offsets_cap,
                       static_cast<std::size_t>(nq) + 1);
  buffers.offsets[0] = 0;
  for (int i = 0; i < nq; ++i) {
    buffers.offsets[i + 1] =
        buffers.offsets[i] + ctx.counts[static_cast<std::size_t>(i)];
  }
  const int64_t total = buffers.offsets[nq];
  if (!output_within_limit(total)) {
    return -1;
  }
  ensure_device<int64_t>(q, ctx.doffsets, ctx.doffsets_cap,
                         static_cast<std::size_t>(nq) + 1);
  q.memcpy(ctx.doffsets, buffers.offsets,
           (static_cast<std::size_t>(nq) + 1) * sizeof(int64_t));
  if (async_output) {
    ensure_device<int32_t>(q, ctx.dout_async, ctx.dout_async_cap,
                           static_cast<std::size_t>(total) + 1);
    dout = ctx.dout_async;
  } else {
    ensure_device<int32_t>(q, ctx.dout, ctx.dout_cap,
                           static_cast<std::size_t>(total) + 1);
    dout = ctx.dout;
  }
  doffsets = ctx.doffsets;
  return total;
}

// Copies the rows to the slot's host buffer. With async_output the copy runs
// on the copy queue and apfrnn_sycl_wait() completes it.
void copy_rows(sycl::queue &q, ApfrnnSyclContext &ctx, int slot,
               const int32_t *dout, int64_t total, bool async_output) {
  SlotBuffers &buffers = slot_buffers(ctx, slot);
  ensure_host<int32_t>(q, buffers.rows, buffers.rows_cap,
                       static_cast<std::size_t>(total) + 1);
  const std::size_t bytes = static_cast<std::size_t>(total) * sizeof(int32_t);
  sycl::queue *copy_q = async_output ? gpu_copy_queue() : nullptr;
  if (copy_q != nullptr) {
    copy_q->memcpy(buffers.rows, dout, bytes);
    ctx.async_pending = true;
    ctx.async_slot = slot;
  } else {
    q.memcpy(buffers.rows, dout, bytes).wait();
  }
}

int64_t search_pair_same_group16(sycl::queue &q, ApfrnnSyclContext &ctx,
                                      const DeviceGrid &target, int slot,
                                      bool async_output) {
  constexpr int kWG = 32;
  constexpr int kVec = 16;
  const int nq = target.n;
  const std::size_t cell_count = target.cell_count;
  if (nq == 0 || cell_count == 0) {
    return 0;
  }
  ensure_device<int>(q, ctx.dcounts, ctx.dcounts_cap,
                     static_cast<std::size_t>(nq));
  const float radius2 = target.radius * target.radius;
  const float *tx = target.x;
  const float *ty = target.y;
  const float *tz = target.z;
  const int *torig = target.original;
  const AosPoint *aos = target.aos;
  const int *tstarts = target.starts;
  const int *tcounts = target.counts;
  const int nx = target.nx, ny = target.ny, nz = target.nz;
  int *dcounts = ctx.dcounts;
  const std::size_t groups = (cell_count + kWG - 1) / kWG;

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kWG),
                          sycl::range<1>(kWG)),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
          using namespace sycl::ext::intel::esimd;
          const int cell = static_cast<int>(item.get_group(0)) * kWG +
                           static_cast<int>(item.get_local_id(0));
          if (cell >= static_cast<int>(cell_count)) {
            return;
          }
          const int cz = cell / (nx * ny);
          const int rem = cell % (nx * ny);
          const int cy = rem / nx;
          const int cx = rem % nx;
          const int begin = tstarts[cell];
          const int end = begin + tcounts[cell];
          simd<unsigned, kVec> lane(0, 1);
          for (int q0 = begin; q0 < end; q0 += kVec) {
            const int m = (end - q0 < kVec) ? end - q0 : kVec;
            const simd_mask<kVec> valid = lane < static_cast<unsigned>(m);
            const simd<float, kVec> qx = block_load<float, kVec>(tx + q0);
            const simd<float, kVec> qy = block_load<float, kVec>(ty + q0);
            const simd<float, kVec> qz = block_load<float, kVec>(tz + q0);
            const simd<int32_t, kVec> qorig =
                block_load<int32_t, kVec>(torig + q0);
            simd<int32_t, kVec> counts = 0;
            for (int dz = -1; dz <= 1; ++dz) {
              const int z = cz + dz;
              if (z < 0 || z >= nz) continue;
              for (int dy = -1; dy <= 1; ++dy) {
                const int y = cy + dy;
                if (y < 0 || y >= ny) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                  const int x = cx + dx;
                  if (x < 0 || x >= nx) continue;
                  const int nc = (z * ny + y) * nx + x;
                  const int nb = tstarts[nc];
                  const int ne = nb + tcounts[nc];
                  const int32_t *fp =
                      reinterpret_cast<const int32_t *>(aos) + 4 * nb;
                  const int32_t *fp_end =
                      reinterpret_cast<const int32_t *>(aos) + 4 * ne;
                  for (; fp < fp_end; fp += 4) {
                    const simd<int32_t, 4> p = block_load<int32_t, 4>(fp);
                    const simd<float, kVec> ddx =
                        qx - sycl::bit_cast<float>(p[0]);
                    const simd<float, kVec> ddy =
                        qy - sycl::bit_cast<float>(p[1]);
                    const simd<float, kVec> ddz =
                        qz - sycl::bit_cast<float>(p[2]);
                    simd_mask<kVec> hit =
                        (ddx * ddx + ddy * ddy + ddz * ddz) <= radius2;
                    hit &= valid;
                    hit &= (qorig != p[3]);
                    simd<int32_t, kVec> add = 0;
                    add.merge(1, hit);
                    counts += add;
                  }
                }
              }
            }
            for (int l = 0; l < m; ++l) {
              dcounts[qorig[l]] = counts[l];
            }
          }
        });
  });
  q.wait();

  int32_t *dout = nullptr;
  int64_t *doffsets = nullptr;
  const int64_t total =
      prepare_rows(q, ctx, nq, slot, async_output, dout, doffsets);
  if (total < 0) {
    return -1;
  }

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kWG),
                          sycl::range<1>(kWG)),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
          using namespace sycl::ext::intel::esimd;
          const int cell = static_cast<int>(item.get_group(0)) * kWG +
                           static_cast<int>(item.get_local_id(0));
          if (cell >= static_cast<int>(cell_count)) {
            return;
          }
          const int cz = cell / (nx * ny);
          const int rem = cell % (nx * ny);
          const int cy = rem / nx;
          const int cx = rem % nx;
          const int begin = tstarts[cell];
          const int end = begin + tcounts[cell];
          simd<unsigned, kVec> lane(0, 1);
          for (int q0 = begin; q0 < end; q0 += kVec) {
            const int m = (end - q0 < kVec) ? end - q0 : kVec;
            const simd_mask<kVec> valid = lane < static_cast<unsigned>(m);
            const simd<float, kVec> qx = block_load<float, kVec>(tx + q0);
            const simd<float, kVec> qy = block_load<float, kVec>(ty + q0);
            const simd<float, kVec> qz = block_load<float, kVec>(tz + q0);
            const simd<int32_t, kVec> qorig =
                block_load<int32_t, kVec>(torig + q0);
            // Keep byte offsets: the scatter address is then ready to use
            // without a per-candidate shift.
            simd<uint32_t, kVec> pos = 0;
            for (int l = 0; l < m; ++l) {
              pos[l] = static_cast<uint32_t>(doffsets[qorig[l]] * 4);
            }
            for (int dz = -1; dz <= 1; ++dz) {
              const int z = cz + dz;
              if (z < 0 || z >= nz) continue;
              for (int dy = -1; dy <= 1; ++dy) {
                const int y = cy + dy;
                if (y < 0 || y >= ny) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                  const int x = cx + dx;
                  if (x < 0 || x >= nx) continue;
                  const int nc = (z * ny + y) * nx + x;
                  const int nb = tstarts[nc];
                  const int ne = nb + tcounts[nc];
                  const int32_t *fp =
                      reinterpret_cast<const int32_t *>(aos) + 4 * nb;
                  const int32_t *fp_end =
                      reinterpret_cast<const int32_t *>(aos) + 4 * ne;
                  for (; fp < fp_end; fp += 4) {
                    const simd<int32_t, 4> p = block_load<int32_t, 4>(fp);
                    const simd<float, kVec> ddx =
                        qx - sycl::bit_cast<float>(p[0]);
                    const simd<float, kVec> ddy =
                        qy - sycl::bit_cast<float>(p[1]);
                    const simd<float, kVec> ddz =
                        qz - sycl::bit_cast<float>(p[2]);
                    simd_mask<kVec> hit =
                        (ddx * ddx + ddy * ddy + ddz * ddz) <= radius2;
                    hit &= valid;
                    hit &= (qorig != p[3]);
                    scatter<int32_t, kVec>(dout, pos, p[3], hit);
                    simd<uint32_t, kVec> inc = 0;
                    inc.merge(4u, hit);
                    pos += inc;
                  }
                }
              }
            }
          }
        });
  });
  q.wait();
  copy_rows(q, ctx, slot, dout, total, async_output);
  return total;
}

int64_t search_pair_same_cell_major(sycl::queue &q, ApfrnnSyclContext &ctx,
                                          const DeviceGrid &target, int slot,
                                          bool async_output = false) {
  constexpr int kCellsPerWG = 4;
  constexpr int kLanesPerCell = 8;
  constexpr int kWG = kCellsPerWG * kLanesPerCell;
  const int nq = target.n;
  const std::size_t cell_count = target.cell_count;
  if (nq == 0 || cell_count == 0) {
    return 0;
  }
  ensure_device<int>(q, ctx.dcounts, ctx.dcounts_cap,
                     static_cast<std::size_t>(nq));
  const float radius2 = target.radius * target.radius;
  const int nx = target.nx, ny = target.ny, nz = target.nz;
  const float *tx = target.x;
  const float *ty = target.y;
  const float *tz = target.z;
  const int *torig = target.original;
  const int *tstarts = target.starts;
  const int *tcounts = target.counts;
  int *dcounts = ctx.dcounts;

  q.parallel_for(
      sycl::nd_range<1>(
          sycl::range<1>(((cell_count + kCellsPerWG - 1) / kCellsPerWG) * kWG),
          sycl::range<1>(kWG)),
      [=](sycl::nd_item<1> item) {
        const int group = static_cast<int>(item.get_group(0));
        const int lane = static_cast<int>(item.get_local_id(0));
        const int sub = lane / kLanesPerCell;
        const int qlane = lane % kLanesPerCell;
        const int cell = group * kCellsPerWG + sub;
        if (cell >= static_cast<int>(cell_count)) {
          return;
        }
        const int cz = cell / (nx * ny);
        const int rem = cell % (nx * ny);
        const int cy = rem / nx;
        const int cx = rem % nx;
        const int begin = tstarts[cell];
        const int end = begin + tcounts[cell];
        for (int q = begin + qlane; q < end; q += kLanesPerCell) {
          const int qi = torig[q];
          const float qx = tx[q];
          const float qy = ty[q];
          const float qz = tz[q];
          int count = 0;
          for (int dz = -1; dz <= 1; ++dz) {
            const int z = cz + dz;
            if (z < 0 || z >= nz) continue;
            for (int dy = -1; dy <= 1; ++dy) {
              const int y = cy + dy;
              if (y < 0 || y >= ny) continue;
              for (int dx = -1; dx <= 1; ++dx) {
                const int x = cx + dx;
                if (x < 0 || x >= nx) continue;
                const int nc = (z * ny + y) * nx + x;
                const int nb = tstarts[nc];
                const int ne = nb + tcounts[nc];
                for (int j = nb; j < ne; ++j) {
                  const float ddx = tx[j] - qx;
                  const float ddy = ty[j] - qy;
                  const float ddz = tz[j] - qz;
                  if (ddx * ddx + ddy * ddy + ddz * ddz <= radius2 &&
                      torig[j] != qi) {
                    ++count;
                  }
                }
              }
            }
          }
          dcounts[qi] = count;
        }
      });
  q.wait();

  int32_t *dout = nullptr;
  int64_t *doffsets = nullptr;
  const int64_t total =
      prepare_rows(q, ctx, nq, slot, async_output, dout, doffsets);
  if (total < 0) {
    return -1;
  }

  q.parallel_for(
      sycl::nd_range<1>(
          sycl::range<1>(((cell_count + kCellsPerWG - 1) / kCellsPerWG) * kWG),
          sycl::range<1>(kWG)),
      [=](sycl::nd_item<1> item) {
        const int group = static_cast<int>(item.get_group(0));
        const int lane = static_cast<int>(item.get_local_id(0));
        const int sub = lane / kLanesPerCell;
        const int qlane = lane % kLanesPerCell;
        const int cell = group * kCellsPerWG + sub;
        if (cell >= static_cast<int>(cell_count)) {
          return;
        }
        const int cz = cell / (nx * ny);
        const int rem = cell % (nx * ny);
        const int cy = rem / nx;
        const int cx = rem % nx;
        const int begin = tstarts[cell];
        const int end = begin + tcounts[cell];
        for (int q = begin + qlane; q < end; q += kLanesPerCell) {
          const int qi = torig[q];
          const float qx = tx[q];
          const float qy = ty[q];
          const float qz = tz[q];
          int64_t pos = doffsets[qi];
          for (int dz = -1; dz <= 1; ++dz) {
            const int z = cz + dz;
            if (z < 0 || z >= nz) continue;
            for (int dy = -1; dy <= 1; ++dy) {
              const int y = cy + dy;
              if (y < 0 || y >= ny) continue;
              for (int dx = -1; dx <= 1; ++dx) {
                const int x = cx + dx;
                if (x < 0 || x >= nx) continue;
                const int nc = (z * ny + y) * nx + x;
                const int nb = tstarts[nc];
                const int ne = nb + tcounts[nc];
                for (int j = nb; j < ne; ++j) {
                  const float ddx = tx[j] - qx;
                  const float ddy = ty[j] - qy;
                  const float ddz = tz[j] - qz;
                  if (ddx * ddx + ddy * ddy + ddz * ddz <= radius2 &&
                      torig[j] != qi) {
                    dout[pos++] = torig[j];
                  }
                }
              }
            }
          }
        }
      });
  q.wait();
  copy_rows(q, ctx, slot, dout, total, async_output);
  return total;
}

int64_t search_pair_cross_cell_major(sycl::queue &q, ApfrnnSyclContext &ctx,
                                           const DeviceGrid &query,
                                           const DeviceGrid &target,
                                           int slot) {
  constexpr int kCellsPerWG = 4;
  constexpr int kLanesPerCell = 8;
  constexpr int kWG = kCellsPerWG * kLanesPerCell;
  const int nq = query.n;
  if (nq == 0 || query.cell_count == 0 || target.cell_count == 0) {
    return 0;
  }
  ensure_device<int>(q, ctx.dcounts, ctx.dcounts_cap,
                     static_cast<std::size_t>(nq));
  const float radius2 = target.radius * target.radius;
  const float to_x = target.origin_x;
  const float to_y = target.origin_y;
  const float to_z = target.origin_z;
  const int tnx = target.nx, tny = target.ny, tnz = target.nz;
  const float *qx = query.x;
  const float *qy = query.y;
  const float *qz = query.z;
  const int *qorig = query.original;
  const int *qstarts = query.starts;
  const int *qcounts = query.counts;
  const float *tx = target.x;
  const float *ty = target.y;
  const float *tz = target.z;
  const int *torig = target.original;
  const int *tstarts = target.starts;
  const int *tcounts = target.counts;
  int *dcounts = ctx.dcounts;

  q.parallel_for(
      sycl::nd_range<1>(
          sycl::range<1>(((query.cell_count + kCellsPerWG - 1) /
                          kCellsPerWG) * kWG),
          sycl::range<1>(kWG)),
      [=](sycl::nd_item<1> item) {
        const int group = static_cast<int>(item.get_group(0));
        const int lane = static_cast<int>(item.get_local_id(0));
        const int sub = lane / kLanesPerCell;
        const int qlane = lane % kLanesPerCell;
        const int cell = group * kCellsPerWG + sub;
        if (cell >= static_cast<int>(query.cell_count)) {
          return;
        }
        const int begin = qstarts[cell];
        const int end = begin + qcounts[cell];
        for (int qidx = begin + qlane; qidx < end; qidx += kLanesPerCell) {
          const int qi = qorig[qidx];
          const float px = qx[qidx];
          const float py = qy[qidx];
          const float pz = qz[qidx];
          const int tcx = static_cast<int>(std::floor((px - to_x) * target.inv_radius));
          const int tcy = static_cast<int>(std::floor((py - to_y) * target.inv_radius));
          const int tcz = static_cast<int>(std::floor((pz - to_z) * target.inv_radius));
          int count = 0;
          if (tcx >= 0 && tcx < tnx && tcy >= 0 && tcy < tny && tcz >= 0 &&
              tcz < tnz) {
            for (int dz = -1; dz <= 1; ++dz) {
              const int z = tcz + dz;
              if (z < 0 || z >= tnz) continue;
              for (int dy = -1; dy <= 1; ++dy) {
                const int y = tcy + dy;
                if (y < 0 || y >= tny) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                  const int x = tcx + dx;
                  if (x < 0 || x >= tnx) continue;
                  const int nc = (z * tny + y) * tnx + x;
                  const int nb = tstarts[nc];
                  const int ne = nb + tcounts[nc];
                  for (int j = nb; j < ne; ++j) {
                    const float ddx = tx[j] - px;
                    const float ddy = ty[j] - py;
                    const float ddz = tz[j] - pz;
                    if (ddx * ddx + ddy * ddy + ddz * ddz <= radius2) {
                      ++count;
                    }
                  }
                }
              }
            }
          }
          dcounts[qi] = count;
        }
      });
  q.wait();

  int32_t *dout = nullptr;
  int64_t *doffsets = nullptr;
  const int64_t total =
      prepare_rows(q, ctx, nq, slot, false, dout, doffsets);
  if (total < 0) {
    return -1;
  }

  q.parallel_for(
      sycl::nd_range<1>(
          sycl::range<1>(((query.cell_count + kCellsPerWG - 1) /
                          kCellsPerWG) * kWG),
          sycl::range<1>(kWG)),
      [=](sycl::nd_item<1> item) {
        const int group = static_cast<int>(item.get_group(0));
        const int lane = static_cast<int>(item.get_local_id(0));
        const int sub = lane / kLanesPerCell;
        const int qlane = lane % kLanesPerCell;
        const int cell = group * kCellsPerWG + sub;
        if (cell >= static_cast<int>(query.cell_count)) {
          return;
        }
        const int begin = qstarts[cell];
        const int end = begin + qcounts[cell];
        for (int qidx = begin + qlane; qidx < end; qidx += kLanesPerCell) {
          const int qi = qorig[qidx];
          const float px = qx[qidx];
          const float py = qy[qidx];
          const float pz = qz[qidx];
          const int tcx = static_cast<int>(std::floor((px - to_x) * target.inv_radius));
          const int tcy = static_cast<int>(std::floor((py - to_y) * target.inv_radius));
          const int tcz = static_cast<int>(std::floor((pz - to_z) * target.inv_radius));
          int64_t pos = doffsets[qi];
          if (tcx >= 0 && tcx < tnx && tcy >= 0 && tcy < tny && tcz >= 0 &&
              tcz < tnz) {
            for (int dz = -1; dz <= 1; ++dz) {
              const int z = tcz + dz;
              if (z < 0 || z >= tnz) continue;
              for (int dy = -1; dy <= 1; ++dy) {
                const int y = tcy + dy;
                if (y < 0 || y >= tny) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                  const int x = tcx + dx;
                  if (x < 0 || x >= tnx) continue;
                  const int nc = (z * tny + y) * tnx + x;
                  const int nb = tstarts[nc];
                  const int ne = nb + tcounts[nc];
                  for (int j = nb; j < ne; ++j) {
                    const float ddx = tx[j] - px;
                    const float ddy = ty[j] - py;
                    const float ddz = tz[j] - pz;
                    if (ddx * ddx + ddy * ddy + ddz * ddz <= radius2) {
                      dout[pos++] = torig[j];
                    }
                  }
                }
              }
            }
          }
        }
      });
  q.wait();

  copy_rows(q, ctx, slot, dout, total, false);
  return total;
}

}  // namespace

extern "C" {

int apfrnn_sycl_available(void) { return gpu_queue() != nullptr ? 1 : 0; }

ApfrnnSyclContext *apfrnn_sycl_create(void) {
  return gpu_queue() != nullptr ? new ApfrnnSyclContext() : nullptr;
}

void apfrnn_sycl_destroy(ApfrnnSyclContext *ctx) {
  if (ctx == nullptr) {
    return;
  }
  sycl::queue *q = gpu_queue();
  if (q != nullptr) {
    apfrnn_sycl_wait(ctx);
    sycl::free(ctx->dcounts, *q);
    sycl::free(ctx->doffsets, *q);
    sycl::free(ctx->dout, *q);
    sycl::free(ctx->dout_async, *q);
    for (SlotBuffers &buffers : ctx->slots) {
      sycl::free(buffers.rows, *q);
      sycl::free(buffers.offsets, *q);
    }
    for (CachedGrid &cached : ctx->static_grids) {
      free_grid(*q, cached.grid);
    }
    free_grid(*q, ctx->last_query.grid);
  }
  delete ctx;
}

void apfrnn_sycl_wait(ApfrnnSyclContext *ctx) {
  if (ctx == nullptr || !ctx->async_pending) {
    return;
  }
  sycl::queue *copy_q = gpu_copy_queue();
  if (copy_q != nullptr) {
    copy_q->wait();
  }
  ctx->async_pending = false;
  ctx->async_slot = -1;
}

int apfrnn_sycl_exact_pair(ApfrnnSyclContext *ctx,
                           const ApfrnnSyclPoints *query,
                           const ApfrnnSyclPoints *target, float radius,
                           int32_t same_set, int32_t slot, int32_t defer_copy,
                           ApfrnnSyclResult *result) {
  if (ctx == nullptr || query == nullptr || target == nullptr ||
      result == nullptr || query->xyz == nullptr || target->xyz == nullptr ||
      query->n < 0 || target->n < 0 || slot < 0 || !(radius > 0.0f)) {
    return 1;
  }
  result->neighbors = nullptr;
  result->offsets = nullptr;
  result->total = 0;
  result->nq = query->n;

  sycl::queue *q = gpu_queue();
  if (q == nullptr) {
    return 2;
  }
  // The in-flight copy may target this slot, and dout_async is reused.
  if (ctx->async_pending && (defer_copy || slot == ctx->async_slot)) {
    apfrnn_sycl_wait(ctx);
  }

  // Returns the grid of `points`: a still-valid cached one, or a rebuilt
  // one stored in `slot_grid` (freeing what it held before).
  auto cached_grid = [&](const ApfrnnSyclPoints &points,
                         CachedGrid &slot_grid) -> DeviceGrid * {
    if (!slot_grid.matches(points, radius)) {
      free_grid(*q, slot_grid.grid);
      slot_grid.grid = build_grid_gpu(*q, points.xyz, points.n, radius);
      slot_grid.id = points.id;
      slot_grid.version = points.version;
      slot_grid.n = points.n;
      slot_grid.radius = radius;
    }
    const DeviceGrid &g = slot_grid.grid;
    return g.n == 0 || g.cell_count == 0 || g.x == nullptr ? nullptr
                                                           : &slot_grid.grid;
  };
  auto static_slot = [&](const ApfrnnSyclPoints &points) -> CachedGrid & {
    for (CachedGrid &cached : ctx->static_grids) {
      if (cached.id == points.id) {
        return cached;
      }
    }
    ctx->static_grids.emplace_back();
    return ctx->static_grids.back();
  };

  try {
    DeviceGrid *target_grid = nullptr;
    CachedGrid target_tmp;
    if (target->is_static != 0) {
      target_grid = cached_grid(*target, static_slot(*target));
    } else if (same_set != 0) {
      target_grid = cached_grid(*target, ctx->last_query);
    } else {
      target_grid = cached_grid(*target, target_tmp);
    }

    DeviceGrid *query_grid = target_grid;
    CachedGrid query_tmp;
    if (same_set == 0) {
      if (query->is_static != 0) {
        query_grid = cached_grid(*query, static_slot(*query));
      } else if (ctx->last_query.matches(*query, radius)) {
        query_grid = &ctx->last_query.grid;
      } else {
        query_grid = cached_grid(*query, query_tmp);
      }
    }

    // The temporary grids are still needed by the kernels below. Free them
    // when this scope ends, including when a launch throws.
    struct FreeTemporaries {
      sycl::queue &queue;
      CachedGrid &target;
      CachedGrid &query;
      ~FreeTemporaries() {
        free_grid(queue, target.grid);
        free_grid(queue, query.grid);
      }
    } free_temporaries{*q, target_tmp, query_tmp};
    if (target_grid == nullptr || query_grid == nullptr) {
      return 3;
    }
    // One work item walks one cell's queries against the 27 neighboring
    // cells, so a dense cell concentrates its whole neighborhood in a single
    // item and can reset the device (which then stays lost for the rest of
    // the process). Reject it and let the caller fall back to the CPU.
    constexpr int64_t kMaxCellComparisons = 64LL * 1024 * 1024;
    const auto neighborhood_bound = [](const DeviceGrid &grid) {
      return std::min<int64_t>(int64_t{27} * grid.max_cell,
                               static_cast<int64_t>(grid.n));
    };
    const bool too_dense =
        same_set != 0
            ? static_cast<int64_t>(target_grid->max_cell) *
                      neighborhood_bound(*target_grid) >
                  kMaxCellComparisons
            : static_cast<int64_t>(query_grid->max_cell) *
                      neighborhood_bound(*target_grid) >
                  kMaxCellComparisons;
    if (too_dense) {
      return 8;
    }

    int64_t total = -1;
    // Grouped-query ESIMD for same-set grids up to 500k (the ones built with
    // an AoS copy), the cell-major SIMT kernel otherwise.
    const bool defer_same = defer_copy != 0 && same_set != 0;
    if (same_set == 0) {
      total = search_pair_cross_cell_major(*q, *ctx, *query_grid,
                                           *target_grid, slot);
    } else if (target_grid->aos != nullptr) {
      total = search_pair_same_group16(*q, *ctx, *target_grid, slot,
                                       defer_same);
    } else {
      total = search_pair_same_cell_major(*q, *ctx, *target_grid, slot,
                                          defer_same);
    }
    if (total < 0) {
      return 7;
    }
    const SlotBuffers &buffers = slot_buffers(*ctx, slot);
    result->offsets = buffers.offsets;
    result->neighbors = buffers.rows;
    result->total = total;
    return 0;
  } catch (...) {
    return 6;
  }
}

}  // extern "C"
