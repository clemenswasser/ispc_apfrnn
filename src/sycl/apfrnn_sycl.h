#pragma once

// Interface of the optional SYCL GPU backend (APFRNN_ENABLE_SYCL=ON).
//
// Plain C declarations so that the CPU sources never include SYCL headers;
// the implementation is linked statically from the apfrnn_sycl library.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ApfrnnSyclResult {
  int32_t *neighbors;  // total original target indices
  int64_t *offsets;    // nq + 1 row offsets
  int64_t total;       // total neighbors
  int32_t nq;          // number of query rows
} ApfrnnSyclResult;

// One point set as seen by the backend. Grids are cached per id: a cached
// grid is reused only while version, n and the radius are unchanged, so the
// caller must change version whenever the coordinates change.
typedef struct ApfrnnSyclPoints {
  const float *xyz;   // n interleaved xyz points
  int32_t n;
  int32_t id;         // the caller's stable set id
  uint64_t version;
  int32_t is_static;  // keep this set's grid across calls
} ApfrnnSyclPoints;

// Backend state for one caller (one NSearch): device buffers, result slots
// and cached grids. Independent contexts do not affect each other.
typedef struct ApfrnnSyclContext ApfrnnSyclContext;

// Returns 1 when a SYCL GPU queue can be created.
int apfrnn_sycl_available(void);

// Returns NULL when no GPU is available.
ApfrnnSyclContext *apfrnn_sycl_create(void);
void apfrnn_sycl_destroy(ApfrnnSyclContext *ctx);

// Runs an exact fixed-radius search of query against target. With same_set,
// query and target are the same set and self matches are excluded.
//
// slot (>= 0) selects the host buffers the result rows are written to, so
// several pairs stay readable at once; the result points into them and stays
// valid until the next call for that slot. With defer_copy, the copy of a
// same-set result to the host may still be in flight on return: call
// apfrnn_sycl_wait() before reading it. Returns 0 on success.
int apfrnn_sycl_exact_pair(ApfrnnSyclContext *ctx,
                           const ApfrnnSyclPoints *query,
                           const ApfrnnSyclPoints *target, float radius,
                           int32_t same_set, int32_t slot, int32_t defer_copy,
                           ApfrnnSyclResult *result);

// Waits for the deferred copies of this context.
void apfrnn_sycl_wait(ApfrnnSyclContext *ctx);

#ifdef __cplusplus
}
#endif
