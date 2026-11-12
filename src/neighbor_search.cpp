#include "apfrnn/neighbor_search.hpp"

#include "apfrnn.hpp"
#include "plan.hpp"
#include "sweep.hpp"

#ifdef APFRNN_HAVE_SYCL
#include "sycl/apfrnn_sycl.h"
#endif

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <vector>

namespace apfrnn {

namespace {

constexpr int kPairParallelMaxPoints = 32768;
// Below this, the per-row offset/capacity loads cost more than the padding
// they save, so tiny cross pairs keep the plain fixed layout.
constexpr int kCrossBlockMinPoints = 4096;
// Below this size the fixed cross rows are small enough that the hybrid's
// per-frame layout scan costs more than the padding it removes.
constexpr int kCrossHybridMinPoints = 1 << 17;

SearchBackend search_backend_from_env() {
  const char *backend = std::getenv("APFRNN_BACKEND");
  if (backend == nullptr) {
    return SearchBackend::CpuExact;
  }
  if (std::strcmp(backend, "auto") == 0) {
    return SearchBackend::Auto;
  }
  if (std::strcmp(backend, "gpu") == 0) {
    return SearchBackend::GpuExact;
  }
  return SearchBackend::CpuExact;
}

enum class SourceType {
  None,
  Float,
  Double,
};

// original_to_sorted is the inverse of data.original_index.
void update_sorted_mapping(const NeighborSearchData &data,
                           IntBuffer &original_to_sorted) {
  original_to_sorted.resize(static_cast<std::size_t>(data.num_points));
  tbb::parallel_for(
      tbb::blocked_range<int>(0, data.num_points, 4096),
      [&](const tbb::blocked_range<int> &range) {
        for (int sorted = range.begin(); sorted < range.end(); ++sorted) {
          original_to_sorted[static_cast<std::size_t>(
              data.original_index[static_cast<std::size_t>(sorted)])] = sorted;
        }
      });
}

// Runs after every row writer and again for the frame stats, so for large
// sets a serial pass is a fixed per-frame cost that stops thread scaling
// (~2.5 ms of a 9 ms uniform_1m frame on 96 cores).
int max_row_count(const IntBuffer &counts) {
  constexpr int kGrain = 1 << 16;
  const int n = static_cast<int>(counts.size());
  if (n <= kGrain) {
    return n == 0
               ? 0
               : std::max(0, *std::max_element(counts.begin(), counts.end()));
  }
  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, n, kGrain), 0,
      [&](const tbb::blocked_range<int> &range, int acc) {
        for (int i = range.begin(); i < range.end(); ++i) {
          acc = std::max(acc, counts[static_cast<std::size_t>(i)]);
        }
        return acc;
      },
      [](int a, int b) { return std::max(a, b); });
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

struct NSearch::PointSet {
  bool is_static = false;
  bool dirty = true;
  bool valid = false;
  bool same_lists_valid = false;
  bool fixed_valid = false;
  int fixed_capacity = 128;
  int last_fixed_capacity = 0;
  Bounds bounds;  // of x/y/z, maintained by sync_source
  bool use_sweep = false;
  bool sweep_decision_valid = false;
  bool sweep_decision = false;
  bool refresh_rows = false;
  sweep::Index sweep;
  SourceType source_type = SourceType::None;
  const void *source = nullptr;
  int source_count = 0;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> cached_x;
  std::vector<float> cached_y;
  std::vector<float> cached_z;
  bool target_cache_valid = false;
  std::vector<float> gpu_xyz;
  bool gpu_xyz_valid = false;
  uint64_t gpu_version = 0;  // bumped whenever gpu_xyz is repacked
  int probe_max_cell = -1;
  bool probe_valid = false;
  int probe_age = 0;
  NeighborSearchData data;
  IntBuffer original_to_sorted;
  IntBuffer fixed_neighbors;
  IntBuffer fixed_counts;
  // Sparse same-set rows are indexed by original point, so get_neighbors
  // skips the sorted mapping. Every other row layout keeps the sorted index.
  bool sparse_rows_by_original = false;
  // Fixed rows placed by original point: row p of fixed_neighbors belongs to
  // the caller's point p and slot_counts[p] is its count, so a caller walking
  // its points in order reads the table front to back. fixed_counts stays in
  // sorted order for the layout decisions.
  bool rows_by_original = false;
  IntBuffer slot_counts;
  // Row starts by original point for placed per-point rows (block_rows).
  IntBuffer point_offset;
  // Max row count of the fixed rows, known from the writer; -1 forces the
  // stats pass to rescan (paths that do not track it).
  int fixed_max_count = -1;
  // Sparse target-major cross rows. The counts buffer is kept across
  // invalidations for pairs that use that writer; only the rows it wrote are
  // cleared before the next build.
  std::vector<int> incremental_rows_touched;
  bool incremental_rows_touched_overflow = false;
  bool incremental_rows_owned = false;
  // Block-fixed rows: per-block capacities and per-query row offsets.
  bool block_rows = false;
  // Packed fixed rows: row_offset gives each sorted row's start, with no
  // padding between rows (see write_neighbors_packed_dense_parallel).
  bool packed_rows = false;
  bool block_caps_valid = false;
  // row_offset/row_capacity and the buffer size match block_caps. On an
  // unchanged grid nothing overflows, so later frames skip the layout pass.
  bool block_layout_valid = false;
  int block_total = 0;
  int block_max_capacity = 0;
  int64_t block_layout_budget = 0; // memory_budget_ the layout was checked at
  int block_size = 0;
  IntBuffer block_offsets;
  IntBuffer block_caps;
  IntBuffer row_offset;
  IntBuffer row_capacity;
  // Moving sets (see build_moving_block_rows): per-point row capacities by
  // original index, which, unlike the sorted index, survives a grid rebuild.
  IntBuffer moving_caps;
  bool moving_caps_valid = false;
  int moving_check_wait = 0; // frames until fixed rows are re-checked
  // Bumped whenever the sorted grid (and therefore the cell keys and sorted
  // index order) is rebuilt. Cross-pair caches compare it to detect reuse.
  uint64_t grid_generation = 0;
  // grid_generation at the last same-set list build. Block rows are cached
  // per sorted index, which a grid rebuild reorders, so they are only used
  // while the grid is unchanged.
  static constexpr uint64_t kNoGeneration = ~uint64_t{0};
  uint64_t lists_grid_generation = kNoGeneration;
  // grid_generation at which the fixed layout was last chosen over block
  // rows. The counts of an unchanged grid cannot change that decision, so it
  // is not re-evaluated (three passes over the counts) every frame.
  uint64_t fixed_layout_generation = kNoGeneration;
};

struct NSearch::Pair {
  bool active = true;
  bool valid = false;
  bool same_set = false;
  bool fixed_valid = false;
  int fixed_capacity = 64;
  int last_fixed_capacity = 0;
  bool gpu_rows = false;
  const uint32_t *gpu_neighbors = nullptr;
  const int64_t *gpu_offsets = nullptr;
  int gpu_nq = 0;
  CrossNeighborSearchData cross;
  IntBuffer fixed_neighbors;
  IntBuffer fixed_counts;
  // Max row count of the fixed rows, known from the writer; -1 forces the
  // stats pass to rescan (paths that do not track it).
  int fixed_max_count = -1;
  // Sparse target-major cross rows. The counts buffer is kept across
  // invalidations for pairs that use that writer; only the rows it wrote are
  // cleared before the next build.
  std::vector<int> incremental_rows_touched;
  bool incremental_rows_touched_overflow = false;
  bool incremental_rows_owned = false;
  // Per-block cross rows. The block capacities and row offsets depend only
  // on the source/target grids, so they survive a pair invalidation that
  // merely rewrites unchanged rows.
  bool cross_block_valid = false;
  bool cross_block_active = false;
  int cross_block_size = 0;
  int cross_block_capacity = 0;
  uint64_t cross_block_source_generation = 0;
  uint64_t cross_block_target_generation = 0;
  int cross_block_source_points = 0;
  int cross_block_target_points = 0;
  IntBuffer cross_row_offset;
  IntBuffer cross_row_capacity;
  // Hybrid cross rows: rows up to the base capacity use a fixed stride, rows
  // above it get a per-row offset past the base area. This removes the
  // padding to the global maximum without a per-row load on common rows.
  bool cross_hybrid_valid = false;
  bool cross_hybrid_active = false;
  int cross_hybrid_base = 0;
  IntBuffer cross_hybrid_offset;
  IntBuffer cross_hybrid_capacity;
  // Frames to wait before re-testing a pair whose counts did not fit a
  // hybrid base; the scan is wasted on geometry that cannot use it.
  int cross_hybrid_check_wait = 0;
};

// GPU backend state of one NSearch (device buffers, result slots, cached
// grids). Created on the first GPU search.
struct NSearch::GpuState {
#ifdef APFRNN_HAVE_SYCL
  ApfrnnSyclContext *context = apfrnn_sycl_create();
  GpuState() = default;
  GpuState(const GpuState &) = delete;
  GpuState &operator=(const GpuState &) = delete;
  ~GpuState() { apfrnn_sycl_destroy(context); }
#endif
  uint64_t next_version = 1;
};

NSearch::NSearch(float radius)
    : radius_(radius),
      search_backend_(search_backend_from_env()) {
  if (search_backend_ == SearchBackend::GpuExact && !gpu_available()) {
    search_backend_ = SearchBackend::CpuExact;
  }
}

NSearch::~NSearch() = default;

int NSearch::n_sets() const { return static_cast<int>(sets_.size()); }

bool NSearch::valid_set(int set) const {
  return set >= 0 && set < static_cast<int>(sets_.size());
}

float NSearch::sync_source(int set_index, bool skip_positions) {
  PointSet &set = sets_[static_cast<std::size_t>(set_index)];
  const int n = set.source_count;
  const bool comparable =
      set.valid && set.x.size() == static_cast<std::size_t>(n);

  if (n <= 0 || set.source == nullptr ||
      set.source_type == SourceType::None) {
    const float max_distance_sq =
        set.x.empty() ? 0.0f : std::numeric_limits<float>::infinity();
    set.x.clear();
    set.y.clear();
    set.z.clear();
    set.bounds = Bounds{};
    set.sweep.valid = false;
    set.sweep_decision_valid = false;
    return max_distance_sq;
  }

  if (skip_positions) {
    // Grid builds read the caller's interleaved buffer directly, so the SoA
    // copy is not needed. Keep the vectors sized for the API's index checks;
    // the exact bounds are computed in the grid key pass.
    if (set.x.size() != static_cast<std::size_t>(n)) {
      set.x.resize(static_cast<std::size_t>(n));
      set.y.resize(static_cast<std::size_t>(n));
      set.z.resize(static_cast<std::size_t>(n));
    }
    set.bounds = Bounds{};
    set.sweep.valid = false;
    set.sweep_decision_valid = false;
    return std::numeric_limits<float>::infinity();
  }
  if (!comparable) {
    set.x.resize(static_cast<std::size_t>(n));
    set.y.resize(static_cast<std::size_t>(n));
    set.z.resize(static_cast<std::size_t>(n));
  }
  const SyncResult result =
      set.source_type == SourceType::Float
          ? sync_points(static_cast<const float *>(set.source), n, comparable,
                        set.x, set.y, set.z)
          : sync_points(static_cast<const double *>(set.source), n,
                        comparable, set.x, set.y, set.z);
  set.bounds = result.bounds;
  const float max_distance_sq = comparable
                                    ? result.max_distance_sq
                                    : std::numeric_limits<float>::infinity();
  if (max_distance_sq > 0.0f) {
    // Both the sweep x order and the packed GPU copy are stale once any
    // point moved. The sweep/grid choice may change as well.
    set.sweep.valid = false;
    set.sweep_decision_valid = false;
    set.gpu_xyz_valid = false;
  }
  return max_distance_sq;
}

int NSearch::add_point_set(const float *xyz, int n, bool is_static) {
  return add_source(xyz, false, n, is_static);
}

int NSearch::add_point_set(const double *xyz, int n, bool is_static) {
  return add_source(xyz, true, n, is_static);
}

int NSearch::add_source(const void *xyz, bool is_double, int n,
                        bool is_static) {
  row_views_valid_ = false;
  if (n < 0 || (n > 0 && xyz == nullptr)) {
    return -1;
  }

  const int set_index = static_cast<int>(sets_.size());
  sets_.emplace_back();
  PointSet &set = sets_.back();
  set.is_static = is_static;
  set.source_type = is_double ? SourceType::Double : SourceType::Float;
  set.source = xyz;
  set.source_count = n;
  set.dirty = true;
  set.valid = false;

  const int new_set_count = static_cast<int>(sets_.size());
  for (auto &row : pairs_) {
    row.resize(static_cast<std::size_t>(new_set_count));
  }
  pairs_.resize(static_cast<std::size_t>(new_set_count));
  pairs_.back().resize(static_cast<std::size_t>(new_set_count));

  return set_index;
}

void NSearch::resize_point_set(int set, const float *xyz, int n) {
  resize_source(set, xyz, false, n);
}

void NSearch::resize_point_set(int set, const double *xyz, int n) {
  resize_source(set, xyz, true, n);
}

void NSearch::resize_source(int set, const void *xyz, bool is_double, int n) {
  row_views_valid_ = false;
  if (!valid_set(set) || n < 0 || (n > 0 && xyz == nullptr)) {
    return;
  }

  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  point_set.source_type = is_double ? SourceType::Double : SourceType::Float;
  point_set.source = xyz;
  point_set.source_count = n;
  point_set.dirty = true;
  point_set.valid = false;
  point_set.sweep_decision_valid = false;
  point_set.probe_valid = false;
  point_set.probe_age = 0;
  point_set.gpu_xyz_valid = false;
  point_set.last_fixed_capacity = 0;
  invalidate_pairs_for_set(set);
  for (int other = 0; other < static_cast<int>(sets_.size()); ++other) {
    pairs_[static_cast<std::size_t>(set)][static_cast<std::size_t>(other)]
        .last_fixed_capacity = 0;
    pairs_[static_cast<std::size_t>(other)][static_cast<std::size_t>(set)]
        .last_fixed_capacity = 0;
  }
}

void NSearch::set_point_set_static(int set, bool is_static) {
  row_views_valid_ = false;
  if (!valid_set(set)) {
    return;
  }

  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  if (point_set.is_static == is_static) {
    return;
  }

  point_set.is_static = is_static;
  point_set.dirty = true;
  invalidate_pairs_for_set(set);
}

void NSearch::set_active_search(int set_i, int set_j, bool active) {
  row_views_valid_ = false;
  if (!valid_set(set_i) || !valid_set(set_j)) {
    return;
  }

  Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                     [static_cast<std::size_t>(set_j)];
  pair.active = active;
}

void NSearch::set_all_searches(bool active) {
  row_views_valid_ = false;
  for (auto &row : pairs_) {
    for (Pair &pair : row) {
      pair.active = active;
    }
  }
}

void NSearch::prepare_zsort() {
  row_views_valid_ = false;
  zsort_order_.assign(sets_.size(), {});
  for (int set = 0; set < static_cast<int>(sets_.size()); ++set) {
    PointSet &point_set = sets_[static_cast<std::size_t>(set)];
    const IntBuffer &order = point_set.data.original_index;
    const std::size_t n = static_cast<std::size_t>(point_set.source_count);
    if (!point_set.valid || point_set.use_sweep || n == 0 ||
        order.size() != n) {
      continue;
    }
    bool in_order = true;
    for (std::size_t i = 0; i < n && in_order; ++i) {
      in_order = order[i] == static_cast<int>(i);
    }
    if (in_order) {
      continue;
    }
    zsort_order_[static_cast<std::size_t>(set)].assign(order.begin(),
                                                       order.end());
    // The caller is about to permute this set's points: every position
    // changes, also for a static set, and lists by the old indices are void.
    point_set.dirty = true;
    point_set.valid = false;
    invalidate_pairs_for_set(set);
    // Per-point capacities follow their points.
    if (point_set.moving_caps.size() == n) {
      IntBuffer moved(n);
      for (std::size_t i = 0; i < n; ++i) {
        moved[i] = point_set.moving_caps[static_cast<std::size_t>(order[i])];
      }
      point_set.moving_caps.swap(moved);
    }
  }
}

void NSearch::update_point_sets() {
  row_views_valid_ = false;
  for (int set = 0; set < static_cast<int>(sets_.size()); ++set) {
    if (!sets_[static_cast<std::size_t>(set)].is_static) {
      sets_[static_cast<std::size_t>(set)].dirty = true;
      invalidate_pairs_for_set(set);
    }
  }
}

const char *NSearch::path_name(uint64_t path) {
  switch (path) {
  case kPathGridDense: return "grid_dense";
  case kPathGridHash: return "grid_hash";
  case kPathSweep: return "sweep";
  case kPathSmallCells: return "small_cells";
  case kPathRangeCacheBuilt: return "range_cache_built";
  case kPathRangeCacheShared: return "range_cache_shared";
  case kPathTargetReuse: return "target_reuse";
  case kPathParallelBuild: return "parallel_build";
  case kPathParallelPairs: return "parallel_pairs";
  case kPathSameSweep: return "same_sweep";
  case kPathSameBlockRows: return "same_block_rows";
  case kPathSameIsolated: return "same_isolated";
  case kPathSameRangesDenseFused: return "same_ranges_dense_fused";
  case kPathSameRangesMerged: return "same_ranges_merged";
  case kPathSamePairDense: return "same_pair_dense";
  case kPathSameBlocked: return "same_blocked";
  case kPathSameQueryOct: return "same_query_oct";
  case kPathSameActive2: return "same_active2";
  case kPathSameGrown: return "same_grown";
  case kPathSameCsr: return "same_csr";
  case kPathSameSparseKeys: return "same_sparse_keys";
  case kPathCrossSweep: return "cross_sweep";
  case kPathCrossNoCandidates: return "cross_no_candidates";
  case kPathCrossTargetMajor: return "cross_target_major";
  case kPathCrossPairDense: return "cross_pair_dense";
  case kPathCrossQueryPair: return "cross_query_pair";
  case kPathCrossBlocked: return "cross_blocked";
  case kPathCrossActive2: return "cross_active2";
  case kPathCrossGrown: return "cross_grown";
  case kPathCrossCsr: return "cross_csr";
  case kPathCrossCellRange: return "cross_cell_range";
  case kPathGpu: return "gpu";
  }
  return "unknown";
}

void NSearch::run() {
  row_views_valid_ = false;
  run_frame();
  build_row_views();
}

void NSearch::run_frame() {
  last_stats_ = RunStats{};
  skipped_position_sync_ = false;
  path_bits_.store(0, std::memory_order_relaxed);
  if (sets_.empty()) {
    return;
  }

  const int set_count = static_cast<int>(sets_.size());
  update_cell_size();
  if (use_small_cells_) {
    note_path(kPathSmallCells);
  }
  last_stats_.pair_ms.assign(static_cast<std::size_t>(set_count) *
                                 static_cast<std::size_t>(set_count),
                             0.0);

  std::vector<bool> is_source(static_cast<std::size_t>(set_count), false);
  std::vector<bool> needs_same_set(static_cast<std::size_t>(set_count), false);
  for (int set = 0; set < set_count; ++set) {
    const std::vector<Pair> &row = pairs_[static_cast<std::size_t>(set)];
    is_source[static_cast<std::size_t>(set)] =
        std::any_of(row.begin(), row.end(),
                    [](const Pair &pair) { return pair.active; });
    needs_same_set[static_cast<std::size_t>(set)] =
        row[static_cast<std::size_t>(set)].active;
  }
  // How many pairs read each set's ranges as a target; decides whether the
  // grid build shares one range cache between them.
  for (int target = 0; target < set_count; ++target) {
    int uses = 0;
    for (int query = 0; query < set_count; ++query) {
      uses += pairs_[static_cast<std::size_t>(query)]
                    [static_cast<std::size_t>(target)]
                        .active
                  ? 1
                  : 0;
    }
    sets_[static_cast<std::size_t>(target)].data.range_cache_uses = uses;
  }

  sync_dynamic_sets(is_source);

  effective_backend_ = search_backend_;
  if (search_backend_ == SearchBackend::Auto) {
    // Static sets are normally synced inside the build loop below, but the
    // policy needs their coordinates before choosing a backend.
    for (int set = 0; set < set_count; ++set) {
      PointSet &point_set = sets_[static_cast<std::size_t>(set)];
      if (point_set.is_static && (!point_set.valid || point_set.dirty)) {
        sync_source(set);
      }
    }
    effective_backend_ = (gpu_available() && gpu_policy_ok())
                             ? SearchBackend::GpuExact
                             : SearchBackend::CpuExact;
  }

  const auto build_start = std::chrono::steady_clock::now();
  build_point_sets(is_source, needs_same_set);
  last_stats_.sync_build_ms = elapsed_ms(build_start);

  // Cross-sweep pairs benefit from an x-sorted source. Build that index once,
  // serially, before the pair loop: pairs are built in parallel and two pairs
  // sharing a source would otherwise race on the index.
  {
    std::vector<char> needs_source_index(static_cast<std::size_t>(set_count),
                                         0);
    std::vector<char> needs_target_index(static_cast<std::size_t>(set_count),
                                         0);
    for (int set_i = 0; set_i < set_count; ++set_i) {
      for (int set_j = 0; set_j < set_count; ++set_j) {
        if (set_i == set_j) {
          continue;
        }
        const Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                                 [static_cast<std::size_t>(set_j)];
        if (!pair.active || pair.valid) {
          continue;
        }
        PointSet &source = sets_[static_cast<std::size_t>(set_i)];
        PointSet &target = sets_[static_cast<std::size_t>(set_j)];
        if (!source.use_sweep && target.use_sweep &&
            target.x.size() <= 1000000 && target.bounds.min_x >= 0.0f &&
            target.bounds.min_y >= 0.0f && target.bounds.min_z >= 0.0f) {
          ensure_grid(target);
        }
        const bool sparse_target =
            target.data.num_points > 0 && !target.data.dense_valid;
        const bool short_window =
            sparse_target &&
            sweep::estimated_window(target.bounds, target.data.num_points,
                                    radius_) <= 256.0;
        bool cell_range = cell_range_cross_candidate(source, target);
        if (cell_range) {
          ensure_cell_x_index(target.data);
          if (target.data.num_points >= 100000) {
            ensure_occupancy(target.data);
          }
          cell_range = target.data.cell_x_valid;
        }
        if (!cell_range && (target.use_sweep || short_window)) {
          needs_source_index[static_cast<std::size_t>(set_i)] = 1;
          needs_target_index[static_cast<std::size_t>(set_j)] = 1;
        }
      }
    }
    for (int set = 0; set < set_count; ++set) {
      PointSet &point_set = sets_[static_cast<std::size_t>(set)];
      const bool source =
          needs_source_index[static_cast<std::size_t>(set)] != 0;
      const bool target =
          needs_target_index[static_cast<std::size_t>(set)] != 0;
      if ((source || target) && !point_set.use_sweep) {
        if (!point_set.sweep.valid && point_set.valid &&
            (point_set.data.cell_keys_signed_order ||
             point_set.bounds.min_x >= 0.0f) &&
            point_set.data.num_points > 0 &&
            point_set.data.num_cells == point_set.data.num_points &&
            point_set.data.sorted_x.size() ==
                static_cast<std::size_t>(point_set.data.num_points)) {
          // One point per cell means the cell-key order is strictly x-sorted;
          // the cross sweep can reuse the grid order instead of sorting the
          // moving set by x again.
          sweep::adopt_sorted_order(point_set.sweep, point_set.data.sorted_x,
                                    point_set.data.sorted_y,
                                    point_set.data.sorted_z,
                                    point_set.data.original_index,
                                    point_set.data.scratch_unique_keys);
        } else {
          sweep::ensure_index(point_set.sweep, point_set.x, point_set.y,
                              point_set.z);
        }
      }
    }
  }

  // Sparse target-major cross pairs: when the target is much smaller than the
  // source and both are sparse unique-cell sets, iterate the target's points
  // against the source grid. This cuts per-query index overhead by the
  // source/target size ratio while producing the same pair set.
  {
    const auto target_major_start = std::chrono::steady_clock::now();
    for (int set_i = 0; set_i < set_count; ++set_i) {
      PointSet &source = sets_[static_cast<std::size_t>(set_i)];
      if (source.use_sweep || !source.valid) {
        continue;
      }
      for (int set_j = 0; set_j < set_count; ++set_j) {
        if (set_j == set_i) {
          continue;
        }
        Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                           [static_cast<std::size_t>(set_j)];
        PointSet &target = sets_[static_cast<std::size_t>(set_j)];
        if (!pair.active || pair.valid) {
          continue;
        }
        if (target.use_sweep || !target.valid ||
            target.data.num_points * 2 > source.data.num_points) {
          continue;
        }
        // The searched source must be a sparse cell grid with a valid
        // order-space x-slab index. Multi-point cells are fine (their ranges
        // are materialized); the ordered index also supports negative
        // coordinates from jittered moving sets. Dense grids keep the
        // query-major writer, which amortizes the search over many neighbors.
        if (source.data.cell_stencil != 1 ||
            source.data.num_points < 2000000 ||
            static_cast<long long>(source.data.num_cells) * 4 <
                static_cast<long long>(source.data.num_points)) {
          continue;
        }
        ensure_ordered_cell_x(source.data);
        ensure_occupancy_cells(source.data);
        if (!source.data.ordered_x_valid || source.data.num_cells <= 0 ||
            source.data.scratch_unique_keys.size() !=
                static_cast<std::size_t>(source.data.num_cells)) {
          continue;
        }
        const float radius = radius_;
        for (int capacity : {1, 2, 4, 8, 16, 32}) {
          int max_count = 0;
          if (!write_cross_rows_target_major(
                  source.data, target.data, radius, capacity,
                  pair.fixed_neighbors, pair.fixed_counts,
                  pair.incremental_rows_touched,
                  pair.incremental_rows_touched_overflow, max_count)) {
            continue;
          }
          pair.incremental_rows_owned = true;
          pair.fixed_max_count = max_count;
          pair.fixed_valid = true;
          pair.fixed_capacity = capacity;
          pair.last_fixed_capacity =
              plan::cross_fitted_capacity(max_count, capacity);
          pair.cross_block_active = false;
          pair.same_set = false;
          pair.valid = true;
          pair.cross.radius = radius;
          note_path(kPathCrossTargetMajor);
          break;
        }
      }
    }
    last_stats_.cross_set_ms += elapsed_ms(target_major_start);
  }

  auto build_cpu_pair = [&](Pair &pair, int set_i, int set_j) {
    if (set_i == set_j) {
      build_same_pair(pair, set_i);
    } else {
      build_cross_pair(pair, set_i, set_j);
    }
  };

  int pair_count = 0;
  int max_pair_points = 0;
  for (int set_i = 0; set_i < set_count; ++set_i) {
    for (int set_j = 0; set_j < set_count; ++set_j) {
      const Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                               [static_cast<std::size_t>(set_j)];
      if (!pair.active || pair.valid) {
        continue;
      }
      ++pair_count;
      max_pair_points = std::max(
          max_pair_points,
          std::max(sets_[static_cast<std::size_t>(set_i)].data.num_points,
                   sets_[static_cast<std::size_t>(set_j)].data.num_points));
    }
  }

  // Small pairs each start their own TBB region, so a scene with many sets
  // spends more time waking workers than searching. Run those pairs as one
  // parallel job; large pairs keep the intra-pair parallelism.
  bool pair_parallel =
      effective_backend_ == SearchBackend::CpuExact && pair_count > 4 &&
      max_pair_points <= kPairParallelMaxPoints;
  if (pair_parallel) {
    for (const PointSet &point_set : sets_) {
      if (point_set.use_sweep) {
        pair_parallel = false;
        break;
      }
    }
  }
  if (pair_parallel) {
    note_path(kPathParallelPairs);
    struct Job {
      int set_i;
      int set_j;
    };
    std::vector<Job> jobs;
    jobs.reserve(static_cast<std::size_t>(pair_count));
    for (int set_i = 0; set_i < set_count; ++set_i) {
      for (int set_j = 0; set_j < set_count; ++set_j) {
        const Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                                 [static_cast<std::size_t>(set_j)];
        if (pair.active && !pair.valid) {
          jobs.push_back(Job{set_i, set_j});
        }
      }
    }
    for (const Job &job : jobs) {
      ensure_point_blocks(sets_[static_cast<std::size_t>(job.set_j)].data);
    }

    std::vector<double> pair_times(jobs.size(), 0.0);
    parallel_pairs_ = true;
    parallel_overflows_.store(0, std::memory_order_relaxed);
    // Same-set pairs establish each source's fitted capacity and write the
    // point-set rows. They touch one set each, so run them in parallel, then
    // let cross pairs read the settled sets.
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, jobs.size(), 1),
        [&](const tbb::blocked_range<std::size_t> &range) {
          for (std::size_t index = range.begin(); index < range.end();
               ++index) {
            const Job job = jobs[index];
            if (job.set_i != job.set_j) {
              continue;
            }
            Pair &pair = pairs_[static_cast<std::size_t>(job.set_i)]
                               [static_cast<std::size_t>(job.set_j)];
            const auto pair_start = std::chrono::steady_clock::now();
            build_cpu_pair(pair, job.set_i, job.set_j);
            pair_times[index] = elapsed_ms(pair_start);
          }
        });
    // Cross pairs only read their sets' grids and ranges: every set that is
    // an active target has its range cache built before this loop, and the
    // same-set phase has materialized the cell ranges. Run all pending pairs
    // at once instead of serializing every pair that shares a set.
    for (int set = 0; set < set_count; ++set) {
      PointSet &point_set = sets_[static_cast<std::size_t>(set)];
      if (!point_set.valid) {
        continue;
      }
      ensure_sorted_coords(point_set.data);
      ensure_cell_ranges(point_set.data);
      if (!point_set.data.dense_valid) {
        ensure_hash_table(point_set.data);
      }
    }
    std::vector<std::size_t> pending;
    for (std::size_t index = 0; index < jobs.size(); ++index) {
      if (jobs[index].set_i != jobs[index].set_j) {
        pending.push_back(index);
      }
    }
    // All-to-all scenes test the same point pairs for (i,j) and (j,i). When
    // both directions use equal-size dense sets and fixed rows, build one
    // direction and transpose its rows into the other.
    auto can_transpose = [&](int i, int j) {
      const PointSet &a = sets_[static_cast<std::size_t>(i)];
      const PointSet &b = sets_[static_cast<std::size_t>(j)];
      if (!a.valid || !b.valid || a.use_sweep || b.use_sweep ||
          !a.data.dense_valid || !b.data.dense_valid ||
          a.data.cell_stencil != 1 || b.data.cell_stencil != 1 ||
          a.data.cell_radius != b.data.cell_radius ||
          a.data.num_points != b.data.num_points ||
          a.data.num_points < 512 ||
          a.data.num_points >= kCrossBlockMinPoints) {
        return false;
      }
      const Pair &forward = pairs_[static_cast<std::size_t>(i)]
                                  [static_cast<std::size_t>(j)];
      const Pair &reverse = pairs_[static_cast<std::size_t>(j)]
                                  [static_cast<std::size_t>(i)];
      if (!forward.active || !reverse.active || forward.valid ||
          reverse.valid) {
        return false;
      }
      const plan::CrossPlan forward_plan = plan::cross_set(
          a.data, a.last_fixed_capacity, b.data, forward.last_fixed_capacity,
          memory_budget_);
      const plan::CrossPlan reverse_plan = plan::cross_set(
          b.data, b.last_fixed_capacity, a.data, reverse.last_fixed_capacity,
          memory_budget_);
      return forward_plan.fixed_rows && reverse_plan.fixed_rows &&
             forward_plan.first_capacity > 0 &&
             reverse_plan.first_capacity > 0;
    };
    std::vector<int> transpose_reverse(jobs.size(), -1);
    std::vector<char> transposed(jobs.size(), 0);
    {
      std::vector<int> index(static_cast<std::size_t>(set_count) *
                                 static_cast<std::size_t>(set_count),
                             -1);
      for (std::size_t k = 0; k < jobs.size(); ++k) {
        index[static_cast<std::size_t>(jobs[k].set_i) *
                  static_cast<std::size_t>(set_count) +
              static_cast<std::size_t>(jobs[k].set_j)] = static_cast<int>(k);
      }
      for (int i = 0; i < set_count; ++i) {
        for (int j = i + 1; j < set_count; ++j) {
          const int forward =
              index[static_cast<std::size_t>(i) *
                        static_cast<std::size_t>(set_count) +
                    static_cast<std::size_t>(j)];
          const int reverse =
              index[static_cast<std::size_t>(j) *
                        static_cast<std::size_t>(set_count) +
                    static_cast<std::size_t>(i)];
          if (forward < 0 || reverse < 0 || !can_transpose(i, j)) {
            continue;
          }
          transpose_reverse[static_cast<std::size_t>(forward)] = reverse;
          transposed[static_cast<std::size_t>(reverse)] = 1;
        }
      }
    }
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, pending.size(), 1),
        [&](const tbb::blocked_range<std::size_t> &range) {
          for (std::size_t slot = range.begin(); slot < range.end(); ++slot) {
            const std::size_t index = pending[slot];
            if (transposed[index]) {
              continue;
            }
            const Job job = jobs[index];
            Pair &pair = pairs_[static_cast<std::size_t>(job.set_i)]
                               [static_cast<std::size_t>(job.set_j)];
            const auto pair_start = std::chrono::steady_clock::now();
            build_cpu_pair(pair, job.set_i, job.set_j);
            pair_times[index] = elapsed_ms(pair_start);
            const int reverse_index = transpose_reverse[index];
            if (reverse_index >= 0) {
              const Job reverse_job = jobs[static_cast<std::size_t>(
                  reverse_index)];
              Pair &reverse = pairs_[static_cast<std::size_t>(
                                         reverse_job.set_i)]
                                     [static_cast<std::size_t>(
                                         reverse_job.set_j)];
              const auto transpose_start = std::chrono::steady_clock::now();
              if (!transpose_cross_pair(
                      pair, reverse, sets_[static_cast<std::size_t>(job.set_i)],
                      sets_[static_cast<std::size_t>(job.set_j)])) {
                build_cpu_pair(reverse, reverse_job.set_i, reverse_job.set_j);
              }
              pair_times[static_cast<std::size_t>(reverse_index)] =
                  elapsed_ms(transpose_start);
            }
          }
        });
    parallel_pairs_ = false;
    last_stats_.fixed_overflows += parallel_overflows_.exchange(0);

    for (std::size_t index = 0; index < jobs.size(); ++index) {
      const Job job = jobs[index];
      const double pair_ms = pair_times[index];
      if (job.set_i == job.set_j) {
        last_stats_.same_set_ms += pair_ms;
      } else {
        last_stats_.cross_set_ms += pair_ms;
      }
      last_stats_.pair_ms[static_cast<std::size_t>(job.set_i) *
                              static_cast<std::size_t>(set_count) +
                          static_cast<std::size_t>(job.set_j)] += pair_ms;
    }
  } else {
    // A pair that converts a sweep set to a grid invalidates that set's
    // earlier pairs (see ensure_grid), so repeat until every active pair is
    // valid. Each set converts at most once per run.
    for (bool built = true; built;) {
      built = false;
      for (int set_i = 0; set_i < set_count; ++set_i) {
        for (int set_j = 0; set_j < set_count; ++set_j) {
          Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                             [static_cast<std::size_t>(set_j)];
          if (!pair.active || pair.valid) {
            continue;
          }
          built = true;

          const auto pair_start = std::chrono::steady_clock::now();
          if (effective_backend_ == SearchBackend::GpuExact) {
            const bool defer_same = set_i == set_j;
            if (build_gpu_pair(pair, set_i, set_j, defer_same)) {
              note_path(kPathGpu);
            } else {
              // The GPU path is an opt-in accelerator; on any failure restart the
              // run on the canonical CPU Exact path. Drain any copy already
              // queued for an earlier pair first.
  #ifdef APFRNN_HAVE_SYCL
              apfrnn_sycl_wait(gpu_ ? gpu_->context : nullptr);
  #endif
              set_search_backend(SearchBackend::CpuExact);
              run();
              return;
            }
          } else {
            build_cpu_pair(pair, set_i, set_j);
          }

          const double pair_ms = elapsed_ms(pair_start);
          if (set_i == set_j) {
            last_stats_.same_set_ms += pair_ms;
          } else {
            last_stats_.cross_set_ms += pair_ms;
          }
          last_stats_.pair_ms[static_cast<std::size_t>(set_i) *
                                  static_cast<std::size_t>(set_count) +
                              static_cast<std::size_t>(set_j)] += pair_ms;
        }
      }
    }
  }

  collect_memory_stats();
  last_stats_.paths = path_bits_.load(std::memory_order_relaxed);
}

void NSearch::update_cell_size() {
  bool small_cells = false;
  for (const PointSet &point_set : sets_) {
    small_cells = small_cells ||
                  plan::wants_small_cells(point_set.data,
                                          point_set.last_fixed_capacity);
  }
  if (small_cells != use_small_cells_) {
    use_small_cells_ = small_cells;
    for (PointSet &point_set : sets_) {
      point_set.dirty = true;
      point_set.valid = false;
    }
  }
}

void NSearch::sync_dynamic_sets(const std::vector<bool> &is_source) {
  // Dynamic positions are synced on every run. A pure target set that did
  // not move keeps its grid (target cache).
  for (int set = 0; set < static_cast<int>(sets_.size()); ++set) {
    PointSet &point_set = sets_[static_cast<std::size_t>(set)];
    if (point_set.is_static) {
      continue;
    }
    // Large moving float source sets always build a grid (the sweep path is
    // disabled at this size), so the per-frame SoA copy is redundant: the key
    // pass reads the caller's interleaved buffer and computes the AABB.
    const bool skip_positions =
        search_backend_ == SearchBackend::CpuExact &&
        is_source[static_cast<std::size_t>(set)] &&
        point_set.source_type == SourceType::Float &&
        point_set.source != nullptr && point_set.source_count >= 1000000 &&
        !point_set.is_static;
    if (skip_positions) {
      skipped_position_sync_ = true;
    }
    const float max_distance_sq = sync_source(set, skip_positions);
    if (max_distance_sq > 0.0f) {
      // The packed GPU copy and the cached grid are stale once any point
      // moved. A static frame keeps both across runs.
      point_set.gpu_xyz_valid = false;
    }
    last_stats_.max_displacement_sq =
        std::max(last_stats_.max_displacement_sq, max_distance_sq);
    // An unchanged set keeps its sorted coordinates, grid,
    // ranges and rows. This is not just a target cache: a source set that did
    // not move needs no rebuild either.
    if (point_set.valid && max_distance_sq == 0.0f) {
      point_set.dirty = false;
      // Keep the unchanged grid, sorted order, ranges and cells. Source sets
      // still rewrite their rows so consumers always see a fresh run().
      point_set.refresh_rows =
          is_source[static_cast<std::size_t>(set)];
      ++last_stats_.reused_sets;
      note_path(kPathTargetReuse);
      continue;
    }
    const bool target_unchanged =
        !is_source[static_cast<std::size_t>(set)] &&
        point_set.valid && point_set.target_cache_valid &&
        point_set.cached_x == point_set.x &&
        point_set.cached_y == point_set.y && point_set.cached_z == point_set.z;
    if (target_unchanged) {
      point_set.dirty = false;
      ++last_stats_.reused_sets;
      note_path(kPathTargetReuse);
    } else {
      point_set.dirty = true;
      ++last_stats_.rebuilt_sets;
    }
  }
}

void NSearch::build_point_sets(const std::vector<bool> &is_source,
                               const std::vector<bool> &needs_same_set) {
  const int set_count = static_cast<int>(sets_.size());
  if (effective_backend_ == SearchBackend::GpuExact) {
    for (int set = 0; set < set_count; ++set) {
      PointSet &point_set = sets_[static_cast<std::size_t>(set)];
      if (point_set.is_static && (!point_set.valid || point_set.dirty)) {
        sync_source(set);
      }
      if (point_set.dirty || !point_set.valid || !point_set.is_static) {
        invalidate_pairs_for_set(set);
      }
      point_set.valid = true;
      point_set.dirty = false;
    }
    return;
  }

  std::vector<int> build_sets;
  std::vector<char> grid_rebuilt(static_cast<std::size_t>(set_count), 0);
  int max_points = 0;
  for (int set = 0; set < set_count; ++set) {
    const PointSet &point_set = sets_[static_cast<std::size_t>(set)];
    if (!point_set.dirty && point_set.valid && !point_set.refresh_rows) {
      continue;
    }
    build_sets.push_back(set);
    const int points =
        point_set.is_static ? point_set.source_count
                            : static_cast<int>(point_set.x.size());
    max_points = std::max(max_points, points);
  }
  for (int set : build_sets) {
    invalidate_pairs_for_set(set);
  }

  std::vector<double> grid_ms(static_cast<std::size_t>(set_count), 0.0);
  std::vector<double> same_ms(static_cast<std::size_t>(set_count), 0.0);
  auto build_one = [&](int set) {
    PointSet &point_set = sets_[static_cast<std::size_t>(set)];
    const bool rebuild_grid =
        !point_set.valid || point_set.dirty || !point_set.refresh_rows;
    if (rebuild_grid) {
      const auto grid_start = std::chrono::steady_clock::now();
      if (point_set.is_static) {
        sync_source(set);
      }
      build_point_grid(set);
      grid_ms[static_cast<std::size_t>(set)] = elapsed_ms(grid_start);
      grid_rebuilt[static_cast<std::size_t>(set)] = 1;
      if (!is_source[static_cast<std::size_t>(set)]) {
        point_set.cached_x = point_set.x;
        point_set.cached_y = point_set.y;
        point_set.cached_z = point_set.z;
        point_set.target_cache_valid = true;
      }
    }
    point_set.refresh_rows = false;
    if (needs_same_set[static_cast<std::size_t>(set)]) {
      const auto same_start = std::chrono::steady_clock::now();
      build_same_lists(set);
      same_ms[static_cast<std::size_t>(set)] = elapsed_ms(same_start);
    }
  };

  if (build_sets.size() > 1 && max_points <= 32768) {
    note_path(kPathParallelBuild);
    parallel_pairs_ = true;
    parallel_overflows_.store(0, std::memory_order_relaxed);
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, build_sets.size(), 1),
        [&](const tbb::blocked_range<std::size_t> &range) {
          for (std::size_t i = range.begin(); i < range.end(); ++i) {
            build_one(build_sets[i]);
          }
        });
    parallel_pairs_ = false;
    last_stats_.fixed_overflows += parallel_overflows_.exchange(0);
  } else {
    for (int set : build_sets) {
      build_one(set);
    }
  }
  // Counted after the builds: they may run in parallel, and a set's grid
  // kind is only known once it is built.
  for (int set : build_sets) {
    if (grid_rebuilt[static_cast<std::size_t>(set)] == 0) {
      continue;
    }
    const PointSet &point_set = sets_[static_cast<std::size_t>(set)];
    if (point_set.use_sweep) {
      ++last_stats_.sweep_sets;
      note_path(kPathSweep);
      continue;
    }
    ++last_stats_.grid_sets;
    last_stats_.grid_cells += point_set.data.num_cells;
    last_stats_.dense_cells +=
        static_cast<int>(point_set.data.dense_cells.size());
    if (point_set.data.dense_valid) {
      ++last_stats_.dense_sets;
      note_path(kPathGridDense);
    } else {
      ++last_stats_.hash_sets;
      note_path(kPathGridHash);
    }
    if (point_set.data.dense_cache_valid) {
      note_path(kPathRangeCacheBuilt);
    }
  }
  for (double ms : grid_ms) {
    last_stats_.grid_ms += ms;
  }
  for (double ms : same_ms) {
    last_stats_.same_build_ms += ms;
  }
}

void NSearch::build_same_pair(Pair &pair, int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  if (!point_set.same_lists_valid) {
    build_same_lists(set);
  }
  if (!point_set.fixed_valid) {
    write_neighbors_parallel(point_set.data);
  }
  pair.same_set = true;
  pair.valid = true;
}

// True when the exact sorted-cell range writer can replace the x-window
// sweep for this pair: a grid source and a sparse, non-negative grid target.
bool NSearch::cell_range_cross_candidate(const PointSet &source,
                                         const PointSet &target) {
  if (source.use_sweep || !source.valid || target.x.empty()) {
    return false;
  }
  if (target.bounds.min_x < 0.0f || target.bounds.min_y < 0.0f ||
      target.bounds.min_z < 0.0f) {
    return false;
  }
  if (target.use_sweep || !target.valid ||
      target.data.cell_stencil != 1 || target.data.max_cell_points > 1 ||
      target.data.num_cells <= 0 ||
      target.data.scratch_unique_keys.size() !=
          static_cast<std::size_t>(target.data.num_cells)) {
    return false;
  }
  const uint64_t *keys = target.data.scratch_unique_keys.data();
  const int span = static_cast<int>(
                       (keys[target.data.num_cells - 1] >> 42) & 0x1FFFFF) -
                   static_cast<int>((keys[0] >> 42) & 0x1FFFFF) + 1;
  return span > 0 && span <= 4 * target.data.num_cells + 65536;
}

void NSearch::build_cross_pair(Pair &pair, int set_i, int set_j) {
  PointSet &source = sets_[static_cast<std::size_t>(set_i)];
  PointSet &target = sets_[static_cast<std::size_t>(set_j)];
  pair.incremental_rows_owned = false;
  pair.fixed_valid = false;
  pair.fixed_max_count = -1;
  pair.same_set = false;
  pair.valid = true;

  if (cell_range_cross_candidate(source, target) &&
      target.data.cell_x_valid) {
    for (int capacity : {1, 2, 4, 8, 16, 32}) {
      int max_count = 0;
      if (!write_cross_rows_cell_range(
              source.data, target.data, radius_, capacity,
              pair.fixed_neighbors, pair.fixed_counts,
              pair.incremental_rows_touched,
              pair.incremental_rows_touched_overflow, max_count)) {
        continue;
      }
      pair.incremental_rows_owned = true;
      pair.fixed_max_count = max_count;
      pair.fixed_valid = true;
      pair.fixed_capacity = capacity;
      pair.last_fixed_capacity =
          plan::cross_fitted_capacity(max_count, capacity);
      pair.cross_block_active = false;
      note_path(kPathCrossCellRange);
      return;
    }
  }

  // Any writer below reads the source's sorted coordinates directly.
  if (!source.data.sorted_coords_valid) {
    ensure_sorted_coords(source.data);
  }
  const bool sparse_target =
      target.data.num_points > 0 && !target.data.dense_valid;
  const bool short_target_window =
      sparse_target &&
      sweep::estimated_window(target.bounds, target.data.num_points,
                              radius_) <= 256.0;
  if (source.use_sweep || target.use_sweep || short_target_window) {
    if (build_sweep_cross_lists(pair, source, target)) {
      note_path(kPathCrossSweep);
      return;
    }
    ensure_grid(source, &pair);
    ensure_grid(target, &pair);
  }

  if (plan::cross_wants_target_blocks(source.data, target.data)) {
    ensure_point_blocks(target.data);
  }
  const plan::CrossPlan cross_plan =
      plan::cross_set(source.data, source.last_fixed_capacity, target.data,
                      pair.last_fixed_capacity, memory_budget_);

  // Writers that skip the range build still read the pair's radius.
  pair.cross.radius = radius_;
  // A sparse target writes one id into ~0.85 source rows, so the fixed row
  // layout touches a fresh cache line per entry. The compact CSR layout keeps
  // the same ids in a few MB: count the rows exactly, then fill the packed
  // rows (see write_cross_neighbors_target_major_csr_parallel).
  if (cross_plan.fixed_rows &&
      cross_plan.writer == plan::CrossWriter::TargetMajor && source.data.num_points > 0 &&
      target.data.num_points > 0) {
    note_path(kPathCrossTargetMajor);
    pair.fixed_valid = false;
    pair.fixed_max_count = -1;
    pair.cross_block_active = false;
    pair.cross_block_valid = false;
    IntBuffer().swap(pair.fixed_neighbors);
    IntBuffer().swap(pair.fixed_counts);
    IntBuffer().swap(pair.cross_row_offset);
    IntBuffer().swap(pair.cross_row_capacity);
    write_cross_neighbors_target_major_csr_parallel(pair.cross, source.data,
                                                    target.data);
    pair.valid = true;
    return;
  }
  auto try_fixed = [&](int capacity) {
    switch (cross_plan.writer) {
    case plan::CrossWriter::TargetMajor:
      note_path(kPathCrossTargetMajor);
      write_cross_neighbors_target_major_parallel(
          pair.cross, source.data, target.data, capacity,
          pair.fixed_neighbors, pair.fixed_counts);
      break;
    case plan::CrossWriter::PairDense:
      note_path(kPathCrossPairDense);
      write_cross_neighbors_fixed_query_pair_dense_parallel(
          pair.cross, source.data, target.data, capacity,
          pair.fixed_neighbors, pair.fixed_counts);
      break;
    case plan::CrossWriter::QueryPair:
      note_path(kPathCrossQueryPair);
      write_cross_neighbors_fixed_query_pair_parallel(
          pair.cross, target.data, capacity, pair.fixed_neighbors,
          pair.fixed_counts);
      break;
    case plan::CrossWriter::Blocked:
      note_path(kPathCrossBlocked);
      write_cross_neighbors_fixed_query_blocked_parallel(
          pair.cross, target.data, source.data, capacity,
          pair.fixed_neighbors, pair.fixed_counts);
      break;
    case plan::CrossWriter::Active2:
      note_path(kPathCrossActive2);
      write_cross_neighbors_fixed_query_active2_parallel(
          pair.cross, target.data, capacity, pair.fixed_neighbors,
          pair.fixed_counts);
      break;
    }
    const int max_count = max_row_count(pair.fixed_counts);
    if (max_count > capacity) {
      return false;
    }
    pair.fixed_max_count = max_count;
    pair.fixed_valid = true;
    pair.cross_block_active = false;
    pair.fixed_capacity = capacity + row_slack();
    pair.last_fixed_capacity = plan::cross_fitted_capacity(max_count, capacity);
    return true;
  };

  if (cross_plan.fixed_rows) {
    if (cross_plan.build_ranges) {
      build_cross_cell_ranges_from_sorted_inplace(pair.cross, source.data,
                                                  target.data, radius_);
      if (pair.cross.shared_starts != nullptr) {
        note_path(kPathRangeCacheShared);
      }
      if (plan::cross_has_no_candidates(pair.cross)) {
        note_path(kPathCrossNoCandidates);
        pair.fixed_counts.assign(
            static_cast<std::size_t>(source.data.num_points), 0);
        pair.fixed_capacity = 1;
        pair.fixed_neighbors.resize(
            static_cast<std::size_t>(source.data.num_points));
        pair.fixed_valid = true;
        return;
      }
    }
    const bool pair_queries =
        cross_plan.writer == plan::CrossWriter::QueryPair ||
        cross_plan.writer == plan::CrossWriter::PairDense;
    const bool dense_ranges =
        cross_plan.writer == plan::CrossWriter::PairDense;
    const bool block_supported =
        source.data.num_points >= kCrossBlockMinPoints &&
        (cross_plan.writer == plan::CrossWriter::Active2 ||
         cross_plan.writer == plan::CrossWriter::QueryPair ||
         cross_plan.writer == plan::CrossWriter::PairDense);
    if (block_supported &&
        build_cross_block_rows(pair, source, target, pair_queries,
                               dense_ranges)) {
      return;
    }
    if (block_supported &&
        build_cross_hybrid_rows(pair, source, target, pair_queries,
                                dense_ranges)) {
      return;
    }
    if (try_fixed(cross_plan.first_capacity)) {
      remember_cross_block_rows(pair, source, target,
                                cross_plan.first_capacity);
      if (cross_plan.writer == plan::CrossWriter::PairDense) {
        seed_cross_hybrid(pair, source);
      }
      return;
    }
    // The failed fixed write still counted every row exactly, so its counts
    // can seed per-block rows without another counting pass. Compare against
    // the capacity the fixed path would need (not the small starting
    // capacity) so a layout that only wins because of padding is accepted.
    const int max_count = max_row_count(pair.fixed_counts);
    const int grown = plan::cross_grown_capacity(
        max_count, source.data.num_points, memory_budget_);
    if (block_supported) {
      const int compare_capacity =
          grown > 0 ? grown : ((max_count + 63) & ~63);
      if (remember_cross_block_rows(pair, source, target,
                                    compare_capacity) &&
          build_cross_block_rows(pair, source, target, pair_queries,
                                 dense_ranges)) {
        return;
      }
    }
    if (grown > 0) {
      note_path(kPathCrossGrown);
      if (try_fixed(grown)) {
        remember_cross_block_rows(pair, source, target, grown);
        if (cross_plan.writer == plan::CrossWriter::PairDense) {
          seed_cross_hybrid(pair, source);
        }
        return;
      }
    }
    if (parallel_pairs_) {
      parallel_overflows_.fetch_add(1, std::memory_order_relaxed);
    } else {
      ++last_stats_.fixed_overflows;
    }
    pair.fixed_neighbors.clear();
    pair.fixed_counts.clear();
  }

  note_path(kPathCrossCsr);
  pair.cross_block_active = false;
  pair.cross_block_valid = false;
  // The CSR path is the low-memory fallback; release the block-row layout and
  // cached capacities so it does not keep the memory it was meant to save.
  IntBuffer().swap(pair.cross_row_offset);
  IntBuffer().swap(pair.cross_row_capacity);
  build_cross_neighbor_search_data_from_sorted_inplace(
      pair.cross, source.data, target.data, radius_);
  write_cross_neighbors_parallel(pair.cross, target.data);
}

void NSearch::collect_memory_stats() {
  const int set_count = static_cast<int>(sets_.size());
  for (int set_i = 0; set_i < set_count; ++set_i) {
    const PointSet &source = sets_[static_cast<std::size_t>(set_i)];
    last_stats_.grid_bytes +=
        source.data.sorted_x.capacity() * sizeof(float) +
        source.data.sorted_y.capacity() * sizeof(float) +
        source.data.sorted_z.capacity() * sizeof(float) +
        source.data.original_index.capacity() * sizeof(int) +
        source.data.cell_starts.capacity() * sizeof(int) +
        source.data.cell_ends.capacity() * sizeof(int) +
        source.data.cell_neighbor_offset.capacity() * sizeof(int) +
        source.data.cell_num_neighbors.capacity() * sizeof(int) +
        source.data.cell_neighbor_starts.capacity() * sizeof(int) +
        source.data.cell_neighbor_ends.capacity() * sizeof(int) +
        source.data.hash_keys.capacity() * sizeof(uint64_t) +
        source.data.hash_vals.capacity() * sizeof(int) +
        source.data.dense_cells.capacity() * sizeof(int) +
        source.fixed_neighbors.capacity() * sizeof(int) +
        source.fixed_counts.capacity() * sizeof(int) +
        source.slot_counts.capacity() * sizeof(int) +
        source.point_offset.capacity() * sizeof(int);

    for (int set_j = 0; set_j < set_count; ++set_j) {
      const Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                               [static_cast<std::size_t>(set_j)];
      if (!pair.active || !pair.valid) {
        continue;
      }
      if (set_i == set_j) {
        if (source.fixed_valid) {
          ++last_stats_.same_fixed_pairs;
          if (source.block_rows) {
            ++last_stats_.same_block_pairs;
          }
          last_stats_.neighbor_bytes +=
              source.fixed_neighbors.capacity() * sizeof(int) +
              source.fixed_counts.capacity() * sizeof(int) +
              source.slot_counts.capacity() * sizeof(int);
          const int known = source.fixed_max_count;
          last_stats_.max_neighbors =
              std::max(last_stats_.max_neighbors,
                       known >= 0 ? known
                                  : max_row_count(source.fixed_counts));
        } else {
          ++last_stats_.same_csr_pairs;
          last_stats_.neighbor_bytes +=
              source.data.row_ptr.capacity() * sizeof(int) +
              source.data.col_idx.capacity() * sizeof(int);
          last_stats_.max_neighbors = std::max(
              last_stats_.max_neighbors, max_row_count(source.data.counts));
        }
      } else if (pair.fixed_valid) {
        ++last_stats_.cross_fixed_pairs;
        last_stats_.neighbor_bytes +=
            pair.fixed_neighbors.capacity() * sizeof(int) +
            pair.fixed_counts.capacity() * sizeof(int);
        if (pair.cross_hybrid_valid) {
          last_stats_.neighbor_bytes +=
              pair.cross_hybrid_offset.capacity() * sizeof(int) +
              pair.cross_hybrid_capacity.capacity() * sizeof(int);
        }
        const int known = pair.fixed_max_count;
        last_stats_.max_neighbors = std::max(
            last_stats_.max_neighbors,
            known >= 0 ? known : max_row_count(pair.fixed_counts));
      } else {
        ++last_stats_.cross_csr_pairs;
        last_stats_.neighbor_bytes +=
            pair.cross.row_ptr.capacity() * sizeof(int) +
            pair.cross.col_idx.capacity() * sizeof(int);
        last_stats_.max_neighbors = std::max(last_stats_.max_neighbors,
                                             max_row_count(pair.cross.counts));
      }
    }
  }
}

NeighborList NSearch::get_neighbors_checked(int set_i, int set_j,
                                            int point) const {
  if (!valid_set(set_i) || !valid_set(set_j)) {
    return {};
  }

  const Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                           [static_cast<std::size_t>(set_j)];
  if (!pair.active || !pair.valid) {
    return {};
  }

  const PointSet &source = sets_[static_cast<std::size_t>(set_i)];
  if (point < 0 || point >= static_cast<int>(source.x.size())) {
    return {};
  }

  if (pair.gpu_rows) {
#ifdef APFRNN_HAVE_SYCL
    if (gpu_) {
      apfrnn_sycl_wait(gpu_->context);
    }
#endif
    if (point >= pair.gpu_nq) {
      return {};
    }
    const int64_t begin = pair.gpu_offsets[point];
    const int64_t end = pair.gpu_offsets[point + 1];
    return {reinterpret_cast<const uint32_t *>(pair.gpu_neighbors + begin),
            static_cast<std::size_t>(end - begin)};
  }

  if (pair.same_set && source.fixed_valid &&
      (source.sparse_rows_by_original || source.rows_by_original)) {
    const IntBuffer &counts = source.rows_by_original ? source.slot_counts
                                                      : source.fixed_counts;
    const int count = counts[static_cast<std::size_t>(point)];
    const std::size_t start =
        source.rows_by_original && source.block_rows
            ? static_cast<std::size_t>(
                  source.point_offset[static_cast<std::size_t>(point)])
            : static_cast<std::size_t>(point) *
                  static_cast<std::size_t>(source.fixed_capacity);
    const int *neighbors = source.fixed_neighbors.data() + start;
    return {reinterpret_cast<const uint32_t *>(neighbors),
            static_cast<std::size_t>(count)};
  }

  // The source's sorted mapping is rebuilt by every grid build (and is the
  // identity for sweep sets), so it is valid for same-set and cross rows.
  const int sorted_index =
      source.original_to_sorted[static_cast<std::size_t>(point)];

  if (pair.same_set && source.fixed_valid &&
      (source.block_rows || source.packed_rows)) {
    const int count =
        source.fixed_counts[static_cast<std::size_t>(sorted_index)];
    const int *neighbors =
        source.fixed_neighbors.data() +
        static_cast<std::size_t>(
            source.row_offset[static_cast<std::size_t>(sorted_index)]);
    return {reinterpret_cast<const uint32_t *>(neighbors),
            static_cast<std::size_t>(count)};
  }

  if (pair.same_set && source.fixed_valid) {
    const int count =
        source.fixed_counts[static_cast<std::size_t>(sorted_index)];
    const int *neighbors =
        source.fixed_neighbors.data() +
        static_cast<std::size_t>(sorted_index) *
            static_cast<std::size_t>(source.fixed_capacity);
    return {reinterpret_cast<const uint32_t *>(neighbors),
            static_cast<std::size_t>(count)};
  }

  if (!pair.same_set && pair.fixed_valid) {
    const int count =
        pair.fixed_counts[static_cast<std::size_t>(sorted_index)];
    const int *neighbors;
    if (pair.cross_hybrid_active) {
      if (count <= pair.cross_hybrid_base) {
        neighbors = pair.fixed_neighbors.data() +
                    static_cast<std::size_t>(sorted_index) *
                        static_cast<std::size_t>(pair.cross_hybrid_base +
                                                 row_slack());
      } else {
        neighbors = pair.fixed_neighbors.data() +
                    static_cast<std::size_t>(
                        pair.cross_hybrid_offset[
                            static_cast<std::size_t>(sorted_index)]);
      }
    } else if (pair.cross_block_active) {
      neighbors = pair.fixed_neighbors.data() +
                  static_cast<std::size_t>(
                      pair.cross_row_offset[
                          static_cast<std::size_t>(sorted_index)]);
    } else {
      neighbors = pair.fixed_neighbors.data() +
                  static_cast<std::size_t>(sorted_index) *
                      static_cast<std::size_t>(pair.fixed_capacity);
    }
    return {reinterpret_cast<const uint32_t *>(neighbors),
            static_cast<std::size_t>(count)};
  }

  const IntBuffer *row_ptr =
      pair.same_set ? &source.data.row_ptr : &pair.cross.row_ptr;
  const IntBuffer *col_idx =
      pair.same_set ? &source.data.col_idx : &pair.cross.col_idx;
  const int begin = (*row_ptr)[static_cast<std::size_t>(sorted_index)];
  const int end = (*row_ptr)[static_cast<std::size_t>(sorted_index + 1)];
  return {reinterpret_cast<const uint32_t *>(col_idx->data() + begin),
          static_cast<std::size_t>(end - begin)};
}

// Mirrors get_neighbors_checked() for every pair: the same layout tests in
// the same order, so the inline reader returns the same rows.
void NSearch::build_row_views() {
  const int set_count = static_cast<int>(sets_.size());
  row_views_.assign(static_cast<std::size_t>(set_count) *
                        static_cast<std::size_t>(set_count),
                    detail::RowView{});
  view_sets_ = set_count;
  for (int set_i = 0; set_i < set_count; ++set_i) {
    const PointSet &source = sets_[static_cast<std::size_t>(set_i)];
    for (int set_j = 0; set_j < set_count; ++set_j) {
      const Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                               [static_cast<std::size_t>(set_j)];
      detail::RowView &view =
          row_views_[static_cast<std::size_t>(set_i) *
                         static_cast<std::size_t>(set_count) +
                     static_cast<std::size_t>(set_j)];
      if (!pair.active || !pair.valid) {
        view.kind = detail::RowView::Empty;
        continue;
      }
      if (pair.gpu_rows) {
        continue; // the full lookup waits for the device
      }
      view.kind = detail::RowView::Rows;
      view.points = static_cast<unsigned>(source.x.size());
      if (pair.same_set && source.fixed_valid &&
          (source.sparse_rows_by_original || source.rows_by_original)) {
        view.data = source.fixed_neighbors.data();
        view.counts = source.rows_by_original ? source.slot_counts.data()
                                              : source.fixed_counts.data();
        if (source.rows_by_original && source.block_rows) {
          view.offsets = source.point_offset.data();
        } else {
          view.stride = static_cast<std::size_t>(source.fixed_capacity);
        }
        continue;
      }
      view.to_sorted = source.original_to_sorted.data();
      if (pair.same_set && source.fixed_valid) {
        view.data = source.fixed_neighbors.data();
        view.counts = source.fixed_counts.data();
        if (source.block_rows || source.packed_rows) {
          view.offsets = source.row_offset.data();
        } else {
          view.stride = static_cast<std::size_t>(source.fixed_capacity);
        }
        continue;
      }
      if (!pair.same_set && pair.fixed_valid) {
        view.data = pair.fixed_neighbors.data();
        view.counts = pair.fixed_counts.data();
        if (pair.cross_hybrid_active) {
          view.offsets = pair.cross_hybrid_offset.data();
          view.fixed_limit = pair.cross_hybrid_base;
          view.stride = static_cast<std::size_t>(pair.cross_hybrid_base +
                                                 row_slack());
        } else if (pair.cross_block_active) {
          view.offsets = pair.cross_row_offset.data();
        } else {
          view.stride = static_cast<std::size_t>(pair.fixed_capacity);
        }
        continue;
      }
      view.data = pair.same_set ? source.data.col_idx.data()
                                : pair.cross.col_idx.data();
      view.offsets = pair.same_set ? source.data.row_ptr.data()
                                   : pair.cross.row_ptr.data();
    }
  }
  row_views_valid_ = true;
}

int NSearch::get_n_points(int set) const {
  if (!valid_set(set)) {
    return 0;
  }
  return sets_[static_cast<std::size_t>(set)].source_count;
}

void NSearch::set_radius(float radius) {
  row_views_valid_ = false;
  if (radius == radius_) {
    return;
  }

  radius_ = radius;
  for (int set = 0; set < static_cast<int>(sets_.size()); ++set) {
    sets_[static_cast<std::size_t>(set)].dirty = true;
    sets_[static_cast<std::size_t>(set)].valid = false;
    sets_[static_cast<std::size_t>(set)].sweep_decision_valid = false;
    invalidate_pairs_for_set(set);
  }
}

bool NSearch::gpu_available() const {
#ifdef APFRNN_HAVE_SYCL
  return apfrnn_sycl_available() != 0;
#else
  return false;
#endif
}

bool NSearch::set_search_backend(SearchBackend backend) {
  row_views_valid_ = false;
  if (backend == SearchBackend::GpuExact && !gpu_available()) {
    return false;
  }
  if (backend == search_backend_) {
    return true;
  }
  search_backend_ = backend;
  if (backend == SearchBackend::CpuExact) {
    gpu_.reset();  // frees device memory; GPU rows are dropped below
  }
  for (PointSet &set : sets_) {
    set.valid = false;
    set.dirty = true;
    set.probe_valid = false;
    set.probe_age = 0;
    set.fixed_valid = false;
    set.same_lists_valid = false;
  }
  for (std::vector<Pair> &row : pairs_) {
    for (Pair &pair : row) {
      pair.valid = false;
      pair.fixed_valid = false;
      pair.cross_block_active = false;
      pair.gpu_rows = false;
      pair.gpu_neighbors = nullptr;
      pair.gpu_offsets = nullptr;
      pair.gpu_nq = 0;
    }
  }
  return true;
}

void NSearch::ensure_grid(PointSet &point_set, const Pair *building) {
  if (!point_set.use_sweep) {
    return;
  }
  // Rows already built from this set this run (its same-set pair and cross
  // pairs with it as the source) are indexed by the sweep's identity order;
  // the grid below re-sorts the set, so they are rebuilt by run().
  const int set = static_cast<int>(&point_set - sets_.data());
  for (int other = 0; other < static_cast<int>(sets_.size()); ++other) {
    Pair &pair =
        pairs_[static_cast<std::size_t>(set)][static_cast<std::size_t>(other)];
    if (&pair != building && pair.valid) {
      clear_pair(pair);
    }
  }
  point_set.use_sweep = false;
  point_set.sweep.clear();
  // The sweep path replaced original_index with the identity, so any keys
  // saved by an earlier grid build do not describe this order.
  point_set.data.incremental_valid = false;
  const float grid_radius = radius_;
  const float cell_radius = use_small_cells_ ? 0.5f * grid_radius : 0.0f;
  const float *interleaved =
      point_set.source_type == SourceType::Float && point_set.source != nullptr
          ? static_cast<const float *>(point_set.source)
          : nullptr;
  Bounds world;
  const bool inverse_written = build_point_grid_inplace(
      point_set.data, point_set.x, point_set.y, point_set.z, grid_radius,
      cell_radius, interleaved, &world, &point_set.original_to_sorted,
      lazy_sorted_coords(point_set));
  if (!world.empty()) {
    point_set.bounds = world;
  }
  if (!inverse_written) {
    update_sorted_mapping(point_set.data, point_set.original_to_sorted);
  }
  point_set.valid = true;
  point_set.same_lists_valid = false;
  point_set.fixed_valid = false;
  ++point_set.grid_generation;
}

void NSearch::set_memory_budget(int64_t bytes) {
  row_views_valid_ = false;
  memory_budget_ = std::max<int64_t>(0, bytes);
}

bool NSearch::lazy_sorted_coords(const PointSet &point_set) const {
  if (effective_backend_ != SearchBackend::CpuExact) {
    return false;
  }
  if (point_set.source_type != SourceType::Float ||
      point_set.source == nullptr || point_set.data.dense_valid) {
    return false;
  }
  const int set =
      static_cast<int>(&point_set - sets_.data());
  if (set < 0 || set >= static_cast<int>(sets_.size())) {
    return false;
  }
  bool is_source = false;
  bool is_cross_target = false;
  const int set_count = static_cast<int>(sets_.size());
  for (int i = 0; i < set_count; ++i) {
    for (int j = 0; j < set_count; ++j) {
      if (!pairs_[static_cast<std::size_t>(i)]
                 [static_cast<std::size_t>(j)]
                     .active) {
        continue;
      }
      if (i == set) {
        is_source = true;
      }
      if (j == set && i != set) {
        is_cross_target = true;
      }
    }
  }
  return is_source && !is_cross_target;
}

void NSearch::build_point_grid(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  point_set.valid = false;
  point_set.same_lists_valid = false;
  point_set.fixed_valid = false;
  point_set.dirty = false;

  if (!point_set.sweep_decision_valid) {
    point_set.sweep_decision = sweep::should_use(
        point_set.x, point_set.y, point_set.z, point_set.bounds,
        radius_);
    point_set.sweep_decision_valid = true;
  }
  if (point_set.sweep_decision) {
    // Sweep sets keep the caller's order: the sorted mapping is the identity
    // and the grid arrays stay empty.
    NeighborSearchData &data = point_set.data;
    const int num_points = static_cast<int>(point_set.x.size());
    data.num_points = num_points;
    data.radius = radius_;
    data.num_cells = 0;
    for (auto *buffer :
         {&data.cell_starts, &data.cell_ends, &data.cell_neighbor_offset,
          &data.cell_num_neighbors, &data.cell_neighbor_starts,
          &data.cell_neighbor_ends, &data.counts, &data.col_idx,
          &data.row_ptr, &data.hash_vals, &data.dense_cells}) {
      buffer->clear();
    }
    data.sorted_x.clear();
    data.sorted_y.clear();
    data.sorted_z.clear();
    data.hash_keys.clear();
    data.scratch_unique_keys.clear();
    data.hash_mask = 0;
    data.hash_valid = false;
    data.dense_valid = false;
    data.cell_ranges_implicit = false;
    data.row_ptr.push_back(0);
    data.original_index.resize(static_cast<std::size_t>(num_points));
    std::iota(data.original_index.begin(), data.original_index.end(), 0);
    data.incremental_valid = false;
    point_set.original_to_sorted = data.original_index;
    sweep::ensure_index(point_set.sweep, point_set.x, point_set.y,
                        point_set.z);
    point_set.use_sweep = true;
    point_set.valid = true;
    ++point_set.grid_generation;
    return;
  }

  point_set.use_sweep = false;
  point_set.sweep.clear();
  const float grid_radius = radius_;
  const float cell_radius = use_small_cells_ ? 0.5f * grid_radius : 0.0f;
  const float *interleaved =
      point_set.source_type == SourceType::Float && point_set.source != nullptr
          ? static_cast<const float *>(point_set.source)
          : nullptr;
  Bounds world;
  const bool inverse_written = build_point_grid_inplace(
      point_set.data, point_set.x, point_set.y, point_set.z, grid_radius,
      cell_radius, interleaved, &world, &point_set.original_to_sorted,
      lazy_sorted_coords(point_set));
  if (!world.empty()) {
    point_set.bounds = world;
  }
  if (!inverse_written) {
    update_sorted_mapping(point_set.data, point_set.original_to_sorted);
  }
  point_set.valid = true;
  ++point_set.grid_generation;
}

void NSearch::build_same_lists(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  point_set.packed_rows = false;
  if (!point_set.valid) {
    return;
  }
  if (point_set.use_sweep) {
    // Sweep rows own the counts buffer; the sparse grid writer must not
    // reuse its touched-row state after this path.
    point_set.incremental_rows_owned = false;
    point_set.fixed_max_count = -1;
    if (build_sweep_same_lists(set)) {
      // Sweep rows use the plain fixed layout; get_neighbors would read them
      // through the block-row offsets of an earlier frame otherwise.
      point_set.block_rows = false;
      note_path(kPathSameSweep);
      return;
    }
    ensure_grid(point_set);
  }
  build_grid_same_lists(set);
}

namespace {

// Counts by original point from counts by sorted row, written front to back.
void gather_slot_counts(const IntBuffer &counts, const IntBuffer &to_sorted,
                        IntBuffer &slot_counts) {
  const int num_points = static_cast<int>(counts.size());
  slot_counts.resize(counts.size());
  tbb::parallel_for(tbb::blocked_range<int>(0, num_points, 1 << 15),
                    [&](const tbb::blocked_range<int> &range) {
                      for (int point = range.begin(); point != range.end();
                           ++point) {
                        slot_counts[static_cast<std::size_t>(point)] =
                            counts[static_cast<std::size_t>(to_sorted[
                                static_cast<std::size_t>(point)])];
                      }
                    });
}

} // namespace

void NSearch::build_grid_same_lists(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  if (!point_set.valid) {
    return;
  }
  NeighborSearchData &data = point_set.data;
  // A static set's grid is not rebuilt between frames, so restore the radius
  // in case an earlier writer changed it.
  data.radius = radius_;
  point_set.fixed_valid = false;
  point_set.sparse_rows_by_original = false;
  point_set.rows_by_original = false;
  point_set.fixed_max_count = -1;
  // Whether the counts buffer still holds only this writer's rows from the
  // previous frame; every other layout leaves it false.
  const bool sparse_rows_reuse = point_set.incremental_rows_owned;
  point_set.incremental_rows_owned = false;
  // A moving set is re-sorted on every grid rebuild, so cached per-row block
  // capacities no longer match their rows and nearly every frame overflows
  // and rewrites all rows; it uses fixed rows instead. The first build is
  // optimistic.
  const bool first_lists =
      point_set.lists_grid_generation == PointSet::kNoGeneration;
  const bool grid_rebuilt =
      point_set.lists_grid_generation != point_set.grid_generation;
  point_set.lists_grid_generation = point_set.grid_generation;
  const bool block_rows_ok = first_lists || !grid_rebuilt;
  // Cells hold at most a few points, so the sorted cell-key order indexes the
  // 3x3x3 stencil directly. Cells with more than one point are handled by a
  // direct scan of their neighborhood, which keeps the unique-cell walk. The
  // cursor walk's fixed cost dominates on small sets, which keep the planned
  // writer.
  if (grid_rebuilt && data.cell_stencil == 1 && data.max_cell_points <= 8 &&
      data.num_cells > 0 && data.num_points >= 2048 &&
      (data.num_points >= 4096 ||
       data.num_cells == data.num_points) &&
      static_cast<int64_t>(data.num_cells) * 4 >=
          static_cast<int64_t>(data.num_points) * 3 &&
      data.scratch_unique_keys.size() ==
          static_cast<std::size_t>(data.num_cells) &&
      (data.cell_ranges_implicit ||
       (data.cell_starts.size() == static_cast<std::size_t>(data.num_cells) &&
        data.cell_ends.size() == static_cast<std::size_t>(data.num_cells)))) {
    bool reuse_rows = sparse_rows_reuse;
    for (int capacity : {1, 2, 4, 8, 16, 32}) {
      int max_count = 0;
      if (!write_sparse_same_rows(
              data, capacity, point_set.fixed_neighbors,
              point_set.fixed_counts, point_set.incremental_rows_touched,
              point_set.incremental_rows_touched_overflow, reuse_rows,
              max_count)) {
        reuse_rows = true;
        continue;
      }
      point_set.incremental_rows_owned = true;
      point_set.fixed_max_count = max_count;
      point_set.sparse_rows_by_original = true;
      point_set.block_rows = false;
      point_set.fixed_capacity = capacity;
      point_set.last_fixed_capacity =
          plan::same_fitted_capacity(max_count, capacity);
      point_set.fixed_valid = true;
      point_set.same_lists_valid = true;
      note_path(kPathSameSparseKeys);
      return;
    }
  }
  // Sparse moving sets have almost no neighbors. Try a direct fixed row at a
  // tiny capacity before laying out per-block rows; the block layout rewrites
  // all 8M rows every frame even when every row is empty or singleton.
  if (grid_rebuilt && data.max_cell_points <= 1 && data.num_points > 4000000) {
    build_same_set_cell_ranges_inplace(data);
    for (int capacity : {1, 2, 4}) {
      write_neighbors_fixed_query_oct_parallel(
          data, capacity, point_set.fixed_neighbors, point_set.fixed_counts);
      const int max_count = max_row_count(point_set.fixed_counts);
      if (max_count > capacity) {
        continue;
      }
      point_set.fixed_max_count = max_count;
      point_set.block_rows = false;
      point_set.fixed_capacity = capacity + row_slack();
      point_set.last_fixed_capacity =
          plan::same_fitted_capacity(max_count, capacity);
      point_set.fixed_valid = true;
      point_set.same_lists_valid = true;
      note_path(kPathSameQueryOct);
      return;
    }
  }
  if (!data.sorted_coords_valid) {
    ensure_sorted_coords(data);
  }
  if (data.cell_ranges_implicit) {
    ensure_cell_ranges(data);
  }
  if (point_set.block_rows) {
    if (block_rows_ok && build_same_block_rows(set)) {
      note_path(kPathSameBlockRows);
      return;
    }
    point_set.block_rows = false;
    point_set.block_caps_valid = false;
    point_set.block_layout_valid = false;
  }
  if (grid_rebuilt && !first_lists) {
    if (build_moving_block_rows(set)) {
      note_path(kPathSameBlockRows);
      return;
    }
    // A resize invalidates every per-row capacity. Rebuilding a fixed table
    // at the old capacity is expensive for high-degree sets (the growing
    // Fountain/clustered case), so try compact block rows from fresh exact
    // counts before falling back to fixed rows.
    if (point_set.fixed_counts.size() !=
        static_cast<std::size_t>(data.num_points)) {
      if (build_same_block_rows(set, /*min_block=*/0,
                                /*counts_ready=*/false)) {
        note_path(kPathSameBlockRows);
        return;
      }
    }
  } else {
    point_set.moving_caps_valid = false;
  }
  const plan::SamePlan same_plan =
      plan::same_set(data, point_set.last_fixed_capacity, memory_budget_);

  // Per-block rows avoid padding every row to the set's maximum. They need
  // exact counts. On later frames the last
  // frame's counts decide; on the first build the counts are taken below.
  auto wants_block_rows = [&](int &min_block) {
    if (!block_rows_ok ||
        point_set.fixed_layout_generation == point_set.grid_generation) {
      return false;
    }
    min_block = plan::same_min_block(point_set.fixed_counts);
    if (plan::prefer_block_rows(
            plan::same_fixed_slots(point_set.fixed_counts),
            plan::same_block_slots(point_set.fixed_counts, min_block))) {
      return true;
    }
    point_set.fixed_layout_generation = point_set.grid_generation;
    return false;
  };
  int min_block = 0;
  if (same_plan.fixed_rows && point_set.last_fixed_capacity > 0 &&
      point_set.fixed_counts.size() ==
          static_cast<std::size_t>(data.num_points) &&
      wants_block_rows(min_block)) {
    if (build_same_block_rows(set, min_block)) {
      note_path(kPathSameBlockRows);
      return;
    }
  }

  auto try_fixed = [&](int capacity) {
    // Rows go to their original point's slot: the caller reads rows in its
    // own point order, which is unrelated to the grid's sorted order once
    // the points have moved.
    const RowPlacement placement{data.original_index.data()};
    bool placed = false;
    bool packed = false;
    switch (same_plan.writer) {
    case plan::SameWriter::PairDense:
      note_path(kPathSamePairDense);
      if (!write_neighbors_packed_dense_parallel(
              data, capacity, point_set.fixed_neighbors,
              point_set.fixed_counts, point_set.row_offset)) {
        return false;
      }
      packed = true;
      break;
    case plan::SameWriter::Blocked:
      note_path(kPathSameBlocked);
      write_neighbors_fixed_query_blocked_parallel(
          data, capacity, point_set.fixed_neighbors, point_set.fixed_counts);
      break;
    case plan::SameWriter::QueryOct:
      note_path(kPathSameQueryOct);
      placed = write_neighbors_fixed_query_oct_parallel(
          data, capacity, point_set.fixed_neighbors, point_set.fixed_counts,
          placement);
      break;
    case plan::SameWriter::Active2:
      note_path(kPathSameActive2);
      placed = write_neighbors_fixed_query_active2_parallel(
          data, capacity, point_set.fixed_neighbors, point_set.fixed_counts,
          placement);
      break;
    }
    const int max_count = max_row_count(point_set.fixed_counts);
    if (max_count > capacity) {
      return false;
    }
    point_set.fixed_max_count = max_count;
    point_set.fixed_capacity = capacity + row_slack();
    point_set.last_fixed_capacity =
        plan::same_fitted_capacity(max_count, capacity);
    point_set.block_rows = false;
    // Counts in slot order for the reader. A gather over the points writes
    // them front to back; storing them from the writer would add a scattered
    // store per row.
    if (placed) {
      gather_slot_counts(point_set.fixed_counts, point_set.original_to_sorted,
                         point_set.slot_counts);
    }
    point_set.rows_by_original = placed;
    point_set.packed_rows = packed;
    if (packed) {
      // row_offset now holds the packed starts, not a block layout.
      point_set.block_layout_valid = false;
    }
    point_set.fixed_valid = true;
    point_set.same_lists_valid = true;
    return true;
  };

  if (same_plan.fixed_rows) {
    switch (same_plan.ranges) {
    case plan::SameRanges::None:
      break;
    case plan::SameRanges::DenseFused:
      note_path(kPathSameRangesDenseFused);
      build_same_set_cell_ranges_dense_fused_inplace(data);
      break;
    case plan::SameRanges::Merged:
      note_path(kPathSameRangesMerged);
      build_same_set_cell_ranges_inplace(data);
      break;
    }
    if (data.shared_starts != nullptr) {
      note_path(kPathRangeCacheShared);
    }
    if (same_plan.ranges != plan::SameRanges::None &&
        plan::same_set_isolated(data)) {
      note_path(kPathSameIsolated);
      point_set.fixed_counts.assign(static_cast<std::size_t>(data.num_points),
                                    0);
      point_set.fixed_max_count = 0;
      point_set.fixed_capacity = 1;
      point_set.fixed_neighbors.resize(
          static_cast<std::size_t>(data.num_points));
      point_set.fixed_valid = true;
      point_set.same_lists_valid = true;
      return;
    }
    int first_capacity = same_plan.first_capacity;
    if (point_set.last_fixed_capacity == 0) {
      // First build of this set: count exactly instead of writing rows at a
      // guessed capacity, which leaves an oversized buffer resident or, on
      // overflow, rewrites every row padded to the maximum.
      count_same_set_neighbors_inplace(data);
      point_set.fixed_counts = data.counts;
      if (wants_block_rows(min_block) &&
          build_same_block_rows(set, min_block, /*counts_ready=*/true)) {
        note_path(kPathSameBlockRows);
        return;
      }
      first_capacity =
          plan::same_exact_capacity(max_row_count(point_set.fixed_counts),
                                    data.num_points, memory_budget_);
    }
    if (first_capacity > 0 && try_fixed(first_capacity)) {
      return;
    }
    const int grown = plan::same_grown_capacity(
        max_row_count(point_set.fixed_counts), data.num_points,
        memory_budget_);
    if (grown > 0) {
      note_path(kPathSameGrown);
      if (try_fixed(grown)) {
        return;
      }
    }
    if (parallel_pairs_) {
      parallel_overflows_.fetch_add(1, std::memory_order_relaxed);
    } else {
      ++last_stats_.fixed_overflows;
    }
    point_set.fixed_neighbors.clear();
    point_set.fixed_counts.clear();
  }

  if (build_same_block_rows(set)) {
    note_path(kPathSameBlockRows);
    return;
  }
  point_set.block_rows = false;

  note_path(kPathSameCsr);
  build_same_set_neighbor_lists_inplace(data, radius_);
  point_set.same_lists_valid = true;
}

// Block rows for a set whose grid was rebuilt since the last frame. Rows are
// re-sorted on every rebuild, so capacities are kept per original point:
// each frame gathers them into sorted order, lays the rows out with a scan
// and writes them. A point's degree changes slowly, so with the margin of
// plan::moving_row_capacity() few rows overflow; an overflow grows those
// points' capacities and rewrites the rows. Capacities far above a point's
// degree shrink for the next frame.
bool NSearch::build_moving_block_rows(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  NeighborSearchData &data = point_set.data;
  const int num_points = data.num_points;
  if (num_points == 0) {
    return false;
  }
  const std::size_t n = static_cast<std::size_t>(num_points);
  if (!point_set.moving_caps_valid) {
    // Decide from the last frame's counts: slot totals do not depend on the
    // row order. Sets that keep fixed rows are re-checked every 16 frames.
    if (point_set.fixed_counts.size() != n || point_set.moving_check_wait > 0) {
      point_set.moving_check_wait =
          std::max(0, point_set.moving_check_wait - 1);
      return false;
    }
    // When the budget cannot hold fixed rows, per-point rows are the only
    // row layout the plan can use, so keep them even when their saving over
    // the fixed layout is small.
    const plan::SamePlan same_plan =
        plan::same_set(data, point_set.last_fixed_capacity, memory_budget_);
    if (same_plan.fixed_rows &&
        !plan::prefer_block_rows(
            plan::same_fixed_slots(point_set.fixed_counts),
            plan::same_moving_slots(point_set.fixed_counts))) {
      point_set.moving_check_wait = 16;
      return false;
    }
    data.radius = radius_;
    count_same_set_neighbors_inplace(data);
    point_set.fixed_counts = data.counts;
    point_set.moving_caps.resize(n);
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_points, 1 << 14),
        [&](const tbb::blocked_range<int> &range) {
          for (int i = range.begin(); i < range.end(); ++i) {
            point_set
                .moving_caps[static_cast<std::size_t>(data.original_index[i])] =
                plan::moving_row_capacity(point_set.fixed_counts[i]);
          }
        });
    point_set.moving_caps_valid = true;
  } else if (point_set.moving_caps.size() != n) {
    // A growing set keeps the capacities of its existing points. Seed the
    // new points with the average capacity; the writer grows any that
    // overflow, so no exact recount is needed for the resize.
    const std::size_t old = point_set.moving_caps.size();
    if (old > n) {
      point_set.moving_caps.resize(n);
    } else {
      int64_t sum = 0;
      for (int capacity : point_set.moving_caps) {
        sum += capacity;
      }
      const int seed =
          old > 0 ? static_cast<int>(sum / static_cast<int64_t>(old)) : 32;
      point_set.moving_caps.resize(
          n, round_up_to_row_align(std::max(8, seed)));
    }
  }
  data.radius = radius_;
  point_set.row_offset.resize(n);
  point_set.row_capacity.resize(n);
  point_set.fixed_counts.resize(n);
  // The exact rows of a high-degree set can exceed the memory budget by
  // design: the library lets the output exceed it rather than fall back to
  // CSR, and the block-row path uses the same allowance. The per-point layout
  // also carries the degree margin, so its allowance includes that margin.
  int64_t row_budget = memory_budget_;
  if (memory_budget_ >= static_cast<int64_t>(64) << 20) {
    const int64_t exact_neighbors = tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, n, 1 << 14), int64_t{0},
        [&](const tbb::blocked_range<std::size_t> &range, int64_t acc) {
          for (std::size_t i = range.begin(); i < range.end(); ++i) {
            acc += point_set.fixed_counts[i];
          }
          return acc;
        },
        [](int64_t a, int64_t b) { return a + b; });
    row_budget =
        std::max(row_budget,
                 plan::block_output_budget(
                     exact_neighbors + exact_neighbors / 8, num_points));
  }
  // Attempt 0 lays out and writes every row; if its overflowed rows do not
  // fit in the buffer's headroom, attempt 1 lays out again with their grown
  // capacities.
  bool placed = false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    IntBuffer &values = point_set.fixed_neighbors;
    // Rows go in point order when the table is large enough for the order to
    // matter (judged by the buffer in use; a new buffer is never placed).
    const bool eligible = plan::place_rows_by_point(
        static_cast<int64_t>(values.capacity()) * 4);
    // Capacities, and with `offsets` the row starts, into sorted order for
    // the writer.
    auto gather_layout = [&](bool offsets) {
      return tbb::parallel_reduce(
          tbb::blocked_range<int>(0, num_points, 1 << 14), 0,
          [&](const tbb::blocked_range<int> &range, int acc) {
            for (int i = range.begin(); i < range.end(); ++i) {
              const std::size_t point =
                  static_cast<std::size_t>(data.original_index[i]);
              const int capacity = point_set.moving_caps[point];
              point_set.row_capacity[static_cast<std::size_t>(i)] = capacity;
              if (offsets) {
                point_set.row_offset[static_cast<std::size_t>(i)] =
                    point_set.point_offset[point];
              }
              acc = std::max(acc, capacity);
            }
            return acc;
          },
          [](int a, int b) { return std::max(a, b); });
    };
    // Row starts in sorted order by an exclusive scan; returns the slots.
    auto scan_sorted = [&] {
      return tbb::parallel_scan(
          tbb::blocked_range<int>(0, num_points, 1 << 14), int64_t{0},
          [&](const tbb::blocked_range<int> &range, int64_t sum, bool final) {
            for (int i = range.begin(); i < range.end(); ++i) {
              if (final) {
                point_set.row_offset[static_cast<std::size_t>(i)] =
                    static_cast<int>(sum);
              }
              sum += point_set.row_capacity[static_cast<std::size_t>(i)] +
                     row_slack();
            }
            return sum;
          },
          [](int64_t a, int64_t b) { return a + b; });
    };
    int max_capacity = 0;
    int64_t total = 0;
    if (eligible) {
      // Row starts in point order: the caller reads rows by its own points,
      // so rows laid out in that order are read front to back.
      point_set.point_offset.resize(n);
      total = tbb::parallel_scan(
          tbb::blocked_range<std::size_t>(0, n, 1 << 14), int64_t{0},
          [&](const tbb::blocked_range<std::size_t> &range, int64_t sum,
              bool final) {
            for (std::size_t point = range.begin(); point < range.end();
                 ++point) {
              if (final) {
                point_set.point_offset[point] = static_cast<int>(sum);
              }
              sum += point_set.moving_caps[point] + row_slack();
            }
            return sum;
          },
          [](int64_t a, int64_t b) { return a + b; });
    } else {
      max_capacity = gather_layout(false);
      total = scan_sorted();
    }
    const int64_t bytes = total * 4 + static_cast<int64_t>(num_points) *
                                          (eligible ? 3 : 2) * 4;
    if (total > INT_MAX || bytes > row_budget) {
      point_set.moving_caps_valid = false;
      return false;
    }
    const std::size_t slots = static_cast<std::size_t>(total);
    // Keep at least 1/16 headroom for overflowed rows: once the growing
    // capacities ate it, every frame fell back to a full relayout.
    const bool fresh = values.capacity() < slots + slots / 16 ||
                       values.capacity() > 2 * slots;
    if (fresh) {
      // Headroom, so the next frames' small growth reuses the buffer.
      IntBuffer replacement;
      replacement.reserve(slots + slots / 8);
      values.swap(replacement);
    }
    values.resize(slots);
    // A new buffer is laid out in sorted order, so the writer faults its
    // pages in address order (see prepare_fixed_rows).
    placed = eligible && !fresh;
    if (eligible) {
      max_capacity = gather_layout(placed);
      if (!placed) {
        scan_sorted();
      }
    }
    write_same_set_block_rows_parallel(
        data, max_capacity, point_set.fixed_counts, values,
        point_set.row_offset, point_set.row_capacity);
    // Grow the points whose rows overflowed and collect those rows; shrink
    // capacities far above the degree for the next frame (their rows are
    // complete now).
    using Rows = std::vector<int>;
    Rows overflowed = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, num_points, 1 << 14), Rows{},
        [&](const tbb::blocked_range<int> &range, Rows acc) {
          for (int i = range.begin(); i < range.end(); ++i) {
            const std::size_t row = static_cast<std::size_t>(i);
            const int count = point_set.fixed_counts[row];
            const int fitted = plan::moving_row_capacity(count);
            int &capacity = point_set.moving_caps[static_cast<std::size_t>(
                data.original_index[i])];
            if (count > point_set.row_capacity[row]) {
              capacity = fitted;
              acc.push_back(i);
            } else if (2 * fitted < capacity) {
              capacity = fitted;
            }
          }
          return acc;
        },
        [](Rows a, const Rows &b) {
          a.insert(a.end(), b.begin(), b.end());
          return a;
        });
    bool complete = overflowed.empty();
    if (!complete) {
      // A point's degree rarely jumps, so only a few rows overflow (and by
      // little): give them new space after the rows, which the buffer's
      // headroom usually holds, and rewrite only them. Their grown
      // capacities place them inline from the next frame on.
      int64_t spilled = total;
      for (int row : overflowed) {
        spilled += point_set.moving_caps[static_cast<std::size_t>(
                       data.original_index[row])] +
                   row_slack();
      }
      if (spilled <= static_cast<int64_t>(values.capacity()) &&
          spilled * 4 + static_cast<int64_t>(num_points) * 2 * 4 <=
              row_budget) {
        int offset = static_cast<int>(total);
        for (int row : overflowed) {
          const int capacity = point_set.moving_caps[static_cast<std::size_t>(
              data.original_index[row])];
          point_set.row_offset[static_cast<std::size_t>(row)] = offset;
          if (placed) {
            point_set.point_offset[static_cast<std::size_t>(
                data.original_index[row])] = offset;
          }
          point_set.row_capacity[static_cast<std::size_t>(row)] = capacity;
          offset += capacity + row_slack();
        }
        values.resize(static_cast<std::size_t>(spilled));
        IntBuffer queries(overflowed.begin(), overflowed.end());
        write_same_set_listed_rows(data, queries, point_set.fixed_counts,
                                   values, point_set.row_offset,
                                   point_set.row_capacity);
        complete = true;
      }
    }
    if (complete) {
      if (placed) {
        gather_slot_counts(point_set.fixed_counts,
                           point_set.original_to_sorted,
                           point_set.slot_counts);
      }
      point_set.rows_by_original = placed;
      point_set.block_rows = true;
      point_set.block_caps_valid = false;
      point_set.block_layout_valid = false;
      point_set.fixed_valid = true;
      point_set.fixed_capacity = 0;
      point_set.last_fixed_capacity = 0;
      point_set.same_lists_valid = true;
      return true;
    }
  }
  point_set.moving_caps_valid = false;
  return false;
}

bool NSearch::build_same_block_rows(int set, int min_block, bool counts_ready) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  NeighborSearchData &data = point_set.data;
  const int num_points = data.num_points;
  if (num_points == 0) {
    return false;
  }
  data.radius = radius_;

  const bool cached =
      point_set.block_caps_valid && point_set.block_size > 0 &&
      static_cast<int>(point_set.block_caps.size()) ==
          (num_points + point_set.block_size - 1) / point_set.block_size;
  int64_t block_budget = memory_budget_;
  if (!cached) {
    if (!counts_ready) {
      count_same_set_neighbors_inplace(data);
      point_set.fixed_counts = data.counts;
    }
    const IntBuffer &counts = point_set.fixed_counts;
    int64_t exact_neighbors = 0;
    for (int count : counts) {
      exact_neighbors += count;
    }
    const int64_t output_budget =
        plan::block_output_budget(exact_neighbors, num_points);
    if (memory_budget_ >= static_cast<int64_t>(64) << 20) {
      block_budget = std::max(memory_budget_, output_budget);
    }
    point_set.block_size = 0;
    for (int candidate : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
      if (candidate < min_block) {
        continue;
      }
      const int blocks = (num_points + candidate - 1) / candidate;
      int64_t bytes = static_cast<int64_t>(num_points) * 2 * 4 +
                      static_cast<int64_t>(blocks) * 2 * 4;
      std::vector<int> caps(static_cast<std::size_t>(blocks));
      for (int block = 0; block < blocks; ++block) {
        const int begin = block * candidate;
        const int end = std::min(begin + candidate, num_points);
        int max_count = 0;
        for (int i = begin; i < end; ++i) {
          max_count = std::max(max_count, counts[i]);
        }
        const int cap = round_up_to_row_align(max_count + 8);
        caps[static_cast<std::size_t>(block)] = cap;
        bytes += static_cast<int64_t>(end - begin) * (cap + row_slack()) * 4;
      }
      if (bytes <= block_budget) {
        point_set.block_size = candidate;
        point_set.block_caps.assign(caps.begin(), caps.end());
        break;
      }
    }
    if (point_set.block_size == 0) {
      return false;
    }
    point_set.block_caps_valid = true;
    point_set.block_layout_valid = false;
  } else if (!point_set.block_layout_valid ||
             point_set.block_layout_budget != memory_budget_ ||
             point_set.fixed_neighbors.size() !=
                 static_cast<std::size_t>(point_set.block_total)) {
    point_set.block_layout_valid = false;
    if (static_cast<int>(point_set.fixed_counts.size()) != num_points) {
      point_set.fixed_counts.resize(static_cast<std::size_t>(num_points));
    }
    int64_t exact_neighbors = 0;
    for (int count : point_set.fixed_counts) {
      exact_neighbors += count;
    }
    if (memory_budget_ >= static_cast<int64_t>(64) << 20) {
      block_budget = std::max(
          memory_budget_,
          plan::block_output_budget(exact_neighbors, num_points));
    }
  }

  const int block_size = point_set.block_size;
  const int block_count = (num_points + block_size - 1) / block_size;
  point_set.block_offsets.resize(static_cast<std::size_t>(block_count));
  point_set.row_offset.resize(static_cast<std::size_t>(num_points));
  point_set.row_capacity.resize(static_cast<std::size_t>(num_points));

  bool stable = false;
  for (int attempt = 0; attempt < 4; ++attempt) {
    if (!point_set.block_layout_valid) {
      int total = 0;
      int max_capacity = 0;
      for (int block = 0; block < block_count; ++block) {
        const int begin = block * block_size;
        const int end = std::min(begin + block_size, num_points);
        const int capacity =
            point_set.block_caps[static_cast<std::size_t>(block)];
        const int stride = capacity + row_slack();
        point_set.block_offsets[static_cast<std::size_t>(block)] = total;
        for (int i = begin; i < end; ++i) {
          point_set.row_offset[static_cast<std::size_t>(i)] =
              total + (i - begin) * stride;
          point_set.row_capacity[static_cast<std::size_t>(i)] = capacity;
        }
        total += (end - begin) * stride;
        max_capacity = std::max(max_capacity, capacity);
      }
      const int64_t bytes = static_cast<int64_t>(total) * 4 +
                            static_cast<int64_t>(num_points) * 2 * 4 +
                            static_cast<int64_t>(block_count) * 2 * 4;
      if (bytes > block_budget) {
        return false;
      }
      if (point_set.fixed_neighbors.capacity() <
              static_cast<std::size_t>(total) ||
          point_set.fixed_neighbors.capacity() >
              static_cast<std::size_t>(total) * 2) {
        IntBuffer().swap(point_set.fixed_neighbors);
        // After a regrow (a moving set) leave headroom, so the next small
        // growth reuses the buffer instead of faulting in a fresh one.
        if (attempt > 0) {
          point_set.fixed_neighbors.reserve(static_cast<std::size_t>(total) +
                                            static_cast<std::size_t>(total) /
                                                8);
        }
      }
      point_set.fixed_neighbors.resize(static_cast<std::size_t>(total));
      point_set.block_total = total;
      point_set.block_max_capacity = max_capacity;
      point_set.block_layout_budget = memory_budget_;
      point_set.block_layout_valid = true;
    }
    write_same_set_block_rows_parallel(
        data, point_set.block_max_capacity, point_set.fixed_counts,
        point_set.fixed_neighbors, point_set.row_offset,
        point_set.row_capacity);

    // Each block's capacity is independent, so the overflow check runs in
    // parallel; a serial pass over every row is a fixed per-frame cost.
    const bool grew = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, block_count, 4096), false,
        [&](const tbb::blocked_range<int> &range, bool acc) {
          for (int block = range.begin(); block < range.end(); ++block) {
            const int begin = block * block_size;
            const int end = std::min(begin + block_size, num_points);
            int max_count = 0;
            for (int i = begin; i < end; ++i) {
              max_count = std::max(max_count, point_set.fixed_counts[i]);
            }
            int &current =
                point_set.block_caps[static_cast<std::size_t>(block)];
            if (max_count > current) {
              // Only a moving set overflows; a margin keeps its degree
              // fluctuations from rewriting every row on most frames.
              current =
                  round_up_to_row_align(max_count + std::max(8, max_count / 8));
              acc = true;
            }
          }
          return acc;
        },
        [](bool a, bool b) { return a || b; });
    if (!grew) {
      stable = true;
      break;
    }
    point_set.block_layout_valid = false;
  }
  if (!stable) {
    return false;
  }

  point_set.block_rows = true;
  point_set.fixed_valid = true;
  point_set.fixed_capacity = 0;
  point_set.last_fixed_capacity = 0;
  point_set.same_lists_valid = true;
  // Keep per-point capacities for the next frame when the per-point layout
  // is the cheaper one. A growing set can then extend them for its new
  // points instead of recounting every candidate.
  const std::size_t point_count =
      static_cast<std::size_t>(point_set.fixed_counts.size());
  if (point_count != 0 &&
      plan::prefer_block_rows(
          plan::same_fixed_slots(point_set.fixed_counts),
          plan::same_moving_slots(point_set.fixed_counts))) {
    point_set.moving_caps.resize(point_count);
    const int *original = data.original_index.data();
    int *caps = point_set.moving_caps.data();
    const IntBuffer &counts = point_set.fixed_counts;
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, point_count, 1 << 14),
        [&](const tbb::blocked_range<std::size_t> &range) {
          for (std::size_t i = range.begin(); i < range.end(); ++i) {
            caps[static_cast<std::size_t>(original[i])] =
                plan::moving_row_capacity(counts[i]);
          }
        });
    point_set.moving_caps_valid = true;
  } else {
    point_set.moving_caps_valid = false;
  }
  return true;
}

bool NSearch::build_cross_block_rows(Pair &pair, PointSet &source,
                                     PointSet &target, bool pair_queries,
                                     bool dense_ranges) {
  if (!pair.cross_block_valid) {
    return false;
  }
  // Zero means the layout was evaluated and the fixed rows were cheaper.
  if (pair.cross_block_size == 0) {
    return false;
  }
  const int num_points = source.data.num_points;
  if (num_points == 0 ||
      pair.cross_block_source_generation != source.grid_generation ||
      pair.cross_block_target_generation != target.grid_generation ||
      pair.cross_block_source_points != num_points ||
      pair.cross_block_target_points != target.data.num_points ||
      (!dense_ranges && pair.cross.num_points != num_points) ||
      pair.cross_row_offset.size() != static_cast<std::size_t>(num_points) ||
      pair.cross_row_capacity.size() !=
          static_cast<std::size_t>(num_points)) {
    pair.cross_block_valid = false;
    pair.cross_block_active = false;
    return false;
  }
  // A budget lowered after the layout was cached must still be honored.
  if (memory_budget_ < static_cast<int64_t>(64) << 20) {
    const int64_t slots =
        static_cast<int64_t>(
            pair.cross_row_offset[static_cast<std::size_t>(num_points - 1)]) +
        pair.cross_row_capacity[static_cast<std::size_t>(num_points - 1)] +
        row_slack();
    const int64_t bytes = slots * static_cast<int64_t>(sizeof(int)) +
                          static_cast<int64_t>(num_points) * 2 * sizeof(int);
    if (bytes > memory_budget_) {
      pair.cross_block_valid = false;
      pair.cross_block_active = false;
      return false;
    }
  }
  if (dense_ranges) {
    write_cross_neighbors_block_rows_pair_dense_parallel(
        pair.cross, source.data, target.data, pair.cross_block_capacity,
        pair.fixed_neighbors, pair.fixed_counts, pair.cross_row_offset,
        pair.cross_row_capacity);
  } else if (pair_queries) {
    write_cross_neighbors_block_rows_pair_parallel(
        pair.cross, target.data, pair.cross_block_capacity,
        pair.fixed_neighbors, pair.fixed_counts, pair.cross_row_offset,
        pair.cross_row_capacity);
  } else {
    write_cross_neighbors_block_rows_parallel(
        pair.cross, target.data, pair.cross_block_capacity,
        pair.fixed_neighbors, pair.fixed_counts, pair.cross_row_offset,
        pair.cross_row_capacity);
  }
  pair.cross_block_active = true;
  pair.fixed_valid = true;
  return true;
}

bool NSearch::remember_cross_block_rows(Pair &pair, PointSet &source,
                                        PointSet &target, int capacity) {
  pair.cross_block_active = false;
  // A moving set rebuilds its grid every frame, so a block layout learned now
  // cannot be reused and the scan would only add work. An infinite distance is
  // the first sync (no previous coordinates), not observed motion, so a static
  // scene can still build its layout on the first frame.
  if (skipped_position_sync_ ||
      (std::isfinite(last_stats_.max_displacement_sq) &&
       last_stats_.max_displacement_sq > 0.0f)) {
    pair.cross_block_valid = false;
    return false;
  }
  const int num_points = source.data.num_points;
  if (num_points == 0 ||
      pair.fixed_counts.size() != static_cast<std::size_t>(num_points)) {
    pair.cross_block_valid = false;
    return false;
  }
  // A zero block size records that the fixed layout was cheaper for this
  // geometry, so later frames do not rescan the rows.
  if (pair.cross_block_valid && pair.cross_block_size == 0 &&
      pair.cross_block_source_generation == source.grid_generation &&
      pair.cross_block_target_generation == target.grid_generation &&
      pair.cross_block_source_points == num_points &&
      pair.cross_block_target_points == target.data.num_points) {
    return false;
  }
  auto mark_fixed_layout = [&] {
    pair.cross_block_size = 0;
    pair.cross_block_source_generation = source.grid_generation;
    pair.cross_block_target_generation = target.grid_generation;
    pair.cross_block_source_points = num_points;
    pair.cross_block_target_points = target.data.num_points;
    pair.cross_block_valid = true;
    IntBuffer().swap(pair.cross_row_offset);
    IntBuffer().swap(pair.cross_row_capacity);
  };
  // Tiny sources do not amortize the per-row offset/capacity loads.
  if (num_points < kCrossBlockMinPoints) {
    mark_fixed_layout();
    return false;
  }
  pair.cross_block_valid = false;

  // One row per block: each row's capacity is just its own degree. Blocks
  // larger than a row only raise the local maximum without changing the row
  // offsets, so they add padding for no benefit.
  constexpr int kBlockSize = 1;
  int64_t slots = 0;
  int max_capacity = 0;
  int64_t exact_neighbors = 0;
  for (int i = 0; i < num_points; ++i) {
    const int count = pair.fixed_counts[static_cast<std::size_t>(i)];
    // Rows are written with aligned non-temporal stores of a whole gang, so
    // every row start must stay gang-aligned: round to row_align().
    const int cap = round_up_to_row_align(count);
    exact_neighbors += count;
    slots += cap + row_slack();
    max_capacity = std::max(max_capacity, cap);
  }

  // Keep the fixed layout unless per-row capacities remove at least half of
  // the row table. The extra row_offset/row_capacity loads cost more than the
  // saved stores for moderate padding, and the win only appears when the
  // global maximum is far above the typical degree.
  const int64_t fixed_slots =
      static_cast<int64_t>(num_points) * (capacity + row_slack());
  if (slots * 2 > fixed_slots) {
    mark_fixed_layout();
    return false;
  }

  int64_t budget = memory_budget_;
  if (memory_budget_ >= static_cast<int64_t>(64) << 20) {
    budget = std::max(budget, plan::block_output_budget(exact_neighbors,
                                                        num_points));
  }
  const int64_t bytes =
      slots * static_cast<int64_t>(sizeof(int)) +
      static_cast<int64_t>(num_points) * 2 * sizeof(int);
  if (bytes > budget) {
    mark_fixed_layout();
    return false;
  }

  pair.cross_block_size = kBlockSize;
  pair.cross_block_capacity = max_capacity;
  pair.cross_row_offset.resize(static_cast<std::size_t>(num_points));
  pair.cross_row_capacity.resize(static_cast<std::size_t>(num_points));
  int total = 0;
  for (int i = 0; i < num_points; ++i) {
    const int cap =
        round_up_to_row_align(pair.fixed_counts[static_cast<std::size_t>(i)]);
    pair.cross_row_offset[static_cast<std::size_t>(i)] = total;
    pair.cross_row_capacity[static_cast<std::size_t>(i)] = cap;
    total += cap + row_slack();
  }
  pair.cross_block_source_generation = source.grid_generation;
  pair.cross_block_target_generation = target.grid_generation;
  pair.cross_block_source_points = num_points;
  pair.cross_block_target_points = target.data.num_points;
  pair.cross_block_valid = true;
  return true;
}

// Chooses the base capacity of a hybrid cross layout from the exact row
// counts of a fixed write. Most rows fit the base; the longer rows get
// per-row capacities past the base area.
void NSearch::seed_cross_hybrid(Pair &pair, PointSet &source) {
  pair.cross_hybrid_valid = false;
  pair.cross_hybrid_active = false;
  if (pair.cross_hybrid_check_wait > 0) {
    --pair.cross_hybrid_check_wait;
    return;
  }
  const int num_points = source.data.num_points;
  if (num_points < kCrossHybridMinPoints ||
      pair.fixed_counts.size() != static_cast<std::size_t>(num_points) ||
      pair.fixed_capacity <= 0) {
    return;
  }
  // Very wide fixed rows are already read as one contiguous run per query;
  // the base/overflow indirection costs more than the padding it removes.
  if (pair.fixed_capacity >= 64) {
    pair.cross_hybrid_check_wait = 16;
    return;
  }
  const IntBuffer &counts = pair.fixed_counts;
  int max_count = 0;
  for (int count : counts) {
    max_count = std::max(max_count, count);
  }
  constexpr int kMaxCount = 4096;
  if (max_count > kMaxCount) {
    return;
  }
  std::vector<int64_t> histogram(static_cast<std::size_t>(max_count) + 1, 0);
  for (int count : counts) {
    ++histogram[static_cast<std::size_t>(count)];
  }
  const int align = row_align();
  const int slack = row_slack();
  const int64_t fixed_bytes =
      static_cast<int64_t>(num_points) * pair.fixed_capacity;
  int chosen = 0;
  for (int base = align; base <= max_count; base += align) {
    int64_t overflow_rows = 0;
    int64_t overflow_slots = 0;
    for (int count = base + 1; count <= max_count; ++count) {
      const int64_t rows = histogram[static_cast<std::size_t>(count)];
      if (rows == 0) {
        continue;
      }
      overflow_rows += rows;
      const int cap =
          round_up_to_row_align(count + std::max(8, count / 8));
      overflow_slots += rows * (cap + slack);
    }
    // The overflow rows are rewritten by a second pass; keep them rare so
    // that pass never shows up in the frame time.
    if (overflow_rows * 128 > num_points) {
      continue;
    }
    const int64_t slots = static_cast<int64_t>(num_points) * (base + slack) +
                          overflow_slots;
    // Only take the hybrid when it halves the row table; smaller savings do
    // not pay for the extra overflow metadata.
    if (slots * 4 <= fixed_bytes * 3) {
      chosen = base;
      break;
    }
  }
  if (chosen == 0) {
    pair.cross_hybrid_check_wait = 16;
    return;
  }
  pair.cross_hybrid_base = chosen;
  pair.cross_hybrid_offset.resize(static_cast<std::size_t>(num_points));
  pair.cross_hybrid_capacity.resize(static_cast<std::size_t>(num_points));
  pair.cross_hybrid_valid = true;
}

bool NSearch::build_cross_hybrid_rows(Pair &pair, PointSet &source,
                                      PointSet &target, bool pair_queries,
                                      bool dense_ranges) {
  const int num_points = source.data.num_points;
  if (!pair.cross_hybrid_valid || !dense_ranges || !pair_queries ||
      num_points < kCrossBlockMinPoints ||
      pair.cross_hybrid_offset.size() !=
          static_cast<std::size_t>(num_points) ||
      pair.cross_hybrid_capacity.size() !=
          static_cast<std::size_t>(num_points)) {
    return false;
  }
  const int base = pair.cross_hybrid_base;
  const int base_stride = base + row_slack();
  const int64_t base_slots = static_cast<int64_t>(num_points) * base_stride;
  // Pass 1 writes every row at the base capacity and counts it exactly; rows
  // above the base are truncated and rewritten by pass 2 past the base area.
  write_cross_neighbors_fixed_query_pair_dense_parallel(
      pair.cross, source.data, target.data, base, pair.fixed_neighbors,
      pair.fixed_counts);
  const IntBuffer &counts = pair.fixed_counts;
  const int64_t overflow_slots = tbb::parallel_scan(
      tbb::blocked_range<int>(0, num_points, 1 << 14), int64_t{0},
      [&](const tbb::blocked_range<int> &range, int64_t sum, bool final) {
        for (int i = range.begin(); i < range.end(); ++i) {
          const int count = counts[static_cast<std::size_t>(i)];
          if (count > base) {
            const int cap =
                round_up_to_row_align(count + std::max(8, count / 8));
            pair.cross_hybrid_capacity[static_cast<std::size_t>(i)] = cap;
            if (final) {
              pair.cross_hybrid_offset[static_cast<std::size_t>(i)] =
                  static_cast<int>(base_slots + sum);
            }
            sum += cap + row_slack();
          }
        }
        return sum;
      },
      [](int64_t a, int64_t b) { return a + b; });
  const int64_t total = base_slots + overflow_slots;
  int64_t budget = memory_budget_;
  if (memory_budget_ >= static_cast<int64_t>(64) << 20) {
    const int64_t exact_neighbors = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, num_points, 1 << 14), int64_t{0},
        [&](const tbb::blocked_range<int> &range, int64_t acc) {
          for (int i = range.begin(); i < range.end(); ++i) {
            acc += counts[static_cast<std::size_t>(i)];
          }
          return acc;
        },
        [](int64_t a, int64_t b) { return a + b; });
    budget = std::max(budget, plan::block_output_budget(exact_neighbors,
                                                        num_points));
  }
  if (total > INT_MAX || total * 4 > budget) {
    pair.cross_hybrid_valid = false;
    return false;
  }
  IntBuffer &values = pair.fixed_neighbors;
  // Pass 1 already sized and filled the base area; only grow from here so
  // the base rows survive.
  if (values.capacity() < static_cast<std::size_t>(total)) {
    values.reserve(static_cast<std::size_t>(total + total / 8));
  }
  values.resize(static_cast<std::size_t>(total));
  // Pass 2 rewrites the rows above the base into their own slots.
  const int overflow_rows = static_cast<int>(tbb::parallel_reduce(
      tbb::blocked_range<int>(0, num_points, 1 << 14), 0,
      [&](const tbb::blocked_range<int> &range, int acc) {
        for (int i = range.begin(); i < range.end(); ++i) {
          acc += counts[static_cast<std::size_t>(i)] > base ? 1 : 0;
        }
        return acc;
      },
      [](int a, int b) { return a + b; }));
  if (overflow_rows > 0) {
    std::vector<int> queries(static_cast<std::size_t>(overflow_rows));
    tbb::parallel_scan(
        tbb::blocked_range<int>(0, num_points, 1 << 14), 0,
        [&](const tbb::blocked_range<int> &range, int sum, bool final) {
          for (int i = range.begin(); i < range.end(); ++i) {
            if (counts[static_cast<std::size_t>(i)] > base) {
              if (final) {
                queries[static_cast<std::size_t>(sum)] = i;
              }
              ++sum;
            }
          }
          return sum;
        },
        [](int a, int b) { return a + b; });
    const IntBuffer &cell_starts = source.data.cell_starts;
    std::vector<int> query_cells(static_cast<std::size_t>(overflow_rows));
    tbb::parallel_for(
        tbb::blocked_range<int>(0, overflow_rows, 64),
        [&](const tbb::blocked_range<int> &range) {
          for (int k = range.begin(); k < range.end(); ++k) {
            query_cells[static_cast<std::size_t>(k)] = static_cast<int>(
                std::upper_bound(cell_starts.begin(), cell_starts.end(),
                                 queries[static_cast<std::size_t>(k)]) -
                cell_starts.begin()) - 1;
          }
        });
    write_cross_hybrid_rows_parallel(
        pair.cross, source.data, target.data, queries, query_cells,
        pair.fixed_counts, values, pair.cross_hybrid_offset,
        pair.cross_hybrid_capacity);
  }
  note_path(kPathCrossBlocked);
  pair.cross_block_active = false;
  pair.cross_block_valid = false;
  pair.cross_hybrid_active = true;
  pair.fixed_valid = true;
  pair.fixed_max_count = max_row_count(counts);
  pair.fixed_capacity = 0;
  pair.last_fixed_capacity = base + row_slack();
  return true;
}

bool NSearch::transpose_cross_pair(Pair &forward, Pair &reverse,
                                   PointSet &source, PointSet &target) {
  const int n = source.data.num_points;
  if (!forward.fixed_valid || forward.cross_block_active ||
      forward.fixed_capacity <= 0 || forward.same_set || n == 0 ||
      target.data.num_points != n ||
      forward.fixed_counts.size() != static_cast<std::size_t>(n)) {
    return false;
  }
  const plan::CrossPlan reverse_plan = plan::cross_set(
      target.data, target.last_fixed_capacity, source.data,
      reverse.last_fixed_capacity, memory_budget_);
  if (!reverse_plan.fixed_rows || reverse_plan.first_capacity <= 0) {
    return false;
  }
  const int capacity = reverse_plan.first_capacity;
  const int reverse_stride = capacity + row_slack();
  const std::size_t rows = static_cast<std::size_t>(n);
  const std::size_t required =
      rows * static_cast<std::size_t>(reverse_stride);
  reverse.fixed_counts.resize(rows);
  if (required > reverse.fixed_neighbors.capacity() ||
      2 * required < reverse.fixed_neighbors.capacity()) {
    IntBuffer().swap(reverse.fixed_neighbors);
  }
  reverse.fixed_neighbors.resize(required);
  std::fill(reverse.fixed_counts.begin(), reverse.fixed_counts.end(), 0);
  const int forward_stride = forward.fixed_capacity;
  const int *forward_counts = forward.fixed_counts.data();
  const int *forward_rows = forward.fixed_neighbors.data();
  const int *source_original = source.data.original_index.data();
  const int *target_to_sorted = target.original_to_sorted.data();
  int *reverse_counts = reverse.fixed_counts.data();
  int *reverse_rows = reverse.fixed_neighbors.data();
  bool overflow = false;
  for (int row = 0; row < n; ++row) {
    const int count = forward_counts[row];
    if (count <= 0) {
      continue;
    }
    const int source_id = source_original[row];
    const int *neighbors =
        forward_rows + static_cast<std::size_t>(row) *
                           static_cast<std::size_t>(forward_stride);
    for (int k = 0; k < count; ++k) {
      const int reverse_row = target_to_sorted[neighbors[k]];
      int &filled = reverse_counts[reverse_row];
      if (filled < capacity) {
        reverse_rows[static_cast<std::size_t>(reverse_row) *
                         static_cast<std::size_t>(reverse_stride) +
                     filled] = source_id;
      } else {
        overflow = true;
      }
      ++filled;
    }
  }
  if (overflow) {
    return false;
  }
  int max_count = 0;
  for (int row = 0; row < n; ++row) {
    max_count = std::max(max_count, reverse_counts[row]);
  }
  reverse.fixed_max_count = max_count;
  reverse.fixed_capacity = reverse_stride;
  reverse.last_fixed_capacity =
      plan::cross_fitted_capacity(max_count, capacity);
  reverse.fixed_valid = true;
  reverse.cross_block_active = false;
  reverse.same_set = false;
  reverse.valid = true;
  reverse.cross.radius = radius_;
  return true;
}

bool NSearch::build_sweep_same_lists(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  sweep::ensure_index(point_set.sweep, point_set.x, point_set.y,
                      point_set.z);
  const sweep::Cloud cloud{point_set.x, point_set.y, point_set.z,
                           &point_set.sweep};
  int row_stride = 0;
  if (!sweep::write_fixed_rows(cloud, point_set.original_to_sorted, cloud,
                               true, radius_, memory_budget_,
                               point_set.fixed_neighbors,
                               point_set.fixed_counts, row_stride)) {
    return false;
  }
  point_set.fixed_capacity = row_stride;
  point_set.last_fixed_capacity = row_stride;
  point_set.fixed_valid = true;
  point_set.sparse_rows_by_original = false;
  point_set.rows_by_original = false;
  point_set.same_lists_valid = true;
  return true;
}

bool NSearch::build_sweep_cross_lists(Pair &pair, PointSet &source,
                                      PointSet &target) {
  sweep::ensure_index(target.sweep, target.x, target.y, target.z);
  const sweep::Cloud query{source.x, source.y, source.z, &source.sweep};
  const sweep::Cloud cloud{target.x, target.y, target.z, &target.sweep};
  int row_stride = 0;
  if (!sweep::write_fixed_rows(query, source.original_to_sorted, cloud, false,
                               radius_, memory_budget_,
                               pair.fixed_neighbors, pair.fixed_counts,
                               row_stride)) {
    return false;
  }
  pair.fixed_capacity = row_stride;
  pair.last_fixed_capacity = row_stride;
  pair.fixed_valid = true;
  return true;
}

void NSearch::invalidate_pairs_for_set(int set) {
  if (!valid_set(set)) {
    return;
  }

  sets_[static_cast<std::size_t>(set)].same_lists_valid = false;
  sets_[static_cast<std::size_t>(set)].fixed_valid = false;
  sets_[static_cast<std::size_t>(set)].fixed_max_count = -1;
  const int set_count = static_cast<int>(sets_.size());
  for (int other = 0; other < set_count; ++other) {
    clear_pair(pairs_[static_cast<std::size_t>(set)]
                    [static_cast<std::size_t>(other)]);
    if (other != set) {
      clear_pair(pairs_[static_cast<std::size_t>(other)]
                      [static_cast<std::size_t>(set)]);
    }
  }
}

bool NSearch::max_cell_occupancy(PointSet &set, int &max_points) {
  constexpr int kProbeTtl = 16;
  if (set.probe_valid && set.probe_age < kProbeTtl) {
    ++set.probe_age;
    max_points = set.probe_max_cell;
    return max_points >= 0;
  }
  const int n = static_cast<int>(set.x.size());
  if (n == 0) {
    set.probe_max_cell = -1;
    set.probe_valid = true;
    set.probe_age = 0;
    return false;
  }
  const float min_x = set.bounds.min_x, min_y = set.bounds.min_y,
              min_z = set.bounds.min_z;
  const float max_x = set.bounds.max_x, max_y = set.bounds.max_y,
              max_z = set.bounds.max_z;
  const float inv = 1.0f / std::max(radius_, 1.0e-6f);
  const int nx =
      static_cast<int>(std::floor((max_x - (min_x - radius_)) * inv)) + 2;
  const int ny =
      static_cast<int>(std::floor((max_y - (min_y - radius_)) * inv)) + 2;
  const int nz =
      static_cast<int>(std::floor((max_z - (min_z - radius_)) * inv)) + 2;
  const int64_t cells =
      static_cast<int64_t>(nx) * static_cast<int64_t>(ny) * nz;
  if (cells <= 0 || cells > 8000000) {
    set.probe_max_cell = -1;
    set.probe_valid = true;
    set.probe_age = 0;
    return false;
  }
  std::vector<int> counts(static_cast<std::size_t>(cells), 0);
  int max_count = 0;
  for (int i = 0; i < n; ++i) {
    const int cx = static_cast<int>(std::floor(
        (set.x[static_cast<std::size_t>(i)] - (min_x - radius_)) * inv));
    const int cy = static_cast<int>(std::floor(
        (set.y[static_cast<std::size_t>(i)] - (min_y - radius_)) * inv));
    const int cz = static_cast<int>(std::floor(
        (set.z[static_cast<std::size_t>(i)] - (min_z - radius_)) * inv));
    const std::size_t index =
        (static_cast<std::size_t>(cz) * static_cast<std::size_t>(ny) +
         static_cast<std::size_t>(cy)) *
            static_cast<std::size_t>(nx) +
        static_cast<std::size_t>(cx);
    max_count = std::max(max_count, ++counts[index]);
  }
  max_points = max_count;
  set.probe_max_cell = max_count;
  set.probe_valid = true;
  set.probe_age = 0;
  return true;
}

bool NSearch::gpu_policy_ok() {
  constexpr int kMinSourcePoints = 200000;
  constexpr int kMovingDenseCellLimit = 32;
  // Cell-major GPU wins when cells are dense enough to amortize the query
  // grouping, but not so dense that the 4-byte neighbor output dominates.
  // Measured: uniform/slab/1M max cell 21-36; jittered lattice 16 and
  // clustered/surface/thin/two-scale 94-1753.
  constexpr int kMinCellPoints = 20;
  constexpr int kMaxCellPoints = 48;
  bool any = false;
  const int set_count = static_cast<int>(sets_.size());
  for (int i = 0; i < set_count; ++i) {
    const PointSet &source = sets_[static_cast<std::size_t>(i)];
    for (int j = 0; j < set_count; ++j) {
      const Pair &pair = pairs_[static_cast<std::size_t>(i)]
                                [static_cast<std::size_t>(j)];
      if (!pair.active) {
        continue;
      }
      if (static_cast<int>(source.x.size()) < kMinSourcePoints) {
        return false;
      }
      int max_cell = 0;
      if (!max_cell_occupancy(sets_[static_cast<std::size_t>(j)], max_cell) ||
          max_cell > kMaxCellPoints) {
        return false;
      }
      // The lower bound is a same-set signal: jittered/sparse same-set grids
      // do not amortize the cell-major GPU kernel. Cross targets such as a
      // sparse boundary are fine and are judged only by the upper bound.
      if (i == j && max_cell < kMinCellPoints) {
        return false;
      }
      // A moving dense grid pays both the per-frame GPU grid rebuild and the
      // transfer; measured slab-200k loses there while uniform-200k wins.
      if (i == j && last_stats_.max_displacement_sq > 0.0f &&
          max_cell > kMovingDenseCellLimit) {
        return false;
      }
      any = true;
    }
  }
  return any;
}

bool NSearch::build_gpu_pair(Pair &pair, int source, int target,
                             bool defer_same_copy) {
#ifdef APFRNN_HAVE_SYCL
  if (!valid_set(source) || !valid_set(target)) {
    return false;
  }
  if (!gpu_) {
    gpu_ = std::make_unique<GpuState>();
  }
  if (gpu_->context == nullptr) {
    return false;
  }
  PointSet &source_set = sets_[static_cast<std::size_t>(source)];
  PointSet &target_set = sets_[static_cast<std::size_t>(target)];
  const int nq = static_cast<int>(source_set.x.size());
  const int nt = static_cast<int>(target_set.x.size());
  if (nq == 0 || nt == 0) {
    return false;
  }
  // The backend caches grids per set id and version, so every repack gets a
  // new version.
  auto points = [&](PointSet &set, int id) {
    const std::size_t count = set.x.size();
    if (!set.gpu_xyz_valid || set.gpu_xyz.size() != count * 3) {
      set.gpu_xyz.resize(count * 3);
      for (std::size_t i = 0; i < count; ++i) {
        set.gpu_xyz[3 * i + 0] = set.x[i];
        set.gpu_xyz[3 * i + 1] = set.y[i];
        set.gpu_xyz[3 * i + 2] = set.z[i];
      }
      set.gpu_xyz_valid = true;
      set.gpu_version = gpu_->next_version++;
    }
    return ApfrnnSyclPoints{set.gpu_xyz.data(), static_cast<int32_t>(count),
                            id, set.gpu_version, set.is_static ? 1 : 0};
  };
  const ApfrnnSyclPoints query = points(source_set, source);
  const ApfrnnSyclPoints target_points = points(target_set, target);

  const int slot = source * static_cast<int>(sets_.size()) + target;
  ApfrnnSyclResult result{};
  const int rc = apfrnn_sycl_exact_pair(
      gpu_->context, &query, &target_points, radius_, source == target ? 1 : 0,
      slot, defer_same_copy ? 1 : 0, &result);
  if (rc != 0 || result.neighbors == nullptr || result.offsets == nullptr ||
      result.nq != nq) {
    return false;
  }
  pair.gpu_rows = true;
  pair.gpu_neighbors = reinterpret_cast<const uint32_t *>(result.neighbors);
  pair.gpu_offsets = result.offsets;
  pair.gpu_nq = result.nq;
  pair.same_set = source == target;
  pair.fixed_valid = false;
  pair.valid = true;
  return true;
#else
  (void)pair;
  (void)source;
  (void)target;
  (void)defer_same_copy;
  return false;
#endif
}

void NSearch::clear_pair(Pair &pair) {
  pair.valid = false;
  pair.same_set = false;
  pair.fixed_valid = false;
  pair.cross_block_active = false;
  pair.cross_hybrid_active = false;
  pair.gpu_rows = false;
  pair.gpu_neighbors = nullptr;
  pair.gpu_offsets = nullptr;
  pair.gpu_nq = 0;
  pair.cross = CrossNeighborSearchData{};
  pair.fixed_neighbors.clear();
  if (!pair.incremental_rows_owned || !pair.active) {
    pair.fixed_counts.clear();
    pair.incremental_rows_touched.clear();
    pair.incremental_rows_touched_overflow = false;
  }
}

} // namespace apfrnn
