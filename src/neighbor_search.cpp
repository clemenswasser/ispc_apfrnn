#include "apfrnn/neighbor_search.hpp"

#include "apfrnn.hpp"
#include "plan.hpp"
#include "sweep.hpp"

#ifdef APFRNN_HAVE_SYCL
#include "sycl/apfrnn_sycl.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace apfrnn {

namespace {

constexpr int kPairParallelMaxPoints = 32768;

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

int max_row_count(const IntBuffer &counts) {
  int max_count = 0;
  for (int count : counts) {
    max_count = std::max(max_count, count);
  }
  return max_count;
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
  bool skin_lists = false;
  bool active_valid = false;
  int fixed_capacity = 128;
  int last_fixed_capacity = 0;
  Bounds bounds;  // of x/y/z, maintained by sync_source
  bool use_sweep = false;
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
  IntBuffer active_neighbors;
  IntBuffer active_counts;
};

struct NSearch::Pair {
  bool active = true;
  bool valid = false;
  bool same_set = false;
  bool fixed_valid = false;
  bool skin_lists = false;
  bool active_valid = false;
  int fixed_capacity = 64;
  int last_fixed_capacity = 0;
  bool gpu_rows = false;
  const uint32_t *gpu_neighbors = nullptr;
  const int64_t *gpu_offsets = nullptr;
  int gpu_nq = 0;
  CrossNeighborSearchData cross;
  IntBuffer fixed_neighbors;
  IntBuffer fixed_counts;
  IntBuffer active_neighbors;
  IntBuffer active_counts;
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

float NSearch::sync_source(int set_index) {
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
    return max_distance_sq;
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
    // point moved.
    set.sweep.valid = false;
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
  if (!valid_set(set) || n < 0 || (n > 0 && xyz == nullptr)) {
    return;
  }

  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  point_set.source_type = is_double ? SourceType::Double : SourceType::Float;
  point_set.source = xyz;
  point_set.source_count = n;
  point_set.dirty = true;
  point_set.valid = false;
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
  if (!valid_set(set_i) || !valid_set(set_j)) {
    return;
  }

  Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                     [static_cast<std::size_t>(set_j)];
  pair.active = active;
}

void NSearch::set_all_searches(bool active) {
  for (auto &row : pairs_) {
    for (Pair &pair : row) {
      pair.active = active;
    }
  }
}

void NSearch::update_point_sets() {
  for (int set = 0; set < static_cast<int>(sets_.size()); ++set) {
    if (!sets_[static_cast<std::size_t>(set)].is_static) {
      sets_[static_cast<std::size_t>(set)].dirty = true;
      invalidate_pairs_for_set(set);
    }
  }
}

void NSearch::run() {
  last_stats_ = RunStats{};
  if (sets_.empty()) {
    return;
  }

  const int set_count = static_cast<int>(sets_.size());
  update_cell_size();
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
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, jobs.size(), 1),
        [&](const tbb::blocked_range<std::size_t> &range) {
          for (std::size_t index = range.begin(); index < range.end();
               ++index) {
            const Job job = jobs[index];
            if (job.set_i == job.set_j) {
              continue;
            }
            Pair &pair = pairs_[static_cast<std::size_t>(job.set_i)]
                               [static_cast<std::size_t>(job.set_j)];
            const auto pair_start = std::chrono::steady_clock::now();
            build_cpu_pair(pair, job.set_i, job.set_j);
            pair_times[index] = elapsed_ms(pair_start);
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
    for (int set_i = 0; set_i < set_count; ++set_i) {
      for (int set_j = 0; set_j < set_count; ++set_j) {
        Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                           [static_cast<std::size_t>(set_j)];
        if (!pair.active || pair.valid) {
          continue;
        }

        const auto pair_start = std::chrono::steady_clock::now();
        if (effective_backend_ == SearchBackend::GpuExact) {
          const bool defer_same = set_i == set_j;
          if (!build_gpu_pair(pair, set_i, set_j, defer_same)) {
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

  if (skin_ > 0.0f) {
#ifdef APFRNN_HAVE_SYCL
    // Skin filtering reads the rows; make any deferred copy visible first.
    if (gpu_) {
      apfrnn_sycl_wait(gpu_->context);
    }
#endif
    const auto filter_start = std::chrono::steady_clock::now();
    filter_skin_lists();
    last_stats_.filter_ms = elapsed_ms(filter_start);
  }

  collect_memory_stats();
}

void NSearch::update_cell_size() {
  bool small_cells = false;
  if (skin_ <= 0.0f) {
    for (const PointSet &point_set : sets_) {
      small_cells = small_cells || plan::wants_small_cells(
                                       point_set.data,
                                       point_set.last_fixed_capacity);
    }
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
  // Dynamic positions are synced on every run. With a skin, the existing
  // neighbor lists remain valid while every point moves at most skin/2. A
  // pure target set that did not move keeps its grid (target cache).
  const bool target_cache = skin_ <= 0.0f;
  const float reuse_distance_sq = 0.25f * skin_ * skin_;
  for (int set = 0; set < static_cast<int>(sets_.size()); ++set) {
    PointSet &point_set = sets_[static_cast<std::size_t>(set)];
    if (point_set.is_static) {
      continue;
    }
    const float max_distance_sq = sync_source(set);
    if (max_distance_sq > 0.0f) {
      // The packed GPU copy and the cached grid are stale once any point
      // moved. A static frame keeps both across runs.
      point_set.gpu_xyz_valid = false;
    }
    last_stats_.max_displacement_sq =
        std::max(last_stats_.max_displacement_sq, max_distance_sq);
    const bool target_unchanged =
        target_cache && !is_source[static_cast<std::size_t>(set)] &&
        point_set.valid && point_set.target_cache_valid &&
        point_set.cached_x == point_set.x &&
        point_set.cached_y == point_set.y && point_set.cached_z == point_set.z;
    if (target_unchanged) {
      point_set.dirty = false;
      ++last_stats_.reused_sets;
    } else if (skin_ <= 0.0f || point_set.dirty || !point_set.valid ||
               !point_set.skin_lists ||
               max_distance_sq > reuse_distance_sq) {
      point_set.dirty = true;
      ++last_stats_.rebuilt_sets;
    } else {
      ++last_stats_.reused_sets;
    }
  }
}

void NSearch::build_point_sets(const std::vector<bool> &is_source,
                               const std::vector<bool> &needs_same_set) {
  const bool target_cache = skin_ <= 0.0f;
  for (int set = 0; set < static_cast<int>(sets_.size()); ++set) {
    PointSet &point_set = sets_[static_cast<std::size_t>(set)];
    if (effective_backend_ == SearchBackend::GpuExact) {
      if (point_set.is_static && (!point_set.valid || point_set.dirty)) {
        sync_source(set);
      }
      if (point_set.dirty || !point_set.valid || !point_set.is_static) {
        invalidate_pairs_for_set(set);
      }
      point_set.valid = true;
      point_set.dirty = false;
      continue;
    }
    if (!point_set.dirty && point_set.valid) {
      continue;
    }
    invalidate_pairs_for_set(set);

    const auto grid_start = std::chrono::steady_clock::now();
    if (point_set.is_static) {
      sync_source(set);
    }
    build_point_grid(set);
    last_stats_.grid_ms += elapsed_ms(grid_start);
    if (target_cache && !is_source[static_cast<std::size_t>(set)]) {
      point_set.cached_x = point_set.x;
      point_set.cached_y = point_set.y;
      point_set.cached_z = point_set.z;
      point_set.target_cache_valid = true;
    }

    if (needs_same_set[static_cast<std::size_t>(set)]) {
      const auto same_start = std::chrono::steady_clock::now();
      build_same_lists(set);
      last_stats_.same_build_ms += elapsed_ms(same_start);
    }
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

void NSearch::build_cross_pair(Pair &pair, int set_i, int set_j) {
  PointSet &source = sets_[static_cast<std::size_t>(set_i)];
  PointSet &target = sets_[static_cast<std::size_t>(set_j)];
  pair.fixed_valid = false;
  pair.same_set = false;
  pair.valid = true;

  if (source.use_sweep || target.use_sweep) {
    if (build_sweep_cross_lists(pair, source, target)) {
      return;
    }
    ensure_grid(source);
    ensure_grid(target);
  }

  if (plan::cross_wants_target_blocks(source.data, target.data)) {
    ensure_point_blocks(target.data);
  }
  const plan::CrossPlan cross_plan =
      plan::cross_set(source.data, source.last_fixed_capacity, target.data,
                      pair.last_fixed_capacity, memory_budget_);

  auto try_fixed = [&](int capacity) {
    switch (cross_plan.writer) {
    case plan::CrossWriter::TargetMajor:
      write_cross_neighbors_target_major_parallel(
          pair.cross, source.data, target.data, capacity,
          pair.fixed_neighbors, pair.fixed_counts);
      break;
    case plan::CrossWriter::PairDense:
      write_cross_neighbors_fixed_query_pair_dense_parallel(
          pair.cross, source.data, target.data, capacity,
          pair.fixed_neighbors, pair.fixed_counts);
      break;
    case plan::CrossWriter::QueryPair:
      write_cross_neighbors_fixed_query_pair_parallel(
          pair.cross, target.data, capacity, pair.fixed_neighbors,
          pair.fixed_counts);
      break;
    case plan::CrossWriter::Blocked:
      write_cross_neighbors_fixed_query_blocked_parallel(
          pair.cross, target.data, source.data, capacity,
          pair.fixed_neighbors, pair.fixed_counts);
      break;
    case plan::CrossWriter::Active2:
      write_cross_neighbors_fixed_query_active2_parallel(
          pair.cross, target.data, capacity, pair.fixed_neighbors,
          pair.fixed_counts);
      break;
    }
    const int max_count = max_row_count(pair.fixed_counts);
    if (max_count > capacity) {
      return false;
    }
    pair.fixed_valid = true;
    pair.skin_lists = skin_ > 0.0f;
    pair.active_valid = false;
    pair.fixed_capacity = capacity + 8;
    pair.last_fixed_capacity = plan::cross_fitted_capacity(max_count, capacity);
    return true;
  };

  if (cross_plan.fixed_rows) {
    if (cross_plan.build_ranges) {
      build_cross_cell_ranges_from_sorted_inplace(pair.cross, source.data,
                                                  target.data, build_radius());
      if (plan::cross_has_no_candidates(pair.cross)) {
        pair.fixed_counts.assign(
            static_cast<std::size_t>(source.data.num_points), 0);
        pair.fixed_capacity = 1;
        pair.fixed_neighbors.resize(
            static_cast<std::size_t>(source.data.num_points));
        pair.fixed_valid = true;
        pair.skin_lists = false;
        pair.active_valid = false;
        return;
      }
    }
    if (try_fixed(cross_plan.first_capacity)) {
      return;
    }
    const int grown = plan::cross_grown_capacity(
        max_row_count(pair.fixed_counts), source.data.num_points,
        memory_budget_);
    if (grown > 0 && try_fixed(grown)) {
      return;
    }
    if (parallel_pairs_) {
      parallel_overflows_.fetch_add(1, std::memory_order_relaxed);
    } else {
      ++last_stats_.fixed_overflows;
    }
    pair.fixed_neighbors.clear();
    pair.fixed_counts.clear();
  }

  pair.skin_lists = false;
  pair.active_valid = false;
  build_cross_neighbor_search_data_from_sorted_inplace(pair.cross, source.data,
                                                       target.data, radius_);
  write_cross_neighbors_parallel(pair.cross, target.data);
}

void NSearch::filter_skin_lists() {
  const int set_count = static_cast<int>(sets_.size());
  for (int set_i = 0; set_i < set_count; ++set_i) {
    for (int set_j = 0; set_j < set_count; ++set_j) {
      Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                         [static_cast<std::size_t>(set_j)];
      if (!pair.active || !pair.valid) {
        continue;
      }
      if (set_i == set_j &&
          sets_[static_cast<std::size_t>(set_i)].skin_lists) {
        filter_same_active(set_i);
      } else if (set_i != set_j && pair.skin_lists) {
        filter_cross_active(pair, set_i, set_j);
      }
    }
  }
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
        source.fixed_counts.capacity() * sizeof(int);

    for (int set_j = 0; set_j < set_count; ++set_j) {
      const Pair &pair = pairs_[static_cast<std::size_t>(set_i)]
                               [static_cast<std::size_t>(set_j)];
      if (!pair.active || !pair.valid) {
        continue;
      }
      if (set_i == set_j) {
        if (source.fixed_valid) {
          ++last_stats_.same_fixed_pairs;
          last_stats_.neighbor_bytes +=
              source.fixed_neighbors.capacity() * sizeof(int) +
              source.fixed_counts.capacity() * sizeof(int) +
              source.active_neighbors.capacity() * sizeof(int) +
              source.active_counts.capacity() * sizeof(int);
          const IntBuffer &counts =
              (skin_ > 0.0f && source.skin_lists) ? source.active_counts
                                                  : source.fixed_counts;
          for (int count : counts) {
            last_stats_.max_neighbors =
                std::max(last_stats_.max_neighbors, count);
          }
        } else {
          ++last_stats_.same_csr_pairs;
          last_stats_.neighbor_bytes +=
              source.data.row_ptr.capacity() * sizeof(int) +
              source.data.col_idx.capacity() * sizeof(int);
          for (int count : source.data.counts) {
            last_stats_.max_neighbors =
                std::max(last_stats_.max_neighbors, count);
          }
        }
      } else if (pair.fixed_valid) {
        ++last_stats_.cross_fixed_pairs;
        last_stats_.neighbor_bytes +=
            pair.fixed_neighbors.capacity() * sizeof(int) +
            pair.fixed_counts.capacity() * sizeof(int) +
            pair.active_neighbors.capacity() * sizeof(int) +
            pair.active_counts.capacity() * sizeof(int);
        const IntBuffer &counts =
            (skin_ > 0.0f && pair.skin_lists) ? pair.active_counts
                                              : pair.fixed_counts;
        for (int count : counts) {
          last_stats_.max_neighbors =
              std::max(last_stats_.max_neighbors, count);
        }
      } else {
        ++last_stats_.cross_csr_pairs;
        last_stats_.neighbor_bytes +=
            pair.cross.row_ptr.capacity() * sizeof(int) +
            pair.cross.col_idx.capacity() * sizeof(int);
        for (int count : pair.cross.counts) {
          last_stats_.max_neighbors =
              std::max(last_stats_.max_neighbors, count);
        }
      }
    }
  }
}

NeighborList NSearch::get_neighbors(int set_i, int set_j, int point) const {
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

  // The source's sorted mapping is rebuilt by every grid build (and is the
  // identity for sweep sets), so it is valid for same-set and cross rows.
  const int sorted_index =
      source.original_to_sorted[static_cast<std::size_t>(point)];

  if (pair.same_set && source.fixed_valid) {
    if (skin_ > 0.0f && source.skin_lists) {
      const int count =
          source.active_counts[static_cast<std::size_t>(sorted_index)];
      const int *neighbors =
          source.active_neighbors.data() +
          static_cast<std::size_t>(sorted_index) *
              static_cast<std::size_t>(source.fixed_capacity);
      return {reinterpret_cast<const uint32_t *>(neighbors),
              static_cast<std::size_t>(count)};
    }
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
    if (skin_ > 0.0f && pair.skin_lists) {
      const int count =
          pair.active_counts[static_cast<std::size_t>(sorted_index)];
      const int *neighbors =
          pair.active_neighbors.data() +
          static_cast<std::size_t>(sorted_index) *
              static_cast<std::size_t>(pair.fixed_capacity);
      return {reinterpret_cast<const uint32_t *>(neighbors),
              static_cast<std::size_t>(count)};
    }
    const int count =
        pair.fixed_counts[static_cast<std::size_t>(sorted_index)];
    const int *neighbors =
        pair.fixed_neighbors.data() +
        static_cast<std::size_t>(sorted_index) *
            static_cast<std::size_t>(pair.fixed_capacity);
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

int NSearch::get_n_points(int set) const {
  if (!valid_set(set)) {
    return 0;
  }
  return sets_[static_cast<std::size_t>(set)].source_count;
}

void NSearch::set_radius(float radius) {
  if (radius == radius_) {
    return;
  }

  radius_ = radius;
  for (int set = 0; set < static_cast<int>(sets_.size()); ++set) {
    sets_[static_cast<std::size_t>(set)].dirty = true;
    sets_[static_cast<std::size_t>(set)].valid = false;
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
    set.active_valid = false;
    set.skin_lists = false;
  }
  for (std::vector<Pair> &row : pairs_) {
    for (Pair &pair : row) {
      pair.valid = false;
      pair.fixed_valid = false;
      pair.active_valid = false;
      pair.skin_lists = false;
      pair.gpu_rows = false;
      pair.gpu_neighbors = nullptr;
      pair.gpu_offsets = nullptr;
      pair.gpu_nq = 0;
    }
  }
  return true;
}

void NSearch::set_skin(float skin) {
  if (skin < 0.0f || skin == skin_) {
    return;
  }

  skin_ = skin;
  for (int set = 0; set < static_cast<int>(sets_.size()); ++set) {
    sets_[static_cast<std::size_t>(set)].dirty = true;
    sets_[static_cast<std::size_t>(set)].valid = false;
    invalidate_pairs_for_set(set);
  }
}

void NSearch::ensure_grid(PointSet &point_set) {
  if (!point_set.use_sweep) {
    return;
  }
  point_set.use_sweep = false;
  point_set.sweep.clear();
  const float grid_radius = build_radius();
  const float cell_radius = use_small_cells_ ? 0.5f * grid_radius : 0.0f;
  build_point_grid_inplace(point_set.data, point_set.x, point_set.y,
                           point_set.z, grid_radius, cell_radius);
  update_sorted_mapping(point_set.data, point_set.original_to_sorted);
  point_set.valid = true;
  point_set.same_lists_valid = false;
  point_set.fixed_valid = false;
}

void NSearch::set_memory_budget(int64_t bytes) {
  memory_budget_ = std::max<int64_t>(0, bytes);
}

void NSearch::build_point_grid(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  point_set.valid = false;
  point_set.same_lists_valid = false;
  point_set.fixed_valid = false;
  point_set.skin_lists = skin_ > 0.0f;
  point_set.active_valid = false;
  point_set.dirty = false;

  if (sweep::should_use(point_set.bounds, point_set.x.size(),
                        build_radius())) {
    // Sweep sets keep the caller's order: the sorted mapping is the identity
    // and the grid arrays stay empty.
    NeighborSearchData &data = point_set.data;
    const int num_points = static_cast<int>(point_set.x.size());
    data.num_points = num_points;
    data.radius = build_radius();
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
    data.dense_valid = false;
    data.row_ptr.push_back(0);
    data.original_index.resize(static_cast<std::size_t>(num_points));
    std::iota(data.original_index.begin(), data.original_index.end(), 0);
    point_set.original_to_sorted = data.original_index;
    sweep::ensure_index(point_set.sweep, point_set.x);
    point_set.use_sweep = true;
    point_set.valid = true;
    ++last_stats_.sweep_sets;
    return;
  }
  ++last_stats_.grid_sets;
  last_stats_.grid_cells += point_set.data.num_cells;
  last_stats_.dense_cells += static_cast<int>(point_set.data.dense_cells.size());
  if (point_set.data.dense_valid) {
    ++last_stats_.dense_sets;
  } else {
    ++last_stats_.hash_sets;
  }

  point_set.use_sweep = false;
  point_set.sweep.clear();
  const float grid_radius = build_radius();
  const float cell_radius = use_small_cells_ ? 0.5f * grid_radius : 0.0f;
  build_point_grid_inplace(point_set.data, point_set.x, point_set.y,
                           point_set.z, grid_radius, cell_radius);
  update_sorted_mapping(point_set.data, point_set.original_to_sorted);
  point_set.valid = true;
}

void NSearch::build_same_lists(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  if (!point_set.valid) {
    return;
  }
  if (point_set.use_sweep) {
    if (build_sweep_same_lists(set)) {
      return;
    }
    ensure_grid(point_set);
  }
  build_grid_same_lists(set);
}

void NSearch::build_grid_same_lists(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  if (!point_set.valid) {
    return;
  }
  NeighborSearchData &data = point_set.data;
  point_set.fixed_valid = false;
  const plan::SamePlan same_plan =
      plan::same_set(data, point_set.last_fixed_capacity, memory_budget_);

  auto try_fixed = [&](int capacity) {
    switch (same_plan.writer) {
    case plan::SameWriter::PairDense:
      write_neighbors_fixed_query_pair_dense_parallel(
          data, capacity, point_set.fixed_neighbors, point_set.fixed_counts);
      break;
    case plan::SameWriter::Blocked:
      write_neighbors_fixed_query_blocked_parallel(
          data, capacity, point_set.fixed_neighbors, point_set.fixed_counts);
      break;
    case plan::SameWriter::QueryPair:
      write_neighbors_fixed_query_pair_parallel(
          data, capacity, point_set.fixed_neighbors, point_set.fixed_counts);
      break;
    case plan::SameWriter::Active2:
      write_neighbors_fixed_query_active2_parallel(
          data, capacity, point_set.fixed_neighbors, point_set.fixed_counts);
      break;
    }
    const int max_count = max_row_count(point_set.fixed_counts);
    if (max_count > capacity) {
      return false;
    }
    point_set.fixed_capacity = capacity + 8;
    point_set.last_fixed_capacity =
        plan::same_fitted_capacity(max_count, capacity);
    point_set.fixed_valid = true;
    point_set.skin_lists = skin_ > 0.0f;
    point_set.active_valid = false;
    point_set.same_lists_valid = true;
    return true;
  };

  if (same_plan.fixed_rows) {
    switch (same_plan.ranges) {
    case plan::SameRanges::None:
      break;
    case plan::SameRanges::DenseFused:
      build_same_set_cell_ranges_dense_fused_inplace(data);
      break;
    case plan::SameRanges::Merged:
      build_same_set_cell_ranges_inplace(data);
      break;
    }
    if (same_plan.ranges != plan::SameRanges::None &&
        plan::same_set_isolated(data)) {
      point_set.fixed_counts.assign(static_cast<std::size_t>(data.num_points),
                                    0);
      point_set.fixed_capacity = 1;
      point_set.fixed_neighbors.resize(
          static_cast<std::size_t>(data.num_points));
      point_set.fixed_valid = true;
      point_set.skin_lists = false;
      point_set.active_valid = false;
      point_set.same_lists_valid = true;
      return;
    }
    if (try_fixed(same_plan.first_capacity)) {
      return;
    }
    const int grown = plan::same_grown_capacity(
        max_row_count(point_set.fixed_counts), data.num_points,
        memory_budget_);
    if (grown > 0 && try_fixed(grown)) {
      return;
    }
    if (parallel_pairs_) {
      parallel_overflows_.fetch_add(1, std::memory_order_relaxed);
    } else {
      ++last_stats_.fixed_overflows;
    }
    point_set.fixed_neighbors.clear();
    point_set.fixed_counts.clear();
  }

  point_set.skin_lists = false;
  point_set.active_valid = false;
  build_same_set_neighbor_lists_inplace(data, radius_);
  point_set.same_lists_valid = true;
}

bool NSearch::build_sweep_same_lists(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  sweep::ensure_index(point_set.sweep, point_set.x);
  const sweep::Cloud cloud{point_set.x, point_set.y, point_set.z,
                           &point_set.sweep};
  int row_stride = 0;
  if (!sweep::write_fixed_rows(cloud, point_set.original_to_sorted, cloud,
                               true, build_radius(), memory_budget_,
                               point_set.fixed_neighbors,
                               point_set.fixed_counts, row_stride)) {
    return false;
  }
  point_set.fixed_capacity = row_stride;
  point_set.last_fixed_capacity = row_stride;
  point_set.fixed_valid = true;
  point_set.skin_lists = skin_ > 0.0f;
  point_set.active_valid = false;
  point_set.same_lists_valid = true;
  return true;
}

bool NSearch::build_sweep_cross_lists(Pair &pair, PointSet &source,
                                      PointSet &target) {
  sweep::ensure_index(target.sweep, target.x);
  const sweep::Cloud query{source.x, source.y, source.z, &source.sweep};
  const sweep::Cloud cloud{target.x, target.y, target.z, &target.sweep};
  int row_stride = 0;
  if (!sweep::write_fixed_rows(query, source.original_to_sorted, cloud, false,
                               build_radius(), memory_budget_,
                               pair.fixed_neighbors, pair.fixed_counts,
                               row_stride)) {
    return false;
  }
  pair.fixed_capacity = row_stride;
  pair.last_fixed_capacity = row_stride;
  pair.fixed_valid = true;
  pair.skin_lists = skin_ > 0.0f;
  pair.active_valid = false;
  return true;
}

void NSearch::filter_same_active(int set) {
  PointSet &point_set = sets_[static_cast<std::size_t>(set)];
  if (!point_set.fixed_valid || !point_set.skin_lists) {
    point_set.active_valid = false;
    return;
  }

  const int capacity = point_set.fixed_capacity;
  const int num_points = point_set.data.num_points;
  point_set.active_neighbors.resize(static_cast<std::size_t>(num_points) *
                                    static_cast<std::size_t>(capacity));
  point_set.active_counts.resize(static_cast<std::size_t>(num_points));

  const float radius_sq = radius_ * radius_;
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, 256),
      [&](const tbb::blocked_range<int> &range) {
        for (int sorted_index = range.begin(); sorted_index < range.end();
             ++sorted_index) {
          const int original =
              point_set.data.original_index[static_cast<std::size_t>(
                  sorted_index)];
          const float qx = point_set.x[static_cast<std::size_t>(original)];
          const float qy = point_set.y[static_cast<std::size_t>(original)];
          const float qz = point_set.z[static_cast<std::size_t>(original)];
          const int *skin_row =
              point_set.fixed_neighbors.data() +
              static_cast<std::size_t>(sorted_index) *
                  static_cast<std::size_t>(capacity);
          const int skin_count =
              point_set.fixed_counts[static_cast<std::size_t>(sorted_index)];
          int active_count = 0;
          int *active_row =
              point_set.active_neighbors.data() +
              static_cast<std::size_t>(sorted_index) *
                  static_cast<std::size_t>(capacity);
          for (int index = 0; index < skin_count; ++index) {
            const int neighbor = skin_row[index];
            const float dx =
                point_set.x[static_cast<std::size_t>(neighbor)] - qx;
            const float dy =
                point_set.y[static_cast<std::size_t>(neighbor)] - qy;
            const float dz =
                point_set.z[static_cast<std::size_t>(neighbor)] - qz;
            if (dx * dx + dy * dy + dz * dz <= radius_sq) {
              active_row[active_count++] = neighbor;
            }
          }
          point_set.active_counts[static_cast<std::size_t>(sorted_index)] =
              active_count;
        }
      });
  point_set.active_valid = true;
}

void NSearch::filter_cross_active(Pair &pair, int source_set,
                                  int target_set) {
  if (!pair.fixed_valid || !pair.skin_lists) {
    pair.active_valid = false;
    return;
  }

  const PointSet &source = sets_[static_cast<std::size_t>(source_set)];
  const PointSet &target = sets_[static_cast<std::size_t>(target_set)];
  const int capacity = pair.fixed_capacity;
  const int num_points = source.data.num_points;
  pair.active_neighbors.resize(static_cast<std::size_t>(num_points) *
                               static_cast<std::size_t>(capacity));
  pair.active_counts.resize(static_cast<std::size_t>(num_points));

  const float radius_sq = radius_ * radius_;
  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_points, 256),
      [&](const tbb::blocked_range<int> &range) {
        for (int sorted_index = range.begin(); sorted_index < range.end();
             ++sorted_index) {
          const int original =
              source.data.original_index[static_cast<std::size_t>(
                  sorted_index)];
          const float qx = source.x[static_cast<std::size_t>(original)];
          const float qy = source.y[static_cast<std::size_t>(original)];
          const float qz = source.z[static_cast<std::size_t>(original)];
          const int *skin_row =
              pair.fixed_neighbors.data() +
              static_cast<std::size_t>(sorted_index) *
                  static_cast<std::size_t>(capacity);
          const int skin_count =
              pair.fixed_counts[static_cast<std::size_t>(sorted_index)];
          int active_count = 0;
          int *active_row =
              pair.active_neighbors.data() +
              static_cast<std::size_t>(sorted_index) *
                  static_cast<std::size_t>(capacity);
          for (int index = 0; index < skin_count; ++index) {
            const int neighbor = skin_row[index];
            const float dx =
                target.x[static_cast<std::size_t>(neighbor)] - qx;
            const float dy =
                target.y[static_cast<std::size_t>(neighbor)] - qy;
            const float dz =
                target.z[static_cast<std::size_t>(neighbor)] - qz;
            if (dx * dx + dy * dy + dz * dz <= radius_sq) {
              active_row[active_count++] = neighbor;
            }
          }
          pair.active_counts[static_cast<std::size_t>(sorted_index)] =
              active_count;
        }
      });
  pair.active_valid = true;
}

void NSearch::invalidate_pairs_for_set(int set) {
  if (!valid_set(set)) {
    return;
  }

  sets_[static_cast<std::size_t>(set)].same_lists_valid = false;
  sets_[static_cast<std::size_t>(set)].fixed_valid = false;
  sets_[static_cast<std::size_t>(set)].skin_lists = false;
  sets_[static_cast<std::size_t>(set)].active_valid = false;
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
  pair.skin_lists = false;
  pair.active_valid = false;
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
  pair.skin_lists = false;
  pair.active_valid = false;
  pair.gpu_rows = false;
  pair.gpu_neighbors = nullptr;
  pair.gpu_offsets = nullptr;
  pair.gpu_nq = 0;
  pair.cross = CrossNeighborSearchData{};
  pair.fixed_neighbors.clear();
  pair.fixed_counts.clear();
  pair.active_neighbors.clear();
  pair.active_counts.clear();
}

} // namespace apfrnn
