#pragma once

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace apfrnn {

// Lightweight non-owning view of one CSR neighbor row. The returned indices
// are original point indices in the destination point set.
struct NeighborList {
  const uint32_t *data = nullptr;
  std::size_t count = 0;

  const uint32_t *begin() const { return data; }
  const uint32_t *end() const { return data == nullptr ? nullptr : data + count; }
  std::size_t size() const { return count; }
  bool empty() const { return count == 0; }
};

// The default backend can be preselected with APFRNN_BACKEND=cpu|gpu|auto.
enum class SearchBackend {
  CpuExact,
  GpuExact,
  Auto,  // GPU only for large, low-occupancy frames; CPU otherwise
};

// Native multi-set fixed-radius neighbor search for the learnsph workload.
//
// Supported subset:
//   - 3D, fixed radius, float search (double input is converted to float)
//   - arbitrary number of point sets
//   - per-pair activation (search set i against set j)
//   - static point sets whose acceleration data is cached across runs
//   - dynamic point sets, resize, and in-place updates
//   - same-set and cross-set CSR neighbor lists
//
// Not part of this subset: variable per-point radii, periodic boundaries,
// user-side z-sorting, and top-k queries.
class NSearch {
public:
  // Code paths one run() can take, as bits of RunStats::paths. Tests use them
  // to check that every path (and every change of path between frames) is
  // exercised; the bits are diagnostics, not a stable API.
  enum Path : uint64_t {
    kPathGridDense = 1ull << 0,       // a set built a dense cell lookup
    kPathGridHash = 1ull << 1,        // a set built a hashed cell lookup
    kPathSweep = 1ull << 2,           // a set uses the sweep backend
    kPathSmallCells = 1ull << 3,      // half-radius cells for this run
    kPathRangeCacheBuilt = 1ull << 4, // a set built the shared range cache
    kPathRangeCacheShared = 1ull << 5,  // a pair read the shared cache
    kPathTargetReuse = 1ull << 6,     // an unchanged target kept its grid
    kPathParallelBuild = 1ull << 7,   // small sets were built in parallel
    kPathParallelPairs = 1ull << 8,   // small pairs ran as one parallel job
    kPathSameSweep = 1ull << 9,
    kPathSameBlockRows = 1ull << 10,
    kPathSameIsolated = 1ull << 11,
    kPathSameRangesDenseFused = 1ull << 12,
    kPathSameRangesMerged = 1ull << 13,
    kPathSamePairDense = 1ull << 14,
    kPathSameBlocked = 1ull << 15,
    kPathSameQueryOct = 1ull << 16,
    kPathSameActive2 = 1ull << 17,
    kPathSameGrown = 1ull << 18,      // fixed rows overflowed and regrew
    kPathSameCsr = 1ull << 19,
    kPathSameSparseKeys = 1ull << 30, // merged sorted cell-key stencil
    kPathCrossSweep = 1ull << 20,
    kPathCrossNoCandidates = 1ull << 21,
    kPathCrossTargetMajor = 1ull << 22,
    kPathCrossPairDense = 1ull << 23,
    kPathCrossQueryPair = 1ull << 24,
    kPathCrossBlocked = 1ull << 25,
    kPathCrossActive2 = 1ull << 26,
    kPathCrossGrown = 1ull << 27,
    kPathCrossCsr = 1ull << 28,
    kPathGpu = 1ull << 29,
    kPathCrossCellRange = 1ull << 31, // sorted-cell range search
  };
  static constexpr int kPathBits = 32;
  // Short name of one Path bit, e.g. "same_csr".
  static const char *path_name(uint64_t path);

  struct RunStats {
    double sync_build_ms = 0.0;
    double grid_ms = 0.0;
    double same_build_ms = 0.0;
    double same_set_ms = 0.0;
    double cross_set_ms = 0.0;
    std::vector<double> pair_ms;
    std::size_t neighbor_bytes = 0;
    std::size_t grid_bytes = 0;
    int same_fixed_pairs = 0;
    int same_block_pairs = 0;
    int same_csr_pairs = 0;
    int cross_fixed_pairs = 0;
    int cross_csr_pairs = 0;
    int fixed_overflows = 0;
    int max_neighbors = 0;
    int rebuilt_sets = 0;
    int reused_sets = 0;
    int grid_sets = 0;
    int sweep_sets = 0;
    int grid_cells = 0;
    int dense_cells = 0;
    int dense_sets = 0;
    int hash_sets = 0;
    float max_displacement_sq = 0.0f;
    // Bits of Path for every code path this run took.
    uint64_t paths = 0;
  };

  explicit NSearch(float radius);
  ~NSearch();

  NSearch(NSearch const &) = delete;
  NSearch &operator=(NSearch const &) = delete;

  int add_point_set(const float *xyz, int n, bool is_static = false);
  int add_point_set(const double *xyz, int n, bool is_static = false);

  void resize_point_set(int set, const float *xyz, int n);
  void resize_point_set(int set, const double *xyz, int n);

  void set_point_set_static(int set, bool is_static = true);
  void set_active_search(int set_i, int set_j, bool active = true);
  void set_all_searches(bool active);

  // Marks non-static point sets dirty. run() does this automatically, so this
  // is only needed for callers that use the explicit updating convention.
  void update_point_sets();

  void run();

  NeighborList get_neighbors(int set_i, int set_j, int point) const;

  int get_n_points(int set) const;
  int n_sets() const;
  float radius() const { return radius_; }
  void set_radius(float radius);

  // GPU Exact is opt-in and statically linked only in an
  // APFRNN_ENABLE_SYCL=ON build. Returns false when the GPU backend is not
  // available; the search stays on CPU Exact.
  bool set_search_backend(SearchBackend backend);
  SearchBackend search_backend() const { return search_backend_; }
  bool gpu_available() const;
  // Upper bound for one same-set or cross-set fixed-row allocation. Defaults
  // to 3 GiB. Lower it for memory-constrained callers; larger requests fall
  // back to CSR.
  void set_memory_budget(int64_t bytes);
  int64_t memory_budget() const { return memory_budget_; }

  RunStats last_stats() const { return last_stats_; }

  // Compatibility shims for the learnsph adapter. APFRNN always keeps an
  // internal cell order and maps results back to the caller's order, so these
  // are intentionally no-ops.
  void zsort() {}
  void prepare_zsort() {}
  template <typename T> void apply_zsort(int, T *) {}

private:
  // Defined in neighbor_search.cpp; they hold the internal grid types.
  struct PointSet;
  struct Pair;
  struct GpuState;

  bool valid_set(int set) const;
  // True when this grid set is only ever a source (never a cross target) and
  // is not dense, so its sorted coordinates can stay lazy.
  bool lazy_sorted_coords(const PointSet &point_set) const;
  // True when this run skipped a position copy for a large moving source; the
  // per-frame block-layout scan assumes a static scene and must not run.
  bool skipped_position_sync_ = false;
  int add_source(const void *xyz, bool is_double, int n, bool is_static);
  void resize_source(int set, const void *xyz, bool is_double, int n);
  float sync_source(int set, bool skip_positions = false);
  void invalidate_pairs_for_set(int set);
  static void clear_pair(Pair &pair);

  void update_cell_size();
  void sync_dynamic_sets(const std::vector<bool> &is_source);
  void build_point_sets(const std::vector<bool> &is_source,
                        const std::vector<bool> &needs_same_set);
  void build_same_pair(Pair &pair, int set);
  void build_cross_pair(Pair &pair, int source, int target);
  // Writes the reverse of a built cross pair by transposing its fixed rows;
  // the distance test is symmetric, so both lists describe the same pairs.
  bool transpose_cross_pair(Pair &forward, Pair &reverse, PointSet &source,
                            PointSet &target);
  void collect_memory_stats();

  void build_point_grid(int set);
  // Converts a sweep set to a grid; `building` is the pair being built.
  void ensure_grid(PointSet &point_set, const Pair *building = nullptr);
  void build_same_lists(int set);
  bool build_sweep_same_lists(int set);
  void build_grid_same_lists(int set);
  bool build_moving_block_rows(int set);
  bool build_same_block_rows(int set, int min_block = 1,
                             bool counts_ready = false);
  bool build_cross_block_rows(Pair &pair, PointSet &source, PointSet &target,
                              bool pair_queries, bool dense_ranges);
  bool remember_cross_block_rows(Pair &pair, PointSet &source,
                                 PointSet &target, int capacity);
  bool build_sweep_cross_lists(Pair &pair, PointSet &source, PointSet &target);
  static bool cell_range_cross_candidate(const PointSet &source,
                                         const PointSet &target);

  bool build_gpu_pair(Pair &pair, int source, int target,
                      bool defer_same_copy);
  bool gpu_policy_ok();
  bool max_cell_occupancy(PointSet &set, int &max_points);
  void note_path(uint64_t path) {
    path_bits_.fetch_or(path, std::memory_order_relaxed);
  }

  float radius_;
  SearchBackend search_backend_ = SearchBackend::CpuExact;
  SearchBackend effective_backend_ = SearchBackend::CpuExact;
  bool use_small_cells_ = false;
  int64_t memory_budget_ = 3LL * 1024 * 1024 * 1024;
  std::vector<PointSet> sets_;
  std::vector<std::vector<Pair>> pairs_;
  std::unique_ptr<GpuState> gpu_;
  RunStats last_stats_;
  bool parallel_pairs_ = false;
  std::atomic<int> parallel_overflows_{0};
  // Path bits of the current run; pairs and set builds may run in parallel.
  std::atomic<uint64_t> path_bits_{0};
};

} // namespace apfrnn
