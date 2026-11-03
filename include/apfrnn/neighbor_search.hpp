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
  struct RunStats {
    double sync_build_ms = 0.0;
    double grid_ms = 0.0;
    double same_build_ms = 0.0;
    double same_set_ms = 0.0;
    double cross_set_ms = 0.0;
    double filter_ms = 0.0;
    std::vector<double> pair_ms;
    std::size_t neighbor_bytes = 0;
    std::size_t grid_bytes = 0;
    int same_fixed_pairs = 0;
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
  float skin() const { return skin_; }
  void set_radius(float radius);
  void set_skin(float skin);

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
  float build_radius() const { return radius_ + skin_; }
  int add_source(const void *xyz, bool is_double, int n, bool is_static);
  void resize_source(int set, const void *xyz, bool is_double, int n);
  float sync_source(int set);
  void invalidate_pairs_for_set(int set);
  static void clear_pair(Pair &pair);

  void update_cell_size();
  void sync_dynamic_sets(const std::vector<bool> &is_source);
  void build_point_sets(const std::vector<bool> &is_source,
                        const std::vector<bool> &needs_same_set);
  void build_same_pair(Pair &pair, int set);
  void build_cross_pair(Pair &pair, int source, int target);
  void filter_skin_lists();
  void collect_memory_stats();

  void build_point_grid(int set);
  void ensure_grid(PointSet &point_set);
  void build_same_lists(int set);
  bool build_sweep_same_lists(int set);
  void build_grid_same_lists(int set);
  bool build_sweep_cross_lists(Pair &pair, PointSet &source, PointSet &target);
  void filter_same_active(int set);
  void filter_cross_active(Pair &pair, int source_set, int target_set);

  bool build_gpu_pair(Pair &pair, int source, int target,
                      bool defer_same_copy);
  bool gpu_policy_ok();
  bool max_cell_occupancy(PointSet &set, int &max_points);

  float radius_;
  SearchBackend search_backend_ = SearchBackend::CpuExact;
  SearchBackend effective_backend_ = SearchBackend::CpuExact;
  bool use_small_cells_ = false;
  float skin_ = 0.0f;
  int64_t memory_budget_ = 3LL * 1024 * 1024 * 1024;
  std::vector<PointSet> sets_;
  std::vector<std::vector<Pair>> pairs_;
  std::unique_ptr<GpuState> gpu_;
  RunStats last_stats_;
  bool parallel_pairs_ = false;
  std::atomic<int> parallel_overflows_{0};
};

} // namespace apfrnn
