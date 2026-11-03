// Multi-frame differential fuzz test of NSearch against a reference search.
//
// Each seed builds a random multi-set scene and runs it for several frames,
// mutating it between frames the way callers do: motion (rigid and per-point),
// points jumping far away and back (dense grid <-> hash grid <-> sweep),
// resizes, static toggles, pair toggles, skin, radius and memory-budget
// changes, and sets added after a run. Every active pair is checked against
// the reference on every frame, so state that leaks from one frame's plan
// into the next is caught, not only first-frame results.
//
// RunStats::paths records the code paths each run took. Across all seeds the
// test requires every CPU path to be taken, and to be both entered and left
// between consecutive frames, so a new path (or a path that silently became
// unreachable) shows up as a test failure.
//
//   APFRNN_FUZZ_SEEDS=N   number of seeds (default 96; fewer skips the
//                         path checks)
//   APFRNN_FUZZ_FIRST_SEED=F  start at seed F (to soak a new range)
//   APFRNN_FUZZ_SEED=S    run only seed S (to replay a failure)
//   APFRNN_FUZZ_TRACE=1   print each seed's scene and per-frame operations

#include "point_cloud.hpp"
#include "scene.hpp"
#include "apfrnn/neighbor_search.hpp"

#include <doctest/doctest.h>

#include <iostream>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using apfrnn::NSearch;
using apfrnn::support::PointCloud;

int env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  return value != nullptr ? std::atoi(value) : fallback;
}

// Candidate neighbors from a hashed grid of radius-sized cells, decided with
// the float formula of the brute-force helpers. Pairs within float rounding
// of the radius may be reported either way, so they are returned as `maybe`.
class Reference {
public:
  Reference(const PointCloud &target, float radius) : target_(target), r_(radius) {
    for (int i = 0; i < static_cast<int>(target.x.size()); ++i) {
      cells_[key(cell(target.x[i]), cell(target.y[i]), cell(target.z[i]))]
          .push_back(i);
    }
  }

  void query(float qx, float qy, float qz, int skip, std::vector<int> &must,
             std::vector<int> &maybe) const {
    must.clear();
    maybe.clear();
    const int64_t cx = cell(qx), cy = cell(qy), cz = cell(qz);
    const float r2 = r_ * r_;
    for (int64_t dx = -1; dx <= 1; ++dx) {
      for (int64_t dy = -1; dy <= 1; ++dy) {
        for (int64_t dz = -1; dz <= 1; ++dz) {
          const auto found = cells_.find(key(cx + dx, cy + dy, cz + dz));
          if (found == cells_.end()) {
            continue;
          }
          for (int i : found->second) {
            if (i == skip) {
              continue;
            }
            const float tx = target_.x[i], ty = target_.y[i], tz = target_.z[i];
            const float ex = qx - tx, ey = qy - ty, ez = qz - tz;
            const float d2 = ex * ex + ey * ey + ez * ez;
            const float scale = std::max(
                {std::fabs(qx), std::fabs(qy), std::fabs(qz), std::fabs(tx),
                 std::fabs(ty), std::fabs(tz), r_});
            const float band = 64.0f * FLT_EPSILON * scale * scale;
            if (d2 <= r2 - band) {
              must.push_back(i);
            } else if (d2 <= r2 + band) {
              maybe.push_back(i);
            }
          }
        }
      }
    }
  }

private:
  int64_t cell(float v) const {
    return static_cast<int64_t>(std::floor(static_cast<double>(v) / r_));
  }
  struct Key {
    int64_t x, y, z;
    bool operator==(const Key &other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key &k) const {
      uint64_t h = static_cast<uint64_t>(k.x) * 0x9e3779b97f4a7c15ULL;
      h ^= static_cast<uint64_t>(k.y) * 0xc2b2ae3d27d4eb4fULL + (h << 6);
      h ^= static_cast<uint64_t>(k.z) * 0x165667b19e3779f9ULL + (h >> 2);
      return static_cast<std::size_t>(h);
    }
  };
  static Key key(int64_t x, int64_t y, int64_t z) { return {x, y, z}; }

  const PointCloud &target_;
  float r_;
  std::unordered_map<Key, std::vector<int>, KeyHash> cells_;
};

struct FuzzSet {
  PointCloud cloud;
  bool is_double = false;
  bool is_static = false;
  std::vector<float> xyz_f;
  std::vector<double> xyz_d;
};

class Fuzzer {
public:
  explicit Fuzzer(uint32_t seed)
      : seed_(seed), rng_(seed * 2654435761u + 1),
        trace_(env_int("APFRNN_FUZZ_TRACE", 0) != 0) {}

  // Returns false (after reporting) on the first wrong frame.
  bool run(uint64_t &paths_seen, uint64_t &paths_entered,
           uint64_t &paths_left) {
    radius_ = pick({0.6f, 1.0f, 1.0f, 1.7f});
    search_ = std::make_unique<NSearch>(radius_);
    const int sets = uniform(1, 5);
    for (int set = 0; set < sets; ++set) {
      add_set();
    }
    randomize_pairs(0.6);

    uint64_t previous = 0;
    const int frames = uniform(4, 9);
    for (int frame = 0; frame < frames; ++frame) {
      if (frame > 0) {
        mutate();
      }
      search_->run();
      const uint64_t paths = search_->last_stats().paths;
      log() << "frame " << frame << " radius " << radius_ << " skin " << skin_
            << " budget " << budget_ << " paths:" << describe(paths) << "\n";
      paths_seen |= paths;
      if (frame > 0) {
        paths_entered |= paths & ~previous;
        paths_left |= previous & ~paths;
      }
      previous = paths;
      if (!check(frame)) {
        return false;
      }
    }
    return true;
  }

private:
  int uniform(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng_);
  }
  float uniform_f(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng_);
  }
  bool chance(double p) { return std::bernoulli_distribution(p)(rng_); }
  template <typename T> T pick(std::initializer_list<T> values) {
    return *(values.begin() + uniform(0, static_cast<int>(values.size()) - 1));
  }

  PointCloud make_cloud() {
    namespace b = apfrnn::bench;
    const int n = pick({0, 1, 2, 17, 300, 1500, 2500, 4000});
    // Points per unit volume at radius 1: ~4.2x this many neighbors, and
    // 5e-5 is sparse enough for the sweep backend.
    const float density = pick({5e-5f, 0.05f, 0.4f, 1.5f, 6.0f});
    const float side =
        std::max(1.0f, std::cbrt(static_cast<float>(std::max(n, 1)) / density));
    const uint32_t seed = static_cast<uint32_t>(rng_());
    const int distribution = uniform(0, 7);
    log() << "  cloud n " << n << " density " << density << " distribution "
          << distribution << "\n";
    switch (distribution) {
    case 0: return b::make_clustered(n, side, seed);
    case 1: return b::make_slab(n, side, seed);
    case 2: return b::make_surface(n, side, seed, true);
    case 3: return b::make_two_scale(n, side, seed);
    case 4: return b::make_jittered_lattice(n, side, seed, chance(0.5));
    case 5: return b::make_pathological(std::min(n, 1200), seed);
    default: return b::make_uniform(n, side, seed);
    }
  }

  void add_set() {
    FuzzSet set;
    set.cloud = make_cloud();
    set.is_double = chance(0.25);
    set.is_static = chance(0.3);
    sets_.push_back(std::move(set));
    FuzzSet &added = sets_.back();
    fill(added);
    const int n = size(added);
    const int id = added.is_double
                       ? search_->add_point_set(added.xyz_d.data(), n,
                                                added.is_static)
                       : search_->add_point_set(added.xyz_f.data(), n,
                                                added.is_static);
    REQUIRE(id == static_cast<int>(sets_.size()) - 1);
  }

  static int size(const FuzzSet &set) {
    return static_cast<int>(set.cloud.x.size());
  }

  // Writes the cloud into the set's buffer in place (same size, same
  // pointer) or into a fresh buffer.
  static void fill(FuzzSet &set) {
    const std::size_t n = set.cloud.x.size();
    if (set.is_double) {
      set.xyz_d.resize(3 * n);
    } else {
      set.xyz_f.resize(3 * n);
    }
    for (std::size_t i = 0; i < n; ++i) {
      const float v[3] = {set.cloud.x[i], set.cloud.y[i], set.cloud.z[i]};
      for (int k = 0; k < 3; ++k) {
        if (set.is_double) {
          set.xyz_d[3 * i + k] = v[k];
        } else {
          set.xyz_f[3 * i + k] = v[k];
        }
      }
    }
  }

  // Applies a changed cloud: dynamic sets of unchanged size are updated in
  // place, everything else (static sets, size changes, sometimes on purpose)
  // goes through resize_point_set with a new buffer.
  void commit(int id, bool size_changed) {
    FuzzSet &set = sets_[static_cast<std::size_t>(id)];
    if (!set.is_static && !size_changed && chance(0.8)) {
      fill(set);
      return;
    }
    std::vector<float>().swap(set.xyz_f);
    std::vector<double>().swap(set.xyz_d);
    fill(set);
    if (set.is_double) {
      search_->resize_point_set(id, set.xyz_d.data(), size(set));
    } else {
      search_->resize_point_set(id, set.xyz_f.data(), size(set));
    }
  }

  // Every pair is set explicitly, so the tracked matrix is the search's.
  void randomize_pairs(double p) {
    const int n = static_cast<int>(sets_.size());
    active_.assign(static_cast<std::size_t>(n * n), false);
    search_->set_all_searches(false);
    for (int q = 0; q < n; ++q) {
      for (int t = 0; t < n; ++t) {
        const bool on = chance(p);
        active_[static_cast<std::size_t>(q * n + t)] = on;
        search_->set_active_search(q, t, on);
      }
    }
  }

  bool active(int q, int t) const {
    const int n = static_cast<int>(sets_.size());
    return active_[static_cast<std::size_t>(q * n + t)];
  }

  void mutate() {
    const int count = static_cast<int>(sets_.size());
    for (int id = 0; id < count; ++id) {
      FuzzSet &set = sets_[static_cast<std::size_t>(id)];
      PointCloud &c = set.cloud;
      const int n = size(set);
      bool changed = false;
      bool size_changed = false;
      const int op = uniform(0, 9);
      log() << "  set " << id << " op " << op << "\n";
      switch (op) {
      case 0:
      case 1: {  // per-point jitter, sometimes within a skin
        const float amount = radius_ * pick({0.01f, 0.1f, 0.6f});
        for (int i = 0; i < n; ++i) {
          c.x[i] += uniform_f(-amount, amount);
          c.y[i] += uniform_f(-amount, amount);
          c.z[i] += uniform_f(-amount, amount);
        }
        changed = n > 0;
        break;
      }
      case 2: {  // rigid motion
        const float dx = uniform_f(-2.0f, 2.0f) * radius_;
        for (int i = 0; i < n; ++i) {
          c.x[i] += dx;
          c.z[i] -= 0.5f * dx;
        }
        changed = n > 0;
        break;
      }
      case 3:  // one point far away: hash grid for large sets
        if (n > 0) {
          c.x[0] = c.y[0] = c.z[0] = 300.0f * radius_;
          changed = true;
        }
        break;
      case 4:  // extreme outlier: sweep or hash
        if (n > 0) {
          apfrnn::support::set_extreme_outlier(c, 0);
          changed = true;
        }
        break;
      case 5:  // a fresh cloud, usually of another size
        c = make_cloud();
        changed = true;
        size_changed = size(set) != n;
        break;
      case 6:  // static toggle (re-reads the set)
        set.is_static = !set.is_static;
        search_->set_point_set_static(id, set.is_static);
        break;
      default:  // unchanged
        break;
      }
      if (changed) {
        commit(id, size_changed);
      }
      log() << "    n " << size(set) << (set.is_static ? " static" : "")
            << (changed ? " changed" : "") << "\n";
    }

    const int global = uniform(0, 11);
    log() << "  global op " << global << "\n";
    switch (global) {
    case 0: randomize_pairs(0.6); break;
    case 1:
      skin_ = chance(0.5) ? 0.0f : 0.3f * radius_;
      search_->set_skin(skin_);
      break;
    case 2:
      radius_ = pick({0.6f, 1.0f, 1.7f});
      search_->set_radius(radius_);
      break;
    case 3:  // forces the CSR and block-row fallbacks
      // Small budgets force CSR. Block rows need a budget between a
      // high-degree set's exact rows and its fixed rows, which depends on
      // the scene, so the budget steps through that range.
      budget_ = chance(0.25) ? (int64_t(3) << 30)
                             : (int64_t(1) << (16 + 2 * uniform(0, 5)));
      search_->set_memory_budget(budget_);
      break;
    case 4:
      if (sets_.size() < 6) {
        add_set();
        randomize_pairs(0.6);
      }
      break;
    case 5:
      search_->set_all_searches(true);
      active_.assign(active_.size(), true);
      break;
    default: break;
    }
  }

  // The (sorted) neighbors of one query from a new NSearch over the current
  // sets, pairs and settings, run once.
  std::vector<int> fresh_neighbors(int q, int t, int i) const {
    NSearch fresh(radius_);
    fresh.set_skin(skin_);
    fresh.set_memory_budget(budget_);
    for (const FuzzSet &set : sets_) {
      if (set.is_double) {
        fresh.add_point_set(set.xyz_d.data(), size(set), set.is_static);
      } else {
        fresh.add_point_set(set.xyz_f.data(), size(set), set.is_static);
      }
    }
    const int count = static_cast<int>(sets_.size());
    fresh.set_all_searches(false);
    for (int a = 0; a < count; ++a) {
      for (int b = 0; b < count; ++b) {
        fresh.set_active_search(a, b, active(a, b));
      }
    }
    fresh.run();
    const apfrnn::NeighborList list = fresh.get_neighbors(q, t, i);
    std::vector<int> neighbors(list.begin(), list.end());
    std::sort(neighbors.begin(), neighbors.end());
    return neighbors;
  }

  // What is wrong with `got` (sorted) against the reference, or "".
  static std::string verdict(const std::vector<int> &got,
                             const std::vector<int> &must,
                             const std::vector<int> &maybe) {
    std::vector<int> extra;
    std::set_difference(got.begin(), got.end(), must.begin(), must.end(),
                        std::back_inserter(extra));
    std::string out;
    if (std::adjacent_find(got.begin(), got.end()) != got.end()) {
      out += ", duplicates";
    }
    if (!std::includes(got.begin(), got.end(), must.begin(), must.end())) {
      out += ", missing";
    }
    if (!std::includes(maybe.begin(), maybe.end(), extra.begin(),
                       extra.end())) {
      out += ", outside the radius";
    }
    return out;
  }

  bool check(int frame) {
    const int count = static_cast<int>(sets_.size());
    std::vector<int> must, maybe, got;
    for (int t = 0; t < count; ++t) {
      const PointCloud &target = sets_[static_cast<std::size_t>(t)].cloud;
      const Reference reference(target, radius_);
      for (int q = 0; q < count; ++q) {
        if (!active(q, t)) {
          continue;
        }
        const PointCloud &query = sets_[static_cast<std::size_t>(q)].cloud;
        for (int i = 0; i < static_cast<int>(query.x.size()); ++i) {
          reference.query(query.x[i], query.y[i], query.z[i],
                          q == t ? i : -1, must, maybe);
          const apfrnn::NeighborList list = search_->get_neighbors(q, t, i);
          got.assign(list.begin(), list.end());
          std::sort(got.begin(), got.end());
          std::sort(must.begin(), must.end());
          std::sort(maybe.begin(), maybe.end());
          const std::string wrong = verdict(got, must, maybe);
          if (!wrong.empty()) {
            // Whether a fresh search over the same state is also wrong tells
            // a single-frame bug from state carried over between frames.
            const std::string fresh =
                verdict(fresh_neighbors(q, t, i), must, maybe);
            std::ostringstream message;
            message << "seed " << seed_ << " frame " << frame << " pair (" << q
                    << ", " << t << ") point " << i << ": got " << got.size()
                    << " neighbors, expected " << must.size() << " (+"
                    << maybe.size() << " on the radius)" << wrong
                    << "; a fresh search over this state is "
                    << (fresh.empty() ? "correct" : "wrong too")
                    << "; paths:" << describe(search_->last_stats().paths)
                    << "; replay with APFRNN_FUZZ_SEED=" << seed_;
            FAIL_CHECK(message.str());
            return false;
          }
        }
      }
    }
    return true;
  }

public:
  static std::string describe(uint64_t paths) {
    std::string out;
    for (int bit = 0; bit < NSearch::kPathBits; ++bit) {
      if ((paths >> bit) & 1u) {
        out += " ";
        out += NSearch::path_name(1ull << bit);
      }
    }
    return out;
  }

private:
  // Trace output when APFRNN_FUZZ_TRACE is set.
  std::ostream &log() {
    static std::ostream null(nullptr);
    return trace_ ? std::cerr : null;
  }

  uint32_t seed_;
  std::mt19937 rng_;
  bool trace_ = false;
  float radius_ = 1.0f;
  float skin_ = 0.0f;
  int64_t budget_ = 3LL << 30;
  std::unique_ptr<NSearch> search_;
  std::vector<FuzzSet> sets_;
  std::vector<bool> active_;  // row-major (query, target)
};

} // namespace

TEST_CASE("multi-frame fuzz matches the reference and reaches every path") {
  const int replay = env_int("APFRNN_FUZZ_SEED", -1);
  const int seeds = replay >= 0 ? 1 : env_int("APFRNN_FUZZ_SEEDS", 96);
  const int first = replay >= 0 ? replay : env_int("APFRNN_FUZZ_FIRST_SEED", 0);
  uint64_t seen = 0, entered = 0, left = 0;
  int failures = 0;
  for (int seed = first; seed < first + seeds; ++seed) {
    Fuzzer fuzzer(static_cast<uint32_t>(seed));
    if (!fuzzer.run(seen, entered, left)) {
      ++failures;
    }
  }
  CHECK(failures == 0);
  if (replay >= 0 || seeds < 96) {
    return;  // a partial run cannot be expected to reach every path
  }

  // The GPU path needs a SYCL build and a device; everything else is CPU.
  const uint64_t required = ((1ull << NSearch::kPathBits) - 1) &
                            ~static_cast<uint64_t>(NSearch::kPathGpu);
  INFO("never taken:" << Fuzzer::describe(required & ~seen));
  INFO("never entered between frames:"
       << Fuzzer::describe(required & ~entered));
  INFO("never left between frames:" << Fuzzer::describe(required & ~left));
  CHECK((required & ~seen) == 0);
  CHECK((required & ~entered) == 0);
  CHECK((required & ~left) == 0);
}
