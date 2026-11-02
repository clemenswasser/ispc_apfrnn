#include "apfrnn.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

namespace {

// Scalar reference for sync_points_*_ispc on [begin, end).
template <typename T>
std::vector<float> reference(const std::vector<T> &xyz, int begin, int end,
                             bool comparable, std::vector<float> &x,
                             std::vector<float> &y, std::vector<float> &z) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> out = {0.0f, inf, inf, inf, -inf, -inf, -inf};
  for (int i = begin; i < end; ++i) {
    const float nx = static_cast<float>(xyz[3 * i + 0]);
    const float ny = static_cast<float>(xyz[3 * i + 1]);
    const float nz = static_cast<float>(xyz[3 * i + 2]);
    if (comparable) {
      const float dx = nx - x[i], dy = ny - y[i], dz = nz - z[i];
      out[0] = std::max(out[0], dx * dx + dy * dy + dz * dz);
    }
    x[i] = nx;
    y[i] = ny;
    z[i] = nz;
    out[1] = std::min(out[1], nx);
    out[2] = std::min(out[2], ny);
    out[3] = std::min(out[3], nz);
    out[4] = std::max(out[4], nx);
    out[5] = std::max(out[5], ny);
    out[6] = std::max(out[6], nz);
  }
  return out;
}

template <typename T> void check_sync(int n) {
  std::mt19937 gen(static_cast<uint32_t>(n));
  std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
  std::vector<T> xyz(3 * static_cast<std::size_t>(n));
  for (T &v : xyz) {
    v = static_cast<T>(dist(gen));
  }
  std::vector<float> old_x(n), old_y(n), old_z(n);
  for (int i = 0; i < n; ++i) {
    old_x[i] = dist(gen);
    old_y[i] = dist(gen);
    old_z[i] = dist(gen);
  }

  for (bool comparable : {false, true}) {
    std::vector<float> ref_x = old_x, ref_y = old_y, ref_z = old_z;
    std::vector<float> got_x = old_x, got_y = old_y, got_z = old_z;
    const std::vector<float> expected =
        reference(xyz, 0, n, comparable, ref_x, ref_y, ref_z);
    const apfrnn::SyncResult got =
        apfrnn::sync_points(xyz.data(), n, comparable, got_x, got_y, got_z);
    CHECK(got_x == ref_x);
    CHECK(got_y == ref_y);
    CHECK(got_z == ref_z);
    // Bounds are exact; the displacement may differ in the last bits when
    // ISPC contracts dx * dx + ... into FMAs.
    CHECK(got.max_distance_sq == doctest::Approx(expected[0]).epsilon(1e-6));
    CHECK(got.bounds.min_x == expected[1]);
    CHECK(got.bounds.min_y == expected[2]);
    CHECK(got.bounds.min_z == expected[3]);
    CHECK(got.bounds.max_x == expected[4]);
    CHECK(got.bounds.max_y == expected[5]);
    CHECK(got.bounds.max_z == expected[6]);
  }
}

} // namespace

TEST_CASE("coordinate sync matches the scalar reference") {
  // Partial vectors, a single TBB chunk with a tail, and 33333 points, which
  // TBB splits at offsets that are not a multiple of the vector width.
  for (int n : {1, 5, 17, 1000, 33333}) {
    check_sync<float>(n);
    check_sync<double>(n);
  }
}
