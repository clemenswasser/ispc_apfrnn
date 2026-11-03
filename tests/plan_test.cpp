#include "plan.hpp"

#include <doctest/doctest.h>

namespace {

// Grid statistics only; the plan never looks at point data.
apfrnn::NeighborSearchData grid(int num_points, int num_cells,
                                int max_cell_points) {
  apfrnn::NeighborSearchData data;
  data.num_points = num_points;
  data.num_cells = num_cells;
  data.max_cell_points = max_cell_points;
  data.dense_valid = true;
  return data;
}

constexpr int64_t kBudget = 3LL * 1024 * 1024 * 1024;

} // namespace

TEST_CASE("same-set plan picks the on-the-fly pair writer for moderate density") {
  const auto data = grid(200000, 20000, 30);  // 10 points per cell
  const auto plan = apfrnn::plan::same_set(data, 0, kBudget);
  CHECK(plan.fixed_rows);
  CHECK(plan.first_capacity == 128);
  CHECK(plan.ranges == apfrnn::plan::SameRanges::None);
  CHECK(plan.writer == apfrnn::plan::SameWriter::PairDense);
}

TEST_CASE("same-set plan falls back from the pair writers on long rows") {
  const auto data = grid(200000, 20000, 30);
  // A previous frame needed 300 slots: too long for the pair writers.
  const auto plan = apfrnn::plan::same_set(data, 300, kBudget);
  CHECK(plan.first_capacity == 300);
  CHECK(plan.ranges == apfrnn::plan::SameRanges::DenseFused);
  CHECK(plan.writer == apfrnn::plan::SameWriter::Active2);
}

TEST_CASE("same-set plan uses the blocked writer when blocks exist") {
  auto data = grid(200000, 1000, 2000);
  data.blocks_valid = true;
  const auto plan = apfrnn::plan::same_set(data, 0, kBudget);
  CHECK(plan.ranges == apfrnn::plan::SameRanges::Merged);
  CHECK(plan.writer == apfrnn::plan::SameWriter::Blocked);
}

TEST_CASE("same-set plan resets the first capacity to fit the budget") {
  const auto data = grid(1000, 100, 30);
  const int64_t budget = 1000LL * (512 + 8) * 4 - 1;
  CHECK(apfrnn::plan::same_set(data, 512, budget).first_capacity == 128);
  CHECK(apfrnn::plan::same_set(data, 512, budget + 1).first_capacity == 512);
}

TEST_CASE("cross plan picks target-major for sparse dense targets") {
  const auto source = grid(200000, 20000, 30);
  const auto sparse_target = grid(1000, 900, 2);
  const auto dense_target = grid(200000, 20000, 30);
  CHECK(apfrnn::plan::cross_set(source, 64, sparse_target, 0, kBudget).writer ==
        apfrnn::plan::CrossWriter::TargetMajor);
  const auto plan = apfrnn::plan::cross_set(source, 64, dense_target, 0, kBudget);
  CHECK(plan.writer == apfrnn::plan::CrossWriter::PairDense);
  CHECK_FALSE(plan.build_ranges);
  CHECK(plan.first_capacity == 32);
}

TEST_CASE("cross plan uses ranges and active2 outside the pair regime") {
  const auto source = grid(20000, 200, 400);  // 100 points per cell
  const auto target = grid(20000, 200, 400);
  const auto plan = apfrnn::plan::cross_set(source, 0, target, 9000, kBudget);
  CHECK(plan.build_ranges);
  CHECK(plan.writer == apfrnn::plan::CrossWriter::Active2);
  CHECK(plan.first_capacity == 4096);  // clamped to the cross maximum
}

TEST_CASE("fitted capacities keep a margin for degree changes") {
  CHECK(apfrnn::plan::same_fitted_capacity(10, 128) == 32);
  CHECK(apfrnn::plan::same_fitted_capacity(70, 128) == 80);
  CHECK(apfrnn::plan::same_fitted_capacity(8000, 16384) == 8504);
  CHECK(apfrnn::plan::cross_fitted_capacity(70, 128) == 80);
  CHECK(apfrnn::plan::cross_fitted_capacity(70, 64) == 64);
}

TEST_CASE("grown capacities respect limits and budget") {
  CHECK(apfrnn::plan::same_grown_capacity(130, 1000, kBudget) == 192);
  CHECK(apfrnn::plan::same_grown_capacity(40000, 1000, kBudget) == 0);
  CHECK(apfrnn::plan::cross_grown_capacity(5000, 1000, kBudget) == 0);
  CHECK(apfrnn::plan::cross_grown_capacity(130, 1000, 1000) == 0);
}

TEST_CASE("block row budget adds the 64 MiB slack after the per-point term") {
  constexpr int64_t kSlack = int64_t(64) << 20;
  CHECK(apfrnn::plan::block_output_budget(0, 0) == kSlack);
  CHECK(apfrnn::plan::block_output_budget(1000, 10) ==
        1000 * 4 + 10 * 32 + kSlack);
  // The slack must not multiply the exact-neighbor term.
  CHECK(apfrnn::plan::block_output_budget(1, 1) < kSlack + 1024);
}
