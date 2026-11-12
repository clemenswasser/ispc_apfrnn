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

// Smallest capacity >= n whose row stride (capacity + row_slack()) is a whole
// number of 64-byte lines. The slack is the gang width, so the expected
// capacities depend on the ISPC target.
int line_capacity(int n) {
  const int slack = apfrnn::row_slack();
  return (n + slack + 15) / 16 * 16 - slack;
}

} // namespace

TEST_CASE("same-set plan picks the on-the-fly pair writer for moderate density") {
  const auto data = grid(200000, 20000, 30);  // 10 points per cell
  const auto plan = apfrnn::plan::same_set(data, 0, kBudget);
  CHECK(plan.fixed_rows);
  CHECK(plan.first_capacity == 128);
  CHECK(plan.ranges == apfrnn::plan::SameRanges::None);
  CHECK(plan.writer == apfrnn::plan::SameWriter::PairDense);
}

TEST_CASE("same-set plan uses eight queries for a sparse hash grid") {
  auto data = grid(200000, 38170, 5);
  data.dense_valid = false;
  const auto plan = apfrnn::plan::same_set(data, 0, kBudget);
  CHECK(plan.ranges == apfrnn::plan::SameRanges::Merged);
  CHECK(plan.writer == apfrnn::plan::SameWriter::QueryOct);
}

TEST_CASE("same-set plan uses the grouped writer on long rows") {
  const auto data = grid(200000, 20000, 30);
  // A previous frame needed 300 slots.
  const auto plan = apfrnn::plan::same_set(data, 300, kBudget);
  CHECK(plan.first_capacity == 300);
  CHECK(plan.ranges == apfrnn::plan::SameRanges::DenseFused);
  CHECK(plan.writer == apfrnn::plan::SameWriter::QueryOct);
}

TEST_CASE("same-set plan keeps the single-query writer for tiny sets") {
  auto data = grid(512, 512, 5);
  data.dense_valid = false;
  const auto plan = apfrnn::plan::same_set(data, 0, kBudget);
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
  const int64_t budget = 1000LL * (512 + apfrnn::row_slack()) * 4 - 1;
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
  CHECK(plan.first_capacity == line_capacity(32));
}

TEST_CASE("cross plan pairs a low-density source with a dense target") {
  const auto source = grid(20000, 8000, 30);  // 2.5 points per cell
  const auto target = grid(20000, 2000, 30);  // 10 points per cell
  const auto plan =
      apfrnn::plan::cross_set(source, 32, target, 0, kBudget);
  CHECK_FALSE(plan.build_ranges);
  CHECK(plan.writer == apfrnn::plan::CrossWriter::PairDense);
}

TEST_CASE("cross plan uses ranges and active2 outside the pair regime") {
  const auto source = grid(20000, 200, 400);  // 100 points per cell
  const auto target = grid(20000, 200, 400);
  const auto plan = apfrnn::plan::cross_set(source, 0, target, 9000, kBudget);
  CHECK(plan.build_ranges);
  CHECK(plan.writer == apfrnn::plan::CrossWriter::Active2);
  // Clamped to the cross maximum, then rounded up so the row stride
  // (capacity + row_slack()) is a whole number of 64-byte lines.
  CHECK(plan.first_capacity == line_capacity(4096));
}

TEST_CASE("per-point rows are placed only when the table exceeds a cache") {
  // A table of a few rows is read from cache in any order; one of a gigabyte
  // is not. The test binary lowers the limit (see below), so set one here.
  int64_t &limit = apfrnn::plan::row_placement_min_bytes();
  const int64_t saved = limit;
  limit = int64_t{1} << 20;
  CHECK_FALSE(apfrnn::plan::place_rows_by_point(0));
  CHECK_FALSE(apfrnn::plan::place_rows_by_point(4096));
  CHECK(apfrnn::plan::place_rows_by_point(int64_t{1} << 30));
  limit = saved;
}

namespace {
// Most tests use sets far smaller than a cache. Placing per-point rows for
// every table size makes the whole test binary run that layout on reused
// buffers, next to the sorted layout every newly allocated buffer still gets.
const struct PlaceRowsInTests {
  PlaceRowsInTests() { apfrnn::plan::row_placement_min_bytes() = 0; }
} place_rows_in_tests;
} // namespace

TEST_CASE("fitted capacities keep a margin for degree changes") {
  CHECK(apfrnn::plan::same_fitted_capacity(10, 128) == 32);
  CHECK(apfrnn::plan::same_fitted_capacity(70, 128) == 80);
  CHECK(apfrnn::plan::same_fitted_capacity(8000, 16384) ==
        apfrnn::round_up_to_row_align(8500));
  CHECK(apfrnn::plan::cross_fitted_capacity(70, 128) ==
        line_capacity(apfrnn::round_up_to_row_align(78)));
  CHECK(apfrnn::plan::cross_fitted_capacity(70, 64) == line_capacity(64));
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
  // Each row's padding is one gang of slack, so it follows the gang width.
  CHECK(apfrnn::plan::block_output_budget(1000, 10) ==
        1000 * 4 + 10 * apfrnn::row_slack() * 4 + kSlack);
  // The slack must not multiply the exact-neighbor term.
  CHECK(apfrnn::plan::block_output_budget(1, 1) < kSlack + 1024);
}

TEST_CASE("exact capacities fit the maximum without a degree margin") {
  using apfrnn::plan::same_exact_capacity;
  CHECK(same_exact_capacity(5, 1000, kBudget) == 32);
  CHECK(same_exact_capacity(70, 1000, kBudget) ==
        apfrnn::round_up_to_row_align(70));
  CHECK(same_exact_capacity(40000, 1000, kBudget) == 0);   // over the limit
  CHECK(same_exact_capacity(1000, 1000000, 1 << 20) == 0); // over budget
}

TEST_CASE("block rows are preferred only for large padding savings") {
  using apfrnn::plan::prefer_block_rows;
  using apfrnn::plan::same_block_slots;
  using apfrnn::plan::same_fixed_slots;
  // One long row pads every row of the fixed layout.
  apfrnn::IntBuffer outlier(80000, 10);
  outlier[123] = 600;
  CHECK(same_fixed_slots(outlier) ==
        80000LL * (apfrnn::round_up_to_row_align(600) + apfrnn::row_slack()));
  CHECK(prefer_block_rows(same_fixed_slots(outlier),
                          same_block_slots(outlier, 8)));
  // Equal rows: block rows only add headroom and per-row offsets.
  const apfrnn::IntBuffer flat(80000, 30);
  CHECK_FALSE(
      prefer_block_rows(same_fixed_slots(flat), same_block_slots(flat, 8)));
  // A large relative saving that is too small in bytes to matter.
  apfrnn::IntBuffer tiny(50, 10);
  tiny[3] = 600;
  CHECK_FALSE(
      prefer_block_rows(same_fixed_slots(tiny), same_block_slots(tiny, 8)));
}

TEST_CASE("high-degree sets size every block row on its own") {
  CHECK(apfrnn::plan::same_min_block(apfrnn::IntBuffer(1000, 100)) == 1);
  CHECK(apfrnn::plan::same_min_block(apfrnn::IntBuffer(1000, 10)) == 8);
  CHECK(apfrnn::plan::same_min_block(apfrnn::IntBuffer()) == 8);
}

TEST_CASE("moving rows keep a margin above the current degree") {
  using apfrnn::plan::moving_row_capacity;
  CHECK(moving_row_capacity(0) == apfrnn::round_up_to_row_align(8));
  CHECK(moving_row_capacity(10) >= 18);
  CHECK(moving_row_capacity(640) >= 720);
  CHECK(moving_row_capacity(640) % apfrnn::row_align() == 0);
  // Per-point rows of very uneven degrees beat rows padded to the maximum.
  apfrnn::IntBuffer uneven(80000, 20);
  for (std::size_t i = 0; i < uneven.size(); i += 4) {
    uneven[i] = 600;
  }
  CHECK(
      apfrnn::plan::prefer_block_rows(apfrnn::plan::same_fixed_slots(uneven),
                                      apfrnn::plan::same_moving_slots(uneven)));
  // Near-uniform degrees: the margin costs more than the padding saves.
  const apfrnn::IntBuffer even(80000, 30);
  CHECK_FALSE(
      apfrnn::plan::prefer_block_rows(apfrnn::plan::same_fixed_slots(even),
                                      apfrnn::plan::same_moving_slots(even)));
}
