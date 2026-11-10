#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build-coverage}"
line_floor="${APFRNN_COVERAGE_LINE_FLOOR:-90}"
branch_floor="${APFRNN_COVERAGE_BRANCH_FLOOR:-70}"

cmake -S . -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS='-fprofile-instr-generate -fcoverage-mapping' \
  -DCMAKE_EXE_LINKER_FLAGS='-fprofile-instr-generate' \
  -DAPFRNN_BUILD_BASELINES=OFF

cmake --build "${build_dir}" -j 8

rm -f "${build_dir}"/apfrnn-*.profraw
LLVM_PROFILE_FILE="${build_dir}/apfrnn-%p.profraw" \
  ./scripts/run_tests.sh ./"${build_dir}"/tests/ispc_apfrnn_tests >/dev/null

llvm-profdata merge -sparse "${build_dir}"/apfrnn-*.profraw \
  -o "${build_dir}/apfrnn.profdata"
llvm-cov export ./"${build_dir}"/tests/ispc_apfrnn_tests \
  -instr-profile="${build_dir}/apfrnn.profdata" -summary-only \
  -ignore-filename-regex='/(support|tests|bench)/' \
  > "${build_dir}/coverage.json"

line=$(jq -r '.data[0].totals.lines.percent' "${build_dir}/coverage.json")
branch=$(jq -r '.data[0].totals.branches.percent' "${build_dir}/coverage.json")

printf 'C++ line coverage:   %.2f%% (floor %s%%)\n' "${line}" "${line_floor}"
printf 'C++ branch coverage: %.2f%% (floor %s%%)\n' "${branch}" "${branch_floor}"

awk -v actual="${line}" -v floor="${line_floor}" \
  'BEGIN { exit !(actual >= floor) }' || {
  echo "Line coverage below floor" >&2
  exit 1
}
awk -v actual="${branch}" -v floor="${branch_floor}" \
  'BEGIN { exit !(actual >= floor) }' || {
  echo "Branch coverage below floor" >&2
  exit 1
}
