#!/usr/bin/env bash
set -euo pipefail

release_build="${1:-build-ci-release}"
asan_build="${2:-build-ci-asan}"
coverage_build="${3:-build-ci-coverage}"

cmake -S . -B "${release_build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DAPFRNN_BUILD_BASELINES=OFF
cmake --build "${release_build}" -j 8
ctest --test-dir "${release_build}" --output-on-failure

cmake -S . -B "${asan_build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' \
  -DAPFRNN_BUILD_BASELINES=OFF
cmake --build "${asan_build}" -j 8
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ASAN_OPTIONS=detect_leaks=1 \
  "./${asan_build}/tests/ispc_apfrnn_tests"

# TSan is intentionally not part of the gate: the current oneTBB build reports
# races inside its work-stealing partitioner. Re-enable after a TSan-compatible
# TBB build/suppression set is available.
./scripts/coverage.sh "${coverage_build}"
