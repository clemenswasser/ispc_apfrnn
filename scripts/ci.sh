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

# Gang-width matrix: no kernel or host code may assume the 8 lanes of the
# AVX2 default. Both targets run on any AVX2 host (avx2-i32x16 double-pumps
# 16 lanes, the AVX-512 width), so this needs no AVX-512 machine.
for target in sse4-i32x4:4 avx2-i32x16:16; do
  isa="${target%%:*}"
  width="${target##*:}"
  cmake -S . -B "${release_build}-${isa}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_ISPC_INSTRUCTION_SETS="${isa}" \
    -DAPFRNN_BUILD_BENCHMARKS=OFF \
    -DAPFRNN_BUILD_BASELINES=OFF
  cmake --build "${release_build}-${isa}" -j 8
  APFRNN_EXPECT_GANG_WIDTH="${width}" \
    ctest --test-dir "${release_build}-${isa}" --output-on-failure
done

cmake -S . -B "${asan_build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS='-D_GLIBCXX_ASSERTIONS -fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' \
  -DAPFRNN_BUILD_BASELINES=OFF
cmake --build "${asan_build}" -j 8
# _GLIBCXX_ASSERTIONS bounds-checks every std::vector access, and the Debug
# build checks the cell range tables the ISPC kernels read through raw
# pointers. The fuzz test runs fewer seeds here (correctness only; the
# Release run above checks that its seeds reach every code path).
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ASAN_OPTIONS=detect_leaks=1 \
  APFRNN_FUZZ_SEEDS=24 "./${asan_build}/tests/ispc_apfrnn_tests"

# TSan is intentionally not part of the gate: the current oneTBB build reports
# races inside its work-stealing partitioner. Re-enable after a TSan-compatible
# TBB build/suppression set is available.
./scripts/coverage.sh "${coverage_build}"
