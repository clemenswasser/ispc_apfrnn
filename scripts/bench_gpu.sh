#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build-sycl}"
cpus="${APFRNN_CPUS:-0-15}"
filter="${APFRNN_FILTER:-^Frame/apfrnn_(cpu|gpu|auto)/}"

if [[ ! -x "${build_dir}/bench/ispc_apfrnn_bench" ]]; then
  echo "Missing ${build_dir}/bench/ispc_apfrnn_bench" >&2
  echo "Configure a SYCL build with -DAPFRNN_ENABLE_SYCL=ON" >&2
  exit 1
fi

exec taskset -c "${cpus}" "${build_dir}/bench/ispc_apfrnn_bench" \
  --benchmark_filter="${filter}" \
  --benchmark_min_time=1s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
