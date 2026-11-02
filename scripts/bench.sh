#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build}"
cpus="${APFRNN_CPUS:-0-15}"
filter="${APFRNN_FILTER:-^Frame/(apfrnn_cpu|tree|compact)/uniform_(50k|200k|1m)/}"

if [[ ! -x "${build_dir}/bench/ispc_apfrnn_bench" ]]; then
  echo "Missing ${build_dir}/bench/ispc_apfrnn_bench" >&2
  echo "Configure with -DAPFRNN_BUILD_BASELINES=ON" >&2
  exit 1
fi

exec taskset -c "${cpus}" "${build_dir}/bench/ispc_apfrnn_bench" \
  --benchmark_filter="${filter}" \
  --benchmark_min_time=0.5s \
  --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true
