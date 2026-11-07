#!/usr/bin/env bash
# Quick benchmark-verification gate: every selected backend must agree with
# the independent reference checksum (or, for large scenes, with each other).
# Requires a build configured with -DAPFRNN_BUILD_BASELINES=ON.
set -euo pipefail

build_dir="${1:-build}"
cpus="${APFRNN_CPUS:-0-15}"
threads="${APFRNN_THREADS:-8}"

if [[ ! -x "${build_dir}/bench/ispc_apfrnn_bench" ]]; then
  echo "Missing ${build_dir}/bench/ispc_apfrnn_bench" >&2
  echo "Configure with -DAPFRNN_BUILD_BASELINES=ON" >&2
  exit 1
fi

filter='^(Order|Dynamic|Growing|Threads)/(apfrnn_cpu|tree|compact)/'
filter+='(uniform_200k|uniform_200k_unsorted|slab_200k|slab_200k_unsorted|'
filter+='clustered_200k|clustered_200k_unsorted|'
filter+='uniform_200k_dynamic_boundary|slab_200k_dynamic_boundary|'
filter+='clustered_200k_dynamic_boundary|uniform_200k_growing|'
filter+='slab_200k_growing|clustered_200k_growing|'
filter+='uniform_200k|uniform_1m|clustered_200k)(/[0-9]+)?/real_time$'

exec taskset -c "${cpus}" "${build_dir}/bench/ispc_apfrnn_bench" \
  --benchmark_filter="${filter}" \
  --benchmark_min_time=0.2s \
  --benchmark_repetitions=1 \
  --benchmark_report_aggregates_only=true \
  --threads="${threads}"
