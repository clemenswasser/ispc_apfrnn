#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build}"
points="${2:-50000}"
cpus="${APFRNN_CPUS:-0-15}"
session="${APFRNN_PROFILE_SESSION:-apfrnn}"

mkdir -p "${build_dir}/profiles"
profile="${build_dir}/profiles/nsearch-${points}.json"
if (( points % 1000000 == 0 )); then
  suffix="$((points / 1000000))m"
else
  suffix="$((points / 1000))k"
fi

taskset -c "${cpus}" samply record --save-only --presymbolicate \
  -o "${profile}" -- \
  "${build_dir}/bench/ispc_apfrnn_bench" \
  --benchmark_filter="^Api/Frame/static/uniform_${suffix}/" \
  --benchmark_min_time=0.5s

profiler-cli load --with-samply "${profile}" --session "${session}"
profiler-cli profile info --session "${session}"
profiler-cli thread functions --session "${session}" --min-self 1 --limit 30
