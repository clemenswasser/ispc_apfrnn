#!/usr/bin/env bash
# Collects hardware evidence for one benchmark into
# .perf-log/evidence/<experiment-id>/<label>/. Counters and samples cover only
# the timed frames, not setup or validation. Start with summary.txt.
#
# Usage:
#   scripts/perf_evidence.sh <experiment-id> <label> <benchmark-regex> [kernel ...]
#
# The regex must select exactly one benchmark. Kernel names add an llvm-mca
# report through scripts/ispc_analyze.sh.
#
# Environment:
#   APFRNN_PROF_BUILD  profiling build directory (default build-prof)
#   APFRNN_CPUS        CPU list (default 0-15)
#   APFRNN_THREADS     worker threads (default 8)
#   APFRNN_MIN_TIME    seconds per collection (default 2)
#   APFRNN_BENCH_ARGS  extra benchmark arguments
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n '2,15p' "$0" >&2
  exit 2
fi
id="$1"
label="$2"
regex="$3"
shift 3

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
common="$(git -C "${root}" rev-parse --path-format=absolute --git-common-dir)"
out="$(dirname "${common}")/.perf-log/evidence/${id}/${label}"
build="${APFRNN_PROF_BUILD:-${root}/build-prof}"
cpus="${APFRNN_CPUS:-0-15}"
threads="${APFRNN_THREADS:-8}"
min_time="${APFRNN_MIN_TIME:-2}"
bench_args="${APFRNN_BENCH_ARGS:-}"

# The same optimized code as a Release build, plus line tables for annotation.
if [[ ! -f "${build}/build.ninja" ]]; then
  cmake -S "${root}" -B "${build}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_FLAGS=-g \
    -DCMAKE_ISPC_FLAGS=-g \
    -DAPFRNN_BUILD_BASELINES=ON >/dev/null
fi
cmake --build "${build}" --target ispc_apfrnn_bench >/dev/null
bin="${build}/bench/ispc_apfrnn_bench"

matches="$("${bin}" --benchmark_filter="${regex}" --benchmark_list_tests | rg -c '/' || true)"
if [[ "${matches}" != "1" ]]; then
  echo "The regex selects ${matches:-0} benchmarks, it must select exactly one:" >&2
  "${bin}" --benchmark_filter="${regex}" --benchmark_list_tests | head -20 >&2
  exit 2
fi

rm -rf "${out}"
mkdir -p "${out}"
work="$(mktemp -d)"
mkfifo "${work}/ctl" "${work}/ack"
export APFRNN_PERF_CTL="${work}/ctl" APFRNN_PERF_ACK="${work}/ack"
control=(--control="fifo:${work}/ctl,${work}/ack")

# shellcheck disable=SC2206
run=(taskset -c "${cpus}" "${bin}" --benchmark_filter="${regex}"
  --benchmark_min_time="${min_time}s" --threads="${threads}" ${bench_args})

previous="$(tuned-adm active 2>/dev/null | sed -n 's/^Current active profile: //p' || true)"
restore() {
  [[ -n "${previous}" ]] && sudo -n tuned-adm profile "${previous}" >/dev/null 2>&1
  rm -rf "${work}"
}
trap restore EXIT
tuned_profile="${previous:-none}"
if [[ -n "${previous}" ]] && sudo -n tuned-adm profile benchmark-pmu >/dev/null 2>&1; then
  tuned_profile="benchmark-pmu"
fi

{
  echo "date:        $(date -Is)"
  echo "commit:      $(git -C "${root}" rev-parse HEAD)"
  echo "dirty hash:  $(git -C "${root}" diff HEAD | sha256sum | cut -d' ' -f1)"
  echo "benchmark:   ${regex}"
  echo "cpus:        ${cpus}"
  echo "threads:     ${threads}"
  echo "min time:    ${min_time}s"
  echo "tuned:       ${tuned_profile}"
  echo "governor:    $(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "nmi wdog:    $(cat /proc/sys/kernel/nmi_watchdog)"
  echo "paranoid:    $(cat /proc/sys/kernel/perf_event_paranoid)"
  echo "ispc:        $(ispc --version 2>/dev/null | head -1)"
  git -C "${root}" status --short | sed 's/^/changed:     /'
} >"${out}/meta.txt"

stat_metric() {
  perf stat --no-big-num --cputype core -D -1 "${control[@]}" \
    -M "$1" -o "${out}/$2" -- "${run[@]}" >/dev/null 2>&1 ||
    echo "collection failed: $1" >"${out}/$2"
}

env -u APFRNN_PERF_CTL -u APFRNN_PERF_ACK "${run[@]}" --benchmark_repetitions=3 --benchmark_out="${out}/timing.json" \
  --benchmark_out_format=json >"${out}/timing.txt" 2>&1

stat_metric TopdownL1 topdown_l1.txt
stat_metric TopdownL2 topdown_l2.txt
stat_metric tma_memory_bound_group topdown_memory.txt
stat_metric tma_core_bound_group topdown_core.txt
stat_metric tma_frontend_bound_group topdown_frontend.txt
stat_metric tma_bad_speculation_group topdown_bad_speculation.txt

perf stat --no-big-num -D -1 "${control[@]}" -o "${out}/counters.txt" \
  -e cpu_core/cycles/,cpu_core/instructions/,cpu_core/branches/,cpu_core/branch-misses/ \
  -e cpu_core/L1-dcache-load-misses/,cpu_core/LLC-load-misses/,cpu_core/dTLB-load-misses/ \
  -e context-switches,cpu-migrations,page-faults \
  -- "${run[@]}" >/dev/null 2>&1 || echo "collection failed" >"${out}/counters.txt"

perf record -q -o "${work}/cycles.data" -D -1 "${control[@]}" \
  -e cpu_core/cycles/ --call-graph lbr -- "${run[@]}" >/dev/null 2>&1 || true
perf report -i "${work}/cycles.data" --no-children --stdio -g none \
  --sort symbol --percent-limit 0.5 >"${out}/hotspots.txt" 2>/dev/null ||
  echo "collection failed" >"${out}/hotspots.txt"
perf report -i "${work}/cycles.data" --no-children --stdio -g none \
  --sort pid >"${out}/threads.txt" 2>/dev/null || true
perf annotate -i "${work}/cycles.data" --stdio --percent-limit 3 -l 2>/dev/null |
  head -20000 >"${out}/annotate.txt" || true

perf mem record -q -o "${work}/mem.data" -D -1 "${control[@]}" -- "${run[@]}" \
  >/dev/null 2>&1 || true
perf mem report -i "${work}/mem.data" --stdio --sort mem,symbol --percent-limit 0.5 \
  >"${out}/memory_levels.txt" 2>/dev/null || true

# Root sees the memory controllers and cache-line contention. The benchmark
# still runs as this user.
user_run=(sudo -n -u "${USER}" env "APFRNN_PERF_CTL=${APFRNN_PERF_CTL}"
  "APFRNN_PERF_ACK=${APFRNN_PERF_ACK}" "${run[@]}")
sudo -n perf stat -a --no-big-num -D -1 "${control[@]}" -o "${out}/bandwidth.txt" \
  -e uncore_imc_free_running_0/data_read/,uncore_imc_free_running_0/data_write/ \
  -e uncore_imc_free_running_1/data_read/,uncore_imc_free_running_1/data_write/ \
  -- "${user_run[@]}" >"${out}/bandwidth_run.txt" 2>&1 || true
sudo -n chown "${USER}" "${out}/bandwidth.txt" 2>/dev/null || true

sudo -n perf c2c record -o "${work}/c2c.data" -- -D -1 "${control[@]}" -- "${user_run[@]}" \
  >/dev/null 2>&1 || true
if [[ -f "${work}/c2c.data" ]]; then
  sudo -n perf c2c report -i "${work}/c2c.data" --stdio 2>/dev/null |
    head -200 >"${out}/false_sharing.txt" || true
  sudo -n rm -f "${work}/c2c.data"
fi
sudo -n chown -R "${USER}" "${work}" 2>/dev/null || true

if [[ $# -gt 0 ]]; then
  "${root}/scripts/ispc_analyze.sh" "${out}/mca" "$@" >"${out}/mca.log" 2>&1 ||
    echo "llvm-mca failed, see mca.log" >&2
fi

{
  cat "${out}/meta.txt"
  echo
  echo "== timing"
  rg "manual_time" "${out}/timing.txt" | cut -c1-200 || true
  for file in topdown_l1 topdown_l2 topdown_memory topdown_core topdown_frontend topdown_bad_speculation; do
    echo
    echo "== ${file}"
    rg 'tma_|failed' "${out}/${file}.txt" | sed 's/^ *[0-9]* *[^ ]* *//' | cut -c1-100 || true
  done
  echo
  echo "== counters"
  rg -v '^\s*$|^#' "${out}/counters.txt" | cut -c1-90 || true
  echo
  echo "== hotspots"
  rg -v '^#|^$' "${out}/hotspots.txt" | head -20 | cut -c1-110 || true
  echo
  echo "== threads"
  rg -v '^#|^$' "${out}/threads.txt" | head -24 | cut -c1-70 || true
  echo
  echo "== memory controller traffic"
  rg 'data_' "${out}/bandwidth.txt" | cut -c1-80 || true
} >"${out}/summary.txt"

echo "Evidence: ${out}/summary.txt"
