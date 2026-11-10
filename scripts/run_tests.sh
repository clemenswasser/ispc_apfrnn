#!/usr/bin/env bash
# Runs every doctest case of a test binary in its own process, several at a
# time. A single process runs the cases one after another, so the few long
# multi-frame cases set the wall time; separate processes overlap them.
#
#   scripts/run_tests.sh <test binary> [jobs]   (jobs default: APFRNN_TEST_JOBS or 4)
#
# Environment variables (sanitizer options, APFRNN_* knobs,
# LLVM_PROFILE_FILE with %p) pass through to every process. Prints the
# output of failing cases and exits non-zero if any case fails.
set -euo pipefail

binary="$1"
jobs="${2:-${APFRNN_TEST_JOBS:-4}}"
count=$("${binary}" --count --no-intro --no-version |
  sed -n 's/.*passing the current filters: \([0-9]*\).*/\1/p')
[[ "${count}" =~ ^[0-9]+$ && "${count}" -gt 0 ]] || {
  echo "run_tests.sh: no test cases found in ${binary}" >&2
  exit 1
}

logs=$(mktemp -d)
trap 'rm -rf "${logs}"' EXIT

# Case k of the name-ordered list runs alone with --first=k --last=k.
run_case() {
  local k="$1"
  if ! "${binary}" --order-by=name --first="${k}" --last="${k}" \
      --no-intro --no-version >"${logs}/${k}.log" 2>&1; then
    echo "${k}" >"${logs}/${k}.failed"
  fi
}
export -f run_case
export binary logs

seq 1 "${count}" | xargs -P "${jobs}" -I{} bash -c 'run_case {}'

failed=0
for f in "${logs}"/*.failed; do
  [[ -e "${f}" ]] || continue
  k=$(cat "${f}")
  cat "${logs}/${k}.log" >&2
  failed=$((failed + 1))
done
if (( failed > 0 )); then
  echo "run_tests.sh: ${failed} of ${count} test cases failed in ${binary}" >&2
  exit 1
fi
echo "run_tests.sh: ${count} test cases passed in ${binary} (${jobs} jobs)"
