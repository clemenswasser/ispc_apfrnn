#!/usr/bin/env bash
# Captures neighbor-search traces from learnsph into traces/. Works on a copy of
# the learnsph checkout, so that checkout stays untouched.
#
# Usage:
#   scripts/capture_traces.sh [small|medium|large|all|scene-name-filter]
#
# Environment:
#   LEARNSPH_DIR   learnsph checkout (default ../learnsph)
#   APFRNN_TRACES  output directory (default traces)
#   CAPTURE_CPUS   CPU list for the simulation (default 0-19)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
learnsph="${LEARNSPH_DIR:-${root}/../learnsph}"
traces="${APFRNN_TRACES:-${root}/traces}"
cpus="${CAPTURE_CPUS:-0-19}"
selection="${1:-small}"
src="${root}/build-capture/learnsph"
build="${root}/build-capture/build"
searches=48

# tier scene radius pbf times...
scenes='
small  dam_break  0.0025 4 0.10 0.40
small  wall       0.01   4 0.10 0.50
small  bowl       0.01   4 0.10 0.50
small  sink       0.01   4 0.10 0.50
small  droplet    0.01   4 0.05 0.40
small  funnel     0.01   4 0.20 0.60
small  ramp_lv    0.01   0 0.10 0.30
small  ramp_hv    0.01   0 0.10 0.30
medium dam_break  0.0015 4 0.10 0.40
medium wall       0.005  4 0.10 0.40
medium bowl       0.005  4 0.10 0.40
medium sink       0.005  4 0.10 0.40
medium droplet    0.005  4 0.05 0.25
medium funnel     0.006  2 0.20 0.40
large  dam_break  0.001  2 0.10 0.30
large  wall       0.0035 2 0.10 0.30
large  bowl       0.0035 2 0.10 0.30
large  wall       0.0025 1 0.10
'

mkdir -p "${root}/build-capture" "${traces}"
rsync -a --delete --exclude '/res/' --exclude '/build*/' --exclude '/.git/' \
  "${learnsph}/" "${src}/"
mkdir -p "${src}/res"
cp "${root}/bench/capture/trace_recorder.h" "${src}/learnSPH/TraceRecorder.h"
cp "${root}/bench/capture/nsearch_trace_capture.cpp" "${src}/tests/"

python3 - "${src}" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
path = src / "learnSPH/GenericNSearch.h"
text = path.read_text()
edits = [
    ("#include <apfrnn/neighbor_search.hpp>\n",
     '#include <apfrnn/neighbor_search.hpp>\n#include "TraceRecorder.h"\n'),
    ("    return set < 0 ? static_cast<uint32_t>(-1)",
     "    trace_recorder().set_points(set, data, size);\n"
     "    return set < 0 ? static_cast<uint32_t>(-1)"),
    ("                               std::size_t new_size) {\n",
     "                               std::size_t new_size) {\n"
     "    trace_recorder().set_points(static_cast<int>(point_set_index), new_data, new_size);\n"),
    ("                                   bool is_static = true) {\n",
     "                                   bool is_static = true) {\n"
     "    trace_recorder().set_static(static_cast<int>(point_set_index), is_static);\n"),
    ("                                bool active = true) {\n",
     "                                bool active = true) {\n"
     "    trace_recorder().set_active(static_cast<int>(set_i), static_cast<int>(set_j), active);\n"),
    ("  inline void set_all_searches(bool active) {\n",
     "  inline void set_all_searches(bool active) {\n"
     "    trace_recorder().clear_active();\n"),
    ("  inline void run() {\n",
     "  inline void run() {\n    trace_recorder().frame(search_radius);\n"),
]
for old, new in edits:
    if text.count(old) != 1:
        sys.exit(f"cannot patch GenericNSearch.h at: {old!r}")
    text = text.replace(old, new)
path.write_text(text)

path = src / "tests/CMakeLists.txt"
path.write_text(path.read_text() + """
add_executable(nsearch_trace_capture nsearch_trace_capture.cpp)
target_link_libraries(nsearch_trace_capture PRIVATE learnSPH)
target_compile_definitions(nsearch_trace_capture PRIVATE
    LEARNSPH_MESH_DIR="${CMAKE_SOURCE_DIR}/meshes")
""")
PY

if [[ ! -f "${build}/build.ninja" ]]; then
  cmake -S "${src}" -B "${build}" -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
fi
cmake --build "${build}" --target nsearch_trace_capture >/dev/null
capture="$(find "${build}" -type f -name nsearch_trace_capture -executable | head -1)"

while read -r tier scene radius pbf times; do
  [[ -z "${tier}" ]] && continue
  case "${selection}" in
    all | "${tier}" | "${scene}") ;;
    *) continue ;;
  esac
  name="${scene}_r${radius}"
  echo "== ${name}: pbf ${pbf}, windows ${times}"
  # shellcheck disable=SC2086
  (cd "${build}" && taskset -c "${cpus}" "${capture}" "${scene}" "${radius}" "${pbf}" \
    "${traces}/${name}" "${searches}" ${times}) | rg '\.trace:' || echo "FAILED ${name}"
done <<<"${scenes}"
