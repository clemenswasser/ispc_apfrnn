#!/usr/bin/env python3
"""Generate the README benchmark tables from ispc_apfrnn_bench JSON.

The tables live between <!-- bench:NAME:start --> and <!-- bench:NAME:end -->
markers.  Normal use:

    scripts/bench_tables.py --write

This runs the benchmark matrix into build/bench-tables, regenerates every
table and writes README.md.  Review the resulting git diff: a changed row is
the benchmark delta for the commit.

To verify that README is already current without running the benchmarks:

    scripts/bench_tables.py --json-dir build/bench-tables --check

--from-json expects headline.json, stress.json, learnsph.json, motion.json,
droplets_motion.json, k50.json, memory.json and learnsph_memory.json in that
directory.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

# (README label, benchmark scene)
HEADLINE = [
    ("uniform", "uniform_200k"),
    ("uniform, 1M", "uniform_1m"),
    ("uniform, moving", "uniform_200k_moving"),
    ("jittered lattice", "jittered_200k"),
    ("SPH-like slab", "slab_200k"),
    ("slab, moving", "slab_200k_moving"),
    ("huge sparse domain", "huge_domain_200k"),
    ("thin slab", "thin_slab_200k"),
    ("surface", "surface_200k"),
    ("two-scale clusters", "two_scale_200k"),
    ("clustered", "clustered_200k"),
    ("thin pipe", "thin_pipe_200k"),
    ("pathological (10k, one cell)", "pathological_10k"),
]

STRESS = [
    ("clustered 500k", "clustered_500k"),
    ("clustered 1M", "clustered_1m"),
    ("two-scale 1M", "two_scale_1m"),
    ("uniform 2M", "uniform_2m"),
    ("8 sets x 2.5k", "many_sets_20k"),
    ("8 sets x 12.5k", "many_sets_100k"),
    ("16 sets x 1.25k", "many_sets_16x1250"),
    ("droplets 200k", "droplets_200k"),
    ("droplets 200k, jitter", "droplets_200k_jitter"),
]

LEARNSPH = [
    ("ramp_split_lv", "learnsph_ramp_split_lv", "20k / 78k"),
    ("ramp_split_hv", "learnsph_ramp_split_hv", "29k / 78k"),
    ("wall_01", "learnsph_wall_01", "74k / 208k"),
    ("funnel", "learnsph_funnel", "21k / 535k"),
    ("sink", "learnsph_sink", "57k / 13k"),
    ("bowl", "learnsph_bowl", "111k / 82k"),
    ("wall", "learnsph_wall", "107k / 208k"),
]

MOTION = [
    ("uniform 200k", "Frame", "uniform_200k", "uniform_200k_jitter"),
    ("slab 200k", "Frame", "slab_200k", "slab_200k_jitter"),
    ("clustered 200k", "Frame", "clustered_200k", "clustered_200k_jitter"),
    ("thin pipe 50k", "Frame", "thin_pipe_50k", "thin_pipe_50k_jitter"),
    ("droplets 200k", "Stress", "droplets_200k", "droplets_200k_jitter"),
]

BACKENDS = ["apfrnn_cpu", "tree", "compact"]


def scene_names(rows):
    return [scene for _, scene in rows]


def run_bench(build: Path, cpus: str, family: str, scenes, backends,
              out: Path, extra, min_time: str, repetitions: int):
    pattern = (rf"^{family}/({'|'.join(backends)})/"
               rf"({'|'.join(re.escape(s) for s in scenes)})/real_time$")
    cmd = [
        "taskset", "-c", cpus,
        str(build / "bench" / "ispc_apfrnn_bench"),
        f"--benchmark_filter={pattern}",
        f"--benchmark_min_time={min_time}",
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_report_aggregates_only=true",
        "--benchmark_format=json",
        f"--benchmark_out={out}",
        *extra,
    ]
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def load_medians(path: Path):
    data = json.loads(path.read_text())
    result = {}
    for bench in data["benchmarks"]:
        if bench["name"].endswith("_median"):
            result[bench["name"]] = bench
    return result


def expected_name(family: str, backend: str, scene: str):
    return f"{family}/{backend}/{scene}/real_time_median"


def run_matrix(args, work: Path):
    work.mkdir(parents=True, exist_ok=True)

    run_bench(args.build, args.cpus, "Frame", scene_names(HEADLINE), BACKENDS,
              work / "headline.json", [], args.min_time, args.repetitions)
    run_bench(args.build, args.cpus, "Stress", scene_names(STRESS), BACKENDS,
              work / "stress.json", [], args.min_time, args.repetitions)
    run_bench(args.build, args.cpus, "Frame", scene_names(LEARNSPH), BACKENDS,
              work / "learnsph.json",
              [f"--learnsph_res={args.learnsph_res}"], args.min_time,
              args.repetitions)
    run_bench(args.build, args.cpus, "Frame",
              [s for _, _, s, _ in MOTION], ["apfrnn_cpu"],
              work / "motion.json", [], args.min_time, args.repetitions)
    run_bench(args.build, args.cpus, "Stress",
              [s for _, family, _, s in MOTION if family == "Stress"],
              ["apfrnn_cpu"], work / "droplets_motion.json", [],
              args.min_time, args.repetitions)
    memory = {}
    if not args.no_memory:
        for family, rows in (("Frame", HEADLINE), ("Stress", STRESS)):
            for _, scene in rows:
                for backend in BACKENDS:
                    out = work / f"mem_{family}_{backend}_{scene}.json"
                    run_bench(args.build, args.cpus, family, [scene],
                              [backend], out, [], args.memory_time, 1)
                    bench = load_medians(out)
                    # A one-repetition run has no _median suffix.
                    if not bench:
                        data = json.loads(out.read_text())
                        bench = {data["benchmarks"][0]["name"]:
                                 data["benchmarks"][0]}
                    entry = next(iter(bench.values()))
                    memory[f"{family}/{scene}/{backend}"] = [
                        entry.get("NeighborMB", 0.0),
                        entry.get("PeakRSSMB", 0.0),
                    ]
        for _, scene, _ in LEARNSPH:
            for backend in BACKENDS:
                out = work / f"mem_Frame_{backend}_{scene}.json"
                run_bench(args.build, args.cpus, "Frame", [scene], [backend],
                          out, [f"--learnsph_res={args.learnsph_res}"],
                          args.memory_time, 1)
                data = json.loads(out.read_text())
                bench = data["benchmarks"][0]
                memory[f"Frame/{scene}/{backend}"] = [
                    bench.get("NeighborMB", 0.0), bench.get("PeakRSSMB", 0.0)
                ]
        (work / "memory.json").write_text(json.dumps(memory, indent=2,
                                                     sort_keys=True) + "\n")

    # Keep a single memory file for --json-dir reuse.
    if (work / "memory.json").exists():
        (work / "learnsph_memory.json").write_text(
            (work / "memory.json").read_text())


def load_inputs(json_dir: Path):
    names = ["headline", "stress", "learnsph", "motion",
             "droplets_motion"]
    data = {name: load_medians(json_dir / f"{name}.json") for name in names}
    memory_path = json_dir / "memory.json"
    if not memory_path.exists():
        memory_path = json_dir / "learnsph_memory.json"
    memory = json.loads(memory_path.read_text()) if memory_path.exists() else {}
    return data, memory


def get(data, family, backend, scene):
    return data[expected_name(family, backend, scene)]


def fmt_ms(value):
    return f"{value:.2f} ms"


def fmt_ratio(value):
    if value >= 10.0:
        return f"{value:.1f}x"
    return f"{value:.1f}x"


def fmt_mb(value):
    return f"{value:.0f}"


def table_timing(data, rows, family):
    lines = [
        "| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Compact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, scene in rows:
        a = get(data, family, "apfrnn_cpu", scene)
        t = get(data, family, "tree", scene)
        c = get(data, family, "compact", scene)
        ta, tt, tc = a["real_time"], t["real_time"], c["real_time"]
        lines.append(
            f"| {label} | {a['AvgNeighbors']:.0f} | {fmt_ms(ta)} | "
            f"{fmt_ms(tt)} | {fmt_ms(tc)} | {fmt_ratio(tt/ta)} | "
            f"{fmt_ratio(tc/ta)} |")
    return "\n".join(lines)


def table_memory(data, memory, rows, family):
    lines = [
        "| Scene | Rows MB (ours) | Peak RSS ours | Tree | Compact |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, scene in rows:
        a = get(data, family, "apfrnn_cpu", scene)
        rows_a = a.get("NeighborMB", 0.0)
        peak_a = memory.get(f"{family}/{scene}/apfrnn_cpu", [0, 0])[1]
        peak_t = memory.get(f"{family}/{scene}/tree", [0, 0])[1]
        peak_c = memory.get(f"{family}/{scene}/compact", [0, 0])[1]
        lines.append(
            f"| {label} | {fmt_mb(rows_a)} | {fmt_mb(peak_a)} | "
            f"{fmt_mb(peak_t)} | {fmt_mb(peak_c)} |")
    return "\n".join(lines)


def table_learnsph(data, memory):
    lines = [
        "| Frame | Fluid / boundary | ispc_apfrnn | TreeNSearch | CompactNSearch | Peak RSS (ours / Tree / Compact) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    learn_data = data["learnsph"]
    for label, scene, size in LEARNSPH:
        a = get(learn_data, "Frame", "apfrnn_cpu", scene)
        t = get(learn_data, "Frame", "tree", scene)
        c = get(learn_data, "Frame", "compact", scene)
        peak = []
        for backend in BACKENDS:
            peak.append(memory.get(f"Frame/{scene}/{backend}", [0, 0])[1])
        lines.append(
            f"| {label} | {size} | {fmt_ms(a['real_time'])} | "
            f"{fmt_ms(t['real_time'])} | {fmt_ms(c['real_time'])} | "
            f"{fmt_mb(peak[0])} / {fmt_mb(peak[1])} / {fmt_mb(peak[2])} MB |")
    return "\n".join(lines)


def table_motion(data):
    lines = [
        "| Scene | static | jitter | jitter vs static | maximum row, static -> jitter |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, family, static, jitter in MOTION:
        source = data["motion"] if family == "Frame" else data["droplets_motion"]
        a = source[expected_name(family, "apfrnn_cpu", static)]
        b = source[expected_name(family, "apfrnn_cpu", jitter)]
        pct = (b["real_time"] / a["real_time"] - 1.0) * 100.0
        lines.append(
            f"| {label} | {fmt_ms(a['real_time'])} | "
            f"{fmt_ms(b['real_time'])} | {pct:+.0f}% | "
            f"{a['MaxNeighbors']:.0f} -> {b['MaxNeighbors']:.0f} |")
    return "\n".join(lines)


def replace_block(text: str, name: str, content: str):
    start = f"<!-- bench:{name}:start -->"
    end = f"<!-- bench:{name}:end -->"
    left = text.find(start)
    right = text.find(end, left + 1)
    if left < 0 or right < 0:
        raise SystemExit(f"README is missing marker block {name!r}")
    return text[:left] + start + "\n" + content + "\n" + text[right:]


def generate(data, memory):
    text = Path("README.md").read_text()
    replacements = {
        "headline-timing": table_timing(data["headline"], HEADLINE, "Frame"),
        "headline-memory": table_memory(data["headline"], memory, HEADLINE,
                                        "Frame"),
        "stress-timing": table_timing(data["stress"], STRESS, "Stress"),
        "stress-memory": table_memory(data["stress"], memory, STRESS,
                                      "Stress"),
        "motion": table_motion(data),
        "learnsph": table_learnsph(data, memory),
    }
    for name, content in replacements.items():
        text = replace_block(text, name, content)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--cpus", default="0-15")
    parser.add_argument("--json-dir", type=Path,
                        help="reuse JSON from this directory instead of running")
    parser.add_argument("--work-dir", type=Path,
                        default=Path("build/bench-tables"))
    parser.add_argument("--min-time", default="2s")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--memory-time", default="0.5s")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--learnsph-res", default="../learnsph/res")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true",
                       help="write README.md")
    group.add_argument("--check", action="store_true",
                       help="fail if README.md differs (default)")
    args = parser.parse_args()

    if args.json_dir is not None:
        data, memory = load_inputs(args.json_dir)
    else:
        run_matrix(args, args.work_dir)
        data, memory = load_inputs(args.work_dir)

    expected = generate(data, memory)
    current = Path("README.md").read_text()
    if expected == current:
        print("README benchmark tables are up to date.", file=sys.stderr)
        return 0
    diff = "".join(difflib.unified_diff(
        current.splitlines(True), expected.splitlines(True),
        fromfile="README.md (current)", tofile="README.md (generated)"))
    if args.write:
        Path("README.md").write_text(expected)
        print(diff, file=sys.stderr)
        return 0
    print(diff, file=sys.stderr)
    print("README benchmark tables are stale; run with --write.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
