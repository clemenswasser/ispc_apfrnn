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

--json-dir expects headline.json, stress.json, learnsph.json, motion.json,
droplets_motion.json, order.json, dynamic.json, scale.json, growing.json,
threads.json and memory.json in that directory. Pass a SYCL benchmark JSON
with --gpu-json to regenerate the separate GPU table.

Runs use a Morton-sorted scene order by default (TreeNSearch's documented
fast path), 8 worker threads for both TBB and OpenMP, and verify every backend
against an independent reference checksum. Every scene and backend runs in its
own process, so no run inherits another scene's memory state. The benchmark
times a fixed window of frames per scene (its Frames counter), so
--min-time does not change what is measured. --threads changes the shared
worker count; --no-scale skips the 175k..8M dense/sparse run.
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

ORDER = [
    ("uniform 200k", "uniform_200k", "uniform_200k_unsorted"),
    ("slab 200k", "slab_200k", "slab_200k_unsorted"),
    ("clustered 200k", "clustered_200k", "clustered_200k_unsorted"),
]

DYNAMIC = [
    ("uniform 200k", "uniform_200k", "uniform_200k_dynamic_boundary"),
    ("slab 200k", "slab_200k", "slab_200k_dynamic_boundary"),
    ("clustered 200k", "clustered_200k",
     "clustered_200k_dynamic_boundary"),
    ("learnsph funnel", "learnsph_funnel",
     "learnsph_funnel_dynamic_boundary"),
    ("learnsph sink", "learnsph_sink", "learnsph_sink_dynamic_boundary"),
    ("learnsph bowl", "learnsph_bowl", "learnsph_bowl_dynamic_boundary"),
    ("learnsph wall", "learnsph_wall", "learnsph_wall_dynamic_boundary"),
]

SCALE = [
    ("uniform", "uniform_175k"),
    ("uniform", "uniform_500k"),
    ("uniform", "uniform_1m"),
    ("uniform", "uniform_2m"),
    ("uniform", "uniform_4m"),
    ("uniform", "uniform_8m"),
    ("sparse", "huge_domain_175k"),
    ("sparse", "huge_domain_500k"),
    ("sparse", "huge_domain_1m"),
    ("sparse", "huge_domain_2m"),
    ("sparse", "huge_domain_4m"),
    ("sparse", "huge_domain_8m"),
]

GROWING = [
    ("uniform 200k -> 400k", "uniform_200k_growing"),
    ("slab 200k -> 400k", "slab_200k_growing"),
    ("clustered 200k -> 400k", "clustered_200k_growing"),
]

THREADS = [
    ("uniform 200k", "uniform_200k", [1, 2, 4, 8, 16]),
    ("uniform 1M", "uniform_1m", [8, 16]),
    ("clustered 200k", "clustered_200k", [1, 4, 16]),
    ("learnsph ramp_split_lv", "learnsph_ramp_split_lv", [1, 4, 16]),
]

BACKENDS = ["apfrnn_cpu", "tree", "compact"]


def scene_names(rows):
    # Headline/stress rows are (label, scene); learnsph rows also carry the
    # point-count label in the third column.
    return [row[1] for row in rows]


def run_bench(build: Path, cpus: str, family: str, scenes, backends,
              out: Path, extra, min_time: str, repetitions: int,
              threads: int = 8):
    """Runs every scene and backend in its own process and merges the JSON
    into out."""
    merged = None
    for scene in scenes:
        for backend in backends:
            part = out.with_name(f"{out.stem}.{backend}.{scene}.json")
            part.unlink(missing_ok=True)
            pattern = (rf"^{family}/{backend}/{re.escape(scene)}/"
                       rf"(iterations:[0-9]+/)?manual_time$")
            cmd = [
                "taskset", "-c", cpus,
                str(build / "bench" / "ispc_apfrnn_bench"),
                f"--benchmark_filter={pattern}",
                f"--benchmark_min_time={min_time}",
                f"--benchmark_repetitions={repetitions}",
                "--benchmark_report_aggregates_only=true",
                "--benchmark_format=json",
                f"--benchmark_out={part}",
                f"--threads={threads}",
                *extra,
            ]
            print("+", " ".join(cmd), file=sys.stderr)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            if not part.exists() or not part.read_text().strip():
                continue
            data = json.loads(part.read_text())
            if merged is None:
                merged = data
            else:
                merged["benchmarks"].extend(data["benchmarks"])
    out.write_text(json.dumps(merged or {"benchmarks": []}, indent=1) + "\n")


def run_threads(build: Path, cpus: str, scenes, backends, out: Path,
                min_time: str, repetitions: int):
    pattern = (rf"^Threads/({'|'.join(backends)})/"
               rf"({'|'.join(re.escape(s) for s in scenes)})/"
               rf"[0-9]+/(iterations:[0-9]+/)?manual_time$")
    cmd = [
        "taskset", "-c", cpus,
        str(build / "bench" / "ispc_apfrnn_bench"),
        f"--benchmark_filter={pattern}",
        f"--benchmark_min_time={min_time}",
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_report_aggregates_only=true",
        "--benchmark_format=json",
        f"--benchmark_out={out}",
        "--threads=8",
    ]
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def load_medians(path: Path):
    data = json.loads(path.read_text())
    result = {}
    stddev = {}
    for bench in data["benchmarks"]:
        # A fixed iteration count adds /iterations:N to the name.
        bench["name"] = re.sub(r"/iterations:[0-9]+", "", bench["name"])
        if bench["name"].endswith("_median"):
            result[bench["name"]] = bench
        elif bench["name"].endswith("_stddev"):
            stddev[bench["name"]] = bench["real_time"]
    for name, bench in result.items():
        std_name = name[: -len("_median")] + "_stddev"
        if std_name in stddev:
            bench["_stddev"] = stddev[std_name]
    if not result:
        # A one-repetition run with aggregates enabled has no _median entry.
        for bench in data["benchmarks"]:
            if bench["name"].endswith("/manual_time"):
                result[bench["name"].replace("/manual_time",
                                             "/manual_time_median")] = bench
    return result


def expected_name(family: str, backend: str, scene: str):
    return f"{family}/{backend}/{scene}/manual_time_median"


def run_replay(args, work: Path):
    """Runs each trace once per backend, each in its own process, so the
    reported memory belongs to that backend alone."""
    results = {}
    for path in sorted(Path(args.traces).glob("*.trace")):
        scene = path.stem.replace(".", "p")
        results[scene] = {}
        for backend in BACKENDS:
            out = work / f"replay_{backend}_{scene}.json"
            out.unlink(missing_ok=True)
            cmd = [
                "taskset", "-c", args.cpus,
                str(args.build / "bench" / "ispc_apfrnn_bench"),
                f"--benchmark_filter=^Replay/{backend}/{re.escape(scene)}/",
                f"--benchmark_repetitions={args.replay_repetitions}",
                "--benchmark_report_aggregates_only=true",
                "--benchmark_format=json",
                f"--benchmark_out={out}",
                f"--threads={args.threads}",
                f"--traces={args.traces}",
            ]
            print("+", " ".join(cmd), file=sys.stderr)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            if not out.exists() or not out.read_text().strip():
                continue
            medians = load_medians(out)
            if medians:
                results[scene][backend] = next(iter(medians.values()))
    (work / "replay.json").write_text(json.dumps(results, indent=1) + "\n")


def run_matrix(args, work: Path):
    work.mkdir(parents=True, exist_ok=True)
    wanted = set(re.findall(r"bench:([\w-]+):start",
                            Path("README.md").read_text()))

    if "replay" in wanted:
        run_replay(args, work)
    if "headline-timing" in wanted:
        run_bench(args.build, args.cpus, "Frame", scene_names(HEADLINE),
                  BACKENDS, work / "headline.json", [], args.min_time,
                  args.repetitions, args.threads)
    if "stress-timing" in wanted:
        run_bench(args.build, args.cpus, "Stress", scene_names(STRESS),
                  BACKENDS, work / "stress.json", [], args.min_time,
                  args.repetitions, args.threads)
    if "learnsph" in wanted:
        run_bench(args.build, args.cpus, "Frame", scene_names(LEARNSPH),
                  BACKENDS, work / "learnsph.json",
                  [f"--learnsph_res={args.learnsph_res}"], args.min_time,
                  args.repetitions, args.threads)
    if "order" in wanted:
        run_bench(args.build, args.cpus, "Order",
                  [scene for row in ORDER for scene in row[1:]], BACKENDS,
                  work / "order.json", [], args.min_time, args.repetitions,
                  args.threads)
    if "dynamic" in wanted:
        run_bench(args.build, args.cpus, "Dynamic",
                  [scene for row in DYNAMIC for scene in row[1:]], BACKENDS,
                  work / "dynamic.json",
                  [f"--learnsph_res={args.learnsph_res}"], args.min_time,
                  args.repetitions, args.threads)
    if "scale" in wanted and not args.no_scale:
        run_bench(args.build, args.cpus, "Scale", scene_names(SCALE), BACKENDS,
                  work / "scale.json", [],
                  args.min_time, args.repetitions, args.threads)
    if "growing" in wanted:
        run_bench(args.build, args.cpus, "Growing", scene_names(GROWING),
                  ["apfrnn_cpu", "tree"], work / "growing.json", [],
                  args.min_time, args.repetitions, args.threads)
    if "threads" in wanted:
        run_threads(args.build, args.cpus, scene_names(THREADS), BACKENDS,
                    work / "threads.json", args.min_time, args.repetitions)
    if "motion" in wanted:
        motion_scenes = {
            family: [name for _, row_family, static, jitter in MOTION
                     if row_family == family for name in (static, jitter)]
            for family in ("Frame", "Stress")
        }
        run_bench(args.build, args.cpus, "Frame", motion_scenes["Frame"],
                  ["apfrnn_cpu"], work / "motion.json", [], args.min_time,
                  args.repetitions, args.threads)
        run_bench(args.build, args.cpus, "Stress", motion_scenes["Stress"],
                  ["apfrnn_cpu"], work / "droplets_motion.json", [],
                  args.min_time, args.repetitions, args.threads)
    memory = {}
    if not args.no_memory:
        for family, rows, marker in (("Frame", HEADLINE, "headline-memory"),
                                     ("Stress", STRESS, "stress-memory")):
            if marker not in wanted:
                continue
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
        for _, scene, _ in LEARNSPH if "learnsph" in wanted else []:
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
             "droplets_motion", "order", "dynamic", "scale", "growing",
             "threads"]
    data = {}
    for name in names:
        path = json_dir / f"{name}.json"
        if path.exists():
            data[name] = load_medians(path)
        else:
            data[name] = {}
    replay_path = json_dir / "replay.json"
    data["replay"] = (json.loads(replay_path.read_text())
                      if replay_path.exists() else {})
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


def gpu_label(scene):
    special = {
        "uniform_1m": "uniform, 1M",
        "uniform_200k": "uniform, 200k",
        "uniform_50k": "uniform, 50k",
        "uniform_200k_moving": "uniform, moving",
        "jittered_200k": "jittered lattice, 200k",
        "jittered_50k": "jittered lattice, 50k",
        "slab_200k": "SPH-like slab, 200k",
        "slab_50k": "SPH-like slab, 50k",
        "slab_200k_moving": "slab, moving",
        "surface_200k": "surface, 200k",
        "surface_50k": "surface, 50k",
        "clustered_200k": "clustered, 200k",
        "thin_slab_200k": "thin slab, 200k",
        "huge_domain_200k": "huge sparse domain, 200k",
        "two_scale_200k": "two-scale clusters, 200k",
        "thin_pipe_200k": "thin pipe, 200k",
        "thin_pipe_50k": "thin pipe, 50k",
        "pathological_10k": "pathological, 10k",
    }
    if scene in special:
        return special[scene]
    if scene.startswith("learnsph_"):
        return scene.replace("_", " ")
    return scene.replace("_", " ")


def table_gpu(data):
    lines = [
        "| Scene | CPU | GPU | Auto | GPU / CPU |",
        "|---|---:|---:|---:|---:|",
    ]
    fallback = False
    for name, cpu in data.items():
        if "/apfrnn_cpu/" not in name:
            continue
        parts = name.split("/")
        if len(parts) != 4:
            continue
        family, _, scene = parts[:3]
        gpu_name = f"{family}/apfrnn_gpu/{scene}/manual_time_median"
        auto_name = f"{family}/apfrnn_auto/{scene}/manual_time_median"
        if gpu_name not in data or auto_name not in data:
            continue
        gpu = data[gpu_name]
        auto = data[auto_name]
        star = ""
        if gpu.get("Fallback", 0):
            star = "*"
            fallback = True
        label = gpu_label(scene)
        if family != "Frame":
            label = f"{family}: {label}"
        lines.append(
            f"| {label} | {fmt_ms(cpu['real_time'])} | "
            f"{fmt_ms(gpu['real_time'])}{star} | {fmt_ms(auto['real_time'])} | "
            f"{gpu['real_time']/cpu['real_time']:.2f}x |")
    if fallback:
        lines.append("")
        lines.append("\\* CPU fallback.")
    return "\n".join(lines)


def search_ratio(other, ours):
    """Ratio of the search time alone (run(), without the density pass)."""
    if not other or not other.get("RunMs") or not ours.get("RunMs"):
        return "n/a"
    return fmt_ratio(other["RunMs"] / ours["RunMs"])


def table_timing(data, rows, family):
    lines = [
        "| Scene | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, scene in rows:
        a = get(data, family, "apfrnn_cpu", scene)
        t = get(data, family, "tree", scene)
        c = get(data, family, "compact", scene)
        ta, tt, tc = a["real_time"], t["real_time"], c["real_time"]
        lines.append(
            f"| {label} | {a['AvgNeighbors']:.0f} | {fmt_ms(ta)} | "
            f"{fmt_ms(tt)} | {fmt_ms(tc)} | {fmt_ratio(tt/ta)} | "
            f"{search_ratio(t, a)} | {fmt_ratio(tc/ta)} |")
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


def available(data, family, backend, scene):
    return expected_name(family, backend, scene) in data


def table_order(data):
    order = data["order"]
    lines = [
        "| Scene | ispc z-sorted | Tree z-sorted | vs Tree | ispc unsorted | Tree unsorted | vs Tree |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, sorted_scene, unsorted_scene in ORDER:
        if not available(order, "Order", "tree", sorted_scene):
            continue
        a_s = get(order, "Order", "apfrnn_cpu", sorted_scene)
        t_s = get(order, "Order", "tree", sorted_scene)
        a_u = get(order, "Order", "apfrnn_cpu", unsorted_scene)
        t_u = get(order, "Order", "tree", unsorted_scene)
        lines.append(
            f"| {label} | {fmt_ms(a_s['real_time'])} | "
            f"{fmt_ms(t_s['real_time'])} | "
            f"{fmt_ratio(t_s['real_time'] / a_s['real_time'])} | "
            f"{fmt_ms(a_u['real_time'])} | {fmt_ms(t_u['real_time'])} | "
            f"{fmt_ratio(t_u['real_time'] / a_u['real_time'])} |")
    return "\n".join(lines)


def table_dynamic(data):
    dynamic = data["dynamic"]
    lines = [
        "| Scene | Boundary | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, static_scene, dynamic_scene in DYNAMIC:
        for tag, scene in (("static", static_scene),
                           ("all dynamic", dynamic_scene)):
            if not available(dynamic, "Dynamic", "tree", scene):
                continue
            a = get(dynamic, "Dynamic", "apfrnn_cpu", scene)
            t = get(dynamic, "Dynamic", "tree", scene)
            c = get(dynamic, "Dynamic", "compact", scene)
            lines.append(
                f"| {label} | {tag} | {fmt_ms(a['real_time'])} | "
                f"{fmt_ms(t['real_time'])} | {fmt_ms(c['real_time'])} | "
                f"{fmt_ratio(t['real_time'] / a['real_time'])} |")
    return "\n".join(lines)


def replay_label(scene):
    match = re.fullmatch(r"(.+)_r([0-9p]+)_t([0-9p]+)", scene)
    if not match:
        return scene
    name, radius, time = match.groups()
    return f"{name.replace('_', ' ')}, t={time.replace('p', '.')} s"


def fmt_points(value):
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    return f"{value / 1e3:.0f}k"


def table_replay(data):
    rows = []
    for scene, backends in data["replay"].items():
        a = backends.get("apfrnn_cpu")
        if a:
            rows.append((a["MovingPoints"], scene, a, backends.get("tree"),
                         backends.get("compact")))
    lines = [
        "| Simulation | Moving particles | Avg neighbors | ispc_apfrnn | TreeNSearch | "
        "CompactNSearch | vs Tree | vs Tree (search) | vs Compact | "
        "Memory ours / Tree / Compact (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def cell(entry, key, fmt):
        return fmt(entry[key]) if entry else "n/a"
    def ratio(entry, a):
        return fmt_ratio(entry["real_time"] / a["real_time"]) if entry else "n/a"
    def mb(value):
        return f"{value:.0f}"
    for _, scene, a, t, c in sorted(rows, key=lambda row: (row[0], row[1])):
        lines.append(
            f"| {replay_label(scene)} | {fmt_points(a['MovingPoints'])} | "
            f"{a['Neighbors'] / max(1, a['MovingPoints']):.0f} | "
            f"{fmt_ms(a['real_time'])} | "
            f"{cell(t, 'real_time', fmt_ms)} | {cell(c, 'real_time', fmt_ms)} | "
            f"{ratio(t, a)} | {search_ratio(t, a)} | {ratio(c, a)} | "
            f"{mb(a['AnonMB'])} / "
            f"{cell(t, 'AnonMB', mb)} / {cell(c, 'AnonMB', mb)} |")
    return "\n".join(lines)


def table_scale(data):
    scale = data["scale"]
    lines = [
        "| Distribution | Points | Avg degree | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree | vs Tree (search) | vs Compact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for distribution, scene in SCALE:
        if not available(scale, "Scale", "tree", scene):
            continue
        a = get(scale, "Scale", "apfrnn_cpu", scene)
        t = get(scale, "Scale", "tree", scene)
        c = get(scale, "Scale", "compact", scene)
        points = scene.rsplit("_", 1)[-1].upper()
        if points.endswith("K"):
            points = points[:-1] + "k"
        lines.append(
            f"| {distribution} | {points} | {a['AvgNeighbors']:.0f} | "
            f"{fmt_ms(a['real_time'])} | {fmt_ms(t['real_time'])} | "
            f"{fmt_ms(c['real_time'])} | "
            f"{fmt_ratio(t['real_time'] / a['real_time'])} | "
            f"{search_ratio(t, a)} | "
            f"{fmt_ratio(c['real_time'] / a['real_time'])} |")
    return "\n".join(lines)


def table_growing(data):
    growing = data["growing"]
    lines = [
        "| Scene | ispc_apfrnn | TreeNSearch | vs Tree | vs Tree (search) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, scene in GROWING:
        if not available(growing, "Growing", "tree", scene):
            continue
        a = get(growing, "Growing", "apfrnn_cpu", scene)
        t = get(growing, "Growing", "tree", scene)
        lines.append(
            f"| {label} | {fmt_ms(a['real_time'])} | "
            f"{fmt_ms(t['real_time'])} | "
            f"{fmt_ratio(t['real_time'] / a['real_time'])} | "
            f"{search_ratio(t, a)} |")
    return "\n".join(lines)


def table_threads(data):
    threads = data["threads"]
    lines = [
        "| Scene | Threads | ispc_apfrnn | TreeNSearch | CompactNSearch | vs Tree |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, scene, counts in THREADS:
        for count in counts:
            name = f"{scene}/{count}"
            if not available(threads, "Threads", "tree", name):
                continue
            a = get(threads, "Threads", "apfrnn_cpu", name)
            t = get(threads, "Threads", "tree", name)
            c = get(threads, "Threads", "compact", name)
            lines.append(
                f"| {label} | {count} | {fmt_ms(a['real_time'])} | "
                f"{fmt_ms(t['real_time'])} | {fmt_ms(c['real_time'])} | "
                f"{fmt_ratio(t['real_time'] / a['real_time'])} |")
    return "\n".join(lines)


def replace_block(text: str, name: str, content: str):
    start = f"<!-- bench:{name}:start -->"
    end = f"<!-- bench:{name}:end -->"
    left = text.find(start)
    right = text.find(end, left + 1)
    if left < 0 or right < 0:
        # README may intentionally omit optional benchmark tables.
        return text
    return text[:left] + start + "\n" + content + "\n" + text[right:]


def generate(data, memory, gpu_data=None):
    text = Path("README.md").read_text()
    tables = {
        "replay": lambda: table_replay(data),
        "headline-timing": lambda: table_timing(data["headline"], HEADLINE, "Frame"),
        "headline-memory": lambda: table_memory(data["headline"], memory, HEADLINE,
                                                "Frame"),
        "stress-timing": lambda: table_timing(data["stress"], STRESS, "Stress"),
        "stress-memory": lambda: table_memory(data["stress"], memory, STRESS,
                                              "Stress"),
        "motion": lambda: table_motion(data),
        "learnsph": lambda: table_learnsph(data, memory),
        "order": lambda: table_order(data),
        "dynamic": lambda: table_dynamic(data),
        "scale": lambda: table_scale(data),
        "growing": lambda: table_growing(data),
        "threads": lambda: table_threads(data),
    }
    wanted = set(re.findall(r"bench:([\w-]+):start", text))
    replacements = {name: table() for name, table in tables.items()
                    if name in wanted}
    if gpu_data is not None:
        replacements["gpu"] = table_gpu(gpu_data)
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
    parser.add_argument("--replay-repetitions", type=int, default=3)
    parser.add_argument("--traces", default="traces")
    parser.add_argument("--threads", type=int, default=8,
                        help="worker threads for Frame/Stress/Cold runs")
    parser.add_argument("--no-scale", action="store_true",
                        help="skip the 175k..8M dense/sparse scale run")
    parser.add_argument("--memory-time", default="0.5s")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--learnsph-res", default="../learnsph/res")
    parser.add_argument("--gpu-json", type=Path,
                        help="optional JSON from a SYCL GPU benchmark run")
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

    gpu_data = load_medians(args.gpu_json) if args.gpu_json else None
    expected = generate(data, memory, gpu_data)
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
