#!/usr/bin/env python3
"""Paired A/B timing of ispc_apfrnn between a baseline commit and a candidate
build, on our backend only.

Rounds alternate the two builds (base first on even rounds, candidate first on
odd ones), so drift in clock, temperature or background load hits both. Each
scene's result is the mean of the per-round candidate/base log ratios. A scene
counts as faster or slower only when a paired t-test rejects no change at 95%
and the change is at least 1%; the +- column is that 95% interval. Every run checks its lists against the reference and the script
stops on a mismatch. The benchmark times a fixed window of frames per scene
(its Frames counter), the same window the README tables time; a scene whose
window differs between the builds is reported instead of compared.

Usage:
  scripts/ab.py [--base REF] [--candidate DIR] [--tier quick|full]
                [--filter REGEX] [--rounds N]

  --base       commit to compare against (default HEAD). Built in build-ab/.
  --candidate  build directory of the candidate (default build). It is rebuilt
               from its own source tree first.
  --tier       quick: a few replays and synthetic scenes, for iterating.
               full: every replay, frame, stress, scale and growing scene,
               required before accepting a change.
  --filter     regex over scene names (family/scene) instead of a tier.

Results go to .perf-log/ab/<time>.json. Run in the quiet benchmark profile.
"""

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = "apfrnn_cpu"

QUICK = [
    "Replay/dam_break_r0p0025_t0p10",
    "Replay/droplet_r0p01_t0p05",
    "Replay/funnel_r0p006_t0p20",
    "Replay/sink_r0p005_t0p40",
    "Replay/wall_r0p005_t0p10",
    "Frame/uniform_200k_moving",
    "Frame/clustered_200k_jitter",
    "Stress/droplets_200k",
]
FULL_FAMILIES = ["Replay", "Frame", "Stress", "Scale", "Growing"]


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


def git(*args, cwd=ROOT):
    return run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def common_root():
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent


def build_base(ref, candidate):
    sha = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    work = common_root() / "build-ab"
    src = work / "base-src"
    build = work / "base"
    work.mkdir(exist_ok=True)
    if not (src / ".git").exists():
        run(["git", "worktree", "add", "--detach", str(src), sha], cwd=ROOT,
            capture_output=True)
    else:
        run(["git", "-C", str(src), "checkout", "--detach", "--quiet", sha])
    # Same configuration as the candidate, so code layout differences come
    # from the source change and not from build options.
    options = candidate_options(candidate)
    stamp = build / "ab-options.txt"
    if not (build / "build.ninja").exists() or (
            not stamp.exists() or stamp.read_text() != "\n".join(options)):
        run(["cmake", "-S", str(src), "-B", str(build), "-G", "Ninja",
             *options], stdout=subprocess.DEVNULL)
        stamp.write_text("\n".join(options))
    print(f"building base {sha[:9]} in {build}", file=sys.stderr)
    run(["cmake", "--build", str(build), "--target", "ispc_apfrnn_bench"],
        stdout=subprocess.DEVNULL)
    return sha, build


def candidate_options(build):
    keep = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
            "CMAKE_ISPC_FLAGS", "CMAKE_EXE_LINKER_FLAGS")
    options = []
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([A-Za-z_]+):[A-Z]+=(.*)$", line)
        if match and (match[1] in keep or match[1].startswith("APFRNN_")):
            options.append(f"-D{match[1]}={match[2]}")
    return sorted(options)


def build_candidate(build):
    print(f"building candidate in {build}", file=sys.stderr)
    run(["cmake", "--build", str(build), "--target", "ispc_apfrnn_bench"],
        stdout=subprocess.DEVNULL)


def scene_regex(args):
    if args.filter:
        families = "|".join(FULL_FAMILIES)
        return f"^({families})/{BACKEND}/", re.compile(args.filter)
    if args.tier == "full":
        return f"^({'|'.join(FULL_FAMILIES)})/{BACKEND}/", None
    parts = []
    for name in QUICK:
        family, scene = name.split("/", 1)
        parts.append(f"{family}/{BACKEND}/{re.escape(scene)}/")
    return "^(" + "|".join(parts) + ")", None


def list_scenes(bench, regex, select, traces):
    out = run([str(bench), f"--benchmark_filter={regex}",
               "--benchmark_list_tests=true", f"--traces={traces}"],
              cwd=ROOT, capture_output=True, text=True).stdout.split()
    names = []
    for line in out:
        family, _, scene = line.split("/")[:3]
        name = f"{family}/{scene}"
        if select is None or select.search(name):
            names.append(name)
    return names


def bench_once(bench, names, args, traces):
    parts = []
    for name in names:
        family, scene = name.split("/", 1)
        parts.append(f"{family}/{BACKEND}/{re.escape(scene)}/")
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
        cmd = ["taskset", "-c", args.cpus, str(bench),
               "--benchmark_filter=^(" + "|".join(parts) + ")",
               "--benchmark_format=json", f"--benchmark_out={out.name}",
               f"--threads={args.threads}", f"--traces={traces}"]
        result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            sys.exit(f"benchmark failed ({result.returncode}) in {bench}:\n"
                     + result.stderr[-2000:])
        data = json.loads(Path(out.name).read_text())
    times = {}
    for entry in data["benchmarks"]:
        family, _, scene = entry["run_name"].split("/")[:3]
        memory = entry.get("NeighborMB", 0.0) + entry.get("GridMB", 0.0)
        times[f"{family}/{scene}"] = (entry["real_time"], memory,
                                      entry.get("Frames", 0.0))
    return times


# Two-sided Student t quantiles at the 0.95 level, by degrees of freedom.
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
       9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}
MIN_EFFECT = 0.01


def t95(dof):
    keys = [k for k in sorted(T95) if k <= dof]
    return T95[keys[-1]] if keys else T95[1]


def summarize(name, base, cand):
    logs = [math.log(c[0] / b[0]) for b, c in zip(base, cand)]
    n = len(logs)
    mean = sum(logs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / (n - 1)) if n > 1 else 0.0
    half = t95(n - 1) * sd / math.sqrt(n) if n > 1 else math.inf
    windows = {sample[2] for sample in base} | {sample[2] for sample in cand}
    if len(windows) > 1:
        verdict = "window"
    elif n < 3 or abs(mean) <= half or abs(mean) < math.log1p(MIN_EFFECT):
        verdict = "noise"
    else:
        verdict = "faster" if mean < 0 else "slower"
    median = lambda values: sorted(values)[len(values) // 2]
    return {
        "scene": name,
        "base_ms": median([b[0] for b in base]),
        "cand_ms": median([c[0] for c in cand]),
        "speedup": math.exp(-mean),
        "noise_pct": 100.0 * (math.exp(half) - 1.0),
        "base_mb": median([b[1] for b in base]),
        "cand_mb": median([c[1] for c in cand]),
        "verdict": verdict,
    }


def tuned_profile():
    try:
        out = subprocess.run(["tuned-adm", "active"], capture_output=True,
                             text=True).stdout
    except OSError:
        return ""
    return out.strip().rsplit(":", 1)[-1].strip()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--candidate", type=Path, default=ROOT / "build")
    parser.add_argument("--tier", choices=["quick", "full"], default="quick")
    parser.add_argument("--filter")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--cpus", default="0-15")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--traces", type=Path, default=None)
    args = parser.parse_args()
    rounds = args.rounds or (7 if args.tier == "quick" and not args.filter else 5)
    traces = args.traces or common_root() / "traces"

    profile = tuned_profile()
    if profile and profile != "benchmark-quiet":
        print(f"warning: tuned profile is {profile}, not benchmark-quiet; "
              "run ~/benchmark_tuner.py apply quiet", file=sys.stderr)

    candidate = args.candidate.resolve()
    build_candidate(candidate)
    sha, base = build_base(args.base, candidate)
    builds = {"base": base / "bench" / "ispc_apfrnn_bench",
              "cand": candidate / "bench" / "ispc_apfrnn_bench"}

    regex, select = scene_regex(args)
    names = list_scenes(builds["cand"], regex, select, traces)
    base_names = set(list_scenes(builds["base"], regex, select, traces))
    names = [name for name in names if name in base_names]
    if not names:
        sys.exit("no scenes selected")
    print(f"{len(names)} scenes, {rounds} rounds", file=sys.stderr)

    samples = {"base": {}, "cand": {}}
    for index in range(rounds):
        order = ["base", "cand"] if index % 2 == 0 else ["cand", "base"]
        for side in order:
            for name, value in bench_once(builds[side], names, args,
                                          traces).items():
                samples[side].setdefault(name, []).append(value)
        print(f"round {index + 1}/{rounds} done", file=sys.stderr)

    rows = [summarize(name, samples["base"][name], samples["cand"][name])
            for name in names
            if name in samples["base"] and name in samples["cand"]]
    rows.sort(key=lambda row: row["speedup"])

    print(f"base {sha[:9]} vs candidate {candidate}, {rounds} rounds")
    print(f"{'scene':44} {'base ms':>10} {'cand ms':>10} {'speedup':>8} "
          f"{'+-':>6} {'base MB':>8} {'cand MB':>8}  verdict")
    for row in rows:
        print(f"{row['scene']:44} {row['base_ms']:10.3f} {row['cand_ms']:10.3f} "
              f"{row['speedup']:7.3f}x {row['noise_pct']:5.1f}% "
              f"{row['base_mb']:8.1f} {row['cand_mb']:8.1f}  {row['verdict']}")
    counts = {verdict: sum(row["verdict"] == verdict for row in rows)
              for verdict in ("faster", "slower", "noise")}
    geomean = math.exp(sum(math.log(row["speedup"]) for row in rows) / len(rows))
    print(f"faster {counts['faster']}, slower {counts['slower']}, "
          f"noise {counts['noise']}, geomean speedup {geomean:.3f}x")
    for row in rows:
        if row["verdict"] == "window":
            print(f"warning: {row['scene']} times a different frame window in "
                  "the two builds (its neighbor count or the benchmark changed); "
                  "compare it by hand", file=sys.stderr)
    flagged = [row["scene"] for row in rows
               if row["verdict"] in ("faster", "slower")]
    if flagged and len(rows) > len(flagged):
        # Each scene is tested on its own, so about one scene in twenty is flagged
        # by chance.
        print("confirm flagged scenes: scripts/ab.py --rounds 9 --filter '^("
              + "|".join(re.escape(name) for name in flagged) + ")$'")

    log_dir = common_root() / ".perf-log" / "ab"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {"base": sha, "candidate": str(candidate),
              "candidate_head": git("rev-parse", "HEAD", cwd=candidate.parent)
              if (candidate.parent / ".git").exists() else "",
              "tier": args.filter or args.tier, "rounds": rounds,
              "profile": profile, "rows": rows}
    path = log_dir / f"{stamp}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"saved {path.relative_to(common_root())}")


if __name__ == "__main__":
    main()
