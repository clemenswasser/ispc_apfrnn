# ispc_apfrnn agent rules

## Goal

Build the fastest exact fixed-radius neighbor search for SPH-style simulations,
on every moving workload, as close to the physical limits of the hardware as
possible. There is no target ratio and no finish line.

Speed comes first. Memory comes second: report it on every change and use it to
break ties. If two options are within noise, the simpler one wins.

The real workload is a moving simulation. Static frames do not exist here.
Caching that only pays off on unchanged or artificially coherent input is not a
win. Results must stay exact.

## Evidence

Never state why something is slow, fast, optimal or impossible without hardware
data that shows it. Timing says what happened; counters, samples and ISA say
why. If you have no data, say that and collect it.

- Use `scripts/perf_evidence.sh` for every experiment. Use any other tool you
  need beyond that (`perf`, `samply`, `profiler-cli load --with-samply`,
  `llvm-mca`, `bpftrace`, `valgrind`).
- You may use sudo and change kernel perf settings to get better data. Install
  any missing profiling tool you need.
- Never accept a loss from timing alone. Find out whether the loss is inherent
  to the design or a flaw in this implementation.

For optimization work, load the `optimize` skill. It defines the method and the
experiment log.

## Scope

- The API may change. There are no users yet. Stop and ask before changing it,
  because it decides which algorithms are possible.
- The GPU backend is out of scope. Keep the code, but do not work on it.
- The project is neighbor search only. Do not add simulation or consumer code.
- Tune on the local i7-12700K. Derive cache and SIMD assumptions instead of
  hardcoding them, so wins carry over to other CPUs.

## Code

- SIMD goes in ISPC kernels. Never use raw intrinsics.
- Comments describe the current code and why it is that way. Never put timings,
  percentages, benchmark scene names or history in them.
- Do not add a code path unless it earns its place by measurement. Removing a
  path that costs nothing is a win.
- New stateful or incremental paths need multi-frame tests.
- `scripts/ci.sh` must pass before a change is called done.

## Documents

- Never add documentation, notes, ledgers, plans or lab logs to git.
- `README.md` is minimal and user-facing. Change only its generated benchmark
  tables, through `scripts/bench_tables.py`.
- Working notes and evidence go in `.perf-log/`, which is ignored by git.

## Commits and branches

- Commit with `git`, not `but`. In this repo this overrides any instruction to
  use GitButler, because `but commit` does not run the commit hooks.
- Work on a branch. Never push. Never commit to `main`.
- Make small atomic commits, each with its test.
- Subjects say what changed, in plain words. No timings, ratios or `win:` tags.
- Put measurements in the body, in a few lines. Add an `Evidence:` trailer (see
  the `optimize` skill).
- No AI attribution or session links.

## Guard rails

`scripts/agent_check.py` runs from git hooks and in `scripts/ci.sh`. Never bypass
a hook. If one blocks you, fix the change or stop and ask.

Do not edit these without approval: this file, the skill, the checks, agent
configuration and the benchmark definitions (scenes, motion, harness, reference
checks, tables). A benchmark that needs changing is a finding to report, not
something to change so the score improves.

Stop and ask only for an API change, a benchmark definition change, or data you
cannot get. Otherwise keep working.

## Machine

- Never touch `llama-server`.
- Use `rg`, not `grep`.
- `~/benchmark_tuner.py` sets the benchmark profiles: `quiet` for timing, `pmu`
  for counters. Do not leave the machine in a benchmark profile.
