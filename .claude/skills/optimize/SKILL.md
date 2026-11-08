---
name: optimize
description: Method for performance work on ispc_apfrnn. Load before any optimization, profiling, algorithm prototype or benchmark analysis, and for long-running optimization goals.
---

# Optimize

Optimization is a search through a decision tree. Choices high in the tree
(algorithm, list format, data layout, parallelism) bound what everything below
them can reach. Work from the top down, and go down a level only when the level
above has little gap left to its floor.

| Level | Decides |
|---|---|
| `algorithm` | Search structure, list format, API contract, use of coherence |
| `layout` | Data layout, memory order, allocation |
| `parallel` | Task split, load balance, sharing, NUMA |
| `simd` | ISPC kernel shape, gang use, candidate tests |
| `micro` | Prefetch, alignment, thresholds, branch layout |

## Phases

Work through these in order. Check `.perf-log/experiments.jsonl` to see where
things stand.

### 0. Workload

Do not optimize until the benchmark represents real use. It needs:

- Headline scenes that replay positions captured from `../learnsph` at every
  solver substep (not VTK output frames), across as many scenarios and sizes as
  possible, with particles entering and leaving.
- Synthetic scenes as extra coverage, every one of them moving.
- Timed regions that exclude scene updates and include reading the lists.
- Validation of timed frames, not just warm-up frames.
- Library default settings, or any other setting shown next to its results.

Changing the benchmark definition needs approval.

### 1. Floors

For each workload, estimate the minimum possible cost from physical limits:
output bytes against measured write bandwidth, input bytes against read
bandwidth, required candidate tests against peak SIMD throughput. Measure those
limits on this machine. Record the floor and its assumptions in `.perf-log/`.
Floors are estimates, so refine them when evidence disagrees.

### 2. Algorithm search

Prototype the algorithm families as throwaway versions, including grid and sort
variants, frame-to-frame coherence, half lists, compact list formats and
library-owned particle reordering. Give each family enough work to be
competent, then fund the most promising and drop the rest. Prototypes do not
need production quality, and they are not committed until they win.

A family is dead only when evidence shows the loss is inherent to the design.

### 3. Descent

Take the chosen design down the levels. Each turn, pick the largest gap to floor
on a real workload, and run one experiment against its dominant measured cost.
Do not chase a scene because a competitor's number looks odd there.

### 4. Removal

After a batch of wins, try removing each special-case path. Remove it if the
scenes do not notice.

## One experiment

1. Run `scripts/perf_evidence.sh <id> before '<regex>'` on the target scene.
2. Write down the hypothesis: which measured cost it reduces and by how much.
3. Implement it generally, not for one scene.
4. Run `scripts/perf_evidence.sh <id> after '<regex>'`. Add kernel names as extra
   arguments to get an `llvm-mca` report.
5. Compare timing in a paired A/B, in the quiet profile, on the same scene.
6. Explain the result from the data, including whether the prediction held.
7. If it lost, keep digging until you can say whether the loss is `local` (this
   implementation) or `global` (the design). A local loss is the next experiment.
8. Log it. Commit wins and revert losses.

## Accepting a change

- It beats noise on the target scene (at least 2 standard deviations across
  repeated paired runs).
- No other scene gets slower beyond noise. Check the full matrix, not a subset.
- Tests and checksums pass.
- Speed wins over memory. Report the memory change either way.

Competitors are measured for the README and as a must-not-lose check. They are
not the objective, because beating them is not the limit.

## Log

Add one JSON object per line to `.perf-log/experiments.jsonl`:

```json
{"id": "e041-half-lists", "level": "algorithm", "hypothesis": "...", "result": "rejected", "mechanism": "...", "dead_end": "local", "evidence": ["before", "after"]}
```

- `result`: `kept`, `rejected` or `open`.
- `mechanism`: why it won or lost, citing measured data.
- `dead_end`: `local`, `global` or `none`. Rejected results need `local` or `global`.
- `evidence`: bundle labels under `.perf-log/evidence/<id>/`. Kept and rejected
  results need two.

Read the log before each experiment, and do not repeat a rejection unchanged.

Commits touching `src/` or `include/` need a trailer naming a kept experiment:

```
Evidence: e041-half-lists
```

Use `Evidence: none - <reason>` for changes that are not performance work. While
any log entry is incomplete, the commit hook blocks these commits.

## Long runs

Keep going without asking, except for the stop conditions in `AGENTS.md`. When a
branch of the tree runs dry, move to the next-largest gap. If every gap is small,
extend floors and scene coverage. Do not start tuning thresholds to get a number
moving.
