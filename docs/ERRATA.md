# Errata

Public log of corrections to BenchCAD ground-truth items and scoring. We fix
flawed items in the open and bump the version rather than freezing a known-bad
set.

## Resolved

### 2026-06-27 — `double_simplex_sprocket` GT timeouts (≈26 Vision2Code records)

**Problem.** The Vision2Code ground-truth STEP solids are generated locally at
download time by running each GT CadQuery program in a subprocess with a 90 s
timeout (`scoring/exec_cq.py::execute_cq_to_step`). The `double_simplex_sprocket`
family is unusually slow to tessellate/export (its slowest instances take ~50–68 s
even on fast hardware). On slower machines, under parallel evaluation, or with
subprocess + import overhead, the slowest of these exceed 90 s and fail to produce
a STEP, so they get **omitted** — independent evaluators (including Anthropic, who
reported evaluating 17,874 of 17,900 files) drop the same ~26 records.

**Root cause.** Not broken code: all 17,900 GT programs execute and export a valid
STEP given enough time (verified — 0 hard failures in the pinned
`cadquery 2.3.0` / `cadquery-ocp 7.9.3.0` environment). The ~26 omissions are a
**timeout/hardware sensitivity** of one slow-geometry family, not a data defect.

**Fix.** Raised the `execute_cq_to_step` default timeout from 90 s to 300 s (the
slowest valid program is ~68 s on fast hardware, leaving generous margin for slower
machines and contention). No dataset change is needed — GT STEPs are produced
locally on download — so with the larger timeout all **17,900** records are
scorable. Anthropic's published numbers are over the 17,874-record subset; results
on the full set are directly comparable once the slow records build.

## Per-record corrections

Each entry: date · `record_id` · what was wrong · the fix · version.

| Date | record_id | Problem | Fix | Version |
|---|---|---|---|---|
| — | — | _none reported yet_ | — | — |

Found a wrong item? See [CONTRIBUTING.md](../CONTRIBUTING.md) section C.
