# Errata

Public log of corrections to BenchCAD ground-truth items and scoring. We fix
flawed items in the open and bump the version rather than freezing a known-bad
set.

## Resolved

### 2026-07 — Negative-gold `dim` answers were unscoreable (18 Code QA rows)

**Problem.** `QA/scoring/qa_score.py::qa_score_single` scored `dim`/`ratio` answers
with a symmetric min/max ratio and returned 0 whenever either side was ≤ 0. The
"sum of all extrude/cutBlind depth values" template (tier_a_L3) has **18 Code QA
records with a negative gold**: `cutBlind` depths are negative in CadQuery, so the
signed sum of the depth literals is negative. An exactly-correct answer therefore
scored 0, which capped the attainable Code QA score at ~0.9925 — for all models
equally. Reported by @jaemoshin (#33).

**Not a data defect.** The golds are correct: for all 18 rows the parsed signed sum
of the extrude/cutBlind depth literals equals the stored gold.

**Fix.** `qa_score_single` now compares magnitudes with a sign-match check —
`min(|pred|,|gt|) / max(|pred|,|gt|)`, 0 if the signs differ, and a 0 answer scores
1.0 only when `pred == gt`. Positive-answer scoring is unchanged; regression test in
`QA/tests/test_qa_score.py`. Code QA scores rise slightly for models that answered
these rows correctly, so listed Code QA numbers should be re-graded.

### 2026-06 — 26 Vision2Code GT records that don't always build a STEP

**Symptom.** Vision2Code scores each prediction against a ground-truth STEP solid.
STEPs are not shipped in the dataset — they are built at download time by running
each record's GT CadQuery program in a subprocess
(`scoring/exec_cq.py::execute_cq_to_step`). The slowest tail of programs — led by
the `double_simplex_sprocket` family — can exceed the execution timeout on slower
hardware or under parallel contention; the subprocess is killed, no STEP is
written, and those records are dropped. Independent evaluators hit the same
records: Anthropic's Fable 5 / Mythos 5 system card reports evaluating 17,874 of
17,900 Vision2Code files, omitting "26 records whose CadQuery code failed to
produce a STEP file."

**Not a data defect.** We rebuilt the entire `double_simplex_sprocket` family (the
slow-family culprit) from its GT code: **185 / 185 execute and export a valid,
scorable STEP** — 0 execution failures, 0 tessellation failures — in the pinned
`cadquery 2.3.0` / `cadquery-ocp 7.9.3.0` environment. Under 6-way parallel
contention on a fast (Apple-silicon) workstation the slowest built in 39 s; none
exceeded 90 s. The omissions are a timeout-margin / hardware sensitivity of the
build-GT-locally step — the same valid programs just need longer than the old
budget on slower machines — not bad ground truth.

**Interim fix (merged).** Raised the default `execute_cq_to_step` timeout from
90 s to 300 s.

**Definitive fix.** `Vision2Code/tools/build_gt_steps.py` builds every GT STEP once
(parallel, with a serial retry for the slow tail) and packages them as a
`gt_steps.parquet`. Shipping these pre-built STEPs in the dataset removes the
local-execution step entirely, so the full 17,900-record set scores identically on
any machine — no timeout, no hardware sensitivity — and the tool's report gives an
authoritative build status for every record. Until the STEPs ship, Anthropic's
numbers are over the 17,874-record subset and are directly comparable to full-set
numbers once the slow records build.

## Per-record corrections

Each entry: date · `record_id` · what was wrong · the fix · version.

| Date | record_id | Problem | Fix | Version |
|---|---|---|---|---|
| — | — | _none reported yet_ | — | — |

Found a wrong item? See [CONTRIBUTING.md](../CONTRIBUTING.md) section C.
