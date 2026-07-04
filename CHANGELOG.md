# Changelog

All notable changes to the BenchCAD dataset and evaluation harness. The dataset
is versioned with Hugging Face revision tags on `BenchCAD/BenchCAD`; scoring and
harness changes that can move reported numbers are called out explicitly.

## [Unreleased]

### Scoring — Vision2Code
- **Corrected swapped camera positions in the system prompt.** The four
  diagonal-view labels named the *antipodal* octant (top-left ↔ top-right, and
  bottom-left ↔ bottom-right of the 2×2 composite). A model that followed the
  stated positions reconstructed the part rotated 180°, which voxel IoU
  penalizes — it normalizes center and scale, not rotation. The labels now match
  the renderer (`Vision2Code/pipeline/prompt.py`, merged in #1). Reported scores
  improve, most visibly in agentic / tool-use settings.
- **Grading accepts a raw shape, not only a `Workplane`.** The prompt asks for
  "the final solid in `result`"; `result` may be a raw CadQuery `Shape`
  (`Solid` / `Compound` / …) or a `Workplane`. Export now calls `.val()` only
  when it exists, so raw-shape outputs are scored instead of failing
  (`benchcad_core/scoring/exec_cq.py`). Aggregate effect on scores is negligible.

### Scoring — QA
- **Negative `dim` answers are now scoreable.** `qa_score_single` returned 0 for
  any `dim`/`ratio` pair with a non-positive value, so the 18 Code QA rows whose
  gold is a negative signed extrude/cutBlind depth sum scored 0 even when exactly
  right — capping Code QA at ~0.9925, for all models equally. Now uses a
  sign-checked magnitude ratio. See `docs/ERRATA.md`. (reported in #33)

### Harness
- **GT STEP build timeout raised 90 s → 300 s**, plus a new
  `Vision2Code/tools/build_gt_steps.py` to pre-build every GT STEP once. GT STEPs
  are otherwise produced locally by executing each GT program; the slowest tail
  (led by the `double_simplex_sprocket` family) exceeded the old 90 s budget on
  slower hardware / under parallel contention, so independent runs scored 17,874
  of 17,900 records. See `docs/ERRATA.md`. Shipping the tool's pre-built STEPs
  removes the local-execution dependency entirely and makes the full
  17,900-record set score identically on any machine.
- Contribution infrastructure: `CONTRIBUTING.md`, `tools/regrade.py` (re-grade
  submitted predictions), errata process.

## [0.1.0] — 2026-06
- Initial release: `code_gen` (17,900 samples / 106 part families),
  `QA` (2,400 numeric questions / 200 parts), `edit-bench` (748 held-out edit set).
