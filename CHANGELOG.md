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
- **Per-record token usage + cost are recorded** in `results.jsonl`
  (`prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens`,
  `cost_usd`; wall-time `lat_s` was already there). `call_model` now returns a
  `(text, usage)` `Completion`; per-call cost comes from an editable
  `benchcad_core/models/pricing.yaml` (models not listed → `cost_usd: null`).
- **Model calls + CadQuery execution are configurable per run** via an optional
  `gen:` block in each task config (`max_tokens`, `timeout`, `exec_timeout`).
  Defaults are unchanged except the per-call timeout, raised 120 s → 600 s (and
  `prod` configs set 3600 s so long reasoning passes aren't cut off); OpenAI
  reasoning models floor `max_tokens` to 32000 so reasoning doesn't starve the
  answer. See `benchcad_core/run_config.py`.
- **OpenRouter models can now request reasoning tokens** via a `:reasoning=`
  model-id suffix (`high|medium|low` effort, an integer reasoning-token budget, or
  `off`), wired to OpenRouter's unified `reasoning` field. The suffix was
  previously honored for OpenAI / Anthropic models but silently ignored for
  `openrouter/*`. Reported scores for OpenRouter reasoning models can change.
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

### Environments (Prime Intellect hub)
- **`benchcad-vision2code` 0.1.0 could not be installed from the hub at all**;
  fixed in 0.1.2. The published wheel declared `nlopt==2.10.0` and
  `numpy==1.26.4`, which cannot resolve together — nlopt 2.8.0+ declares
  `numpy>=2,<3`. Locally this was reconciled by `[tool.uv]
  override-dependencies`, which is workspace config and is never written into
  wheel metadata, so `prime env install benchcad/benchcad-vision2code` failed for
  everyone while every local check passed. nlopt is now pinned to 2.7.1 (the last
  release declaring `numpy>=1.14`) and the package carries no uv overrides.
  Scoring is unaffected — nlopt backs cadquery's sketch constraint solver, not
  execution or voxel IoU. (reported in #48)
- **`requires-python` narrowed to `>=3.11,<3.13`.** It claimed `<3.14`, so
  installers picked Python 3.13, for which the pinned numpy 1.26.4 has no wheel.
- **`exec_timeout` is now actually available on the hub.** It was added in 0.1.1
  and documented in the environment README, but 0.1.1 was never pushed — hub
  users got 0.1.0, where the argument was silently swallowed by `**kwargs` and
  the execution timeout stayed at its default.
- CI now builds each `environments/` package and installs it into a bare venv
  from its own metadata, then smoke-tests the CAD stack, so a package that only
  resolves inside this repo fails the build (`.github/workflows/ci.yml`,
  `tests/test_env_packages.py`).

## [0.1.0] — 2026-06
- Initial release: `code_gen` (17,900 samples / 106 part families),
  `QA` (2,400 numeric questions / 200 parts), `edit-bench` (748 instruction-guided edit pairs).
