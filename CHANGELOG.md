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
- **xAI (Grok) models can be evaluated directly**, via `grok-*` or `xai/<slug>`
  model ids and an `XAI_API_KEY` (or `GROK_API_KEY`) — previously they were only
  reachable through `openrouter/x-ai/*`. Calls go to xAI's own Responses API
  (`https://api.x.ai/v1`), which takes the same request shape as OpenAI's, so
  images and the `:reasoning=` suffix work as they do elsewhere; the effort
  ladder is xAI's (`none|low|medium|high|xhigh`) and is validated before the call
  rather than 400-ing mid-run. `grok-*` covers the published line; the `xai/`
  form is the escape hatch for models xAI serves under some other name, and
  passes the slug through verbatim. See `benchcad_core/models/xai_adapter.py`.
- **`gen.max_tokens` is not sent to xAI, deliberately.** There it behaves as a
  reasoning *target* rather than a ceiling, so sending one makes runs slower
  rather than safer: the same image->CadQuery record used 32395 output tokens in
  398 s with `max_output_tokens=16000`, and 10089 tokens in 134 s with the
  parameter absent. Both completed with a valid program, and the capped call
  overshot its own cap 2x — it only ever bounds the visible answer, which is a
  few dozen tokens. Raising it to "remove the limit" is the worst case.
  The per-call timeout is the only effective bound, and the
  adapter floors it to 900 s — a normal high-effort call on a hard task measured
  4–5 minutes, but an identical re-run can take far longer, so the tail is
  stochastic rather than a property of the prompt. The openai SDK's
  `max_retries=2` is left alone for that same reason (a re-run may well succeed,
  and any concurrent run will meet 429s), which is why the floor is 900 s rather
  than an hour — a hopeless call costs up to 3x it. `:reasoning=low` answers in
  tens of seconds instead, at a real accuracy cost — in a 4-record Vision2Code
  smoke it lost most of the score on one part by choosing the wrong base plane,
  which voxel IoU punishes because it does not normalize rotation.
- Contribution infrastructure: `CONTRIBUTING.md`, `tools/regrade.py` (re-grade
  submitted predictions), errata process.

## [0.1.0] — 2026-06
- Initial release: `code_gen` (17,900 samples / 106 part families),
  `QA` (2,400 numeric questions / 200 parts), `edit-bench` (748 instruction-guided edit pairs).
