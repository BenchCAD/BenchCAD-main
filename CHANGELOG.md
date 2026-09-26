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
- **`gen.max_tokens` is not forwarded to xAI, deliberately.** There a cap near
  the model's working range acts as a reasoning *target* rather than a ceiling,
  so sending one makes runs slower rather than safer. The same image->CadQuery
  record used 32395 output tokens in 398 s with `max_output_tokens=16000`, and
  10089 tokens in 134 s without it. Both completed with a valid program, and the
  capped call overshot its own cap 2x: the cap only ever bounds the visible
  answer, which is a few dozen tokens. The adapter instead sends a fixed
  `max_output_tokens=512000` backstop, far above anything observed (~33k), so the
  request stays bounded without shaping the answer.
- **xAI requests are streamed, and `gen.timeout` is used exactly as configured.**
  The endpoint buffers its answer either way. A streamed request sends a
  keepalive frame every ~15 s, while a non-streamed one is silent for the whole
  generation and does not survive a long reasoning pass: in paired runs on
  identical requests, 4/4 streamed and 2/4 non-streamed calls returned. Because
  the SDK's timeout applies per read, it now bounds a stalled connection rather
  than a long generation. Earlier revisions of this adapter clamped the timeout
  up to 900 s and then 3000 s. Both values were sized from latencies that silently
  included the SDK's two retries, so they overrode the run config and made every
  dead call cost 3x the floor. `:reasoning=low` answers in tens of seconds
  instead, at a real accuracy cost.
- **Grok is sampled at temperature 0.7**, the endpoint's own default, pinned
  explicitly, rather than the 0.0 the other adapters send. Its reasoning length
  varies from run to run regardless, so 0.0 would not make a run reproducible. A
  model that rejects `temperature` is retried once without it.
- **Records can be evaluated concurrently** via an optional `concurrency:` block
  (`api_workers`, `score_workers`); the default `api_workers: 1` leaves existing
  runs sequential and unchanged. The two pools are deliberately separate: a model
  call costs a socket and blocks for minutes, while scoring spawns a ~0.5 GB
  CadQuery/OCP subprocess and finishes in seconds, so a single pool either
  starves the API or exhausts memory (64 concurrent scorers need ~25 GB). A
  thread waiting on the API holds no scoring slot, which is what lets a large API
  pool sit in front of a small scoring pool. `benchcad_core/parallel.py` also
  handles three things that are only visible under load: ground-truth composites
  are rendered up front on the main thread (VTK aborts the process if a worker
  builds a render window, and leaks a graphics context per render), the
  `results.jsonl` read-modify-write is serialised (it silently loses rows under
  threads), and a worker exception fails one record instead of the batch.
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
