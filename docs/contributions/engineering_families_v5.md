# Six engineering families: validation and scope

This source-data contribution adds bearing_plunger, cycloidal_cartridge,
dual_drive_metering_head, planar_arm_geartrain, profiled_support_plate and
tolerance_stack. Each family declares 50 seeds at each of three difficulty tiers,
two representative parts with 12 numeric QA each, and six T1 edits.

## Reproduction

From a checkout containing the contributions, run `uv sync --dev`,
`uv run ruff check .` and `uv run pytest -q`. Run, for each of the six families:

```sh
uv run python tools/validate_family.py contributions/bearing_plunger --per-difficulty 3
```

`--per-difficulty 50` covers all declared seeds. `validate_task.py` accepts each
examples/<stem> directory. The per-family generation_manifest.json records all
150 generated-source SHA-256 values. No new runtime dependencies are introduced.

## Checks performed

At upstream commit 77fd16a80a4e5fc39964ee23b6cc3225beefac8f:

- All 900 declared sources were generated and executed with the pinned upstream
  export, render and normalized-geometry hash helpers. Positive-volume valid
  solids, STEP export and four-view rendering passed; all 900 hashes are distinct.
- The submission checkout passed all six official family gates at three seeds
  per difficulty (54 samples), including all 36 geometry-changing edits.
- Local ruff passed. Pytest: 134 passed, 7 skipped, 2 expected failures.
  This is a local macOS check, not a claim of hosted Linux CI completion.
- 900 code_gen asset rows passed an upstream-Features/Parquet image/code round-trip.
  QA structure, numeric scorer controls and independent reference calculations
  were checked. Model scores were never used to define ground truth.

## V5 wording regression

A prior v4 pilot used 228 single-shot requests. V5 removes 12 wording hints without
changing any answer, type, level, CAD source or edit. This v5 retest ran 70 new
requests over all seven affected 12-question banks: code input and no-CAD controls
(two repeats each), plus seed-49 parameter perturbations, using deepseek-flash and
[次]gemini-4-argon. Output cap: 16,384 tokens; no tools or automatic retries.
DeepSeek used high reasoning; the channel model used its recorded default thinking
configuration and temperature 0. These are not equal-compute comparisons.

All 12 modified target positions have a valid full-score source-conditioned answer
from both models. 23 of the 24 model/target cells were verified in grouped answers.
For the remaining cell, DeepSeek twice returned 11 rather than 12 numbers for the
easy cycloidal bank. Two separately labelled one-question diagnostics of the exact
same changed question and CAD source returned the correct answer. The original
grouped failures and zero scores are retained, not replaced by those diagnostics.
The JSON companion gives per-target evidence and strict full-context scores.

This is a targeted regression check, not a leaderboard submission or evidence that
training on these data improves a model. T1 edits were at ceiling for completed
answers in the prior pilot; T2–T5 edit discrimination is not claimed.

## Explicit review items

- Geometry comparison used all 20 public data shards at HF revision
  5919f578ab09ec283603a082fab07c7639ab56eb. Of 19,040 independent inputs, 18,877
  produced hashes using the official helper. No candidate hash matched these
  inputs or the GitHub main example. The remaining 163 inputs could not be hashed
  locally; complete accepted-corpus deduplication is not established. This is exact
  normalized-mesh hashing, not a near-duplicate or functional-equivalence proof.
- Canonical 0.55 framing touches image boundaries for 297 variants (150 tolerance
  specimens and 147 profiled plates); visual samples confirm clipping. Separate
  0.90 review previews preserve the full shape but are not substituted for official
  benchmark inputs. Framing/input-version handling needs maintainer review (#54).
- QA was tested in code mode. Source-operation questions are not claimed solvable
  from unannotated normalized images; image-only applicability needs separate
  curation. Engineering assumptions are stated in the questions.
- Family labels and difficulty tiers are proposed. Several families are simplified
  assemblies or mechanism analogues, not production or contact/stress validation.
  Please review whether taxonomy should add families or extend existing ones.

Only new contribution source, metadata and small verification documents are in
this PR. No full official dataset, API responses, credentials, private renders or
historical model outputs are committed. Source citations and applicable MIT
notices are included in each family.
