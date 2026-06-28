# Contributing to BenchCAD

Thanks for helping improve BenchCAD. There are three kinds of contribution, each
with its own intake and quality gate. Pick the one that matches what you have.

> **Leaderboard numbers are re-graded, never self-reported.** New parts go
> through an automated executable check plus human review. This is what keeps the
> benchmark trustworthy now that it is used to evaluate frontier models.

---

## A. Contribute a new part family (data)

BenchCAD is organized by **part family** (a mechanical archetype — `hex_bolt`,
`washer`, `bevel_gear`, …), not by single parts. The live dataset has 106
families, each contributing parts to three benchmarks; a new family must follow
the same conventions so it slots in alongside the existing ones.

You submit **raw source on GitHub** — you never push to the HuggingFace dataset.
HF is the published artifact; maintainers regenerate it from accepted families
(see *How data gets accepted* below).

### Per-family requirements (matched to the live dataset)

| What | Requirement | Why (live-dataset reference) |
|---|---|---|
| **Vision2Code parts** | a parametric generator `build(difficulty, seed) -> CadQuery code` that yields parts across **all three difficulties** `easy / medium / hard`, **≥ ~50 per difficulty** | every family covers all 3; median ≈ 55–58 parts per difficulty (~169 total) |
| `meta` per part | `family`, `variant` (default `"standard"`), `difficulty`, `base_plane` ∈ `{XY,XZ,YZ}`, `standard` (ISO/DIN/ASME or `null`) | exact code_gen schema |
| **QA** | QA on **≥ 2 representative parts**, **exactly 12 questions each**; mix the types ~ `integer` (incl. count/yes-no) **≈70%**, `dim` **≈15%**, `ratio` **≈15%**; spread difficulty `level` across **L1–L6** (weight L3–L4) | 200 annotated parts × 12 QA; types 73/14/13%; levels L1–L6 |
| answer types | numeric only — `integer` / `count` / `dim` / `ratio` / `boolean` | — |
| **CodeEdit** | each edit takes a **prototype part** (a generated Vision2Code part, which carries its `easy/medium/hard` difficulty) and changes it. Cover **all three prototype difficulties, ≈2–3 edits each** (≈6–9 total). Every edit must change the geometry (resulting IoU < 1). | 748 edits, ~6–7 per family, all non-trivial (median IoU 0.765) |
| edit `category` | tag each edit with its **edit type** `T1`–`T5`: `T1` literal_replace · `T2` chain_transform · `T3` relative_compute · `T4` feature_edit · `T5` geometry_rebuild (this is the *kind* of edit, independent of the prototype difficulty) | edit-bench category taxonomy |

### Layout

```
contributions/<family>/
├── family.json          # {"family","standard","base_plane","description","source","contributor"}
├── generator.py         # defines build(difficulty, seed) -> str  (a CadQuery program)
├── qa/<stem>.json        # ≥2 files, 12 questions each: [{"question","answer","type","level"}]
└── edits/<name>.json     # cover all 3 prototype difficulties, ~2-3 each:
                          #   {"prototype_difficulty":"easy|medium|hard",
                          #    "category":"T1..T5", "edit_type":"dim|add_hole|rotate|...",
                          #    "instruction":"...", "orig_code":"...", "gt_code":"..."}
```

(A single self-contained part — like `contributions/example_plate/` — is also
accepted for smoke-testing the gate, but a *family* must meet the table above.)

**Validation gate** — run `tools/validate_task.py` per part and
`tools/validate_family.py` for the whole family before opening the PR (no API
keys needed). They check:
1. each part executes and exports a valid STEP solid;
2. it renders to the 4-view composite without error;
3. `meta` is well-formed (required keys, valid `difficulty` / `base_plane` / `type`);
4. its geometry hash does not duplicate an existing part;
5. family-level coverage (difficulty counts, ≥2 QA×12, 2–3 edits/difficulty) is met.

(CI on every PR runs `ruff` + the offline scoring tests; the validation gate
above is run by you and re-checked by a maintainer on review.)

**Human gate:** a maintainer reviews that `family` / `standard` labels are
engineering-correct, and that questions are unambiguous and answerable from the
part (machines can't judge this).

### How data gets accepted (maintainer flow)
1. PR merges into `contributions/` after both gates pass.
2. A maintainer runs `tools/ingest_to_hf.py`, which renders the parts, packs them
   into the dataset schema, appends to the HF shards
   (`code_gen/`, `QA/`, `edit-bench/`), **bumps the HF revision tag** (e.g.
   `v1.0` → `v1.1`), updates the dataset card and `CHANGELOG.md`, and credits
   contributors. Contributors never need HF write access.

---

## B. Contribute a model result (leaderboard)

Do **not** send a score. Send the model's **raw outputs**; we re-grade them with
the official scorer.

1. Produce `submission.jsonl`, one object per record:
   ```
   {"record_id": "...", "prediction": "<raw model output text>"}
   ```
2. Re-grade locally to sanity-check (we run the same command on our side):
   ```bash
   uv run python tools/regrade.py --task codeqa \
       --gt path/to/gt_records.jsonl --pred submission.jsonl --model "<model name>"
   ```
   `--task` is one of `codeqa` / `codegen` / `codeedit`. `codegen`/`codeedit`
   re-execute the predicted CadQuery and score by voxel IoU; `codeqa` scores by
   symmetric ratio accuracy. No judge model is involved.
3. Open an issue using the **Model result submission** template and attach
   `submission.jsonl` + your run config. We re-grade, add a row to
   [`leaderboard.json`](leaderboard.json), and run
   `uv run python tools/build_leaderboard.py` to refresh
   [`LEADERBOARD.md`](LEADERBOARD.md). For the private held-out split, you submit
   predictions and we grade against the hidden answers.

`leaderboard.json` is the single source of truth for the board (same schema the
website renders); every number on it is one we reproduced, never self-reported.

---

## C. Contribute code or fix a wrong item (errata)

Normal GitHub flow:

- One PR = one purpose; keep the diff minimal and match the surrounding style.
- Every behavior change ships a test that fails before and passes after
  (see `QA/tests/`, `Vision2Code/tests/`).
- Run the checks locally:
  ```bash
  uv run ruff check .
  uv run pytest -q
  ```
- **Found a wrong ground-truth** (a part that doesn't match its claimed standard,
  or a bad answer)? Fix it, then record it in `docs/ERRATA.md` and `CHANGELOG.md`
  and bump the dataset version. We publish errata openly rather than freezing
  flawed items.

---

## Dev setup

```bash
uv sync                 # Python 3.11
cp .env.example .env    # LLM API keys only — benchmark data is public
```

Scoring is execution-grounded and deterministic (voxel IoU on executed STEP
solids; symmetric ratio accuracy on numeric answers) — there is no LLM judge, so
re-grading any submission reproduces the same number.
