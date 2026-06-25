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
| **CodeGen parts** | a parametric generator that yields parts across **all three difficulties** `easy / medium / hard`, **≥ ~50 per difficulty** | every family covers all 3; median ≈ 55–58 parts per difficulty (~169 total) |
| `meta` per part | `family`, `variant` (default `"standard"`), `difficulty`, `base_plane` ∈ `{XY,XZ,YZ}`, `standard` (ISO/DIN/ASME or `null`) | exact code_gen schema |
| **CodeQA** | QA on **≥ 2 representative parts**, **exactly 12 questions each**; mix the types ~ `integer` (incl. count/yes-no) **≈70%**, `dim` **≈15%**, `ratio` **≈15%**; spread difficulty `level` across **L1–L6** (weight L3–L4) | 200 annotated parts × 12 QA; types 73/14/13%; levels L1–L6 |
| answer types | numeric only — `integer` / `count` / `dim` / `ratio` / `boolean` | — |
| **CodeEdit** | **2–3 edits per difficulty** (`easy / medium / hard`, ≈6–9 total); each edit changes the geometry (resulting IoU < 1) and carries an `instruction` + `edit_type` | 748 edits, ~6–7 per family, all non-trivial (median IoU 0.765) |

### Layout

```
contributions/<family>/
├── family.json          # {"family","standard","base_plane","description","source","contributor"}
├── generator.py         # parametric: `build(difficulty, seed) -> result` (CadQuery solid)
├── qa/<stem>.json        # ≥2 files, 12 questions each: [{"question","answer","type","level"}]
└── edits/<name>.json     # ≥2-3 per difficulty: {"difficulty","edit_type","instruction",
                          #                        "orig_code","gt_code"}
```

(A single self-contained part — like `contributions/example_plate/` — is also
accepted for smoke-testing the gate, but a *family* must meet the table above.)

**Automated gate** (CI runs `tools/validate_task.py` per generated part; no API keys):
1. each part executes and exports a valid STEP solid;
2. it renders to the 4-view composite without error;
3. `meta` is well-formed (required keys, valid `difficulty` / `base_plane` / `type`);
4. its geometry hash does not duplicate an existing part;
5. family-level coverage (difficulty counts, ≥2 QA×12, 2–3 edits/difficulty) is met.

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
   `submission.jsonl` + your run config. We re-grade and update the board. For the
   private held-out split, you submit predictions and we grade against the hidden
   answers.

---

## C. Contribute code or fix a wrong item (errata)

Normal GitHub flow:

- One PR = one purpose; keep the diff minimal and match the surrounding style.
- Every behavior change ships a test that fails before and passes after
  (see `CodeQA/tests/`, `CodeGen/tests/`).
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
