# Contributing to BenchCAD

Thanks for helping improve BenchCAD. There are three kinds of contribution, each
with its own intake and quality gate. Pick the one that matches what you have.

> **Leaderboard numbers are re-graded, never self-reported.** New parts go
> through an automated executable check plus human review. This is what keeps the
> benchmark trustworthy now that it is used to evaluate frontier models.

---

## A. Contribute a new part / QA item (data)

You submit **raw source on GitHub** — you do **not** push to the HuggingFace
dataset. HF is the published artifact; maintainers regenerate it from accepted
contributions (see *How data gets accepted* below).

Open a PR adding one directory under `contributions/`:

```
contributions/<your_part_name>/
├── part.py        # a CadQuery program that binds the final solid to `result`
├── meta.json      # {"family": "...", "standard": "ISO/DIN/ASME ... or null",
│                  #  "difficulty": "easy|medium|hard", "source": "...", "contributor": "..."}
└── qa.json        # [{"question": "...", "answer": 12, "type": "integer|count|dim|ratio|boolean"}]
                   #   (omit for a pure CodeGen/CodeEdit part)
```

**Automated gate** (CI runs `tools/validate_task.py`, no API keys needed):
1. `part.py` must execute and export a valid STEP solid.
2. It must render to the 4-view composite without error.
3. Every numeric `answer` is **re-derived from the geometry** and must match what
   you wrote (dimensions/counts/ratios are computed from the B-rep, not trusted).
4. The geometry hash must not duplicate an existing part.

**Human gate:** a maintainer reviews that the `family` / `standard` labels are
engineering-correct (machines can't judge this).

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
