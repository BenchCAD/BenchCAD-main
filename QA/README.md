# QA

CadQuery part → numeric answers. The model sees the part (as **code**, as a
**rendered image**, or **both**) plus a list of numeric questions about it
(dimensions, counts, ratios). Scoring by **mean ratio accuracy** between
predicted and ground-truth numbers (`min(pred, gt) / max(pred, gt)`, symmetric).

Three input modes, set via `mode:` in the config:
- `code` — text-only, CadQuery source (default)
- `img`  — vision-only, rendered composite PNG
- `both` — multimodal, code + image

← Back to [main README](../README.md)

> **All commands below assume `cd QA` first.**

## Configs

What to run lives in YAML — one per setup. Four ship out of the box:

| Config | Mode | Purpose |
|---|---|---|
| `configs/test.yaml`     | code | 4-record smoke (`test_data/`), 1 model. Default. |
| `configs/prod.yaml`     | code | Full code bench (`code-qa-bench` on HF), 7+ models. |
| `configs/test_img.yaml` | img  | 2-stem image smoke (`test_data_img/`), 1 model. |
| `configs/prod_img.yaml` | img  | Full image bench (`QA/qa_2400` on HF, 200 stems × 12 QA). |

Each config sets:
```yaml
mode:     code            # code | img | both  (default: code)
data_dir: test_data
out_dir:  results_test
models: [gpt-4o, claude-opus-4-7, ...]
```

Paths are **relative to `QA/`**. Copy either one to make your own
(`configs/myrun.yaml`) and pass `--config`.

## Download full-bench data (one-time, for prod)

```bash
cd QA

# code bench (text-only):
uv run python tools/download_qa_bench.py             # → data/

# image bench (composite_png + 12 QA per stem):
uv run python tools/download_qa_img.py               # → data_qa_img/
uv run python tools/download_qa_img.py --limit 2 \
        --out test_data_img                          # tiny smoke fixture
```

## Quick smoke run

```bash
cd QA

# Run with default config (configs/test.yaml):
uv run python main.py

# Plot from whatever's in <out_dir>/results.jsonl:
uv run python main.py --plot
```

## Custom run

```bash
cd QA
uv run python main.py --config configs/prod.yaml
uv run python main.py --config configs/myrun.yaml --plot
```

Debug overrides (don't change config): `--records r1 r2` for a subset, `--limit N` to cap.

Output (per config's `out_dir`, relative to `QA/`):
```
<out_dir>/
├── results.jsonl                 (model, record_id) → qa_score + per_qa breakdown
├── plot.png                      written by --plot only
└── outputs/<safe_model>/<record_id>.{txt,json}
```

`results.jsonl` is **overwrite-keyed** by `(model, record_id)` —
re-running a tuple replaces the old row, never duplicates.

## Score: type-aware

Per-pair score depends on the question's `type`:

| Type                              | Score function                                |
|---|---|
| `integer` / `count` / `boolean`   | exact match — `1.0 if pred == gt else 0.0`    |
| `dim` / `ratio` / (anything else) | symmetric ratio — `min(pred, gt) / max(pred, gt)`, `0` if either non-positive |

Then `qa_score = mean(score_one for q in qa_pairs)`. Exact match is required
for boolean (0/1) answers, where ratio accuracy collapses to 0 even when
`pred == gt == 0`; it also matches natural semantics for counts and line
numbers. `qa_score = 1` when every answer matches exactly.

## Folder structure

```
QA/
├── main.py                      CLI: --config, --plot, --records, --limit
├── configs/                     YAML configs (test.yaml, prod.yaml, ...)
├── tools/
│   ├── download_qa_bench.py     code bench: HF code-qa-bench → data/
│   └── download_qa_img.py       image bench: HF QA/qa_2400.parquet → data_qa_img/
├── pipeline/
│   ├── runner.py                per-record loop + overwrite store
│   ├── prompt.py                (system, user_text, image_paths) builder
│   ├── store.py                 results.jsonl helpers
│   └── plot.py                  results.jsonl → bar plot (husl palette)
├── scoring/
│   └── qa_score.py              JSON parser + ratio accuracy
├── models/                      provider adapters (openai / anthropic / gemini / openrouter)
├── test_data/                   4 records committed for code-mode smoke runs
├── test_data_img/               tiny img-mode smoke fixture (populate via download_qa_img.py --limit)
├── data/                        full code bench (gitignored)
└── data_qa_img/                 full image bench (gitignored)
```
