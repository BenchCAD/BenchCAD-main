# CodeEdit

Text instruction → modify an existing CadQuery program. Scoring by **normalized
IoU** (improvement over baseline) between the model's STEP output and the
ground-truth STEP.

← Back to [main README](../README.md)

> **All commands below assume `cd CodeEdit` first.**

## Configs

What to run lives in YAML — one per setup. Two ship out of the box:

| Config | Purpose |
|---|---|
| `configs/test.yaml` | 4-record smoke (`test_data/`), 1 model. Default. |
| `configs/prod.yaml` | Full bench (`BenchCAD/BenchCAD/edit-bench` on HF), 7+ models. |

Each config sets:
```yaml
data_dir: test_data
out_dir:  results_test
models: [gpt-4o, claude-opus-4-7, ...]
modes:  [instruction]
```

Paths are **relative to `CodeEdit/`**. Copy either one to make your own
(`configs/myrun.yaml`) and pass `--config`.

## Download full-bench data (one-time, for prod)

```bash
cd CodeEdit
uv run python tools/download_edit_bench.py     # → data/  (~339 MB, 748 records)
```

## Quick smoke run

```bash
cd CodeEdit

# Run with default config (configs/test.yaml):
uv run python main.py

# Plot from whatever's in <out_dir>/results.jsonl:
uv run python main.py --plot
```

## Custom run

```bash
cd CodeEdit
uv run python main.py --config configs/prod.yaml
uv run python main.py --config configs/myrun.yaml --plot
```

Debug overrides (don't change config): `--records r1 r2` for a subset, `--limit N` to cap.

Output (per config's `out_dir`, relative to `CodeEdit/`):
```
<out_dir>/
├── results.jsonl                 (mode, model, record_id) → norm_iou + paths
├── plot.png                      written by --plot only
└── outputs/<mode>/<safe_model>/<record_id>.{py,step,png}
```

`results.jsonl` is **overwrite-keyed** by `(mode, model, record_id)` —
re-running a tuple replaces the old row, never duplicates.

## Score: norm_iou

For each `(model, record)`:

```
model_iou    = voxel-IoU(model_step, gt_step)        # 64³ grid, normalized
baseline_iou = record["iou"]                          # = IoU(orig_step, gt_step)
norm_iou     = clip( (model_iou - baseline_iou) / (1 - baseline_iou), 0, 1 )
```

`norm_iou = 0` when the model didn't improve over baseline (no-op or
exec_fail). `norm_iou = 1` when the model produced a perfect match.

## Folder structure

```
CodeEdit/
├── main.py                      CLI: --config, --plot, --records, --limit
├── configs/                     YAML configs (test.yaml, prod.yaml, ...)
├── tools/
│   └── download_edit_bench.py   pulls HF parquet → data/ (matches test_data shape)
├── pipeline/
│   ├── modes.py                 (system, user_text, image_paths) per mode
│   ├── runner.py                per-record loop + overwrite store
│   ├── store.py                 results.jsonl helpers
│   ├── plot.py                  results.jsonl → bar plot (husl palette)
│   └── prompts/                 reference prompt templates
├── scoring/
│   ├── exec_cq.py               extract code → run subprocess → STEP file
│   ├── iou.py                   voxel IoU + norm_iou
│   └── views.py                 STEP → 4-view composite PNG (Tiffany blue)
├── models/                      provider adapters (openai / anthropic / gemini / openrouter)
├── test_data/                   4 records committed for smoke runs
└── data/                        full bench, populated by tools/download_edit_bench.py (gitignored)
```
