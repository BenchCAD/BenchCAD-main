# CodeGen

Image → CadQuery code. The model is shown a 2x2 composite of 4 diagonal views
of a mechanical part and must write a CadQuery program that reproduces the
geometry. Scoring by **voxel IoU** between the model's STEP output and the
ground-truth STEP (no edit baseline — generation from scratch).

← Back to [main README](../README.md)

> **All commands below assume `cd CodeGen` first.**

## Configs

What to run lives in YAML — one per setup. Two ship out of the box:

| Config | Purpose |
|---|---|
| `configs/test.yaml` | 4-record smoke (`test_data/`), 1 model. Default. |
| `configs/prod.yaml` | Full bench (`BenchCAD/BenchCAD/code-gen-bench` on HF), 7+ models. |

Each config sets:
```yaml
data_dir: test_data
out_dir:  results_test
models: [gpt-4o, claude-opus-4-7, ...]
```

Paths are **relative to `CodeGen/`**. Copy either one to make your own
(`configs/myrun.yaml`) and pass `--config`.

## Download full-bench data (one-time, for prod)

```bash
cd CodeGen
uv run python tools/download_codegen_bench.py     # → data/
```

## Quick smoke run

```bash
cd CodeGen

# Run with default config (configs/test.yaml):
uv run python main.py

# Plot from whatever's in <out_dir>/results.jsonl:
uv run python main.py --plot
```

## Custom run

```bash
cd CodeGen
uv run python main.py --config configs/prod.yaml
uv run python main.py --config configs/myrun.yaml --plot
```

Debug overrides (don't change config): `--records r1 r2` for a subset, `--limit N` to cap.

Output (per config's `out_dir`, relative to `CodeGen/`):
```
<out_dir>/
├── results.jsonl                 (model, record_id) → iou + paths
├── plot.png                      written by --plot only
└── outputs/<safe_model>/<record_id>.{py,step,png}
```

`results.jsonl` is **overwrite-keyed** by `(model, record_id)` —
re-running a tuple replaces the old row, never duplicates.

## Render-and-verify (tool-assisted runs)

By default each record is **single-shot** — the model writes CadQuery once from the target views.
To reproduce the *with Python tools* setting (the model renders its own candidate and visually
verifies it before submitting, which lifts Vision2Code scores substantially), add `render_verify`
to a config:

```yaml
render_verify:
  rounds: 3        # total attempts incl. the first single-shot (1 = plain single-shot)
```

Round 1 is the normal single-shot. In each later round the current candidate is rendered in the
**same 4 views** (the same `scoring/views` renderer used everywhere) and shown to the model next to
the target; the model returns a corrected program, kept only if it still executes to a valid STEP.
Scoring (voxel IoU vs GT) is unchanged, so the number stays directly comparable to a single-shot
run. Each `results.jsonl` row gains a `rounds` field (attempts actually used).

## Score: voxel IoU

For each `(model, record)`:

```
iou = voxel-IoU(model_step, gt_step)        # 64³ grid, normalized
```

Both STEPs are tessellated, normalized (bbox center → [0.5,0.5,0.5], longest
axis → [0,1]), voxelized at pitch 1/64. `iou = 0` when the model's code didn't
exec or geometry is degenerate. `iou = 1` when the voxelized solids match.

## Folder structure

```
CodeGen/
├── main.py                      CLI: --config, --plot, --records, --limit
├── configs/                     YAML configs (test.yaml, prod.yaml, ...)
├── tools/
│   └── download_codegen_bench.py   pulls HF parquet → data/
├── pipeline/
│   ├── runner.py                per-record loop + overwrite store
│   ├── prompt.py                (system, user_text, image_paths) builder
│   ├── store.py                 results.jsonl helpers
│   └── plot.py                  results.jsonl → bar plot (husl palette)
├── scoring/
│   ├── exec_cq.py               extract code → run subprocess → STEP file
│   ├── iou.py                   voxel IoU (64³ grid)
│   └── views.py                 STEP → 4-view composite PNG (Tiffany blue)
├── models/                      provider adapters (openai / anthropic / gemini / openrouter)
├── test_data/                   4 records committed for smoke runs
└── data/                        full bench, populated by tools/download_codegen_bench.py (gitignored)
```
