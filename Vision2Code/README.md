# Vision2Code

Image → CadQuery code. The model is shown a 2x2 composite of 4 diagonal views
of a mechanical part and must write a CadQuery program that reproduces the
geometry. Scoring by **voxel IoU** between the model's STEP output and the
ground-truth STEP (no edit baseline — generation from scratch).

← Back to [main README](../README.md)

> **All commands below assume `cd Vision2Code` first.**

## Configs

What to run lives in YAML — one per setup. Two ship out of the box:

| Config | Purpose |
|---|---|
| `configs/test.yaml` | 4-record smoke (`test_data/`), 1 model. Default. |
| `configs/prod.yaml` | Full bench (`BenchCAD/BenchCAD` config `code_gen` on HF), 7+ models. |

Each config sets:
```yaml
data_dir: test_data
out_dir:  results_test
models: [gpt-4o, claude-opus-4-7, ...]
```

Paths are **relative to `Vision2Code/`**. Copy either one to make your own
(`configs/myrun.yaml`) and pass `--config`.

## Download full-bench data (one-time, for prod)

```bash
cd Vision2Code
uv run python tools/download_codegen_bench.py     # → data/
```

## Quick smoke run

```bash
cd Vision2Code

# Run with default config (configs/test.yaml):
uv run python main.py

# Plot from whatever's in <out_dir>/results.jsonl:
uv run python main.py --plot
```

## Custom run

```bash
cd Vision2Code
uv run python main.py --config configs/prod.yaml
uv run python main.py --config configs/myrun.yaml --plot
```

Debug overrides (don't change config): `--records r1 r2` for a subset, `--limit N` to cap.

Output (per config's `out_dir`, relative to `Vision2Code/`):
```
<out_dir>/
├── results.jsonl                 (model, record_id) → iou + paths
├── plot.png                      written by --plot only
└── outputs/<safe_model>/<record_id>.{py,step,png}
```

`results.jsonl` is **overwrite-keyed** by `(model, record_id)` —
re-running a tuple replaces the old row, never duplicates.

## Agentic setting (Python sandbox)

`configs/agentic.yaml` runs Vision2Code with tools. The model gets a working
directory holding the target composite and its four views, plus a `tools.py`
whose `render()` draws a STEP with the same renderer and framing as the target.
It works through two native tools. `run_python` executes code in a Docker
container with no network and returns stdout, stderr and any images the code
wrote. `submit` ends the episode with the program to score. Scoring is identical
to single-shot (execute → STEP → voxel IoU), and an episode that never calls
`submit` scores 0.

```bash
# once, from the repo root: build the sandbox image for your architecture
docker build -f docker/sandbox.arm64.Dockerfile -t benchcad-sandbox:arm64 .   # Apple silicon
docker build -f docker/sandbox.Dockerfile       -t benchcad-sandbox:amd64 .   # x86-64

# from Vision2Code/
uv run python main.py --config configs/agentic.yaml
```

`agentic.max_rounds` (default 100) is the round budget. The config lists the
measured cost per record at each budget, which is far above single-shot. For
the paired single-shot baseline, run the same records with
`agentic.enabled: false`. The ablation switches are environment variables, all
off by default:

| Variable | Effect |
|---|---|
| `BENCH_NO_IMAGE_FEEDBACK=1` | the images the model's code writes are not attached to the next turn; everything else is byte-identical |
| `BENCH_VIEW_HINT=anchor` | the prompt states only the top-right camera direction |
| `BENCH_VIEW_HINT=perturb` | wording for targets whose other three views were re-rendered off the corner directions |
| `BENCH_SANDBOX_IMAGE` | sandbox image to use instead of the architecture default |

## Score: voxel IoU

For each `(model, record)`:

```
iou = voxel-IoU(model_step, gt_step)        # 64³ grid, normalized
```

Both STEPs are tessellated, normalized (bbox center → [0.5,0.5,0.5], longest
axis → [0,1]), voxelized at pitch 1/64. `iou = 0` when the model's code didn't
exec or geometry is degenerate. `iou = 1` when the voxelized solids match.

This raw voxel IoU is the default and matches the metric Anthropic reports.

## Score: composite (`--score composite`)

`--score composite` reports the fused **BenchCAD total** instead of raw IoU:

```
score = 0.60·IoU + 0.20·essential-op + 0.10·Feature-F1
      + 0.05·Chamfer-score + 0.05·Hausdorff-score
```

- **IoU** — the raw voxel IoU above (so `--score iou` is exactly the 0.60 term).
- **essential-op** — per-family hand-curated essential operations must all be
  present in the generated code (`scoring/canonical_ops.yaml`). Families with no
  spec drop this term and the remaining 0.80 is rescaled to `[0,1]`.
- **Feature-F1** — F1 over hole/fillet/chamfer features (B-rep for holes).
- **Chamfer / Hausdorff** — surface-distance terms, each mapped to a bounded
  `[0,1]` score.

The composite scorer (`scoring/composite.py`, `composite_metrics.py`,
`canonical_ops.{py,yaml}`) is vendored from the BenchCAD generation repo, which
is the source of truth for the weights and the essential-op spec.

## Folder structure

```
Vision2Code/
├── main.py                      CLI: --config, --plot, --records, --limit
├── configs/                     YAML configs (test.yaml, prod.yaml, ...)
├── tools/
│   └── download_codegen_bench.py   pulls HF parquet → data/
├── pipeline/
│   ├── runner.py                per-record loop + overwrite store
│   ├── agentic.py               sandbox episode loop (configs/agentic.yaml)
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
