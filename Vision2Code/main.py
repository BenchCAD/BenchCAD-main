"""Vision2Code runner — config-driven (image → CadQuery code).

Two independent operations:

  (1) RUN: read config → for each (model × record) build prompt (4-view image →
      CadQuery code) → call model → exec CadQuery → score (voxel IoU vs GT) →
      write to <out_dir>/results.jsonl (overwrite by (model, record_id)).

  (2) PLOT: read <out_dir>/results.jsonl → husl bar plot (mean IoU per model).

Examples (run from Vision2Code/)
----------------------------
    # Smoke run (default config: configs/test.yaml):
    uv run python main.py

    # Full bench:
    uv run python main.py --config configs/prod.yaml

    # Plot whatever's in results.jsonl for that config:
    uv run python main.py --plot

    # Debug overrides:
    uv run python main.py --records r1 r2
    uv run python main.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.runner import run_record  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "test.yaml"
REQUIRED_FIELDS = ("data_dir", "out_dir", "models")


def parse_args():
    p = argparse.ArgumentParser(description="Vision2Code runner (config-driven)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"YAML config (default: {DEFAULT_CONFIG.relative_to(ROOT)})")
    p.add_argument("--plot", action="store_true",
                   help="Render bar plot from <out_dir>/results.jsonl. No model calls.")
    p.add_argument("--records", nargs="*", default=None,
                   help="Debug override: only run these record_ids.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap to N records (first N, or a random N with --seed).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random-sample --limit records with this seed (reproducible).")
    p.add_argument("--score", choices=["iou", "composite"], default="iou",
                   help="iou = raw voxel IoU (default, Anthropic's method); "
                        "composite = fused BenchCAD total (IoU+essential+F1+CD+HD).")
    return p.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    missing = [k for k in REQUIRED_FIELDS if k not in cfg]
    if missing:
        sys.exit(f"config {path} missing fields: {missing}")
    return cfg


def load_records(data_dir: Path) -> list[dict]:
    jsonl = data_dir / "records.jsonl"
    if not jsonl.exists():
        sys.exit(f"records.jsonl not found at {jsonl}")
    return [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]


def do_plot(out_dir: Path) -> None:
    from pipeline.plot import make_bar
    jsonl = out_dir / "results.jsonl"
    if not jsonl.exists():
        sys.exit(f"results.jsonl not found at {jsonl}")
    out_png = out_dir / "plot.png"
    make_bar(jsonl, out_png)
    print(f"plot → {out_png}")


def _print_results_summary(out_dir: Path) -> None:
    jsonl = out_dir / "results.jsonl"
    if not jsonl.exists():
        return
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    if not rows:
        return
    bucket: dict[str, list[float]] = {}
    for r in rows:
        bucket.setdefault(r["model"], []).append(float(r.get("score", r["iou"])))
    stype = rows[-1].get("score_type", "iou")
    print(f"\nresults  → {jsonl}  ({len(rows)} rows)")
    print(f"summary  → mean {stype} per model:")
    width = max(len(m) for m in bucket) + 2
    for model in sorted(bucket):
        vals = bucket[model]
        print(f"    {model:<{width}}n={len(vals):<3} mean_{stype}={sum(vals)/len(vals):.3f}")


def do_run(cfg: dict, args) -> None:
    data_dir = Path(cfg["data_dir"])
    out_dir  = Path(cfg["out_dir"])
    models   = list(cfg["models"])

    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(data_dir)
    if args.records:
        wanted = set(args.records)
        records = [r for r in records if r["record_id"] in wanted]
    if args.limit:
        if args.seed is not None:
            import random
            records = random.Random(args.seed).sample(records, min(args.limit, len(records)))
        else:
            records = records[: args.limit]

    print(f"config: {args.config}")
    print(f"data:   {data_dir}")
    print(f"out:    {out_dir}")
    print(f"runs:   {len(records)} record(s) × {len(models)} model(s)")
    for model in models:
        for i, rec in enumerate(records, 1):
            print(f"  [{model}] {i}/{len(records)} {rec['record_id']}", end=" ... ", flush=True)
            row = run_record(record=rec, data_dir=data_dir, results_root=out_dir,
                             model=model, score=args.score)
            print(f"{row['status']:10s} {row['score_type']}={row['score']:.3f}  ({row['lat_s']:.1f}s)")
    _print_results_summary(out_dir)


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    out_dir = Path(cfg["out_dir"])

    if args.plot:
        do_plot(out_dir)
        return
    do_run(cfg, args)


if __name__ == "__main__":
    main()
