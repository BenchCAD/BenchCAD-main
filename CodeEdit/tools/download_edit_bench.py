"""Download BenchCAD/BenchCAD edit-bench parquets → data/.

Run from CodeEdit/:
    uv run python tools/download_edit_bench.py

Layout matches test_data/, so configs/prod.yaml works unchanged:

    data/
    ├── records.jsonl     {record_id, family, edit_type, instruction, iou,
    │                      *_path, pos_id, category, category_label}
    ├── codes/<rid>_{orig,gt}.py
    └── steps/<rid>_{orig,gt}.step

Source: https://huggingface.co/datasets/BenchCAD/BenchCAD/tree/main/edit-bench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

REPO = "BenchCAD/BenchCAD"
SUBDIR = "edit-bench"
ROOT = Path(__file__).resolve().parents[1]  # CodeEdit/
DEFAULT_OUT = ROOT / "data"


def main():
    ap = argparse.ArgumentParser(description="Download BenchCAD edit-bench → data/")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Extract dir (default: {DEFAULT_OUT.relative_to(ROOT)}/)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="HF download cache (default: $HF_HOME)")
    args = ap.parse_args()

    print(f"snapshot_download {REPO} :: {SUBDIR}/* ...")
    snap = Path(snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=f"{SUBDIR}/*",
        cache_dir=args.cache_dir,
    ))
    parquet_files = sorted((snap / SUBDIR).rglob("*.parquet"))
    if not parquet_files:
        sys.exit(f"no .parquet files under {snap / SUBDIR}")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    print(f"  loaded {len(df)} rows from {len(parquet_files)} parquet shard(s)")

    out: Path = args.out
    (out / "codes").mkdir(parents=True, exist_ok=True)
    (out / "steps").mkdir(parents=True, exist_ok=True)

    rows = []
    for r in df.itertuples(index=False):
        rid = r.record_id
        paths = {
            "orig_code_path": f"codes/{rid}_orig.py",
            "gt_code_path":   f"codes/{rid}_gt.py",
            "orig_step_path": f"steps/{rid}_orig.step",
            "gt_step_path":   f"steps/{rid}_gt.step",
        }
        (out / paths["orig_code_path"]).write_text(r.orig_code)
        (out / paths["gt_code_path"]).write_text(r.gt_code)
        (out / paths["orig_step_path"]).write_bytes(r.orig_step)
        (out / paths["gt_step_path"]).write_bytes(r.gt_step)
        rows.append({
            "record_id":      rid,
            "family":         r.family,
            "edit_type":      r.edit_type,
            "instruction":    r.instruction,
            "iou":            float(r.iou),
            **paths,
            "pos_id":         int(r.pos_id) if getattr(r, "pos_id", None) is not None else -1,
            "category":       getattr(r, "category", "") or "",
            "category_label": getattr(r, "category_label", "") or "",
        })

    records_jsonl = out / "records.jsonl"
    with records_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records → {records_jsonl}")
    print(f"        codes/ + steps/ → {out}/")


if __name__ == "__main__":
    main()
