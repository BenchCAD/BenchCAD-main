"""Download BenchCAD/BenchCAD code-gen-bench parquets → data/.

Run from Vision2Code/:
    uv run python tools/download_codegen_bench.py

Layout matches test_data/, so configs/prod.yaml works unchanged:

    data/
    ├── records.jsonl     {record_id, family, code_path, step_path}
    ├── codes/<rid>.py
    └── steps/<rid>.step

Source: https://huggingface.co/datasets/BenchCAD/BenchCAD/tree/main/code_gen

Parquet columns (current upstream schema): stem, family, code, view_*_png,
composite_png. There is no `gt_step` column — STEP files are produced locally
by executing each row's `code` through the same cadquery subprocess used at
scoring time (scoring.exec_cq.execute_cq_to_step). Records whose GT code
fails to execute are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for benchcad_core
from benchcad_core.scoring.exec_cq import execute_cq_to_step  # noqa: E402

REPO = "BenchCAD/BenchCAD"
SUBDIR = "code_gen/data"
ROOT = Path(__file__).resolve().parents[1]  # Vision2Code/
DEFAULT_OUT = ROOT / "data"


def main():
    ap = argparse.ArgumentParser(description="Download BenchCAD code-gen-bench → data/")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Extract dir (default: {DEFAULT_OUT.relative_to(ROOT)}/)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="HF download cache (default: $HF_HOME)")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="Only download the first N parquet shards (default: all).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only materialize the first N records (default: all).")
    args = ap.parse_args()

    if args.max_shards is not None:
        patterns = [f"{SUBDIR}/code_gen-{i:05d}-of-*.parquet" for i in range(args.max_shards)]
    else:
        patterns = f"{SUBDIR}/*.parquet"
    print(f"snapshot_download {REPO} :: {patterns} ...")
    snap = Path(snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=patterns,
        cache_dir=args.cache_dir,
    ))
    parquet_files = sorted((snap / SUBDIR).rglob("*.parquet"))
    if not parquet_files:
        sys.exit(f"no .parquet files under {snap / SUBDIR}")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    print(f"  loaded {len(df)} rows from {len(parquet_files)} parquet shard(s)")
    if args.limit is not None:
        df = df.head(args.limit)
        print(f"  --limit {args.limit}: processing first {len(df)} rows")

    out: Path = args.out
    (out / "codes").mkdir(parents=True, exist_ok=True)
    (out / "steps").mkdir(parents=True, exist_ok=True)

    rows = []
    skipped: list[tuple[str, str]] = []
    for r in df.itertuples(index=False):
        rid = r.stem
        code_path = f"codes/{rid}.py"
        step_path = f"steps/{rid}.step"
        (out / code_path).write_text(r.code)
        try:
            execute_cq_to_step(r.code, out / step_path)
        except Exception as e:
            skipped.append((rid, f"{type(e).__name__}: {e}"))
            print(f"  ! skip {rid}: GT code failed to exec — {type(e).__name__}")
            continue
        rows.append({
            "record_id": rid,
            "family":    getattr(r, "family", "") or "",
            "code_path": code_path,
            "step_path": step_path,
        })

    records_jsonl = out / "records.jsonl"
    with records_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records → {records_jsonl}")
    print(f"        codes/ + steps/ → {out}/")
    if skipped:
        print(f"        skipped {len(skipped)} record(s) whose GT code failed to exec")


if __name__ == "__main__":
    main()
