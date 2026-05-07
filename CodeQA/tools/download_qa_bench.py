"""Download BenchCAD/BenchCAD code-qa-bench parquets → data/.

Run from CodeQA/:
    uv run python tools/download_qa_bench.py

Layout matches test_data/, so configs/prod.yaml works unchanged:

    data/
    ├── records.jsonl     {record_id, family, code_path, qa_pairs[]}
    └── codes/<rid>.py

Source: https://huggingface.co/datasets/BenchCAD/BenchCAD/tree/main/code-qa-bench

Expected parquet columns: record_id, family, gt_code (str), qa_pairs (str /
list of dicts with question / answer / type).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

REPO = "BenchCAD/BenchCAD"
SUBDIR = "code-qa-bench"
ROOT = Path(__file__).resolve().parents[1]  # CodeQA/
DEFAULT_OUT = ROOT / "data"


def _coerce_qa(val) -> list[dict]:
    """qa_pairs may arrive as JSON-string, list-of-dicts, or list-of-JSON-strings."""
    if isinstance(val, list):
        out = []
        for x in val:
            if isinstance(x, dict):
                out.append(x)
            elif isinstance(x, str):
                try:
                    out.append(json.loads(x))
                except Exception:
                    pass
        return out
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def main():
    ap = argparse.ArgumentParser(description="Download BenchCAD code-qa-bench → data/")
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

    rows = []
    for r in df.itertuples(index=False):
        rid = r.record_id
        code_path = f"codes/{rid}.py"
        (out / code_path).write_text(r.gt_code)
        rows.append({
            "record_id": rid,
            "family":    getattr(r, "family", "") or "",
            "code_path": code_path,
            "qa_pairs":  _coerce_qa(getattr(r, "qa_pairs", None)),
        })

    records_jsonl = out / "records.jsonl"
    with records_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records → {records_jsonl}")
    print(f"        codes/ → {out}/codes/")


if __name__ == "__main__":
    main()
