"""Download the BenchCAD/BenchCAD QA benchmark → data/.

Run from QA/:
    uv run python tools/download_qa_bench.py
    uv run python tools/download_qa_bench.py --out test_data --limit 4

Layout matches test_data/, so configs/prod.yaml works unchanged:

    data/
    ├── records.jsonl     {record_id, family, code_path, qa_pairs[]}
    └── codes/<rid>.py

Source: https://huggingface.co/datasets/BenchCAD/BenchCAD (config `QA`,
`QA/qa_2400.parquet`). The parquet has one row per question
(columns: stem, family, gt_code, qa [JSON: question/answer/type/level], ...);
rows are grouped by `stem` into one record with its list of `qa_pairs`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "BenchCAD/BenchCAD"
PARQUET = "QA/qa_2400.parquet"
ROOT = Path(__file__).resolve().parents[1]  # QA/
DEFAULT_OUT = ROOT / "data"


def _qa_from_row(qa_val) -> dict | None:
    """One QA dict from the `qa` column (a JSON string)."""
    try:
        d = qa_val if isinstance(qa_val, dict) else json.loads(qa_val)
    except Exception:
        return None
    if "question" not in d or "answer" not in d:
        return None
    return {"question": d["question"], "answer": d["answer"], "type": d.get("type", "dim")}


def main():
    ap = argparse.ArgumentParser(description="Download BenchCAD QA bench → data/")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Extract dir (default: {DEFAULT_OUT.relative_to(ROOT)}/)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="HF download cache (default: $HF_HOME)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only write the first N records/parts (for small smoke sets)")
    args = ap.parse_args()

    print(f"hf_hub_download {REPO} :: {PARQUET} ...")
    df = pd.read_parquet(hf_hub_download(
        REPO, PARQUET, repo_type="dataset", cache_dir=args.cache_dir))
    print(f"  loaded {len(df)} QA rows over {df['stem'].nunique()} parts")

    out: Path = args.out
    (out / "codes").mkdir(parents=True, exist_ok=True)

    rows = []
    for stem, group in df.groupby("stem", sort=False):
        if args.limit is not None and len(rows) >= args.limit:
            break
        qa_pairs = [qa for qa in (_qa_from_row(v) for v in group["qa"]) if qa]
        if not qa_pairs:
            continue
        code_path = f"codes/{stem}.py"
        (out / code_path).write_text(group.iloc[0]["gt_code"])
        rows.append({
            "record_id": stem,
            "family": group.iloc[0].get("family", "") or "",
            "code_path": code_path,
            "qa_pairs": qa_pairs,
        })

    records_jsonl = out / "records.jsonl"
    with records_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records ({sum(len(r['qa_pairs']) for r in rows)} QA) → {records_jsonl}")
    print(f"        codes/ → {out}/codes/")


if __name__ == "__main__":
    main()
