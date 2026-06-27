"""Download BenchCAD/BenchCAD QA/qa_2400.parquet → data_qa_img/.

Run from QA/:
    uv run python tools/download_qa_img.py
    uv run python tools/download_qa_img.py --out test_data_img --limit 2

Layout:
    <out>/
    ├── records.jsonl     {record_id, family, code_path, image_path, qa_pairs[]}
    ├── codes/<rid>.py
    └── images/<rid>.png

Source: https://huggingface.co/datasets/BenchCAD/BenchCAD/blob/main/QA/qa_2400.parquet

Parquet schema (one row per QA, 200 stems × 12 QAs = 2400 rows):
    stem, family, gt_code (str), composite_png ({bytes, path}), standard,
    qa (JSON: {question, answer, type, source, level}), qa_level, qa_id

Rows are grouped by `stem` → one record per stem with 12 qa_pairs.
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
DEFAULT_OUT = ROOT / "data_qa_img"


def _qa_from_row(qa_str: str) -> dict | None:
    try:
        d = json.loads(qa_str)
    except Exception:
        return None
    if not isinstance(d, dict) or "question" not in d or "answer" not in d:
        return None
    return {
        "question": d["question"],
        "answer":   d["answer"],
        "type":     d.get("type", "dim"),
        "level":    d.get("level"),
        "source":   d.get("source"),
    }


def main():
    ap = argparse.ArgumentParser(description="Download BenchCAD QA/qa_2400 → data_qa_img/")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Extract dir (default: {DEFAULT_OUT.relative_to(ROOT)}/)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="HF download cache (default: $HF_HOME)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of stems (records) for smoke runs.")
    args = ap.parse_args()

    print(f"hf_hub_download {REPO} :: {PARQUET} ...")
    p = hf_hub_download(
        repo_id=REPO, filename=PARQUET, repo_type="dataset",
        cache_dir=args.cache_dir,
    )
    df = pd.read_parquet(p)
    print(f"  loaded {len(df)} rows, {df['stem'].nunique()} stems")

    out: Path = args.out
    (out / "codes").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    stems = list(dict.fromkeys(df["stem"].tolist()))  # preserve first-seen order
    if args.limit:
        stems = stems[: args.limit]

    rows = []
    for rid in stems:
        sub = df[df["stem"] == rid]
        first = sub.iloc[0]

        code_path = f"codes/{rid}.py"
        (out / code_path).write_text(first["gt_code"])

        png = first["composite_png"]
        png_bytes = png["bytes"] if isinstance(png, dict) else None
        image_path = None
        if png_bytes:
            image_path = f"images/{rid}.png"
            (out / image_path).write_bytes(png_bytes)

        qa_pairs = [q for q in (_qa_from_row(s) for s in sub["qa"]) if q]

        rows.append({
            "record_id":  rid,
            "family":     first.get("family", "") or "",
            "code_path":  code_path,
            "image_path": image_path,
            "qa_pairs":   qa_pairs,
        })

    records_jsonl = out / "records.jsonl"
    with records_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records → {records_jsonl}")
    print(f"        codes/  → {out}/codes/")
    print(f"        images/ → {out}/images/")


if __name__ == "__main__":
    main()
