"""Append-only JSONL result store.

Per-record row schema (see pipeline.runner.run_record):
    record_id, model, mode, status, iou, lat_s, err, code_path, step_path
"""

from __future__ import annotations

import json
from pathlib import Path


def append_result(jsonl_path: Path, row: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
