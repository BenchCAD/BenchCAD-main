#!/usr/bin/env python
"""Re-grade submitted model predictions for a BenchCAD task.

Leaderboard numbers are never self-reported. A contributor submits *raw model
outputs* (not scores); this script re-runs the official scorer on them so every
number on the board is reproduced by us, not trusted as reported.

Usage
-----
    uv run python tools/regrade.py --task codeqa \
        --gt   path/to/gt_records.jsonl \
        --pred path/to/submission.jsonl \
        --model "claude-opus-4-8" \
        --out  leaderboard_entry.json

Formats (one JSON object per line)
----------------------------------
    gt   (codeqa):   {"record_id": "...", "qa_pairs": [{"answer": 12, "type": "integer"}, ...]}
    gt   (codegen):  {"record_id": "...", "gt_code": "<CadQuery source>"}
    gt   (codeedit): {"record_id": "...", "gt_code": "<target>", "orig_code": "<pre-edit source>"}
    pred (codeqa):   {"record_id": "...", "prediction": "[12, 2.5, ...]"}
    pred (codegen):  {"record_id": "...", "prediction": "```python\nimport cadquery...```"}
    pred (codeedit): {"record_id": "...", "prediction": "```python\nimport cadquery...```"}

codegen/codeedit re-execute model code in a subprocess and need the CAD deps
(cadquery / cadquery-ocp / trimesh / vtk); codeqa is pure-numeric and needs none.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import _taskmods


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _index(records: list[dict]) -> dict:
    return {str(r["record_id"]): r for r in records}


def regrade_codeqa(gt: list[dict], pred: list[dict]) -> dict:
    qa = _taskmods.qa_score()
    gt_by_id = _index(gt)
    pred_by_id = _index(pred)

    per_record, missing = [], 0
    for rid, g in gt_by_id.items():
        pairs = g["qa_pairs"]
        p = pred_by_id.get(rid)
        if p is None:
            missing += 1
            score = 0.0
        else:
            answers = qa.parse_json_numbers(p.get("prediction", ""), len(pairs))
            score = qa.qa_score(answers, pairs) if answers is not None else 0.0
        per_record.append({"record_id": rid, "qa_score": score})

    mean = sum(r["qa_score"] for r in per_record) / len(per_record) if per_record else 0.0
    return {"metric": "ratio_accuracy", "score": round(mean, 4),
            "n_records": len(per_record), "n_missing_submissions": missing,
            "per_record": per_record}


def regrade_codegen(gt: list[dict], pred: list[dict]) -> dict:
    exec_cq = _taskmods.exec_cq()
    iou_mod = _taskmods.iou()
    gt_by_id = _index(gt)
    pred_by_id = _index(pred)

    per_record, missing = [], 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for rid, g in gt_by_id.items():
            p = pred_by_id.get(rid)
            if p is None:
                missing += 1
                per_record.append({"record_id": rid, "iou": 0.0})
                continue
            gt_step, pr_step = td / f"{rid}_gt.step", td / f"{rid}_pred.step"
            try:
                exec_cq.execute_cq_to_step(g["gt_code"], gt_step)
                exec_cq.execute_cq_to_step(exec_cq.extract_code(p["prediction"]), pr_step)
                iou = iou_mod.iou_step_vs_step(gt_step, pr_step)
            except Exception:
                iou = 0.0  # non-executable prediction scores 0, never crashes the run
            per_record.append({"record_id": rid, "iou": round(iou, 4)})

    mean = sum(r["iou"] for r in per_record) / len(per_record) if per_record else 0.0
    return {"metric": "voxel_iou", "score": round(mean, 4),
            "n_records": len(per_record), "n_missing_submissions": missing,
            "per_record": per_record}


def regrade_codeedit(gt: list[dict], pred: list[dict]) -> dict:
    """CodeEdit metric is headroom-normalized IoU, not raw IoU.

    norm_iou = (model_iou - baseline_iou) / (1 - baseline_iou), clipped to [0, 1],
    where baseline_iou is the unedited program vs the target. We re-execute the
    orig program here so the baseline is reproduced, not trusted from the gt file.
    """
    exec_cq = _taskmods.exec_cq()
    iou_mod = _taskmods.iou()
    gt_by_id = _index(gt)
    pred_by_id = _index(pred)

    per_record, missing = [], 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for rid, g in gt_by_id.items():
            p = pred_by_id.get(rid)
            if p is None:
                missing += 1
                per_record.append({"record_id": rid, "norm_iou": 0.0})
                continue
            gt_step = td / f"{rid}_gt.step"
            orig_step = td / f"{rid}_orig.step"
            pr_step = td / f"{rid}_pred.step"
            try:
                exec_cq.execute_cq_to_step(g["gt_code"], gt_step)
                exec_cq.execute_cq_to_step(g["orig_code"], orig_step)
                exec_cq.execute_cq_to_step(exec_cq.extract_code(p["prediction"]), pr_step)
                baseline = iou_mod.iou_step_vs_step(orig_step, gt_step)
                model_iou = iou_mod.iou_step_vs_step(pr_step, gt_step)
                n_iou = iou_mod.norm_iou(model_iou, baseline)
            except Exception:
                n_iou = 0.0  # non-executable prediction scores 0, never crashes the run
            per_record.append({"record_id": rid, "norm_iou": round(n_iou, 4)})

    mean = sum(r["norm_iou"] for r in per_record) / len(per_record) if per_record else 0.0
    return {"metric": "norm_iou", "score": round(mean, 4),
            "n_records": len(per_record), "n_missing_submissions": missing,
            "per_record": per_record}


_REGRADERS = {"codeqa": regrade_codeqa, "codegen": regrade_codegen, "codeedit": regrade_codeedit}


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-grade BenchCAD submission predictions.")
    ap.add_argument("--task", required=True, choices=list(_REGRADERS))
    ap.add_argument("--gt", required=True, type=Path, help="ground-truth records jsonl")
    ap.add_argument("--pred", required=True, type=Path, help="submission predictions jsonl")
    ap.add_argument("--model", default="unknown", help="model name for the leaderboard entry")
    ap.add_argument("--out", type=Path, default=None, help="write leaderboard entry json here")
    args = ap.parse_args()

    result = _REGRADERS[args.task](_read_jsonl(args.gt), _read_jsonl(args.pred))
    entry = {"model": args.model, "task": args.task, **{k: v for k, v in result.items() if k != "per_record"}}
    print(json.dumps(entry, indent=2))

    if args.out:
        result["model"], result["task"] = args.model, args.task
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
