"""Per-record loop: build prompt → call model → parse JSON → score → save.

Per-record output goes to:
    <results_root>/outputs/<safe_model>/<record_id>.{txt,json}

Aggregated scores live in `<results_root>/results.jsonl`, one row per
(model, record_id). Re-running a tuple replaces its old row.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scoring.qa_score import parse_json_numbers, qa_score, qa_score_single

from pipeline.prompt import build as build_prompt


def _safe_model(m: str) -> str:
    return m.replace(":", "_").replace("=", "_").replace("/", "_")


def _outputs_paths(results_root: Path, model: str, rid: str) -> dict:
    base = results_root / "outputs" / _safe_model(model)
    base.mkdir(parents=True, exist_ok=True)
    return {
        "raw":  base / f"{rid}.txt",
        "json": base / f"{rid}.json",
    }


def _read_results(jsonl: Path) -> dict[tuple[str, str], dict]:
    out: dict = {}
    if not jsonl.exists():
        return out
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["model"], r["record_id"])] = r
    return out


def _write_results(jsonl: Path, rows: dict) -> None:
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_record(*, record: dict, data_dir: Path, results_root: Path,
               model: str, mode: str = "code") -> dict:
    """Run one (model, record) → write the result row to results.jsonl."""
    rid = record["record_id"]
    paths = _outputs_paths(results_root, model, rid)
    qa_pairs = record.get("qa_pairs", []) or []

    if not qa_pairs:
        row = {
            "record_id": rid, "model": model, "status": "no_qa",
            "qa_score": 0.0, "n_qa": 0, "lat_s": 0.0,
            "err": "record has no qa_pairs", "per_qa": [],
            "raw_path": None, "json_path": None,
        }
        jsonl = results_root / "results.jsonl"
        rows = _read_results(jsonl); rows[(model, rid)] = row; _write_results(jsonl, rows)
        return row

    # 1. Build prompt
    system, user_text, image_paths = build_prompt(record, data_dir, mode=mode)

    # 2. Call model (text-only)
    from benchcad_core.models import call_model
    t0 = time.time()
    try:
        raw = call_model(model=model, system=system, user_text=user_text,
                         image_paths=image_paths, max_tokens=4096)
        api_err = None
    except Exception as e:
        raw, api_err = "", f"{type(e).__name__}: {e}"
    lat = time.time() - t0
    paths["raw"].write_text(raw or "")

    # 3. Parse JSON array of numbers
    if api_err:
        status, err_msg, answers = "api_fail", api_err, None
    else:
        answers = parse_json_numbers(raw, len(qa_pairs))
        if answers is None:
            status, err_msg = "parse_fail", f"could not extract {len(qa_pairs)} numbers from response"
        else:
            status, err_msg = "ok", None

    # 4. Score
    if status == "ok":
        per_qa = [
            {
                "q": q["question"],
                "type": q.get("type", "dim"),
                "gt": q["answer"],
                "pred": p,
                "score": qa_score_single(p, q["answer"], q.get("type", "dim")),
            }
            for q, p in zip(qa_pairs, answers, strict=True)
        ]
        score = qa_score(answers, qa_pairs)
    else:
        per_qa = [{"q": q["question"], "type": q.get("type", "dim"), "gt": q["answer"],
                   "pred": None, "score": 0.0} for q in qa_pairs]
        score = 0.0

    paths["json"].write_text(json.dumps(
        {"answers": answers, "per_qa": per_qa, "qa_score": score},
        ensure_ascii=False, indent=2,
    ))

    # 5. Persist with overwrite by (model, record_id)
    row = {
        "record_id": rid,
        "model": model,
        "mode": mode,
        "status": status,
        "qa_score": round(float(score), 4),
        "n_qa": len(qa_pairs),
        "lat_s": round(lat, 2),
        "err": err_msg,
        "pred_answers": answers,
        "gt_answers": [q["answer"] for q in qa_pairs],
        "per_qa": per_qa,
        "raw_path":  str(paths["raw"].relative_to(results_root))  if paths["raw"].exists()  else None,
        "json_path": str(paths["json"].relative_to(results_root)) if paths["json"].exists() else None,
    }
    jsonl = results_root / "results.jsonl"
    rows = _read_results(jsonl)
    rows[(model, rid)] = row
    _write_results(jsonl, rows)
    return row
