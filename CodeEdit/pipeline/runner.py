"""Per-record run loop: build prompt → call model → exec → score → render → save.

Per-record output goes to:
    <results_root>/outputs/<mode>/<safe_model>/<record_id>.{py,step,png}

Aggregated scores live in `<results_root>/results.jsonl`, with one row per
(mode, model, record_id) — overwrite mechanism: re-running a tuple replaces
its old row.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from benchcad_core.models.pricing import cost_usd
from benchcad_core.run_config import (
    DEFAULT_EXEC_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
)
from benchcad_core.scoring.exec_cq import execute_cq_to_step, extract_code
from benchcad_core.scoring.iou import iou_step_vs_step, norm_iou
from pipeline.modes import build as build_prompt


def _safe_model(m: str) -> str:
    return m.replace(":", "_").replace("=", "_").replace("/", "_")


def _outputs_paths(results_root: Path, mode: str, model: str, rid: str) -> dict:
    base = results_root / "outputs" / mode / _safe_model(model)
    base.mkdir(parents=True, exist_ok=True)
    return {
        "code": base / f"{rid}.py",
        "step": base / f"{rid}.step",
        "png":  base / f"{rid}.png",
    }


def _read_results(jsonl: Path) -> dict[tuple[str, str, str], dict]:
    """Read existing results.jsonl into a dict keyed by (mode, model, record_id)."""
    out: dict = {}
    if not jsonl.exists():
        return out
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["mode"], r["model"], r["record_id"])
        out[key] = r
    return out


def _write_results(jsonl: Path, rows: dict) -> None:
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_record(*, record: dict, data_dir: Path, results_root: Path,
               model: str, mode: str,
               max_tokens: int = DEFAULT_MAX_TOKENS, timeout: int = DEFAULT_TIMEOUT,
               exec_timeout: int = DEFAULT_EXEC_TIMEOUT) -> dict:
    """Run one (mode, model, record) → write the result row to results.jsonl."""
    rid = record["record_id"]
    paths = _outputs_paths(results_root, mode, model, rid)

    # 1. Build prompt
    system, user_text, image_paths = build_prompt(record, data_dir, mode)

    # 2. Call model
    from benchcad_core.models import call_model
    t0 = time.time()
    usage: dict = {}
    try:
        raw, usage = call_model(
            model=model, system=system, user_text=user_text,
            image_paths=image_paths, max_tokens=max_tokens, timeout=timeout,
        )
        api_err = None
    except Exception as e:
        raw, api_err = "", f"{type(e).__name__}: {e}"
    lat = time.time() - t0

    # 3. Extract + execute → STEP
    code = extract_code(raw) if raw else ""
    paths["code"].write_text(code or raw or "")
    if api_err:
        status, err_msg = "api_fail", api_err
    elif not code.strip():
        status, err_msg = "no_code", "no parseable code in response"
    else:
        try:
            execute_cq_to_step(code, paths["step"], timeout=exec_timeout)
            status, err_msg = "ok", None
        except Exception as e:
            status, err_msg = "exec_fail", f"{type(e).__name__}: {e}"

    # 4. Score (only if exec succeeded). norm_iou uses baseline iou from records.jsonl.
    if status == "ok" and paths["step"].exists():
        gt_step = data_dir / record["gt_step_path"]
        try:
            mi = iou_step_vs_step(paths["step"], gt_step)
        except Exception as e:
            mi = 0.0
            err_msg = f"iou_fail: {type(e).__name__}: {e}"
            status = "score_fail"
    else:
        mi = 0.0

    baseline = float(record.get("iou", 0.0))
    n_iou = norm_iou(mi, baseline)

    # 5. Render model STEP → composite PNG (Tiffany blue) for inspection
    if status == "ok" and paths["step"].exists():
        try:
            from benchcad_core.scoring.views import composite_for_step
            composite_for_step(paths["step"], paths["png"])
        except Exception:
            pass  # render is non-critical for scoring

    # 6. Persist with overwrite by (mode, model, record_id)
    row = {
        "record_id": rid,
        "model": model,
        "mode": mode,
        "status": status,
        "norm_iou": round(float(n_iou), 4),
        "lat_s": round(lat, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": cost_usd(model, usage.get("prompt_tokens"), usage.get("completion_tokens")),
        "err": err_msg,
        "code_path": str(paths["code"].relative_to(results_root)) if paths["code"].exists() else None,
        "step_path": str(paths["step"].relative_to(results_root)) if paths["step"].exists() else None,
        "png_path":  str(paths["png"].relative_to(results_root))  if paths["png"].exists()  else None,
    }
    jsonl = results_root / "results.jsonl"
    rows = _read_results(jsonl)
    rows[(mode, model, rid)] = row
    _write_results(jsonl, rows)
    return row
