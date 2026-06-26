"""Render-and-verify eval mode — the *with Python tools* setting from the system card (§8.16.4).

Vanilla Vision2Code (`pipeline.runner.run_record`) is single-shot: the model sees the 4-view target
and writes CadQuery once. Frontier-model evaluations show a large lift when the model is additionally
allowed to **render its own candidate and visually verify it before submitting**. This module adds
that loop while reusing the exact same renderer (`scoring.views`), executor and IoU as scoring, so the
number stays comparable:

    round 1            : single-shot (identical to run_record)
    rounds 2..N        : render the current candidate in the SAME 4 views, show it to the model next
                         to the target, ask it to fix the mismatch; keep the revision only if it still
                         executes to a valid STEP.

Enable from a config by adding:

    render_verify:
      rounds: 3        # total attempts including the first single-shot

`run_record_with_tools` writes the same `results.jsonl` row shape as `run_record`, plus `rounds`
(how many attempts were actually used).
"""

from __future__ import annotations

import time

from pipeline.prompt import SYSTEM_PROMPT, build as build_prompt
from pipeline.runner import _outputs_paths, _read_results, _write_results
from scoring.exec_cq import execute_cq_to_step, extract_code
from scoring.iou import iou_step_vs_step

REFINE_PROMPT = (
    "The FIRST image is the TARGET part (2x2 of 4 diagonal views). The SECOND image is YOUR current "
    "CadQuery program rendered in the SAME 4 views, same cameras and normalization.\n\n"
    "Compare them view-by-view and correct every mismatch: orientation / axis, proportions, absolute "
    "thickness (keep flat parts flat), and feature counts (holes, teeth, fins, slots, lugs). If the two "
    "already match, return your program unchanged.\n\n"
    "Output ONLY the corrected full program as a single ```python fenced block: start with "
    "`import cadquery as cq` and store the final solid in `result`."
)


def _exec_ok(code: str, step_path) -> bool:
    """Execute `code` to `step_path`; True iff it produced a STEP. Never raises."""
    try:
        execute_cq_to_step(code, step_path)
        return step_path.exists()
    except Exception:
        return False


def run_record_with_tools(*, record: dict, data_dir, results_root, model: str,
                          rounds: int = 3, max_tokens: int = 8192) -> dict:
    """Run one (model, record) with render-and-verify refinement → write its results.jsonl row."""
    from models import call_model
    from scoring.views import composite_for_step

    rid = record["record_id"]
    paths = _outputs_paths(results_root, model, rid)
    system, user_text, image_paths = build_prompt(record, data_dir)
    target_png = image_paths[0]

    t0 = time.time()
    try:
        raw = call_model(model=model, system=system, user_text=user_text,
                         image_paths=image_paths, max_tokens=max_tokens)
        api_err = None
    except Exception as e:
        raw, api_err = "", f"{type(e).__name__}: {e}"

    best_code = extract_code(raw) if raw else ""
    used = 1
    have_step = bool(best_code.strip()) and _exec_ok(best_code, paths["step"])

    # rounds 2..N: render the current best, ask the model to verify + fix; keep only executing revisions.
    cand_png = paths["png"].with_name(f"{rid}_cand.png")
    try_step = paths["step"].with_name(f"{rid}_try.step")
    for r in range(2, rounds + 1):
        if api_err or not have_step:
            break
        try:
            composite_for_step(paths["step"], cand_png)
        except Exception:
            break  # can't render the candidate → nothing to verify against
        try:
            raw = call_model(model=model, system=SYSTEM_PROMPT, user_text=REFINE_PROMPT,
                             image_paths=[target_png, cand_png], max_tokens=max_tokens)
        except Exception:
            break  # transient API failure → submit the best so far
        used = r
        new_code = extract_code(raw)
        if new_code.strip() and _exec_ok(new_code, try_step):
            try_step.replace(paths["step"])
            best_code = new_code

    paths["code"].write_text(best_code or raw or "")
    lat = time.time() - t0

    # score (raw voxel IoU vs GT — identical to run_record)
    err_msg = api_err
    if api_err:
        status, iou = "api_fail", 0.0
    elif not best_code.strip():
        status, iou = "no_code", 0.0
        err_msg = "no parseable code in response"
    elif not paths["step"].exists():
        status, iou = "exec_fail", 0.0
        err_msg = "no valid solid after refinement"
    else:
        try:
            iou = iou_step_vs_step(paths["step"], data_dir / record["step_path"])
            status = "ok"
        except Exception as e:
            status, iou, err_msg = "score_fail", 0.0, f"iou_fail: {type(e).__name__}: {e}"
        try:
            composite_for_step(paths["step"], paths["png"])  # final render for inspection
        except Exception:
            pass

    row = {
        "record_id": rid, "model": model, "status": status, "iou": round(float(iou), 4),
        "rounds": used, "lat_s": round(lat, 2), "err": err_msg,
        "code_path": str(paths["code"].relative_to(results_root)) if paths["code"].exists() else None,
        "step_path": str(paths["step"].relative_to(results_root)) if paths["step"].exists() else None,
        "png_path":  str(paths["png"].relative_to(results_root))  if paths["png"].exists()  else None,
    }
    jsonl = results_root / "results.jsonl"
    rows = _read_results(jsonl)
    rows[(model, rid)] = row
    _write_results(jsonl, rows)
    return row
