"""Shared loaders for the per-task scoring / rendering modules.

The three contributor tools (regrade, validate_task, ingest_to_hf) all need the
official scoring and rendering code that lives under `QA/scoring/` and
`Vision2Code/scoring/`. Those two dirs both ship a package literally named `scoring`,
so a plain `import scoring.*` collides. Load each module once, by path, here —
so no tool re-implements scoring/rendering and none duplicate this loader.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def qa_score():
    """QA numeric scoring (parse_json_numbers, qa_score, qa_score_single)."""
    return _load("benchcad_qa_score", "QA/scoring/qa_score.py")


def exec_cq():
    """Vision2Code CadQuery execution (extract_code, execute_cq_to_step)."""
    return _load("benchcad_exec_cq", "Vision2Code/scoring/exec_cq.py")


def iou():
    """Vision2Code voxel IoU (iou_step_vs_step, norm_iou)."""
    return _load("benchcad_iou", "Vision2Code/scoring/iou.py")


def views():
    """Vision2Code STEP rendering (composite_for_step, _step_to_normalized_mesh)."""
    return _load("benchcad_views", "Vision2Code/scoring/views.py")
