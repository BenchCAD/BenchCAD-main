"""Offline unit tests for Vision2Code scoring helpers. No API keys, no network,
no CAD execution (those paths are exercised by the full prod run)."""

import importlib.util
import pathlib

# Load Vision2Code/scoring/*.py by path under unique names so they don't collide
# with QA's same-named `scoring` package during a shared pytest run.
_scoring = pathlib.Path(__file__).resolve().parents[2] / "benchcad_core" / "scoring"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _scoring / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_exec_cq = _load("codegen_exec_cq", "exec_cq.py")
_iou = _load("codegen_iou", "iou.py")
extract_code = _exec_cq.extract_code
iou_step_vs_step = _iou.iou_step_vs_step
norm_iou = _iou.norm_iou


# --- norm_iou --------------------------------------------------------------

def test_norm_iou_basic():
    # halfway between baseline 0.5 and perfect 1.0 -> 0.5
    assert norm_iou(0.75, 0.5) == 0.5
    # model below baseline clips to 0
    assert norm_iou(0.4, 0.5) == 0.0
    # model at perfect -> 1
    assert norm_iou(1.0, 0.5) == 1.0


def test_norm_iou_perfect_baseline():
    # baseline already perfect: only a perfect model scores 1
    assert norm_iou(1.0, 1.0) == 1.0
    assert norm_iou(0.9, 1.0) == 0.0


# --- iou_step_vs_step ------------------------------------------------------

def test_iou_missing_files_returns_zero():
    # robustness contract: any failure -> 0.0, never raises
    missing = pathlib.Path("/nonexistent/a.step")
    assert iou_step_vs_step(missing, missing) == 0.0


# --- extract_code ----------------------------------------------------------

def test_extract_fenced_python():
    raw = "Here:\n```python\nimport cadquery as cq\nresult = cq.Workplane()\n```\nDone."
    code = extract_code(raw)
    assert "import cadquery as cq" in code
    assert "```" not in code


def test_extract_unfenced_cadquery():
    raw = "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)"
    assert "cq.Workplane" in extract_code(raw)


def test_extract_no_code_returns_empty():
    assert extract_code("just some prose, no code") == ""
    assert extract_code("") == ""
