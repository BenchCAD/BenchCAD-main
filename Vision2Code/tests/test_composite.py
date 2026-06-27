"""Offline unit tests for the composite (fused) Vision2Code score.

Guards the scorer vendored from Cadance: the weighting formula, the per-family
essential-op check, op extraction, and the CD/HD bounded-score curves. No API
keys, no network; the geometry terms (IoU/Chamfer/Hausdorff on real STEPs) are
exercised by the full prod run.
"""

import importlib.util
import pathlib

_scoring = pathlib.Path(__file__).resolve().parents[1] / "scoring"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _scoring / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_M = _load("codegen_composite_metrics", "composite_metrics.py")
_canon = _load("codegen_canonical_ops", "canonical_ops.py")


def test_combined_score_perfect_is_one():
    # 0.60·1 + 0.20·1 + 0.10·1 + 0.05·1 + 0.05·1 = 1.0
    assert _M.combined_score(1.0, 1.0, 0.0, 0.0, essential_pass=True) == 1.0


def test_combined_score_weights():
    # IoU-only contributes its 0.60 weight (essential False, others 0/inf).
    assert _M.combined_score(0.0, 1.0, float("inf"), float("inf"),
                             essential_pass=False) == 0.60
    # essential alone contributes 0.20.
    assert _M.combined_score(0.0, 0.0, float("inf"), float("inf"),
                             essential_pass=True) == 0.20


def test_combined_score_na_rescales():
    # essential None (family N/A) → drop the 0.20 term, rescale 0.80 by ×1.25.
    assert _M.combined_score(1.0, 1.0, 0.0, 0.0, essential_pass=None) == 1.0
    assert _M.combined_score(0.0, 0.0, float("inf"), float("inf"),
                             essential_pass=None) == 0.0


def test_cd_hd_to_score_bounds():
    assert _M.cd_to_score(0.0) == 1.0
    assert _M.cd_to_score(float("inf")) == 0.0
    assert _M.hd_to_score(0.0) == 1.0
    assert _M.hd_to_score(float("inf")) == 0.0


def test_find_ops_extracts_features():
    ops = _canon.find_ops(
        "import cadquery as cq\n"
        "result = cq.Workplane('XY').box(1, 1, 1).faces('>Z').hole(2).edges().chamfer(0.5)"
    )
    assert "hole" in ops
    assert "chamfer" in ops


def test_essential_pass_none_for_unknown_family():
    assert _canon.essential_pass("not_a_real_family", {"hole"}) is None
