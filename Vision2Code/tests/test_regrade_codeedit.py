"""regrade.py must score CodeEdit by norm_iou (headroom-normalized improvement),
NOT raw voxel IoU. Regression test for the bug where `--task codeedit` was routed
to the codegen regrader and reported raw IoU.

A prediction equal to the *unedited* program made no progress → norm_iou 0, even
though its raw IoU vs the target is > 0. A prediction equal to the *target* →
norm_iou 1. Needs CAD execution (cadquery), no network.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools"))

import regrade  # noqa: E402

_GT = "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 2)"
_ORIG = "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)"


def _fence(code):
    return f"```python\n{code}\n```"


def test_codeedit_routes_to_norm_iou_regrader():
    assert regrade._REGRADERS["codeedit"] is regrade.regrade_codeedit


def test_codeedit_prediction_equal_to_orig_scores_zero():
    gt = [{"record_id": "t1", "gt_code": _GT, "orig_code": _ORIG}]
    pred = [{"record_id": "t1", "prediction": _fence(_ORIG)}]  # no edit made
    out = regrade.regrade_codeedit(gt, pred)
    assert out["metric"] == "norm_iou"
    assert out["per_record"][0]["norm_iou"] == 0.0  # raw-IoU regrader would report > 0


def test_codeedit_prediction_equal_to_target_scores_one():
    gt = [{"record_id": "t1", "gt_code": _GT, "orig_code": _ORIG}]
    pred = [{"record_id": "t1", "prediction": _fence(_GT)}]  # perfect edit
    out = regrade.regrade_codeedit(gt, pred)
    assert out["per_record"][0]["norm_iou"] == 1.0
