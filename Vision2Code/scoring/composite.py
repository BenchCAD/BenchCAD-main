"""Composite Vision2Code score — the BenchCAD "total" / fused metric.

    score = 0.60·IoU + 0.20·essential-op + 0.10·Feature-F1
          + 0.05·Chamfer-score + 0.05·Hausdorff-score

Default Vision2Code scoring is the raw voxel IoU (the 0.60·IoU term, identical to
the method Anthropic reports). Pass `--score composite` to use this fused score.

Vendored from the BenchCAD generation repo (`bench/metrics` +
`bench/research/canonical_ops`), which is the source of truth for the weights and
the per-family essential-op spec (`canonical_ops.yaml`). Keep in sync with it.
"""

from __future__ import annotations

from pathlib import Path

from . import composite_metrics as _M
from .canonical_ops import essential_pass as _essential_pass
from .canonical_ops import find_ops as _find_ops


def composite_score(gt_step, gen_step, gen_code: str, gt_code: str, family: str) -> float:
    """Fused score for one (gen, gt) pair. Mirrors the generation repo's eval.

    gen_step may be missing (non-executable prediction): geometry terms drop to 0
    and only Feature-F1 + essential-op contribute (combined_score handles the
    N/A re-scaling).
    """
    gen_ok = bool(gen_step) and Path(gen_step).exists()
    gt_ok = bool(gt_step) and Path(gt_step).exists()
    gen_feats = _M.extract_features(gen_code or "", str(gen_step) if gen_ok else None)
    gt_feats = _M.extract_features(gt_code or "", str(gt_step) if gt_ok else None)
    f1 = _M.feature_f1(gen_feats, gt_feats)
    ep = _essential_pass(family, _find_ops(gen_code or ""))
    if not gen_ok or not gt_ok:
        return _M.combined_score(f1, 0.0, float("inf"), float("inf"), essential_pass=ep)
    iou, _ = _M.compute_iou(str(gt_step), str(gen_step))
    cd, _ = _M.compute_chamfer(str(gt_step), str(gen_step))
    hd, _ = _M.compute_hausdorff(str(gt_step), str(gen_step))
    return _M.combined_score(f1, iou, cd, hd, essential_pass=ep)
