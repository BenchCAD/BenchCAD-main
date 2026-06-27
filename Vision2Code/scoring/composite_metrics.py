"""Geometry and feature metrics."""

from __future__ import annotations

import re

LD = __import__("os").environ.get("LD_LIBRARY_PATH", "/workspace/.local/lib")

# ── Feature extraction ────────────────────────────────────────────────────────

_FEATURE_PATTERNS = {
    "has_hole": re.compile(r"\b(hole|cutThruAll|cboreHole|cskHole)\s*\(", re.I),
    "has_fillet": re.compile(r"\bfillet\s*\(", re.I),
    "has_chamfer": re.compile(r"\bchamfer\s*\(", re.I),
}


def _step_has_hole(step_path: str) -> bool:
    """Detect cylindrical inner bore in STEP B-rep (appendix §D.7 Method B).

    REVERSED-oriented cylindrical face with radius ≥ 0.5 mm = inner wall.
    Iterates faces via TopExp to bypass cq.faces() hashCode dependency.
    """
    try:
        import cadquery as cq
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        shape = cq.importers.importStep(step_path)
        root = shape.val().wrapped
        exp = TopExp_Explorer(root, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            ad = BRepAdaptor_Surface(face)
            if (
                ad.GetType() == GeomAbs_Cylinder
                and face.Orientation() == TopAbs_REVERSED
                and ad.Cylinder().Radius() >= 0.5
            ):
                return True
            exp.Next()
    except Exception:
        pass
    return False


def extract_features(code: str, step_path: str | None = None) -> dict[str, bool]:
    """AST regex over code; has_hole prefers STEP B-rep when step_path is given.

    Appendix §D.6 production decision: Method B (STEP-only) for has_hole when
    geometry is available; AST regex used only as fallback for exec_fail samples
    where no gen STEP exists. Reasons:
      - B is geometrically grounded; A is name-matching (Type II/III mismatches).
      - On 1000-sample reliability study (n=1000, 106 families): B F1=0.922,
        C=A OR B F1=0.931 (+0.009, within label-convention noise).
      - C inflates FP by ~2.5pp on hollow-shell families with no benefit on
        F1-style metrics.
    """
    feats = {k: bool(pat.search(code)) for k, pat in _FEATURE_PATTERNS.items()}
    if step_path:
        feats["has_hole"] = _step_has_hole(step_path)
    return feats


def feature_f1(pred: dict, gt: dict) -> float:
    keys = list(gt.keys())
    if not keys:
        return 1.0
    tp = sum(1 for k in keys if pred.get(k) and gt.get(k))
    fp = sum(1 for k in keys if pred.get(k) and not gt.get(k))
    fn = sum(1 for k in keys if not pred.get(k) and gt.get(k))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ── Geometry normalization ────────────────────────────────────────────────────


def _load_normalized_mesh(step_path: str):
    """Load STEP → tessellate → normalize bbox center→[0.5,0.5,0.5], longest→[0,1]³."""
    import cadquery as cq
    import trimesh

    shape = cq.importers.importStep(step_path)
    solid = shape.val()
    if solid is None:
        solids = shape.solids().vals()
        if not solids:
            raise ValueError(f"no solids in {step_path}")
        solid = solids[0]

    verts_raw, tris_raw = solid.tessellate(0.05)
    verts = __import__("numpy").array([[v.x, v.y, v.z] for v in verts_raw], dtype=float)
    tris = __import__("numpy").array(
        [[t[0], t[1], t[2]] for t in tris_raw], dtype=__import__("numpy").int64
    )

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
    out[o[0] : e[0], o[1] : e[1], o[2] : e[2]] = m[
        : e[0] - o[0], : e[1] - o[1], : e[2] - o[2]
    ]
    return out


def compute_iou(gt_step: str, gen_step: str) -> tuple[float, str | None]:
    try:
        import numpy as np

        gt_mesh = _load_normalized_mesh(gt_step)
        gen_mesh = _load_normalized_mesh(gen_step)

        res = 64
        gt_vox = gt_mesh.voxelized(pitch=1.0 / res).fill()
        gen_vox = gen_mesh.voxelized(pitch=1.0 / res).fill()
        gt_d = _vox_dense(gt_vox, res + 4)
        gen_d = _vox_dense(gen_vox, res + 4)
        inter = np.logical_and(gt_d, gen_d).sum()
        union = np.logical_or(gt_d, gen_d).sum()
        return (float(inter / union), None) if union else (0.0, "union empty")
    except Exception as e:
        return 0.0, str(e)[:100]


# ── Rotation-invariant IoU ────────────────────────────────────────────────────


def compute_chamfer(
    gt_step: str, gen_step: str, n_points: int = 2048
) -> tuple[float, str | None]:
    try:
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree

        gt_mesh = _load_normalized_mesh(gt_step)
        gen_mesh = _load_normalized_mesh(gen_step)
        gt_pts = trimesh.sample.sample_surface(gt_mesh, n_points)[0]
        gen_pts = trimesh.sample.sample_surface(gen_mesh, n_points)[0]
        d1 = cKDTree(gen_pts).query(gt_pts)[0]
        d2 = cKDTree(gt_pts).query(gen_pts)[0]
        return float(np.mean(d1**2) + np.mean(d2**2)), None
    except Exception as e:
        return float("inf"), str(e)[:100]


def compute_hausdorff(
    gt_step: str, gen_step: str, n_points: int = 2048
) -> tuple[float, str | None]:
    """Symmetric Hausdorff (max-of-mins both ways) on normalized meshes."""
    try:
        import trimesh
        from scipy.spatial import cKDTree

        gt_mesh = _load_normalized_mesh(gt_step)
        gen_mesh = _load_normalized_mesh(gen_step)
        gt_pts = trimesh.sample.sample_surface(gt_mesh, n_points)[0]
        gen_pts = trimesh.sample.sample_surface(gen_mesh, n_points)[0]
        d1 = cKDTree(gen_pts).query(gt_pts)[0]
        d2 = cKDTree(gt_pts).query(gen_pts)[0]
        return float(max(d1.max(), d2.max())), None
    except Exception as e:
        return float("inf"), str(e)[:100]


# ── CD / HD → bounded score (3-piece linear, see appendix scoring section) ────

_CD_LOW, _CD_HIGH = 0.001, 0.2  # cap=1 below LOW; 0 above HIGH; linear in between
_HD_LOW, _HD_HIGH = 0.05, 0.5  # self-self HD ≈ 0.05 due to sampling noise


def cd_to_score(cd: float) -> float:
    if cd is None or cd != cd or cd == float("inf"):
        return 0.0
    if cd <= _CD_LOW:
        return 1.0
    if cd >= _CD_HIGH:
        return 0.0
    return (_CD_HIGH - cd) / (_CD_HIGH - _CD_LOW)


def hd_to_score(hd: float) -> float:
    if hd is None or hd != hd or hd == float("inf"):
        return 0.0
    if hd <= _HD_LOW:
        return 1.0
    if hd >= _HD_HIGH:
        return 0.0
    return (_HD_HIGH - hd) / (_HD_HIGH - _HD_LOW)


def combined_score(
    feature_f1: float,
    iou: float,
    cd: float,
    hd: float,
    essential_pass: bool | None = None,
    iou_rot: float | None = None,  # noqa: ARG001 — kept in API for back-compat / sidecar reporting; NOT used in score
) -> float:
    """Bench final score — see bench/SCORING.md.

        score = 0.60·IoU + 0.20·essential + 0.10·Feat-F1
              + 0.05·cd_score + 0.05·hd_score

    IoU is the raw fixed-orientation voxel IoU. iou_rot24 is reported per
    stem as a diagnostic (orientation tolerance) but is NEVER added to the
    final score — model is judged on whether it built the correct shape in
    the correct orientation, not on rotation tolerance.

    essential_pass = True / False → counted at full weight (0.20).
                   = None (family is N/A) → drop the 0.20 essential term and
                     rescale the remaining 0.80 weight back to [0, 1] by ×1.25.
    """
    geom = (
        0.60 * iou
        + 0.10 * feature_f1
        + 0.05 * cd_to_score(cd)
        + 0.05 * hd_to_score(hd)
    )  # cumulative 0.80
    if essential_pass is None:
        return round(geom * 1.25, 4)
    return round(geom + 0.20 * (1.0 if essential_pass else 0.0), 4)
