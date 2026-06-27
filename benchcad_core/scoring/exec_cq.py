"""Run model-generated CadQuery code → write a STEP file.

Two helpers:
    extract_code(raw)         pull python from a fenced block
    execute_cq_to_step(code, step_path, timeout=300)
        runs the code in a subprocess; if no .exportStep is present we append one
        for `result`. Raises RuntimeError on subprocess failure or missing output.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_FENCE = re.compile(r"```(?:python|py|cadquery)?\s*\n(.*?)```", re.DOTALL)


def extract_code(raw: str) -> str:
    """First ```python ... ``` block, or whole text if no fence."""
    m = _FENCE.search(raw or "")
    if m:
        return m.group(1).strip()
    if raw and ("import cadquery" in raw or "cq.Workplane" in raw):
        return raw.strip()
    return ""


_OCP_HASHCODE_FIX = """
# cadquery 2.3 ↔ cadquery-ocp 7.9 compat: OCP removed TopoDS_*.HashCode but
# cq's exporter still calls it. Restore as identity-based stub.
from OCP.TopoDS import (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
    TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid)
for _cls in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
             TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid):
    if not hasattr(_cls, "HashCode"):
        _cls.HashCode = lambda self, ub=2147483647: id(self) % ub
# Stub show_object (defined only inside CQ-editor) so model code with it doesn't crash
def show_object(*a, **k): pass
"""


def _patch_export(code: str, out_step: Path) -> str:
    """Prepend OCP/cq compat shim, then replace last .exportStep("...") to write
    to out_step (append if model didn't include one)."""
    out_lit = str(out_step).replace("\\", "\\\\")
    patched = re.sub(
        r'(\.exportStep\s*\()["\'].*?["\']\s*\)',
        f'.exportStep("{out_lit}")',
        code,
    )
    if ".exportStep" not in patched:
        # The prompt asks for "the final solid in `result`", not for a Workplane:
        # raw Shapes (Solid/Compound/...) have .exportStep directly, only
        # Workplane needs .val() first.
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
        r = subprocess.run(
            [sys.executable, tmp],
            env=os.environ.copy(),
            timeout=timeout,
            capture_output=True,
        )
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
