"""Build (system, user_text, image_paths) for image → CadQuery code.

A `record` is one row from records.jsonl with paths relative to `data_dir`.
The composite PNG is rendered lazily from `step_path` via scoring.views.
"""

from __future__ import annotations

import os
from pathlib import Path

# How much of the camera geometry the prompt hands over. The default states all
# four cameras; `anchor` states one and leaves the rest as a range. 86% of the
# 299 agentic runs transcribed these vectors into their own code and used them,
# so the block is load-bearing rather than decorative -- which is exactly why
# withholding most of it is worth measuring. Selected by env var so the two arms
# differ in this one block and nothing else.
_VIEWS_FULL = """Views (cameras at, all looking at part center [0.5, 0.5, 0.5]):
- Top-left:     [-1, -1, -1]
- Top-right:    [ 1,  1,  1]
- Bottom-left:  [ 1, -1,  1]
- Bottom-right: [-1,  1, -1]"""

_VIEWS_ANCHOR = """Views (all looking at part center [0.5, 0.5, 0.5]):
- Top-right: camera at [ 1,  1,  1].
- The other three are corner views as well -- each camera sits at a distinct
  direction whose three components are each either +1 or -1 -- but which view
  is in which quadrant is not given."""

# `perturb` goes with targets whose three undisclosed views were re-rendered
# from cameras rotated 15 degrees off the corner directions. The wording has to
# change with them: under `anchor` the components really are +-1, and once the
# cameras move that sentence would be false. The magnitude is deliberately not
# stated -- naming it would hand over the recipe for searching the offset back.
_VIEWS_PERTURB = """Views (all looking at part center [0.5, 0.5, 0.5]):
- Top-right: camera at [ 1,  1,  1].
- The other three are diagonal views as well, from directions near -- but not
  exactly at -- the remaining corner directions. Neither the exact directions
  nor which view sits in which quadrant is given."""

_VIEWS = {"anchor": _VIEWS_ANCHOR,
          "perturb": _VIEWS_PERTURB}.get(os.environ.get("BENCH_VIEW_HINT"),
                                         _VIEWS_FULL)

SYSTEM_PROMPT = f"""You are an expert CAD engineer. Given a 2x2 composite of 4 diagonal views of a mechanical part, write a CadQuery Python program that reproduces the geometry.

{_VIEWS}

Renders are normalized: bbox centered at [0.5, 0.5, 0.5], longest side maps to [0,1]. Match orientation exactly — world XYZ in your code must match world XYZ in the renders.

Output ONLY a single ```python fenced block:
- start with `import cadquery as cq`
- store the final solid in `result`
- no prose, no comments outside the fence"""

USER_PROMPT = (
    "Generate CadQuery code to recreate this industrial part shown in the "
    "4-view composite render."
)


def build(record: dict, data_dir: Path) -> tuple[str, str, list[Path]]:
    from benchcad_core.scoring.views import composite_for_step
    step = data_dir / record["step_path"]
    png = composite_for_step(step)
    return SYSTEM_PROMPT, USER_PROMPT, [png]
