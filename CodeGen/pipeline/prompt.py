"""Build (system, user_text, image_paths) for image → CadQuery code.

A `record` is one row from records.jsonl with paths relative to `data_dir`.
The composite PNG is rendered lazily from `step_path` via scoring.views.
"""

from __future__ import annotations

from pathlib import Path

SYSTEM_PROMPT = """You are an expert CAD engineer. Given a 2x2 composite of 4 diagonal views of a mechanical part, write a CadQuery Python program that reproduces the geometry.

Views (cameras at, all looking at part center [0.5, 0.5, 0.5]):
- Top-left:     [ 1,  1,  1]
- Top-right:    [-1, -1, -1]
- Bottom-left:  [-1,  1, -1]
- Bottom-right: [ 1, -1,  1]

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
    from scoring.views import composite_for_step
    step = data_dir / record["step_path"]
    png = composite_for_step(step)
    return SYSTEM_PROMPT, USER_PROMPT, [png]
