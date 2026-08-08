"""Post-install smoke for the publishable packages under `environments/`.

Run inside a bare venv holding only the package's own declared dependencies —
the way the Prime Intellect hub installs it. This checks what a metadata-only
guard cannot: that the pinned stack actually imports and computes together, in
particular that the nlopt pin (chosen so the package resolves against numpy 1.26
at all — see issue #48) still drives cadquery's constraint solver.

Usage: python .github/scripts/env_smoke.py <benchcad-vision2code|benchcad-qa>
"""

import pathlib
import sys
import tempfile

_PART = (
    'import cadquery as cq\n'
    'result = cq.Workplane("XY").box(10, 10, 2).faces(">Z").workplane().hole(3)\n'
)


def smoke_vision2code() -> None:
    import cadquery as cq
    import nlopt
    import numpy
    from benchcad_vision2code import (
        execute_cq_to_step,
        iou_step_vs_step,
        load_environment,
    )

    print(f"numpy {numpy.__version__} · nlopt {getattr(nlopt, '__version__', '?')}")
    assert callable(load_environment)

    # cadquery reaches nlopt only through the sketch constraint solver
    (
        cq.Sketch()
        .segment((0.0, 0.0), (2.0, 0.0), "a")
        .segment((2.0, 0.0), (2.0, 2.0), "b")
        .constrain("a", "Fixed", None)
        .constrain("a", "b", "Coincident", None)
        .constrain("a", "b", "Angle", 90.0)
        .solve()
    )
    print("sketch constraint solve (nlopt) ok")

    # the scoring path end to end: execute -> STEP -> mesh -> voxel IoU
    with tempfile.TemporaryDirectory() as d:
        a, b = pathlib.Path(d) / "a.step", pathlib.Path(d) / "b.step"
        execute_cq_to_step(_PART, a)
        execute_cq_to_step(_PART, b)
        iou = iou_step_vs_step(a, b)
    assert iou > 0.99, f"self-IoU should be ~1.0, got {iou}"
    print(f"execute + voxel IoU ok (self-IoU {iou})")


def smoke_qa() -> None:
    from benchcad_qa import load_environment

    assert callable(load_environment)
    print("import ok")


if __name__ == "__main__":
    name = sys.argv[1]
    {"benchcad-vision2code": smoke_vision2code, "benchcad-qa": smoke_qa}[name]()
    print(f"{name}: smoke passed")
