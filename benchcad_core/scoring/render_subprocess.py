"""Render a STEP to its 2x2 composite in a child process.

VTK's Cocoa backend can only build a render window on the main thread, and it
leaks a graphics context per render — a few hundred in-process renders exhaust
memory and a render from a worker aborts the whole process with an NSException.
The agentic runner renders the model's own geometry every round, from inside a
worker pool, so it cannot use `composite_for_step` directly.

Running each render in a child process sidesteps both problems: the child has
its own main thread, and its leaks die with it.

    python -m benchcad_core.scoring.render_subprocess <step> <png>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# A render of a pathological solid can hang in tessellation; bound it so a
# single bad part cannot stall an agentic round.
DEFAULT_TIMEOUT = 120


def render(step: Path, png: Path, timeout: int = DEFAULT_TIMEOUT) -> Path | None:
    """Render `step` to `png` out of process. Returns the path, or None on failure.

    Failure is not raised: a preview the model cannot see is a degraded round,
    not a reason to fail the record.
    """
    step, png = Path(step), Path(png)
    if not step.exists():
        return None
    png.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "benchcad_core.scoring.render_subprocess",
             str(step), str(png)],
            capture_output=True, timeout=timeout,
            cwd=Path(__file__).resolve().parents[2],
        )
    except subprocess.TimeoutExpired:
        return None
    return png if (r.returncode == 0 and png.exists()) else None


def _main() -> int:
    step, png = Path(sys.argv[1]), Path(sys.argv[2])
    from benchcad_core.scoring.views import composite_for_step
    composite_for_step(step, png)
    return 0 if png.exists() else 1


if __name__ == "__main__":
    raise SystemExit(_main())
