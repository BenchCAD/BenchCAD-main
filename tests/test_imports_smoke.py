"""Import-smoke tests: every task's runner, its download tool(s), the launcher,
and the contributor tools import cleanly — no API calls, no network.

These catch structural import breakage (e.g. a module moved into benchcad_core
but an importer left pointing at the old path) that the offline unit tests do NOT
exercise — those load scoring by file path, rather than importing the runners and
download tools the way they actually run. (A real such break — the codegen
download tool still doing `from scoring.exec_cq import ...` after the move — got
through the unit suite and was only caught by a manual smoke run; this is that
smoke, automated.)

Each case runs in its own subprocess to mirror how the launcher invokes each task
and to avoid same-name package collisions (every task ships a `pipeline` /
`scoring` package).
"""

import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Import the task runner + every download_*.py tool. The download tools execute
# CAD code and import benchcad_core, so they're the most likely to break on a move.
_TASK_SNIPPET = (
    "from pipeline.runner import run_record\n"
    "import importlib.util, pathlib\n"
    "for dl in sorted(pathlib.Path('tools').glob('download_*.py')):\n"
    "    spec = importlib.util.spec_from_file_location(dl.stem, dl)\n"
    "    spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
)

_CASES = [
    ("benchcad-launcher", _ROOT, "import benchcad"),
    ("vision2code", _ROOT / "Vision2Code", _TASK_SNIPPET),
    ("codeedit", _ROOT / "CodeEdit", _TASK_SNIPPET),
    ("qa", _ROOT / "QA", _TASK_SNIPPET),
    ("contributor-tools", _ROOT, "import sys; sys.path.insert(0, 'tools'); import _taskmods, regrade"),
]


@pytest.mark.parametrize("label,cwd,snippet", _CASES, ids=[c[0] for c in _CASES])
def test_imports_clean(label, cwd, snippet):
    # Replicate each main.py's sys.path: the task dir + the repo root (benchcad_core).
    bootstrap = f"import sys; sys.path[:0] = [r'{cwd}', r'{_ROOT}']\n"
    r = subprocess.run(
        [sys.executable, "-c", bootstrap + snippet],
        cwd=str(cwd), capture_output=True, text=True,
    )
    assert r.returncode == 0, f"{label} import failed:\n{r.stderr[-1500:]}"
