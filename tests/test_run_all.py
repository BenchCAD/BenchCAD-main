"""Integrity tests for the one-click runner (run_all.py). No model calls — just
verifies the task map points at real, runnable task dirs so a directory rename or
a missing config can't silently break `run_all.py`.
"""

import importlib.util
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run_all():
    spec = importlib.util.spec_from_file_location("benchcad_run_all", _ROOT / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tasks_map_to_real_runnable_dirs():
    tasks = _run_all().TASKS
    # The three benchmark tasks, regardless of the CLI key names.
    assert set(tasks.values()) == {"Vision2Code", "CodeEdit", "QA"}
    for key, dirname in tasks.items():
        d = _ROOT / dirname
        assert d.is_dir(), f"task key {key!r} -> {dirname} is not a directory"
        assert (d / "main.py").exists(), f"{dirname}/main.py is missing"
        assert (d / "configs" / "test.yaml").exists(), \
            f"{dirname}/configs/test.yaml (smoke config) is missing"
