"""One-click runner for all three benchmarks (CodeEdit / Vision2Code / QA).

Each task is also independently runnable via its own `main.py`. This wrapper
just dispatches to one or all of them with a shared config name (`test` or
`prod`) and shared debug overrides.

Usage
-----
    # All three with the smoke config (4 records each, gpt-4o):
    uv run python run_all.py

    # All three with the full config (requires HF download per task):
    uv run python run_all.py --config prod

    # Just one:
    uv run python run_all.py --task codegen
    uv run python run_all.py --task codeqa --config prod

    # Plot results for all three (or one):
    uv run python run_all.py --plot
    uv run python run_all.py --task codeedit --plot

    # Debug overrides forwarded to each task's main.py:
    uv run python run_all.py --records topup_clevis_axial_hole
    uv run python run_all.py --limit 2

The config name maps to `<Task>/configs/<config>.yaml` per task.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASKS = {
    "codeedit": "CodeEdit",
    "codegen":  "Vision2Code",
    "codeqa":   "QA",
}


def _run_task(task_dir: Path, config_name: str, extras: list[str]) -> int:
    config_path = task_dir / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        print(f"  !! {task_dir.name}: config not found {config_path.relative_to(ROOT)} — skipping")
        return 1
    cmd = [sys.executable, "main.py", "--config", str(config_path.relative_to(task_dir)), *extras]
    print(f"\n>>> {task_dir.name}: {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=task_dir).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="One-click runner: CodeEdit / Vision2Code / QA")
    ap.add_argument("--task", choices=[*TASKS, "all"], default="all",
                    help="Which task to run (default: all)")
    ap.add_argument("--config", default="test",
                    help="Config NAME (without .yaml). Looked up in <Task>/configs/. Default: test")
    ap.add_argument("--plot", action="store_true",
                    help="Plot from existing results.jsonl. No model calls.")
    ap.add_argument("--records", nargs="*", default=None,
                    help="Debug override: only run these record_ids.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap each task to N records (first N, or a random N with --seed).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random-sample --limit records with this seed (reproducible).")
    ap.add_argument("--score", choices=["iou", "composite"], default="iou",
                    help="Vision2Code scoring: iou (default, Anthropic's method) or "
                         "composite (fused BenchCAD total). Ignored by other tasks.")
    args = ap.parse_args()

    extras: list[str] = []
    if args.plot:
        extras.append("--plot")
    if args.records:
        extras += ["--records", *args.records]
    if args.limit is not None:
        extras += ["--limit", str(args.limit)]
    if args.seed is not None:
        extras += ["--seed", str(args.seed)]

    targets = list(TASKS) if args.task == "all" else [args.task]
    rcs: dict[str, int] = {}
    for key in targets:
        # --score only applies to Vision2Code; other tasks don't accept the flag.
        # Match on the task dir, not the key, so this is robust to key renames.
        task_extras = extras + (["--score", args.score] if TASKS[key] == "Vision2Code" else [])
        rcs[key] = _run_task(ROOT / TASKS[key], args.config, task_extras)

    print("\n=== summary ===")
    for key in targets:
        rc = rcs[key]
        print(f"  {TASKS[key]:<13}{'OK' if rc == 0 else f'FAIL rc={rc}'}")
    sys.exit(0 if all(rc == 0 for rc in rcs.values()) else 1)


if __name__ == "__main__":
    main()
