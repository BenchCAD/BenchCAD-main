"""BenchCAD one-click runner.

    benchcad --task all --num all --model gpt-4o
    benchcad --task vision2code --num 100 --model claude-opus-4-7 --seed 42
    benchcad --model gpt-4o                       # quick smoke (all tasks, 5 records)

Flags
-----
    --task   all | vision2code | codeedit | qa     (default: all)
    --num    all | N                               (default: 5 — a quick smoke)
    --model  MODEL [MODEL ...]                      (required, unless --plot)
    --seed   N        reproducible random N-record sample (with --num N)
    --score  iou | composite     Vision2Code scoring (default: iou)
    --plot   plot existing results, no model calls

Data is pulled from HuggingFace on demand (sliced to --num) into each task's
gitignored data/ folder — nothing to download by hand.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASKS = {"vision2code": "Vision2Code", "codeedit": "CodeEdit", "qa": "QA"}
DOWNLOADERS = {
    "vision2code": "tools/download_codegen_bench.py",
    "codeedit":    "tools/download_edit_bench.py",
    "qa":          "tools/download_qa_bench.py",
}


def _records_count(task_dir: Path) -> int:
    f = task_dir / "data" / "records.jsonl"
    return sum(1 for ln in f.open() if ln.strip()) if f.exists() else 0


def _ensure_data(key: str, task_dir: Path, num: int | None) -> None:
    """Pull data from HF if what's on disk isn't enough for this run.

    A `.full` marker records that the complete split was downloaded, so repeated
    `--num all` runs don't re-fetch. For `--num N`, we only fetch when fewer than
    N records are present.
    """
    data = task_dir / "data"
    marker = data / ".full"
    if num is None:
        if marker.exists():
            return
        dl_args: list[str] = []
    else:
        if _records_count(task_dir) >= num:
            return
        dl_args = ["--limit", str(num)]
    n = "all" if num is None else num
    print(f"  [{key}] fetching data from HuggingFace ({n} records) ...", flush=True)
    rc = subprocess.run([sys.executable, DOWNLOADERS[key], *dl_args], cwd=task_dir).returncode
    if rc != 0:
        sys.exit(f"  !! {key}: data download failed (rc={rc})")
    if num is None:
        marker.touch()


def _run_task(task_dir: Path, extras: list[str]) -> int:
    cmd = [sys.executable, "main.py", "--config", "configs/prod.yaml", *extras]
    print(f"\n>>> {task_dir.name}: {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=task_dir).returncode


def main() -> None:
    ap = argparse.ArgumentParser(prog="benchcad", description="BenchCAD one-click runner")
    ap.add_argument("--task", choices=[*TASKS, "all"], default="all",
                    help="which task to run (default: all)")
    ap.add_argument("--num", default="5",
                    help="all, or N records per task (default: 5 — a quick smoke)")
    ap.add_argument("--model", nargs="+", default=None,
                    help="one or more model names (required unless --plot)")
    ap.add_argument("--seed", type=int, default=None,
                    help="reproducible random N-record sample (with --num N)")
    ap.add_argument("--score", choices=["iou", "composite"], default="iou",
                    help="Vision2Code scoring: iou (default) or composite; other tasks ignore it")
    ap.add_argument("--plot", action="store_true",
                    help="plot from existing results.jsonl. No model calls / no download.")
    args = ap.parse_args()

    if not args.plot and not args.model:
        ap.error("--model is required (one or more model names), e.g. --model gpt-4o")

    if args.num == "all":
        num: int | None = None
    else:
        try:
            num = int(args.num)
            if num <= 0:
                raise ValueError
        except ValueError:
            ap.error(f"--num must be 'all' or a positive integer, got {args.num!r}")

    targets = list(TASKS) if args.task == "all" else [args.task]
    rcs: dict[str, int] = {}
    for key in targets:
        task_dir = ROOT / TASKS[key]
        if args.plot:
            rcs[key] = _run_task(task_dir, ["--plot"])
            continue
        _ensure_data(key, task_dir, num)
        extras = ["--model", *args.model]
        if num is not None:
            extras += ["--limit", str(num)]
        if args.seed is not None:
            extras += ["--seed", str(args.seed)]
        # --score only applies to Vision2Code; matched on dir so it survives renames.
        if TASKS[key] == "Vision2Code":
            extras += ["--score", args.score]
        rcs[key] = _run_task(task_dir, extras)

    print("\n=== summary ===")
    for key in targets:
        rc = rcs[key]
        print(f"  {TASKS[key]:<13}{'OK' if rc == 0 else f'FAIL rc={rc}'}")
    sys.exit(0 if all(rc == 0 for rc in rcs.values()) else 1)


if __name__ == "__main__":
    main()
