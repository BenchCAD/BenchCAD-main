"""Pre-build the Vision2Code ground-truth STEP solids, once.

Vision2Code scores against a GT STEP per record, but the dataset ships GT *code*,
not STEP. STEPs are otherwise built on the fly by executing each program
(`scoring.exec_cq.execute_cq_to_step`). That is slow for a tail of programs and
timeout/hardware-sensitive, so independent runs drop a handful of the slowest
records — see `docs/ERRATA.md` (the "26 records" issue) and the note in
Anthropic's system card that omits "26 records whose CadQuery code failed to
produce a STEP file".

Building every STEP once and shipping it removes the local-exec dependency:
scoring becomes identical on any machine, over the full 17,900-record set.

Output (under ``--out``):
    steps/<stem>.step     one STEP per successfully built record
    gt_steps.parquet      columns: stem, gt_step (bytes), build_ok, build_s, error
    report.json           summary + the list of any records that failed to build

``gt_steps.parquet`` is the shippable artifact: publish it as a ``gt_step`` config
on the HF dataset and have `download_codegen_bench.py` read it instead of executing
GT code. Re-runs are resumable — records whose STEP already exists are skipped.

Usage (run from Vision2Code/):
    uv run python tools/build_gt_steps.py --out gt_out --workers 8
    # second pass: retry the slow tail serially with a generous budget
    uv run python tools/build_gt_steps.py --out gt_out --retry-timeout 900
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for benchcad_core
from benchcad_core.scoring.exec_cq import execute_cq_to_step  # noqa: E402

REPO = "BenchCAD/BenchCAD"
SUBDIR = "code_gen/data"


def _load_code(cache_dir: Path | None, max_shards: int | None) -> pd.DataFrame:
    if max_shards is not None:
        patterns = [f"{SUBDIR}/code_gen-{i:05d}-of-*.parquet" for i in range(max_shards)]
    else:
        patterns = f"{SUBDIR}/*.parquet"
    snap = Path(snapshot_download(repo_id=REPO, repo_type="dataset",
                                  allow_patterns=patterns, cache_dir=cache_dir))
    pqs = sorted((snap / SUBDIR).rglob("*.parquet"))
    if not pqs:
        sys.exit(f"no parquet shards under {snap / SUBDIR}")
    return pd.concat([pd.read_parquet(p, columns=["stem", "code"]) for p in pqs],
                     ignore_index=True)


def _build_one(stem: str, code: str, steps_dir: Path, timeout: int) -> tuple[str, bool, float, str]:
    step = steps_dir / f"{stem}.step"
    if step.exists() and step.stat().st_size > 0:
        return stem, True, 0.0, "cached"
    t0 = time.time()
    try:
        execute_cq_to_step(code, step, timeout=timeout)
        return stem, True, time.time() - t0, ""
    except Exception as e:  # noqa: BLE001 — record the reason, keep going
        return stem, False, time.time() - t0, f"{type(e).__name__}: {e}"[:200]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-build Vision2Code GT STEP solids.")
    ap.add_argument("--out", type=Path, required=True, help="output dir")
    ap.add_argument("--workers", type=int, default=8, help="parallel build workers")
    ap.add_argument("--timeout", type=int, default=300, help="per-record exec timeout (first pass)")
    ap.add_argument("--retry-timeout", type=int, default=0,
                    help="if >0, retry the failures serially with this larger timeout")
    ap.add_argument("--cache-dir", type=Path, default=None, help="HF download cache")
    ap.add_argument("--max-shards", type=int, default=None, help="only first N parquet shards")
    ap.add_argument("--limit", type=int, default=None, help="only first N records")
    args = ap.parse_args()

    df = _load_code(args.cache_dir, args.max_shards)
    if args.limit is not None:
        df = df.head(args.limit)
    steps = args.out / "steps"
    steps.mkdir(parents=True, exist_ok=True)
    jobs = [(r.stem, r.code) for r in df.itertuples(index=False)]
    print(f"building {len(jobs)} GT STEPs → {steps}  ({args.workers} workers, timeout {args.timeout}s)")

    results: dict[str, tuple[bool, float, str]] = {}
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_build_one, s, c, steps, args.timeout): s for s, c in jobs}
        for fut in as_completed(futs):
            stem, ok, dt, err = fut.result()
            results[stem] = (ok, dt, err)
            done += 1
            if done % 500 == 0 or not ok:
                print(f"  {done}/{len(jobs)}  {'ok  ' if ok else 'FAIL'} {stem} {dt:4.0f}s {err[:55]}")

    fails = [s for s, (ok, _, _) in results.items() if not ok]
    if args.retry_timeout and fails:
        code_by = {r.stem: r.code for r in df.itertuples(index=False)}
        print(f"retrying {len(fails)} failure(s) serially, timeout {args.retry_timeout}s")
        for s in list(fails):
            _, ok, dt, err = _build_one(s, code_by[s], steps, args.retry_timeout)
            results[s] = (ok, dt, err)
        fails = [s for s, (ok, _, _) in results.items() if not ok]

    recs = []
    for r in df.itertuples(index=False):
        ok, dt, err = results[r.stem]
        sp = steps / f"{r.stem}.step"
        recs.append({"stem": r.stem,
                     "gt_step": sp.read_bytes() if (ok and sp.exists()) else None,
                     "build_ok": ok, "build_s": round(dt, 2), "error": err})
    pd.DataFrame(recs).to_parquet(args.out / "gt_steps.parquet")

    times = [dt for _, dt, _ in results.values() if dt > 0]
    report = {
        "total": len(df),
        "built": sum(1 for ok, _, _ in results.values() if ok),
        "failed": len(fails),
        "max_build_s": round(max(times), 1) if times else 0.0,
        "over_90s": sum(1 for t in times if t > 90),
        "failures": [{"stem": s, "error": results[s][2]} for s in fails],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nbuilt {report['built']}/{report['total']}   failed {report['failed']}   "
          f"max {report['max_build_s']}s   >90s: {report['over_90s']}")
    print(f"artifact → {args.out / 'gt_steps.parquet'}   report → {args.out / 'report.json'}")
    if fails:
        print("failed to build:", fails[:20])


if __name__ == "__main__":
    main()
