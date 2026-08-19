"""Vision2Code runner — config-driven (image → CadQuery code).

Two independent operations:

  (1) RUN: read config → for each (model × record) build prompt (4-view image →
      CadQuery code) → call model → exec CadQuery → score (voxel IoU vs GT) →
      write to <out_dir>/results.jsonl (overwrite by (model, record_id)).

  (2) PLOT: read <out_dir>/results.jsonl → husl bar plot (mean IoU per model).

Examples (run from Vision2Code/)
----------------------------
    # Smoke run (default config: configs/test.yaml):
    uv run python main.py

    # Full bench:
    uv run python main.py --config configs/prod.yaml

    # Plot whatever's in results.jsonl for that config:
    uv run python main.py --plot

    # Debug overrides:
    uv run python main.py --records r1 r2
    uv run python main.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # repo root, for benchcad_core

from pipeline.runner import run_record  # noqa: E402

from benchcad_core.run_config import agentic_params, concurrency_params, gen_params  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "test.yaml"
REQUIRED_FIELDS = ("data_dir", "out_dir", "models")


def parse_args():
    p = argparse.ArgumentParser(description="Vision2Code runner (config-driven)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"YAML config (default: {DEFAULT_CONFIG.relative_to(ROOT)})")
    p.add_argument("--plot", action="store_true",
                   help="Render bar plot from <out_dir>/results.jsonl. No model calls.")
    p.add_argument("--records", nargs="*", default=None,
                   help="Debug override: only run these record_ids.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap to N records (first N, or a random N with --seed).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random-sample --limit records with this seed (reproducible).")
    p.add_argument("--score", choices=["iou", "composite"], default="iou",
                   help="iou = raw voxel IoU (default, Anthropic's method); "
                        "composite = fused BenchCAD total (IoU+essential+F1+CD+HD).")
    p.add_argument("--model", nargs="*", default=None,
                   help="Override the config's model list (one or more model names).")
    return p.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    missing = [k for k in REQUIRED_FIELDS if k not in cfg]
    if missing:
        sys.exit(f"config {path} missing fields: {missing}")
    return cfg


def load_records(data_dir: Path) -> list[dict]:
    jsonl = data_dir / "records.jsonl"
    if not jsonl.exists():
        sys.exit(f"records.jsonl not found at {jsonl}")
    return [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]


def do_plot(out_dir: Path) -> None:
    from pipeline.plot import make_bar
    jsonl = out_dir / "results.jsonl"
    if not jsonl.exists():
        sys.exit(f"results.jsonl not found at {jsonl}")
    out_png = out_dir / "plot.png"
    make_bar(jsonl, out_png)
    print(f"plot → {out_png}")


def _print_results_summary(out_dir: Path) -> None:
    jsonl = out_dir / "results.jsonl"
    if not jsonl.exists():
        return
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    if not rows:
        return
    bucket: dict[str, list[float]] = {}
    for r in rows:
        bucket.setdefault(r["model"], []).append(float(r.get("score", r["iou"])))
    stype = rows[-1].get("score_type", "iou")
    print(f"\nresults  → {jsonl}  ({len(rows)} rows)")
    print(f"summary  → mean {stype} per model:")
    width = max(len(m) for m in bucket) + 2
    for model in sorted(bucket):
        vals = bucket[model]
        print(f"    {model:<{width}}n={len(vals):<3} mean_{stype}={sum(vals)/len(vals):.3f}")
    total_tok = sum(r.get("total_tokens") or 0 for r in rows)
    total_cost = sum(r.get("cost_usd") or 0.0 for r in rows)
    if total_tok:
        line = f"tokens   → {total_tok:,} total"
        if total_cost:
            line += f"  ·  cost ≈ ${total_cost:.4f}"
        print(line)


def do_run(cfg: dict, args) -> None:
    data_dir = Path(cfg["data_dir"])
    out_dir  = Path(cfg["out_dir"])
    models   = args.model if args.model else list(cfg["models"])
    gp       = gen_params(cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(data_dir)
    if args.records:
        wanted = set(args.records)
        records = [r for r in records if r["record_id"] in wanted]
    if args.limit:
        if args.seed is not None:
            import random
            records = random.Random(args.seed).sample(records, min(args.limit, len(records)))
        else:
            records = records[: args.limit]

    cp = concurrency_params(cfg)
    ap = agentic_params(cfg)
    print(f"config: {args.config}")
    print(f"data:   {data_dir}")
    print(f"out:    {out_dir}")
    print(f"gen:    max_tokens={gp['max_tokens']} timeout={gp['timeout']}s exec_timeout={gp['exec_timeout']}s")
    print(f"conc:   api_workers={cp['api_workers']} score_workers={cp['score_workers']}")
    print(f"mode:   {'agentic (sandbox, %d rounds)' % ap['max_rounds'] if ap['enabled'] else 'single-shot'}")
    print(f"runs:   {len(records)} record(s) × {len(models)} model(s)")

    def one(model, rec):
        return run_record(record=rec, data_dir=data_dir, results_root=out_dir,
                          model=model, score=args.score,
                          max_tokens=gp["max_tokens"], timeout=gp["timeout"],
                          exec_timeout=gp["exec_timeout"],
                          agentic=ap["enabled"], max_rounds=ap["max_rounds"])

    if cp["api_workers"] == 1:
        for model in models:
            for i, rec in enumerate(records, 1):
                print(f"  [{model}] {i}/{len(records)} {rec['record_id']}", end=" ... ", flush=True)
                row = one(model, rec)
                print(f"{row['status']:10s} {row['score_type']}={row['score']:.3f}  ({row['lat_s']:.1f}s)")
        _print_results_summary(out_dir)
        return

    # Concurrent path. Render the ground-truth composites here, on the main
    # thread: VTK aborts the process if a worker builds a render window, and it
    # leaks a graphics context per render. Workers then only hit the disk cache.
    from pipeline import runner as runner_mod
    from pipeline.prompt import build as build_prompt

    import benchcad_core.scoring.views as views
    from benchcad_core import parallel

    def _unrenderable(rec, err):
        print(f"  !! skipping {rec['record_id']}: ground truth will not render "
              f"({type(err).__name__})", flush=True)

    ready = parallel.prerender(records, build_prompt, data_dir, on_error=_unrenderable)
    if len(ready) != len(records):
        print(f"  pre-rendered {len(ready)}/{len(records)} composites "
              f"({len(records) - len(ready)} dropped)", flush=True)

    for model in models:
        def report(done, rec, row, err, _m=model):
            if err is not None:
                print(f"  [{_m}] {done}/{len(ready)} {rec['record_id']} ... "
                      f"FAILED {type(err).__name__}: {str(err)[:80]}", flush=True)
            else:
                print(f"  [{_m}] {done}/{len(ready)} {rec['record_id']} ... "
                      f"{row['status']:10s} {row['score_type']}={row['score']:.3f}  "
                      f"({row['lat_s']:.1f}s)", flush=True)

        # The preview render of each generated STEP is a second VTK call and
        # would abort the process from a worker, so it is queued and drawn here
        # on the main thread once the pool has drained.
        with parallel.serialized_results(runner_mod), \
             parallel.bounded_scoring(runner_mod, cp["score_workers"]), \
             parallel.deferred_previews(views) as pending:
            parallel.map_records(ready, lambda r, _m=model: one(_m, r),
                                 cp["api_workers"], report)
        drawn = parallel.replay_previews(views, pending)
        if drawn:
            print(f"  rendered {drawn} preview image(s) on the main thread", flush=True)
    _print_results_summary(out_dir)


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    out_dir = Path(cfg["out_dir"])

    if args.plot:
        do_plot(out_dir)
        return
    do_run(cfg, args)


if __name__ == "__main__":
    main()
