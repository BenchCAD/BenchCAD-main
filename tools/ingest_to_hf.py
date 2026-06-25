#!/usr/bin/env python
"""Publish accepted contributions to the HuggingFace dataset (maintainer-only).

Contributors never touch HF — they open a GitHub PR (see CONTRIBUTING.md) and a
maintainer runs this after the PR is merged. It executes each part, renders it
with the official renderer (reused via `_taskmods`, not reimplemented), builds
rows that match the live dataset schema exactly, and — only with --push —
appends them to `BenchCAD/BenchCAD` and tags a new revision.

    # dry run (default): build + validate the rows locally, push nothing
    uv run python tools/ingest_to_hf.py --config code_gen contributions/my_part

    # publish for real
    HF_TOKEN=... uv run python tools/ingest_to_hf.py --config QA --push --tag v1.1 contributions/*

Schemas (verified against the live dataset):
    code_gen: stem, family, variant, difficulty, base_plane, standard, code,
              view_0_png..view_3_png, composite_png
    QA:       stem, family, gt_code, standard, qa, qa_level, qa_id, composite_png
              (one row per QA pair)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import _taskmods

_COLOR = (110, 195, 192)


def _render(views_mod, step: Path, size: int):
    """Reuse the official renderer: 4 single views + their 2x2 composite."""
    verts, tris = views_mod._step_to_normalized_mesh(step)
    color01 = tuple(c / 255.0 for c in _COLOR)
    imgs = [views_mod._render_one_view(verts, tris, f, color01, size)
            for f in views_mod.CAMERA_FRONTS]
    composite = views_mod._composite_2x2(imgs, size_each=size)
    return imgs, composite


def _build_rows(part_dirs, config: str, size: int) -> list[dict]:
    exec_cq = _taskmods.exec_cq()
    views_mod = _taskmods.views()
    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as td:
        for d in part_dirs:
            d = Path(d)
            meta = json.loads((d / "meta.json").read_text())
            code = (d / "part.py").read_text()
            stem = meta.get("stem") or d.name
            standard = meta.get("standard") or ""

            step = Path(td) / f"{stem}.step"
            exec_cq.execute_cq_to_step(code, step)
            views, composite = _render(views_mod, step, size)

            if config == "code_gen":
                rows.append({
                    "stem": stem, "family": meta["family"],
                    "variant": meta.get("variant", ""),
                    "difficulty": meta["difficulty"],
                    "base_plane": meta.get("base_plane", "XY"),
                    "standard": standard, "code": code,
                    "view_0_png": views[0], "view_1_png": views[1],
                    "view_2_png": views[2], "view_3_png": views[3],
                    "composite_png": composite,
                })
            else:  # QA: one row per QA pair
                qa_items = json.loads((d / "qa.json").read_text())
                for item in qa_items:
                    qa_id = hashlib.sha256(
                        f"{stem}|{item['question']}".encode()).hexdigest()[:16]
                    rows.append({
                        "stem": stem, "family": meta["family"], "gt_code": code,
                        "composite_png": composite, "standard": standard,
                        "qa": json.dumps(item, ensure_ascii=False),
                        "qa_level": str(item.get("level", "")), "qa_id": qa_id,
                    })
    return rows


def _features(config: str):
    from datasets import Features, Image, Value
    s = Value("string")
    if config == "code_gen":
        return Features({
            "stem": s, "family": s, "variant": s, "difficulty": s,
            "base_plane": s, "standard": s, "code": s,
            "view_0_png": Image(), "view_1_png": Image(), "view_2_png": Image(),
            "view_3_png": Image(), "composite_png": Image(),
        })
    return Features({
        "stem": s, "family": s, "gt_code": s, "composite_png": Image(),
        "standard": s, "qa": s, "qa_level": s, "qa_id": s,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest accepted parts into the HF dataset.")
    ap.add_argument("parts", nargs="+", type=Path, help="contribution directories")
    ap.add_argument("--config", required=True, choices=["code_gen", "QA"])
    ap.add_argument("--repo", default="BenchCAD/BenchCAD")
    ap.add_argument("--size", type=int, default=256, help="render size per view")
    ap.add_argument("--push", action="store_true", help="actually upload (default: dry run)")
    ap.add_argument("--tag", default=None, help="dataset revision tag to create on --push")
    args = ap.parse_args()

    from datasets import Dataset

    rows = _build_rows(args.parts, args.config, args.size)
    ds = Dataset.from_list(rows, features=_features(args.config))
    print(f"built {ds.num_rows} new rows for config '{args.config}'")
    print("features:", {k: str(v) for k, v in ds.features.items()})
    # round-trip sanity: the first image decodes
    img0 = ds[0]["composite_png"]
    print(f"sample composite image: {img0.size} {img0.mode}")

    if not args.push:
        print("\n[dry run] nothing uploaded. Re-run with --push to publish.")
        return

    from datasets import concatenate_datasets, load_dataset
    from huggingface_hub import HfApi

    split = args.config
    existing = load_dataset(args.repo, args.config, split=split)
    merged = concatenate_datasets([existing, ds])
    print(f"appending: {existing.num_rows} + {ds.num_rows} -> {merged.num_rows}")
    merged.push_to_hub(args.repo, config_name=args.config, split=split)
    if args.tag:
        HfApi().create_tag(args.repo, tag=args.tag, repo_type="dataset")
        print(f"tagged {args.tag}")
    print("pushed.")


if __name__ == "__main__":
    main()
