#!/usr/bin/env python
"""Validate a contributed *family* against the per-family requirements in
CONTRIBUTING.md (section A). Reuses `validate_task` for every per-part check and
`_taskmods` for execution / IoU — nothing is reimplemented.

    uv run python tools/validate_family.py contributions/<family> [--per-difficulty 3]

Checks
------
  family.json : required keys, base_plane in {XY,XZ,YZ}
  generator   : build(difficulty, seed) -> CadQuery code; sampled parts at each of
                easy/medium/hard execute, render, and are mutually distinct
  qa/         : >= 2 files, each exactly 12 questions, valid type/level
  edits/      : cover all 3 prototype difficulties (>= 2 each); each edit's
                orig_code and gt_code both execute and change the geometry (IoU < 1)
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import _taskmods
import validate_task

_DIFFS = ("easy", "medium", "hard")
_FAMILY_KEYS = ("family", "standard", "base_plane", "description", "source", "contributor")
_CATEGORIES = {"T1", "T2", "T3", "T4", "T5"}
_QA_PER_PART = 12
_LEVELS = {"L1", "L2", "L3", "L4", "L5", "L6"}


def _import_generator(path: Path):
    spec = importlib.util.spec_from_file_location("contrib_generator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_family_json(fam_dir: Path, log: list) -> dict | None:
    p = fam_dir / "family.json"
    if not p.exists():
        log.append((False, "family.json: missing"))
        return None
    meta = json.loads(p.read_text())
    missing = [k for k in _FAMILY_KEYS if k not in meta]
    if missing:
        log.append((False, f"family.json: missing keys {missing}"))
        return None
    if meta["base_plane"] not in validate_task._BASE_PLANE:
        log.append((False, f"family.json: base_plane must be one of {validate_task._BASE_PLANE}"))
        return None
    log.append((True, "family.json ok"))
    return meta


def _check_generator(fam_dir: Path, meta: dict, n: int, log: list) -> None:
    gen_path = fam_dir / "generator.py"
    if not gen_path.exists():
        log.append((False, "generator.py: missing"))
        return
    gen = _import_generator(gen_path)
    if not hasattr(gen, "build"):
        log.append((False, "generator.py: must define build(difficulty, seed)"))
        return

    hashes: set[str] = set()
    with tempfile.TemporaryDirectory() as td:
        for diff in _DIFFS:
            ok_count = 0
            for seed in range(n):
                code = gen.build(diff, seed)
                if not isinstance(code, str):
                    log.append((False, f"build({diff!r}, {seed}) must return CadQuery code (str)"))
                    continue
                part = Path(td) / f"{diff}_{seed}"
                part.mkdir(parents=True, exist_ok=True)
                (part / "part.py").write_text(code)
                (part / "meta.json").write_text(json.dumps({
                    "family": meta["family"], "variant": "standard", "difficulty": diff,
                    "base_plane": meta["base_plane"], "standard": meta["standard"],
                    "source": meta["source"], "contributor": meta["contributor"],
                }))
                with contextlib.redirect_stdout(io.StringIO()):
                    passed, gh = validate_task.validate(part, hashes)
                if passed:
                    ok_count += 1
                    hashes.add(gh)
            log.append((ok_count == n, f"generator {diff}: {ok_count}/{n} sampled parts valid & distinct"))


def _check_qa(fam_dir: Path, log: list) -> None:
    qa_dir = fam_dir / "qa"
    files = sorted(qa_dir.glob("*.json")) if qa_dir.exists() else []
    if len(files) < 2:
        log.append((False, f"qa/: need >= 2 files, found {len(files)}"))
        return
    types: Counter = Counter()
    ok = True
    for f in files:
        items = json.loads(f.read_text())
        if len(items) != _QA_PER_PART:
            log.append((False, f"qa/{f.name}: needs exactly {_QA_PER_PART} questions, has {len(items)}"))
            ok = False
            continue
        for i, q in enumerate(items):
            if not {"question", "answer", "type", "level"} <= set(q):
                log.append((False, f"qa/{f.name}[{i}]: needs question/answer/type/level"))
                ok = False
            elif q["type"] not in validate_task._QA_TYPES:
                log.append((False, f"qa/{f.name}[{i}]: bad type {q['type']!r}"))
                ok = False
            elif q["level"] not in _LEVELS:
                log.append((False, f"qa/{f.name}[{i}]: level must be L1..L6"))
                ok = False
            elif not isinstance(q["answer"], (int, float)):
                log.append((False, f"qa/{f.name}[{i}]: answer must be numeric"))
                ok = False
            else:
                types[q["type"]] += 1
    if ok:
        log.append((True, f"qa/: {len(files)} parts x {_QA_PER_PART}, types {dict(types)}"))


def _check_edits(fam_dir: Path, log: list) -> None:
    edits_dir = fam_dir / "edits"
    files = sorted(edits_dir.glob("*.json")) if edits_dir.exists() else []
    if not files:
        log.append((False, "edits/: none found"))
        return
    exec_cq = _taskmods.exec_cq()
    iou_mod = _taskmods.iou()
    per_diff: Counter = Counter()
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            e = json.loads(f.read_text())
            if not {"prototype_difficulty", "category", "edit_type",
                    "instruction", "orig_code", "gt_code"} <= set(e):
                log.append((False, f"edits/{f.name}: missing required keys"))
                ok = False
                continue
            if e["prototype_difficulty"] not in _DIFFS:
                log.append((False, f"edits/{f.name}: prototype_difficulty must be easy/medium/hard"))
                ok = False
            if e["category"] not in _CATEGORIES:
                log.append((False, f"edits/{f.name}: category must be T1..T5"))
                ok = False
            a, b = Path(td) / f"{f.stem}_o.step", Path(td) / f"{f.stem}_g.step"
            try:
                exec_cq.execute_cq_to_step(e["orig_code"], a)
                exec_cq.execute_cq_to_step(e["gt_code"], b)
                iou = iou_mod.iou_step_vs_step(a, b)
            except Exception as ex:  # noqa: BLE001
                log.append((False, f"edits/{f.name}: orig/gt failed to execute ({str(ex)[:80]})"))
                ok = False
                continue
            if iou >= 1.0:
                log.append((False, f"edits/{f.name}: edit does not change geometry (IoU {iou:.3f})"))
                ok = False
            else:
                per_diff[e["prototype_difficulty"]] += 1
    short = [d for d in _DIFFS if per_diff[d] < 2]
    if short:
        log.append((False, f"edits/: need >= 2 per prototype difficulty; short on {short} (have {dict(per_diff)})"))
        ok = False
    if ok:
        log.append((True, f"edits/: {sum(per_diff.values())} valid, per-difficulty {dict(per_diff)}"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a contributed BenchCAD family.")
    ap.add_argument("family_dir", type=Path)
    ap.add_argument("--per-difficulty", type=int, default=3,
                    help="how many generator seeds to sample per difficulty")
    args = ap.parse_args()

    log: list[tuple[bool, str]] = []
    print(f"Validating family {args.family_dir} ...")
    meta = _check_family_json(args.family_dir, log)
    if meta is not None:
        _check_generator(args.family_dir, meta, args.per_difficulty, log)
    _check_qa(args.family_dir, log)
    _check_edits(args.family_dir, log)

    for passed, msg in log:
        print(f"  [{'PASS' if passed else 'FAIL'}] {msg}")
    ok = all(p for p, _ in log)
    print(f"\n{'FAMILY OK' if ok else 'FAMILY VALIDATION FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
