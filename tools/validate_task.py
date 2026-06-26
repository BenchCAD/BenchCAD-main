#!/usr/bin/env python
"""Automated quality gate for a contributed BenchCAD part.

Reuses the official execution / rendering / mesh code (via `_taskmods`) — it does
not reimplement any of it.

    uv run python tools/validate_task.py contributions/my_part [--known-hashes hashes.txt]

Gates (all hard unless noted):
  1. executable : part.py runs and exports a valid STEP solid          (exec_cq)
  2. renderable : the STEP renders to the 4-view composite             (views)
  3. metadata   : meta.json has required keys; qa.json (if present) is well-formed
  4. dedup      : normalized-geometry hash is not already in --known-hashes (views mesh)

Exit 0 iff every gate passes. The geometry hash (printed on success) is what you
add to the known-hashes set once a part is accepted.

Note: re-deriving each numeric `answer` from the B-rep is done by the datagen QA
generator, not here — this gate checks that answers are present and well-typed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import _taskmods

_REQUIRED_META = ("family", "variant", "difficulty", "base_plane", "standard",
                  "source", "contributor")
_DIFFICULTY = {"easy", "medium", "hard"}
_BASE_PLANE = {"XY", "XZ", "YZ"}
_QA_TYPES = {"integer", "count", "dim", "ratio", "boolean", "bool"}


def _geometry_hash(views_mod, step: Path) -> str:
    """Stable hash of the normalized geometry (sorted, rounded vertices)."""
    import numpy as np

    verts, _ = views_mod._step_to_normalized_mesh(step)
    q = np.round(np.asarray(verts), 3)
    q = q[np.lexsort(q.T[::-1])]  # canonical order, independent of tessellation order
    return hashlib.sha256(q.tobytes()).hexdigest()[:16]


def _check_metadata(part_dir: Path, results: list) -> bool:
    meta_path = part_dir / "meta.json"
    if not meta_path.exists():
        results.append(("metadata", False, "missing meta.json"))
        return False
    meta = json.loads(meta_path.read_text())
    missing = [k for k in _REQUIRED_META if k not in meta]
    if missing:
        results.append(("metadata", False, f"meta.json missing keys: {missing}"))
        return False
    if meta.get("difficulty") not in _DIFFICULTY:
        results.append(("metadata", False, f"difficulty must be one of {_DIFFICULTY}"))
        return False
    if meta.get("base_plane") not in _BASE_PLANE:
        results.append(("metadata", False, f"base_plane must be one of {_BASE_PLANE}"))
        return False

    qa_path = part_dir / "qa.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text())
        if not isinstance(qa, list) or not qa:
            results.append(("metadata", False, "qa.json must be a non-empty list"))
            return False
        for i, item in enumerate(qa):
            if not {"question", "answer", "type"} <= set(item):
                results.append(("metadata", False, f"qa[{i}] needs question/answer/type"))
                return False
            if item["type"] not in _QA_TYPES:
                results.append(("metadata", False, f"qa[{i}] bad type {item['type']!r}"))
                return False
            if not isinstance(item["answer"], (int, float)):
                results.append(("metadata", False, f"qa[{i}] answer must be numeric"))
                return False
    results.append(("metadata", True, "ok"))
    return True


def validate(part_dir: Path, known_hashes: set[str]) -> tuple[bool, str | None]:
    exec_cq = _taskmods.exec_cq()
    views_mod = _taskmods.views()
    results: list[tuple[str, bool, str]] = []

    part_py = part_dir / "part.py"
    geo_hash = None
    if not part_py.exists():
        results.append(("executable", False, "missing part.py"))
    else:
        with tempfile.TemporaryDirectory() as td:
            step = Path(td) / "part.step"
            try:
                exec_cq.execute_cq_to_step(part_py.read_text(), step)
                results.append(("executable", True, "STEP exported"))
            except Exception as e:  # noqa: BLE001 - report any failure as a gate fail
                results.append(("executable", False, str(e)[:160]))
                step = None

            if step and step.exists():
                try:
                    out_png = Path(td) / "composite.png"
                    views_mod.composite_for_step(step, out_png)
                    results.append(("renderable", True, "composite rendered"))
                except Exception as e:  # noqa: BLE001
                    results.append(("renderable", False, str(e)[:160]))
                try:
                    geo_hash = _geometry_hash(views_mod, step)
                    dup = geo_hash in known_hashes
                    results.append(("dedup", not dup,
                                    f"duplicate of an existing part (hash {geo_hash})" if dup
                                    else f"unique (hash {geo_hash})"))
                except Exception as e:  # noqa: BLE001
                    results.append(("dedup", False, str(e)[:160]))

    _check_metadata(part_dir, results)

    ok = all(passed for _, passed, _ in results)
    for name, passed, msg in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {msg}")
    return ok, geo_hash


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a contributed BenchCAD part.")
    ap.add_argument("part_dir", type=Path, help="contributions/<part> directory")
    ap.add_argument("--known-hashes", type=Path, default=None,
                    help="file of accepted geometry hashes (one per line) for dedup")
    args = ap.parse_args()

    known: set[str] = set()
    if args.known_hashes and args.known_hashes.exists():
        known = {ln.strip() for ln in args.known_hashes.read_text().splitlines() if ln.strip()}

    print(f"Validating {args.part_dir} ...")
    ok, geo_hash = validate(args.part_dir, known)
    if ok:
        print(f"\nALL GATES PASS — accepted geometry hash: {geo_hash}")
        sys.exit(0)
    print("\nVALIDATION FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
