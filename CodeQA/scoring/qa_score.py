"""Numeric QA scoring — type-aware.

Per-pair score depends on `type`:
    integer / count / boolean / bool   → exact match  (1.0 if pred == gt else 0.0)
    dim / ratio / anything else        → symmetric ratio accuracy
                                         min(pred, gt) / max(pred, gt),
                                         0.0 if either non-positive

Exact match is required for boolean (0/1) answers, where ratio accuracy
collapses to 0 even when pred == gt == 0. It also matches the natural
semantics of counts and line numbers (off-by-one is wrong, not "92% right").

Also includes a JSON-array parser robust to leading prose / fenced blocks.
"""

from __future__ import annotations

import json
import re

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", flags=re.M)
_FENCE_CLOSE = re.compile(r"```\s*$", flags=re.M)
_ARRAY = re.compile(r"\[[^\[\]]*\]", re.S)

_EXACT_TYPES = {"integer", "count", "boolean", "bool"}


def parse_json_numbers(raw: str, n_expected: int) -> list[float] | None:
    """Extract a JSON array of n_expected numbers. Returns None on mismatch."""
    s = (raw or "").strip()
    s = _FENCE_OPEN.sub("", s)
    s = _FENCE_CLOSE.sub("", s).strip()
    m = _ARRAY.search(s)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(arr, list) or len(arr) != n_expected:
        return None
    out: list[float] = []
    for x in arr:
        try:
            out.append(float(x))
        except Exception:
            return None
    return out


def qa_score_single(pred: float, gt: float, type_: str = "dim") -> float:
    """Score one (pred, gt) given its question `type_`."""
    pred = float(pred)
    gt = float(gt)
    if (type_ or "").lower() in _EXACT_TYPES:
        return 1.0 if pred == gt else 0.0
    if gt <= 0 or pred <= 0:
        return 0.0
    return round(min(pred, gt) / max(pred, gt), 4)


def qa_score(pred_answers: list[float], qa_pairs: list[dict]) -> float:
    """Mean per-pair score across all QA pairs. 0.0 if no pairs."""
    if not qa_pairs:
        return 0.0
    scores = [qa_score_single(p, q["answer"], q.get("type", "dim"))
              for p, q in zip(pred_answers, qa_pairs)]
    return round(sum(scores) / len(scores), 4)
