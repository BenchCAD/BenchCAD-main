"""Numeric QA scoring — rule iclrNov15.

Per question (pred = parsed model answer, None/NaN if unanswered). Both values
are compared by magnitude: p = |pred|, g = |gt|.
    no answer (None / NaN)   → 0
    g == 0 (e.g. "no")       → 1 if p == 0 else 0
    g > 0, p == 0            → 0
    g > 0, p > 0             → min(p, g) / max(p, g), rounded to 4 dp
        yes/no (gt 1): pred 1 → 1, pred 0 → 0
        count gt 4, pred 3 → 0.75;  ratio gt 3.75, pred 5 → 0.75
        signed sum gt -31.8, pred -31.8 → 1;  pred 31.8 → 1 (the sign is not scored)

The absolute value matters for the 18 bank questions whose gold is a negative
signed sum of extrude/cutBlind depths: without it a correct answer would score 0
and a positive answer would score below 0.

The rule is type-agnostic: `type` is accepted for API compatibility but
ignored. It is the rule behind the paper's qa_img vs qa_code gap table
(named iclrNov15, 2026-11-15).

Also includes a JSON-array parser robust to leading prose / fenced blocks.
"""

from __future__ import annotations

import json
import math
import re

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", flags=re.M)
_FENCE_CLOSE = re.compile(r"```\s*$", flags=re.M)
_ARRAY = re.compile(r"\[[^\[\]]*\]", re.S)

_EPS = 1e-6


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


def qa_score_single(pred: float | None, gt: float, type_: str = "dim") -> float:
    """Score one (pred, gt) under iclrNov15. `type_` is ignored."""
    if pred is None:
        return 0.0
    pred = abs(float(pred))
    gt = abs(float(gt))
    if math.isnan(pred):
        return 0.0
    if gt == 0:
        return 1.0 if pred < _EPS else 0.0
    if pred == 0:
        return 0.0
    return round(min(pred, gt) / max(pred, gt), 4)


def qa_score(pred_answers: list[float], qa_pairs: list[dict]) -> float:
    """Mean per-pair score across all QA pairs. 0.0 if no pairs."""
    if not qa_pairs:
        return 0.0
    scores = [qa_score_single(p, q["answer"], q.get("type", "dim"))
              for p, q in zip(pred_answers, qa_pairs)]
    return round(sum(scores) / len(scores), 4)
