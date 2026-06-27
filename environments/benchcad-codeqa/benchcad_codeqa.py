"""BenchCAD QA — Prime Intellect Environments Hub.

Given rendered views of a mechanical part and one numeric question (a dimension,
count, or ratio), the model answers with a single number. Reward = symmetric
ratio accuracy `min(pred, gt) / max(pred, gt)` for dimensions/ratios, exact match
for counts/integers/yes-no. Deterministic — no judge model, no CAD execution.

Benchmark: https://github.com/BenchCAD/BenchCAD-main  ·  paper: arXiv:2605.10865
Data:      https://huggingface.co/datasets/BenchCAD/BenchCAD  (config `QA`)

The scoring below mirrors the canonical scorer (`QA/scoring/qa_score.py`); it
is vendored so this environment is a self-contained, publishable package.
"""

from __future__ import annotations

import base64
import io
import json
import re

import verifiers as vf
from datasets import Dataset, load_dataset

RULES = (
    "You are an expert CAD engineer. You are shown a 2x2 composite of four diagonal "
    "rendered views of a mechanical part, and one numeric question about it.\n\n"
    "Answer with a SINGLE number and nothing else — no words, no units. For yes/no "
    "questions output 1 for yes and 0 for no. For counts output an integer. For "
    "ratios output a decimal. For dimensions answer in millimetres."
)

# Vendored from BenchCAD/BenchCAD-main QA/scoring/qa_score.py
_EXACT_TYPES = {"integer", "count", "boolean", "bool"}


def _to_number(x) -> float | None:
    """First numeric value in a model answer, or None."""
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x or ""))
    return float(m.group()) if m else None


def _score_one(pred: float | None, gt: float, qa_type: str) -> float:
    if pred is None:
        return 0.0
    if (qa_type or "").lower() in _EXACT_TYPES:
        return 1.0 if pred == gt else 0.0
    if gt <= 0 or pred <= 0:
        return 0.0
    return min(pred, gt) / max(pred, gt)


def _img_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _completion_text(completion) -> str:
    content = completion[-1]["content"] if completion else ""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def load_environment(num_examples: int | None = None, **kwargs) -> vf.Environment:
    """Load the BenchCAD QA (vision) environment.

    Args:
        num_examples: if set, evaluate on the first N questions (quick runs).
    """
    raw = load_dataset("BenchCAD/BenchCAD", "QA", split="QA")
    if num_examples is not None:
        raw = raw.select(range(min(num_examples, len(raw))))

    # One user message whose content is a list (text + image). We do not pass
    # system_prompt: verifiers would prepend a string-content system message, and
    # mixing string- and list-typed content in one column breaks Arrow.
    items = []
    for r in raw:
        qa = json.loads(r["qa"])
        items.append({
            "prompt": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{RULES}\n\nQuestion: {qa['question']}"},
                    {"type": "image_url", "image_url": {"url": _img_to_data_url(r["composite_png"])}},
                ],
            }],
            "answer": str(qa["answer"]),
            "info": {"qa_type": qa.get("type", "dim"), "level": qa.get("level"),
                     "family": r["family"], "standard": r["standard"]},
        })
    ds = Dataset.from_list(items)

    async def ratio_reward(completion, answer, info, **_) -> float:
        pred = _to_number(_completion_text(completion))
        return _score_one(pred, float(answer), info.get("qa_type", "dim"))

    rubric = vf.Rubric(funcs=[ratio_reward])
    return vf.SingleTurnEnv(dataset=ds, rubric=rubric, **kwargs)
