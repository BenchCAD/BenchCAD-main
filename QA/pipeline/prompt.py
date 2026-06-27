"""Build (system, user_text, image_paths) for numeric QA over CadQuery parts.

A `record` is one row from records.jsonl with `qa_pairs` (list of
{question, answer, type}) and one or both of:
    - code_path  (relative to data_dir)
    - image_path (relative to data_dir)

Three modes:
    code  : show the CadQuery source only (text-only)
    img   : show one rendered image only (vision-only)
    both  : show source + image (multimodal)
"""

from __future__ import annotations

import json
from pathlib import Path

_RULES = """Rules:
- Output ONLY a JSON array of numbers, one per question, in the same order.
- No text, no keys, no explanation. Just the array.
- For yes/no questions, use 1 for yes and 0 for no.
- For count questions, use an integer (e.g. 12, not "twelve").
- For ratio questions, use a decimal (e.g. 2.5).
- For dimensional questions, answer in millimetres."""

_EXAMPLE = """Example questions: ["How many teeth?", "What is the module?"]
Example output: [20, 2.5]"""

SYSTEM_BY_MODE = {
    "code": "You are an expert CAD engineer. You will be shown CadQuery Python code "
            "for a mechanical part and a list of numeric questions about the part it "
            "produces.\n\n" + _RULES + "\n\n" + _EXAMPLE,
    "img":  "You are an expert CAD engineer. You will be shown rendered views of a "
            "mechanical part and a list of numeric questions about that part.\n\n"
            + _RULES + "\n\n" + _EXAMPLE,
    "both": "You are an expert CAD engineer. You will be shown CadQuery Python code "
            "AND rendered views of the mechanical part it produces, plus a list of "
            "numeric questions. Use either source as needed.\n\n"
            + _RULES + "\n\n" + _EXAMPLE,
}


def build(record: dict, data_dir: Path,
          mode: str = "code") -> tuple[str, str, list[Path]]:
    if mode not in SYSTEM_BY_MODE:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(SYSTEM_BY_MODE)}")

    questions = [q["question"] for q in record["qa_pairs"]]
    q_block = (
        "\n\nQuestions (answer each with a single number, JSON array):\n"
        + json.dumps(questions)
    )

    parts: list[str] = []
    images: list[Path] = []

    if mode in ("code", "both"):
        cp = record.get("code_path")
        if not cp:
            raise ValueError(f"record {record.get('record_id')!r} has no code_path for mode={mode!r}")
        code = (data_dir / cp).read_text(errors="ignore")
        parts.append("CadQuery code:\n```python\n" + code + "\n```")

    if mode in ("img", "both"):
        ip = record.get("image_path")
        if not ip:
            raise ValueError(f"record {record.get('record_id')!r} has no image_path for mode={mode!r}")
        img = data_dir / ip
        if not img.exists():
            raise FileNotFoundError(f"image not found: {img}")
        images.append(img)
        parts.append("Rendered views of the part are attached as image(s).")

    user_text = "\n\n".join(parts) + q_block
    return SYSTEM_BY_MODE[mode], user_text, images
