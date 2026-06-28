"""Build per-record (system, user_text, image_paths) for the code-edit task.

A `record` is one row from records.jsonl, paths in the record are relative to
`data_dir`.

Returns a 3-tuple consumable by `models.call_model`:
    (system, user_text, image_paths)
where image_paths is a list of PNG paths (always empty: text-instruction task).
"""

from __future__ import annotations

from pathlib import Path

SYSTEM_PROMPT = """You are an expert CAD engineer. You will be given:
1. A CadQuery Python script that builds a parametric mechanical part.
2. A natural-language edit instruction describing a change to one specific feature.

Your task: return the script with the MINIMAL edit that achieves the requested feature change. Only the feature(s) the instruction explicitly mentions should change; every other feature, dimension, and detail must stay exactly as in the original.

Rules:
- Output ONLY executable Python code, no explanation, no markdown fences.
- Make the smallest set of changes that realize the instruction. Do NOT touch unrelated literals, operations, or structure.
- Do NOT refactor, rename variables, reorder operations, or add/remove imports.
- If the instruction adds a new feature: append the operation in the most natural place; keep the rest of the chain intact.
- If the instruction removes a feature: delete just that operation block; do not rewrite surrounding code.
- If the instruction modifies a dimension/parameter: update only the literal(s) that depend on it (and, if present, the matching parameter comment)."""


def build(record: dict, data_dir: Path, mode: str = "instruction") -> tuple[str, str, list[Path]]:
    if mode != "instruction":
        raise ValueError(f"Unknown mode: {mode}")
    orig_code = _read(data_dir / record["orig_code_path"])
    user = (
        f"Instruction: {record['instruction']}\n\n"
        f"Original code:\n```python\n{orig_code}\n```"
    )
    return SYSTEM_PROMPT, user, []


def _read(path: Path) -> str:
    return path.read_text(errors="ignore")
