"""Resolve the optional `gen:` config block shared by all task runners.

A task config may include a `gen:` mapping to tune model calls and execution:

    gen:
      max_tokens: 16000    # per-call output/token budget
      timeout: 3600        # per model-call timeout, seconds
      exec_timeout: 300    # CadQuery -> STEP execution timeout, seconds

All keys are optional; omitted keys fall back to the defaults below. OpenAI
reasoning models additionally floor `max_tokens` inside their adapter so a long
reasoning pass doesn't starve the output.
"""

from __future__ import annotations

# Per-call output budget. 16k comfortably holds a CadQuery program plus a short
# preamble; reasoning models floor it higher inside their adapters.
DEFAULT_MAX_TOKENS = 16000
# Per model-call timeout, seconds. Generous vs the raw 120s API default so a slow
# reasoning response isn't cut off; prod configs raise it further (up to ~1 hr).
DEFAULT_TIMEOUT = 600
# CadQuery -> STEP execution timeout (subprocess), seconds. Matches the scorer.
DEFAULT_EXEC_TIMEOUT = 300


def gen_params(cfg: dict) -> dict:
    """Return `{max_tokens, timeout, exec_timeout}` from `cfg['gen']` (with defaults)."""
    gen = cfg.get("gen") or {}
    if not isinstance(gen, dict):
        raise SystemExit(f"config `gen:` must be a mapping, got {type(gen).__name__}")
    return {
        "max_tokens":   int(gen.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "timeout":      int(gen.get("timeout", DEFAULT_TIMEOUT)),
        "exec_timeout": int(gen.get("exec_timeout", DEFAULT_EXEC_TIMEOUT)),
    }
