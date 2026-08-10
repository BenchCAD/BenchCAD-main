"""Resolve the optional `gen:` config block shared by all task runners.

A task config may include a `gen:` mapping to tune model calls and execution:

    gen:
      max_tokens: 16000    # per-call output/token budget
      timeout: 3600        # per model-call timeout, seconds
      exec_timeout: 300    # CadQuery -> STEP execution timeout, seconds

    concurrency:
      api_workers: 8       # concurrent model calls (cheap: a socket each)
      score_workers: 2     # concurrent CadQuery runs (~0.5 GB each)

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

# Concurrent model calls. 1 keeps the historical sequential behaviour; raise it
# to overlap the minutes each reasoning call spends blocked on the network.
DEFAULT_API_WORKERS = 1
# Concurrent CadQuery executions. Deliberately small and separate from
# api_workers: each one is a ~0.5 GB OCP subprocess, so this is the knob that
# bounds peak memory. A handful keeps up with dozens of API workers, because
# scoring takes seconds where a call takes minutes.
DEFAULT_SCORE_WORKERS = 2


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


def concurrency_params(cfg: dict) -> dict:
    """Return `{api_workers, score_workers}` from `cfg['concurrency']`.

    These are separate pools on purpose. A model call costs a socket and blocks
    for minutes; a scoring run spawns a ~0.5 GB CadQuery subprocess and finishes
    in seconds. Tying them together either starves the API or exhausts memory —
    64 concurrent scorers need roughly 25 GB.
    """
    con = cfg.get("concurrency") or {}
    if not isinstance(con, dict):
        raise SystemExit(
            f"config `concurrency:` must be a mapping, got {type(con).__name__}")
    api = int(con.get("api_workers", DEFAULT_API_WORKERS))
    score = int(con.get("score_workers", DEFAULT_SCORE_WORKERS))
    if api < 1 or score < 1:
        raise SystemExit(
            f"concurrency workers must be >= 1 (got api_workers={api}, score_workers={score})")
    return {"api_workers": api, "score_workers": score}
