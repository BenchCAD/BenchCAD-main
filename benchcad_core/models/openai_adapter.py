"""OpenAI adapter — Responses API for EVERY call (Chat Completions removed).

Model-id suffixes:
    *:reasoning=minimal|low|medium|high|xhigh|max → reasoning.effort (Responses API)
    *-thinking                                    → effort=high (+ chat-latest variant)

Bare model ids (no `:reasoning=` suffix) call Responses without a `reasoning`
param — i.e. the model's own default effort. The Responses API is the only
endpoint exposing the full effort ladder (none < minimal < low < medium < high <
xhigh < max), so routing every tier through it keeps the endpoint from being a
confound in an effort-vs-score comparison.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _img_b64, usage_dict


def _supports_temperature(model: str) -> bool:
    # Reasoning models (gpt-5*, o1/o3) reject any temperature != default; older
    # non-reasoning models still take temperature=0 for determinism.
    return not model.startswith(("o1", "o3", "gpt-5"))


def generate(*, model: str, system: str, user_text: str,
             image_paths: list[Path], max_tokens: int, timeout: int) -> tuple[str, dict]:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in env")

    real_model = model
    reasoning_effort = None
    if ":reasoning=" in model:
        real_model, reasoning_effort = model.split(":reasoning=", 1)
    elif model.endswith("-thinking"):
        base = model[: -len("-thinking")]
        real_model = f"{base}-chat-latest"
        reasoning_effort = "high"
    if reasoning_effort:
        # Reasoning tokens are billed against the output budget; floor it so a
        # long reasoning pass doesn't consume it all and leave no room for the
        # actual answer.
        max_tokens = max(max_tokens, 32000)

    content: list = [{"type": "input_text", "text": user_text}]
    for p in image_paths:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{_img_b64(p)}",
            "detail": "high",
        })
    kwargs: dict = {
        "model": real_model,
        "instructions": system,
        "input": [{"role": "user", "content": content}],
        "timeout": timeout,
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if _supports_temperature(real_model):
        kwargs["temperature"] = 0.0
    # effort=max reasons long. With NO cap, ~14% of B30 records spiral for 1-2h
    # until the connection drops (api_fail) — and they re-spiral on retry, so it's
    # systematic, not transient. A 128000 cap bounds a spiral to ~25min (→ no_code
    # instead of a connection-killing 2h call) while staying high enough not to
    # truncate legit long reasoning (max observed output ~107k). Other tiers keep
    # the config's token budget.
    if reasoning_effort == "max":
        kwargs["max_output_tokens"] = max(max_tokens, 128000)
    else:
        kwargs["max_output_tokens"] = max_tokens

    client = openai.OpenAI(api_key=api_key)
    resp = client.responses.create(**kwargs)
    return (resp.output_text or ""), _usage_from_responses(resp)


def _usage_from_responses(resp) -> dict:
    """Extract usage from a Responses-API response (input/output_tokens naming)."""
    u = getattr(resp, "usage", None)
    if u is None:
        return usage_dict()
    details = getattr(u, "output_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
    return usage_dict(
        prompt=getattr(u, "input_tokens", None),
        completion=getattr(u, "output_tokens", None),
        reasoning=reasoning,
        total=getattr(u, "total_tokens", None),
    )
