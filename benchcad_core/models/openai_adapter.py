"""OpenAI / o-series adapter.

Chat Completions by default. Model-id suffixes:
    *:reasoning=low|medium|high|xhigh → reasoning_effort (Chat Completions)
    *:reasoning=minimal|max           → routed to the Responses API, the ONLY
                                         endpoint exposing these tiers (Chat
                                         Completions caps at xhigh and 400s on max)
    *-thinking                        → reasoning_effort=high (+ chat-latest variant)
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _img_b64, usage_dict, usage_from_openai

# Effort tiers the Chat Completions `reasoning_effort` param rejects (400); only
# the Responses API `reasoning.effort` accepts them. Full ladder on Responses:
# none < minimal < low < medium < high < xhigh < max.
_RESPONSES_ONLY_EFFORTS = ("minimal", "max")


def _max_tokens_kwarg(model: str, n: int) -> dict:
    key = "max_completion_tokens" if model.startswith(("gpt-5", "o1", "o3")) else "max_tokens"
    return {key: n}


def _supports_temperature(model: str) -> bool:
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
        # Reasoning tokens are billed against max_completion_tokens; floor the
        # budget so a long reasoning pass doesn't consume it all and leave no
        # room for the actual answer.
        max_tokens = max(max_tokens, 32000)

    client = openai.OpenAI(api_key=api_key)
    # OPENAI_FORCE_RESPONSES=1 routes ALL reasoning tiers through the Responses API
    # (for controlled API-vs-API comparisons, e.g. xhigh-on-Responses vs max); by
    # default only minimal/max — which Chat Completions can't do — take that path.
    force_responses = os.environ.get("OPENAI_FORCE_RESPONSES") == "1"
    if reasoning_effort and (force_responses or reasoning_effort in _RESPONSES_ONLY_EFFORTS):
        return _generate_responses(
            client=client, model=real_model, system=system, user_text=user_text,
            image_paths=image_paths, max_tokens=max_tokens, timeout=timeout,
            effort=reasoning_effort,
        )

    user_content: list = [{"type": "text", "text": user_text}]
    for p in image_paths:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_img_b64(p)}", "detail": "high"},
        })
    kwargs: dict = {
        "model": real_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "timeout": timeout,
        **_max_tokens_kwarg(real_model, max_tokens),
    }
    if _supports_temperature(real_model):
        kwargs["temperature"] = 0.0
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    return text, usage_from_openai(resp)


def _generate_responses(*, client, model: str, system: str, user_text: str,
                        image_paths: list[Path], max_tokens: int, timeout: int,
                        effort: str) -> tuple[str, dict]:
    """Responses API path — the only endpoint exposing the `minimal`/`max` effort
    tiers. Same 4-view image + system prompt as the Chat Completions path, but the
    Responses request shape differs (input parts, instructions, max_output_tokens)."""
    content: list = [{"type": "input_text", "text": user_text}]
    for p in image_paths:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{_img_b64(p)}",
            "detail": "high",
        })
    kwargs: dict = {
        "model": model,
        "instructions": system,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": effort},
        "timeout": timeout,
    }
    # effort=max reasons without a fixed budget; matching the model-card eval,
    # leave max_output_tokens UNSET (no cap → model's own limit) so a long
    # reasoning+answer isn't truncated — ~11% of samples exceed 32k output, and
    # truncating them to 0 drags mean IoU down (0.72 → 0.64). Lower tiers keep
    # the token cap.
    if effort != "max":
        kwargs["max_output_tokens"] = max_tokens
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
