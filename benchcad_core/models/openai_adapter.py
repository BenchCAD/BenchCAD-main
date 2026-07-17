"""OpenAI / o-series adapter (Chat Completions API).

Handles model-id suffixes:
    *:reasoning=high|medium → reasoning_effort
    *-thinking              → reasoning_effort=high (and chat-latest variant)
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _img_b64


def _max_tokens_kwarg(model: str, n: int) -> dict:
    key = "max_completion_tokens" if model.startswith(("gpt-5", "o1", "o3")) else "max_tokens"
    return {key: n}


def _supports_temperature(model: str) -> bool:
    return not model.startswith(("o1", "o3", "gpt-5"))


def generate(*, model: str, system: str, user_text: str,
             image_paths: list[Path], max_tokens: int, timeout: int) -> str:
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

    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
