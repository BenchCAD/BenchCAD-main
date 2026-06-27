"""Anthropic / Claude adapter.

Handles `:reasoning=high` suffix → enables extended thinking.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _img_b64


def generate(*, model: str, system: str, user_text: str,
             image_paths: list[Path], max_tokens: int, timeout: int) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")

    real_model = model
    thinking = None
    if ":reasoning=" in model:
        real_model, level = model.split(":reasoning=", 1)
        if level in ("high", "medium"):
            thinking = {"type": "enabled", "budget_tokens": 8000 if level == "high" else 4000}
            # Extended thinking is drawn from max_tokens; floor it so the thinking
            # budget doesn't starve the final code block. Anthropic requires
            # max_tokens > budget_tokens, with room left over for the output.
            max_tokens = max(max_tokens, 16000)
            # Extended thinking can take ~2 min; floor the request timeout so a
            # slow thinking response isn't cut off (call_model default is 120s).
            timeout = max(timeout, 600)

    content: list = [{"type": "text", "text": user_text}]
    for p in image_paths:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _img_b64(p)},
        })

    kwargs: dict = {
        "model": real_model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": content}],
        "timeout": timeout,
    }
    if thinking:
        kwargs["thinking"] = thinking

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(**kwargs)
    # Concatenate text blocks
    parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    return "".join(parts)
