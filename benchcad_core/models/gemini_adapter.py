"""Google Gemini adapter (google-genai SDK).

Handles `:thinking=off` suffix → sets thinking_budget=0.
"""

from __future__ import annotations

import os
from pathlib import Path


def generate(*, model: str, system: str, user_text: str,
             image_paths: list[Path], max_tokens: int, timeout: int) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in env")

    real_model = model
    thinking_budget: int | None = None
    if model.endswith(":thinking=off"):
        real_model = model[: -len(":thinking=off")]
        thinking_budget = 0

    parts = [types.Part.from_text(text=user_text)]
    for p in image_paths:
        parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type="image/png"))

    config_kwargs: dict = {
        "system_instruction": system,
        # Gemini's max_output_tokens INCLUDES thinking tokens, so add the thinking
        # budget on top to leave the full max_tokens available for the code output.
        "max_output_tokens": max_tokens + (thinking_budget or 0),
        "temperature": 0.0,
    }
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    # Gemini otherwise ignores timeout entirely — a hung connection would stall
    # the eval loop indefinitely. HttpOptions.timeout is in ms; floor at 5 min.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=max(timeout, 300) * 1000),
    )
    resp = client.models.generate_content(
        model=real_model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return resp.text or ""
