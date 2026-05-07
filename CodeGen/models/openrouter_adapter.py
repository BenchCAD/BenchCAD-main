"""OpenRouter adapter (OpenAI-compatible endpoint).

Model id form: `openrouter/<provider>/<model>[:free]`, e.g.
    openrouter/openai/gpt-oss-120b:free
    openrouter/nvidia/nemotron-3-super-120b-a12b:free
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _img_b64


def generate(*, model: str, system: str, user_text: str,
             image_paths: list[Path], max_tokens: int, timeout: int) -> str:
    import openai

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in env")

    real_model = model[len("openrouter/"):]
    user_content: list = [{"type": "text", "text": user_text}]
    for p in image_paths:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_img_b64(p)}"},
        })

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout,
    )
    resp = client.chat.completions.create(
        model=real_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""
