"""OpenRouter adapter (OpenAI-compatible endpoint).

Model id form: `openrouter/<provider>/<model>[:free][:reasoning=<spec>]`, e.g.
    openrouter/openai/gpt-oss-120b:free
    openrouter/nvidia/nemotron-3-super-120b-a12b:free
    openrouter/deepseek/deepseek-r1:reasoning=high

The optional `:reasoning=<spec>` suffix maps to OpenRouter's unified `reasoning`
request field (https://openrouter.ai/docs/use-cases/reasoning-tokens):
    high | medium | low   → {"effort": <level>}
    <integer>             → {"max_tokens": <n>}   (reasoning-token budget)
    off | none | 0        → {"enabled": false}
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _img_b64


def _reasoning_extra_body(real_model: str) -> tuple[str, dict]:
    """Split a trailing `:reasoning=<spec>` off the model slug.

    Returns `(model_without_suffix, extra_body)`, where `extra_body` is `{}` or
    `{"reasoning": {...}}` for OpenRouter's unified reasoning API. Everything
    before the suffix — including the `:free` variant tag — is preserved.
    """
    if ":reasoning=" not in real_model:
        return real_model, {}
    base, spec = real_model.rsplit(":reasoning=", 1)
    spec = spec.strip().lower()
    if spec in ("high", "medium", "low"):
        return base, {"reasoning": {"effort": spec}}
    if spec in ("off", "none", "0"):
        return base, {"reasoning": {"enabled": False}}
    if spec.isdigit():
        return base, {"reasoning": {"max_tokens": int(spec)}}
    raise ValueError(
        f"bad :reasoning= spec {spec!r} for {base!r} (want high|medium|low|off|<int>)"
    )


def generate(*, model: str, system: str, user_text: str,
             image_paths: list[Path], max_tokens: int, timeout: int) -> str:
    import openai

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in env")

    real_model, extra_body = _reasoning_extra_body(model[len("openrouter/"):])
    if extra_body:
        # A reasoning pass can run well past the 120s call_model default; floor the
        # request timeout so a slow reasoning response isn't cut off (mirrors the
        # anthropic adapter's extended-thinking floor).
        timeout = max(timeout, 600)

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
        extra_body=extra_body or None,
    )
    return resp.choices[0].message.content or ""
