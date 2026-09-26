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

from . import ToolCall, _img_b64, usage_from_openai


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


def _user_content(text: str, image_paths) -> list:
    """Text plus any images, in the chat-completions content-part form."""
    out: list = [{"type": "text", "text": text}]
    for p in image_paths:
        out.append({"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_img_b64(Path(p))}"}})
    return out


def _messages(turns) -> list:
    """Flatten `Turn`s into chat-completions messages.

    Three shapes matter. A user turn may carry images. An assistant turn that
    asked for tools carries them in `tool_calls`, and its content is null rather
    than "" when the model said nothing alongside the call. A tool turn answers
    one call by id -- text only, since no provider accepts an image on a tool
    result; the harness appends returned renders as a following user turn.
    """
    msgs: list = []
    for t in turns:
        if t.role == "tool":
            msgs.append({"role": "tool", "tool_call_id": t.call_id,
                         "content": t.text})
        elif t.role == "assistant":
            m: dict = {"role": "assistant", "content": t.text or None}
            if t.tool_calls:
                m["tool_calls"] = [
                    {"id": c.call_id, "type": "function",
                     "function": {"name": c.name, "arguments": c.arguments}}
                    for c in t.tool_calls]
            msgs.append(m)
        else:
            msgs.append({"role": "user",
                         "content": _user_content(t.text, t.images)
                         if t.images else t.text})
    return msgs


def _tools_param(tools) -> list:
    """The harness states tools in the Responses API's flat form; chat
    completions wants the same fields nested under `function`."""
    out = []
    for t in tools:
        if "function" in t:                       # already nested
            out.append(t)
        else:
            out.append({"type": "function",
                        "function": {k: v for k, v in t.items()
                                     if k in ("name", "description", "parameters")}})
    return out


def generate(*, model: str, system: str, user_text: str,
             image_paths: list[Path], max_tokens: int, timeout: int,
             turns: list | None = None, tools: list | None = None):
    import openai

    # OpenCode Zen is a second OpenAI-compatible gateway with its own key and
    # base url; the request and response shapes are identical, so it rides this
    # adapter rather than earning a near-duplicate one.
    if model.startswith("opencode/"):
        base_url = "https://opencode.ai/zen/v1"
        api_key = os.environ.get("OPENCODE_API_KEY") or os.environ.get("ZEN_API_KEY")
        if not api_key:
            raise RuntimeError("OPENCODE_API_KEY not set in env "
                               "(run `opencode auth login` and export the key)")
        slug = model[len("opencode/"):]
    else:
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in env")
        slug = model[len("openrouter/"):]

    real_model, extra_body = _reasoning_extra_body(slug)
    if extra_body:
        # A reasoning pass can run long; floor the request timeout so a low
        # configured timeout can't cut a slow reasoning response off (mirrors the
        # anthropic adapter's extended-thinking floor).
        timeout = max(timeout, 600)

    messages = [{"role": "system", "content": system}]
    if turns is None:
        messages.append({"role": "user",
                         "content": _user_content(user_text, image_paths)})
    else:
        messages.extend(_messages(turns))

    kwargs = dict(model=real_model, messages=messages, max_tokens=max_tokens,
                  temperature=0.0, extra_body=extra_body or None)
    if tools is not None:
        kwargs["tools"] = _tools_param(tools)
        # Let the model answer in prose when it has nothing to run; forcing a
        # call would turn "I am finished" into a spurious one.
        kwargs["tool_choice"] = "auto"

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    try:
        resp = client.chat.completions.create(**kwargs)
    except openai.BadRequestError as e:
        # Which models reject `temperature` is not reliably documented, and
        # getting it wrong would kill a whole run over a determinism nicety.
        # Drop it and retry once; re-raise anything else untouched.
        if "temperature" not in kwargs or "temperature" not in str(e).lower():
            raise
        kwargs.pop("temperature")
        resp = client.chat.completions.create(**kwargs)

    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = usage_from_openai(resp)
    if tools is None:
        return text, usage
    calls = tuple(
        ToolCall(c.id, c.function.name, c.function.arguments or "{}")
        for c in (choice.message.tool_calls or []))
    # A reply cut off at the token ceiling looks like "the model chose not to
    # call a tool", which the runner would count as a wasted round rather than
    # the truncation it is. Say so instead of failing silently.
    if not calls and getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError(
            f"{real_model} hit the {max_tokens}-token output ceiling before "
            f"emitting a tool call; raise max_tokens for this model")
    return text, usage, calls
