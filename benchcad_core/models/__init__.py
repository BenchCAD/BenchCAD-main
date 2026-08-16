"""Model dispatcher.

Public:
    call_model(model: str, system: str, user_text: str,
               image_paths: list[Path] | None = None,
               max_tokens: int = 4096, timeout: int = 600) -> Completion

`Completion` is a (text, usage) NamedTuple. `usage` is a dict with keys
prompt_tokens / completion_tokens / reasoning_tokens / total_tokens — any value
may be None if the provider didn't report it.

Dispatches by model id prefix:
    gpt-* / o3 / o1 / o-*       → openai
    claude-*                    → anthropic
    gemini-*                    → gemini
    grok-* / xai/*              → xai
    openrouter/*                → openrouter

Special model-id suffixes (all routed inside the relevant adapter):
    :reasoning=high|medium|low  → reasoning effort (openai / anthropic / openrouter / xai)
    :reasoning=<int>            → reasoning-token budget (openrouter)
    :thinking=off               → gemini, disables thinking
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

# Lazy-load .env once on import.
try:
    from dotenv import load_dotenv
    # Repo root is .../BenchCAD/<task>/models/__init__.py → up 3
    _root = Path(__file__).resolve().parents[2]
    load_dotenv(_root / ".env", override=False)
except ImportError:
    pass


class Turn(NamedTuple):
    """One message in a multi-turn exchange.

    `role` is "user" or "assistant"; `images` are attached to user turns only
    (no provider accepts images on an assistant turn).
    """
    role: str
    text: str
    images: tuple = ()


class Completion(NamedTuple):
    """A model response: generated `text` plus a `usage` token-count dict."""
    text: str
    usage: dict


def usage_dict(prompt=None, completion=None, reasoning=None, total=None) -> dict:
    """Normalize token counts to the common usage schema (missing → None).

    `total` is derived from prompt+completion when the provider didn't report it.
    `reasoning` (when present) is a subset of `completion` — reasoning / thinking
    tokens are billed as output.
    """
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
    }


def usage_from_openai(resp) -> dict:
    """Extract usage from an OpenAI-compatible response (openai + openrouter)."""
    u = getattr(resp, "usage", None)
    if u is None:
        return usage_dict()
    details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
    return usage_dict(
        prompt=getattr(u, "prompt_tokens", None),
        completion=getattr(u, "completion_tokens", None),
        reasoning=reasoning,
        total=getattr(u, "total_tokens", None),
    )


def usage_from_responses(resp) -> dict:
    """Extract usage from a Responses-API response (openai + xai).

    The Responses API names its counts input_tokens / output_tokens, where Chat
    Completions says prompt_tokens / completion_tokens — hence the separate
    reader from `usage_from_openai`.
    """
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


def _route(model: str) -> str:
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith(("gpt-", "o3", "o1")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "gemini"
    if model.startswith("xai/") or model.startswith("grok-"):
        return "xai"
    raise ValueError(f"Unknown model family: {model}")


def call_model(model: str, system: str, user_text: str,
               image_paths: list[Path] | None = None,
               max_tokens: int = 4096, timeout: int = 600,
               turns: list | None = None) -> Completion:
    """Single-shot by default; pass `turns` for a multi-turn exchange.

    `turns` is a list of `Turn`, and when given it replaces `user_text` /
    `image_paths` entirely — the agentic runner needs the model to see its own
    previous programs as its own turns, not as text quoted back at it.
    """
    images = [Path(p) for p in (image_paths or [])]
    provider = _route(model)
    if provider == "openai":
        from .openai_adapter import generate
    elif provider == "anthropic":
        from .anthropic_adapter import generate
    elif provider == "gemini":
        from .gemini_adapter import generate
    elif provider == "xai":
        from .xai_adapter import generate
    else:
        from .openrouter_adapter import generate
    kw = dict(model=model, system=system, user_text=user_text,
              image_paths=images, max_tokens=max_tokens, timeout=timeout)
    if turns is not None:
        import inspect
        if "turns" not in inspect.signature(generate).parameters:
            raise NotImplementedError(
                f"multi-turn is not implemented for the {provider} adapter yet; "
                f"it is available for openai and xai (both Responses API)")
        kw["turns"] = turns
    text, usage = generate(**kw)
    return Completion(text, usage)


def _img_b64(path: Path) -> str:
    """Read a PNG and return base64 string (no header)."""
    import base64
    return base64.b64encode(path.read_bytes()).decode()
