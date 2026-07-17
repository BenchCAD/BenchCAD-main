"""Model dispatcher.

Public:
    call_model(model: str, system: str, user_text: str,
               image_paths: list[Path] | None = None,
               max_tokens: int = 4096, timeout: int = 120) -> str

Dispatches by model id prefix:
    gpt-* / o3 / o1 / o-*       → openai
    claude-*                    → anthropic
    gemini-*                    → gemini
    openrouter/*                → openrouter

Special model-id suffixes (all routed inside the relevant adapter):
    :reasoning=high|medium|low  → reasoning effort (openai / anthropic / openrouter)
    :reasoning=<int>            → reasoning-token budget (openrouter)
    :thinking=off               → gemini, disables thinking
"""

from __future__ import annotations

from pathlib import Path

# Lazy-load .env once on import.
try:
    from dotenv import load_dotenv
    # Repo root is .../BenchCAD/<task>/models/__init__.py → up 3
    _root = Path(__file__).resolve().parents[2]
    load_dotenv(_root / ".env", override=False)
except ImportError:
    pass


def _route(model: str) -> str:
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith(("gpt-", "o3", "o1")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "gemini"
    raise ValueError(f"Unknown model family: {model}")


def call_model(model: str, system: str, user_text: str,
               image_paths: list[Path] | None = None,
               max_tokens: int = 4096, timeout: int = 120) -> str:
    images = [Path(p) for p in (image_paths or [])]
    provider = _route(model)
    if provider == "openai":
        from .openai_adapter import generate
    elif provider == "anthropic":
        from .anthropic_adapter import generate
    elif provider == "gemini":
        from .gemini_adapter import generate
    else:
        from .openrouter_adapter import generate
    return generate(model=model, system=system, user_text=user_text,
                    image_paths=images, max_tokens=max_tokens, timeout=timeout)


def _img_b64(path: Path) -> str:
    """Read a PNG and return base64 string (no header)."""
    import base64
    return base64.b64encode(path.read_bytes()).decode()
