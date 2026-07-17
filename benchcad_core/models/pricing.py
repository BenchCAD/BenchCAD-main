"""Estimate model-call cost (USD) from token usage.

Prices live in `pricing.yaml` (USD per 1M tokens, input/output) so a model can be
added or corrected without a code change. A model that isn't listed → cost None
(recorded as `cost_usd: null`). Reasoning / thinking tokens are billed as output,
so they're already inside `completion_tokens` and need no separate term.
"""

from __future__ import annotations

import functools
from pathlib import Path

_PRICING_PATH = Path(__file__).with_name("pricing.yaml")


@functools.lru_cache(maxsize=1)
def _prices() -> dict:
    if not _PRICING_PATH.exists():
        return {}
    import yaml
    return yaml.safe_load(_PRICING_PATH.read_text()) or {}


def base_model(model: str) -> str:
    """Strip the routing prefix and behavior suffixes to a pricing-table key.

    `openrouter/openai/gpt-oss-120b:free` → `openai/gpt-oss-120b`
    `claude-opus-4-7:reasoning=high`      → `claude-opus-4-7`
    """
    m = model
    if m.startswith("openrouter/"):
        m = m[len("openrouter/"):]
    for sep in (":reasoning=", ":thinking="):
        m = m.split(sep, 1)[0]
    for suffix in (":free", "-thinking"):
        if m.endswith(suffix):
            m = m[: -len(suffix)]
    return m


def cost_usd(model: str, prompt_tokens, completion_tokens) -> float | None:
    """USD cost for one call, or None if token counts or the price are unknown."""
    if prompt_tokens is None and completion_tokens is None:
        return None
    price = _prices().get(base_model(model))
    if not price:
        return None
    pin = float(price.get("input", 0.0))
    pout = float(price.get("output", 0.0))
    return round((prompt_tokens or 0) / 1e6 * pin + (completion_tokens or 0) / 1e6 * pout, 6)
