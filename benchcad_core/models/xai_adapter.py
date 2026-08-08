"""xAI (Grok) adapter — Responses API at https://api.x.ai/v1.

Model id form: `grok-*` or `xai/<slug>`, either with an optional
`:reasoning=<effort>` suffix:
    grok-4.5
    grok-4.5:reasoning=low
    xai/<slug>                     # any slug not named grok-*

`grok-*` covers the published line. The `xai/` prefix is the escape hatch: not
every model xAI serves is named `grok-something`, and a key may be scoped by ACL
to a subset of them, so routing on the `grok-` prefix alone would send a valid
model to the wrong provider. Anything after `xai/` is passed through verbatim.

xAI serves an OpenAI-shaped `/v1/responses`, and its own docs use the OpenAI SDK
pointed at `base_url="https://api.x.ai/v1"` — so this is the openai adapter with
a different key, base URL and effort ladder, not a separate protocol.

Reads `XAI_API_KEY` (xAI's own convention), falling back to `GROK_API_KEY` —
both names are in circulation and a key under the wrong one is an unhelpful way
to lose a run.

**The configured output budget is passed through unchanged**, which matters more
here than for other providers. An xAI reasoning model at high effort can spend
essentially the whole output budget on reasoning — raising `max_output_tokens`
raises wall-clock roughly proportionally, rather than only raising a ceiling
that is usually not reached. At an observed ~36 tok/s that puts a 32000-token
budget around 15 minutes per call, past the per-call timeout the run configs
set; the openai SDK then retries a timeout twice, so the record burns ~3x that
before failing, with no log line in between.

An earlier version of this adapter floored the budget to keep reasoning from
starving the answer. Measurement says that was the wrong trade in both
directions: the API does not truncate the answer to fit — a request capped at
2000 output tokens returned 2047 tokens and a complete program — so the floor
bought nothing, while doubling wall-clock and overriding a budget the run config
had set deliberately. Size `gen.max_tokens` and `gen.timeout` together.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _img_b64, usage_from_responses

_BASE_URL = "https://api.x.ai/v1"

# `none` < `low` < `medium` < `high` (grok-4.5 default), plus `xhigh` which
# grok-4.20-multi-agent reads as an agent count rather than a depth. Update this
# when xAI adds a tier — the ladder has moved more than once.
_EFFORTS = ("none", "low", "medium", "high", "xhigh")

# Measured throughput, used only to explain the timeout guidance above. At
# effort=high, wall-clock per call is roughly max_output_tokens / this.
_TOKENS_PER_SEC = 36


def split_effort(model: str) -> tuple[str, str | None]:
    """Strip the optional `xai/` prefix and a trailing `:reasoning=<effort>`.

    Returns `(slug, effort)` with `effort` None when the suffix is absent —
    which means the model's own default, not "no reasoning".
    """
    if model.startswith("xai/"):
        model = model[len("xai/"):]
    if ":reasoning=" not in model:
        return model, None
    base, effort = model.rsplit(":reasoning=", 1)
    effort = effort.strip().lower()
    if effort not in _EFFORTS:
        raise ValueError(
            f"bad :reasoning= effort {effort!r} for {base!r} (want {'|'.join(_EFFORTS)})"
        )
    return base, effort


def supports_temperature(model: str) -> bool:
    """Whether to send `temperature=0.0` for determinism.

    xAI's docs say temperature is "not supported by grok-3 and reasoning
    models", but that is not what the API does — a reasoning model verified
    against the live endpoint accepted `temperature: 0` and echoed it back.
    Since a benchmark wants determinism wherever it can get it, temperature is
    sent by default and withheld only from grok-3, the one family the docs name
    concretely. `generate()` retries without it if a model disagrees.
    """
    return not model.startswith("grok-3")


def generate(*, model: str, system: str, user_text: str,
             image_paths: list[Path], max_tokens: int, timeout: int) -> tuple[str, dict]:
    import openai

    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set in env")

    real_model, effort = split_effort(model)

    content: list = [{"type": "input_text", "text": user_text}]
    for p in image_paths:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{_img_b64(p)}",
            "detail": "high",
        })

    kwargs: dict = {
        "model": real_model,
        "instructions": system,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
        "timeout": timeout,
    }
    if effort is not None:
        kwargs["reasoning"] = {"effort": effort}
    if supports_temperature(real_model):
        kwargs["temperature"] = 0.0

    client = openai.OpenAI(api_key=api_key, base_url=_BASE_URL)
    try:
        resp = client.responses.create(**kwargs)
    except openai.BadRequestError as e:
        # Which models reject `temperature` is not reliably documented, and
        # getting it wrong would kill a whole run over a determinism nicety.
        # Drop it and retry once; re-raise anything else untouched.
        if "temperature" not in kwargs or "temperature" not in str(e).lower():
            raise
        kwargs.pop("temperature")
        resp = client.responses.create(**kwargs)
    return (resp.output_text or ""), usage_from_responses(resp)
