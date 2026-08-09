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

**`max_tokens` is not forwarded at all.** On every other provider it is a
ceiling; here it behaves as a reasoning *target*, and sending one makes runs
slower rather than safer. Measured on one image→CadQuery record at high effort:

    max_output_tokens=16000  ->  32395 output tokens (32158 reasoning), 398 s
    max_output_tokens absent ->  10089 output tokens (10022 reasoning), 134 s

Both returned `status=completed` with a complete program, so the cap is not
bounding anything — note the first row overshoots its own cap 2x. It only
bounds the visible answer, which is a few dozen tokens and never approaches it.
Sending a large value to "remove the limit" is the worst case: it inflates
reasoning threefold and triples wall-clock for no gain.

The per-call timeout is therefore the only effective bound, and the floor below
is the one real safeguard.
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

# Floor for the per-call timeout, mirroring the anthropic/openrouter adapters'
# floors for their reasoning paths. A normal high-effort call on a hard task
# measured 4-5 minutes, but the same request re-run can take far longer — the
# latency tail is stochastic, not a property of the prompt. 900 s leaves ample
# headroom over the normal case while keeping a stalled call from running away.
#
# The openai SDK's default `max_retries=2` is deliberately left alone. Retries
# are worth keeping here precisely because the tail *is* stochastic (a re-run of
# an identical request has a real chance of completing), and because a run at
# any concurrency will meet 429s, which is exactly what retries are for. That
# does mean a hopeless call costs up to 3x this timeout, which is the reason the
# floor is 900 s and not an hour: bound the worst case here, not in the retries.
_MIN_TIMEOUT = 900


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
    timeout = max(timeout, _MIN_TIMEOUT)

    content: list = [{"type": "input_text", "text": user_text}]
    for p in image_paths:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{_img_b64(p)}",
            "detail": "high",
        })

    # `max_tokens` is deliberately not forwarded — see the module docstring.
    kwargs: dict = {
        "model": real_model,
        "instructions": system,
        "input": [{"role": "user", "content": content}],
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
