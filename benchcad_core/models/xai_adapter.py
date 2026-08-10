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
# floors for their reasoning paths. Sized from the measured distribution of a
# high-effort image->CadQuery workload: successful calls had a median latency of
# ~1190 s and a maximum of ~2520 s. A floor much below that cuts off requests
# that are working -- and because the openai SDK then retries, a call killed
# early costs 3x the timeout and frequently still fails. An earlier revision
# used 900 s, chosen to "bound the worst case" from a 20-record sample where
# retries happened to rescue every call; over 299 records that margin vanished
# and most requests were being truncated mid-flight.
#
# The SDK's default `max_retries=2` is deliberately left alone: the latency tail
# is stochastic, so a re-run of an identical request has a real chance of
# completing, and any concurrent run will meet 429s. Bound the worst case by
# sizing this floor to the work, not by removing the retries.
_MIN_TIMEOUT = 3000

# Sampling temperature. Other adapters send 0.0 for determinism, which is worth
# little here anyway — reasoning length varies run to run regardless, so an
# identical request is not reproducible even at 0. 0.7 is the value the endpoint
# reports when none is sent, i.e. the provider's own default; it is passed
# explicitly so the run is pinned to a known value rather than to whatever the
# default becomes later.
_TEMPERATURE = 0.7

# Sent as a backstop, not a budget. Measured behaviour: a value near the model's
# working range acts as a reasoning target rather than a ceiling (a request
# capped at 2000 returned 5360 tokens), so a binding cap distorts the run. This
# is set far above anything observed (max ~33k) so it never binds, while still
# keeping the request bounded. Verified not to affect the timeout behaviour:
# 512000, 16000 and no cap at all fail identically when the endpoint is
# degraded, so this is not a reliability knob.
_MAX_OUTPUT_TOKENS = 512_000


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
    """Whether to send `temperature` at all.

    xAI's docs say temperature is "not supported by grok-3 and reasoning
    models", but that is not what the API does — a reasoning model verified
    against the live endpoint accepted an explicit temperature and echoed it
    back. It is therefore sent to everything except grok-3, the one family the
    docs name concretely, and `generate()` retries without it if a model
    disagrees.
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

    kwargs: dict = {
        "model": real_model,
        "instructions": system,
        "input": [{"role": "user", "content": content}],
        # A ceiling far above anything the model emits, so it never shapes the
        # answer, but the request is still bounded rather than open-ended. The
        # run config's `max_tokens` is not forwarded — see the module docstring.
        "max_output_tokens": _MAX_OUTPUT_TOKENS,
        "timeout": timeout,
    }
    if effort is not None:
        kwargs["reasoning"] = {"effort": effort}
    if supports_temperature(real_model):
        kwargs["temperature"] = _TEMPERATURE

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
