"""Unit tests: xAI (Grok) routing, `:reasoning=` parsing, temperature policy.

Pure — no network and no `openai` import (that happens inside `generate()`).
"""

import pytest

from benchcad_core.models import _route
from benchcad_core.models.pricing import base_model, cost_usd
from benchcad_core.models.xai_adapter import split_effort, supports_temperature


@pytest.mark.parametrize("model", [
    "grok-4.5",
    "grok-4.3",
    "grok-4.20-0309-reasoning",
    "grok-4.5:reasoning=low",
    "xai/some-slug",                  # models not named grok-* need the prefix
    "xai/some-slug:reasoning=high",
])
def test_grok_ids_route_to_xai(model):
    assert _route(model) == "xai"


def test_xai_prefix_is_stripped_from_the_slug_sent_upstream():
    assert split_effort("xai/some-slug") == ("some-slug", None)
    assert split_effort("xai/some-slug:reasoning=low") == ("some-slug", "low")


def test_openrouter_hosted_grok_still_routes_to_openrouter():
    # The openrouter prefix wins — it is checked before any family prefix.
    assert _route("openrouter/x-ai/grok-4.5") == "openrouter"


def test_no_suffix_means_provider_default_not_no_reasoning():
    assert split_effort("grok-4.5") == ("grok-4.5", None)


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh"])
def test_effort_levels(effort):
    assert split_effort(f"grok-4.5:reasoning={effort}") == ("grok-4.5", effort)


def test_effort_is_case_insensitive():
    assert split_effort("grok-4.5:reasoning=HIGH") == ("grok-4.5", "high")


def test_dotted_and_dashed_ids_survive_the_split():
    assert split_effort("grok-4.20-multi-agent-0309:reasoning=xhigh") == (
        "grok-4.20-multi-agent-0309",
        "xhigh",
    )


@pytest.mark.parametrize("spec", ["turbo", "max", "minimal", "off", "2048"])
def test_bad_effort_raises_rather_than_reaching_the_api(spec):
    # OpenAI's ladder has `minimal`/`max` and OpenRouter takes `off` and integer
    # budgets; neither is valid at xAI, and a 400 mid-run is worse than a parse
    # error before it.
    with pytest.raises(ValueError):
        split_effort(f"grok-4.5:reasoning={spec}")


def test_temperature_sent_except_to_grok_3():
    # Verified live: a reasoning model echoed back `temperature: 0.0`, despite
    # the docs saying reasoning models reject it. Only grok-3, the family the
    # docs name concretely, is withheld from.
    assert supports_temperature("some-slug")
    assert supports_temperature("grok-4.5")
    assert not supports_temperature("grok-3")
    assert not supports_temperature("grok-3-mini")


def test_pricing_key_strips_the_reasoning_suffix():
    assert base_model("grok-4.5:reasoning=high") == "grok-4.5"


# --- the request body itself -------------------------------------------------
# No XAI_API_KEY here, so `generate()` is exercised against a stubbed client.
# This is what the parser tests can't show: that the call we would put on the
# wire is the one xAI's Responses API documents.

_LAST_SEEN: dict = {}


class _FakeUsage:
    input_tokens, output_tokens, total_tokens = 11, 22, 33
    output_tokens_details = type("D", (), {"reasoning_tokens": 7})()


class _FakeResp:
    output_text = "hello"
    usage = _FakeUsage()


def _capture(monkeypatch, tmp_path, model, _key_env="XAI_API_KEY", **overrides):
    """Run generate() against a stub client; return (client_kwargs, request)."""
    import openai

    from benchcad_core.models import xai_adapter

    seen = {}

    class _FakeStream:
        def __init__(self, kw):
            seen["request"] = kw
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __iter__(self): return iter(())          # keepalive frames, nothing to read
        def get_final_response(self): return _FakeResp()

    class _FakeClient:
        def __init__(self, **kw):
            seen["client"] = kw
            self.responses = self

        def with_options(self, **kw):                 # timeout is applied here now
            seen.setdefault("options", {}).update(kw)
            return self

        def stream(self, **kw):
            return _FakeStream(kw)

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv(_key_env, "test-key")
    kwargs = dict(model=model, system="SYS", user_text="USER", image_paths=[],
                  max_tokens=4096, timeout=600)
    kwargs.update(overrides)
    text, usage = xai_adapter.generate(**kwargs)
    assert text == "hello"
    assert usage == {"prompt_tokens": 11, "completion_tokens": 22,
                     "reasoning_tokens": 7, "total_tokens": 33}
    _LAST_SEEN.clear(); _LAST_SEEN.update(seen)
    return seen["client"], seen["request"]


def test_request_targets_the_xai_endpoint(monkeypatch, tmp_path):
    client, _ = _capture(monkeypatch, tmp_path, "grok-4.5")
    assert client["base_url"] == "https://api.x.ai/v1"
    assert client["api_key"] == "test-key"


def test_missing_key_fails_before_any_call(monkeypatch):
    from benchcad_core.models import xai_adapter

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        xai_adapter.generate(model="grok-4.5", system="s", user_text="u",
                             image_paths=[], max_tokens=10, timeout=1)


def test_grok_api_key_is_accepted_as_a_fallback(monkeypatch, tmp_path):
    client, _ = _capture(monkeypatch, tmp_path, "grok-4.5", _key_env="GROK_API_KEY")
    assert client["api_key"] == "test-key"


def test_system_prompt_goes_to_instructions_not_a_system_turn(monkeypatch, tmp_path):
    _, req = _capture(monkeypatch, tmp_path, "grok-4.5")
    assert req["instructions"] == "SYS"
    assert [m["role"] for m in req["input"]] == ["user"]


def test_bare_id_sends_no_reasoning_param(monkeypatch, tmp_path):
    _, req = _capture(monkeypatch, tmp_path, "grok-4.5")
    assert "reasoning" not in req


@pytest.mark.parametrize("model", ["grok-4.5", "grok-4.5:reasoning=high",
                                   "grok-4.5:reasoning=none"])
def test_config_max_tokens_is_not_forwarded(monkeypatch, tmp_path, model):
    """A cap near the model's working range acts as a reasoning target rather
    than a ceiling (a request capped at 2000 returned 5360 tokens), so the run
    config's budget is not passed through. A non-binding backstop is sent
    instead, far above anything observed."""
    _, req = _capture(monkeypatch, tmp_path, model, max_tokens=4096)
    assert req["max_output_tokens"] == 512_000


@pytest.mark.parametrize("timeout", [450, 600, 3600])
def test_configured_timeout_is_passed_through_unclamped(monkeypatch, tmp_path, timeout):
    """Earlier revisions floored this to 900s and then 3000s, silently
    overriding the caller. Both floors were derived from latencies that included
    SDK retries; a real successful call takes 280-410s, so clamping only made
    dead calls cost 3x the floor. On the streaming path the timeout is per-read
    and the server's keepalive resets it, so it no longer truncates a long pass."""
    _capture(monkeypatch, tmp_path, "grok-4.5", timeout=timeout)
    assert _LAST_SEEN["options"]["timeout"] == timeout


def test_temperature_sent_by_default(monkeypatch, tmp_path):
    _, req = _capture(monkeypatch, tmp_path, "xai/some-slug")
    assert req["temperature"] == 0.7


def test_temperature_omitted_for_grok_3(monkeypatch, tmp_path):
    _, req = _capture(monkeypatch, tmp_path, "grok-3")
    assert "temperature" not in req


def test_temperature_rejection_is_retried_once_without_it(monkeypatch, tmp_path):
    """A model that rejects `temperature` must not take the whole run down."""
    import openai

    from benchcad_core.models import xai_adapter

    attempts = []

    class _FakeClient:
        def __init__(self, **kw):
            self.responses = self

        def with_options(self, **kw): return self

        def stream(self, **kw):
            attempts.append(kw)
            if "temperature" in kw:
                raise openai.BadRequestError(
                    "temperature is not supported by this model",
                    response=_httpx_response(), body=None,
                )
            return _ok_stream()

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    monkeypatch.setenv("XAI_API_KEY", "k")
    text, _ = xai_adapter.generate(model="grok-4.5", system="s", user_text="u",
                                   image_paths=[], max_tokens=10, timeout=1)
    assert text == "hello"
    assert len(attempts) == 2
    assert "temperature" in attempts[0] and "temperature" not in attempts[1]


def test_unrelated_bad_request_is_not_retried(monkeypatch, tmp_path):
    import openai

    from benchcad_core.models import xai_adapter

    attempts = []

    class _FakeClient:
        def __init__(self, **kw):
            self.responses = self

        def with_options(self, **kw): return self

        def stream(self, **kw):
            attempts.append(kw)
            raise openai.BadRequestError(
                "The model foo does not exist", response=_httpx_response(), body=None,
            )

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    monkeypatch.setenv("XAI_API_KEY", "k")
    with pytest.raises(openai.BadRequestError):
        xai_adapter.generate(model="grok-4.5", system="s", user_text="u",
                             image_paths=[], max_tokens=10, timeout=1)
    assert len(attempts) == 1


class _ok_stream:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter(())
    def get_final_response(self): return _FakeResp()


def _httpx_response():
    import httpx

    return httpx.Response(400, request=httpx.Request("POST", "https://api.x.ai/v1/responses"))


def test_images_are_inlined_as_png_data_urls(monkeypatch, tmp_path):
    img = tmp_path / "views.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake")
    _, req = _capture(monkeypatch, tmp_path, "grok-4.5", image_paths=[img])
    content = req["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "USER"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert content[1]["detail"] == "high"


def test_grok_is_priced():
    # 1M in + 1M out at grok-4.5's listed 2.00 / 6.00.
    assert cost_usd("grok-4.5:reasoning=low", 1_000_000, 1_000_000) == 8.0
