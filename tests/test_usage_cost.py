"""Unit test: token-usage normalization + cost estimation. Offline, no network."""

import pytest

from benchcad_core.models import Completion, usage_dict, usage_from_openai
from benchcad_core.models.pricing import base_model, cost_usd


def test_usage_dict_derives_total():
    assert usage_dict(10, 20) == {
        "prompt_tokens": 10, "completion_tokens": 20,
        "reasoning_tokens": None, "total_tokens": 30, "cached_tokens": None,
    }


def test_usage_dict_all_none():
    assert usage_dict() == {
        "prompt_tokens": None, "completion_tokens": None,
        "reasoning_tokens": None, "total_tokens": None, "cached_tokens": None,
    }


def test_usage_dict_keeps_explicit_total():
    assert usage_dict(10, 20, total=99)["total_tokens"] == 99


def test_completion_unpacks_as_tuple():
    text, usage = Completion("hi", usage_dict(1, 2))
    assert text == "hi"
    assert usage["total_tokens"] == 3


class _Details:
    reasoning_tokens = 5


class _Usage:
    prompt_tokens = 100
    completion_tokens = 40
    total_tokens = 140
    completion_tokens_details = _Details()


class _Resp:
    usage = _Usage()


def test_usage_from_openai_extracts_reasoning():
    assert usage_from_openai(_Resp()) == {
        "prompt_tokens": 100, "completion_tokens": 40,
        "reasoning_tokens": 5, "total_tokens": 140, "cached_tokens": None,
    }


def test_usage_from_openai_no_usage():
    class R:
        usage = None
    assert usage_from_openai(R())["total_tokens"] is None


@pytest.mark.parametrize("model,expected", [
    ("gpt-4o", "gpt-4o"),
    ("openrouter/openai/gpt-oss-120b:free", "openai/gpt-oss-120b"),
    ("claude-opus-4-7:reasoning=high", "claude-opus-4-7"),
    ("gpt-5.3-chat-latest-thinking", "gpt-5.3-chat-latest"),
    ("gemini-3-pro:thinking=off", "gemini-3-pro"),
])
def test_base_model_normalization(model, expected):
    assert base_model(model) == expected


def test_cost_known_model():
    # gpt-4o priced 2.50 in / 10.00 out per 1M tokens
    assert cost_usd("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.5)


def test_cost_strips_suffix_before_lookup():
    assert cost_usd("gpt-4o:reasoning=high", 1_000_000, 0) == pytest.approx(2.5)


def test_cost_unknown_model_is_none():
    assert cost_usd("some-unlisted-model", 100, 100) is None


def test_cost_none_tokens_is_none():
    assert cost_usd("gpt-4o", None, None) is None


def test_cost_bills_cached_prompt_tokens_at_the_cache_rate():
    """A multi-turn loop resends its transcript every turn, and most of it is a
    cache hit. Billing that at the input rate overstates the cost of the very
    setting the benchmark is measuring."""
    full = cost_usd("xai/grok-4.5", 212, 21)
    cached = cost_usd("xai/grok-4.5", 212, 21, 128)
    assert cached < full
    assert cached == pytest.approx(0.000332, abs=1e-6)   # live xAI call: $0.000330


def test_cost_without_cache_info_is_the_upper_bound():
    assert cost_usd("xai/grok-4.5", 100, 10, None) == cost_usd("xai/grok-4.5", 100, 10, 0)


def test_cached_cannot_exceed_prompt():
    assert cost_usd("xai/grok-4.5", 50, 0, 999) == cost_usd("xai/grok-4.5", 50, 0, 50)
