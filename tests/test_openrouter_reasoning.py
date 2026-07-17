"""Unit test: OpenRouter `:reasoning=` suffix → unified `reasoning` request body.

Pure parser only — no network, no `openai` import (that happens inside
`generate()`, not at module import).
"""

import pytest

from benchcad_core.models.openrouter_adapter import _reasoning_extra_body


def test_no_suffix_is_noop():
    assert _reasoning_extra_body("openai/gpt-oss-120b") == ("openai/gpt-oss-120b", {})


def test_free_variant_preserved_without_reasoning():
    assert _reasoning_extra_body("openai/gpt-oss-120b:free") == (
        "openai/gpt-oss-120b:free",
        {},
    )


@pytest.mark.parametrize("level", ["high", "medium", "low"])
def test_effort_levels(level):
    assert _reasoning_extra_body(f"deepseek/deepseek-r1:reasoning={level}") == (
        "deepseek/deepseek-r1",
        {"reasoning": {"effort": level}},
    )


def test_effort_alongside_free_variant():
    # The `:free` tag must survive; only the trailing `:reasoning=` is stripped.
    assert _reasoning_extra_body("openai/gpt-oss-120b:free:reasoning=high") == (
        "openai/gpt-oss-120b:free",
        {"reasoning": {"effort": "high"}},
    )


def test_integer_token_budget():
    assert _reasoning_extra_body("x/y:reasoning=2048") == (
        "x/y",
        {"reasoning": {"max_tokens": 2048}},
    )


@pytest.mark.parametrize("spec", ["off", "none", "0"])
def test_disable(spec):
    assert _reasoning_extra_body(f"x/y:reasoning={spec}") == (
        "x/y",
        {"reasoning": {"enabled": False}},
    )


def test_case_insensitive():
    assert _reasoning_extra_body("x/y:reasoning=HIGH") == (
        "x/y",
        {"reasoning": {"effort": "high"}},
    )


def test_bad_spec_raises():
    with pytest.raises(ValueError):
        _reasoning_extra_body("x/y:reasoning=turbo")
