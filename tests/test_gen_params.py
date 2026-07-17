"""Unit test: the shared `gen:` config resolver (`benchcad_core.run_config`).

Pure — no network, no model imports.
"""

import pytest

from benchcad_core.run_config import (
    DEFAULT_EXEC_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    gen_params,
)


def test_absent_gen_uses_defaults():
    assert gen_params({"models": ["gpt-4o"]}) == {
        "max_tokens": DEFAULT_MAX_TOKENS,
        "timeout": DEFAULT_TIMEOUT,
        "exec_timeout": DEFAULT_EXEC_TIMEOUT,
    }


def test_empty_or_null_gen_uses_defaults():
    assert gen_params({"gen": None})["timeout"] == DEFAULT_TIMEOUT
    assert gen_params({"gen": {}})["max_tokens"] == DEFAULT_MAX_TOKENS


def test_partial_override_keeps_other_defaults():
    assert gen_params({"gen": {"timeout": 3600}}) == {
        "max_tokens": DEFAULT_MAX_TOKENS,
        "timeout": 3600,
        "exec_timeout": DEFAULT_EXEC_TIMEOUT,
    }


def test_full_override():
    assert gen_params(
        {"gen": {"max_tokens": 32000, "timeout": 7200, "exec_timeout": 600}}
    ) == {"max_tokens": 32000, "timeout": 7200, "exec_timeout": 600}


def test_string_values_coerced_to_int():
    got = gen_params({"gen": {"timeout": "3600"}})
    assert got["timeout"] == 3600 and isinstance(got["timeout"], int)


def test_non_mapping_gen_rejected():
    with pytest.raises(SystemExit):
        gen_params({"gen": [1, 2, 3]})
