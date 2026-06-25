"""Offline unit tests for CodeQA numeric scoring. No API keys, no network."""

import importlib.util
import pathlib

# Load CodeQA/scoring/qa_score.py by path under a unique name so it doesn't
# collide with CodeGen's same-named `scoring` package during a shared pytest run.
_path = pathlib.Path(__file__).resolve().parents[1] / "scoring" / "qa_score.py"
_spec = importlib.util.spec_from_file_location("codeqa_qa_score", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_json_numbers = _mod.parse_json_numbers
qa_score = _mod.qa_score
qa_score_single = _mod.qa_score_single


# --- qa_score_single -------------------------------------------------------

def test_exact_match_types():
    # integer/count/boolean require exact equality
    assert qa_score_single(3, 3, "integer") == 1.0
    assert qa_score_single(3, 4, "count") == 0.0
    # boolean 0 == 0 must score 1.0 (ratio accuracy would wrongly give 0)
    assert qa_score_single(0, 0, "boolean") == 1.0
    assert qa_score_single(1, 0, "bool") == 0.0


def test_ratio_accuracy_symmetric():
    # dim/ratio use symmetric min/max ratio
    assert qa_score_single(8, 10, "dim") == 0.8
    assert qa_score_single(10, 8, "dim") == 0.8  # symmetric
    assert qa_score_single(5, 5, "ratio") == 1.0


def test_non_positive_is_zero():
    assert qa_score_single(-1, 10, "dim") == 0.0
    assert qa_score_single(10, 0, "dim") == 0.0


def test_qa_score_mean_and_empty():
    pairs = [{"answer": 10, "type": "dim"}, {"answer": 4, "type": "integer"}]
    # preds: 8 -> 0.8, 4 -> 1.0  => mean 0.9
    assert qa_score([8, 4], pairs) == 0.9
    assert qa_score([1, 2], []) == 0.0  # no pairs -> 0.0


# --- parse_json_numbers ----------------------------------------------------

def test_parse_plain_array():
    assert parse_json_numbers("[1, 2.5, 3]", 3) == [1.0, 2.5, 3.0]


def test_parse_fenced_with_prose():
    raw = "Sure, here are the answers:\n```json\n[4, 5]\n```"
    assert parse_json_numbers(raw, 2) == [4.0, 5.0]


def test_parse_wrong_length_returns_none():
    assert parse_json_numbers("[1, 2, 3]", 2) is None


def test_parse_garbage_returns_none():
    assert parse_json_numbers("no array here", 2) is None
    assert parse_json_numbers("", 1) is None
