"""Unit tests for the concurrent record runner (`benchcad_core.parallel`).

Every case here corresponds to a failure that actually happened while running a
299-record sweep, not a hypothetical: peak memory blowing up because scoring
inherited the API pool size, rows vanishing from results.jsonl under threads, a
worker exception ending the batch, and a record whose ground truth cannot be
rendered being mistaken for a model failure.

Pure — no network, no CadQuery, no VTK.
"""

import json
import threading
import time
import types

import pytest

from benchcad_core import parallel
from benchcad_core.run_config import concurrency_params

# --- config ------------------------------------------------------------------

def test_defaults_preserve_sequential_behaviour():
    # api_workers=1 must stay the default so existing configs are unchanged.
    assert concurrency_params({}) == {"api_workers": 1, "score_workers": 2}


def test_concurrency_block_is_read():
    cfg = {"concurrency": {"api_workers": 16, "score_workers": 3}}
    assert concurrency_params(cfg) == {"api_workers": 16, "score_workers": 3}


@pytest.mark.parametrize("bad", [{"api_workers": 0}, {"score_workers": 0}])
def test_zero_workers_is_rejected(bad):
    with pytest.raises(SystemExit):
        concurrency_params({"concurrency": bad})


def test_non_mapping_concurrency_is_rejected():
    with pytest.raises(SystemExit):
        concurrency_params({"concurrency": [1, 2]})


# --- the memory bound --------------------------------------------------------

def _runner_stub():
    """A stand-in for pipeline.runner with the two heavy calls it binds."""
    mod = types.SimpleNamespace()
    mod.peak = 0
    mod.live = 0
    mod.lock = threading.Lock()

    def heavy(*_a, **_k):
        with mod.lock:
            mod.live += 1
            mod.peak = max(mod.peak, mod.live)
        time.sleep(0.02)
        with mod.lock:
            mod.live -= 1

    mod.execute_cq_to_step = heavy
    mod.iou_step_vs_step = heavy
    return mod


def test_scoring_is_capped_independently_of_api_workers():
    """The whole point: 32 API threads must not mean 32 CadQuery subprocesses.

    Unbounded, this is what needs ~25 GB and takes the machine down.
    """
    runner = _runner_stub()
    with parallel.bounded_scoring(runner, 2):
        parallel.map_records(
            [{"record_id": str(i)} for i in range(32)],
            lambda r: runner.execute_cq_to_step(),
            workers=32,
            on_result=lambda *a: None,
        )
    assert runner.peak <= 2, f"scoring reached {runner.peak} concurrent, cap was 2"


def test_scoring_patches_are_reverted():
    runner = _runner_stub()
    before = runner.execute_cq_to_step
    with parallel.bounded_scoring(runner, 1):
        assert runner.execute_cq_to_step is not before
    assert runner.execute_cq_to_step is before


def test_invalid_slot_count_rejected():
    with pytest.raises(ValueError):
        with parallel.bounded_scoring(_runner_stub(), 0):
            pass


# --- results.jsonl under threads ---------------------------------------------

def test_concurrent_writes_do_not_lose_rows(tmp_path):
    """run_record persists read-modify-write; interleaving drops rows."""
    jsonl = tmp_path / "results.jsonl"

    def read(path):
        if not path.exists():
            return {}
        return {(r["model"], r["record_id"]): r
                for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}

    def write(path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))

    runner = types.SimpleNamespace(_read_results=read, _write_results=write)

    with parallel.serialized_results(runner):
        def persist(rec):
            rows = read(jsonl)                    # the racy read, as run_record does it
            time.sleep(0.001)                     # widen the window
            rows[("m", rec["record_id"])] = {"model": "m", "record_id": rec["record_id"]}
            runner._write_results(jsonl, rows)

        parallel.map_records([{"record_id": str(i)} for i in range(40)],
                             persist, workers=16, on_result=lambda *a: None)

    assert len(read(jsonl)) == 40, "rows were lost to interleaved read-modify-write"


# --- worker isolation ---------------------------------------------------------

def test_one_failing_record_does_not_kill_the_batch():
    seen = []

    def work(rec):
        if rec["record_id"] == "3":
            raise RuntimeError("boom")
        return {"ok": rec["record_id"]}

    parallel.map_records([{"record_id": str(i)} for i in range(6)], work, workers=4,
                         on_result=lambda done, rec, res, err: seen.append((rec["record_id"], err)))
    assert len(seen) == 6, "every record must be reported"
    assert sum(1 for _, err in seen if err is not None) == 1
    assert next(err for rid, err in seen if rid == "3") is not None


# --- unrenderable ground truth ------------------------------------------------

def test_prerender_drops_unrenderable_records_and_reports_them():
    """A GT STEP that will not tessellate is a data defect. Dropping it here
    keeps it from surfacing later as if the model had failed."""
    records = [{"record_id": "good1"}, {"record_id": "bad"}, {"record_id": "good2"}]
    reported = []

    def build(rec, _data_dir):
        if rec["record_id"] == "bad":
            raise AttributeError("'NoneType' object has no attribute 'NbNodes'")

    ready = parallel.prerender(records, build, "data",
                              on_error=lambda rec, e: reported.append(rec["record_id"]))
    assert [r["record_id"] for r in ready] == ["good1", "good2"]
    assert reported == ["bad"]


def test_prerender_runs_on_the_calling_thread():
    """VTK's Cocoa backend aborts the process if a render happens off the main
    thread, so this must never be moved into the worker pool."""
    calls = []
    parallel.prerender([{"record_id": "a"}, {"record_id": "b"}],
                       lambda rec, _d: calls.append(threading.current_thread().name),
                       "data")
    assert set(calls) == {threading.current_thread().name}
