"""Run per-record work concurrently without multiplying peak memory.

The two halves of a record cost wildly different amounts of memory. A model call
is a socket plus a base64 image — a few MB, and it spends minutes blocked on the
network. Scoring spawns a CadQuery/OCP subprocess (~0.5 GB resident) and
voxelises two meshes. Sizing a single pool for both is a trap in either
direction: sized for scoring, the API sits idle; sized for the API, the box dies
(64 concurrent scorers want ~25 GB).

So the two are pooled separately. `api_workers` threads issue model calls, and a
semaphore admits only `score_workers` of them into the execute+score section at
a time. A thread waiting on the API holds no scoring slot, which is what lets a
large API pool sit in front of a small scoring pool without deadlocking or
blowing up memory. The ratio is very forgiving in practice — a call takes minutes
while scoring takes seconds, so a couple of scoring slots keep up with dozens of
API workers.

Three further constraints, each learned from a real failure rather than
anticipated:

* **VTK renders only on the main thread.** Its Cocoa backend aborts the whole
  process with an NSException if a worker builds a render window, and it leaks a
  graphics context per render, so rendering hundreds in-process exhausts memory.
  `prerender()` therefore warms the on-disk composite cache up front, on the main
  thread, and workers only ever hit that cache.
* **`results.jsonl` is persisted read-modify-write**, which silently loses rows
  under concurrency. `serialized_results()` re-reads and merges under a lock.
* **A worker must not raise.** One unscoreable record should cost that record,
  not the batch.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any


@contextmanager
def bounded_scoring(runner, slots: int):
    """Admit at most `slots` threads into execution + IoU at once.

    Patches the names bound in the runner module rather than their defining
    modules, because the runner imports them at module load.
    """
    if slots < 1:
        raise ValueError(f"score_workers must be >= 1, got {slots}")
    sem = threading.BoundedSemaphore(slots)
    original = {name: getattr(runner, name)
                for name in ("execute_cq_to_step", "iou_step_vs_step")
                if hasattr(runner, name)}

    def guard(fn):
        def wrapper(*a, **k):
            with sem:
                return fn(*a, **k)
        return wrapper

    for name, fn in original.items():
        setattr(runner, name, guard(fn))
    try:
        yield
    finally:
        for name, fn in original.items():
            setattr(runner, name, fn)


@contextmanager
def serialized_results(runner):
    """Make the runner's results.jsonl persistence safe under threads.

    `run_record` reads every row, inserts one, and writes the file back. Two
    threads interleaving that lose whichever row was written in between, so the
    write re-reads and merges while holding a lock.
    """
    lock = threading.Lock()
    read, write = runner._read_results, runner._write_results

    def safe_write(jsonl, rows):
        with lock:
            merged = read(jsonl)
            merged.update(rows)
            write(jsonl, merged)

    runner._write_results = safe_write
    try:
        yield
    finally:
        runner._write_results = write


def prerender(records: Iterable[dict], build_prompt: Callable, data_dir,
              on_error: Callable[[dict, Exception], None] | None = None) -> list[dict]:
    """Warm the on-disk prompt/render cache on the main thread.

    Returns the records that rendered successfully. A record whose ground truth
    cannot be tessellated is dropped here rather than failing inside a worker —
    that is a data defect, and it would otherwise look like a model failure.
    """
    ok = []
    for rec in records:
        try:
            build_prompt(rec, data_dir)
            ok.append(rec)
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            if on_error:
                on_error(rec, e)
    return ok


def map_records(records: Sequence[dict], work: Callable[[dict], Any],
                workers: int, on_result: Callable[[int, dict, Any, Exception | None], None]
                ) -> list[Any]:
    """Run `work` over `records` with `workers` threads, reporting as each lands.

    Results arrive in completion order, not submission order — with reasoning
    models the spread is large, so waiting for order would stall reporting behind
    the slowest record. Exceptions are handed to `on_result` instead of
    propagating, so one bad record cannot end the batch.
    """
    out: list[Any] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, rec): rec for rec in records}
        for done, future in enumerate(as_completed(futures), 1):
            rec = futures[future]
            try:
                result = future.result()
                out.append(result)
                on_result(done, rec, result, None)
            except Exception as e:  # noqa: BLE001 - one record, not the batch
                on_result(done, rec, None, e)
    return out
