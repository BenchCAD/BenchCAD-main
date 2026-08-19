"""Vision2Code with a Python sandbox, following Anthropic's tool ablation.

Single-shot Vision2Code asks the model to read four rendered views and emit a
CadQuery program blind. This gives it a working directory instead: the target
views are files it can open, it can render its own solid through the same
renderer that produced them, crop either image to check a detail, and compare
them numerically before committing.

Their reported effect on a 1,000-file subset, five runs averaged, max effort:
Mythos 5 went 0.379 -> 0.650 with tools, Mythos Preview 0.356 -> 0.610. The
mechanism is measurement rather than inspection — a model that can only look at
a 268px render fixes orientation and gross proportion, while one that can crop
and diff can chase dimensions.

Shape of the loop: the model writes a ```python block, the sandbox runs it, and
the model gets back stdout, stderr, and any images the code produced. It submits
by emitting a ```cadquery block, which is what gets scored — through the same
`execute_cq_to_step` + `iou_step_vs_step` as a single-shot run, so the two
settings stay comparable.

Orchestrated turns rather than native tool-calling, because provider tool
schemas differ and this benchmark exists to compare providers; the exchange uses
only text and images, which all of them support identically.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from benchcad_core.models import Turn, call_model
from benchcad_core.sandbox import Sandbox
from benchcad_core.scoring.exec_cq import execute_cq_to_step

MAX_ROUNDS = 10
# One dead round discards every round before it, since they share a conversation,
# so a round is worth retrying harder than a single-shot call would be.
CALL_ATTEMPTS = 5
# Backoff between attempts. Without it the three retries fired within a couple of
# seconds of each other and a record died in 17s -- against a transient the
# retries are just the same failure three times, and under concurrency they add
# load to whatever caused it. Doubling from 5s spreads the last attempt ~75s out.
RETRY_BACKOFF_S = 5
_FENCE = re.compile(r"```(python|cadquery)\s*\n(.*?)```", re.DOTALL)

# Deliberately close to the mechanical minimum: what each fenced block does,
# that the directory resets, the round budget, and that the answer is whatever
# geometry the model judges best. An earlier version also prescribed how to
# spend the rounds and argued that diffing a render beats reading the target's
# pixels. That was strategy, not interface, and it steered the model into
# maximising agreement between two silhouettes -- a proxy that is not monotone
# in the voxel IoU actually being scored. One record swept parameters to a
# self-reported 0.60 on its own pixel metric and scored 0.000, down from a
# single-shot 0.696. What the tools are good for is the model's call.
SYSTEM_SUFFIX = """

You are working in a directory, not answering in one shot.

  target.png                 {target_w}x{target_h} — the composite to reproduce
  view_0.png .. view_3.png   {view_w}x{view_h} — its four quadrants, in reading order
  tools.py                   export(result, path) -> STEP
                             render(step, png)    -> {target_w}x{target_h} composite
                             views(png)           -> four quadrants
                             crop(png, box)       -> region

render draws your STEP with the same renderer that drew target.png -- same
projection, same four cameras, same layout, same colour, same size -- so the two
images are directly comparable, pixel for pixel.

cadquery, numpy and PIL are available. There is no network.

```python    runs in the directory; you get back stdout, stderr, and any
             images it wrote
```cadquery  your final answer — it ends the episode and is what gets scored

The directory persists across rounds: files you write are still there next round. You
have {rounds} rounds, and submit the geometry you judge best.
"""


def _blocks(raw: str) -> tuple[str | None, str | None]:
    """Return (python, cadquery) — the last block of each kind, if any."""
    py = cq = None
    for kind, body in _FENCE.findall(raw or ""):
        if kind == "python":
            py = body
        else:
            cq = body
    return py, cq


def _observation(rnd: int, res, max_rounds: int) -> Turn:
    parts = [f"Round {rnd}/{max_rounds} — exit {res.returncode}"]
    if res.stdout.strip():
        parts.append(f"stdout:\n{res.stdout.strip()[:2000]}")
    if res.stderr.strip():
        parts.append(f"stderr:\n{res.stderr.strip()[:2000]}")
    if res.images:
        parts.append(f"images produced: {', '.join(p.name for p in res.images)} (shown below)")
    elif res.ok:
        parts.append("no images were written")
    if rnd >= max_rounds - 1:
        # Without this the episode simply stops: the last round's output was
        # never shown and no turn ever asked for an answer, so a model that
        # budgeted its rounds for investigation was cut off mid-investigation
        # having submitted nothing. Every record in the first working run
        # ended this way.
        parts.append("This is your final observation — no further ```python "
                     "block will be run. Reply now with your ```cadquery "
                     "answer, using the best geometry you have.")
    return Turn("user", "\n\n".join(parts), tuple(res.images[:3]))


def _quadrants(target_png: Path, work_dir: Path) -> dict:
    """Split the composite into view_0..view_3, the files the prompt promises.

    Without these the prompt is describing a directory that does not exist: the
    model opened view_0.png on its first round exactly as told, and lost the
    round to a FileNotFoundError.
    """
    from PIL import Image

    stage = Path(work_dir).parent / (Path(work_dir).name + "_seed")
    stage.mkdir(parents=True, exist_ok=True)
    im = Image.open(target_png)
    w, h = im.size
    boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
             (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
    out = {}
    for i, b in enumerate(boxes):
        p = stage / f"view_{i}.png"
        im.crop(b).save(p)
        out[f"view_{i}.png"] = p
    return out


def run_agentic(*, record: dict, data_dir: Path, work_dir: Path, model: str,
                system: str, user_text: str, target_png: Path,
                max_tokens: int, timeout: int,
                max_rounds: int = MAX_ROUNDS, exec_timeout: int = 300) -> dict:
    """Run one record. Returns the transcript plus the program to score."""
    # tools.render is the same renderer that produced target.png -- same cameras,
    # same 268x268 layout, same palette, verified pixel-identical -- so the model
    # can diff its own render against the target directly. An earlier version
    # rendered with BenchCAD-main's renderer instead, which differs in size and
    # style; the model detected the shape mismatch, skipped its own comparison,
    # and was left optimising blind.
    quads = _quadrants(target_png, Path(work_dir))
    views = {"target.png": target_png, **quads}
    box = Sandbox(work_dir, views, Path(__file__).resolve().parents[2])

    from PIL import Image
    tw, th = Image.open(target_png).size
    # The budget stated in the prompt is the budget actually enforced. It came
    # from the module constant, so a per-call max_rounds of 25 still told the
    # model it had 10 while the observations counted to 25.
    suffix = SYSTEM_SUFFIX.format(target_w=tw, target_h=th,
                                  view_w=tw // 2, view_h=th // 2,
                                  rounds=max_rounds)

    turns = [Turn("user", user_text, (target_png,))]
    rounds: list[dict] = []
    submitted, last_step = "", None
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0,
                   "reasoning_tokens": 0, "total_tokens": 0}

    for rnd in range(1, max_rounds + 1):
        raw, usage, err = "", {}, None
        for attempt in range(CALL_ATTEMPTS):
            try:
                raw, usage = call_model(model=model, system=system + suffix,
                                        user_text="", max_tokens=max_tokens,
                                        timeout=timeout, turns=turns)
                err = None
                break
            except Exception as e:                    # noqa: BLE001 - retried
                err = f"{type(e).__name__}: {e}"
                if attempt < CALL_ATTEMPTS - 1:
                    time.sleep(RETRY_BACKOFF_S * 2 ** attempt)
        if err is not None:
            rounds.append({"round": rnd, "action": "api_fail", "err": err})
            break

        for k in usage_total:
            usage_total[k] += (usage.get(k) or 0)
        turns.append(Turn("assistant", raw or "", ()))
        py, cq = _blocks(raw)

        if cq is not None:
            # No gate. An earlier version refused a submission from an episode
            # that had never rendered, which is a strategy requirement dressed
            # as a protocol one: it forces the model to spend a round producing
            # an image whether or not that is the best use of the round, and it
            # pushed toward the same silhouette-matching proxy the prompt no
            # longer argues for. Whether the tools are worth using is the
            # measurement.
            submitted = cq
            rounds.append({"round": rnd, "action": "submit"})
            break

        if py is None:
            rounds.append({"round": rnd, "action": "no_block"})
            turns.append(Turn("user", "No ```python or ```cadquery block found. "
                                      "Send one.", ()))
            continue

        res = box.run(py, timeout=exec_timeout)
        rounds.append({"round": rnd, "action": "exec", "returncode": res.returncode,
                       "n_images": len(res.images)})
        # Always show the result, including on the last round -- the final round
        # is the model's chance to answer, and it cannot answer from output it
        # was never shown.
        turns.append(_observation(rnd, res, max_rounds))

    # The rounds are for investigation; answering is separate. A model that
    # spends its whole budget measuring has done the right thing and must still
    # be asked for the program, or the episode scores whatever happened to be on
    # disk. One extra call, only when nothing was submitted.
    if not submitted and rounds and rounds[-1]["action"] != "api_fail":
        turns.append(Turn("user",
            "Rounds are finished. Reply with only a ```cadquery block: the "
            "complete program for your best geometry, leaving the final solid "
            "in `result`.", ()))
        for attempt in range(CALL_ATTEMPTS):
            try:
                raw, usage = call_model(model=model, system=system + suffix,
                                        user_text="", max_tokens=max_tokens,
                                        timeout=timeout, turns=turns)
                for k in usage_total:
                    usage_total[k] += (usage.get(k) or 0)
                _, cq = _blocks(raw)
                if cq is not None:
                    submitted = cq
                    rounds.append({"round": len(rounds) + 1, "action": "submit_final"})
                break
            except Exception:                         # noqa: BLE001 - retried
                if attempt < CALL_ATTEMPTS - 1:
                    time.sleep(RETRY_BACKOFF_S * 2 ** attempt)

    # Score the submitted program; fall back to the last STEP the model built,
    # so a run that exhausted its rounds mid-investigation is not scored as zero
    # when it already had working geometry on disk.
    code = submitted
    if code.strip():
        step = Path(work_dir) / "final.step"
        try:
            execute_cq_to_step(code, step, timeout=exec_timeout)
            last_step = step
        except Exception:                             # noqa: BLE001 - reported as status
            last_step = None
    if last_step is None:
        # From the log, which holds every round's geometry in order, so the
        # newest surviving solid is found even if the model overwrote or removed
        # the file it built earlier in the episode.
        cands = box.artifacts(".step")
        last_step = cands[-1] if cands else None

    return {"code": code, "step": last_step, "rounds": rounds,
            "n_rounds": len(rounds), "usage": usage_total,
            "submitted": bool(submitted),
            "final_status": rounds[-1]["action"] if rounds else "no_rounds"}
