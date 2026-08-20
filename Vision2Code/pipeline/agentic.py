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

import json
import time
from pathlib import Path

from benchcad_core.models import Turn, call_model
from benchcad_core.sandbox import Sandbox
from benchcad_core.scoring.exec_cq import execute_cq_to_step

# The measured working length, not a guess. Given 100 rounds the model stops on
# its own at a median of 70 and only 4 records in 50 reach the cap; given 10 it
# hits the cap 98% of the time, so every number collected at 10 was measuring
# "what it can do in ten rounds", not what it can do. Paired over 50 records,
# 10 -> 100 is +0.266 IoU (95% CI [+0.176, +0.360], 43 wins to 7) at roughly 60x
# the tokens. See configs/agentic.yaml for the cost curve and cheaper settings.
MAX_ROUNDS = 100
# One dead round discards every round before it, since they share a conversation,
# so a round is worth retrying harder than a single-shot call would be.
CALL_ATTEMPTS = 5
NUDGE_BUDGET = 5          # extra calls a run may spend recovering the format

_NUDGE = ("You replied without calling a tool, so nothing ran and the round was "
          "not spent. Call run_python to investigate, or submit when you have "
          "the geometry you want scored.")

TOOLS = [
    {"type": "function", "name": "run_python",
     "description": ("Run Python in the working directory and get back stdout, "
                     "stderr and any images it wrote. cadquery, numpy and PIL "
                     "are importable, tools.py is on the path, and files you "
                     "write persist into the next call."),
     "parameters": {"type": "object", "additionalProperties": False,
                    "required": ["code"],
                    "properties": {"code": {
                        "type": "string",
                        "description": "The program to run."}}}},
    {"type": "function", "name": "submit",
     "description": ("Submit the CadQuery program for the geometry you judge "
                     "best. This ends the episode and is what gets scored, so "
                     "call it once and last."),
     "parameters": {"type": "object", "additionalProperties": False,
                    "required": ["code"],
                    "properties": {"code": {
                        "type": "string",
                        "description": ("A complete CadQuery program leaving the "
                                        "final solid in `result`.")}}}},
]

RETRY_BACKOFF_S = 5

# Per-execution wall clock. 300s was censoring real work rather than catching
# runaways: 10 of 353 executions in the first half hour of a run hit it, each
# on a different record and each on a program that was building geometry, not
# looping forever. A round lost that way is not fatal -- the model is told
# "timeout after 300s" and continues -- but it is a round spent on nothing.
# The same value bounds the final scoring execution, so a submitted program
# slower than the cap scored zero for being slow.
EXEC_TIMEOUT = 600

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

You are working in a directory over {rounds} rounds, not answering in one shot.

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

Work through the run_python tool. A round is one run_python call, and the
directory persists between them. When you have the geometry you want scored,
call submit; that ends the episode.
"""


def _arg_code(call) -> str | None:
    """The `code` argument of a tool call, or None if the model mangled it.

    A schema constrains what the model is asked for, not what arrives: a
    truncated generation can still deliver unparseable JSON, and that is a
    malformed call rather than a decision to stop.
    """
    try:
        code = json.loads(call.arguments or "{}").get("code")
    except (ValueError, AttributeError):
        return None
    return code if isinstance(code, str) and code.strip() else None


def _keep_raw(box, n: int, raw: str, calls=()) -> None:
    """Persist what the model actually returned, before anything interprets it.

    The sandbox only archives a call that ran code, so the calls worth
    diagnosing left nothing behind at all. That made the largest failure in the
    text-fence version of this harness unreadable for weeks; the same blind
    spot would hide a malformed tool call just as well.
    """
    try:
        d = box.log_dir / "raw"               # flat: a call is not a round
        d.mkdir(parents=True, exist_ok=True)
        (d / f"call_{n:03d}.json").write_text(json.dumps(
            {"text": raw or "",
             "tool_calls": [{"name": c.name, "arguments": c.arguments}
                            for c in calls]},
            ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass                                  # diagnosis must never end a run


def _observation(rnd: int, res, max_rounds: int) -> str:
    """The tool result text. No truncation here -- the sandbox already bounds it."""
    parts = [f"Round {rnd}/{max_rounds} — exit {res.returncode}"]
    if res.stdout.strip():
        parts.append(f"stdout:\n{res.stdout.strip()}")
    if res.stderr.strip():
        parts.append(f"stderr:\n{res.stderr.strip()}")
    if res.images:
        parts.append(f"images produced: {', '.join(p.name for p in res.images)}"
                     f" (attached to the next message)")
    elif res.ok:
        parts.append("no images were written")
    if rnd >= max_rounds:
        parts.append("That was the last round. Call submit now.")
    return "\n\n".join(parts)


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
                max_rounds: int = MAX_ROUNDS,
                exec_timeout: int = EXEC_TIMEOUT) -> dict:
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

    rnd, calls = 0, 0
    while rnd < max_rounds and calls < max_rounds + NUDGE_BUDGET:
        calls += 1
        raw, usage, tcalls, err = "", {}, (), None
        for attempt in range(CALL_ATTEMPTS):
            try:
                raw, usage, tcalls = call_model(
                    model=model, system=system + suffix, user_text="",
                    max_tokens=max_tokens, timeout=timeout, turns=turns,
                    tools=TOOLS)
                err = None
                break
            except Exception as e:                    # noqa: BLE001 - retried
                err = f"{type(e).__name__}: {e}"
                if attempt < CALL_ATTEMPTS - 1:
                    time.sleep(RETRY_BACKOFF_S * 2 ** attempt)
        if err is not None:
            rounds.append({"round": rnd + 1, "action": "api_fail", "err": err})
            break

        for k in usage_total:
            usage_total[k] += (usage.get(k) or 0)
        _keep_raw(box, calls, raw, tcalls)
        turns.append(Turn("assistant", raw or "", (), tuple(tcalls)))

        if not tcalls:
            # Not a round. The model produced no work, so charging it a round
            # would spend the budget on the protocol rather than the task.
            rounds.append({"round": rnd + 1, "action": "no_call"})
            turns.append(Turn("user", _NUDGE, ()))
            continue

        # Every call the model made needs a result, or the next request is
        # rejected for an unanswered call -- including calls we do not act on.
        images: list = []
        done = False
        for call in tcalls:
            code = _arg_code(call)
            if code is None:
                turns.append(Turn("tool", "That call had no usable `code` "
                                          "argument. Send it again.", (),
                                  (), call.call_id))
                rounds.append({"round": rnd + 1, "action": "bad_args"})
                continue
            if call.name == "submit":
                submitted = code
                rounds.append({"round": rnd + 1, "action": "submit"})
                turns.append(Turn("tool", "submitted", (), (), call.call_id))
                done = True
                continue
            if done:
                turns.append(Turn("tool", "not run: the episode ended with "
                                          "submit.", (), (), call.call_id))
                continue
            rnd += 1
            res = box.run(code, timeout=exec_timeout)
            rounds.append({"round": rnd, "action": "exec",
                           "returncode": res.returncode,
                           "n_images": len(res.images)})
            turns.append(Turn("tool", _observation(rnd, res, max_rounds), (),
                              (), call.call_id))
            images.extend(res.images[:3])
        if done:
            break
        # Images cannot ride on a tool result in this API, so they follow as a
        # user turn. Without this the model is told an image exists and never
        # sees it, which is the whole point of the sandbox undone.
        if images:
            turns.append(Turn("user", "Images from that call:", tuple(images[:3])))

    # The rounds are for investigation; answering is separate. A model that
    # spends its whole budget measuring has done the right thing and must still
    # be asked for the program, or the episode scores whatever happened to be on
    # disk. One extra call, only when nothing was submitted.
    if not submitted and rounds and rounds[-1]["action"] != "api_fail":
        turns.append(Turn("user",
            "Rounds are finished. Call submit now with the complete CadQuery "
            "program for your best geometry, leaving the final solid in "
            "`result`.", ()))
        for attempt in range(CALL_ATTEMPTS):
            try:
                raw, usage, tcalls = call_model(
                    model=model, system=system + suffix, user_text="",
                    max_tokens=max_tokens, timeout=timeout, turns=turns,
                    tools=TOOLS)
                for k in usage_total:
                    usage_total[k] += (usage.get(k) or 0)
                _keep_raw(box, calls + 1, raw, tcalls)
                for call in tcalls:
                    code = _arg_code(call)
                    if code and call.name == "submit":
                        submitted = code
                        rounds.append({"round": len(rounds) + 1,
                                       "action": "submit_final"})
                        break
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
