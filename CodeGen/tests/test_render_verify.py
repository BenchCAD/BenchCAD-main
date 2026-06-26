"""Loop logic for the render-and-verify eval mode (pipeline.render_verify).

Externals (model call, CadQuery exec, VTK render, voxel IoU) are mocked so the test is fast and
deterministic and exercises only the orchestration: N attempts, keep-only-executing revisions, and
the results.jsonl row shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # CodeGen/ on the path

import pipeline.render_verify as rv  # noqa: E402


def _wire(monkeypatch, tmp_path, responses):
    """Mock build_prompt / call_model / exec / render / iou. `responses` = list of model replies."""
    target = tmp_path / "target.png"; target.write_text("PNG")
    monkeypatch.setattr(rv, "build_prompt", lambda record, data_dir: ("SYS", "USER", [target]))
    monkeypatch.setattr(rv, "iou_step_vs_step", lambda gen, gt: 0.77)

    def fake_exec(code, step_path, timeout=90):       # "BAD" in code → exec failure
        if "BAD" in code:
            raise RuntimeError("exec fail")
        Path(step_path).write_text("STEP")
    monkeypatch.setattr(rv, "execute_cq_to_step", fake_exec)

    calls = []
    it = iter(responses)

    def fake_call(**kw):
        calls.append(kw)
        return next(it)
    import models
    monkeypatch.setattr(models, "call_model", fake_call)

    import scoring.views as views

    def fake_render(step, out_png=None):
        p = Path(out_png or step)
        p.write_text("PNG")
        return p
    monkeypatch.setattr(views, "composite_for_step", fake_render)
    return calls


def _final_code(tmp_path, model, rid):
    return (tmp_path / "out" / "outputs" / model.replace("/", "_") / f"{rid}.py").read_text().strip()


def test_three_rounds_keep_last_executing(tmp_path, monkeypatch):
    calls = _wire(monkeypatch, tmp_path,
                  ["```python\nv1\n```", "```python\nv2\n```", "```python\nv3\n```"])
    (tmp_path / "gt.step").write_text("GT")
    row = rv.run_record_with_tools(record={"record_id": "r1", "step_path": "gt.step"},
                                   data_dir=tmp_path, results_root=tmp_path / "out",
                                   model="claude-x", rounds=3)
    assert row["status"] == "ok"
    assert row["rounds"] == 3                 # 1 single-shot + 2 refine rounds
    assert row["iou"] == 0.77
    assert len(calls) == 3
    # refine rounds get the target + the candidate render (2 images)
    assert len(calls[1]["image_paths"]) == 2
    assert _final_code(tmp_path, "claude-x", "r1") == "v3"


def test_failed_revision_is_rejected(tmp_path, monkeypatch):
    # round 1 good (v1); round 2 revision fails to execute → keep v1, still scored ok
    _wire(monkeypatch, tmp_path, ["```python\nv1\n```", "```python\nBAD\n```"])
    (tmp_path / "gt.step").write_text("GT")
    row = rv.run_record_with_tools(record={"record_id": "r2", "step_path": "gt.step"},
                                   data_dir=tmp_path, results_root=tmp_path / "out",
                                   model="claude-x", rounds=2)
    assert row["status"] == "ok"
    assert row["iou"] == 0.77
    assert _final_code(tmp_path, "claude-x", "r2") == "v1"   # bad revision discarded


def test_rounds_one_is_single_shot(tmp_path, monkeypatch):
    calls = _wire(monkeypatch, tmp_path, ["```python\nv1\n```"])
    (tmp_path / "gt.step").write_text("GT")
    row = rv.run_record_with_tools(record={"record_id": "r3", "step_path": "gt.step"},
                                   data_dir=tmp_path, results_root=tmp_path / "out",
                                   model="claude-x", rounds=1)
    assert row["rounds"] == 1 and len(calls) == 1          # no refine when rounds<=1
    assert _final_code(tmp_path, "claude-x", "r3") == "v1"
