"""Guard the leaderboard data + that LEADERBOARD.md was regenerated from it."""
import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_leaderboard_json_well_formed():
    d = json.loads((_ROOT / "leaderboard.json").read_text())
    assert d.get("tasks"), "no tasks"
    for name, t in d["tasks"].items():
        assert t.get("columns"), f"{name}: no columns"
        assert t.get("rows"), f"{name}: no rows"


def test_leaderboard_md_lists_every_task():
    # If this fails after editing leaderboard.json, run tools/build_leaderboard.py.
    md = (_ROOT / "LEADERBOARD.md").read_text()
    d = json.loads((_ROOT / "leaderboard.json").read_text())
    for t in d["tasks"].values():
        assert f"## {t['label']}" in md, f"{t['label']} missing — run tools/build_leaderboard.py"
