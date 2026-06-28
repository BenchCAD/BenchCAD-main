"""Render leaderboard.json -> LEADERBOARD.md.

    uv run python tools/build_leaderboard.py

leaderboard.json is the single source of truth (the same schema benchcad.com
renders). A new result = add a row — re-graded by us from raw outputs, never
self-reported (see CONTRIBUTING.md) — then re-run this script.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "leaderboard.json").read_text())
meta = data["meta"]
MARK = {"ours": " ⭐", "specialist": " 🔧", "control": " ·"}


def fmt(v, c):
    if v is None or v == "":
        return "—"
    f = c.get("fmt")
    if f == "f4":
        return f"{v:.4f}"
    if f == "pct1":
        return f"{v:.1f}%"
    return str(v)


out = [
    "# BenchCAD Leaderboard", "",
    f"_{meta['tagline']}_", "",
    f"**Scoring** — {meta['scoring']}", "",
    f"**Reproduce** — `{meta['run_command']}`", "",
    "> Numbers are **re-graded by us from raw model outputs, never self-reported.** "
    "To get on the board, submit predictions per "
    "[CONTRIBUTING.md § B](CONTRIBUTING.md).", "",
    "<!-- GENERATED from leaderboard.json by tools/build_leaderboard.py — do not hand-edit -->", "",
]
for t in data["tasks"].values():
    cols, primary = t["columns"], t.get("primary")
    out += [f"## {t['label']}", ""]
    if t.get("blurb"):
        out += [t["blurb"], ""]
    out.append("| " + " | ".join(c["label"] for c in cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    rows = t["rows"]
    if primary:
        pc = next((c for c in cols if c["key"] == primary), {})
        rev = pc.get("better", "high") != "low"
        rows = sorted(rows, key=lambda r: (r.get(primary) if r.get(primary) is not None else -1e9), reverse=rev)
    for r in rows:
        cells = []
        for c in cols:
            if c["key"] == "model":
                cells.append(str(r.get("model", "")) + MARK.get(r.get("class", ""), ""))
            else:
                cells.append(fmt(r.get(c["key"]), c))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
out += ["---", "", "⭐ BenchCAD's own model · 🔧 CAD specialist · re-graded, never self-reported.",
        "Generated from `leaderboard.json` — `uv run python tools/build_leaderboard.py`."]
(ROOT / "LEADERBOARD.md").write_text("\n".join(out) + "\n")
print("wrote LEADERBOARD.md")
