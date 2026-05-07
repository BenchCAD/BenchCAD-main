"""Bar plot of mean IoU per model, husl palette.

x-axis : models present in results.jsonl
y-axis : mean IoU (averaged across records for each model)
One coloured bar per model; value labelled above each bar.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load(jsonl: Path) -> list[dict]:
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"No rows in {jsonl}")
    return rows


def _aggregate(rows: list[dict]) -> dict[str, dict]:
    bucket: dict[str, list[float]] = {}
    for r in rows:
        bucket.setdefault(r["model"], []).append(float(r["iou"]))
    return {m: {"mean": sum(v) / len(v), "n": len(v)} for m, v in bucket.items()}


def make_bar(results_jsonl: Path, out_png: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    rows = _load(results_jsonl)
    agg = _aggregate(rows)
    models = sorted(agg)
    palette = sns.color_palette("husl", len(models))

    fig, ax = plt.subplots(figsize=(max(6.0, 2.0 + 1.4 * len(models)), 4.6))
    x = np.arange(len(models))
    for i, m in enumerate(models):
        ax.bar(x[i], agg[m]["mean"], 0.7, color=palette[i], edgecolor="white", linewidth=0.6)
        ax.text(x[i], agg[m]["mean"] + 0.012, f"{agg[m]['mean']:.2f}",
                ha="center", va="bottom", fontsize=9, color=palette[i], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean voxel IoU", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png
