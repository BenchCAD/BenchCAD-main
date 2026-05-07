"""Bar plot of mean norm-IoU per (model, mode), fig3-style.

Layout
------
    x-axis : models present in results.jsonl
    y-axis : mean norm-IoU (averaged across records for each (model, mode))
    Each model gets 1 husl colour.
    Each model has up-to-3 sub-bars (one per mode), differentiated by FILL:
        instruction → white fill + coloured edge + hatch "////"  (text-only)
        image       → solid colour                                (image-only)
        ablation    → solid colour, alpha=0.55, hatch ".."        (text+image)

Adaptive: only the (model, mode) pairs actually present in results.jsonl are
drawn — empty cells are skipped (no zero-bar). Number of husl hues = number
of distinct models.
"""

from __future__ import annotations

import json
from pathlib import Path

MODE_ORDER = ("instruction", "image", "ablation")
MODE_STYLE = {
    "instruction": dict(facecolor="white", hatch="////", alpha=1.0, edgecolor=None, edgewidth=1.4),
    "image":       dict(facecolor=None,    hatch=None,   alpha=1.0, edgecolor="white", edgewidth=0.6),
    "ablation":    dict(facecolor=None,    hatch="..",   alpha=0.55, edgecolor="white", edgewidth=0.6),
}
MODE_LABEL = {
    "instruction": "instruction only (text)",
    "image":       "image only",
    "ablation":    "instruction + image (ablation)",
}


def _load(jsonl: Path) -> list[dict]:
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"No rows in {jsonl}")
    return rows


def _aggregate(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Return {(model, mode): {mean_norm, n}} averaged over records."""
    bucket: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        bucket.setdefault((r["model"], r["mode"]), []).append(float(r["norm_iou"]))
    return {k: {"mean_norm": sum(v) / len(v), "n": len(v)} for k, v in bucket.items()}


def make_bar(results_jsonl: Path, out_png: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    import seaborn as sns

    rows = _load(results_jsonl)
    agg = _aggregate(rows)
    models = sorted({m for m, _ in agg})
    modes_present = [m for m in MODE_ORDER if any((mod, m) in agg for mod in models)]
    n_models = len(models)
    n_modes = len(modes_present)
    palette = sns.color_palette("husl", n_models)
    color_for = dict(zip(models, palette))

    # Figure width scales with number of models (min 6 inches, +1 inch per model).
    fig, ax = plt.subplots(figsize=(max(6.0, 2.0 + 1.6 * n_models), 4.6))
    bar_w = 0.86 / n_modes
    x = np.arange(n_models)

    for j, mode in enumerate(modes_present):
        offsets = (j - (n_modes - 1) / 2) * bar_w
        for i, model in enumerate(models):
            cell = agg.get((model, mode))
            if cell is None:
                continue
            color = color_for[model]
            style = MODE_STYLE[mode]
            face = color if style["facecolor"] is None else style["facecolor"]
            edge = color if style["edgecolor"] is None else style["edgecolor"]
            ax.bar(
                x[i] + offsets, cell["mean_norm"], bar_w,
                facecolor=face,
                edgecolor=edge,
                linewidth=style["edgewidth"],
                alpha=style["alpha"],
                hatch=style["hatch"],
            )
            ax.text(
                x[i] + offsets, cell["mean_norm"] + 0.012,
                f"{cell['mean_norm']:.2f}",
                ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean norm-IoU", fontsize=12)
    ax.set_title("")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend: only mode styles (model is encoded by both colour and x-tick).
    legend_handles = []
    for mode in modes_present:
        st = MODE_STYLE[mode]
        face = "#888" if st["facecolor"] is None else st["facecolor"]
        edge = "#444" if st["edgecolor"] is None else st["edgecolor"]
        legend_handles.append(Patch(
            facecolor=face, edgecolor=edge, alpha=st["alpha"],
            hatch=st["hatch"], linewidth=st["edgewidth"],
            label=MODE_LABEL[mode],
        ))
    ax.legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), title="Mode",
        frameon=False, fontsize=10, title_fontsize=10,
    )
    fig.subplots_adjust(right=0.78)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png
