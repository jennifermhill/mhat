"""Merge-ablation delta plot: Fluo-C3DL-MDA231 / 02_cells.

Mimics the merge ablation figure but (a) uses SEG in place of LNK and (b) plots
each ablation condition as a *delta from baseline* (metric_condition - metric_baseline)
rather than the raw score. Formatting follows the leaderboard comparison figure:
all three metrics (TRA, DET, SEG) on one axes, touching bars per condition, a
legend mapping color -> condition, no title.

Baseline is the zero reference (drawn as the dashed line at Δ = 0), so only the
three ablation conditions get bars; every bar is negative because each ablation
hurts every metric.

Data provenance
---------------
All values load live from each condition's track_metrics.json (CTCMetrics), so
the 2026-07-20 SEG fix -- which required re-evaluating every non-baseline
condition -- is reflected. tracking_uids are the mda231_02cells conditions in
configs/evaluation/merge_ablation.toml. Palette matches merge_ablation_figure.py.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path(
    "Y:/jennifer/mhat/experiments/evaluation/Fluo-C3DL-MDA231/02_cells"
)

BASELINE_UID = "2026-06-01_14-41-03"
# label -> (tracking_uid, color)  (merge_ablation_figure.py Wong palette)
CONDITIONS = [
    ("- Coh/Adh",    "2026-06-25_15-22-14", "#CC79A7"),  # reddish purple
    ("- Affinities", "2026-06-01_16-00-08", "#F0E442"),  # yellow
    ("- Merges",     "2026-06-01_16-01-29", "#000000"),  # black
]
BASELINE_COLOR = "#0072B2"  # blue
METRICS = ["TRA", "DET", "SEG"]


def load_ctc(uid):
    path = EVAL_DIR / uid / "track_metrics.json"
    return json.loads(path.read_text())["CTCMetrics"]


def main():
    base = load_ctc(BASELINE_UID)
    # deltas[label][metric] = metric_condition - metric_baseline
    deltas = {}
    for label, uid, _ in CONDITIONS:
        m = load_ctc(uid)
        deltas[label] = {k: m[k] - base[k] for k in METRICS}

    x = np.arange(len(METRICS))
    n = len(CONDITIONS)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, _uid, color) in enumerate(CONDITIONS):
        offset = (i - (n - 1) / 2) * width
        heights = [deltas[label][k] for k in METRICS]
        bars = ax.bar(x + offset, heights, width, label=label, color=color,
                      edgecolor="white", linewidth=1.0)
        for bar, v in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, v - 0.0012, f"{v:+.3f}",
                    rotation=90, ha="center", va="top",
                    fontsize=7.5, color="#333333")

    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Δ vs Baseline")
    ymin = min(deltas[l][k] for l, _, _ in CONDITIONS for k in METRICS)
    ax.set_ylim(ymin * 1.25, abs(ymin) * 0.18)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CONDITIONS]
    labels = [l for l, _, _ in CONDITIONS]
    handles.append(plt.Line2D([], [], color=BASELINE_COLOR, linestyle="--",
                              linewidth=1.0))
    labels.append("Baseline (Δ=0)")
    ax.legend(handles, labels, frameon=False, loc="upper center",
              ncol=len(labels), fontsize=9, bbox_to_anchor=(0.5, -0.08),
              columnspacing=1.4, handlelength=1.4, handletextpad=0.5)

    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = EVAL_DIR / "merge_ablation_delta_mda231_02.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
