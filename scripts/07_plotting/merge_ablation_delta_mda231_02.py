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
Condition tracking_uids, labels, and the eval directory are read live from
configs/evaluation/merge_ablation.toml (the mda231_02cells section by default),
so this figure stays in sync with the other merge figures when the TOML's uids
change. Values load from each condition's track_metrics.json (CTCMetrics), so the
2026-07-20 SEG fix is reflected. SEG is used in place of LNK and isn't in the
TOML metric list, so the three metrics are defined locally here; the palette is
keyed by condition name and defined locally (it no longer matches
merge_ablation_figure.py).

Usage:
    python scripts/07_plotting/merge_ablation_delta_mda231_02.py \
        [config] [--dataset mda231_02cells] [--output foo.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml

# Baseline first, then the three delta bars (matches merge_ablation_figure.py).
CONDITION_ORDER = ["baseline", "no_cohesion", "no_affinities", "no_merges"]

# Wong colorblind-friendly palette, per-concept. Baseline is the Δ=0 reference
# line, not a bar, and uses the same light blue that marks MHAT in the
# leaderboard comparison figure.
PALETTE = {
    "baseline": "#56B4E9",       # sky blue
    "no_cohesion": "#009E73",    # bluish green
    "no_affinities": "#000000",  # black
    "no_merges": "#CC79A7",      # reddish purple
}

# SEG replaces LNK here (all three read from CTCMetrics).
METRICS = ["TRA", "DET", "SEG"]


def load_ctc(eval_dir, uid):
    path = eval_dir / uid / "track_metrics.json"
    return json.loads(path.read_text())["CTCMetrics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?",
                        default="configs/evaluation/merge_ablation.toml",
                        help="Path to merge_ablation.toml")
    parser.add_argument("--dataset", default="mda231_02cells",
                        help="Top-level dataset key in the TOML")
    parser.add_argument("--output", default=None, help="Override output PNG path")
    args = parser.parse_args()

    config = toml.load(args.config)
    if args.dataset not in config:
        raise KeyError(f"dataset {args.dataset!r} not in {args.config}; "
                       f"available: {list(config.keys())}")
    dataset_cfg = config[args.dataset]
    eval_dir = (
        Path(dataset_cfg["eval_base_dir"])
        / dataset_cfg["experiment"]
        / dataset_cfg["dataset_dir"]
    )
    conditions = dataset_cfg["conditions"]

    base = load_ctc(eval_dir, conditions["baseline"]["tracking_uid"])
    # (label, color, deltas) for each non-baseline condition, in fixed order.
    rows = []
    for cond_name in CONDITION_ORDER[1:]:
        cond = conditions[cond_name]
        m = load_ctc(eval_dir, cond["tracking_uid"])
        deltas = {k: m[k] - base[k] for k in METRICS}
        rows.append((cond["label"], PALETTE[cond_name], deltas))

    x = np.arange(len(METRICS))
    n = len(rows)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, color, deltas) in enumerate(rows):
        offset = (i - (n - 1) / 2) * width
        heights = [deltas[k] for k in METRICS]
        bars = ax.bar(x + offset, heights, width, label=label, color=color,
                      edgecolor="white", linewidth=1.0)
        for bar, v in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, v - 0.0012, f"{v:+.3f}",
                    rotation=90, ha="center", va="top",
                    fontsize=7.5, color="#333333")

    baseline_color = PALETTE["baseline"]
    ax.axhline(0, color=baseline_color, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Δ vs Baseline")
    ymin = min(d[k] for _, _, d in rows for k in METRICS)
    ax.set_ylim(ymin * 1.25, abs(ymin) * 0.18)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in rows]
    labels = [l for l, _, _ in rows]
    handles.append(plt.Line2D([], [], color=baseline_color, linestyle="--",
                              linewidth=1.0))
    labels.append("Baseline (Δ=0)")
    ax.legend(handles, labels, frameon=False, loc="upper center",
              ncol=len(labels), fontsize=9, bbox_to_anchor=(0.5, -0.08),
              columnspacing=1.4, handlelength=1.4, handletextpad=0.5)

    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path(args.output or eval_dir / "merge_ablation_delta_mda231_02.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
