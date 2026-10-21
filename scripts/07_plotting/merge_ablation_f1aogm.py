"""Ablation/addition plot on TE and FN detections.

Sibling of merge_ablation_figure.py, on the metrics of the competitor figure
(mhat_vs_competitor_mda231.py) and without the `- Affinities` condition. One panel
per metric, each on its own axis and titled with the metric name and an arrow for
the better direction, one bar per condition with its absolute value: the full
model, `- Coh/Adh` and `- Merges`. (An AOGM panel was dropped 2026-10-19.)

A table may set ``condition_order`` and give each condition a ``color``; without
them the MDA231 merge-ablation order and palette below apply. That is how the
NC281-sparse solver-addition figure (None, + Coh/Adh, + Flow, Full) uses this
script: scratch_configs/evaluation/NC281-sparse-label/solver_addition_f1aogm.toml.

Data provenance
---------------
Each condition's run and exact eval (see plot_spec.py) and its label are read from
the spec (scratch_configs/evaluation/Fluo-C3DL-MDA231/merge_ablation_f1aogm.toml).
The eval must carry F1AOGMMetrics and TrackOverlapMetrics, i.e. metrics = ["ctc",
"f1_aogm", "track_overlap"] with the ctc matcher (or "basic" in place of "ctc" with
the point matcher). A condition still at the placeholder is skipped; any other named
eval that is missing raises.

Usage:
    python scripts/07_plotting/merge_ablation_f1aogm.py <spec.toml> \
        [--dataset mda231_02cells] [--output foo.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import plot_spec
import toml

CONDITION_ORDER = ["baseline", "no_cohesion", "no_merges"]

PALETTE = {
    "baseline": "#000000",     # black
    "no_cohesion": "#009E73",  # bluish green
    "no_merges": "#999999",    # gray
}

# (key, label, json_path, higher_is_better, unit_interval, fmt)
METRICS = [
    ("te", "Target effectiveness", ["TrackOverlapMetrics", "target_effectiveness"],
     True, True, "{:.3f}"),
    ("fn", "False-negative detections", ["F1AOGMMetrics", "Detection FN"],
     False, False, "{:.0f}"),
]


def load_values(table, cond):
    """{metric key: value} of the condition's eval; None for a placeholder."""
    track_metrics = plot_spec.load_metrics(table, cond)
    if track_metrics is None:
        return None
    return {key: float(plot_spec.walk_json_path(track_metrics, path))
            for key, _, path, *_ in METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the merge_ablation_f1aogm spec")
    parser.add_argument("--dataset", default="mda231_02cells",
                        help="Top-level dataset key in the TOML")
    parser.add_argument("--output", default=None, help="Override output PNG path")
    args = parser.parse_args()

    config = toml.load(args.config)
    if args.dataset not in config:
        raise KeyError(f"dataset {args.dataset!r} not in {args.config}; "
                       f"available: {list(config.keys())}")
    table = config[args.dataset]
    conditions = table["conditions"]

    # (label, color, values) for each condition, in order.
    rows = []
    for cond_name in table.get("condition_order", CONDITION_ORDER):
        cond = conditions[cond_name]
        values = load_values(table, cond)
        if values is None:
            print(f"warning: {cond_name!r} is still a placeholder, skipping")
            continue
        print(cond["label"], {k: round(v, 4) for k, v in values.items()})
        rows.append((cond["label"], cond.get("color", PALETTE.get(cond_name)), values))
    if not rows:
        raise RuntimeError("every condition is still a placeholder; nothing to plot")

    # 3.8 in per panel for the three-bar merge ablation, wider with more bars so the
    # tick labels keep their room.
    panel_width = 3.8 * max(1.0, len(rows) / 3)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(panel_width * len(METRICS), 4.0))
    x = np.arange(len(rows))
    for ax, metric in zip(axes, METRICS, strict=True):
        key, label, _, higher_is_better, unit_interval, fmt = metric
        heights = [values[key] for _, _, values in rows]
        bars = ax.bar(x, heights, 0.72, color=[color for _, color, _ in rows])

        top = 1.0 if unit_interval else 1.18 * max(heights)
        for bar, v in zip(bars, heights, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012 * top, fmt.format(v),
                    ha="center", va="bottom", fontsize=12, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl, _, _ in rows], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(0, top)
        arrow = "↑" if higher_is_better else "↓"
        ax.set_title(f"{label} ({arrow})", fontsize=15)
        ax.set_ylabel(label, fontsize=13)

        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)

    if table.get("title"):
        fig.suptitle(table["title"], fontsize=16)
    fig.tight_layout()
    out = plot_spec.figure_path(
        table, f"merge_ablation_f1aogm_{args.dataset}.png", args.output
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
