"""Unified merge ablation plot.

Reads each condition's run and eval and the metric specs from merge_ablation.toml
(see plot_spec.py), loads each condition's track_metrics.json, and renders a bar
chart per metric. A condition still at the placeholder is skipped; a named eval
that is missing raises.

Usage:
    python scripts/07_plotting/merge_ablation_figure.py configs/evaluation/merge_ablation.toml --dataset {mda231,nc281}
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import toml

import plot_spec

# Fixed condition order so the palette mapping stays stable.
CONDITION_ORDER = ["baseline", "no_cohesion", "no_affinities", "no_merges"]

# Wong colorblind-friendly palette, per-concept consistent across figures.
PALETTE = {
    "baseline": "#CC79A7",       # reddish purple
    "no_cohesion": "#D55E00",    # vermillion (reddish orange)
    "no_affinities": "#009E73",  # bluish green
    "no_merges": "#000000",      # black
}


def walk_json_path(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            available = list(cur.keys()) if isinstance(cur, dict) else type(cur).__name__
            raise KeyError(f"json_path step {key!r} not found; available: {available}")
        cur = cur[key]
    return cur


def reduce_value(value, reduction):
    if reduction == "scalar":
        return float(value)
    if reduction == "mean":
        if isinstance(value, list):
            return float(np.mean(value))
        return float(value)
    raise ValueError(f"unknown reduction: {reduction!r}")


def load_condition_metrics(cond_cfg, dataset_cfg, metric_specs, eval_key="eval"):
    track_metrics = plot_spec.load_metrics(dataset_cfg, cond_cfg, eval_key)
    if track_metrics is None:
        return None
    return {
        key: reduce_value(walk_json_path(track_metrics, spec["json_path"]), spec["reduction"])
        for key, spec in metric_specs.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to merge_ablation.toml")
    parser.add_argument("--dataset", required=True, help="Top-level dataset key in the TOML (e.g., mda231, nc281)")
    parser.add_argument("--output", default=None, help="Override output_png from TOML")
    args = parser.parse_args()

    config = toml.load(args.config)
    if args.dataset not in config:
        raise KeyError(f"dataset {args.dataset!r} not in {args.config}; available: {list(config.keys())}")
    dataset_cfg = config[args.dataset]

    metric_specs = dataset_cfg["metrics"]
    conditions = dataset_cfg["conditions"]

    # Collect per-condition values for each metric, in fixed order.
    labels = []
    colors = []
    metric_values = {key: [] for key in metric_specs}
    for cond_name in CONDITION_ORDER:
        if cond_name not in conditions:
            print(f"warning: condition {cond_name!r} missing from TOML, skipping")
            continue
        cond_cfg = conditions[cond_name]
        values = load_condition_metrics(cond_cfg, dataset_cfg, metric_specs)
        if values is None:
            print(f"warning: {cond_name!r} is still a placeholder, skipping")
            continue
        labels.append(cond_cfg["label"])
        colors.append(PALETTE[cond_name])
        for key, val in values.items():
            metric_values[key].append(val)

    if not labels:
        raise RuntimeError("every condition is still a placeholder; nothing to plot")

    n_metrics = len(metric_specs)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics + 1, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (key, spec) in zip(axes, metric_specs.items()):
        values = metric_values[key]
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(spec["display_name"], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.axhline(y=values[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(dataset_cfg["suptitle"], fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = plot_spec.figure_path(
        dataset_cfg, f"merge_ablation_results_{args.dataset}.png", args.output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
