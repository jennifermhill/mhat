"""Unified merge ablation plot.

Reads tracking_uids and metric specs from merge_ablation.toml, loads each
condition's track_metrics.json, and renders a bar chart per metric.

Usage:
    python merge_ablation_figure.py merge_ablation.toml --dataset {mda231,nc281}
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml

# Fixed condition order so the palette mapping stays stable.
CONDITION_ORDER = ["baseline", "no_cohesion", "no_affinities", "no_merges"]

# Wong colorblind-friendly palette, per-concept consistent across figures.
PALETTE = {
    "baseline": "#0072B2",       # blue
    "no_cohesion": "#CC79A7",    # reddish purple
    "no_affinities": "#F0E442",  # yellow
    "no_merges": "#000000",      # black
}

PLACEHOLDER = "<fill in>"


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


def load_condition_metrics(cond_cfg, dataset_cfg, metric_specs):
    tracking_uid = cond_cfg["tracking_uid"]
    if tracking_uid == PLACEHOLDER:
        return None
    metrics_filename = dataset_cfg.get("metrics_filename", "track_metrics.json")
    metrics_path = (
        Path(dataset_cfg["eval_base_dir"])
        / dataset_cfg["experiment"]
        / dataset_cfg["dataset_dir"]
        / tracking_uid
        / metrics_filename
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(f"track_metrics.json not found at {metrics_path}")
    with open(metrics_path) as f:
        track_metrics = json.load(f)
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
            print(f"warning: tracking_uid for {cond_name!r} is unfilled, skipping")
            continue
        labels.append(cond_cfg["label"])
        colors.append(PALETTE[cond_name])
        for key, val in values.items():
            metric_values[key].append(val)

    if not labels:
        raise RuntimeError("no conditions had a filled-in tracking_uid; nothing to plot")

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

    output_path = Path(args.output or dataset_cfg["output_png"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
