"""Unified solver addition / ablation plot.

Reads each condition's run and eval (see plot_spec.py), condition order, colors,
and metric specs from a solver experiment TOML (solver_ablation.toml or
solver_addition.toml), loads each condition's track_metrics.json, and renders a
bar chart per metric. A placeholder condition is skipped; a named eval that is
missing raises. The same
script drives both experiment types -- the only difference lives in the TOML.

Unlike merge_ablation_figure.py, the condition set varies per dataset, so the
order, labels, and per-bar colors are read from the TOML rather than hardcoded.

Usage:
    python scripts/07_plotting/solver_figure.py configs/evaluation/solver_ablation.toml --dataset mda231
    python scripts/07_plotting/solver_figure.py configs/evaluation/solver_addition.toml --dataset nc281 --output foo.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml

import plot_spec


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
    # Each condition names its own eval, so one whose run scored a different
    # metrics file (e.g. point-matched metrics split off as their own eval) just
    # names that eval.
    track_metrics = plot_spec.load_metrics(dataset_cfg, cond_cfg)
    if track_metrics is None:
        return None
    return {
        key: reduce_value(walk_json_path(track_metrics, spec["json_path"]), spec["reduction"])
        for key, spec in metric_specs.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to solver_ablation.toml or solver_addition.toml")
    parser.add_argument("--dataset", required=True, help="Top-level dataset key in the TOML (e.g., mda231, nc281)")
    parser.add_argument("--output", default=None, help="Override output_png from TOML")
    args = parser.parse_args()

    config = toml.load(args.config)
    if args.dataset not in config:
        raise KeyError(f"dataset {args.dataset!r} not in {args.config}; available: {list(config.keys())}")
    dataset_cfg = config[args.dataset]

    metric_specs = dataset_cfg["metrics"]
    conditions = dataset_cfg["conditions"]
    condition_order = dataset_cfg["condition_order"]
    rotation = dataset_cfg.get("xtick_rotation", 0)
    reference = dataset_cfg.get("reference", "first")  # "first" | "last" | "none"

    # Collect per-condition values in the TOML-specified order.
    labels = []
    colors = []
    metric_values = {key: [] for key in metric_specs}
    for cond_name in condition_order:
        if cond_name not in conditions:
            print(f"warning: condition {cond_name!r} in condition_order but missing from TOML, skipping")
            continue
        cond_cfg = conditions[cond_name]
        values = load_condition_metrics(cond_cfg, dataset_cfg, metric_specs)
        if values is None:
            print(f"warning: {cond_name!r} is still a placeholder, skipping")
            continue
        labels.append(cond_cfg["label"])
        colors.append(cond_cfg["color"])
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
        if rotation:
            ax.set_xticklabels(labels, fontsize=9, rotation=rotation, ha="right")
        else:
            ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        if reference == "first":
            ax.axhline(y=values[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        elif reference == "last":
            ax.axhline(y=values[-1], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
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
        dataset_cfg, f"{Path(args.config).stem}_results_{args.dataset}.png", args.output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
