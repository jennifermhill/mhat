"""MHAT vs competing trackers on primary_nk_cells / 01_cells.

The GT is sparse (MTrackJ points, corrected 2026-07-30), so every metric shown
must ignore false positives: an unannotated true cell is otherwise counted as
an error. One panel per metric, one bar per method, each panel on its own axis.

Each method names its ``run`` and exact ``eval`` (see plot_spec.py); values are
loaded live from that ``track_metrics.json``. A metric gives a ``json_path``, or
``json_paths`` whose values are summed (e.g. AOGM without its FP terms).

Spec schema (TOML) -- see scratch_configs/evaluation/mhat_vs_competitor_nk_cells.toml:

    output_png   = ".../figures/mhat_vs_competitor_nk_cells.png"
    output_dir   = ".../experiments/primary_nk_cells/01_cells"
    metric_order = ["te", "fn", "aogm"]
    [metrics.aogm]
    label            = "AOGM"
    json_paths       = [["AOGMMetrics", "fn_nodes"], ["AOGMMetrics", "fn_edges"]]
    higher_is_better = false
    unit_interval    = false        # own y range instead of 0-1
    fmt              = "{:.0f}"
    method_order = ["mhat", "ultrack"]
    [methods.mhat]   label, color, run, eval

Usage:
    python mhat_vs_competitor_nk_cells.py <spec.toml> [--output foo.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import toml

import plot_spec


def metric_value(track_metrics, metric):
    """One metric's value: ``json_path``, or the sum over ``json_paths``."""
    paths = metric.get("json_paths", [metric.get("json_path")])
    return sum(float(plot_spec.walk_json_path(track_metrics, p)) for p in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the mhat_vs_competitor_nk_cells TOML")
    parser.add_argument("--output", default=None, help="Override output_png from TOML")
    args = parser.parse_args()

    cfg = toml.load(args.config)
    metrics = cfg["metrics"]
    methods = cfg["methods"]
    metric_keys = cfg.get("metric_order", list(metrics.keys()))
    method_keys = cfg.get("method_order", list(methods.keys()))

    values = {}
    for mk in method_keys:
        track_metrics = plot_spec.load_metrics(cfg, methods[mk])
        if track_metrics is None:
            raise ValueError(f"{methods[mk]['label']} is still a placeholder")
        values[mk] = {key: metric_value(track_metrics, metrics[key]) for key in metric_keys}
        print(methods[mk]["label"])
        for key in metric_keys:
            print(f"  {key:<14} {values[mk][key]:.4f}")

    fig, axes = plt.subplots(
        1, len(metric_keys), figsize=(3.1 * len(metric_keys), 3.4)
    )
    for ax, key in zip(np.atleast_1d(axes), metric_keys):
        metric = metrics[key]
        heights = [values[mk][key] for mk in method_keys]
        colors = [methods[mk]["color"] for mk in method_keys]
        names = [methods[mk]["label"] for mk in method_keys]

        x = np.arange(len(method_keys))
        bars = ax.bar(x, heights, 0.72, color=colors)
        top = 1.0 if metric.get("unit_interval", True) else 1.18 * max(heights)
        fmt = metric.get("fmt", "{:.3f}")
        for bar, v in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012 * top, fmt.format(v),
                    ha="center", va="bottom", fontsize=8, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_xlim(-0.6, len(method_keys) - 0.4)
        ax.set_ylim(0, top)
        arrow = "↑" if metric.get("higher_is_better", True) else "↓"
        ax.set_ylabel(f"{metric['label']} ({arrow})", fontsize=10)

        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)

    if cfg.get("title"):
        fig.suptitle(cfg["title"])
    fig.tight_layout()
    out = plot_spec.figure_path(cfg, "mhat_vs_competitor_nk_cells.png", args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
