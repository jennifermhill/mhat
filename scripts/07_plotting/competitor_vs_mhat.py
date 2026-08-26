"""MHAT vs a competing tracker, on a single dataset.

Generic: the competitor, the metrics, the colors, and the bar order all come
from the config, so one script renders every head-to-head figure. Current
configs:

    configs/evaluation/trackmate_vs_mhat.toml        NC281-sparse / train
    configs/evaluation/ultrack_vs_mhat_nk_cells.toml primary_nk_cells / 01_cells

Renders a grouped bar chart: one x-group per metric (e.g. target effectiveness,
track fractions, recalls), with two touching bars per group -- MHAT first, then
the competitor -- plus a legend. Every value is loaded from the corresponding
tracking result's track_metrics.json -- the config only says which result each
method points at, so re-running picks up any re-evaluation.

Config schema (TOML) -- see configs/evaluation/trackmate_vs_mhat.toml:

    output_png    = "..."
    title         = "..."
    ylabel        = "Score (higher is better)"
    experiment    = "NC281-sparse-label"
    dataset_dir   = "02_nuclei_denoised_train"
    eval_base_dir = "Y:/jennifer/mhat/experiments/evaluation"

    metric_order = ["target_effectiveness", "track_fractions", "track_purity"]
    [metrics.target_effectiveness]
    label     = "Target\neffectiveness"
    json_path = ["TrackOverlapMetrics", "target_effectiveness"]

    method_order = ["mhat", "trackmate"]
    [methods.mhat]
    label        = "MHAT"
    color        = "#CC79A7"
    tracking_uid = "2026-07-02_10-55-33"

Usage:
    python competitor_vs_mhat.py configs/evaluation/trackmate_vs_mhat.toml [--output foo.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml


def walk_json_path(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            available = list(cur.keys()) if isinstance(cur, dict) else type(cur).__name__
            raise KeyError(f"json_path step {key!r} not found; available: {available}")
        cur = cur[key]
    return cur


def load_metric(cfg, uid, json_path):
    """Return the metric value for one (method, metric), or None if missing."""
    path = (
        Path(cfg["eval_base_dir"])
        / cfg["experiment"]
        / cfg["dataset_dir"]
        / uid
        / "track_metrics.json"
    )
    if not path.is_file():
        print(f"warning: metrics missing for {uid} ({path})")
        return None
    with open(path) as f:
        track_metrics = json.load(f)
    return float(walk_json_path(track_metrics, json_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a <competitor>_vs_mhat TOML")
    parser.add_argument("--output", default=None, help="Override output_png from TOML")
    args = parser.parse_args()

    cfg = toml.load(args.config)
    metrics = cfg["metrics"]
    methods = cfg["methods"]
    metric_keys = cfg.get("metric_order", list(metrics.keys()))
    method_keys = cfg.get("method_order", list(methods.keys()))

    # values[method_key] -> list over metrics
    values = {mk: [] for mk in method_keys}
    for metric_key in metric_keys:
        json_path = metrics[metric_key]["json_path"]
        for mk in method_keys:
            values[mk].append(load_metric(cfg, methods[mk]["tracking_uid"], json_path))

    x = np.arange(len(metric_keys))
    n = len(method_keys)
    width = 0.8 / n  # bars touch within a group, gap between groups

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, mk in enumerate(method_keys):
        offset = (i - (n - 1) / 2) * width
        vals = values[mk]
        plot_vals = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(
            x + offset, plot_vals, width,
            label=methods[mk]["label"], color=methods[mk]["color"],
            edgecolor="white", linewidth=1.0,  # 2px surface gap between touching bars
        )
        for bar, v in zip(bars, vals):
            if v is None:
                bar.set_hatch("xxx")
                bar.set_alpha(0.35)
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "no result",
                        rotation=90, ha="center", va="bottom", fontsize=7, color="#444")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([metrics[k]["label"] for k in metric_keys])
    ax.set_ylabel(cfg.get("ylabel", "Score (higher is better)"))
    ax.set_ylim(0, 1.0)
    ax.set_title(cfg.get("title", ""))
    ax.legend(frameon=False, loc="upper left")

    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path = Path(args.output or cfg["output_png"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
