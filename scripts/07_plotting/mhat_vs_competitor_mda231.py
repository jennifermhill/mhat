"""MHAT vs competing trackers on Fluo-C3DL-MDA231 (02_cells, the held-out test movie).

Replaces ``leaderboard_comparison_mda231.py`` (MHAT vs the CTC leaderboard top-3).
One panel per metric, one bar per method, each panel on its own axis: target
effectiveness, false-negative detections and AOGM.

Each method names its ``run`` and exact ``eval`` (see plot_spec.py); values are
loaded live from that ``track_metrics.json``. A method whose run or eval is still
``plot_spec.PLACEHOLDER`` is drawn as a hatched "no result" bar, so the figure can
be laid out before the Ultrack and TrackMate runs are final. Any other named eval
that is missing raises.

Spec schema (TOML) -- see scratch_configs/evaluation/Fluo-C3DL-MDA231/mhat_vs_competitor_mda231_02.toml:

    output_dir   = ".../experiments/Fluo-C3DL-MDA231/02_cells"
    metric_order = ["te", "fn", "aogm"]
    [metrics.fn]
    label            = "False-negative detections"
    json_path        = ["F1AOGMMetrics", "Detection FN"]
    higher_is_better = false
    unit_interval    = false        # own y range instead of 0-1
    fmt              = "{:.0f}"
    method_order = ["mhat", "ultrack", "trackmate"]
    [methods.mhat]   label, color, run, eval

Usage:
    python mhat_vs_competitor_mda231.py <spec.toml> [--output foo.png]
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
    parser.add_argument("config", help="Path to the mhat_vs_competitor_mda231 TOML")
    parser.add_argument("--output", default=None, help="Override output_png from TOML")
    args = parser.parse_args()

    cfg = toml.load(args.config)
    metrics = cfg["metrics"]
    methods = cfg["methods"]
    metric_keys = cfg.get("metric_order", list(metrics.keys()))
    method_keys = cfg.get("method_order", list(methods.keys()))

    # values[method][metric] is a float, or None for a placeholder method
    values = {}
    for mk in method_keys:
        track_metrics = plot_spec.load_metrics(cfg, methods[mk])
        if track_metrics is None:
            values[mk] = {key: None for key in metric_keys}
            print(f"{methods[mk]['label']}: placeholder")
            continue
        values[mk] = {key: metric_value(track_metrics, metrics[key]) for key in metric_keys}
        print(methods[mk]["label"])
        for key in metric_keys:
            print(f"  {key:<14} {values[mk][key]:.4f}")

    fig, axes = plt.subplots(
        1, len(metric_keys), figsize=(3.1 * len(metric_keys), 3.4)
    )
    for ax, key in zip(np.atleast_1d(axes), metric_keys):
        metric = metrics[key]
        vals = [values[mk][key] for mk in method_keys]
        real = [v for v in vals if v is not None]
        colors = [methods[mk]["color"] for mk in method_keys]
        names = [methods[mk]["label"] for mk in method_keys]

        x = np.arange(len(method_keys))
        bars = ax.bar(x, [v if v is not None else 0.0 for v in vals], 0.72, color=colors)
        # Placeholders can't set the scale; with no real bar fall back to 1.
        top = 1.0 if metric.get("unit_interval", True) else 1.18 * max(real, default=1.0)
        fmt = metric.get("fmt", "{:.3f}")
        for bar, v in zip(bars, vals):
            if v is None:
                bar.set_hatch("xxx")
                bar.set_alpha(0.35)
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02 * top, "no result",
                        rotation=90, ha="center", va="bottom", fontsize=7, color="#444")
            else:
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
    out = plot_spec.figure_path(cfg, "mhat_vs_competitor_mda231_02.png", args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
