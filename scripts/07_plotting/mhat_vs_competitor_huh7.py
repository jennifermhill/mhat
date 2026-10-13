"""MHAT vs competing trackers on Fluo-C2DL-Huh7, averaged over 01_cells and 02_cells.

Ultrack (Bergamaschi et al. 2025) reports Detection F1, TRA and false-negative
detections for Huh7 as the mean over videos 01 and 02, so every method is
averaged over the same two videos here. One panel per metric, one bar per
method; F1 and TRA share a 0-1 axis, the FN count gets its own.

Data provenance
---------------
A method is either *run* by us or *published*:

* A run method (MHAT, TrackMate) lists one ``datasets`` entry per video, each
  naming ``output_dir`` / ``run`` / exact ``eval`` (see plot_spec.py); its
  values are loaded live from those ``track_metrics.json`` files and averaged
  with equal weight per video, as Ultrack's numbers are.
* A published method (Ultrack) gives its values under ``published``, keyed by
  metric. They are not a run of ours.

Spec schema (TOML) -- see scratch_configs/evaluation/mhat_vs_competitor_huh7.toml:

    output_png   = ".../figures/mhat_vs_competitor_huh7.png"
    metric_order = ["detection_f1", "tra", "fn"]
    [metrics.detection_f1]
    label            = "Detection F1"
    json_path        = ["F1AOGMMetrics", "Detection F1"]
    higher_is_better = true
    fmt              = "{:.3f}"
    method_order = ["mhat", "ultrack", "trackmate"]
    [methods.mhat]         label, color, [[methods.mhat.datasets]] label/output_dir/run/eval
    [methods.ultrack]      label, color, published = {detection_f1 = 0.921, ...}

Usage:
    python mhat_vs_competitor_huh7.py <spec.toml> [--output foo.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import toml

import plot_spec


def method_values(method, metrics, metric_keys):
    """{metric: [value per video]} for a run method, or {metric: [published]}."""
    if "published" in method:
        return {key: [float(method["published"][key])] for key in metric_keys}
    values = {key: [] for key in metric_keys}
    for dataset in method["datasets"]:
        track_metrics = plot_spec.load_metrics(dataset, dataset)
        if track_metrics is None:
            raise ValueError(
                f"{method['label']} {dataset['label']!r} is still a placeholder"
            )
        for key in metric_keys:
            path = metrics[key]["json_path"]
            values[key].append(float(plot_spec.walk_json_path(track_metrics, path)))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the mhat_vs_competitor_huh7 TOML")
    parser.add_argument("--output", default=None, help="Override output_png from TOML")
    args = parser.parse_args()

    cfg = toml.load(args.config)
    metrics = cfg["metrics"]
    methods = cfg["methods"]
    metric_keys = cfg.get("metric_order", list(metrics.keys()))
    method_keys = cfg.get("method_order", list(methods.keys()))
    values = {mk: method_values(methods[mk], metrics, metric_keys) for mk in method_keys}

    for mk in method_keys:
        method = methods[mk]
        videos = (["published"] if "published" in method
                  else [d["label"] for d in method["datasets"]])
        print(f"{method['label']} ({', '.join(videos)})")
        for key in metric_keys:
            per_video = "  ".join(f"{v:.4f}" for v in values[mk][key])
            print(f"  {key:<14} {per_video}  -> mean {np.mean(values[mk][key]):.4f}")

    fig, axes = plt.subplots(
        1, len(metric_keys), figsize=(3.1 * len(metric_keys), 3.4)
    )
    for ax, key in zip(np.atleast_1d(axes), metric_keys):
        metric = metrics[key]
        heights = [float(np.mean(values[mk][key])) for mk in method_keys]
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
        direction = "higher" if metric.get("higher_is_better", True) else "lower"
        ax.set_ylabel(f"{metric['label']}\n({direction} is better)", fontsize=10)

        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)

    if cfg.get("title"):
        fig.suptitle(cfg["title"])
    fig.tight_layout()
    out = plot_spec.figure_path(cfg, "mhat_vs_competitor_huh7.png", args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
