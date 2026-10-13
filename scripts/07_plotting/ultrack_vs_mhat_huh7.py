"""MHAT vs Ultrack on Fluo-C2DL-Huh7, averaged over 01_cells and 02_cells.

Ultrack (Bergamaschi et al. 2025) reports Detection F1, TRA and false-negative
detections for Huh7 as the mean over videos 01 and 02, so MHAT is averaged over
the same two videos here. One panel per metric, each with an MHAT bar and an
Ultrack bar; F1 and TRA share a 0-1 axis, the FN count gets its own.

Data provenance
---------------
* MHAT: loaded live from each video's eval (``track_metrics.json``), named in
  the spec by ``output_dir`` / ``run`` / exact ``eval`` (see plot_spec.py), and
  averaged with equal weight per video, as Ultrack's numbers are.
* Ultrack: published values, written into the spec's ``ultrack`` key of each
  metric. They are not a run of ours.

Spec schema (TOML) -- see scratch_configs/evaluation/ultrack_vs_mhat_huh7.toml:

    output_png   = ".../figures/ultrack_vs_mhat_huh7.png"
    metric_order = ["detection_f1", "tra", "fn"]
    [metrics.detection_f1]
    label            = "Detection F1"
    json_path        = ["F1AOGMMetrics", "Detection F1"]
    ultrack          = 0.921
    higher_is_better = true
    fmt              = "{:.3f}"
    [methods.mhat]      label, color
    [methods.ultrack]   label, color
    [[datasets]]        label, output_dir, run, eval   (one per video)

Usage:
    python ultrack_vs_mhat_huh7.py <spec.toml> [--output foo.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import toml

import plot_spec


def mhat_values(cfg, metric_keys):
    """{metric: [value per dataset]} for MHAT, in spec dataset order."""
    values = {key: [] for key in metric_keys}
    for dataset in cfg["datasets"]:
        track_metrics = plot_spec.load_metrics(dataset, dataset)
        if track_metrics is None:
            raise ValueError(f"dataset {dataset['label']!r} is still a placeholder")
        for key in metric_keys:
            path = cfg["metrics"][key]["json_path"]
            values[key].append(float(plot_spec.walk_json_path(track_metrics, path)))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the ultrack_vs_mhat_huh7 TOML")
    parser.add_argument("--output", default=None, help="Override output_png from TOML")
    args = parser.parse_args()

    cfg = toml.load(args.config)
    metrics = cfg["metrics"]
    methods = cfg["methods"]
    metric_keys = cfg.get("metric_order", list(metrics.keys()))
    per_dataset = mhat_values(cfg, metric_keys)

    labels = [d["label"] for d in cfg["datasets"]]
    print(f"{'metric':<16}" + "".join(f"{lab:>12}" for lab in labels)
          + f"{'MHAT mean':>12}{'Ultrack':>12}")
    for key in metric_keys:
        print(f"{key:<16}" + "".join(f"{v:>12.4f}" for v in per_dataset[key])
              + f"{np.mean(per_dataset[key]):>12.4f}{metrics[key]['ultrack']:>12.4f}")

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(2.3 * len(metric_keys), 3.4))
    for ax, key in zip(np.atleast_1d(axes), metric_keys):
        metric = metrics[key]
        heights = [float(np.mean(per_dataset[key])), float(metric["ultrack"])]
        colors = [methods["mhat"]["color"], methods["ultrack"]["color"]]
        names = [methods["mhat"]["label"], methods["ultrack"]["label"]]

        x = np.arange(2)
        bars = ax.bar(x, heights, 0.72, color=colors)
        top = 1.0 if metric.get("unit_interval", True) else 1.18 * max(heights)
        fmt = metric.get("fmt", "{:.3f}")
        for bar, v in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012 * top, fmt.format(v),
                    ha="center", va="bottom", fontsize=8, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, top)
        direction = "higher" if metric.get("higher_is_better", True) else "lower"
        ax.set_ylabel(f"{metric['label']}\n({direction} is better)", fontsize=10)

        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)

    if cfg.get("title"):
        fig.suptitle(cfg["title"])
    fig.tight_layout()
    out = plot_spec.figure_path(cfg, "ultrack_vs_mhat_huh7.png", args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
