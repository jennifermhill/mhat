"""Hand-tuned vs SSVM-fit comparison figure across datasets.

Renders a single grouped bar chart: one x-group per dataset, two bars per group
(hand-tuned, SSVM-fit). Because the primary metric differs per dataset (TRA for a
CTC-matched dataset, TE for a point-matched one), each dataset carries its own
`json_path`, and its group label names the metric. Bar colors encode the
condition (hand-tuned vs SSVM), shared across datasets, so a single legend reads
the whole figure.

Config schema (TOML) -- see ssvm_vs_handtuned.toml:

    output_png   = "..."
    suptitle     = "..."
    ylabel       = "Score"
    dataset_order = ["mda231", "nc281_train"]

    [conditions.hand_tuned]  # shared style for the hand-tuned bar
    label = "Hand-tuned"
    color = "#0072B2"
    [conditions.ssvm]
    label = "SSVM-fit"
    color = "#D55E00"

    [datasets.mda231]
    name        = "MDA231 (01_cells)\nTRA"
    experiment  = "Fluo-C3DL-MDA231"
    dataset_dir = "01_cells"
    eval_base_dir = "Y:/jennifer/mhat/experiments/evaluation"
    json_path   = ["CTCMetrics", "TRA"]
    reduction   = "scalar"      # or "mean"
    [datasets.mda231.hand_tuned]
    tracking_uid = "2026-05-15_10-55-56"
    [datasets.mda231.ssvm]
    tracking_uid = "ssvm_refit"

Usage:
    python ssvm_vs_handtuned_figure.py ssvm_vs_handtuned.toml [--output foo.png]
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


def reduce_value(value, reduction):
    if reduction == "scalar":
        return float(value)
    if reduction == "mean":
        return float(np.mean(value)) if isinstance(value, list) else float(value)
    raise ValueError(f"unknown reduction: {reduction!r}")


def load_metric(ds_cfg, cond_cfg):
    """Return the metric value for one (dataset, condition), or None if missing."""
    uid = cond_cfg["tracking_uid"]
    metrics_filename = cond_cfg.get("metrics_filename", ds_cfg.get("metrics_filename", "track_metrics.json"))
    path = (
        Path(ds_cfg["eval_base_dir"])
        / ds_cfg["experiment"]
        / ds_cfg["dataset_dir"]
        / uid
        / metrics_filename
    )
    if not path.is_file():
        print(f"warning: metrics missing for {ds_cfg['experiment']}/{ds_cfg['dataset_dir']}/{uid} ({path})")
        return None
    with open(path) as f:
        track_metrics = json.load(f)
    return reduce_value(walk_json_path(track_metrics, ds_cfg["json_path"]), ds_cfg.get("reduction", "scalar"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to ssvm_vs_handtuned.toml")
    parser.add_argument("--output", default=None, help="Override output_png from TOML")
    args = parser.parse_args()

    cfg = toml.load(args.config)
    cond_style = cfg["conditions"]  # {hand_tuned: {label,color}, ssvm: {...}}
    cond_keys = ["hand_tuned", "ssvm"]
    datasets = cfg["datasets"]
    order = cfg.get("dataset_order", list(datasets.keys()))

    # values[cond_key] -> list over datasets
    values = {ck: [] for ck in cond_keys}
    group_labels = []
    for ds_key in order:
        ds_cfg = datasets[ds_key]
        group_labels.append(ds_cfg["name"])
        for ck in cond_keys:
            values[ck].append(load_metric(ds_cfg, ds_cfg[ck]))

    n_groups = len(order)
    x = np.arange(n_groups)
    width = 0.38

    fig, ax = plt.subplots(figsize=(2.6 * n_groups + 2, 5))
    for i, ck in enumerate(cond_keys):
        offset = (i - (len(cond_keys) - 1) / 2) * width
        vals = values[ck]
        plot_vals = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(
            x + offset,
            plot_vals,
            width,
            label=cond_style[ck]["label"],
            color=cond_style[ck]["color"],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v is None:
                bar.set_hatch("xxx")
                bar.set_alpha(0.35)
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "no result",
                        rotation=90, ha="center", va="bottom", fontsize=7, color="#444")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(cfg.get("ylabel", "Score"))
    ax.set_title(cfg.get("suptitle", "Hand-tuned vs SSVM-fit"), fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    output_path = Path(args.output or cfg["output_png"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
