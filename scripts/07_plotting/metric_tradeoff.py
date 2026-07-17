"""Two-metric trade-off plot for the solver ablation + addition study.

Each condition is a point in a 2-metric plane (x vs y), coloured by a third
metric. Full/Baseline is a filled star, None an open star. Arrows go Full -> each
ablation (solid) and None -> each addition (dashed), so you can read which lever
each cost is: e.g. removing an edge cost pulls a point along the linking axis, a
node cost along the detection axis.

Metrics, axes and colour are read from the experiment TOML, so the same script
works for CTC datasets (DET / LNK / TRA) and overlap datasets (TE / Purity / TF).
Defaults: x=DET|TE, y=LNK|Purity, colour=TRA|TF; override with --x/--y/--color.

Usage:
    python metric_tradeoff.py --ablation solver_ablation.toml \
        --addition solver_addition.toml --dataset mda231 \
        [--x DET --y LNK --color TRA] [--mode both|ablation|addition] [--output f.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml


def walk(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def reduce_value(value, reduction):
    if reduction == "mean":
        return float(np.mean(value)) if isinstance(value, list) else float(value)
    return float(value)  # "scalar"


def mfile_for(cond, dataset_cfg):
    # per-condition override wins (e.g. sparse baseline/full store point-matched
    # metrics in track_metrics_point.json), else the dataset default.
    return cond.get("metrics_filename",
                    dataset_cfg.get("metrics_filename", "track_metrics.json"))


def load(dataset_cfg, uid, metric_specs, mfile):
    p = (Path(dataset_cfg["eval_base_dir"]) / dataset_cfg["experiment"]
         / dataset_cfg["dataset_dir"] / uid / mfile)
    if not p.is_file():
        print(f"warning: metrics missing for uid={uid}; omitting")
        return None
    with open(p) as f:
        m = json.load(f)
    return {k: reduce_value(walk(m, s["json_path"]), s["reduction"])
            for k, s in metric_specs.items()}


def annotate(ax, x, y, text, dx=0.0, dy=0.006):
    ax.annotate(text, (x, y), (x + dx, y + dy), fontsize=8.5, ha="center", va="bottom")


def pick_default(metric_specs, preferred, fallback_idx):
    for n in preferred:
        if n in metric_specs:
            return n
    keys = list(metric_specs)
    return keys[min(fallback_idx, len(keys) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--addition", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--x", default=None, help="metric key for x-axis")
    ap.add_argument("--y", default=None, help="metric key for y-axis")
    ap.add_argument("--color", default=None, help="metric key for colour")
    ap.add_argument("--mode", choices=["both", "ablation", "addition"], default="both")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    abl = toml.load(args.ablation)[args.dataset]
    add = toml.load(args.addition)[args.dataset]
    metric_specs = abl["metrics"]

    x_key = args.x or pick_default(metric_specs, ["DET", "TE"], 0)
    y_key = args.y or pick_default(metric_specs, ["LNK", "Purity"], 1)
    c_key = args.color or pick_default(metric_specs, ["TRA", "TF"], 2)
    for k in (x_key, y_key, c_key):
        if k not in metric_specs:
            raise SystemExit(f"metric {k!r} not in TOML metrics {list(metric_specs)}")
    xd = metric_specs[x_key]["display_name"]
    yd = metric_specs[y_key]["display_name"]
    cd = metric_specs[c_key]["display_name"]

    def load_pt(cfg, cond):
        v = load(cfg, cond["tracking_uid"], metric_specs, mfile_for(cond, cfg))
        if v is None:
            return (0.0, 0.0, 0.0), True
        return (v[x_key], v[y_key], v[c_key]), False

    full, full_miss = load_pt(abl, abl["conditions"]["baseline"])
    none, none_miss = load_pt(add, add["conditions"]["none"])

    abl_pts, add_pts = [], []
    if args.mode in ("both", "ablation"):
        for key, cond in abl["conditions"].items():
            if key in ("baseline", "no_all"):
                continue
            (x, y, c), miss = load_pt(abl, cond)
            abl_pts.append((cond["label"], x, y, c, miss))
    if args.mode in ("both", "addition"):
        for key, cond in add["conditions"].items():
            if key in ("none", "full"):
                continue
            (x, y, c), miss = load_pt(add, cond)
            add_pts.append((cond["label"], x, y, c, miss))

    # Colour scale over conditions that actually returned a result.
    present = ([full[2]] if not full_miss else []) + ([none[2]] if not none_miss else []) \
        + [p[3] for p in abl_pts if not p[4]] + [p[3] for p in add_pts if not p[4]]
    vmin, vmax = min(present), max(present)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Arrows first (under points). Skip when either endpoint has no result.
    for lbl, x, y, c, miss in abl_pts:
        if not miss and not full_miss:
            ax.annotate("", (x, y), (full[0], full[1]),
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2, alpha=0.8))
    for lbl, x, y, c, miss in add_pts:
        if not miss and not none_miss:
            ax.annotate("", (x, y), (none[0], none[1]),
                        arrowprops=dict(arrowstyle="->", color="#c8a0c8", lw=1.2,
                                        ls="--", alpha=0.8))

    # Missing runs would sit at (0,0) and collapse the axes, so omit + note them.
    omitted = []

    def scat(pts, marker):
        for lbl, x, y, c, miss in pts:
            if miss:
                omitted.append(lbl)
                continue
            ax.scatter(x, y, c=[c], cmap=cmap, vmin=vmin, vmax=vmax, s=150,
                       marker=marker, edgecolors="black", linewidths=0.7, zorder=3)
            annotate(ax, x, y, lbl)

    scat(abl_pts, "o")
    scat(add_pts, "^")

    if full_miss:
        omitted.append("Full/Baseline")
    else:
        ax.scatter(full[0], full[1], c=[full[2]], cmap=cmap, vmin=vmin, vmax=vmax,
                   s=520, marker="*", edgecolors="black", linewidths=1.2, zorder=4)
        annotate(ax, full[0], full[1], "Full / Baseline", dy=0.009)
    if none_miss:
        omitted.append("None")
    else:
        ax.scatter(none[0], none[1], facecolors="white", edgecolors="black",
                   s=430, marker="*", linewidths=1.2, zorder=4)
        annotate(ax, none[0], none[1], "None", dy=0.009)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, label=cd, shrink=0.85)

    ax.set_xlabel(xd, fontsize=12)
    ax.set_ylabel(yd, fontsize=12)
    base = abl.get("suptitle", "").replace("Ablation", "").strip() or args.dataset
    ax.set_title(f"{base}  {xd} – {yd} Trade-off", fontsize=14, fontweight="bold")
    if omitted:
        ax.text(0.02, 0.02, "no result (omitted): " + ", ".join(omitted),
                transform=ax.transAxes, fontsize=8.5, color="#b00000", va="bottom")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#555", markersize=10,
                   label="ablation (Full − cost)"),
        plt.Line2D([], [], marker="^", ls="", color="#555", markersize=10,
                   label="addition (None + cost)"),
        plt.Line2D([], [], marker="*", ls="", color="#555", markersize=15,
                   label="Full / None reference"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out = Path(args.output or (Path(abl["eval_base_dir"]) / abl["experiment"]
               / abl["dataset_dir"] / f"metric_tradeoff_{args.dataset}.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
