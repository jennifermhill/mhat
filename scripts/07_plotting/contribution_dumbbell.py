"""Marginal-contribution dumbbell plot for the solver ablation + addition study.

For each cost term it draws two markers on a signed Δ-metric axis:
  * addition marginal = metric(None + X) - metric(None)      [open circle]
  * ablation marginal = metric(Full)     - metric(Full - X)  [filled circle]
connected by a line. Both are "contribution" estimates; the GAP between them is
the interaction/redundancy effect that running both experiments was meant to
expose (invisible in the separate bar charts):

  both large ................ essential / independent
  add>>abl .................. redundant (others compensate in Full)
  add<<abl .................. synergistic (only pays off alongside others)
  both small ................ inert

Reads the SAME solver_ablation.toml / solver_addition.toml the bar figure uses,
so it works unchanged on the SSVM TOMLs later. Costs are matched between the two
files by stripping the no_/plus_ prefix (curvature has no ablation twin -> only
its addition dot is drawn).

Usage:
    python scripts/07_plotting/contribution_dumbbell.py --ablation configs/evaluation/solver_ablation.toml \
        --addition configs/evaluation/solver_addition.toml --dataset mda231 [--output foo.png]
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


MISSING = set()


def load_metrics(dataset_cfg, uid, metric_specs, mfile="track_metrics.json"):
    p = (
        Path(dataset_cfg["eval_base_dir"])
        / dataset_cfg["experiment"]
        / dataset_cfg["dataset_dir"]
        / uid
        / mfile
    )
    if not p.is_file():
        print(f"warning: metrics missing for uid={uid}; treating as zeros")
        MISSING.add(uid)
        return {k: 0.0 for k in metric_specs}
    with open(p) as f:
        m = json.load(f)
    return {k: reduce_value(walk(m, spec["json_path"]), spec["reduction"])
            for k, spec in metric_specs.items()}


def strip_prefix(key):
    for pre in ("no_", "plus_"):
        if key.startswith(pre):
            return key[len(pre):]
    return key


def nice_label(cond_label):
    return cond_label.lstrip("+-").strip() if cond_label else cond_label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--addition", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    abl = toml.load(args.ablation)[args.dataset]
    add = toml.load(args.addition)[args.dataset]
    metric_specs = abl["metrics"]

    # Reference points.
    full_cond = abl["conditions"]["baseline"]
    none_cond = add["conditions"]["none"]
    full_uid = full_cond["tracking_uid"]
    none_uid = none_cond["tracking_uid"]
    full = load_metrics(abl, full_uid, metric_specs, mfile_for(full_cond, abl))
    none = load_metrics(add, none_uid, metric_specs, mfile_for(none_cond, add))

    # cost key -> {ablation deltas, addition deltas, label, color}
    costs = {}
    for key, cond in abl["conditions"].items():
        if key in ("baseline", "no_all"):
            continue
        c = strip_prefix(key)
        m = load_metrics(abl, cond["tracking_uid"], metric_specs, mfile_for(cond, abl))
        # Signed change from removing the cost: metric(Full - X) - metric(Full).
        # Negative when removing hurts, so an essential cost spans a wide range
        # (big negative removal + big positive addition) and a useless one hugs 0.
        costs.setdefault(c, {})["abl"] = {k: m[k] - full[k] for k in metric_specs}
        costs[c]["abl_missing"] = cond["tracking_uid"] in MISSING
        costs[c].setdefault("label", nice_label(cond.get("label")))
        costs[c].setdefault("color", cond.get("color", "#444444"))
    for key, cond in add["conditions"].items():
        if key in ("none", "full"):
            continue
        c = strip_prefix(key)
        m = load_metrics(add, cond["tracking_uid"], metric_specs, mfile_for(cond, add))
        costs.setdefault(c, {})["add"] = {k: m[k] - none[k] for k in metric_specs}
        costs[c]["add_missing"] = cond["tracking_uid"] in MISSING
        costs[c]["label"] = nice_label(cond.get("label"))  # prefer addition label
        costs[c].setdefault("color", cond.get("color", "#444444"))
    none_missing = none_uid in MISSING

    # Order rows by total spread (addition gain minus signed ablation change) on
    # the first metric, so the most essential cost is at the top.
    tra = list(metric_specs)[0]

    def spread(c):
        a = costs[c].get("abl", {}).get(tra, 0.0)
        d = costs[c].get("add", {}).get(tra, 0.0)
        return d - a  # add is +gain, abl is -loss -> essential = large spread

    order = sorted(costs, key=spread)  # ascending -> largest spread on top

    n = len(metric_specs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 1, 0.7 * len(order) + 2.2), sharey=True)
    if n == 1:
        axes = [axes]

    y = np.arange(len(order))
    for ax, (key, spec) in zip(axes, metric_specs.items()):
        for i, c in enumerate(order):
            col = costs[c]["color"]
            a = costs[c].get("abl", {}).get(key)
            d = costs[c].get("add", {}).get(key)
            # Red X only when a dot's OWN run is missing. When just the None
            # reference is missing, the open (addition) dot still has data -- it
            # shows an absolute metric instead of a marginal (see footnote), so
            # we keep it as an open dot but drop the connecting line.
            d_missing = costs[c].get("add_missing", False)
            a_missing = costs[c].get("abl_missing", False)
            line_ok = (a is not None and d is not None and not d_missing
                       and not a_missing and not none_missing)
            if line_ok:
                ax.plot([d, a], [i, i], color=col, lw=2, alpha=0.55, zorder=1)
            if d is not None:
                if d_missing:
                    ax.scatter(d, i, s=70, marker="x", color="#d62728",
                               linewidths=2, zorder=3)
                else:
                    ax.scatter(d, i, s=90, facecolors="white", edgecolors=col,
                               linewidths=2, zorder=3)
            if a is not None:
                if a_missing:
                    ax.scatter(a, i, s=70, marker="x", color="#d62728",
                               linewidths=2, zorder=3)
                else:
                    ax.scatter(a, i, s=95, color=col, zorder=3)
        ax.axvline(0, color="gray", lw=0.9, ls="--", alpha=0.7)
        ax.set_title(spec["display_name"], fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Δ {spec['display_name']}   (− removed · + added)")
        ax.margins(x=0.18)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([costs[c]["label"] for c in order], fontsize=11)

    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor="#333", markeredgewidth=2, markersize=10,
                   label="added to None → gain (+)"),
        plt.Line2D([], [], marker="o", ls="", color="#333", markersize=10,
                   label="removed from Full → loss (−)"),
    ]
    if MISSING:
        handles.append(plt.Line2D([], [], marker="x", ls="", color="#d62728",
                                  markersize=9, markeredgewidth=2, label="no result"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=10)
    if none_missing:
        fig.text(0.5, -0.02,
                 "None baseline returned no result → treated as 0; addition dots "
                 "(open) therefore show absolute metric, not a true marginal.",
                 ha="center", fontsize=8, color="#666")
    fig.suptitle(add.get("suptitle", "").replace("Addition", "Cost Contribution")
                 or "Cost Contribution", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()

    out = Path(args.output or (Path(abl["eval_base_dir"]) / abl["experiment"]
               / abl["dataset_dir"] / f"contribution_dumbbell_{args.dataset}.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
