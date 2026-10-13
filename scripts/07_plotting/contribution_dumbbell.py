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

Optional table keys: ``full_condition`` in the ablation table (default
"baseline") and ``base_condition`` in the addition table (default "none") name
the two reference bars, for a dataset whose addition series starts from
something other than an empty model. A metric may give ``json_paths`` (summed)
in place of ``json_path``; one with ``higher_is_better = false`` keeps its true
signed deltas but has its x axis flipped, so improvement points right in every
panel.

For an addition series that is not "None + X" for every X (primary_nk_cells:
None -> Flow -> Flow + X), an addition condition may set ``added_to`` (the
condition its gain is measured from; default the base) and ``dumbbell_label``
(its row name, when the bar label is e.g. "Flow + Volume"), and the addition
table ``addition_legend`` (the open-circle legend text).

Usage:
    python scripts/07_plotting/contribution_dumbbell.py --ablation configs/evaluation/solver_ablation.toml \
        --addition configs/evaluation/solver_addition.toml --dataset mda231 [--output foo.png]
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import toml

import plot_spec


def walk(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def reduce_value(value, reduction):
    if reduction == "mean":
        return float(np.mean(value)) if isinstance(value, list) else float(value)
    return float(value)  # "scalar"


def load_metrics(dataset_cfg, cond, metric_specs):
    """The condition's metrics (its run and eval, see plot_spec.py). A named eval
    that is missing raises; a placeholder has no place in this figure either."""
    m = plot_spec.load_metrics(dataset_cfg, cond)
    if m is None:
        raise ValueError(f"condition {cond.get('label')!r} is still a placeholder")
    return {k: sum(reduce_value(walk(m, path), spec["reduction"])
                   for path in spec.get("json_paths", [spec.get("json_path")]))
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
    full_key = abl.get("full_condition", "baseline")
    base_key = add.get("base_condition", "none")
    full_cond = abl["conditions"][full_key]
    none_cond = add["conditions"][base_key]
    full = load_metrics(abl, full_cond, metric_specs)
    none = load_metrics(add, none_cond, metric_specs)

    # cost key -> {ablation deltas, addition deltas, label, color}
    costs = {}
    for key, cond in abl["conditions"].items():
        if key in (full_key, "no_all"):
            continue
        c = strip_prefix(key)
        m = load_metrics(abl, cond, metric_specs)
        # Signed change from removing the cost: metric(Full - X) - metric(Full).
        # Negative when removing hurts, so an essential cost spans a wide range
        # (big negative removal + big positive addition) and a useless one hugs 0.
        costs.setdefault(c, {})["abl"] = {k: m[k] - full[k] for k in metric_specs}
        costs[c].setdefault("label", nice_label(cond.get("label")))
        costs[c].setdefault("color", cond.get("color", "#444444"))
    for key, cond in add["conditions"].items():
        if key in (base_key, "full"):
            continue
        c = strip_prefix(key)
        m = load_metrics(add, cond, metric_specs)
        ref = none
        if "added_to" in cond:
            ref = load_metrics(add, add["conditions"][cond["added_to"]], metric_specs)
        costs.setdefault(c, {})["add"] = {k: m[k] - ref[k] for k in metric_specs}
        # prefer the addition label
        costs[c]["label"] = cond.get("dumbbell_label") or nice_label(cond.get("label"))
        costs[c].setdefault("color", cond.get("color", "#444444"))

    # Order rows by total spread (addition gain minus signed ablation change) on
    # the first metric, so the most essential cost is at the top.
    tra = list(metric_specs)[0]

    def spread(c):
        a = costs[c].get("abl", {}).get(tra, 0.0)
        d = costs[c].get("add", {}).get(tra, 0.0)
        return d - a  # add is +gain, abl is -loss -> essential = large spread

    # Explicit top-to-bottom order from the TOML (`cost_order`, stripped cost keys)
    # if given; any unlisted costs fall to the bottom, spread-sorted. Otherwise the
    # default is spread-sorted with the largest spread on top. y=0 is the bottom
    # row, so the plotting order is the reverse of top-to-bottom.
    cost_order = abl.get("cost_order")
    if cost_order:
        listed = [c for c in cost_order if c in costs]
        extras = sorted((c for c in costs if c not in listed), key=spread)
        order = list(reversed(listed + extras))
    else:
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
            if a is not None and d is not None:
                ax.plot([d, a], [i, i], color=col, lw=2, alpha=0.55, zorder=1)
            if d is not None:
                ax.scatter(d, i, s=90, facecolors="white", edgecolors=col,
                           linewidths=2, zorder=3)
            if a is not None:
                ax.scatter(a, i, s=95, color=col, zorder=3)
        ax.axvline(0, color="gray", lw=0.9, ls="--", alpha=0.7)
        title = spec["display_name"]
        if spec.get("higher_is_better", True):
            signs = "(− removed · + added)"
        else:
            # Lower is better: a helpful cost lowers the metric when added and
            # raises it when removed. Flip the axis so "better" is still right.
            signs = "(+ removed · − added)"
            ax.invert_xaxis()
        if "higher_is_better" in spec:
            title += " (↑)" if spec["higher_is_better"] else " (↓)"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Δ {spec['display_name']}   {signs}")
        ax.margins(x=0.18)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([costs[c]["label"] for c in order], fontsize=11)

    # The signs in the legend only hold where higher is better.
    flipped = any(not s.get("higher_is_better", True) for s in metric_specs.values())
    base_label = none_cond.get("label", "None")
    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor="#333", markeredgewidth=2, markersize=10,
                   label=add.get("addition_legend", f"added to {base_label} → gain")
                   + ("" if flipped else " (+)")),
        plt.Line2D([], [], marker="o", ls="", color="#333", markersize=10,
                   label="removed from Full → loss" + ("" if flipped else " (−)")),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=10)
    fig.suptitle(add.get("suptitle", "").replace("Addition", "Cost Contribution")
                 or "Cost Contribution", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()

    out = plot_spec.figure_path(
        abl, f"contribution_dumbbell_{args.dataset}.png", args.output,
        key="dumbbell_output_png",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
