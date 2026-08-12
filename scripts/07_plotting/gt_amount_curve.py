"""GT-amount learning curve: test-set score vs how much ground truth was annotated.

One line per annotation arm, mean over seeds with error bars, individual seed draws
scattered behind. The full-GT run is a single seed-independent point shared by
every arm and is drawn without an error bar.

The x-axis is `x_field` in the figure config, default `n_tracks`. Set it to
`n_gt_nodes_annotated` to put the sparse arms (A/B, parameterized by track count)
and the dense-crop arm (C, parameterized by volume fraction) on one axis — the
annotated object-frame count is the budget they actually share.

Runs that did not produce a usable result (empty solution, failed fit) are drawn
as open markers just above 0 and excluded from the mean — a degenerate fit is not
a score of zero, and averaging it in would hide the failure.

Pass several metrics to put them side by side in one figure; every panel shows
both arms.

One config can describe SEVERAL datasets, under `[datasets.<name>]`. Keys at the
top level (arms, x_field, xticks, xlabel) are shared; each dataset table supplies
its own csv_path / output_png / metric_column / baselines, and overrides anything
shared. Each dataset renders into its own experiment tree, so a single no-flag
invocation regenerates every figure the experiment owns. A config with no
`[datasets]` table is treated as one unnamed dataset, which is the original schema.

Config schema (TOML) -- see configs/evaluation/gt_amount_curve.toml.

Usage:
    # every dataset in the config: curve + panels each
    python scripts/07_plotting/gt_amount_curve.py configs/evaluation/gt_amount_curve.toml
    # one dataset only
    python scripts/07_plotting/gt_amount_curve.py <cfg> --dataset Fluo-C3DL-MDA231
    # ad-hoc metric set (needs a single dataset, or it would collapse two figures
    # onto one path)
    python scripts/07_plotting/gt_amount_curve.py <cfg> --dataset NC281-sparse-label \
        --metrics target_effectiveness edge_recall track_purity --output panels.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml

OK_STATUSES = {"ok", "not_converged"}

DEFAULT_METRIC_LABELS = {
    "target_effectiveness": "Target effectiveness",
    "track_purity": "Track purity",
    "track_fractions": "Track fractions (mean)",
    "node_recall": "Node recall",
    "edge_recall": "Edge recall",
    "node_f1": "Node F1",
    "edge_f1": "Edge F1",
    # CTC family — datasets matched with CTCMatcher report these instead.
    "tra": "Tracking (TRA)",
    "det": "Detection (DET)",
    "lnk": "Linking (LNK)",
    "seg": "Segmentation (SEG)",
}


def load_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_key(row):
    """The condition a row was run at — what its seeds should be averaged over.

    Never the x value itself: a budget-sized crop realizes a slightly different node
    count per seed, so keying on x would put every seed in its own group and average
    nothing. Budget-sized crops group on their requested budget, fraction-sized crops
    on the fraction, sparse runs on track count — so the protocols never collide even
    when they land on the same x.
    """
    target = as_float(row.get("crop_target_gt_nodes"))
    if target is not None:
        return ("crop_budget", target)
    fraction = as_float(row.get("crop_fraction"))
    if fraction is not None:
        return ("crop_fraction", fraction)
    return ("tracks", as_float(row.get("n_tracks")))


def draw_metric(ax, rows, metric, cfg, errorbar="sem", annotate_full=True):
    """Draw every arm's curve for one metric onto `ax`. Returns legend handles."""
    arms_cfg = cfg["arms"]
    # The sparse arms are parameterized by track count and the dense-crop arm by
    # volume fraction, so the only axis they share is the annotation budget they
    # actually spent: n_gt_nodes_annotated (objects x frames drawn). Set
    # x_field = "n_gt_nodes_annotated" in the figure config to compare them.
    x_field = cfg.get("x_field", "n_tracks")

    def x_of(row):
        value = as_float(row.get(x_field))
        if value is None:
            raise SystemExit(
                f"run {row.get('run')} has no {x_field!r}; re-run collect_gt_amount.py "
                "so the CSV carries the annotation-budget columns"
            )
        return value

    full_rows = [r for r in rows if r["arm"] == "full"]
    full_n = x_of(full_rows[0]) if full_rows else None
    full_value = as_float(full_rows[0].get(metric)) if full_rows else None

    handles = []
    for arm_index, (arm_key, style) in enumerate(arms_cfg.items()):
        arm_rows = [r for r in rows if r["arm"] == arm_key]
        if not arm_rows and full_value is None:
            continue

        # Seeds are grouped by the CONDITION they were run at, never by their x value.
        # A crop's annotated-node budget is measured, not requested, so every seed of
        # one crop fraction lands on a slightly different x; keying on x would put each
        # seed in its own group and silently average nothing.
        by_group = defaultdict(list)
        failed = defaultdict(list)
        for row in arm_rows:
            value = as_float(row.get(metric))
            key = group_key(row)
            if row["status"] in OK_STATUSES and value is not None:
                by_group[key].append((x_of(row), value))
            else:
                failed[key].append((x_of(row), row["seed"]))

        # Each condition is summarised at the mean of its seeds' budgets.
        summary = {}
        for key, pairs in by_group.items():
            group_x = np.asarray([p[0] for p in pairs], dtype=float)
            group_y = np.asarray([p[1] for p in pairs], dtype=float)
            if group_y.size < 2:
                err = 0.0
            elif errorbar == "sem":
                err = float(group_y.std(ddof=1) / np.sqrt(group_y.size))
            else:
                err = float(group_y.std(ddof=1))
            summary[key] = (float(group_x.mean()), float(group_y.mean()), err, group_x, group_y)

        ordered = sorted(summary, key=lambda k: summary[k][0])
        xs = [summary[k][0] for k in ordered]
        means = [summary[k][1] for k in ordered]
        errs = [summary[k][2] for k in ordered]

        # attach the shared full-GT endpoint to every arm's line
        line_x, line_y, line_e = list(xs), list(means), list(errs)
        if full_value is not None and full_n is not None and full_n not in xs:
            line_x.append(full_n)
            line_y.append(full_value)
            line_e.append(0.0)
            order = np.argsort(line_x)
            line_x = [line_x[i] for i in order]
            line_y = [line_y[i] for i in order]
            line_e = [line_e[i] for i in order]

        container = ax.errorbar(
            line_x,
            line_y,
            yerr=line_e,
            label=style.get("label", arm_key),
            color=style.get("color", None),
            marker=style.get("marker", "o"),
            markersize=6,
            markerfacecolor=(
                "white" if style.get("linestyle", "-") != "-" else style.get("color", None)
            ),
            capsize=3,
            linewidth=2,
            linestyle=style.get("linestyle", "-"),
            zorder=3,
        )
        handles.append(container)

        # individual seed draws at their own budgets, jittered only where seeds share
        # an x exactly (the track arms) so overlapping points stay readable
        rng = np.random.default_rng(0)
        seed_alpha = 0.25 if style.get("linestyle", "-") == "-" else 0.15
        for key in ordered:
            _, _, _, group_x, group_y = summary[key]
            if np.ptp(group_x) == 0:
                group_x = group_x * (1 + 0.012 * rng.standard_normal(group_x.size))
            ax.scatter(
                group_x, group_y, color=style.get("color", None), alpha=seed_alpha, s=18, zorder=2
            )

        # failed runs: open markers just above 0, annotated with how many seeds
        # collapsed. Never averaged in.
        for key, entries in failed.items():
            fail_x = [e[0] for e in entries]
            n_ok = len(by_group.get(key, []))
            ax.scatter(
                fail_x,
                [0.012] * len(fail_x),
                facecolors="none",
                edgecolors=style.get("color", None),
                s=50,
                linewidths=1.5,
                zorder=4,
            )
            # Stagger by series so two conditions failing at the same budget don't
            # overwrite each other's label.
            ax.annotate(
                f"{len(entries)}/{len(entries) + n_ok} empty",
                (float(np.mean(fail_x)), 0.012),
                textcoords="offset points",
                xytext=(-4 + 8 * arm_index, 10),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.5,
                color=style.get("color", None),
            )

    if full_value is not None and annotate_full:
        ax.annotate(
            f"full GT ({full_n:g})",
            (full_n, full_value),
            textcoords="offset points",
            xytext=(-8, -16),
            ha="right",
            fontsize=9,
            color="#333333",
        )

    # Reference lines only belong on the metric they were measured with.
    bases = [
        b
        for b in cfg.get("baselines", {}).values()
        if b.get("metric", "target_effectiveness") == metric
    ]
    # Two references can sit a few thousandths apart (on MDA231 the hand-tuned and
    # old-default-SSVM lines differ by 0.005), which puts their labels on top of each
    # other. Draw the lines where they belong but stack the LABELS upward whenever the
    # previous one would still be occupying that height.
    bases.sort(key=lambda b: b["value"])
    label_gap = 0.030  # in data units; ~1.5 line heights on a 0-1 axis
    last_label_y = None
    for base in bases:
        ax.axhline(
            base["value"],
            color=base.get("color", "#888888"),
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )
        label_y = base["value"]
        if last_label_y is not None and label_y - last_label_y < label_gap:
            label_y = last_label_y + label_gap
        last_label_y = label_y
        ax.annotate(
            f"{base['label']} ({base['value']:.3f})",
            (0.0, label_y),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(4, 3),
            ha="left",
            fontsize=8,
            color=base.get("color", "#888888"),
        )

    ticks = cfg.get("xticks")
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    if cfg.get("xscale") == "log":
        ax.set_xscale("log")
    default_xlabel = (
        "Annotated GT nodes used for fitting (objects × frames)"
        if x_field == "n_gt_nodes_annotated"
        else "Annotated GT tracks used for fitting"
    )
    ax.set_xlabel(cfg.get("xlabel", default_xlabel))
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    return handles


def resolve_datasets(cfg, selected):
    """Flatten the config into (name, dataset_cfg) pairs.

    A dataset's config is the shared top-level keys with its own table merged over
    them, so `arms` / `x_field` / `xticks` are written once and a dataset only
    states what differs. No `[datasets]` table means the whole file is one dataset,
    which is the pre-existing single-dataset schema.
    """
    shared = {k: v for k, v in cfg.items() if k != "datasets"}
    tables = cfg.get("datasets")
    if not tables:
        if selected:
            raise SystemExit("--dataset given but the config has no [datasets] table")
        return [(None, shared)]

    unknown = [name for name in (selected or []) if name not in tables]
    if unknown:
        raise SystemExit(
            f"unknown dataset(s) {unknown}; the config defines {sorted(tables)}"
        )
    return [
        (name, {**shared, **table})
        for name, table in tables.items()
        if not selected or name in selected
    ]


def render(cfg, metrics, output_path, errorbar) -> None:
    """Draw one figure — one panel per metric — and save it."""
    rows = load_rows(Path(cfg["csv_path"]))
    if not rows:
        raise SystemExit(f"no rows in {cfg['csv_path']}")

    labels = {**DEFAULT_METRIC_LABELS, **cfg.get("metric_labels", {})}

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6.5 * len(metrics), 5), squeeze=False, sharey=True
    )
    axes = axes[0]

    handles = []
    for ax, metric in zip(axes, metrics):
        handles = draw_metric(ax, rows, metric, cfg, errorbar=errorbar)
        if len(metrics) == 1:
            ax.set_ylabel(cfg.get("ylabel", labels.get(metric, metric)))
        else:
            ax.set_title(labels.get(metric, metric), fontsize=11)
    if len(metrics) > 1:
        axes[0].set_ylabel("Score on held-out test (higher is better)")

    if len(metrics) == 1:
        axes[0].legend(frameon=False, fontsize=10, loc="lower right")
    else:
        # one shared legend below the panels, so it never covers a curve
        fig.legend(
            handles=handles,
            frameon=False,
            fontsize=10,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, -0.04),
        )

    if cfg.get("suptitle"):
        fig.suptitle(cfg["suptitle"], fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--metric", default=None, help="single metric (back-compat)")
    parser.add_argument(
        "--metrics", nargs="+", default=None, help="several metrics -> one panel each"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--dataset", nargs="+", default=None, help="render only these [datasets.*] entries"
    )
    parser.add_argument("--errorbar", choices=("sem", "sd"), default="sem")
    args = parser.parse_args()

    cfg = toml.load(args.config)
    datasets = resolve_datasets(cfg, args.dataset)

    # --metrics / --metric / --output describe ONE figure. Applying them across
    # several datasets would either mislabel the axes or write both figures to the
    # same path, so require the caller to narrow the selection first.
    ad_hoc = args.metrics or args.metric or args.output
    if ad_hoc and len(datasets) > 1:
        raise SystemExit(
            "--metric/--metrics/--output apply to a single figure; narrow with "
            f"--dataset (config defines {[n for n, _ in datasets]})"
        )

    for name, dcfg in datasets:
        if name:
            print(f"--- {name} ---")
        if ad_hoc:
            metrics = args.metrics or [
                args.metric or dcfg.get("metric_column", "target_effectiveness")
            ]
            render(dcfg, metrics, args.output or dcfg["output_png"], args.errorbar)
            continue

        # The routine invocation: the headline curve, plus the panel figure when the
        # dataset declares one. Both are declared in the config, so regenerating an
        # experiment's figures never depends on remembering the right flags.
        render(
            dcfg,
            [dcfg.get("metric_column", "target_effectiveness")],
            dcfg["output_png"],
            args.errorbar,
        )
        panel_metrics = dcfg.get("panel_metrics")
        if panel_metrics:
            panels_png = dcfg.get("output_panels_png")
            if not panels_png:
                raise SystemExit(
                    f"{name or 'config'} sets panel_metrics but no output_panels_png"
                )
            render(dcfg, list(panel_metrics), panels_png, args.errorbar)


if __name__ == "__main__":
    main()
