"""Collect the metrics of a parameter sweep launched by ``launch_sweep.py``.

Reads the sweep's ``manifest.toml``, pulls each run's ``track_metrics.json`` from
``<eval_base_dir>/evaluation/<experiment>/<dataset>/<exp_uid>/``, writes
``results.csv`` next to the manifest, and prints a table sorted by the primary
metric.

It also reports, per condition and per swept parameter, whether the winning value
sits at an **edge** of the range that was sampled. An edge winner means the
optimum has not been bracketed yet and the sweep should be extended in that
direction; an interior winner means it is safe to stop.

Usage::

    conda run -n mhat2 python scripts/05_evaluation/collect_sweep.py \\
        experiments/tracking/Fluo-C3DL-MDA231/01_cells/sweeps/<sweep_id>/manifest.toml \\
        [--baseline TRA DET LNK] [--metric-set ctc|overlap]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import toml

# The ILP cost knobs a sweep is allowed to tune (CLAUDE.md). Used only to report
# which of them a condition never varied.
TUNABLE_KEYS = {
    "drift_weight", "drift_constant", "area_weight", "area_constant",
    "intensity_weight", "intensity_constant", "curvature_weight", "curvature_constant",
    "cohesion_weight", "cohesion_constant", "adhesion_weight", "adhesion_constant",
    "appear_constant", "disappear_constant", "base_edge_constant",
}

METRIC_SETS = {
    # (json group, [(column name, key)...]); first column is the primary metric.
    "ctc": (
        "CTCMetrics",
        [
            ("TRA", "TRA"),
            ("DET", "DET"),
            ("LNK", "LNK"),
            ("fp", "fp_nodes"),
            ("fn", "fn_nodes"),
            ("ns", "ns_nodes"),
            ("fn_e", "fn_edges"),
            ("fp_e", "fp_edges"),
        ],
    ),
    "overlap": (
        "TrackOverlapMetrics",
        [
            ("TE", "target_effectiveness"),
            ("Purity", "track_purity"),
        ],
    ),
}


def load_metrics(path, group, columns):
    if not path.is_file():
        return None
    with open(path) as handle:
        payload = json.load(handle)
    if group not in payload:
        return None
    values = payload[group]
    return {name: values.get(key) for name, key in columns}


def edge_report(rows, columns):
    """For each condition, flag swept parameters whose winner is at a range edge."""
    primary = columns[0][0]
    by_condition = {}
    for row in rows:
        if row["_metrics"] is None:
            continue
        by_condition.setdefault(row["condition"] or "(unnamed)", []).append(row)

    lines = []
    for condition, group_rows in sorted(by_condition.items()):
        scored = [r for r in group_rows if r["_metrics"].get(primary) is not None]
        if not scored:
            continue
        best = max(scored, key=lambda r: r["_metrics"][primary])
        # A parameter counts as swept if its *effective* value varies across the
        # condition's runs -- the reference run states no overrides at all, so
        # reading the override tables alone would miss one end of every range.
        swept = {}
        for row in scored:
            for key, value in row["effective"].items():
                # Only scalars can be swept; drift_distance and friends are lists.
                if isinstance(value, (int, float, str, bool)):
                    swept.setdefault(key, set()).add(value)
        verdicts = []
        for key, values in sorted(swept.items()):
            numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(numeric) < 2 or len(values) != len(numeric):
                continue
            won = best["effective"].get(key)
            if not isinstance(won, (int, float)):
                continue
            if won == min(numeric):
                verdicts.append(f"{key}={won} at LOW edge -> extend downward")
            elif won == max(numeric):
                verdicts.append(f"{key}={won} at HIGH edge -> extend upward")
            else:
                verdicts.append(f"{key}={won} interior -> bracketed")
        lines.append(
            f"  {condition}: best {best['label']} "
            f"{primary}={best['_metrics'][primary]:.4f} {best['overrides']}"
        )
        for verdict in verdicts:
            lines.append(f"      {verdict}")
        if not verdicts:
            lines.append("      (single point or non-numeric overrides -- nothing to bracket)")
        # A parameter that was never varied inside this condition produces no
        # verdict at all, so "everything bracketed" can hide an axis that simply
        # went untested -- which matters when the same axis moved the winner in
        # another condition.
        untested = sorted(
            key for key, values in swept.items()
            if len(values) == 1 and key in TUNABLE_KEYS
        )
        if untested:
            lines.append(f"      never varied here: {', '.join(untested)}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "manifest", nargs="+",
        help="one or more manifest.toml paths. Pass every round of a sweep at once: "
             "a parameter is only bracketed with respect to all the values tried, so "
             "judging convergence one round at a time re-flags ranges already closed.",
    )
    parser.add_argument("--metric-set", choices=sorted(METRIC_SETS), default="ctc")
    parser.add_argument("--metrics-filename", default="track_metrics.json")
    args = parser.parse_args()

    manifest_paths = [Path(p).resolve() for p in args.manifest]
    group, columns = METRIC_SETS[args.metric_set]

    rows = []
    for manifest_path in manifest_paths:
        manifest = toml.load(manifest_path)
        eval_root = (
            Path(manifest["eval_base_dir"]) / "evaluation"
            / manifest["experiment"] / manifest["dataset"]
        )
        for run in manifest["runs"]:
            metrics_path = eval_root / run["exp_uid"] / args.metrics_filename
            rows.append(
                {
                    "label": run["label"],
                    "condition": run.get("condition", ""),
                    "exp_uid": run["exp_uid"],
                    "overrides": run.get("overrides", {}),
                    "effective": run.get("effective", {}),
                    "_metrics": load_metrics(metrics_path, group, columns),
                }
            )

    primary = columns[0][0]
    done = [row for row in rows if row["_metrics"] is not None]
    pending = [row for row in rows if row["_metrics"] is None]
    done.sort(key=lambda r: (r["_metrics"].get(primary) is None, -(r["_metrics"].get(primary) or 0)))

    csv_path = manifest_paths[-1].parent / "results.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "label", "exp_uid", "overrides"] + [c[0] for c in columns])
        for row in done:
            writer.writerow(
                [row["condition"], row["label"], row["exp_uid"],
                 "; ".join(f"{k}={v}" for k, v in row["overrides"].items())]
                + [row["_metrics"].get(c[0]) for c in columns]
            )

    header = f"{'condition':<16} {'label':<18} " + " ".join(f"{c[0]:>8}" for c in columns)
    print(header)
    print("-" * len(header))
    for row in done:
        cells = []
        for name, _ in columns:
            value = row["_metrics"].get(name)
            if isinstance(value, float):
                cells.append(f"{value:>8.4f}")
            else:
                cells.append(f"{'' if value is None else value:>8}")
        print(f"{row['condition']:<16} {row['label']:<18} " + " ".join(cells))

    if pending:
        print(f"\n{len(pending)} run(s) with no metrics yet: "
              + ", ".join(row["label"] for row in pending))

    print("\nConvergence check (winner at range edge => keep sweeping):")
    for line in edge_report(rows, columns):
        print(line)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
