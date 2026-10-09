"""Collect the metrics of a parameter sweep launched by ``launch_sweep.py``.

Reads the sweep's ``manifest.toml`` (or a ``launch_evals.py`` one), finds each
run's eval inside the run, writes ``results.csv`` next to the manifest, and prints
a table sorted by the primary metric.

An eval is found by its exact name, ``<eval_name>_<sorted metrics>_<timestamp>``,
with the ``eval_name`` and metrics the manifest records. Only an eval holding a
``track_metrics.json`` counts. None: the run is pending. More than one (an eval
repeated by hand under the sweep's id): the collector stops and lists them
rather than pick one. A ``manifest.migrated.toml`` (old sweeps, after the layout
migration) lists each run's evals directly: the one the old collector read
(``sweep_eval``) if it holds this metric set, else the one that does, and more
than one is refused the same way.

It also reports, per condition and per swept parameter, whether the winning value
sits at an **edge** of the range that was sampled. An edge winner means the
optimum has not been bracketed yet and the sweep should be extended in that
direction; an interior winner means it is safe to stop.

Usage::

    conda run -n mhat-cluster python scripts/05_evaluation/collect_sweep.py \\
        experiments/Fluo-C3DL-MDA231/01_cells/tracking/sweeps/<sweep_id>/manifest.toml \\
        [--baseline TRA DET LNK] [--metric-set ctc|overlap]

``--metric-set ctc`` reports TRA/DET/LNK for CTC-ground-truth datasets;
``--metric-set overlap`` reports TE/TF plus the edge and node recalls, for the
sparse-ground-truth datasets where precision and purity are not optimizable;
``--metric-set dro`` reports TE/TF with CTC fn/SEG/LNK diagnostics, for
Fluo-N3DL-DRO, whose GT is sparse *and* matched many-to-one (so no BasicMetrics).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import toml

# The ILP cost knobs a sweep is allowed to tune (CLAUDE.md). Used only to report
# which of them a condition never varied.
TUNABLE_KEYS = {
    "drift_weight", "drift_constant", "area_weight", "area_constant",
    "intensity_weight", "intensity_constant", "curvature_weight", "curvature_constant",
    "cohesion_weight", "cohesion_constant", "adhesion_weight", "adhesion_constant",
    "appear_constant", "disappear_constant", "base_edge_constant",
    "division_weight",
}

METRIC_SETS = {
    # [(column name, json group, key)...]; first column is the primary metric,
    # used for sorting and for picking each condition's winner. Each column names
    # its own group, because a sparse-GT run is judged on TrackOverlapMetrics and
    # BasicMetrics together.
    "ctc": [
        ("TRA", "CTCMetrics", "TRA"),
        ("DET", "CTCMetrics", "DET"),
        ("LNK", "CTCMetrics", "LNK"),
        ("fp", "CTCMetrics", "fp_nodes"),
        ("fn", "CTCMetrics", "fn_nodes"),
        ("ns", "CTCMetrics", "ns_nodes"),
        ("fn_e", "CTCMetrics", "fn_edges"),
        ("fp_e", "CTCMetrics", "fp_edges"),
    ],
    # Sparse-GT datasets (NC281-Fl2mSiH2B, primary_nk_cells): precision, purity
    # and F1 are unreliable because unannotated objects count as FPs, so TE/TF
    # lead and the recalls are the diagnostics. Purity is kept last, reported but
    # not optimized.
    "overlap": [
        ("TE", "TrackOverlapMetrics", "target_effectiveness"),
        ("TF", "TrackOverlapMetrics", "track_fractions"),
        ("EdgeR", "BasicMetrics", "Edge Recall"),
        ("NodeR", "BasicMetrics", "Node Recall"),
        ("FN_e", "BasicMetrics", "False Negative Edges"),
        ("Purity", "TrackOverlapMetrics", "track_purity"),
    ],
    # Fluo-N3DL-DRO: sparse GT *and* a many-to-one CTC matcher, so BasicMetrics
    # (which requires one-to-one matching) cannot exist here and TRA/DET/purity
    # are artifacts of the 222k unannotated-but-correct objects. TE leads; the
    # fn_* columns are diagnostics only -- note the edge report assumes
    # higher-is-better, which holds for the first column and not for those.
    "dro": [
        ("TE",   "TrackOverlapMetrics", "target_effectiveness"),
        ("TF",   "TrackOverlapMetrics", "track_fractions"),
        ("fn_n", "CTCMetrics", "fn_nodes"),
        ("fn_e", "CTCMetrics", "fn_edges"),
        ("SEG",  "CTCMetrics", "SEG"),
        ("LNK",  "CTCMetrics", "LNK"),
    ],
}


EVAL_TIMESTAMP = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"


def find_eval(run_dir, eval_name, metrics):
    """The one eval of ``run_dir`` named ``<eval_name>_<metrics>_<timestamp>``.

    Returns None if there is none with a ``track_metrics.json`` yet. The name is
    matched exactly, not as a prefix: ids and metric names contain underscores,
    so ``regenF01_*`` would also match ``regenF01_baseline_*``.

    Raises:
        SystemExit: If more than one eval matches.
    """
    pattern = re.compile(
        rf"{re.escape(eval_name)}_{re.escape('+'.join(sorted(metrics)))}_{EVAL_TIMESTAMP}"
    )
    parent = Path(run_dir) / "evaluation"
    if not parent.is_dir():
        return None
    found = sorted(
        entry for entry in parent.iterdir()
        if entry.is_dir() and pattern.fullmatch(entry.name)
        and (entry / "track_metrics.json").is_file()
    )
    if len(found) > 1:
        raise SystemExit(
            f"{run_dir} holds {len(found)} evals named {eval_name}_{'+'.join(sorted(metrics))}"
            f"_<time>: {[e.name for e in found]}. Decide which one counts (move the other "
            "out of evaluation/) and collect again."
        )
    return found[0] if found else None


def run_metrics_path(manifest, run, columns):
    """Path of a run's track_metrics.json, or None while the run is pending."""
    output_dir = Path(manifest["output_dir"])
    if "new_run" in run:  # manifest.migrated.toml: the evals are listed
        # The eval the old collector read for this run, when the migration knew it.
        sweep_eval = run.get("sweep_eval")
        if sweep_eval:
            path = output_dir / sweep_eval / "track_metrics.json"
            if load_metrics(path, columns) is not None:
                return path
        usable = [
            output_dir / e / "track_metrics.json" for e in run.get("new_evals", [])
            if load_metrics(output_dir / e / "track_metrics.json", columns) is not None
        ]
        if len(usable) > 1:
            raise SystemExit(
                f"run {run['new_run']} has {len(usable)} migrated evals with these "
                f"metrics: {[str(p.parent.name) for p in usable]}. Pick one."
            )
        return usable[0] if usable else None
    run_dir = output_dir / manifest.get("stage", "tracking") / run["exp_uid"]
    found = find_eval(run_dir, manifest["eval_name"], manifest["metrics"])
    return found / "track_metrics.json" if found else None


def read_manifest(path):
    manifest = toml.load(path)
    runs = manifest.get("runs", [])
    if "output_dir" not in manifest and runs and "new_run" in runs[0]:
        # A migrated manifest sits at <output_dir>/<stage>/sweeps/<id>/: take the
        # output directory from there, so it also collects on a copy of the tree.
        manifest["output_dir"] = str(Path(os.path.abspath(path)).parents[3])
        return manifest
    if "output_dir" not in manifest:
        migrated = Path(path).with_name("manifest.migrated.toml")
        raise SystemExit(
            f"{path} was written for the old results layout. After the layout "
            f"migration, collect {migrated} instead."
        )
    return manifest


def load_metrics(path, columns):
    if not path.is_file():
        return None
    with open(path) as handle:
        payload = json.load(handle)
    # A file that has none of the groups this metric set reads is as good as
    # missing -- report it as pending rather than as a row of blanks.
    if not any(group in payload for _, group, _ in columns):
        return None
    return {
        name: payload.get(group, {}).get(key) for name, group, key in columns
    }


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
    args = parser.parse_args()

    manifest_paths = [Path(p).resolve() for p in args.manifest]
    columns = METRIC_SETS[args.metric_set]

    rows = []
    for manifest_path in manifest_paths:
        manifest = read_manifest(manifest_path)
        for run in manifest["runs"]:
            metrics_path = run_metrics_path(manifest, run, columns)
            rows.append(
                {
                    "label": run["label"],
                    "condition": run.get("condition", ""),
                    "exp_uid": run["exp_uid"],
                    "overrides": run.get("overrides", {}),
                    "effective": run.get("effective", {}),
                    "_metrics": load_metrics(metrics_path, columns) if metrics_path else None,
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
        for name, *_ in columns:
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
