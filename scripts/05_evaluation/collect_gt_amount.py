"""Collect GT-amount sweep results into a single CSV.

Joins, per run token of the fits sweep (``<output_dir>/ssvm/sweeps/<fits_id>/runs/``):

  - metrics            the eval named after the scored sweep, found by exact name:
                       ``--side test`` (default) the held-out solve in
                       ``<test output_dir>/tracking/sweeps/<solves_id>/runs/<token>``,
                       ``--side train`` the fit itself, scored on the GT it was fit on
  - fit bookkeeping    <fit>/fit_summary.json
  - learned weights    <fit>/learned_weights.toml
  - SSVM convergence   <fit>/fit_weights_ssvm_*.log  (iteration count, final eps)

Two status columns, because they can disagree:

  `status`        the outcome on the dataset being SCORED — this is what the curve
                  plots. `ok`, `not_converged`, `empty_solution_test`, `fit_failed`,
                  `missing`.
  `train_status`  what the fit itself reported, kept for diagnostics.

They differ for a fit that converged and wrote weights whose solution was empty on the
TRAIN graph. That is not evidence the same weights are degenerate on the test graph, so
such runs are solved there and reported with their real score; they are drawn as
failures only if the TEST solve is empty too. Runs with no usable result keep an
explicit status alongside a 0.0 score, so a degenerate fit is never silently plotted
as a genuine score of zero.

The CSV goes into the scored sweep's folder (``gt_amount_results.csv``) unless
``--output`` says otherwise.

Usage:
    python scripts/05_evaluation/collect_gt_amount.py <gt_amount_config.toml> [--side train]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import toml

from collect_sweep import find_eval

from mhat.tracking.gt_subsets import (
    check_gt_amount_config,
    iter_sweep_runs,
    parse_run_token,
    sweep_dir,
    sweep_id,
)

DEFAULT_EVAL_METRICS = ["basic", "track_overlap"]

METRIC_FIELDS = [
    ("target_effectiveness", ["TrackOverlapMetrics", "target_effectiveness"]),
    ("track_purity", ["TrackOverlapMetrics", "track_purity"]),
    ("track_fractions", ["TrackOverlapMetrics", "track_fractions"]),
    ("node_recall", ["BasicMetrics", "Node Recall"]),
    ("node_precision", ["BasicMetrics", "Node Precision"]),
    ("node_f1", ["BasicMetrics", "Node F1"]),
    ("edge_recall", ["BasicMetrics", "Edge Recall"]),
    ("edge_precision", ["BasicMetrics", "Edge Precision"]),
    ("edge_f1", ["BasicMetrics", "Edge F1"]),
    ("fp_nodes", ["BasicMetrics", "False Positive Nodes"]),
    ("fn_nodes", ["BasicMetrics", "False Negative Nodes"]),
    ("fn_edges", ["BasicMetrics", "False Negative Edges"]),
    # CTC family. A dataset scored with the CTC matcher cannot also run BasicMetrics
    # (incompatible matching types), so in practice a row carries one block or the
    # other and the unused one stays blank. The counts are prefixed because CTC and
    # BasicMetrics disagree about what an FP node is.
    ("tra", ["CTCMetrics", "TRA"]),
    ("det", ["CTCMetrics", "DET"]),
    ("lnk", ["CTCMetrics", "LNK"]),
    ("seg", ["CTCMetrics", "SEG"]),
    ("aogm", ["CTCMetrics", "AOGM"]),
    ("ctc_fp_nodes", ["CTCMetrics", "fp_nodes"]),
    ("ctc_fn_nodes", ["CTCMetrics", "fn_nodes"]),
    ("ctc_ns_nodes", ["CTCMetrics", "ns_nodes"]),
    ("ctc_fp_edges", ["CTCMetrics", "fp_edges"]),
    ("ctc_fn_edges", ["CTCMetrics", "fn_edges"]),
]

#: Fields that are SCORES in [0, 1]. An empty solution really does score 0 on these,
#: and writing 0.0 is what makes the plot draw it as a failed run rather than drop it.
#: Everything else in METRIC_FIELDS is a count or a cost, where 0 would be a lie
#: (an empty solution has zero false positives), so those stay blank.
SCORE_FIELDS = frozenset(
    {
        "target_effectiveness",
        "track_purity",
        "track_fractions",
        "node_recall",
        "node_precision",
        "node_f1",
        "edge_recall",
        "edge_precision",
        "edge_f1",
        "tra",
        "det",
        "lnk",
        "seg",
    }
)

WEIGHT_FIELDS = [
    "drift_weight",
    "drift_constant",
    "area_weight",
    "area_constant",
    "intensity_weight",
    "intensity_constant",
    "curvature_weight",
    "curvature_constant",
    "cohesion_weight",
    "cohesion_constant",
    "adhesion_weight",
    "adhesion_constant",
    "appear_constant",
    "disappear_constant",
]

FIT_FIELDS = [
    "n_fit_nodes",
    "n_fit_edges",
    "n_masked_nodes",
    "n_kept_gt_labels",
    "n_nodes_pos",
    "n_nodes_neg",
    "n_edges_pos",
    "n_edges_neg",
    "n_orphan_positives",
    # Labeled negatives overlapping no GT at all — "this detection is simply wrong"
    # supervision. 0 for every sparse run by construction; non-zero only for crops.
    "n_neg_no_gt_overlap",
    "n_solution_nodes",
    "n_solution_edges",
]

# Annotation-budget bookkeeping. `n_gt_nodes_annotated` is the only budget measure
# the sparse and dense-crop protocols share (an annotator pays per object per frame),
# so it is the x-axis the two arms can honestly be compared on — `n_tracks` counts a
# crop-truncated track the same as a full-length one.
BUDGET_FIELDS = [
    "kind",
    "n_gt_nodes_annotated",
    "crop_fraction",
    "crop_target_gt_nodes",
    "crop_membership",
]

_ITER_RE = re.compile(r"iteration\s+(\d+)")
_EPS_RE = re.compile(r"ε\s+is:\s*([-\d.eE+]+)")


def walk_json_path(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def parse_fit_log(fit_dir: Path):
    """Newest logfile's final bundle-method iteration count and epsilon."""
    logs = sorted(fit_dir.glob("fit_weights_ssvm_*.log"))
    if not logs:
        return None, None, 0
    text = logs[-1].read_text(encoding="utf-8", errors="replace")
    iters = _ITER_RE.findall(text)
    epss = _EPS_RE.findall(text)
    n_iter = int(iters[-1]) if iters else None
    final_eps = float(epss[-1]) if epss else None
    return n_iter, final_eps, len(logs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the GT-amount sweep config (train side)")
    parser.add_argument(
        "--side", choices=("test", "train"), default="test",
        help="test: the held-out solves' scores (the curve); train: each fit scored "
        "on the GT it was fit on, used for model selection",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = toml.load(args.config)
    check_gt_amount_config(config)
    prefix = config.get("run_name_prefix", "gta")
    metrics_set = config.get("eval_metrics", DEFAULT_EVAL_METRICS)

    # Fits are read from the train side; metrics from whichever sweep was scored:
    # the fits themselves (3a) or the held-out solves on the test movie (3b).
    fits = sweep_id(config, "fits_id")
    fits_dir = sweep_dir(config["output_dir"], "ssvm", fits)
    if args.side == "train":
        scored_dir, scored_id = fits_dir, fits
    else:
        if not config.get("test_template"):
            raise SystemExit("--side test needs test_template in the config")
        test_output_dir = toml.load(config["test_template"])["output_dir"]
        scored_id = sweep_id(config, "solves_id")
        scored_dir = sweep_dir(test_output_dir, "tracking", scored_id)

    rows = []
    for fit_dir in iter_sweep_runs(config["output_dir"], "ssvm", fits):
        parsed = parse_run_token(fit_dir.name, prefix)
        if not parsed:
            continue
        token = fit_dir.name
        row = {
            "run": token,
            "arm": parsed["arm"],
            # Crop tokens carry a volume fraction, not a track count; the realized
            # track count comes from the fit summary below.
            "n_tracks": parsed["n_tracks"],
            "crop_fraction": parsed["crop_fraction"],
            "crop_target_gt_nodes": parsed["crop_target_gt_nodes"],
            "seed": parsed["seed"],
            "status": "ok",
        }

        summary_path = fit_dir / "fit_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            row["status"] = summary.get("status", "ok")
            for key in FIT_FIELDS + BUDGET_FIELDS:
                if summary.get(key) is not None:
                    row[key] = summary[key]
            if row.get("n_tracks") is None:
                row["n_tracks"] = summary.get("n_tracks", summary.get("n_kept_gt_labels"))

        if row.get("n_tracks") is None:
            # Nothing downstream copes with a blank x value — merge and plotting both
            # cast it — so say why it is missing rather than emitting an empty cell.
            print(f"warning: {token} has no track count (missing fit_summary.json?)")
            row["n_tracks"] = 0

        learned_path = fit_dir / "learned_weights.toml"
        if learned_path.is_file():
            learned = toml.load(learned_path)
            for key in WEIGHT_FIELDS:
                if key in learned:
                    row[key] = learned[key]

        n_iter, final_eps, n_logs = parse_fit_log(fit_dir)
        row["ssvm_iterations"] = n_iter
        row["final_eps"] = final_eps
        row["n_logfiles"] = n_logs
        if n_logs > 1:
            print(f"warning: {token} has {n_logs} logfiles; using the newest")
        max_iter = config.get("ssvm_max_iter", 100)
        if row["status"] == "ok" and n_iter is not None and n_iter >= max_iter - 1:
            row["status"] = "not_converged"

        # `status` describes the outcome on the dataset being SCORED, because that is
        # what the curve plots; `train_status` preserves what the fit itself reported.
        # They differ for a fit that converged and wrote weights whose solution was
        # empty on the train graph: that is not evidence the same weights are empty on
        # the test graph, so the run is solved there and gets its real score. A run is
        # only drawn as a failure if it is empty on the dataset being plotted.
        row["train_status"] = row["status"]

        scored_run = scored_dir / "runs" / token
        found = find_eval(scored_run, scored_id, metrics_set)
        metrics_path = found / "track_metrics.json" if found else None
        if metrics_path is not None:
            metrics = json.loads(metrics_path.read_text())
            for name, path in METRIC_FIELDS:
                row[name] = walk_json_path(metrics, path)
            if row["status"] == "empty_solution":
                row["status"] = "ok"
        else:
            if row["status"] in ("ok", "empty_solution"):
                # Distinguish "the weights gave an empty solution on the test set"
                # (a real, reportable outcome) from "this run was never evaluated".
                scored_pred = scored_run / "pred_tracks.zarr"
                empty = "empty_solution_test" if args.side == "test" else "empty_solution"
                row["status"] = "missing" if scored_pred.is_dir() else empty
            for name, _ in METRIC_FIELDS:
                row[name] = 0.0 if name in SCORE_FIELDS else None

        rows.append(row)

    if not rows:
        print(f"No runs matching {prefix}_* found under {fits_dir / 'runs'}")
        return

    fieldnames = (
        ["run", "arm", "n_tracks", "seed", "status", "train_status"]
        + BUDGET_FIELDS
        + [name for name, _ in METRIC_FIELDS]
        + FIT_FIELDS
        + ["ssvm_iterations", "final_eps", "n_logfiles"]
        + WEIGHT_FIELDS
    )
    rows.sort(key=lambda r: (r["arm"], -int(r["n_tracks"]), r["seed"]))

    out_path = args.output or (scored_dir / "gt_amount_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")

    # Train-side numbers score each run against its own annotation, which is correct
    # but makes them non-comparable ACROSS protocols: the sparse arms keep whole
    # tracks while a crop truncates the ones leaving its box, and TE rewards short
    # tracks. Warn exactly where the mistake would be made — a CSV that mixes
    # protocols and is about to become one figure.
    if args.side == "train":
        kinds = {row.get("kind") for row in rows if row.get("kind")}
        if {"crop", "tracks"} <= kinds:
            # ASCII only: this goes to a Windows console under cp1252, where an
            # em-dash arrives as a replacement char and the warning reads as noise.
            print(
                "\nWARNING: this is a TRAIN-side table spanning both annotation "
                "protocols.\n"
                "  Those scores are each measured against a different shape of GT - "
                "sparse arms\n"
                "  keep whole tracks, crops truncate the ones leaving the box, and "
                "target_effectiveness\n"
                "  rewards shorter tracks. Use it for within-arm decisions "
                "(e.g. ssvm_reg), never to\n"
                "  rank arm C against A/B. The cross-arm comparison belongs on the "
                "held-out test split."
            )


if __name__ == "__main__":
    main()
