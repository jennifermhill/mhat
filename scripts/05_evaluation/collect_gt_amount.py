"""Collect GT-amount sweep results into a single CSV.

Joins three things per run token:
  - test-set metrics   experiments/evaluation/<exp>/<test_dataset>/<token>/track_metrics.json
  - fit bookkeeping    experiments/tracking/<exp>/<train_dataset>/<token>/fit_summary.json
  - learned weights    .../<token>/learned_weights.toml
  - SSVM convergence   .../<token>/fit_weights_ssvm_*.log  (iteration count, final eps)

Runs that produced no usable result keep an explicit `status` (empty_solution,
fit_failed, not_converged, missing) alongside a 0.0 metric, so a degenerate fit
is never silently plotted as a genuine score of zero.

Usage:
    python scripts/05_evaluation/collect_gt_amount.py \
        configs/tracking/NC281-sparse-label/02_nuclei_denoised_train/gt_amount/gt_amount.toml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import toml

from mhat.tracking.gt_subsets import iter_run_dirs, parse_run_token, run_dir

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
]

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
        "--eval-dataset",
        "--test-dataset",
        dest="eval_dataset",
        default="02_nuclei_denoised_test",
        help="dataset whose evaluation/ results to collect; use the train dataset to "
        "gather the train-side (annotated-subset) scores used for model selection",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = toml.load(args.config)
    prefix = config.get("run_name_prefix", "gta")

    input_base_dir = Path(config["input_base_dir"])
    experiment = config["experiment"]
    train_root = input_base_dir / "tracking" / experiment / config["dataset"]
    output_base_dir = Path(config["output_base_dir"])
    eval_root = output_base_dir / "evaluation" / experiment / args.eval_dataset

    # Fits are read from the train dataset, metrics from whichever dataset was
    # scored — which may be the train one (3a) or the test one (3b), and the two
    # sides can be grouped under different subdirectories.
    runs_subdir = config.get("runs_subdir", "")
    test_runs_subdir = config.get("test_runs_subdir", "")
    eval_subdir = runs_subdir if args.eval_dataset == config["dataset"] else test_runs_subdir

    rows = []
    for fit_dir in iter_run_dirs(train_root, runs_subdir):
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

        metrics_path = run_dir(eval_root, eval_subdir, token) / "track_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            for name, path in METRIC_FIELDS:
                row[name] = walk_json_path(metrics, path)
        else:
            if row["status"] == "ok":
                # Distinguish "the weights gave an empty solution on the test set"
                # (a real, reportable outcome) from "this run was never evaluated".
                test_pred = (
                    run_dir(
                        input_base_dir / "tracking" / experiment / args.eval_dataset,
                        eval_subdir,
                        token,
                    )
                    / "pred_tracks.zarr"
                )
                row["status"] = "missing" if test_pred.is_dir() else "empty_solution_test"
            for name, _ in METRIC_FIELDS:
                row[name] = 0.0 if name == "target_effectiveness" else None

        rows.append(row)

    if not rows:
        print(f"No runs matching {prefix}_* found under {train_root / runs_subdir}")
        return

    fieldnames = (
        ["run", "arm", "n_tracks", "seed", "status"]
        + BUDGET_FIELDS
        + [name for name, _ in METRIC_FIELDS]
        + FIT_FIELDS
        + ["ssvm_iterations", "final_eps", "n_logfiles"]
        + WEIGHT_FIELDS
    )
    rows.sort(key=lambda r: (r["arm"], -int(r["n_tracks"]), r["seed"]))

    out_path = args.output or (eval_root / "gt_amount_results.csv")
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
    if args.eval_dataset == config["dataset"]:
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
