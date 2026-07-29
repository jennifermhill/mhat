"""Analyze properties of missed (FN) vs matched (TP) GT edges.

Compares edge attributes and ILP costs between true positive and false negative
GT edges to understand why certain edges are missed by the tracker, and what the
solution picked instead.

This runs automatically as part of ``evaluate_tracks.py``; use this script to
re-run it on its own (it recomputes the matching).

Usage: conda run -n mhat2 --no-capture-output python scripts/05_evaluation/analyze_fn_edges.py scripts/05_evaluation/eval_config.toml
"""
import argparse
from pathlib import Path

import toml

from mhat.evaluation.diagnostics import DIAGNOSTICS_SUBDIR, load_run_config
from mhat.evaluation.evaluate_tracking import match_tracking
from mhat.evaluation.fn_analysis import analyze_fn_edges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--no-plots", action="store_true", help="Skip the histogram PNGs.")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config.get("output_base_dir", config["input_base_dir"]))
    experiment = config["experiment"]
    dataset = config["dataset"]
    track_result = config["track_result"]

    gt_data_dir = input_base_dir / "tracking" / experiment / dataset
    pred_data_dir = gt_data_dir / track_result
    output_dir = output_base_dir / "evaluation" / experiment / dataset / track_result / DIAGNOSTICS_SUBDIR

    print(f"Analyzing FN edges for {experiment}/{dataset}/{track_result}")
    print(f"GT: {gt_data_dir}")
    print(f"Pred: {pred_data_dir}")

    matched = match_tracking(config, gt_data_dir, pred_data_dir)
    analyze_fn_edges(
        matched,
        pred_data_dir,
        load_run_config(pred_data_dir),
        output_dir=output_dir,
        plots=not args.no_plots,
    )
    print(f"\nReport written to {output_dir}")
