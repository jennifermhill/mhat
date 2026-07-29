"""Compare raw vs flow-corrected drift distances for GT edges.

Reports the displacement the ILP would see for a perfect linking, both without
flow and using the optical flow field(s) of the tracking run — Farneback and
Lucas-Kanade side by side when both exist for that flow result.

This runs automatically as part of ``evaluate_tracks.py``; use this script to
re-run it on its own.

Usage: conda run -n mhat2 --no-capture-output python scripts/05_evaluation/compare_gt_drift.py scripts/05_evaluation/eval_config.toml
"""
import argparse
from pathlib import Path

import toml

from mhat.evaluation.diagnostics import DIAGNOSTICS_SUBDIR, load_run_config
from mhat.evaluation.gt_drift import compare_gt_drift

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--no-plots", action="store_true", help="Skip the histogram PNG.")
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

    print(f"Comparing GT drift for {experiment}/{dataset}/{track_result}")

    compare_gt_drift(
        config,
        gt_data_dir,
        pred_data_dir,
        load_run_config(pred_data_dir),
        output_dir=output_dir,
        plots=not args.no_plots,
    )
    print(f"\nReport written to {output_dir}")
