"""Analyze unmatched GT nodes to determine why they are FN.

For each GT node without a pred match, determines if:
1. No segmentation candidate exists at that location (segmentation miss)
2. A candidate exists but the ILP didn't select it (solver miss)
3. A candidate was selected but the matcher didn't match it (matcher miss)

This runs automatically as part of ``evaluate_tracks.py``; use this script to
re-run it on its own (it recomputes the matching).

Usage: conda run -n mhat2 --no-capture-output python scripts/05_evaluation/analyze_fn_nodes.py scripts/05_evaluation/eval_config.toml
"""
import argparse
from pathlib import Path

import toml

from mhat.evaluation.diagnostics import DIAGNOSTICS_SUBDIR, load_run_config
from mhat.evaluation.evaluate_tracking import match_tracking
from mhat.evaluation.fn_analysis import analyze_fn_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--proximity-threshold", type=float, default=None,
                        help="Max distance (world units) for an object to count as "
                             "being at the GT location.")
    parser.add_argument("--max-listed", type=int, default=None,
                        help="Cap on the number of FN nodes listed individually.")
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

    print(f"Analyzing FN nodes for {experiment}/{dataset}/{track_result}")

    matched = match_tracking(config, gt_data_dir, pred_data_dir)
    analyze_fn_nodes(
        matched,
        config,
        pred_data_dir,
        load_run_config(pred_data_dir),
        output_dir=output_dir,
        proximity_threshold=args.proximity_threshold,
        max_listed=args.max_listed,
    )
    print(f"\nReport written to {output_dir}")
