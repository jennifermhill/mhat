import json
import argparse
from pathlib import Path

import toml
import geff
import zarr

from mhat.evaluation.evaluate_tracking import evaluate_tracking
from mhat.evaluation.from_ctc_to_geff import from_ctc_to_geff


def run_evaluation(config, gt_data_dir, pred_data_dir):

    gt_tracks_path = gt_data_dir / "correct_tracks.zarr"
    pred_tracks_path = pred_data_dir / "pred_tracks.zarr"
    pred_segmentation_path = pred_data_dir / "pred_seg.zarr"
    if pred_segmentation_path.is_dir():
        pred_segmentation = zarr.open(pred_segmentation_path)
    else:
        print(f"Warning: Predicted segmentation zarr not found at {pred_segmentation_path}")
        pred_segmentation = None

        # Check for CTC metrics
    if "ctc" in config.get("metrics", []):
        # Check if gt has already been converted to geff format
        if not gt_tracks_path.is_dir():
            # If not convert to geff format
            print(f"Converting GT tracks to geff format at {gt_tracks_path}")
            from_ctc_to_geff(
                ctc_path=gt_data_dir / "01_GT" / "TRA",
                geff_path=gt_tracks_path,
                segmentation_store=gt_data_dir / "correct_seg.zarr",
            )

    (gt_graph, gt_metadata) = geff.read(gt_tracks_path)
    (pred_graph, pred_metadata) = geff.read(pred_tracks_path)

    results = evaluate_tracking(
        config,
        gt_graph, 
        pred_graph, 
        gt_segmentation=None,
        pred_segmentation=None,
    )

    track_metrics = {}
    for metric in results:
        metric_name = metric['metric']['name']
        metric_results = metric['results']
        track_metrics[metric_name] = metric_results

    return track_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    dataset: str = config["dataset"]
    experiment: str = config["experiment"]
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    gt_data_dir = input_base_dir / "tracking" / experiment / dataset
    assert gt_data_dir.is_dir(), f"GT data dir {gt_data_dir} is missing"

    pred_data_dir = input_base_dir / "tracking" / experiment / dataset / config["track_result"]
    assert pred_data_dir.is_dir(), f"Pred data dir {pred_data_dir} is missing"

    output_dir = output_base_dir / "evaluation" / experiment / dataset / config["track_result"]
    output_dir.mkdir(parents=True, exist_ok=True)

    track_metrics = run_evaluation(config, gt_data_dir, pred_data_dir)
    tracksfile = output_dir / "track_metrics.json"
    with open(tracksfile, 'w') as f:
        json.dump(track_metrics, f)