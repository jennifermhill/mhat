import json
import shutil
import argparse
from pathlib import Path

import toml
import geff
import zarr

from mhat.evaluation.diagnostics import run_diagnostics
from mhat.evaluation.evaluate_tracking import compute_ctc_seg, evaluate_tracking
from mhat.evaluation.from_ctc_to_geff import from_ctc_to_geff


def run_evaluation(config, gt_data_dir, pred_data_dir):

    gt_tracks_path = gt_data_dir / "correct_tracks.zarr"
    pred_tracks_path = pred_data_dir / "pred_tracks.zarr"

    (pred_graph, pred_metadata) = geff.read(pred_tracks_path)

    ctc_gt = config.get("ctc_gt", "01_GT")

    # Check for CTC metrics
    if "ctc" in config.get("metrics", []):
        if not gt_tracks_path.is_dir():
            print(f"Converting GT tracks to geff format at {gt_tracks_path}")
            axes = pred_metadata.axes
            from_ctc_to_geff(
                ctc_path=gt_data_dir / ctc_gt / "TRA",
                geff_path=gt_tracks_path,
                segmentation_store=gt_data_dir / "correct_seg.zarr",
                axes=axes,
            )

    results, matched = evaluate_tracking(
        config,
        gt_data_dir,
        pred_data_dir,
        return_matched=True,
    )

    track_metrics = {}
    for metric in results:
        metric_name = metric['metric']['name']
        metric_results = metric['results']
        track_metrics[metric_name] = metric_results

    # CTC SEG uses the sparse `SEG` ground-truth folder (pixel-accurate,
    # per-slice), not the coarse `TRA` markers used for TRA/DET/LNK matching.
    if "ctc" in config.get("metrics", []):
        seg_gt_dir = gt_data_dir / ctc_gt / "SEG"
        pred_seg_path = pred_data_dir / "pred_seg.zarr"
        if seg_gt_dir.is_dir() and pred_seg_path.exists():
            seg_score = compute_ctc_seg(seg_gt_dir, pred_seg_path)
            if seg_score is not None:
                track_metrics.setdefault("CTCMetrics", {})["SEG"] = seg_score
        else:
            print(f"Skipping SEG: missing {seg_gt_dir} or {pred_seg_path}")

    return track_metrics, matched


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

    track_metrics, matched = run_evaluation(config, gt_data_dir, pred_data_dir)
    tracksfile = output_dir / "track_metrics.json"
    with open(tracksfile, 'w') as f:
        json.dump(track_metrics, f)

    tracking_config = pred_data_dir / "config.toml"
    if tracking_config.is_file():
        shutil.copy2(tracking_config, output_dir / "tracking_config.toml")

    # Save the eval config used for this run alongside the results for provenance
    with open(output_dir / "eval_config.toml", "w") as config_file:
        toml.dump(config, config_file)

    # FN edge / FN node / GT drift diagnostics, reusing the matching computed above
    run_diagnostics(config, gt_data_dir, pred_data_dir, output_dir, matched=matched)