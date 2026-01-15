import argparse
import json
from pathlib import Path

import toml
import zarr
from mhat.evaluation.eval_io import check_video_dir
from mhat.evaluation.evaluate_segmentation import (
    evaluate_segmentation,
    save_seg_results,
)
from mhat.evaluation.evaluate_tracking import evaluate_tracking
from mhat.tracking.tracks_io import load_tracks_from_csv, read_gt_tracks


def run_evaluation(config, gt_data_dir: Path, pred_data_dir: Path):
    """
    Args:
        config (_type_): _description_
        data_dir (_type_): directory containing data.zarr and {exp_name}/pred_tracks.csv
    """
    exp_name: str = config["exp_name"]
    gt_zarr_path = gt_data_dir / "data.zarr"
    pred_zarr_path = pred_data_dir / "data.zarr"
    assert gt_zarr_path.is_dir(), "data.zarr is missing from {zarr_path}"
    assert pred_zarr_path.is_dir(), "data.zarr is missing from {zarr_path}"

    gt_zarr_root = zarr.open(gt_zarr_path)
    gt_seg_group = "mask"
    assert (
        gt_seg_group in gt_zarr_root
    ), f"Group {gt_seg_group} missing from zarr {gt_zarr_path}"

    pred_zarr_root = zarr.open(pred_zarr_path)
    pred_seg_group = exp_name + "_pred_mask"
    assert (
        pred_seg_group in pred_zarr_root
    ), f"Group {pred_seg_group} missing from zarr {pred_zarr_path}"

    assert pred_data_dir.is_dir(), f"experiment dir {pred_data_dir} is missing"
    pred_tracks_csv = pred_data_dir / exp_name / "pred_tracks.csv"
    assert pred_tracks_csv.is_file(), f"Tracks file {pred_tracks_csv} is missing"
    pred_tracks = load_tracks_from_csv(pred_tracks_csv)

    gt_tracks_file = gt_data_dir / "gt_tracks.csv"
    assert gt_tracks_file.is_file(), f"GT tracks file {gt_tracks_file} is missing"
    gt_tracks = read_gt_tracks(gt_zarr_path, gt_tracks_file)

    # evaluate segmentation
    gt_mask = gt_zarr_root[gt_seg_group][:]
    pred_mask = pred_zarr_root[pred_seg_group][:]
    seg_results = evaluate_segmentation(gt_mask, pred_mask, config["iou_threshold"])

    # evaluate tracking
    tracking_results = evaluate_tracking(
        gt_tracks, gt_mask, pred_tracks, pred_mask, config["iou_threshold"]
    )
    return seg_results, tracking_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    config = toml.load(args.config)
    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    dataset: str = config["dataset"]
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    input_data_dir = input_base_dir / dataset
    assert input_data_dir.is_dir()

    exp_name: str = config["exp_name"]

    for video_dir in input_data_dir.iterdir():
        if check_video_dir(video_dir):
            vid_name = video_dir.stem
            out_data_dir = output_base_dir / dataset / vid_name
            out_exp_dir = out_data_dir / exp_name
            assert out_exp_dir.is_dir(), f"Out dir {out_exp_dir} is missing"
            seg_results, track_results = run_evaluation(config, video_dir, out_data_dir)
            segfile = out_exp_dir / "segmentation_metrics.csv"
            save_seg_results(seg_results, segfile)
            tracksfile = out_exp_dir / "tracking_metrics.json"
            with open(tracksfile, "w") as f:
                json.dump(track_results, f)
