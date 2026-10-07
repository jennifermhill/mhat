import argparse
import json
from pathlib import Path

import geff
import toml

from mhat.dataset import Dataset, absolute_path, require_dir
from mhat.evaluation.evaluate_tracking import compute_ctc_seg, evaluate_tracking
from mhat.utils import spatial_axis_names


def report_axis_consistency(gt_tracks_path, pred_axes):
    """Report -- and only report -- a GT/prediction axis-name disagreement.

    A ground truth converted under different axes than the current prediction
    would have its coordinates matched against the wrong axes. Worth saying
    early and in context.

    This deliberately does not act. Hand-annotated ground truth (NC281,
    primary_nk_cells, ...) can be the ONLY copy of those annotations, so a
    disagreement is something to repair in place, never something to resolve
    by deleting the store.
    """
    if pred_axes is None:
        return
    gt_axes = geff.GeffMetadata.read(gt_tracks_path).axes
    if gt_axes is None:
        return
    gt_names = spatial_axis_names(gt_axes)
    pred_names = spatial_axis_names(pred_axes)
    if gt_names != pred_names:
        print(
            f"WARNING: ground truth at {gt_tracks_path} has spatial axes "
            f"{gt_names} but the prediction has {pred_names}. Matching them "
            f"would compare different coordinates. Repair the ground truth "
            f"store in place -- do NOT delete it, it may be the only copy of "
            f"those annotations."
        )


def gt_paths(config):
    """The ground-truth paths from the config, each checked to exist if given.

    Returns:
        (gt_tracks, gt_seg, ctc_seg_dir); the last two may be None.
    """
    gt_tracks = Path(config["gt_tracks"])
    if not gt_tracks.is_dir():
        raise FileNotFoundError(
            f"Ground-truth tracks {gt_tracks} are missing. For Cell Tracking "
            f"Challenge ground truth, create them first with "
            f"scripts/05_evaluation/convert_ctc_gt.py."
        )
    gt_seg = config.get("gt_seg")
    gt_seg = require_dir(Path(gt_seg), "Ground-truth segmentation") if gt_seg else None
    ctc_seg_dir = config.get("ctc_seg_dir")
    ctc_seg_dir = (
        require_dir(Path(ctc_seg_dir), "CTC SEG folder") if ctc_seg_dir else None
    )
    return gt_tracks, gt_seg, ctc_seg_dir


def run_evaluation(config, gt_tracks, gt_seg, ctc_seg_dir, pred_data_dir):

    pred_tracks_path = pred_data_dir / "pred_tracks.zarr"
    pred_axes = geff.GeffMetadata.read(pred_tracks_path).axes

    report_axis_consistency(gt_tracks, pred_axes)

    results = evaluate_tracking(config, gt_tracks, gt_seg, pred_data_dir)

    track_metrics = {}
    for metric in results:
        metric_name = metric['metric']['name']
        metric_results = metric['results']
        track_metrics[metric_name] = metric_results

    # CTC SEG uses the sparse `SEG` ground-truth folder (pixel-accurate,
    # per-slice), not the coarse `TRA` markers used for TRA/DET/LNK matching.
    if "ctc" in config.get("metrics", []):
        pred_seg_path = pred_data_dir / "pred_seg.zarr"
        if ctc_seg_dir is not None and pred_seg_path.exists():
            seg_score = compute_ctc_seg(ctc_seg_dir, pred_seg_path)
            if seg_score is not None:
                track_metrics.setdefault("CTCMetrics", {})["SEG"] = seg_score
        else:
            print(f"Skipping SEG: no ctc_seg_dir given, or no {pred_seg_path}")

    return track_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    ds = Dataset.from_config(config)
    track_uid = config["track_result"]
    pred_data_dir = require_dir(ds.track_dir(track_uid), "Tracking run")
    gt_tracks, gt_seg, ctc_seg_dir = gt_paths(config)

    track_metrics = run_evaluation(
        config, gt_tracks, gt_seg, ctc_seg_dir, pred_data_dir
    )

    # Evaluations live inside the run they score, one directory each, named
    # after the ground truth and the metrics so none overwrites another.
    gt_label = config.get("eval_name") or absolute_path(gt_tracks).parent.name
    metrics = config.get("metrics", ["basic", "track_overlap"])
    output_dir = ds.new_eval_dir(track_uid, gt_label, metrics)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "track_metrics.json", 'w') as f:
        json.dump(track_metrics, f)

    # Save the eval config used for this run alongside the results for provenance
    with open(output_dir / "eval_config.toml", "w") as config_file:
        toml.dump(config, config_file)
