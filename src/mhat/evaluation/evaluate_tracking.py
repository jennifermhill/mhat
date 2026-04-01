from pathlib import Path

import geff
import numpy as np

from geff_spec import Axis
from traccuracy import TrackingGraph, run_metrics
import traccuracy.matchers as matchers
import traccuracy.metrics as metrics

from funtracks.import_export import import_from_geff

metrics_dict = {
    "basic": metrics.BasicMetrics,
    "cca": metrics.CellCycleAccuracy,
    "complete_tracks": metrics.CompleteTracks,
    "aogm": metrics.AOGMMetrics,
    "ctc": metrics.CTCMetrics,
    "division": metrics.DivisionMetrics,
    "track_overlap": metrics.TrackOverlapMetrics,
}

matchers_dict = {
    "point": matchers.PointMatcher,
    "point_to_seg": matchers.PointSegMatcher,
    "iou": matchers.IOUMatcher,
    "ctc": matchers.CTCMatcher,
}

def remap_seg_to_track_ids(geff_path, graph, segmentation):
    """Remap segmentation labels to match the graph's track_id attributes.

    import_from_geff renumbers track_id, so seg labels no longer match.
    This reads the raw geff to find which node property corresponds to the
    seg labels (via related_objects.label_prop), then remaps per frame.
    """
    (raw_graph, metadata) = geff.read(geff_path)

    # Determine which raw geff property maps to seg labels
    label_prop = "track_id"
    if metadata.related_objects:
        for ro in metadata.related_objects:
            if ro.type == "labels" and ro.label_prop:
                label_prop = ro.label_prop
                break

    remapped = np.zeros_like(segmentation)
    for node, raw_data in raw_graph.nodes(data=True):
        orig_label = raw_data.get(label_prop, node)
        new_track_id = graph.nodes[node]["track_id"]
        t = graph.nodes[node]["time"]
        mask = segmentation[t] == orig_label
        if mask.any():
            remapped[t][mask] = new_track_id
    return remapped


def evaluate_tracking(
    config, gt_data_dir: Path, pred_data_dir: Path
):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        config (dict): Evaluation configuration dictionary.
        gt_data_dir (Path): Path to ground truth data directory.
        pred_data_dir (Path): Path to predicted data directory.

    Returns:
        results (dict): Dictionary of metric results.
    """

    # Use import_from_geff to get graph and segmentation in correct format
    name_map = {
        "time": "time",
        "x": "x", 
        "y": "y",
        "z": "z",
        "id": "track_id",    # track_id stays constant across frames
    }

    # Read scale from metadata
    (_, metadata) = geff.read(pred_data_dir / "pred_tracks.zarr")
    axes = metadata.axes
    if axes is None:
        scale = [1.0, 1.0, 1.0, 1.0]  # Default to isotropic scaling if no axes info
    else:
        scale = [a.scale for a in axes if a.scale is not None]  # Extract scale values

    gt_seg_path = gt_data_dir / "correct_seg.zarr"
    gt_seg_path = gt_seg_path if gt_seg_path.exists() else None
    gt_tracks = import_from_geff(
        gt_data_dir / "correct_tracks.zarr",
        name_map,
        segmentation_path=gt_seg_path,
        scale=scale,
    )

    if gt_tracks.segmentation is not None:
        gt_seg = remap_seg_to_track_ids(
            gt_data_dir / "correct_tracks.zarr", gt_tracks.graph, gt_tracks.segmentation
        )
    else:
        gt_seg = None

    gt_graph = TrackingGraph(
        graph=gt_tracks.graph,
        frame_key="time",
        label_key="track_id",
        location_keys="pos",
        segmentation=gt_seg,
    )

    pred_seg_path = pred_data_dir / "pred_seg.zarr"
    pred_seg_path = pred_seg_path if pred_seg_path.exists() else None
    pred_tracks = import_from_geff(
        pred_data_dir / "pred_tracks.zarr",
        name_map,
        segmentation_path=pred_seg_path,
        scale=scale,
    )

    if pred_tracks.segmentation is not None:
        pred_seg = remap_seg_to_track_ids(
            pred_data_dir / "pred_tracks.zarr", pred_tracks.graph, pred_tracks.segmentation
        )
    else:
        pred_seg = None

    pred_graph = TrackingGraph(
        graph=pred_tracks.graph,
        frame_key="time",
        label_key="track_id",
        location_keys="pos",
        segmentation=pred_seg,
    )

    # match_threshold = config.get("match_threshold", 5.0)
    metrics = config.get("metrics", ["basic", "track_overlap"])
    matcher = config.get("matcher", "point")
    threshold = config.get("threshold", config.get("match_threshold", None))

    # Check that the specified metrics are valid
    for metric in metrics:
        if metric not in metrics_dict:
            raise ValueError(f"Invalid metric specified: {metric}\nValid metrics are: {list(metrics_dict.keys())}")

    matcher_fn = matchers_dict[matcher]
    kwargs = {"threshold": threshold} if threshold is not None else {}

    results, matched = run_metrics(
        gt_graph,
        pred_graph,
        matcher=matcher_fn(**kwargs),
        metrics=[metrics_dict[m]() for m in metrics],
    )

    return results
