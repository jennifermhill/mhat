from pathlib import Path

import geff
import numpy as np

from geff_spec import Axis
from traccuracy import TrackingGraph, run_metrics
import traccuracy.matchers as matchers
import traccuracy.metrics as metrics

from traccuracy.matchers._compute_overlap import get_labels_with_overlap, graph_bbox_and_labels

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


def compute_seg_metric(gt_graph, pred_graph):
    """Compute CTC SEG: average Jaccard index of matched reference objects.

    For each GT label, finds the predicted label with IoGT > 0.5 (the CTC
    detection criterion). If matched, computes the Jaccard similarity index
    (IoU) for that pair. Unmatched GT labels contribute J=0. SEG is the
    mean of all J values across all GT labels in all frames.
    """
    gt_label_key = gt_graph.label_key
    pred_label_key = pred_graph.label_key
    mask_gt = gt_graph.segmentation
    mask_pred = pred_graph.segmentation

    total_jaccard = 0.0
    total_gt_labels = 0

    for t in range(gt_graph.start_frame, gt_graph.end_frame):
        i = t - gt_graph.start_frame
        gt_frame = mask_gt[i]
        pred_frame = mask_pred[i]
        gt_frame_nodes = gt_graph.nodes_by_frame[t]
        pred_frame_nodes = pred_graph.nodes_by_frame[t]

        gt_boxes, gt_labels = graph_bbox_and_labels(
            gt_graph.graph, gt_frame_nodes, gt_label_key)
        pred_boxes, pred_labels = graph_bbox_and_labels(
            pred_graph.graph, pred_frame_nodes, pred_label_key)

        # Get IoGT overlaps to determine matches (CTC detection criterion)
        iogt_overlaps = get_labels_with_overlap(
            gt_frame, pred_frame,
            gt_boxes=gt_boxes, res_boxes=pred_boxes,
            gt_labels=gt_labels, res_labels=pred_labels,
            overlap="iogt",
        )

        # For each GT label, find the pred label with IoGT > 0.5
        # (at most one can satisfy this per the CTC spec)
        matched = {}  # gt_label -> pred_label
        for gt_lab, pred_lab, iogt in iogt_overlaps:
            if iogt > 0.5:
                matched[gt_lab] = pred_lab

        # Get IoU overlaps for the matched pairs
        iou_overlaps = get_labels_with_overlap(
            gt_frame, pred_frame,
            gt_boxes=gt_boxes, res_boxes=pred_boxes,
            gt_labels=gt_labels, res_labels=pred_labels,
            overlap="iou",
        )

        # Index IoU by (gt_label, pred_label)
        iou_map = {}
        for gt_lab, pred_lab, iou in iou_overlaps:
            iou_map[(gt_lab, pred_lab)] = iou

        # Compute Jaccard for each GT label
        gt_labels_set = {gt_graph.graph.nodes[n][gt_label_key]
                         for n in gt_frame_nodes}
        for lab in gt_labels_set:
            if lab in matched:
                total_jaccard += iou_map.get((lab, matched[lab]), 0.0)
            # else: unmatched, contributes J=0
            total_gt_labels += 1

    return total_jaccard / total_gt_labels if total_gt_labels > 0 else 0.0


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

    if "ctc" in metrics and gt_seg is not None and pred_seg is not None:
        seg_score = compute_seg_metric(gt_graph, pred_graph)
        for r in results:
            if r["metric"]["name"] == "CTCMetrics":
                r["results"]["SEG"] = seg_score
                break

    return results
