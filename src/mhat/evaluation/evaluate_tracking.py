from traccuracy import TrackingGraph, run_metrics
from traccuracy.matchers import PointMatcher
import traccuracy.metrics as metrics

metrics_dict = {
    "basic": metrics.BasicMetrics,
    "cca": metrics.CellCycleAccuracy,
    "complete_tracks": metrics.CompleteTracks,
    "aogm": metrics.AOGMMetrics,
    "ctc": metrics.CTCMetrics,
    "division": metrics.DivisionMetrics,
    "track_overlap": metrics.TrackOverlapMetrics,
}

def evaluate_tracking(
    config, gt_graph, pred_graph, gt_segmentation=None, pred_segmentation=None
):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        config (dict): Evaluation configuration dictionary.
        gt_graph (networkx.DiGraph): Ground truth graph.
        gt_segmentation (np.ndarray): Ground truth detections.
        pred_graph (networkx.DiGraph): Predicted graph.
        pred_segmentation (np.ndarray): Predicted dense segmentation.

    Returns:
        results (dict): Dictionary of metric results.
    """

    gt_graph = TrackingGraph(
        graph=gt_graph,
        frame_key="time",
        label_key="track_id",
        location_keys=("x", "y", "z"),
        segmentation=gt_segmentation,
    )

    pred_graph = TrackingGraph(
        graph=pred_graph,
        frame_key="time",
        label_key="track_id",
        location_keys=("x", "y", "z"),
        segmentation=pred_segmentation,
    )

    match_threshold = config.get("match_threshold", 5.0)
    metrics = config.get("metrics", ["basic", "track_overlap"])

    # Check that the specified metrics are valid
    for metric in metrics:
        if metric not in metrics_dict:
            raise ValueError(f"Invalid metric specified: {metric}\nValid metrics are: {list(metrics_dict.keys())}")

    results, matched = run_metrics(
        gt_graph,
        pred_graph,
        matcher=PointMatcher(threshold=match_threshold),
        metrics=[metrics_dict[m]() for m in metrics],
    )

    return results
