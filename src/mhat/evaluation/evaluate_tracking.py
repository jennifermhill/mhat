from traccuracy import TrackingGraph, run_metrics
from traccuracy.matchers import PointMatcher
from traccuracy.metrics import BasicMetrics, TrackOverlapMetrics


def evaluate_tracking(
    gt_graph, pred_graph, gt_segmentation=None, pred_segmentation=None, match_threshold=5
):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        gt_graph (networkx.DiGraph): Ground truth graph.
        gt_segmentation (np.ndarray): Ground truth detections.
        pred_graph (networkx.DiGraph): Predicted graph.
        pred_segmentation (np.ndarray): Predicted dense segmentation.
        match_threshold (float): Distance threshold for matching tracks.

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

    results, matched = run_metrics(
        gt_graph,
        pred_graph,
        matcher=PointMatcher(threshold=match_threshold),
        metrics=[BasicMetrics(), TrackOverlapMetrics()],
    )

    return results
