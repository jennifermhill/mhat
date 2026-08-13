import re
from pathlib import Path

import geff
import numpy as np
import tifffile
import zarr

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


def _slice_seg_jaccard(ref, pred):
    """Sum of CTC-SEG Jaccard scores over the reference objects in one 2D slice.

    For each reference label R (background 0 excluded), the predicted label S
    with the largest overlap is a match iff |R ∩ S| > 0.5 |R| (the CTC detection
    criterion, which at most one predicted label can satisfy). A matched pair
    contributes J = |R ∩ S| / |R ∪ S|; unmatched reference objects contribute 0.

    Returns (jaccard_sum, n_reference_objects).
    """
    jaccard_sum = 0.0
    ref_labels = np.unique(ref)
    ref_labels = ref_labels[ref_labels != 0]
    for rl in ref_labels:
        ref_mask = ref == rl
        ref_area = int(ref_mask.sum())
        overlap_pred = pred[ref_mask]
        overlap_pred = overlap_pred[overlap_pred != 0]
        if overlap_pred.size:
            vals, counts = np.unique(overlap_pred, return_counts=True)
            best = vals[np.argmax(counts)]
            inter = int(counts.max())
            if inter > 0.5 * ref_area:  # CTC detection criterion
                pred_area = int((pred == best).sum())
                union = ref_area + pred_area - inter
                jaccard_sum += inter / union
    return jaccard_sum, len(ref_labels)


def compute_ctc_seg(seg_gt_dir, pred_seg_path):
    """Compute the CTC SEG measure from sparse ``man_seg`` reference slices.

    The CTC segmentation ground truth (the ``SEG`` folder) is sparse: only a
    subset of objects/slices are pixel-accurately annotated, unlike the coarse
    ``TRA`` markers. For 3D data each file is a single annotated 2D z-slice
    named ``man_seg_{t}_{z}.tif``; for 2D data it is ``man_seg{t}.tif``. Each
    reference slice is scored against the matching slice of the predicted
    segmentation, and SEG is the mean Jaccard over all annotated reference
    objects.

    Args:
        seg_gt_dir: Path to the CTC ``SEG`` folder holding ``man_seg*.tif``.
        pred_seg_path: Path to ``pred_seg.zarr`` with axes (t, z, y, x) for 3D
            or (t, y, x) for 2D.

    Returns:
        The SEG score in [0, 1], or None if no reference slices were found.
    """
    seg_gt_dir = Path(seg_gt_dir)
    pred_seg = zarr.open(str(pred_seg_path), mode="r")

    # man_seg_{t}_{z}.tif (3D sparse slice) or man_seg{t}.tif (2D frame)
    pat_3d = re.compile(r"man_seg_(\d+)_(\d+)\.tif$")
    pat_2d = re.compile(r"man_seg_?(\d+)\.tif$")

    total_jaccard = 0.0
    total_objs = 0
    n_slices = 0
    for tif in sorted(seg_gt_dir.glob("man_seg*.tif")):
        m3 = pat_3d.search(tif.name)
        if m3 is not None:
            t, z = int(m3.group(1)), int(m3.group(2))
            pred_slice = np.asarray(pred_seg[t, z])
        else:
            m2 = pat_2d.search(tif.name)
            if m2 is None:
                continue
            t = int(m2.group(1))
            pred_slice = np.asarray(pred_seg[t])
        ref = tifffile.imread(tif)
        j_sum, n = _slice_seg_jaccard(ref, pred_slice)
        total_jaccard += j_sum
        total_objs += n
        n_slices += 1

    if n_slices == 0 or total_objs == 0:
        return None
    return total_jaccard / total_objs


def load_tracking_graphs(config, gt_data_dir: Path, pred_data_dir: Path):
    """Load GT and predicted tracks (with segmentations) as traccuracy graphs.

    Shared by ``evaluate_tracking`` and the post-evaluation diagnostics so both
    see exactly the same graphs, positions and scale.

    Args:
        config (dict): Evaluation configuration dictionary (unused today, kept
            for symmetry with the other entry points).
        gt_data_dir (Path): Path to ground truth data directory.
        pred_data_dir (Path): Path to predicted data directory.

    Returns:
        tuple: (gt_graph, pred_graph, scale) where the graphs are
        ``traccuracy.TrackingGraph`` objects and scale is the per-axis scale
        read from the predicted geff metadata.
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

    gt_seg_path = None
    for candidate in ("correct_seg.zarr", "segmentation"):
        p = gt_data_dir / candidate
        if p.exists():
            gt_seg_path = p
            break
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

    return gt_graph, pred_graph, scale


def build_matcher(config):
    """Instantiate the matcher named by the config, with its threshold kwargs."""
    matcher = config.get("matcher", "point")
    threshold = config.get("threshold", config.get("match_threshold", None))

    matcher_fn = matchers_dict[matcher]
    if matcher == "iou":
        # IOUMatcher expects iou_threshold (default 0.6); also supports one_to_one.
        kwargs = {}
        iou_threshold = config.get("iou_threshold", threshold)
        if iou_threshold is not None:
            kwargs["iou_threshold"] = iou_threshold
        if "one_to_one" in config:
            kwargs["one_to_one"] = config["one_to_one"]
    else:
        kwargs = {"threshold": threshold} if threshold is not None else {}

    return matcher_fn(**kwargs)


def evaluate_tracking(
    config, gt_data_dir: Path, pred_data_dir: Path, return_matched: bool = False
):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        config (dict): Evaluation configuration dictionary.
        gt_data_dir (Path): Path to ground truth data directory.
        pred_data_dir (Path): Path to predicted data directory.
        return_matched (bool): If True, also return the traccuracy ``Matched``
            object so callers can run further diagnostics without re-matching.

    Returns:
        results (dict): Dictionary of metric results, or (results, matched) if
        ``return_matched`` is True.
    """
    gt_graph, pred_graph, _ = load_tracking_graphs(config, gt_data_dir, pred_data_dir)

    # match_threshold = config.get("match_threshold", 5.0)
    metrics = config.get("metrics", ["basic", "track_overlap"])

    # Check that the specified metrics are valid
    for metric in metrics:
        if metric not in metrics_dict:
            raise ValueError(f"Invalid metric specified: {metric}\nValid metrics are: {list(metrics_dict.keys())}")

    results, matched = run_metrics(
        gt_graph,
        pred_graph,
        matcher=build_matcher(config),
        metrics=[metrics_dict[m]() for m in metrics],
    )

    # The CTC SEG measure uses the sparse `SEG` ground truth folder, not the
    # coarse `TRA` markers loaded here, so it is computed separately by the
    # caller (see compute_ctc_seg / run_evaluation).
    if return_matched:
        return results, matched
    return results


def match_tracking(config, gt_data_dir: Path, pred_data_dir: Path):
    """Load the graphs and run only the matcher, returning the Matched object.

    Used by the standalone diagnostic scripts, which need the matching but not
    the metrics.
    """
    gt_graph, pred_graph, _ = load_tracking_graphs(config, gt_data_dir, pred_data_dir)
    return build_matcher(config).compute_mapping(gt_graph, pred_graph)
