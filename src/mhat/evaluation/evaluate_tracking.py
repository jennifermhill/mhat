import re
from pathlib import Path

import geff
import numpy as np
import tifffile
import zarr

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

def remap_seg_to_track_ids(graph, segmentation):
    """Relabel a node-id-labelled segmentation to the graph's track_ids.

    Every segmentation this pipeline reads is labelled by graph node id --
    ``pred_seg.zarr`` from run_tracking, ``correct_seg.zarr`` from
    from_ctc_to_geff -- which is also the invariant funtracks relies on
    (``Tracks.get_pixels`` does ``segmentation[time] == node``).
    ``import_from_geff`` renumbers ``track_id``, so the labels still have to be
    mapped node id -> track_id before traccuracy, which matches on ``track_id``,
    sees them.

    The mapping is applied per frame through a lookup table: one pass over each
    frame, rather than one full-volume comparison per node.
    """
    # Group (node id -> track_id) by frame. Doing this per frame rather than
    # globally keeps the mapping correct even if a label were ever reused
    # across frames.
    per_frame: dict[int, list[tuple[int, int]]] = {}
    max_track_id = 0
    for node, data in graph.nodes(data=True):
        track_id = int(data["track_id"])
        per_frame.setdefault(int(data["time"]), []).append((int(node), track_id))
        max_track_id = max(max_track_id, track_id)

    dtype = np.promote_types(np.min_scalar_type(max_track_id), np.uint16)
    remapped = np.zeros(segmentation.shape, dtype=dtype)
    for t, pairs in per_frame.items():
        frame = np.asarray(segmentation[t])
        # Labels with no node in this frame index a 0 entry and stay background.
        size = max(int(frame.max()), max(node for node, _ in pairs)) + 1
        lut = np.zeros(size, dtype=dtype)
        for node, track_id in pairs:
            lut[node] = track_id
        remapped[t] = lut[frame]
    return remapped


def _import_tracks(geff_path, seg_path, node_name_map, scale, kind):
    """import_from_geff with a clearer error when the seg convention is wrong.

    funtracks checks that the label under one node's centroid equals that
    node's id, and raises a bare "Error testing seg id:" when it doesn't. That
    happens for stores written before segmentations were labelled by node id --
    regenerate them rather than working around it here.
    """
    try:
        return import_from_geff(
            geff_path,
            node_name_map=node_name_map,
            segmentation_path=seg_path,
            scale=scale,
        )
    except ValueError as err:
        if "seg id" not in str(err):
            raise
        raise ValueError(
            f"{kind} segmentation at {seg_path} is not labelled by graph node id "
            f"(funtracks said: {err}). Stores written before this convention "
            f"labelled the segmentation by track_id; delete {geff_path} and its "
            f"segmentation and let them be regenerated."
        ) from err


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
    node_name_map = {
        "time": "time",
        "x": "x",
        "y": "y",
        "z": "z",
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
    gt_tracks = _import_tracks(
        gt_data_dir / "correct_tracks.zarr", gt_seg_path, node_name_map, scale, "GT"
    )

    if gt_tracks.segmentation is not None:
        gt_seg = remap_seg_to_track_ids(gt_tracks.graph, gt_tracks.segmentation)
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
    pred_tracks = _import_tracks(
        pred_data_dir / "pred_tracks.zarr", pred_seg_path, node_name_map, scale,
        "Prediction",
    )

    if pred_tracks.segmentation is not None:
        pred_seg = remap_seg_to_track_ids(pred_tracks.graph, pred_tracks.segmentation)
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

    # The CTC SEG measure uses the sparse `SEG` ground truth folder, not the
    # coarse `TRA` markers loaded here, so it is computed separately by the
    # caller (see compute_ctc_seg / run_evaluation).
    return results
