import re
from pathlib import Path

import dask.array as da
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

def build_node_id_lut(graph):
    """Lookup table from segmentation label (= graph node id) to ``track_id``.

    Segmentations in this pipeline are labelled by graph node id -- both
    ``pred_seg.zarr`` from run_tracking and ``correct_seg.zarr`` from
    from_ctc_to_geff -- which is the invariant funtracks relies on as well
    (``Tracks.get_pixels`` does ``segmentation[time] == node``).
    ``import_from_geff`` renumbers ``track_id``, so labels still need mapping
    before traccuracy, which matches on ``track_id``, sees them.

    Node ids are unique per node, so one table covers the whole movie rather
    than one per frame. It is indexed by label, so it is sized by the largest
    node id: under a megabyte even for a 230k-node graph. Labels with no node
    map to 0 and stay background.
    """
    max_node = max(graph.nodes(), default=0)
    max_track_id = max(
        (int(data["track_id"]) for _, data in graph.nodes(data=True)), default=0
    )
    dtype = np.promote_types(np.min_scalar_type(max_track_id), np.uint16)
    lut = np.zeros(int(max_node) + 1, dtype=dtype)
    for node, data in graph.nodes(data=True):
        lut[int(node)] = int(data["track_id"])
    return lut


def _apply_lut(block, lut, seg_path):
    """Map one block of labels through ``lut``, refusing out-of-range labels."""
    if block.size:
        highest = int(block.max())
        if highest >= lut.size:
            raise ValueError(
                f"Segmentation at {seg_path} holds label {highest}, which is not "
                f"a graph node id (largest is {lut.size - 1}). The segmentation "
                f"and its tracks store are out of step -- regenerate them."
            )
    return lut[block]


def remap_seg_to_track_ids(graph, seg_path):
    """Relabel a node-id-labelled segmentation to the graph's ``track_id``s.

    Reads the zarr a frame at a time and maps each through
    :func:`build_node_id_lut`, so peak memory is the returned array rather than
    the input plus a copy of it, and the cost is one pass per frame instead of
    one full-volume comparison per node.

    We load the segmentation here rather than letting ``import_from_geff`` do
    it, because funtracks validates the store by scaling every coordinate by
    its axis scale -- including ``time``, which is a frame index, not a physical
    coordinate. On a dataset with a real frame interval (DRO: 30 s/frame) it
    reads frame ``int(t / 30)`` and rejects a perfectly good segmentation. Note
    that ``t`` below is used directly as an index, never scaled.

    Returns an in-memory array; use :func:`remap_seg_to_track_ids_lazy` when the
    whole volume does not need to be resident (a full-resolution label volume is
    ~10 GB on a dataset the size of Fluo-N3DL-DRO).
    """
    lut = build_node_id_lut(graph)
    segmentation = zarr.open(str(seg_path), mode="r")
    remapped = np.zeros(segmentation.shape, dtype=lut.dtype)
    checked = False
    for t in range(segmentation.shape[0]):
        frame = np.asarray(segmentation[t])
        if not checked:
            nodes_here = [n for n, d in graph.nodes(data=True) if int(d["time"]) == t]
            if nodes_here:
                _check_labelled_by_node_id(frame, nodes_here, seg_path, t)
                checked = True
        remapped[t] = _apply_lut(frame, lut, seg_path)
    return remapped


def remap_seg_to_track_ids_lazy(graph, seg_path):
    """``remap_seg_to_track_ids`` as a lazy dask array, for viewing.

    The mapping is elementwise, so it applies per chunk and nothing beyond the
    chunks actually touched is ever read: pulling one z-slice of a 10 GB volume
    costs a few MB. Use this wherever the segmentation is displayed rather than
    measured -- napari pulls only the slice on screen.

    Unlike the eager version this cannot check the node-id convention up front,
    since that would mean reading a frame; a mislabelled store shows up as
    background instead of raising.
    """
    lut = build_node_id_lut(graph)
    return da.from_zarr(seg_path).map_blocks(
        _apply_lut, lut=lut, seg_path=seg_path, dtype=lut.dtype
    )


def _check_labelled_by_node_id(frame, nodes, seg_path, t, sample=5):
    """Fail loudly if a segmentation frame is not labelled by graph node id.

    Every node in a frame came from an object in that frame, so under the
    convention its node id must appear as a label. A store written before the
    convention is labelled by track_id instead, where the node ids are absent --
    which would otherwise map every object to background in silence.
    """
    missing = [n for n in nodes[:sample] if not np.any(frame == n)]
    if missing:
        raise ValueError(
            f"Segmentation at {seg_path} is not labelled by graph node id: "
            f"node ids {missing} have no pixels in frame {t}. Stores written "
            f"before this convention labelled the segmentation by track_id -- "
            f"delete the tracks store and its segmentation and let them be "
            f"regenerated."
        )


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
    # Segmentations are loaded by remap_seg_to_track_ids, not funtracks -- see
    # the note there about funtracks scaling the time index.
    gt_tracks = import_from_geff(
        gt_data_dir / "correct_tracks.zarr",
        node_name_map=node_name_map,
        segmentation_path=None,
        scale=scale,
    )

    if gt_seg_path is not None:
        gt_seg = remap_seg_to_track_ids(gt_tracks.graph, gt_seg_path)
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
        node_name_map=node_name_map,
        segmentation_path=None,
        scale=scale,
    )

    if pred_seg_path is not None:
        pred_seg = remap_seg_to_track_ids(pred_tracks.graph, pred_seg_path)
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
