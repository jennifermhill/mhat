"""Shared multi-hypothesis candidate graph construction.

Used by both run_tracking.py and the SSVM fit/inspect scripts so they all
build the candidate graph the same way. Everything here is rank-agnostic: the
data's dimensionality is read off the fragments array (``ndim = fragments.ndim
- 1``) and never assumed, so the same code serves 2D+time and 3D+time movies.
"""

from __future__ import annotations

from pathlib import Path

import geff
import motile
import networkx as nx
import numpy as np
import zarr

from funtracks.import_export import import_from_geff
from mhat.evaluation.evaluate_tracking import (
    read_name_map_and_scale,
    remap_seg_to_track_ids,
)
from mhat.opticalflow.utils import open_flow_raw
from mhat.tracking import create_multihypo_graph, utils
from mhat.utils import get_axes_metadata, seg_chunks
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids


def data_ndim(seg_dir: Path) -> int:
    """Spatial rank of a segmentation run, from its fragments array (metadata only)."""
    return zarr.open(seg_dir / "data.zarr", mode="r")["fragments"].ndim - 1


def resolve_flow_dirs(config, seg_dir: Path) -> dict:
    """Which optical-flow directories a run needs, by the data's rank.

    Mirrors run_tracking.py. 3D data must have a 3D flow (nothing else can
    estimate axial motion); a 2D flow beside it is optional and, when present,
    supplies the better-resolved in-plane components. 2D data has no 3D flow to
    look for, so the 2D one is required.
    """
    flow_result = config.get("flow_result", None)
    if flow_result is None:
        return {"2d": None, "3d": None}

    input_base_dir = Path(config["input_base_dir"])
    flow_base = input_base_dir / "opticalflow" / config["experiment"] / config["dataset"]
    flow_dir_2d = flow_base / "opticalflow_2d" / flow_result
    flow_dir_3d = flow_base / "opticalflow_3d" / flow_result
    if data_ndim(seg_dir) == 3:
        assert flow_dir_3d.is_dir(), f"3D optical flow data directory {flow_dir_3d} is missing"
        if not flow_dir_2d.is_dir():
            print(f"2D optical flow directory {flow_dir_2d} does not exist, using 3D flow only.")
            flow_dir_2d = None
    else:
        assert flow_dir_2d.is_dir(), (
            f"2D optical flow data directory {flow_dir_2d} is missing -- "
            "2D data has no 3D flow to fall back on"
        )
        flow_dir_3d = None
    return {"2d": flow_dir_2d, "3d": flow_dir_3d}


def resolve_input_dirs(config) -> tuple[Path, Path, dict, Path]:
    """Turn the TOML path keys into the directories the pipeline scripts need.

    Returns:
        (raw_dir, seg_dir, flow_dirs, gt_data_dir)

    ``gt_data_dir`` defaults to the usual ``<input>/tracking/<exp>/<dataset>``
    convention but can be overridden with a ``gt_data_dir`` config key — used to
    point a fit or an evaluation at a reduced GT subset (see gt_subsets.py).
    """
    raw_base_dir = Path(config["raw_base_dir"])
    input_base_dir = Path(config["input_base_dir"])
    experiment = config["experiment"]
    dataset = config["dataset"]
    assert raw_base_dir.is_dir(), f"Raw base directory {raw_base_dir} is missing"
    assert input_base_dir.is_dir(), f"Input base directory {input_base_dir} is missing"

    raw_dir = raw_base_dir / experiment / f"{dataset}.zarr"
    assert raw_dir.is_dir(), f"Raw data directory {raw_dir} is missing"

    seg_dir = input_base_dir / "segmentation" / experiment / dataset / config["seg_result"]
    assert seg_dir.is_dir(), f"Segmentation data directory {seg_dir} is missing"

    flow_dirs = resolve_flow_dirs(config, seg_dir)

    if config.get("gt_data_dir"):
        gt_data_dir = Path(config["gt_data_dir"])
    else:
        gt_data_dir = input_base_dir / "tracking" / experiment / dataset

    return raw_dir, seg_dir, flow_dirs, gt_data_dir


def build_track_graph(config, raw_dir: Path, seg_dir: Path, flow_dirs: dict):
    """Load arrays, run multi-hypo construction, attach attributes.

    Raw, fragments and flow are all read one timepoint at a time, so the
    movie is never held in memory while the graph is built.

    Returns:
        (track_graph, fragments, merge_history, exclusion_sets, scale, axes)
    """
    raw_zarr_path = raw_dir
    seg_zarr_path = seg_dir / "data.zarr"
    flow_2d_zarr_path = flow_dirs["2d"] / "flow.zarr" if flow_dirs["2d"] is not None else None
    flow_3d_zarr_path = flow_dirs["3d"] / "flow.zarr" if flow_dirs["3d"] is not None else None
    merge_history_csv_path = seg_dir / "merge_history.csv"

    raw_zarr = zarr.open(raw_zarr_path)
    seg_zarr_root = zarr.open(seg_zarr_path)
    fragments = seg_zarr_root["fragments"]

    if flow_2d_zarr_path is not None:
        flow_2d_zarr = open_flow_raw(zarr.open(flow_2d_zarr_path))
    else:
        flow_2d_zarr = None
    if flow_3d_zarr_path is not None:
        flow_3d_root = zarr.open(flow_3d_zarr_path)
        flow_3d_zarr = open_flow_raw(flow_3d_root)  # (vz, vy, vx)
        if "confidence" in flow_3d_root:
            confidence_3d_zarr = flow_3d_root["confidence"]
        else:
            confidence_3d_zarr = None
    else:
        flow_3d_zarr = None
        confidence_3d_zarr = None

    def load_timepoint(zarr_arr, timepoint, channel=None):
        """Load a single timepoint from a zarr, or zeros if it is past the end.

        Flow zarrs are one frame shorter than the movie (the last frame has no
        successor), so those timepoints come back as zeros. Passing a channel
        selects it while reading, so the other channels are never fetched.
        """
        if zarr_arr is None:
            return None
        if timepoint >= zarr_arr.shape[0]:
            frame_shape = zarr_arr.shape[1:] if channel is None else zarr_arr.shape[2:]
            return np.zeros(frame_shape, dtype=zarr_arr.dtype)
        return zarr_arr[timepoint] if channel is None else zarr_arr[timepoint, channel]

    # Keep the zarr handles around so downstream is-not-None checks still work.
    flow_2d = flow_2d_zarr
    flow_3d = flow_3d_zarr
    confidence_3d = confidence_3d_zarr

    axes = get_axes_metadata(seg_zarr_root["fragments"])
    scale = [axis["scale"] for axis in axes]
    img_shape = fragments.shape
    # One source of truth for the rank: the fragments array itself, which the
    # axes metadata always agrees with. Everything downstream (positions, geff
    # axis names, the flow gating) follows from it rather than assuming 3D.
    ndim = fragments.ndim - 1
    assert len(axes) == ndim + 1, (
        f"{seg_zarr_path}/fragments is {fragments.ndim}D but its axes metadata "
        f"names {len(axes)} axes: {[a['name'] for a in axes]}"
    )
    assert ndim == 2 or flow_2d is None or flow_3d is not None, (
        "3D data with a 2D flow but no 3D flow: nothing can estimate axial motion"
    )
    # Node times are frame indices, not world units, so time stays unscaled here.
    img_shape_scaled = [img_shape[0]] + [
        int(img_shape[i] * scale[i]) for i in range(1, len(img_shape))
    ]

    merge_history = create_multihypo_graph.load_merge_history(merge_history_csv_path)
    # A cellpose / skip-merges segmentation has no merge hierarchy. Mirror
    # run_tracking.py: build nodes straight from the segmentation and skip
    # normalize/renumber (normalize_costs would call np.min on an empty array).
    no_merges = len(merge_history) == 0
    if no_merges:
        print("No merge history found. Building candidate graph in no-merge mode.")
    else:
        # renumber_merge_history needs the largest fragment id up front; take it
        # one frame at a time rather than materializing the whole array. Only
        # this path needs it, so no-merge mode skips the read entirely.
        max_node_id = max(int(fragments[t].max()) for t in range(img_shape[0]))
        merge_history = create_multihypo_graph.normalize_costs(merge_history)
        merge_history = create_multihypo_graph.renumber_merge_history(merge_history, max_node_id)

    z_flow_conf_threshold = config.get("z_flow_conf_threshold", None)
    z_flow_min_pass_pixels = config.get("z_flow_min_pass_pixels", 10)
    if ndim == 2 and z_flow_conf_threshold is not None:
        print(
            "Warning: z_flow_conf_threshold / z_flow_min_pass_pixels filter the "
            "axial flow component, which 2D data does not have. Ignoring them."
        )
        z_flow_conf_threshold = None

    all_cand_graph = None
    all_exclusion_sets: list = []
    for t in range(img_shape[0]):
        # Lazy-load raw, fragment and flow data for this timepoint
        raw_tp = load_timepoint(raw_zarr, t, channel=0)
        frag_tp = load_timepoint(fragments, t)
        flow_2d_tp = load_timepoint(flow_2d_zarr, t)
        flow_3d_tp = load_timepoint(flow_3d_zarr, t)
        conf_3d_tp = load_timepoint(confidence_3d_zarr, t)
        if no_merges:
            cand_graph = utils.nodes_from_segmentation(
                frag_tp,
                raw_img=raw_tp,
                flow_3d=flow_3d_tp,
                flow_2d=flow_2d_tp,
                confidence_3d=conf_3d_tp,
                z_flow_conf_threshold=z_flow_conf_threshold,
                z_flow_min_pass_pixels=z_flow_min_pass_pixels,
                size_threshold=config["size_threshold"],
                tp=t,
                scale=scale,
            )
            for node in cand_graph.nodes():
                cand_graph.nodes[node]["num_leaves"] = 1
            exclusion_sets = []
        else:
            cand_graph, exclusion_sets = create_multihypo_graph.nodes_from_fragments(
                frag_tp,
                merge_history[merge_history[:, 4] == t],
                min_cost=config.get("min_merge_cost", config.get("min_merge_score", 0.0)),
                max_cost=config.get("max_merge_cost", config.get("max_merge_score", 1.0)),
                raw_img=raw_tp,
                flow_2d=flow_2d_tp,
                flow_3d=flow_3d_tp,
                confidence_3d=conf_3d_tp,
                z_flow_conf_threshold=z_flow_conf_threshold,
                z_flow_min_pass_pixels=z_flow_min_pass_pixels,
                size_threshold=config["size_threshold"],
                # A frame with no merges cannot read its timepoint off the (empty)
                # merge-history slice, so say which frame this is.
                timepoint=t,
                scale=scale,
            )
        if all_cand_graph is None:
            all_cand_graph = cand_graph
            all_exclusion_sets = exclusion_sets
        else:
            all_cand_graph = nx.compose(all_cand_graph, cand_graph)
            all_exclusion_sets.extend(exclusion_sets)

    utils.add_cand_edges(
        all_cand_graph, config["max_edge_distance"], max_children=config["max_children"]
    )
    all_cand_graph = utils.add_hyperedges(
        all_cand_graph, divisions=config["divisions"]
    )
    utils.add_appear_ignore_attr(all_cand_graph)
    utils.add_disappear(all_cand_graph, img_shape_scaled)
    track_graph = motile.TrackGraph(all_cand_graph, frame_attribute="time")

    # Either flow is enough to predict motion. On 3D data the assert above
    # guarantees a 3D flow whenever there is any flow; on 2D data the 2D flow
    # is the only flow there is.
    if flow_3d is not None or flow_2d is not None:
        utils.add_flow_dist_attr(track_graph)
    elif "drift_distance" in config:
        utils.add_drift_dist_attr(track_graph, drift=config["drift_distance"])
    else:
        utils.add_drift_dist_attr(track_graph, drift=0)
    utils.add_area_diff_attr(track_graph)
    utils.add_intensity_diff_attr(track_graph)
    utils.add_division_attr(track_graph)

    return track_graph, fragments, merge_history, all_exclusion_sets, scale, axes


def load_gt(gt_data_dir: Path, scale):
    """Load a geff ground truth and its label volume, remapped to ``track_id``.

    ``scale`` is the *prediction's* scale (the one ``build_track_graph`` read off
    the segmentation), applied to the GT exactly as ``evaluate_tracking`` does,
    so the fit and the evaluation see the same GT coordinates. The name map is
    built from the GT store's own axes, so a 2D geff loads as 2D rather than
    silently skipping a ``z`` it does not have.

    Returns:
        (gt_graph, gt_seg): the funtracks graph (``track_id`` renumbered by
        funtracks) and the ``correct_seg.zarr`` volume relabelled to match it.
    """
    gt_data_dir = Path(gt_data_dir)
    gt_tracks_path = gt_data_dir / "correct_tracks.zarr"
    gt_seg_path = gt_data_dir / "correct_seg.zarr"
    if not gt_tracks_path.is_dir():
        raise FileNotFoundError(
            f"GT tracks not found at {gt_tracks_path}. Run evaluate_tracks.py "
            "once on a prior tracking result to trigger CTC->geff conversion."
        )
    if not gt_seg_path.is_dir():
        raise RuntimeError(
            f"GT segmentation not found at {gt_seg_path}; cannot compute IoU matches."
        )
    name_map, own_scale = read_name_map_and_scale(gt_tracks_path)
    assert len(own_scale) == len(scale), (
        f"{gt_tracks_path} has {len(own_scale)} axes but the prediction has "
        f"{len(scale)}: the ground truth and the data are not the same rank"
    )
    # The label volume is loaded by remap_seg_to_track_ids rather than funtracks,
    # which would scale the time index by its axis scale (see evaluate_tracking).
    gt_tracks = import_from_geff(
        gt_tracks_path, node_name_map=name_map, segmentation_path=None, scale=list(scale)
    )
    gt_seg = remap_seg_to_track_ids(gt_tracks.graph, gt_seg_path)
    return gt_tracks.graph, gt_seg


def geff_axis_kwargs(axes) -> dict:
    """``axis_names`` / ``axis_types`` for ``geff.write`` from zarr axes metadata.

    Names come from the data rather than a 3D literal, so a 2D run writes
    ``["time", "y", "x"]`` and does not declare a z axis that no node property
    backs. On every 3D dataset in this repo the metadata already reads exactly
    time/z/y/x, so the output there is unchanged.
    """
    return {
        "axis_names": [axis["name"] for axis in axes],
        "axis_types": [
            axis.get("type", "time" if i == 0 else "space") for i, axis in enumerate(axes)
        ],
    }


def write_tracking_outputs(solution_graph, fragments, merge_history, scale, axes, output_dir):
    """Write ``pred_seg.zarr`` and ``pred_tracks.zarr`` the way run_tracking.py does.

    The segmentation is labelled by solution node id (the convention
    ``evaluate_tracking.remap_seg_to_track_ids`` relies on) and chunked one
    tile of one slice, whatever the rank. ``fragments`` may be a lazy zarr: it
    is read one frame at a time. Assigns tracklet ids on ``solution_graph``.
    """
    output_dir = Path(output_dir)
    frag_ids = set()
    for t in range(fragments.shape[0]):
        frag_ids.update(int(v) for v in np.unique(fragments[t]))
    frag_ids.discard(0)
    lookup = utils.get_solution_lookup(
        merge_history, solution_graph, frag_ids,
        max(frag_ids) if frag_ids else 0, fragments.dtype,
    )
    assign_tracklet_ids(solution_graph)

    output_seg_path = output_dir / "pred_seg.zarr"
    output_zarr_root = zarr.open(
        output_seg_path, mode="w", shape=fragments.shape,
        chunks=seg_chunks(fragments.shape[1:]), dtype=np.uint32,
    )
    output_zarr_root.attrs["axes"] = axes
    for t in range(fragments.shape[0]):
        output_zarr_root[t] = lookup[fragments[t]]

    metadata = geff.GeffMetadata(
        directed=True,
        related_objects=[{"type": "labels", "path": "../pred_seg.zarr", "label_prop": "label"}],
        node_props_metadata={},
        edge_props_metadata={},
    )
    geff.write(
        solution_graph,
        output_dir / "pred_tracks.zarr",
        axis_scales=list(scale),
        metadata=metadata,
        overwrite=True,
        **geff_axis_kwargs(axes),
    )
