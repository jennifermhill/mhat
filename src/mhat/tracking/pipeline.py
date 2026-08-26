"""Shared multi-hypothesis candidate graph construction.

Used by both run_tracking.py and the SSVM fit/inspect scripts so they all
build the candidate graph the same way.
"""

from __future__ import annotations

from pathlib import Path

import motile
import networkx as nx
import numpy as np
import zarr

from mhat.tracking import create_multihypo_graph, utils
from mhat.utils import get_axes_metadata


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

    flow_result = config.get("flow_result", None)
    if flow_result is not None:
        flow_base = input_base_dir / "opticalflow" / experiment / dataset
        flow_dir_2d = flow_base / "opticalflow_2d" / flow_result
        flow_dir_3d = flow_base / "opticalflow_3d" / flow_result
        if not flow_dir_2d.is_dir():
            print(f"2D optical flow directory {flow_dir_2d} does not exist, using 3D only.")
            flow_dir_2d = None
        assert flow_dir_3d.is_dir(), f"Optical flow directory {flow_dir_3d} is missing"
        flow_dirs = {"2d": flow_dir_2d, "3d": flow_dir_3d}
    else:
        flow_dirs = {"2d": None, "3d": None}

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
        flow_2d_zarr = zarr.open(flow_2d_zarr_path)["flow_raw"]
    else:
        flow_2d_zarr = None
    if flow_3d_zarr_path is not None:
        flow_3d_root = zarr.open(flow_3d_zarr_path)
        flow_3d_zarr = flow_3d_root["flow_raw"]
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

    if flow_3d is not None:
        utils.add_flow_dist_attr(track_graph)
    elif "drift_distance" in config:
        utils.add_drift_dist_attr(track_graph, drift=config["drift_distance"])
    else:
        utils.add_drift_dist_attr(track_graph, drift=0)
    utils.add_area_diff_attr(track_graph)
    utils.add_intensity_diff_attr(track_graph)
    utils.add_division_attr(track_graph)

    return track_graph, fragments, merge_history, all_exclusion_sets, scale, axes
