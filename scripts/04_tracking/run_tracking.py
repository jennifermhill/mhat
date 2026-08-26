import csv
import argparse
import datetime
from pathlib import Path

import motile
import numpy as np
import toml
import tifffile as tiff
import zarr
import geff
import networkx as nx

from mhat.evaluation.eval_io import check_video_dir
from mhat.tracking import create_multihypo_graph, solve_with_motile, utils
from mhat.tracking.utils import report_graph_statistics
from mhat.tracking.tracks_io import save_tracks_to_csv
from mhat.utils import get_axes_metadata
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids


def get_solution_lookup(merge_history, solution_graph, frag_ids, max_frag_id, dtype):
    """Build a leaf-fragment-id -> solution-node-id lookup table.

    Applying it to one frame with lookup[frame] relabels that frame in a single
    vectorized pass, so the movie is never relabelled in memory all at once.
    """
    merge_dict = {}
    for merge in merge_history:
        a, b, c, cost, tp = merge
        a = int(a)
        b = int(b)
        c = int(c)
        children = [a, b]
        if a in merge_dict:
            children.extend(merge_dict[a])
        if b in merge_dict:
            children.extend(merge_dict[b])
        merge_dict[c] = children

    # Merged/intermediate ids are all > max(fragments) (see
    # renumber_merge_history), so they never index into the volume and only
    # leaf slots are needed.
    lookup = np.zeros(max_frag_id + 1, dtype=dtype)

    for node in solution_graph.nodes():
        if node in merge_dict:
            children = merge_dict[node]
        else:
            assert node in frag_ids, f"Node {node} not in merge dict or frag ids"
            children = [node]

        for child in children:
            if child > max_frag_id:
                continue  # intermediate/merged id, never present in the volume
            # Each leaf fragment may be claimed by at most one selected node (the ILP
            # ExclusiveNodes invariant). This preserves the original per-fragment
            # assertion as an O(children) check instead of a full-volume scan.
            assert lookup[child] == 0, (
                f"Child {child} fragment already assigned to node {lookup[child]}, "
                f"cannot reassign to {node}"
            )
            lookup[child] = node

    return lookup


def run_tracking(config, raw_dir: Path, seg_dir: Path, flow_dirs: dict, output_dir: Path, stats_only: bool = False):

    raw_zarr_path = raw_dir
    seg_zarr_path = seg_dir / "data.zarr"
    flow_2d_zarr_path = flow_dirs["2d"] / "flow.zarr" if flow_dirs["2d"] is not None else None
    flow_3d_zarr_path = flow_dirs["3d"] / "flow.zarr" if flow_dirs["3d"] is not None else None
    output_seg_path = output_dir / "pred_seg.zarr"
    merge_history_csv_path = seg_dir / "merge_history.csv"
    normalized_merge_history_csv_path = output_dir / "normalized_merge_history.csv"
    config_filepath = output_dir / "config.toml"
    output_filepath_geff = output_dir / "pred_tracks.zarr"

    with open(config_filepath, "w") as config_file:
        toml.dump(config, config_file)

    seg_group = "fragments"
    flow_group = "flow_raw"

    max_edge_distance = config["max_edge_distance"]
    max_timepoints = config.get("max_timepoints", None)

    # Raw, fragments and flow are all left as zarr arrays and read one timepoint
    # at a time; nothing here pulls a whole movie into memory.
    raw_zarr = zarr.open(raw_zarr_path)
    seg_zarr_root = zarr.open(seg_zarr_path)
    fragments = seg_zarr_root[seg_group]
    n_total_frames = fragments.shape[0]
    n_frames = (
        min(max_timepoints, n_total_frames) if max_timepoints is not None
        else n_total_frames
    )
    if n_frames < n_total_frames:
        print(f"Truncating to first {n_frames} timepoints (of {n_total_frames})")
    if flow_2d_zarr_path is not None:
        flow_2d_zarr_root = zarr.open(flow_2d_zarr_path)
        flow_2d_zarr = flow_2d_zarr_root[flow_group]
        flow_2d_shape = flow_2d_zarr.shape
    else:
        flow_2d_zarr = None
        flow_2d_shape = None
    if flow_3d_zarr_path is not None:
        flow_3d_zarr_root = zarr.open(flow_3d_zarr_path)
        flow_3d_zarr = flow_3d_zarr_root[flow_group]
        flow_3d_shape = flow_3d_zarr.shape
        # Try to load the per-pixel confidence array from the same zarr (optional).
        if "confidence" in flow_3d_zarr_root:
            confidence_3d_zarr = flow_3d_zarr_root["confidence"]
        else:
            confidence_3d_zarr = None
    else:
        flow_3d_zarr = None
        flow_3d_shape = None
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

    # Create wrapper objects that support [timepoint] indexing for lazy loading
    # and is-not-None checks for downstream flow detection
    flow_2d = flow_2d_zarr  # None if no 2D flow
    flow_3d = flow_3d_zarr  # None if no 3D flow
    confidence_3d = confidence_3d_zarr  # None if no confidence

    # max_timepoints truncates the run to a prefix of the movie. img_shape is the
    # shape actually processed, so the timepoint loops, add_disappear and the
    # output zarr all follow from it rather than from the zarr on disk.
    img_shape = (n_frames,) + tuple(fragments.shape[1:])
    raw_shape = (n_frames,) + raw_zarr.shape[2:]
    print(f"Raw image shape: {raw_shape}, segmentation shape: {img_shape}, flow_2d shape: {flow_2d_shape if flow_2d_zarr is not None else None}, flow_3d shape: {flow_3d_shape if flow_3d_zarr is not None else None}")
    axes = get_axes_metadata(seg_zarr_root[seg_group])
    scale = [axis["scale"] for axis in axes]
    # renumber_merge_history needs the largest fragment id up front; take it one
    # frame at a time rather than holding the whole array.
    max_node_id = max(int(fragments[t].max()) for t in range(n_frames))
    img_shape_scaled = [img_shape[0]] + [
        int(img_shape[i] * scale[i]) for i in range(1, len(img_shape))
    ]

    merge_history = create_multihypo_graph.load_merge_history(merge_history_csv_path)
    # Drop merges for timepoints that are not processed. max_node_id above only
    # covers the frames we keep, so leaving them in would let
    # renumber_merge_history hand out ids that collide with fragment ids from
    # the dropped frames and corrupt the merge chains.
    merge_history = merge_history[merge_history[:, 4] < n_frames]
    no_merges = len(merge_history) == 0

    fields = ["a", "b", "c", "cost", "timepoint"]
    if no_merges:
        print("No merge history found. Running in no-merge (fragments-only) mode.")
        # Warn if cohesion/adhesion config is nonzero
        coh_adh_keys = ["cohesion_weight", "cohesion_constant", "adhesion_weight", "adhesion_constant"]
        if any(config.get(k, 0.0) != 0.0 for k in coh_adh_keys):
            print("Warning: cohesion/adhesion weights/constants are nonzero but will be ignored in no-merge mode")
        # Write empty normalized merge history for consistency
        with open(normalized_merge_history_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
    else:
        merge_history = create_multihypo_graph.normalize_costs(merge_history)
        merge_history = create_multihypo_graph.renumber_merge_history(
            merge_history, max_node_id
        )

        # Save the normalized and renumbered merge history
        with open(normalized_merge_history_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for row in merge_history:
                writer.writerow(row)

    z_flow_conf_threshold = config.get("z_flow_conf_threshold", None)
    z_flow_min_pass_pixels = config.get("z_flow_min_pass_pixels", 10)
    if confidence_3d is not None and z_flow_conf_threshold is not None:
        print(
            f"Confidence-based Z flow filtering enabled: threshold={z_flow_conf_threshold}, "
            f"min passing pixels={z_flow_min_pass_pixels}"
        )

    frag_ids = set()
    for timepoint in range(img_shape[0]):
        print(f"Processing timepoint {timepoint}")
        # Lazy-load raw, fragment and flow data for this timepoint
        raw_tp = load_timepoint(raw_zarr, timepoint, channel=0)
        frag_tp = load_timepoint(fragments, timepoint)
        flow_2d_tp = load_timepoint(flow_2d_zarr, timepoint)
        flow_3d_tp = load_timepoint(flow_3d_zarr, timepoint)
        conf_3d_tp = load_timepoint(confidence_3d_zarr, timepoint)
        # Gathered here, while the frame is in hand, instead of in a separate
        # pass over the whole movie
        frag_ids.update(int(v) for v in np.unique(frag_tp))
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
                tp=timepoint,
                scale=scale,
            )
            for node in cand_graph.nodes():
                cand_graph.nodes[node]["num_leaves"] = 1
            exclusion_sets = []
        else:
            cand_graph, exclusion_sets = create_multihypo_graph.nodes_from_fragments(
                frag_tp,
                merge_history[merge_history[:, 4] == timepoint],
                min_cost=config.get("min_merge_cost", config.get("min_merge_score", 0.0)),
                max_cost=config.get("max_merge_cost", config.get("max_merge_score", 1.0)),
                raw_img=raw_tp,
                flow_2d=flow_2d_tp,
                flow_3d=flow_3d_tp,
                confidence_3d=conf_3d_tp,
                z_flow_conf_threshold=z_flow_conf_threshold,
                z_flow_min_pass_pixels=z_flow_min_pass_pixels,
                size_threshold=config["size_threshold"],
                tp=timepoint,
                scale=scale,
            )
        if timepoint == 0:
            all_cand_graph = cand_graph
            all_exclusion_sets = exclusion_sets
        else:
            all_cand_graph = nx.compose(all_cand_graph, cand_graph)
            all_exclusion_sets.extend(exclusion_sets)

    if confidence_3d is not None and z_flow_conf_threshold is not None:
        n_total = all_cand_graph.number_of_nodes()
        n_unreliable = sum(
            1 for _, d in all_cand_graph.nodes(data=True)
            if not d.get("z_flow_reliable", True)
        )
        print(
            f"Z flow reliability: {n_unreliable}/{n_total} nodes flagged unreliable "
            f"({100 * n_unreliable / n_total:.1f}%) — these use XY-only drift_dist"
        )

    utils.add_cand_edges(all_cand_graph, max_edge_distance, max_children=config["max_children"])
    print("Edges before hyperedges: ", all_cand_graph.number_of_edges())
    all_cand_graph = utils.add_hyperedges(all_cand_graph, divisions=config["divisions"])
    print("Edges after hyperedges: ", all_cand_graph.number_of_edges())
    utils.add_appear_ignore_attr(all_cand_graph)
    utils.add_disappear(all_cand_graph, img_shape_scaled)
    track_graph = motile.TrackGraph(all_cand_graph, frame_attribute="time")
    if flow_3d is not None:   
        print("Calculating drift distances using optical flow...")
        utils.add_flow_dist_attr(track_graph)
    elif "drift_distance" in config:
        print("Calculating drift distances using drift_distance parameter...")
        utils.add_drift_dist_attr(track_graph, drift=config["drift_distance"])
    else:
        print("No drift distance or flow provided; setting drift distances to zero.")
        utils.add_drift_dist_attr(track_graph, drift=0)
    utils.add_area_diff_attr(track_graph)
    utils.add_intensity_diff_attr(track_graph)
    utils.add_division_attr(track_graph)

    if stats_only:
        report_graph_statistics(config, track_graph)
        print("Stats-only mode: skipping ILP solve and result saving.")
        return

    # Save candidate edge list for analysis (simple edges only, skip hyperedges)
    cand_edges_path = output_dir / "candidate_edges.npy"
    simple_edges = [(e[0], e[1]) for e in track_graph.edges if isinstance(e[0], (int, np.integer)) and isinstance(e[1], (int, np.integer))]
    if simple_edges:
        cand_edge_list = np.array(simple_edges, dtype=np.int64)
    else:
        cand_edge_list = np.empty((0, 2), dtype=np.int64)
    np.save(cand_edges_path, cand_edge_list)
    print(f"Saved {len(cand_edge_list)} candidate edges to {cand_edges_path}")

    print("Solving tracking with motile...")
    solution_graph = solve_with_motile(config, track_graph, all_exclusion_sets, no_merges=no_merges)

    print("Saving results...")

    frag_ids.discard(0)
    lookup = get_solution_lookup(
        merge_history, solution_graph, frag_ids, max_node_id, fragments.dtype
    )

    assign_tracklet_ids(solution_graph)

    output_zarr_root = zarr.open(output_seg_path, mode="a", shape=img_shape, chunks=(1, 1, 512, 512), dtype=np.uint32)
    output_zarr_root.attrs["axes"] = axes
    for timepoint in range(img_shape[0]):
        output_zarr_root[timepoint] = lookup[fragments[timepoint]]

    # Save tracks to geff file format
    metadata = geff.GeffMetadata(directed=True, 
                                 related_objects=[{
                                     "type": "labels", 
                                     "path": "../pred_seg.zarr",
                                     "node_prop": "label"
                                 }],
                                 node_props_metadata={},
                                 edge_props_metadata={},
                                 )
    geff.write(solution_graph,
               output_filepath_geff,
               axis_names=["time", "z", "y", "x"],
               axis_types=["time", "space", "space", "space"],
               axis_scales=scale,
               metadata=metadata,
               overwrite=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--stats-only", action="store_true",
                        help="Build candidate graph and print attribute statistics, then exit.")
    args = parser.parse_args()
    config = toml.load(args.config)

    raw_base_dir = Path(config["raw_base_dir"])
    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    dataset: str = config["dataset"]
    experiment: str = config["experiment"]
    assert raw_base_dir.is_dir()
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    raw_dir = raw_base_dir / experiment / f"{dataset}.zarr"
    print(f"Loading raw data from {raw_dir}")
    assert raw_dir.is_dir(), f"Raw data directory {raw_dir} is missing"

    seg_dir = input_base_dir / "segmentation" / experiment / dataset / config["seg_result"]
    print(f"Loading segmentation data from {seg_dir}")
    assert seg_dir.is_dir(), f"Segmentation data directory {seg_dir} is missing"

    flow_result = config.get("flow_result", None)
    if flow_result is not None:
        flow_dir_2d = input_base_dir / "opticalflow" / experiment / dataset / "opticalflow_2d" / config["flow_result"]
        flow_dir_3d = input_base_dir / "opticalflow" / experiment / dataset / "opticalflow_3d" / config["flow_result"]
        if not flow_dir_2d.is_dir():
            print(f"2D optical flow directory {flow_dir_2d} does not exist, using 3D flow only.")
            flow_dir_2d = None
        else:
            print(f"Loading 2D optical flow data from {flow_dir_2d}")
        print(f"Loading 3D optical flow data from {flow_dir_3d}")
        assert flow_dir_3d.is_dir(), f"3D optical flow data directory {flow_dir_3d} is missing"
        flow_dirs = {"2d": flow_dir_2d, "3d": flow_dir_3d}
    else:
        flow_dirs = {"2d": None, "3d": None}

    exp_uid = config.get("exp_uid")
    if not exp_uid:
        exp_uid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config["exp_uid"] = exp_uid

    output_dir = output_base_dir / "tracking" / experiment / dataset / exp_uid
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")

    run_tracking(config, raw_dir, seg_dir, flow_dirs, output_dir, stats_only=args.stats_only)
