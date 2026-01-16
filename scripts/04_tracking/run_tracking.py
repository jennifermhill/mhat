import csv
import argparse
import datetime
from pathlib import Path

import motile
import numpy as np
import toml
import zarr
import geff
import networkx as nx

from mhat.evaluation.eval_io import check_video_dir
from mhat.tracking import create_multihypo_graph, solve_with_motile, utils
from mhat.tracking.tracks_io import save_tracks_to_csv
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids


def get_solution_seg(fragments, merge_history, solution_graph):
    solution_seg = np.zeros_like(fragments)

    merge_dict = {}
    for merge in merge_history:
        a, b, c, score, tp = merge
        a = int(a)
        b = int(b)
        c = int(c)
        children = [a, b]
        if a in merge_dict:
            children.extend(merge_dict[a])
        if b in merge_dict:
            children.extend(merge_dict[b])
        merge_dict[c] = children

    frag_ids = set(np.unique(fragments))
    frag_ids.remove(0)
    
    for node in solution_graph.nodes():
        if node in merge_dict:
            children = merge_dict[node]
        else:
            assert node in frag_ids, f"Node {node} not in merge dict or frag ids"
            children = [node]

        for child in children:
            assert np.all(
                [solution_seg[fragments == child] == 0]
            ), f"Child {child} fragment is already selected"
            solution_seg[fragments == child] = node

    return solution_seg


def run_tracking(config, seg_dir: Path, flow_dir: Path, output_dir: Path):

    seg_zarr_path = seg_dir / "data.zarr"
    flow_zarr_path = flow_dir / "data.zarr" if flow_dir is not None else None
    output_seg_path = output_dir / "pred_seg.zarr"
    merge_history_csv_path = seg_dir / "merge_history.csv"
    normalized_merge_history_csv_path = output_dir / "normalized_merge_history.csv"
    config_filepath = output_dir / "config.toml"
    output_filepath_csv = output_dir / "pred_tracks.csv"
    output_filepath_graphml = output_dir / "pred_tracks.graphml"
    output_filepath_geff = output_dir / "pred_tracks.zarr"

    with open(config_filepath, "w") as config_file:
        toml.dump(config, config_file)

    seg_group = "fragments"

    max_edge_distance = config["max_edge_distance"]

    seg_zarr_root = zarr.open(seg_zarr_path)
    fragments = seg_zarr_root[seg_group][:]
    axes = seg_zarr_root[seg_group].attrs.get("axes", None)
    if axes is not None:
        for axis in axes:
            if axis["scale"] is None:
                scale
            else:
                axis["scale"] = float(axis["scale"])
        scale = [axis["scale"] for axis in axes if "scale" in axis]
    else:
        scale = [1.0, 1.0, 1.0, 1.0]
    max_node_id = np.max(fragments)
    img_shape = fragments.shape
    img_shape_scaled = [
        int(img_shape[i] * scale[i]) for i in range(len(img_shape))
    ]

    merge_history = create_multihypo_graph.load_merge_history(merge_history_csv_path)
    merge_history = create_multihypo_graph.normalize_scores(merge_history)
    merge_history = create_multihypo_graph.renumber_merge_history(
        merge_history, max_node_id
    )

    # Save the normalized and renumbered merge history
    fields = ["a", "b", "c", "score", "timepoint"]

    with open(normalized_merge_history_csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in merge_history:
            writer.writerow(row)

    for timepoint in range(img_shape[0]):
        print(f"Processing timepoint {timepoint}")
        cand_graph, exclusion_sets = create_multihypo_graph.nodes_from_fragments(
            fragments[timepoint],
            merge_history[merge_history[:, 4] == timepoint],
            min_score=config["min_merge_score"],
            max_score=config["max_merge_score"],
            size_threshold=config["size_threshold"],
            scale=scale,
        )
        if timepoint == 0:
            all_cand_graph = cand_graph
            all_exclusion_sets = exclusion_sets
        else:
            all_cand_graph = nx.compose(all_cand_graph, cand_graph)
            all_exclusion_sets.extend(exclusion_sets)

    utils.add_cand_edges(all_cand_graph, max_edge_distance, max_children=config["max_children"])
    print("Edges before hyperedges: ", all_cand_graph.number_of_edges())
    all_cand_graph = utils.add_hyperedges(all_cand_graph)
    print("Edges after hyperedges: ", all_cand_graph.number_of_edges())
    utils.add_appear_ignore_attr(all_cand_graph)
    utils.add_disappear(all_cand_graph, img_shape_scaled)
    track_graph = motile.TrackGraph(all_cand_graph, frame_attribute="time")
    utils.add_drift_dist_attr(track_graph, drift=config["drift_distance"])
    utils.add_area_diff_attr(track_graph)

    print("Solving tracking with motile...")
    solution_graph = solve_with_motile(config, track_graph, all_exclusion_sets)

    print("Saving results...")

    save_tracks_to_csv(solution_graph, output_filepath_csv)
    # Save tracks to geff file format
    geff.write(solution_graph, output_filepath_geff, axis_names=["time", "z", "y", "x"], axis_types=["time", "space", "space", "space"], axis_scales=scale)
    nx.write_graphml(solution_graph, output_filepath_graphml)
    solution_seg = get_solution_seg(fragments, merge_history, solution_graph)
    assign_tracklet_ids(solution_graph)
    solution_seg = utils.relabel_segmentation(solution_graph, solution_seg)
    output_zarr_root = zarr.open(output_seg_path, mode="a", shape=fragments.shape, chunks=(1, 1, 512, 512), dtype=np.uint32)
    if axes is not None:
        output_zarr_root.attrs["axes"] = axes
    output_zarr_root[:] = solution_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    dataset: str = config["dataset"]
    experiment: str = config["experiment"]
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    seg_dir = input_base_dir / "segmentation" / experiment / dataset / config["seg_result"]
    print(f"Loading segmentation data from {seg_dir}")
    assert seg_dir.is_dir()

    flow_result = config.get("flow_result", None)
    if flow_result is not None:
        flow_dir = input_base_dir / "opticalflow" / experiment / dataset / config["flow_result"]
        print(f"Loading optical flow data from {flow_dir}")
        assert flow_dir.is_dir()
    else:
        flow_dir = None

    current_datetime = datetime.datetime.now()
    exp_uid = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    config["exp_uid"] = exp_uid

    output_dir = output_base_dir / "tracking" / experiment / dataset / exp_uid
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")

    run_tracking(config, seg_dir, flow_dir, output_dir)