import sys
import argparse
import datetime
from pathlib import Path

import ilpy
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


def run_tracking(config, input_video_path: Path, output_video_path: Path, exp_name):

    exp_path = output_video_path / exp_name
    exp_path.mkdir()
    input_zarr_path = input_video_path / "data.zarr"
    output_seg_path = exp_path / "pred_seg.zarr"
    merge_history_csv_path = input_video_path / "merge_history.csv"
    config_filepath = exp_path / "config.toml"
    output_filepath_csv = exp_path / "pred_tracks.csv"
    output_filepath_graphml = exp_path / "pred_tracks.graphml"
    output_filepath_geff = exp_path / "pred_tracks.zarr"

    with open(config_filepath, "w") as config_file:
        toml.dump(config, config_file)

    seg_group = "fragments"

    max_edge_distance = config["max_edge_distance"]

    input_zarr_root = zarr.open(input_zarr_path)
    fragments = input_zarr_root[seg_group][:]
    max_node_id = np.max(fragments)

    merge_history = create_multihypo_graph.load_merge_history(merge_history_csv_path)
    merge_history = create_multihypo_graph.renumber_merge_history(
        merge_history, max_node_id
    )

    for timepoint in range(fragments.shape[0]):
        print(f"Processing timepoint {timepoint}")
        cand_graph, exclusion_sets = create_multihypo_graph.nodes_from_fragments(
            fragments[timepoint],
            merge_history[merge_history[:, 4] == timepoint],
            min_score=config["min_merge_score"],
            max_score=config["max_merge_score"],
            size_threshold=config["size_threshold"],
        )
        if timepoint == 0:
            all_cand_graph = cand_graph
            all_exclusion_sets = exclusion_sets
        else:
            all_cand_graph = nx.compose(all_cand_graph, cand_graph)
            all_exclusion_sets.extend(exclusion_sets)

    utils.add_cand_edges(all_cand_graph, max_edge_distance)
    print("Edges before hyperedges: ", all_cand_graph.number_of_edges())
    all_cand_graph = utils.add_division_hyperedges(all_cand_graph)
    #all_cand_graph = utils.add_merge_hyperedges(all_cand_graph)
    print("Edges after hyperedges: ", all_cand_graph.number_of_edges())
    utils.add_appear_ignore_attr(all_cand_graph)
    utils.add_disappear(all_cand_graph)
    track_graph = motile.TrackGraph(all_cand_graph, frame_attribute="time")
    utils.add_drift_dist_attr(track_graph, drift=config["drift"])
    utils.add_area_diff_attr(track_graph)

    solution_graph = solve_with_motile(config, track_graph, all_exclusion_sets)

    print("Saving results...")

    save_tracks_to_csv(solution_graph, output_filepath_csv)
    # Save tracks to geff file format
    geff.write(solution_graph, output_filepath_geff)
    nx.write_graphml(solution_graph, output_filepath_graphml)
    solution_seg = get_solution_seg(fragments, merge_history, solution_graph)
    assign_tracklet_ids(solution_graph)
    solution_seg = utils.relabel_segmentation(solution_graph, solution_seg)
    output_zarr_root = zarr.open(output_seg_path, mode="a", shape=fragments.shape, chunks=(1, 1, 256, 256), dtype=np.uint32)
    output_zarr_root[:] = solution_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    dataset: str = config["dataset"]
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    data_dir = input_base_dir / dataset
    assert data_dir.is_dir()

    current_datetime = datetime.datetime.now()
    exp_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    config["exp_name"] = exp_name
    print(exp_name)

    output_dataset_dir = output_base_dir / dataset

    for video_dir in data_dir.iterdir():
        if check_video_dir(video_dir):
            vid_name = video_dir.stem
            output_video_dir = output_dataset_dir / vid_name
            output_video_dir.mkdir(exist_ok=True, parents=True)
            print("Writing tracking output to ", output_video_dir)
            run_tracking(config, video_dir, output_video_dir, exp_name)
