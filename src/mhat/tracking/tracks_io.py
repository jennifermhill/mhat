from __future__ import annotations

from csv import DictReader, DictWriter
from typing import TYPE_CHECKING

import networkx as nx
import zarr
from mhat.tracking.utils import nodes_from_segmentation

if TYPE_CHECKING:
    from pathlib import Path


def load_tracks_from_csv(csv_path: str | Path) -> nx.DiGraph:
    with open(csv_path) as f:
        reader = DictReader(f)
        graph = nx.DiGraph()
        for row in reader:
            node_id = int(float(row["id"]))
            attrs = {
                "time": int(float(row["time"])),
                "x": int(float(row["x"])),
                "y": int(float(row["y"])),
                "pos": [int(float(row["x"])), int(float(row["y"]))],
                "label": node_id,
            }
            parent_id = int(float(row["parent_id"]))
            graph.add_node(node_id, **attrs)
            if parent_id != -1:
                graph.add_edge(parent_id, node_id)
    return graph


def save_tracks_to_csv(tracks: nx.DiGraph, csv_path: str | Path) -> None:
    with open(csv_path, "w") as f:
        writer = DictWriter(f, fieldnames=["time", "x", "y", "z", "id", "parent_id"])
        writer.writeheader()
        for node, data in tracks.nodes(data=True):
            parents = list(tracks.predecessors(node))
            if len(parents) == 1:
                parent_id = parents[0]
            elif len(parents) == 0:
                parent_id = -1
            else:
                raise ValueError(f"Node {node} has too many parents! {parents}")
            row = {
                "time": data["time"],
                "x": data["x"],
                "y": data["y"],
                "z": data["z"],
                "id": node,
                "parent_id": parent_id,
            }
            writer.writerow(row)


def read_gt_tracks(mask_zarr, tracks_file) -> nx.DiGraph:
    """Get the ground truth tracks as a networkx graph
    # TODO: save the gt tracks in the normal csv format with locations

    Args:
        mask_zarr (str | Path): Path to the zarr containing the ground truth mask
        tracks_file (str | Path): path to csv containing ground truth links

    Returns:
        nx.DiGraph: _description_
    """
    data_root = zarr.open(mask_zarr)
    gt_seg = data_root["mask"][:]
    gt_tracks = nodes_from_segmentation(gt_seg)

    with open(tracks_file) as f:
        reader = DictReader(f)
        for row in reader:
            i_d = int(float(row["id"]))
            assert i_d in gt_tracks.nodes, f"node {i_d} not in graph"
            parent_id = int(float(row["parent_id"]))
            if parent_id != -1:
                assert (
                    parent_id in gt_tracks.nodes
                ), f"parent id {parent_id} not in graph"
                gt_tracks.add_edge(parent_id, i_d)
    return gt_tracks
