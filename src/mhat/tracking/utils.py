from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping

import motile
import networkx as nx
import numpy as np
import scipy
from scipy import linalg
import skimage
from motile.variables import NodeSelected
from line_profiler import profile


def nodes_from_segmentation(
    segmentation: np.ndarray,
    raw_img: np.ndarray | None = None,
    flow_3d: np.ndarray | None = None,
    flow_2d: np.ndarray | None = None,
    size_threshold: int | None = None,
    tp: int = 0,
    scale: list[float] = [1.0, 1.0, 1.0, 1.0]
) -> nx.DiGraph:
    """Extract candidate nodes from a segmentation.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (z, y, x).

        flow_3d (np.ndarray | None, optional): A numpy array with 3D flow vectors
            for each pixel in the segmentation. Defaults to None.

        flow_2d (np.ndarray | None, optional): A numpy array with 2D flow vectors
            for each pixel in the segmentation. Defaults to None.

        size_threshold (int): A minimum area for candidate nodes. Nodes smaller
            than this area will not be added to the graph.

        tp (int, optional): The timepoint to assign to the nodes. Defaults to 0.

        scale (list[float], optional): The scaling factors for each axis of
            the fragments array. Defaults to [1.0, 1.0, 1.0, 1.0].

    Returns:
        nx.DiGraph: A candidate graph with only nodes.
    """
    cand_graph = nx.DiGraph()
    # for t in range(len(segmentation)):
    #     seg_frame = segmentation[t]
    props = skimage.measure.regionprops(segmentation)
    for regionprop in props:
        if size_threshold and regionprop.area < size_threshold:
            continue
        node_id = int(regionprop.label)
        region = segmentation == node_id
        centroid = (float(regionprop.centroid[0] * scale[1]), 
                    float(regionprop.centroid[1] * scale[2]), 
                    float(regionprop.centroid[2] * scale[3]))
        region_raw = raw_img[region]
        intensity = np.mean(region_raw)
        if flow_3d is not None:
            if flow_2d is not None:
                flow = (np.mean(flow_3d[region][:, 2]), np.mean(flow_2d[region][:, 1]), np.mean(flow_2d[region][:, 0]))
            else:
                flow = (np.mean(flow_3d[region][:, 2]), np.mean(flow_3d[region][:, 1]), np.mean(flow_3d[region][:, 0]))
        else:
            flow = 0
        attrs = {
            "time": int(tp),
            "x": centroid[2],
            "y": centroid[1],
            "z": centroid[0],
            "centroid": centroid,
            "label": node_id,
            "area": regionprop.area,
            "intensity": intensity,
            "flow": flow
        }
        cand_graph.add_node(node_id, **attrs)

    return cand_graph


def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data["time"]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict


def create_kdtree(
    cand_graph: nx.DiGraph, node_ids: Iterable[Any]
) -> scipy.spatial.KDTree:
    positions = [
        [cand_graph.nodes[node]["x"], cand_graph.nodes[node]["y"], cand_graph.nodes[node]["z"]] for node in node_ids
    ]
    return scipy.spatial.KDTree(positions)


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    max_children: int,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    node_frame_dict = _compute_node_frame_dict(cand_graph)

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in frames:
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        # match indices based on k nearest neighbors
        _, matched_indices = next_kdtree.query(prev_kdtree.data, k=max_children, distance_upper_bound=max_edge_distance)

        if max_children == 1:
            for prev_node_id, next_node_index in zip(prev_node_ids, matched_indices):
                if next_node_index == len(next_node_ids):
                    continue
                next_node_id = next_node_ids[next_node_index]
                cand_graph.add_edge(prev_node_id, next_node_id)
        else:
            for prev_node_id, next_node_indices in zip(prev_node_ids, matched_indices):
                for next_node_index in next_node_indices:
                    if next_node_index == len(next_node_ids):
                        continue
                    next_node_id = next_node_ids[next_node_index]
                    cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree


def relabel_segmentation(
    solution_nx_graph: nx.DiGraph,
    segmentation: np.ndarray,
) -> np.ndarray:
    """Relabel a segmentation based on tracking results so that nodes in same
    track share the same id. IDs do change at division.

    Args:
        solution_nx_graph (nx.DiGraph): Networkx graph with the solution to use
            for relabeling. Nodes not in graph will be removed from seg.
        segmentation (np.ndarray): Original (potentially multi-hypothesis)
            segmentation with dimensions (t,h,[z],y,x), where h is 1 for single
            input segmentation.

    Returns:
        np.ndarray: Relabeled segmentation array where nodes in same track share same
            id with shape (t,1,[z],y,x)
    """
    tracked_masks = np.zeros_like(segmentation)
    id_counter = 1
    parent_nodes = [n for (n, d) in solution_nx_graph.out_degree() if d > 1]
    child_nodes = [n for (n, d) in solution_nx_graph.in_degree() if d > 1]
    soln_copy = solution_nx_graph.copy()
    for parent_node in parent_nodes:
        out_edges = solution_nx_graph.out_edges(parent_node)
        soln_copy.remove_edges_from(out_edges)
    for child_node in child_nodes:
        in_edges = solution_nx_graph.in_edges(child_node)
        for in_edge in in_edges:
            if soln_copy.has_edge(in_edge[0], in_edge[1]):
                soln_copy.remove_edge(in_edge[0], in_edge[1])
    for node_set in nx.weakly_connected_components(soln_copy):
        for node in node_set:
            time_frame = solution_nx_graph.nodes[node]["time"]
            previous_seg_id = node
            previous_seg_mask = segmentation[time_frame] == previous_seg_id
            tracked_masks[time_frame][previous_seg_mask] = id_counter
        id_counter += 1
    return tracked_masks


def add_appear_ignore_attr(cand_graph):
    for node_id, attrs in cand_graph.nodes(data=True):
        if "time" not in attrs:
            continue  # skip hypernodes
        if attrs.get("time") == 0:
            cand_graph.nodes[node_id]["ignore_appear"] = True


def add_disappear(cand_graph, img_shape):
    for node_id, attrs in cand_graph.nodes(data=True):
        if "time" not in attrs:
            continue  # skip hypernodes
        T, Z, Y, X = img_shape
        if attrs.get("time") == T - 1 or attrs.get("z") > Z or attrs.get("y") > Y or attrs.get("x") > X:
            cand_graph.nodes[node_id]["ignore_disappear"] = True


def add_drift_dist_attr(cand_graph: motile.TrackGraph, drift=[0, 0, 0]):
    if isinstance(drift, (int, float)):
        drift = [drift, drift, drift]
    for edge in cand_graph.edges:
        # TODO: fix to include y and z pos
        if cand_graph.is_hyperedge(edge):
            us, vs = edge
            pos_u = np.array([
                (sum(cand_graph.nodes[n]["z"] for n in us)) / len(us),
                (sum(cand_graph.nodes[n]["y"] for n in us)) / len(us),
                (sum(cand_graph.nodes[n]["x"] for n in us)) / len(us)
            ])
            pos_v = np.array([
                (sum(cand_graph.nodes[n]["z"] for n in vs)) / len(vs),
                (sum(cand_graph.nodes[n]["y"] for n in vs)) / len(vs),
                (sum(cand_graph.nodes[n]["x"] for n in vs)) / len(vs)
            ])
        else:
            u, v = edge
            node_u = cand_graph.nodes[u]
            node_v = cand_graph.nodes[v]
            pos_u = np.array([node_u["z"], node_u["y"], node_u["x"]])
            pos_v = np.array([node_v["z"], node_v["y"], node_v["x"]])

        # Add drift to pos_u and compute distance
        drift_dist = linalg.norm(pos_u + drift - pos_v)
        cand_graph.edges[edge]["drift_dist"] = drift_dist

def add_flow_dist_attr(cand_graph: motile.TrackGraph):
    # TODO: combine with drift distance function above and default to drift distance if no flow
    for edge in cand_graph.edges:
        if cand_graph.is_hyperedge(edge):
            us, vs = edge
            flow_u = np.mean([cand_graph.nodes[n]["flow"] for n in us], axis=0)
            pos_u = np.array([
                (sum(cand_graph.nodes[n]["z"] for n in us)) / len(us),
                (sum(cand_graph.nodes[n]["y"] for n in us)) / len(us),
                (sum(cand_graph.nodes[n]["x"] for n in us)) / len(us)
            ])
            pos_v = np.array([
                (sum(cand_graph.nodes[n]["z"] for n in vs)) / len(vs),
                (sum(cand_graph.nodes[n]["y"] for n in vs)) / len(vs),
                (sum(cand_graph.nodes[n]["x"] for n in vs)) / len(vs)
            ])
        else:
            u, v = edge
            node_u = cand_graph.nodes[u]
            node_v = cand_graph.nodes[v]
            flow_u = node_u["flow"]
            pos_u = np.array([node_u["z"], node_u["y"], node_u["x"]])
            pos_v = np.array([node_v["z"], node_v["y"], node_v["x"]])

        # Apply flow to pos_u and compute distance
        flow_dist = linalg.norm(pos_u + np.array(flow_u) - pos_v)
        cand_graph.edges[edge]["drift_dist"] = flow_dist


def add_area_diff_attr(cand_graph: motile.TrackGraph):
    for edge in cand_graph.edges:
        if cand_graph.is_hyperedge(edge):
            us, vs = edge
            area_u = sum(cand_graph.nodes[n]["area"] for n in us)
            area_v = sum(cand_graph.nodes[n]["area"] for n in vs)
        else:
            u, v = edge
            area_u = cand_graph.nodes[u]["area"]
            area_v = cand_graph.nodes[v]["area"]

        area_diff = np.abs(area_u - area_v)/np.mean([area_u, area_v])
        cand_graph.edges[edge]["area_diff"] = area_diff


def add_intensity_diff_attr(cand_graph: motile.TrackGraph):
    for edge in cand_graph.edges:
        if cand_graph.is_hyperedge(edge):
            us, vs = edge
            intensity_u = sum(cand_graph.nodes[n]["intensity"] for n in us)
            intensity_v = sum(cand_graph.nodes[n]["intensity"] for n in vs)
        else:
            u, v = edge
            intensity_u = cand_graph.nodes[u]["intensity"]
            intensity_v = cand_graph.nodes[v]["intensity"]

        intensity_diff = np.abs(intensity_u - intensity_v)
        cand_graph.edges[edge]["intensity_diff"] = intensity_diff


@profile
def add_hyperedges(candidate_graph: nx.DiGraph, divisions: bool = True, merges: bool = True) -> nx.DiGraph:
    """Add hyper edges representing specific merges and divisions to the graph

    Hyperedges are pairwise: a division hyperedge points at exactly two
    successors, and a merge hyperedge at exactly two predecessors. The constraint is
    fixed here rather than with MaxChildren/MaxParents, because motile
    counts a hyperedge as a single outgoing/incoming edge however many nodes it
    connects.

    Args:
        candidate_graph (nx.DiGraph): A candidate graph already populated with
            normal nodes and edges.
        divisions (bool, optional): Whether to add division hyperedges. Defaults to True.
        merges (bool, optional): Whether to add merge hyperedges. Defaults to True.

    Returns:
        nx.DiGraph: The candidate graph with additional hypernodes for each
            possible pairwise merge and division
    """
    nodes_original = list(candidate_graph.nodes)
    hypernodes = []
    hyperedges = []
    for node in nodes_original:
        if divisions:
            successors = candidate_graph.successors(node)
            successor_combos = combinations(successors, 2)
            for succ_combo in successor_combos:
                hypernode_succ = str(node) + "_" + "_".join(map(str, succ_combo))
                hypernodes.append(hypernode_succ)
                hyperedges.append((node, hypernode_succ))
                for item in succ_combo:
                    hyperedges.append((hypernode_succ, item))
        if merges:
            predecessors = candidate_graph.predecessors(node)
            predecessor_combos = combinations(predecessors, 2)
            for pred_combo in predecessor_combos:
                hypernode_pred = str(node) + "_" + "_".join(map(str, pred_combo))
                hypernodes.append(hypernode_pred)
                hyperedges.append((hypernode_pred, node))
                for item in pred_combo:
                    hyperedges.append((item, hypernode_pred))

    candidate_graph.add_nodes_from(hypernodes)
    candidate_graph.add_edges_from(hyperedges)
    
    return candidate_graph

def scale_by_leaves(solver: motile.Solver) -> None:
    """Scale the costs of the node variables by the number of leaves in the merge they represent."""
    node_vars = solver.get_variables(NodeSelected)
    # Access .costs to trigger computation and cache the result
    costs = solver.costs
    for node_id in node_vars:
        index = node_vars._index_map[node_id]
        num_leaves = solver.graph.nodes[node_id].get("num_leaves", 1)
        if num_leaves > 1:
            costs[index] *= num_leaves


def to_nx_graph(graph, flatten_hyperedges: bool = True) -> nx.DiGraph:
    """Convert a this TrackGraph into a networkx DiGraph.
    Args:
        flatten_hyperedges (bool, optional): If True, include one edge for each
            (source, target) combo in a hyperedge. If False, introduce a new
            hypernode to represent hyperedges. Defaults to True.
    Returns:
        networkx.DiGraph: Directed networkx graph with same nodes, edges, and
            attributes.
    """
    nx_graph = nx.DiGraph()
    nodes_list = list(graph.nodes.items())
    nx_graph.add_nodes_from(nodes_list)
    edges_list: list[tuple[Any, Any, Mapping]] = []
    for edge, data in graph.edges.items():
        if graph.is_hyperedge(edge):
            us, vs = edge
            if flatten_hyperedges:
                # flatten the hyperedges into multiple normal edges
                for u in us:
                    for v in vs:
                        edges_list.append((u, v, {}))
            else:
                # add a hypernode to connect all in nodes with all out nodes
                hypernode_id = "_".join(list(map(str, us)) + list(map(str, vs)))
                nx_graph.add_node(hypernode_id, **data)
                for u in us:
                    edges_list.append((u, hypernode_id, {}))
                for v in vs:
                    edges_list.append((hypernode_id, v, {}))
        else:
            u, v = edge  # type: ignore
            edges_list.append((u, v, data))

    nx_graph.add_edges_from(edges_list)
    return nx_graph

def to_ctc_format(solution_nx_graph: nx.DiGraph) -> list[tuple[int, int, int, int]]:
    """Convert a solution graph into the format expected by the Cell Tracking Challenge
    for evaluation.

    Args:
        solution_nx_graph (nx.DiGraph): A networkx DiGraph representing the solution tracks.
    Returns:
        list[tuple[int, int, int, int]]: A list of tuples representing the tracks in the format (track_id, first frame, last frame, parent_id).
    """
    ctc_tracks = []
    # Make list of unique track_ids
    track_ids = set(nx.get_node_attributes(solution_nx_graph, "track_id").values())
    for track_id in track_ids:
        track_nodes = [n for n, d in solution_nx_graph.nodes(data=True) if d.get("track_id") == track_id]
        frames = [solution_nx_graph.nodes[n]["time"] for n in track_nodes]
        first_frame = min(frames)
        last_frame = max(frames)
        parent_id = solution_nx_graph.nodes[track_nodes[0]].get("parent_id", 0)
        ctc_tracks.append((track_id, first_frame, last_frame, parent_id))
    return ctc_tracks
