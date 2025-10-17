from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping

import motile
import networkx as nx
import numpy as np
import scipy
import skimage


def nodes_from_segmentation(
    segmentation: np.ndarray, size_threshold: int | None = None, tp: int = 0
) -> nx.DiGraph:
    """Extract candidate nodes from a segmentation.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x).

        size_threshold (int): A minimum area for candidate nodes. Nodes smaller
            than this area will not be added to the graph.

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
        attrs = {
            "time": int(tp),
            "x": float(regionprop.centroid[2]),
            "y": float(regionprop.centroid[1]),
            "z": float(regionprop.centroid[0]),
            "label": node_id,
            "area": regionprop.area,
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

        # match indices based on a max edge distance
        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        # match indices based on k nearest neighbors
        # _, matched_indices = next_kdtree.query(prev_kdtree.data, k=5)

        for prev_node_id, next_node_indices in zip(prev_node_ids, matched_indices):
            for next_node_index in next_node_indices:
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
    soln_copy = solution_nx_graph.copy()
    for parent_node in parent_nodes:
        out_edges = solution_nx_graph.out_edges(parent_node)
        soln_copy.remove_edges_from(out_edges)
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


def add_disappear(cand_graph):
    for node_id, attrs in cand_graph.nodes(data=True):
        if "time" not in attrs:
            continue  # skip hypernodes
        if attrs.get("time") == 99 or attrs.get("x") > 380:
            cand_graph.nodes[node_id]["ignore_disappear"] = True


def add_drift_dist_attr(cand_graph: motile.TrackGraph, drift=10):
    for edge in cand_graph.edges:
        if cand_graph.is_hyperedge(edge):
            us, vs = edge
            u = us[0]  # assume always one "source" node
            v1, v2 = vs  # assume always two "target" nodes
            pos_u = drift + cand_graph.nodes[u]["x"]
            pos_v = (cand_graph.nodes[v1]["x"] + cand_graph.nodes[v2]["x"]) / 2
        else:
            u, v = edge
            pos_u = drift + cand_graph.nodes[u]["x"]
            pos_v = cand_graph.nodes[v]["x"]

        drift_dist = abs(pos_u - pos_v)
        cand_graph.edges[edge]["drift_dist"] = drift_dist


def add_area_diff_attr(cand_graph: motile.TrackGraph):
    for edge in cand_graph.edges:
        if cand_graph.is_hyperedge(edge):
            us, vs = edge
            u = us[0]  # assume always one "source" node
            v1, v2 = vs  # assume always two "target" nodes
            area_u = cand_graph.nodes[u]["area"]
            area_v = cand_graph.nodes[v1]["area"] + cand_graph.nodes[v2]["area"]
        else:
            u, v = edge
            area_u = cand_graph.nodes[u]["area"]
            area_v = cand_graph.nodes[v]["area"]

        area_diff = np.abs(area_u - area_v)
        cand_graph.edges[edge]["area_diff"] = area_diff


def add_division_hyperedges(candidate_graph: nx.DiGraph) -> nx.DiGraph:
    """Add hyper edges representing specific divisions to the graph

    Args:
        candidate_graph (nx.DiGraph): A candidate graph already populated with
            normal nodes and edges.

    Returns:
        nx.DiGraph: The candidate graph with additional hypernodes for each
            possible division
    """
    nodes_original = list(candidate_graph.nodes)
    for node in nodes_original:
        successors = candidate_graph.successors(node)
        # TODO: iterate through combinations of size 2-5 (5 is max number of edges for single node, set as k nearest neighbors in add_cand_edges)
        # combos = []
        # for i in range(2, 6):
        #     combos.extend(list(combinations(successors, i)))
        pairs = list(combinations(successors, 2))
        for pair in pairs:
            hypernode = str(node) + "_" + str(pair[0]) + "_" + str(pair[1])
            candidate_graph.add_node(hypernode)
            candidate_graph.add_edge(
                node,
                hypernode,
            )
            candidate_graph.add_edge(
                hypernode,
                pair[0],
            )
            candidate_graph.add_edge(
                hypernode,
                pair[1],
            )
    return candidate_graph


def add_merge_hyperedges(candidate_graph: nx.DiGraph) -> nx.DiGraph:
    """Add hyper edges representing specific merges to the graph

    Args:
        candidate_graph (nx.DiGraph): A candidate graph already populated with
            normal nodes and edges.

    Returns:
        nx.DiGraph: The candidate graph with additional hypernodes for each
            possible merge
    """
    nodes_original = list(candidate_graph.nodes)
    for node in nodes_original:
        predecessors = candidate_graph.predecessors(node)
        # TODO: iterate through combinations of size 2-5 (5 is max number of edges for single node, set as k nearest neighbors in add_cand_edges)
        pairs = list(combinations(predecessors, 2))
        for pair in pairs:
            hypernode = str(pair[0]) + "_" + str(pair[1]) + "_" + str(node)
            candidate_graph.add_node(hypernode)
            candidate_graph.add_edge(
                pair[0],
                hypernode,
            )
            candidate_graph.add_edge(
                pair[1],
                hypernode,
            )
            candidate_graph.add_edge(
                hypernode,
                node,
            )
    return candidate_graph


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
