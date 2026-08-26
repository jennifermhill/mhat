from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping

import motile
import networkx as nx
import numpy as np
import scipy
from scipy import linalg
import skimage

from mhat.tracking.edge_pairs import CurvatureCost


def nodes_from_segmentation(
    segmentation: np.ndarray,
    raw_img: np.ndarray | None = None,
    flow_3d: np.ndarray | None = None,
    flow_2d: np.ndarray | None = None,
    confidence_3d: np.ndarray | None = None,
    z_flow_conf_threshold: float | None = None,
    z_flow_min_pass_pixels: int = 10,
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

        confidence_3d (np.ndarray | None, optional): A per-pixel confidence array
            corresponding to flow_3d (shape matches flow_3d without the vector axis).
            If provided with z_flow_conf_threshold, Z flow is computed only over
            pixels whose |confidence| exceeds the threshold. Nodes with too few
            passing pixels are marked as z_flow_reliable=False and their drift cost
            is computed in XY only.

        z_flow_conf_threshold (float | None, optional): Threshold on |confidence|
            for including a pixel's Z flow in the node average. If None, all in-region
            pixels are included (current default behavior). Defaults to None.

        z_flow_min_pass_pixels (int, optional): Minimum number of pixels within a
            node's region that must pass the confidence threshold for the Z flow
            average to be trusted. Below this, the node is marked z_flow_reliable=False.
            Defaults to 10.

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
        # Read only this label's bounding box, not the whole frame
        sl = regionprop.slice
        region = regionprop.image
        centroid = (float(regionprop.centroid[0] * scale[1]),
                    float(regionprop.centroid[1] * scale[2]),
                    float(regionprop.centroid[2] * scale[3]))
        region_raw = raw_img[sl][region]
        intensity = np.mean(region_raw)
        z_flow_reliable = True
        if flow_3d is not None:
            # Optical flow is stored in pixel units; scale to world units to match
            # the voxel-scaled centroids above so drift_dist computations are consistent.
            # Z component of flow can be filtered by confidence: only in-region pixels
            # whose |confidence| exceeds z_flow_conf_threshold contribute to the average.
            vz_pixels = flow_3d[sl][region][:, 2]
            if confidence_3d is not None and z_flow_conf_threshold is not None:
                conf_pixels = np.abs(confidence_3d[sl][region])
                conf_mask = conf_pixels > z_flow_conf_threshold
                n_passing = int(np.sum(conf_mask))
                if n_passing >= z_flow_min_pass_pixels:
                    flow_z = float(np.mean(vz_pixels[conf_mask]) * scale[1])
                else:
                    # Too few high-confidence pixels — mark Z unreliable.
                    # Store 0 as a placeholder; add_flow_dist_attr will skip Z for this node.
                    flow_z = 0.0
                    z_flow_reliable = False
            else:
                flow_z = float(np.mean(vz_pixels) * scale[1])
            if flow_2d is not None:
                flow = (flow_z,
                        float(np.mean(flow_2d[sl][region][:, 1]) * scale[2]),
                        float(np.mean(flow_2d[sl][region][:, 0]) * scale[3]))
            else:
                flow = (flow_z,
                        float(np.mean(flow_3d[sl][region][:, 1]) * scale[2]),
                        float(np.mean(flow_3d[sl][region][:, 0]) * scale[3]))
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
            "flow": flow,
            "z_flow_reliable": z_flow_reliable,
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
        max_children (int): Maximum number of candidate edges per node to the next frame.
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
    for node, data in solution_nx_graph.nodes(data=True):
        time_frame = data["time"]
        track_id = data["track_id"]
        tracked_masks[time_frame][segmentation[time_frame] == node] = track_id
    return tracked_masks


def build_merge_dict(merge_history) -> dict[int, list[int]]:
    """Map each merged node id to every leaf fragment id underneath it.

    The merge history is ordered, so walking it once and extending from the
    entries already recorded gives each parent the full transitive set of its
    leaves. Ids absent from the result are leaves themselves.
    """
    merge_dict: dict[int, list[int]] = {}
    for merge in merge_history:
        a, b, c, _cost, _tp = merge
        a = int(a)
        b = int(b)
        c = int(c)
        children = [a, b]
        if a in merge_dict:
            children.extend(merge_dict[a])
        if b in merge_dict:
            children.extend(merge_dict[b])
        merge_dict[c] = children
    return merge_dict


def get_solution_lookup(merge_history, solution_graph, frag_ids, max_frag_id, dtype):
    """Build a leaf-fragment-id -> solution-node-id lookup table.

    Applying it to one frame with ``lookup[frame]`` relabels that frame in a
    single vectorized pass, so the movie is never relabelled in memory all at
    once.

    Args:
        merge_history: Renumbered merge history (see renumber_merge_history).
        solution_graph: The solved subgraph whose nodes claim the fragments.
        frag_ids: Every leaf fragment id present in the volume, used to check
            that a node without merges really is a fragment.
        max_frag_id: Largest leaf fragment id; sizes the table.
        dtype: dtype of the fragments array, so the remap preserves it.
    """
    merge_dict = build_merge_dict(merge_history)

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
            # us is always length 1 now: only division hyperedges remain.
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
            # us is always length 1 now: only division hyperedges remain.
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
            # Source is reliable only if all source nodes are reliable.
            z_reliable = all(
                cand_graph.nodes[n].get("z_flow_reliable", True) for n in us
            )
        else:
            u, v = edge
            node_u = cand_graph.nodes[u]
            node_v = cand_graph.nodes[v]
            flow_u = node_u["flow"]
            pos_u = np.array([node_u["z"], node_u["y"], node_u["x"]])
            pos_v = np.array([node_v["z"], node_v["y"], node_v["x"]])
            z_reliable = node_u.get("z_flow_reliable", True)

        # Apply flow to pos_u and compute distance. When Z flow is unreliable
        # for this source, drop the Z component from both position and flow so
        # drift_dist reflects XY motion only (not penalizing the edge for Z
        # prediction error we can't estimate).
        predicted = pos_u + np.array(flow_u)
        if z_reliable:
            flow_dist = linalg.norm(predicted - pos_v)
        else:
            flow_dist = linalg.norm(predicted[1:] - pos_v[1:])
        cand_graph.edges[edge]["drift_dist"] = flow_dist


def add_area_diff_attr(cand_graph: motile.TrackGraph):
    for edge in cand_graph.edges:
        if cand_graph.is_hyperedge(edge):
            # us is always length 1 now: only division hyperedges remain.
            us, vs = edge
            area_u = sum(cand_graph.nodes[n]["area"] for n in us)
            area_v = sum(cand_graph.nodes[n]["area"] for n in vs)
        else:
            u, v = edge
            area_u = cand_graph.nodes[u]["area"]
            area_v = cand_graph.nodes[v]["area"]

        area_diff = np.abs(area_u - area_v)/np.mean([area_u, area_v])
        cand_graph.edges[edge]["area_diff"] = area_diff


def _combined_intensity(cand_graph: motile.TrackGraph, nodes) -> float:
    """Mean intensity of a set of nodes treated as a single object.

    The intensity attribute is a per-object mean, so unlike area it does not add
    across nodes. Weighting by area reproduces the mean over the combined region
    exactly, which is what the node on the other side of the hyperedge measures.
    """
    intensities = [cand_graph.nodes[n]["intensity"] for n in nodes]
    areas = [cand_graph.nodes[n]["area"] for n in nodes]
    return float(np.average(intensities, weights=areas))


def add_intensity_diff_attr(cand_graph: motile.TrackGraph):
    for edge in cand_graph.edges:
        if cand_graph.is_hyperedge(edge):
            # us is always length 1 now: only division hyperedges remain.
            us, vs = edge
            intensity_u = _combined_intensity(cand_graph, us)
            intensity_v = _combined_intensity(cand_graph, vs)
        else:
            u, v = edge
            intensity_u = cand_graph.nodes[u]["intensity"]
            intensity_v = cand_graph.nodes[v]["intensity"]

        intensity_diff = np.abs(intensity_u - intensity_v)
        cand_graph.edges[edge]["intensity_diff"] = intensity_diff


def add_division_attr(cand_graph: motile.TrackGraph):
    """Flag division hyperedges with a 0/1 attribute so they can be given a cost.

    motile's built in Split cost cannot price these divisions: it applies to the
    NodeSplit variable, which only turns on when a node has two or more selected
    outgoing edges, whereas a division here is a single selected hyperedge. The
    cost therefore has to sit on the edge, as a weight on this indicator.

    Every edge gets the attribute, including ordinary ones, because EdgeSelection
    looks it up on each edge it is applied to.
    """
    for edge in cand_graph.edges:
        is_division = (
            cand_graph.is_hyperedge(edge) and len(edge[0]) == 1 and len(edge[1]) > 1
        )
        cand_graph.edges[edge]["is_division"] = float(is_division)


def add_hyperedges(candidate_graph: nx.DiGraph, divisions: bool = True) -> nx.DiGraph:
    """Add hyper edges representing specific divisions to the graph

    A division hyperedge points at exactly two successors. That pairwise
    constraint is fixed here rather than with MaxChildren, because motile counts
    a hyperedge as a single outgoing edge however many nodes it connects.

    Track merging — a hyperedge from two predecessors into one successor — is no
    longer supported. Note this is unrelated to the merging of segmentation
    fragments, which happens upstream in create_multihypo_graph and is
    unaffected.

    Args:
        candidate_graph (nx.DiGraph): A candidate graph already populated with
            normal nodes and edges.
        divisions (bool, optional): Whether to add division hyperedges. Defaults to True.

    Returns:
        nx.DiGraph: The candidate graph with additional hypernodes for each
            possible pairwise division
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

    candidate_graph.add_nodes_from(hypernodes)
    candidate_graph.add_edges_from(hyperedges)
    
    return candidate_graph

def report_graph_statistics(config, track_graph):
    """Print mean/std of graph attributes and their ILP costs.

    Node costs are scaled by num_leaves to reflect the actual costs
    seen by the ILP solver (LeavesScaledNodeSelection bakes num_leaves
    into the node feature values).
    """
    # Collect node attributes (only those with ILP cost parameters)
    node_attrs = {"cohesion": ([], []), "adhesion": ([], [])}
    for node_id, data in track_graph.nodes.items():
        num_leaves = data.get("num_leaves", 1)
        for attr in node_attrs:
            if attr in data:
                node_attrs[attr][0].append(data[attr])
                node_attrs[attr][1].append(num_leaves)

    # Collect edge attributes
    edge_attrs = {"drift_dist": [], "area_diff": [], "intensity_diff": []}
    for edge_key, data in track_graph.edges.items():
        for attr in edge_attrs:
            if attr in data:
                edge_attrs[attr].append(data[attr])

    curvature_cost = CurvatureCost(position_attribute="centroid")
    curvature_values = []
    for node in track_graph.nodes:
        in_edges = list(track_graph.prev_edges[node])
        out_edges = list(track_graph.next_edges[node])
        for in_edge in in_edges:
            in_offset = curvature_cost.get_edge_offset(track_graph, in_edge)
            for out_edge in out_edges:
                out_offset = curvature_cost.get_edge_offset(track_graph, out_edge)
                curvature_values.append(np.linalg.norm(out_offset - in_offset))

    # Config parameter mapping
    param_map = {
        "cohesion": ("cohesion_weight", "cohesion_constant"),
        "adhesion": ("adhesion_weight", "adhesion_constant"),
        "drift_dist": ("drift_weight", "drift_constant"),
        "area_diff": ("area_weight", "area_constant"),
        "intensity_diff": ("intensity_weight", "intensity_constant"),
        "curvature": ("curvature_weight", "curvature_constant"),
    }

    print("\n" + "=" * 100)
    print("Graph Attribute Statistics (node costs scaled by num_leaves)")
    print("=" * 100)
    header = f"{'Attribute':<16} {'Count':>6} {'Mean':>10} {'Std':>10} {'Weight':>10} {'Constant':>10} {'Cost Mean':>12} {'Cost Std':>12}"
    print(header)
    print("-" * 100)

    # Node attributes: scale costs by num_leaves
    for attr, (values, leaves) in node_attrs.items():
        if not values:
            continue
        arr = np.array(values)
        leaves_arr = np.array(leaves)
        w_key, c_key = param_map[attr]
        weight = config.get(w_key, 0.0)
        constant = config.get(c_key, 0.0)
        costs = (weight * arr + constant) * leaves_arr
        print(f"{attr:<16} {len(arr):>6} {arr.mean():>10.3f} {arr.std():>10.3f} {weight:>10.1f} {constant:>10.1f} {costs.mean():>12.1f} {costs.std():>12.1f}")

    # Edge attributes: no leaves scaling
    all_edge_attrs = {**edge_attrs, "curvature": curvature_values}
    for attr, values in all_edge_attrs.items():
        if not values:
            continue
        arr = np.array(values)
        w_key, c_key = param_map[attr]
        weight = config.get(w_key, 0.0)
        constant = config.get(c_key, 0.0)
        costs = weight * arr + constant
        print(f"{attr:<16} {len(arr):>6} {arr.mean():>10.3f} {arr.std():>10.3f} {weight:>10.1f} {constant:>10.1f} {costs.mean():>12.1f} {costs.std():>12.1f}")

    print("=" * 100 + "\n")

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
            # us is always length 1 now: only division hyperedges remain.
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
