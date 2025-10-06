import csv
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .utils import nodes_from_segmentation


def load_merge_history(merge_path: Path) -> np.ndarray:
    """Load the merge history from a csv

    Args:
        merge_path (Path): Path to csv containing the merge history

    Returns:
        np.ndarray: The merge history as a nparray with shape (4, N) where
            N is the number of merges. The columns are a, b, c, score.
    """
    merge_history = []
    with open(merge_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["a"])
            b = int(row["b"])
            c = int(row["c"])
            score = float(row["score"])
            timepoint = int(row["timepoint"])
            merge_history.append([a, b, c, score, timepoint])

    merge_history = np.array(merge_history)
    return merge_history


def renumber_merge_history(merge_history: np.ndarray, max_node_id: int) -> np.ndarray:
    """Renumber the merge history to produce a new id from each merge,
    instead of reusing ids.

    Args:
        merge_history (np.ndarray): The merge history array with shape (4, N)
            where N is the number of merges. The columns are a, b, c, score.
        max_node_id (int): The maximum node id from the fragments used to
            generate the merge_history. This is not necessarily the max
            of the first three columns if there is an un-merged fragment with
            high ID.

    Returns:
        np.ndarray: The merge history, but updated so that each merge produces
            a unique ID that is greater than max_node_id
    """
    # renumber all the merges to be new ids
    for idx in range(merge_history.shape[0]):
        max_node_id += 1
        row = merge_history[idx]
        c = row[2]
        merge_history[idx][2] = max_node_id
        # replace all instances of c after this row and later with new node id
        if idx < merge_history.shape[0]:
            merge_history[idx + 1 :][merge_history[idx + 1 :] == c] = max_node_id

    # renumber for each timepoint
    timepoints = np.unique(merge_history[:, 4])
    prev_max_id = np.max(merge_history[merge_history[:, 4] == 0][:, 2])
    for tp in timepoints:
        # Add the previous max id to all a, b, c values in this timepoint
        merge_history[merge_history[:, 4] == tp][:, :3] += prev_max_id

        # Find the new highest node ID from this timepoint's merges
        prev_max_id = np.max(merge_history[merge_history[:, 4] == tp][:, 2])

    return merge_history


def compute_conflicts(
    conflict_sets: dict[int, list[list[int]]], a: int, b: int, c: int
):
    """Keep track of which segmentations conflict with each other through
    the merge history.

    Args:
        conflict_sets (dict[int, list[list[int]]]): a map from
            parent node (largest node in the merge history) to conflict sets
            with that node
        a (int): One child id of the merge
        b (int): Another child id of the merge
        c (int): The resulting "parent" node id of a merge

    Returns:
        dict[int, list[list[int]]]: The updated conflict sets dictionary
    """
    conflicts: list[list[Any]] = []
    for child in [a, b]:
        if child in conflict_sets:
            # child is already a result of a merge
            for cs in conflict_sets[child]:
                # parent node is also conflicting with all children of child
                cs.append(c)
                conflicts.append(cs)
            # remove the child from the parent -> conflicts mapping
            del conflict_sets[child]
        else:
            # child is not a result of a merge
            conflicts.append([child, c])

    if len(conflicts) > 0:
        conflict_sets[c] = conflicts

    return conflict_sets


def nodes_from_fragments(
    fragments: np.ndarray,
    merge_history: np.ndarray,
    min_score: float = 0.0,
    max_score: float = 0.5,
    size_threshold: int | None = None,
) -> tuple[nx.DiGraph, list[tuple]]:
    """Compute the nodes of a candidate graph from a set of fragments and a
    merge history.
    Also defines two scores on each node:
        "cohesion": 1 - LS , where LS is the score of the last merge used to
            create this node, or 0 if the node is a fragment. (Higher is better)
        "adhesion": NS, where NS is the score of the next merge with this node
            as a child, or 1 if the node is never merged with anything else
            in the history. (Higher is better)

    Args:
        fragments (np.ndarray): An array of fragment labels to use to generate
            candidate detections.
        merge_history (np.ndarray): The merge history array with shape (4, N)
            where N is the number of merges. The columns are a, b, c, score.
            It must already be renumbered so all cs are unique, and must also
            be sorted from lowest to highest score.
        min_score (float, optional): Excludes candidates that are merged with a
            score lower than min_score from the graph. Defaults to 0.0.
        max_score (float, optional): Exclude candidates that are merged with a
            score higher than max_score from the graph. Defaults to 0.5.
        size_threshold (int, optional): Exclude candidates with area less than
            size_threshold pixels from the graph. Defaults to None.

    Returns:
        tuple[nx.DiGraph, list[tuple, ...]]: returns a networkx graph with all
        the nodes added, and a list of exclusion sets for nodes in the graph
        (nodes that cannot be selected together).
    """
    # create a dictionary from node_ids to last merge scores used to create the node
    last_scores = {}
    # create a dictionary from node_ids to next merge scores used to merge the node
    next_scores = {}

    fragments = fragments.copy()

    graph: nx.DiGraph | None = None
    conflict_sets = {}

    for merge in merge_history:
        a, b, c, score, tp = merge
        a = int(a)
        b = int(b)
        c = int(c)

        if score >= min_score and graph is None:
            # get the initial fragments we want to populate the cand graph with
            graph = nodes_from_segmentation(fragments, size_threshold=size_threshold)

        # merge the fragments and add to history
        fragments[fragments == a] = c
        fragments[fragments == b] = c
        last_scores[c] = score
        next_scores[a] = score
        next_scores[b] = score

        if score >= min_score and score < max_score:
            # add the new node to the graph
            new_seg_only = np.zeros_like(fragments)
            new_seg_only[fragments == c] = c
            node_graph = nodes_from_segmentation(
                new_seg_only, size_threshold=size_threshold
            )
            graph.add_nodes_from(node_graph.nodes(data=True))

            # add conflicting segs to conflict sets
            conflict_sets = compute_conflicts(conflict_sets, a, b, c)

    for node in graph.nodes():
        cohesion_score = 1 - last_scores.get(node, 0.0)
        adhesion_score = next_scores.get(node, 1.0)
        graph.nodes[node]["cohesion"] = cohesion_score
        graph.nodes[node]["adhesion"] = adhesion_score

    exclusion_sets = []

    # filter elements below size threshold out of conflict sets
    for conflicts in conflict_sets.values():
        for conflict_set in conflicts:
            conflict_set = [node for node in conflict_set if node in graph.nodes]
            if len(conflict_set) > 1:
                exclusion_sets.append(conflict_set)

    return graph, exclusion_sets
