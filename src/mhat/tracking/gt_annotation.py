"""Annotate a multi-hypothesis candidate graph with binary ground-truth labels
for SSVM weight fitting via motile.Solver.fit_weights.

The GT annotation rule (dense GT, e.g. CTC-style on Fluo-C3DL-MDA231):
- For each timepoint, match each GT label to the candidate node with highest IoU
  above a threshold, using Hungarian assignment (one-to-one). Matched candidates
  get gt_selected=1, all other candidates get gt_selected=0.
- For each candidate edge: gt_selected=1 iff both endpoints are matched AND the
  GT graph contains an edge between the matched cells; otherwise gt_selected=0.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_node_to_fragments(track_graph, merge_history) -> dict[int, list[int]]:
    """Map every candidate node ID to the list of leaf fragment IDs it owns.

    Mirrors the merge-walking logic in run_tracking.get_solution_seg, but
    over the full candidate graph rather than the solved subgraph.
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

    node_to_fragments: dict[int, list[int]] = {}
    for node in track_graph.nodes:
        if node in merge_dict:
            node_to_fragments[node] = merge_dict[node]
        else:
            node_to_fragments[node] = [int(node)]
    return node_to_fragments


def _candidate_mask(fragments_t: np.ndarray, fragment_ids: list[int]) -> np.ndarray:
    """Boolean mask in frame t covering all the given fragment IDs."""
    return np.isin(fragments_t, fragment_ids)


def _overlaps_for_candidate(
    cand_mask: np.ndarray,
    gt_seg_t: np.ndarray,
    gt_label_sizes: dict[int, int],
) -> dict[int, tuple[float, float]]:
    """For each overlapping GT label, return (iogt, iou) of cand_mask.

    IoGT = |cand ∩ gt| / |gt| (CTC detection criterion).
    IoU  = |cand ∩ gt| / |cand ∪ gt| (used as tiebreaker preferring tight fits).

    Filtering by IoGT > threshold and ranking by IoU avoids biasing the
    matching toward over-merged candidates (where IoGT stays high as the
    candidate grows but IoU drops).
    """
    cand_size = int(cand_mask.sum())
    if cand_size == 0:
        return {}
    overlapping = gt_seg_t[cand_mask]
    inter_counts = np.bincount(overlapping.ravel())
    out: dict[int, tuple[float, float]] = {}
    for label, gt_size in gt_label_sizes.items():
        if label < inter_counts.size:
            inter = int(inter_counts[label])
            if inter > 0 and gt_size > 0:
                iogt = inter / gt_size
                iou = inter / (cand_size + gt_size - inter)
                out[label] = (iogt, iou)
    return out


def annotate_gt_on_candidate_graph(
    track_graph,
    fragments: np.ndarray,
    gt_graph,
    gt_seg: np.ndarray,
    merge_history: np.ndarray,
    iogt_threshold: float = 0.5,
    gt_label_key: str = "track_id",
    frame_key: str = "time",
) -> dict:
    """Write `gt_selected` (1/0) on every candidate node and edge of `track_graph`.

    Args:
        track_graph: motile.TrackGraph (multi-hypothesis candidate graph).
        fragments: int label array of shape (T, ...). Per-frame slices give the
            fragment masks the candidate graph was built from.
        gt_graph: networkx DiGraph (the GT tracks) with `frame_key` and
            `gt_label_key` set on each node.
        gt_seg: int label array of shape (T, ...) where each pixel is the GT
            track_id at that pixel (already remapped via remap_seg_to_track_ids).
        merge_history: renumbered, normalized merge history array.
        iogt_threshold: minimum intersection-over-GT for a candidate↔GT match to
            count (CTC detection criterion). Defaults to 0.5.
        gt_label_key: attribute on GT graph nodes that holds the seg label.
            (Defaults to "track_id" — matches evaluate_tracking.py.)
        frame_key: attribute holding the timepoint (default "time").

    Returns:
        Dict of stats: {n_nodes_pos, n_nodes_neg, n_edges_pos, n_edges_neg,
        n_frames_with_unmatched_gt, n_total_gt_per_frame_unmatched}.
    """
    node_to_fragments = build_node_to_fragments(track_graph, merge_history)

    # Group candidate nodes by time
    nodes_by_time: dict[int, list] = {}
    for node, data in track_graph.nodes.items():
        t = data.get(frame_key)
        if t is None:
            continue  # hypernodes (none on MDA231 baseline) — leave gt_selected unset
        nodes_by_time.setdefault(int(t), []).append(node)

    # Build (time, track_id) -> gt_node index for edge lookup
    gt_node_by_tt: dict[tuple[int, int], int] = {}
    for gt_node, data in gt_graph.nodes(data=True):
        t = int(data[frame_key])
        L = int(data[gt_label_key])
        gt_node_by_tt[(t, L)] = gt_node

    # cand_node -> matched GT label (used for edge labeling)
    cand_to_gt_label: dict[int, int] = {}

    n_nodes_pos = 0
    n_nodes_neg = 0
    n_unmatched_gt_total = 0
    n_frames_with_unmatched = 0

    T = fragments.shape[0]
    for t in range(T):
        cand_nodes = nodes_by_time.get(t, [])
        if not cand_nodes:
            continue
        gt_seg_t = gt_seg[t] if t < gt_seg.shape[0] else None
        if gt_seg_t is None:
            # No GT for this frame — every candidate is gt_selected=0
            for n in cand_nodes:
                track_graph.nodes[n]["gt_selected"] = 0
            n_nodes_neg += len(cand_nodes)
            continue

        # GT label sizes in this frame
        gt_labels_arr, counts = np.unique(gt_seg_t, return_counts=True)
        gt_label_sizes: dict[int, int] = {
            int(L): int(c) for L, c in zip(gt_labels_arr, counts) if int(L) != 0
        }
        gt_labels = list(gt_label_sizes.keys())
        if not gt_labels:
            for n in cand_nodes:
                track_graph.nodes[n]["gt_selected"] = 0
            n_nodes_neg += len(cand_nodes)
            continue

        # Build IoGT and IoU matrices [num_candidates × num_gt_labels]
        # IoGT gates eligibility (CTC detection criterion). Hungarian ranks
        # by IoU among eligible to prefer the tightest-fitting candidate
        # rather than the largest over-merge.
        fragments_t = fragments[t]
        n_cand = len(cand_nodes)
        n_gt = len(gt_labels)
        gt_label_to_col = {L: i for i, L in enumerate(gt_labels)}
        iogt = np.zeros((n_cand, n_gt), dtype=np.float64)
        iou = np.zeros((n_cand, n_gt), dtype=np.float64)
        for row, cand in enumerate(cand_nodes):
            mask = _candidate_mask(fragments_t, node_to_fragments[cand])
            for L, (g_val, u_val) in _overlaps_for_candidate(
                mask, gt_seg_t, gt_label_sizes
            ).items():
                col = gt_label_to_col[L]
                iogt[row, col] = g_val
                iou[row, col] = u_val

        # Hungarian on IoU, but only over IoGT-eligible (cand, gt) pairs.
        # Mask out ineligible cells with a very low score so they're never picked.
        ranking = np.where(iogt >= iogt_threshold, iou, -1.0)
        row_idx, col_idx = linear_sum_assignment(-ranking)
        # Pre-filter by IoGT threshold, then greedy-resolve fragment conflicts.
        # Sort by IoU (highest first) to prefer the tightest-fitting candidate
        # when resolving overlapping-fragment conflicts.
        candidates_above_threshold = sorted(
            (
                (r, c, iou[r, c])
                for r, c in zip(row_idx, col_idx)
                if iogt[r, c] >= iogt_threshold
            ),
            key=lambda x: -x[2],
        )
        used_fragments: set[int] = set()
        matched_cand_rows: dict[int, int] = {}
        for r, c, _val in candidates_above_threshold:
            cand_frags = set(node_to_fragments[cand_nodes[r]])
            if cand_frags & used_fragments:
                continue
            matched_cand_rows[r] = c
            used_fragments.update(cand_frags)

        for row, cand in enumerate(cand_nodes):
            if row in matched_cand_rows:
                col = matched_cand_rows[row]
                track_graph.nodes[cand]["gt_selected"] = 1
                cand_to_gt_label[cand] = gt_labels[col]
                n_nodes_pos += 1
            else:
                track_graph.nodes[cand]["gt_selected"] = 0
                n_nodes_neg += 1

        # Track GT labels left unmatched after Hungarian + threshold + conflict resolution
        matched_cols = set(matched_cand_rows.values())
        unmatched_gt_count = n_gt - len(matched_cols)
        if unmatched_gt_count > 0:
            n_frames_with_unmatched += 1
            n_unmatched_gt_total += unmatched_gt_count

    # Edges
    n_edges_pos = 0
    n_edges_neg = 0
    n_edges_skipped = 0
    for edge, data in track_graph.edges.items():
        # Skip hyperedges (no time-ordered 2-tuple semantics for our labeling rule)
        if not isinstance(edge, tuple) or len(edge) != 2:
            n_edges_skipped += 1
            continue
        u, v = edge
        u_match = cand_to_gt_label.get(u)
        v_match = cand_to_gt_label.get(v)
        if u_match is None or v_match is None:
            data["gt_selected"] = 0
            n_edges_neg += 1
            continue
        t_u = int(track_graph.nodes[u][frame_key])
        t_v = int(track_graph.nodes[v][frame_key])
        gt_u = gt_node_by_tt.get((t_u, u_match))
        gt_v = gt_node_by_tt.get((t_v, v_match))
        if gt_u is not None and gt_v is not None and gt_graph.has_edge(gt_u, gt_v):
            data["gt_selected"] = 1
            n_edges_pos += 1
        else:
            data["gt_selected"] = 0
            n_edges_neg += 1

    return {
        "n_nodes_pos": n_nodes_pos,
        "n_nodes_neg": n_nodes_neg,
        "n_edges_pos": n_edges_pos,
        "n_edges_neg": n_edges_neg,
        "n_edges_skipped": n_edges_skipped,
        "n_frames_with_unmatched_gt": n_frames_with_unmatched,
        "n_unmatched_gt_total": n_unmatched_gt_total,
    }
