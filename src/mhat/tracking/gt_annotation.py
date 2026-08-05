"""Annotate a multi-hypothesis candidate graph with binary ground-truth labels
for SSVM weight fitting via motile.Solver.fit_weights.

The GT annotation rule (dense GT, e.g. CTC-style on Fluo-C3DL-MDA231):
- For each timepoint, match each GT label to the candidate node with highest IoU
  above a threshold, using Hungarian assignment (one-to-one). Matched candidates
  get gt_selected=1, all other candidates get gt_selected=0.
- For each candidate edge: gt_selected=1 iff both endpoints are matched AND the
  GT graph contains an edge between the matched cells; otherwise gt_selected=0.

The work is split in two so the expensive part can be reused:

- `compute_gt_overlaps` does the per-candidate mask/overlap sweep over the
  fragment volume. It depends only on the candidate graph and the *full* GT, so
  it can be computed once and reused across many GT subsets.
- `assign_gt_labels` does the Hungarian matching and writes `gt_selected`. It is
  cheap, and takes an optional `keep_gt_labels` restricting supervision to a
  subset of GT tracks (see the GT-amount experiment driven by gt_subsets.py).

`annotate_gt_on_candidate_graph` composes the two and is the unchanged entry
point for callers that just want the full-GT annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class FrameOverlaps:
    """Cached candidate↔GT overlap matrices for a single timepoint.

    `iogt` and `iou` are [len(cand_nodes) × len(gt_labels)]. Both are zero where
    the candidate and the GT label do not intersect at all, so `iou > 0` is an
    exact test for "shares at least one voxel".
    """

    cand_nodes: list
    gt_labels: list[int]
    iogt: np.ndarray
    iou: np.ndarray


@dataclass
class GtOverlaps:
    """Everything `assign_gt_labels` needs that is independent of the GT subset."""

    node_to_fragments: dict[int, list[int]]
    frames: dict[int, FrameOverlaps] = field(default_factory=dict)
    gt_node_by_tt: dict[tuple[int, int], int] = field(default_factory=dict)
    all_gt_labels: set[int] = field(default_factory=set)


def compute_gt_overlaps(
    track_graph,
    fragments: np.ndarray,
    gt_graph,
    gt_seg: np.ndarray,
    merge_history: np.ndarray,
    gt_label_key: str = "track_id",
    frame_key: str = "time",
) -> GtOverlaps:
    """Sweep the fragment volume once and cache all candidate↔GT overlaps.

    This is the expensive half of GT annotation (one boolean mask per candidate
    per frame). It depends only on `fragments` and the *full* `gt_seg`, so the
    result can be reused for any subset of GT tracks.
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

    overlaps = GtOverlaps(node_to_fragments=node_to_fragments, gt_node_by_tt=gt_node_by_tt)

    T = fragments.shape[0]
    for t in range(T):
        cand_nodes = nodes_by_time.get(t, [])
        if not cand_nodes:
            continue

        # No GT for this frame -> empty label set, so every candidate ends up
        # gt_selected=0 in assign_gt_labels (same as the pre-split behaviour).
        if t >= gt_seg.shape[0]:
            gt_label_sizes: dict[int, int] = {}
        else:
            gt_labels_arr, counts = np.unique(gt_seg[t], return_counts=True)
            gt_label_sizes = {
                int(L): int(c) for L, c in zip(gt_labels_arr, counts) if int(L) != 0
            }
        gt_labels = list(gt_label_sizes.keys())
        overlaps.all_gt_labels.update(gt_labels)

        # Build IoGT and IoU matrices [num_candidates × num_gt_labels]
        fragments_t = fragments[t]
        n_cand = len(cand_nodes)
        n_gt = len(gt_labels)
        gt_label_to_col = {L: i for i, L in enumerate(gt_labels)}
        iogt = np.zeros((n_cand, n_gt), dtype=np.float64)
        iou = np.zeros((n_cand, n_gt), dtype=np.float64)
        if n_gt:
            gt_seg_t = gt_seg[t]
            for row, cand in enumerate(cand_nodes):
                mask = _candidate_mask(fragments_t, node_to_fragments[cand])
                for L, (g_val, u_val) in _overlaps_for_candidate(
                    mask, gt_seg_t, gt_label_sizes
                ).items():
                    col = gt_label_to_col[L]
                    iogt[row, col] = g_val
                    iou[row, col] = u_val

        overlaps.frames[t] = FrameOverlaps(
            cand_nodes=cand_nodes, gt_labels=gt_labels, iogt=iogt, iou=iou
        )

    return overlaps


def assign_gt_labels(
    track_graph,
    overlaps: GtOverlaps,
    gt_graph,
    iogt_threshold: float = 0.5,
    keep_gt_labels: set[int] | None = None,
    overlap_criterion: str = "any",
    mask_untouched: bool = True,
    extra_masked_nodes: set | None = None,
    frame_key: str = "time",
) -> tuple[dict, set]:
    """Write `gt_selected` (1/0) on every candidate node and edge of `track_graph`.

    Args:
        track_graph: motile.TrackGraph the overlaps were computed against.
        overlaps: cached overlaps from `compute_gt_overlaps`.
        gt_graph: networkx DiGraph (the GT tracks).
        iogt_threshold: minimum intersection-over-GT for a candidate↔GT match to
            count (CTC detection criterion).
        keep_gt_labels: if given, only these GT track labels are treated as
            annotated, and every candidate touching none of them is returned in
            `masked_nodes` — to be dropped from the fit graph (arm A) or marked
            unlabeled (arm B). Deciding this needs only the annotations that
            exist; the hidden tracks are never consulted.

            **Changed 2026-08-03.** Previously the rule was "touches a *removed*
            track and no kept track", which left candidates overlapping no GT at
            all labeled 0. That was wrong for a subset: an annotator cannot tell
            background from an unannotated cell, so asserting "not selected"
            there is supervision they could not have supplied — and deciding it
            required reading the hidden tracks' extent. Results produced before
            this date (the `gta_*`, `gtar_*` and `gtae_*` sweeps) used the old
            rule and carry that small oracle: 74 of 2896 candidate nodes on
            NC281-sparse `02_nuclei_denoised_train`, constant across subset size.
        overlap_criterion: how "touches an annotated track" is decided. "any" =
            shares at least one voxel (the honest reading of "an annotator said
            nothing about this region"); "iogt" = passes `iogt_threshold`, i.e.
            only near-matches count.
        mask_untouched: whether "touches no annotated track" is itself grounds for
            masking. True (default) is the sparse-annotation protocol above. Set
            False for a **dense crop** (see gt_crops.py), where the annotation
            covers a whole region: there, a candidate inside the crop that matches
            nothing is a genuine false positive and `gt_selected = 0` is exactly
            what the annotator's work supports. Keeping those negatives is the
            entire difference between the two protocols, so this flag must follow
            the protocol — with False, the caller is responsible for supplying the
            region mask via `extra_masked_nodes` and for guaranteeing that no
            surviving candidate touches an unannotated track (gt_crops'
            `assert_no_unannotated_overlap` checks precisely that).
        extra_masked_nodes: candidates to mask regardless of what they touch —
            the crop arm's "this candidate is outside the annotated box", a
            decision made from geometry alone. Unioned with whatever
            `mask_untouched` contributes.
        frame_key: attribute holding the timepoint.

    Returns:
        (stats, masked_nodes). `masked_nodes` is empty when `keep_gt_labels` is
        None and no `extra_masked_nodes` are given, in which case this reproduces
        the original dense-GT annotation exactly.
    """
    if overlap_criterion not in ("any", "iogt"):
        raise ValueError(f"overlap_criterion must be 'any' or 'iogt', got {overlap_criterion!r}")

    cand_to_gt_label: dict[int, int] = {}
    masked_nodes: set = set()

    n_nodes_pos = 0
    n_nodes_neg = 0
    n_no_gt_overlap = 0
    n_unmatched_gt_total = 0
    n_frames_with_unmatched = 0

    for _t, fo in overlaps.frames.items():
        cand_nodes = fo.cand_nodes
        n_cand = len(cand_nodes)
        n_gt = len(fo.gt_labels)

        # Which GT labels each candidate touches, under the chosen criterion.
        touches = (fo.iou > 0.0) if overlap_criterion == "any" else (fo.iogt >= iogt_threshold)

        if keep_gt_labels is None:
            # Complete GT: nothing is unknown. A candidate touching no GT at all is
            # a genuine false positive and stays an informative negative.
            keep_cols = np.arange(n_gt)
            masked_rows = np.zeros(n_cand, dtype=bool)
        else:
            is_kept = np.array([L in keep_gt_labels for L in fo.gt_labels], dtype=bool)
            keep_cols = np.flatnonzero(is_kept)
            if mask_untouched:
                # Sparse annotation: a candidate is unlabeled iff it touches NO
                # annotated track. Note this asks only where the *kept* (annotated)
                # tracks are — never where the hidden ones are — so it is decidable
                # from the annotations alone.
                masked_rows = ~touches[:, keep_cols].any(axis=1)
            else:
                # Dense annotation over a region: the region mask arrives via
                # `extra_masked_nodes`, and everything else is genuinely labeled.
                masked_rows = np.zeros(n_cand, dtype=bool)

        if extra_masked_nodes:
            masked_rows = masked_rows | np.array(
                [cand in extra_masked_nodes for cand in cand_nodes], dtype=bool
            )

        # Diagnostic only (never used for supervision): candidates overlapping no GT
        # anywhere. Under a subset these are indistinguishable from unannotated cells.
        n_no_gt_overlap += int((~touches.any(axis=1)).sum())

        alive_rows = np.flatnonzero(~masked_rows)
        for r in np.flatnonzero(masked_rows):
            masked_nodes.add(cand_nodes[int(r)])

        # Hungarian on IoU, but only over IoGT-eligible (cand, gt) pairs.
        # Mask out ineligible cells with a very low score so they're never picked.
        matched_cand_rows: dict[int, int] = {}
        if keep_cols.size and alive_rows.size:
            sub_iogt = fo.iogt[np.ix_(alive_rows, keep_cols)]
            sub_iou = fo.iou[np.ix_(alive_rows, keep_cols)]
            ranking = np.where(sub_iogt >= iogt_threshold, sub_iou, -1.0)
            row_idx, col_idx = linear_sum_assignment(-ranking)
            # Pre-filter by IoGT threshold, then greedy-resolve fragment conflicts.
            # Sort by IoU (highest first) to prefer the tightest-fitting candidate
            # when resolving overlapping-fragment conflicts.
            candidates_above_threshold = sorted(
                (
                    (r, c, sub_iou[r, c])
                    for r, c in zip(row_idx, col_idx)
                    if sub_iogt[r, c] >= iogt_threshold
                ),
                key=lambda x: -x[2],
            )
            used_fragments: set[int] = set()
            for r, c, _val in candidates_above_threshold:
                row = int(alive_rows[r])
                cand_frags = set(overlaps.node_to_fragments[cand_nodes[row]])
                if cand_frags & used_fragments:
                    continue
                matched_cand_rows[row] = int(keep_cols[c])
                used_fragments.update(cand_frags)

        for row, cand in enumerate(cand_nodes):
            if row in matched_cand_rows:
                track_graph.nodes[cand]["gt_selected"] = 1
                cand_to_gt_label[cand] = fo.gt_labels[matched_cand_rows[row]]
                n_nodes_pos += 1
            else:
                track_graph.nodes[cand]["gt_selected"] = 0
                if not masked_rows[row]:
                    n_nodes_neg += 1

        # Track GT labels left unmatched after Hungarian + threshold + conflict resolution
        unmatched_gt_count = int(keep_cols.size) - len(set(matched_cand_rows.values()))
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
        gt_u = overlaps.gt_node_by_tt.get((t_u, u_match))
        gt_v = overlaps.gt_node_by_tt.get((t_v, v_match))
        if gt_u is not None and gt_v is not None and gt_graph.has_edge(gt_u, gt_v):
            data["gt_selected"] = 1
            n_edges_pos += 1
        else:
            data["gt_selected"] = 0
            n_edges_neg += 1

    stats = {
        "n_nodes_pos": n_nodes_pos,
        "n_nodes_neg": n_nodes_neg,
        "n_edges_pos": n_edges_pos,
        "n_edges_neg": n_edges_neg,
        "n_edges_skipped": n_edges_skipped,
        "n_frames_with_unmatched_gt": n_frames_with_unmatched,
        "n_unmatched_gt_total": n_unmatched_gt_total,
        "n_masked_nodes": len(masked_nodes),
        "n_kept_gt_labels": (
            len(overlaps.all_gt_labels) if keep_gt_labels is None else len(keep_gt_labels)
        ),
        "n_no_gt_overlap": n_no_gt_overlap,
    }
    return stats, masked_nodes


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
        n_edges_skipped, n_frames_with_unmatched_gt, n_unmatched_gt_total,
        n_masked_nodes, n_kept_gt_labels, n_no_gt_overlap}.
    """
    overlaps = compute_gt_overlaps(
        track_graph,
        fragments,
        gt_graph,
        gt_seg,
        merge_history,
        gt_label_key=gt_label_key,
        frame_key=frame_key,
    )
    stats, _masked = assign_gt_labels(
        track_graph,
        overlaps,
        gt_graph,
        iogt_threshold=iogt_threshold,
        frame_key=frame_key,
    )
    return stats
