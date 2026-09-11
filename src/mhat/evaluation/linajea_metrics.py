"""Reimplementation of the linajea error metrics, for apples-to-apples comparison.

This reproduces the metric reported in Malin-Mayor et al. 2023 ("Automated
reconstruction of whole-embryo cell lineages by learning from sparse
annotations"), so that MHAT results, linajea's published predictions, and the
TGMM baseline can all be scored by the same code on the same crop.

The metric counts four error types and normalizes each by the number of
ground truth edges:

    FN   false negative edges
    IS   identity switches
    FP-D false positive divisions
    FN-D false negative divisions
    sum  = (FN + IS + FP-D + FN-D) / gt_edges

False positive *edges* are deliberately excluded from ``sum``: with sparse
ground truth a predicted edge between two unannotated cells cannot be
contradicted. ``fp_edges`` is still reported, but as linajea does it — the
number of predicted edges that were close enough to a GT edge to be a
candidate match yet went unmatched ("unselected potential matches") — which is
a proxy, not a true false positive count.

How divisions survive sparse ground truth
-----------------------------------------
The annotation is sparse in *which cells* are annotated, but complete within
each annotated lineage. So a division is only scored when it can be anchored
to annotated ground truth: a predicted division whose child edges match no GT
edge at all is skipped entirely (neither TP nor FP). This mirrors
``Evaluator.get_fp_divisions`` in linajea v1.3 with ``sparse=True``.

Reference implementation
------------------------
https://github.com/funkelab/linajea/tree/v1.3 —
``linajea/evaluation/{match,evaluator}.py``. Two deliberate departures:

* Matching is solved with ``scipy.optimize.linear_sum_assignment`` rather than
  an ILP via ``pylp``. Both solve the same assignment problem, so the optimum
  is identical (ties may break differently).
* Edges here run **parent -> child** (forward in time), matching the rest of
  this repo. linajea stores them child -> parent, so its "in-degree 2" test
  for a division is an out-degree test here.

Units
-----
All coordinates and ``matching_threshold`` must be in the same units. This
repo stores geff positions in micrometers, so linajea's ``matching_threshold =
15`` (in its own world units of xy-pixels) is ``15 * 0.406 = 6.09`` um — see
``MATCHING_THRESHOLD_UM``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

# linajea used matching_threshold = 15 in world units of xy-pixels
# (setup211_simple_eval_side_*.toml); the DRO xy voxel is 0.406 um.
MATCHING_THRESHOLD_UM = 15 * 0.406


@dataclass
class LinajeaReport:
    """Error counts and per-GT-edge rates."""

    gt_edges: int = 0
    rec_edges: int = 0
    matched_edges: int = 0

    fn_edges: int = 0
    fp_edges: int = 0  # unselected potential matches (sparse proxy)
    identity_switches: int = 0
    fp_divisions: int = 0
    fn_divisions: int = 0

    gt_divisions: int = 0
    rec_divisions: int = 0
    tp_divisions: int = 0
    # FN-D broken out the way linajea does
    fn_divisions_no_connection: int = 0
    fn_divisions_one_unconnected_child: int = 0
    fn_divisions_unconnected_parent: int = 0
    # predicted divisions skipped because no adjacent edge matched GT
    skipped_divisions_unannotated: int = 0

    details: dict = field(default_factory=dict)

    @property
    def sum_errors(self) -> int:
        return (
            self.fn_edges
            + self.identity_switches
            + self.fp_divisions
            + self.fn_divisions
        )

    def normalized(self) -> dict[str, float]:
        """Every error count divided by the number of GT edges."""
        n = self.gt_edges
        if n == 0:
            raise ValueError("no ground truth edges to normalize by")
        return {
            "fn_edges": self.fn_edges / n,
            "identity_switches": self.identity_switches / n,
            "fp_divisions": self.fp_divisions / n,
            "fn_divisions": self.fn_divisions / n,
            "sum": self.sum_errors / n,
        }

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "details"}
        d["sum_errors"] = self.sum_errors
        d["normalized"] = self.normalized() if self.gt_edges else {}
        return d


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

DEFAULT_VOXEL_SIZE = (2.03, 0.406, 0.406)


def load_tracks_txt(
    path: str | Path,
    voxel_size: tuple[float, float, float] = DEFAULT_VOXEL_SIZE,
    z_anisotropy: float | None = None,
) -> nx.DiGraph:
    """Load a linajea-style track file into a parent -> child DiGraph.

    Handles both the ground truth files (7 tab-separated columns, no header)
    and the published prediction files (9 comma-separated columns with a
    header: ``t, z, y, x, cell_id, parent_id, track_id, node_score,
    edge_score``). Positions are converted to micrometers using the same
    convention as ``scripts/05_evaluation/linajea_to_geff.py``: the ``z``
    column is a world coordinate in xy-pixel units, so it is divided by the
    anisotropy before scaling.
    """
    path = Path(path)
    if z_anisotropy is None:
        z_anisotropy = voxel_size[0] / voxel_size[2]

    with open(path) as f:
        first = f.readline()
    has_header = "cell_id" in first
    delimiter = "," if first.count(",") >= 6 else None  # None => any whitespace

    # pandas rather than np.loadtxt: the published prediction files run to
    # millions of rows (drosophila_side_1 is 4.2M), where loadtxt is far too
    # slow.
    arr = pd.read_csv(
        path,
        sep=delimiter if delimiter is not None else r"\s+",
        header=0 if has_header else None,
        usecols=range(6),
        names=None if has_header else range(6),
        engine="c" if delimiter is not None else "python",
    ).to_numpy(dtype=float)

    graph = nx.DiGraph()
    for row in arr:
        graph.add_node(
            int(row[4]),
            t=int(row[0]),
            z=(row[1] / z_anisotropy) * voxel_size[0],
            y=row[2] * voxel_size[1],
            x=row[3] * voxel_size[2],
        )
    for row in arr:
        pid = int(row[5])
        if pid != -1 and pid in graph:
            graph.add_edge(pid, int(row[4]))  # parent -> child
    return graph


def load_geff_tracks(path: str | Path) -> nx.DiGraph:
    """Load an MHAT geff (``pred_tracks.zarr`` / ``correct_tracks.zarr``).

    Lets tracking output be scored by the same metric as the published linajea
    and TGMM text files. geff positions are already in micrometers, so no
    rescaling is applied (see ``funtracks.import_from_geff``, which likewise
    does not rescale).
    """
    import geff

    src, _ = geff.read(Path(path))
    graph = nx.DiGraph()
    for n, d in src.nodes(data=True):
        graph.add_node(
            n, t=int(d["time"]), z=float(d["z"]), y=float(d["y"]), x=float(d["x"])
        )
    graph.add_edges_from(src.edges())
    return graph


def crop_time(
    graph: nx.DiGraph, t_min: int, t_max: int, rebase: bool = False
) -> nx.DiGraph:
    """Return the induced subgraph on nodes with ``t_min <= t <= t_max``.

    Edges straddling a boundary disappear with their endpoint, so both GT and
    prediction must be cropped with the same bounds for the comparison to be
    fair.

    With ``rebase`` the kept frames are shifted so ``t_min`` becomes 0, which
    is what you want when comparing against tracking run on a cropped zarr
    (whose own frames start at 0).
    """
    keep = [n for n, d in graph.nodes(data=True) if t_min <= d["t"] <= t_max]
    out = graph.subgraph(keep).copy()
    if rebase:
        for _, d in out.nodes(data=True):
            d["t"] -= t_min
    return out


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------


def _pos(graph: nx.DiGraph, node) -> list[float]:
    d = graph.nodes[node]
    return [d["z"], d["y"], d["x"]]


def match_edges(
    gt: nx.DiGraph,
    rec: nx.DiGraph,
    matching_threshold: float,
) -> tuple[list[tuple], int]:
    """Match GT edges to predicted edges, one frame transition at a time.

    An edge pair is a candidate only when *both* endpoints lie within
    ``matching_threshold`` of their counterparts; the cost is the sum of the
    two endpoint distances. Each frame transition is solved independently,
    with both sides free to go unmatched.

    Returns the list of ``(gt_edge, rec_edge)`` matches and the number of
    predicted edges that were candidates but went unmatched.
    """
    gt_by_frame = _edges_by_child_frame(gt)
    rec_by_frame = _edges_by_child_frame(rec)
    rec_edge_of_child = {v: (u, v) for u, v in rec.edges()}
    # Index rec nodes by frame once: scanning all nodes per frame is O(T*N),
    # which is fatal on the 4.2M-node published predictions.
    rec_nodes_by_frame = _nodes_by_frame(rec)

    matches: list[tuple] = []
    unselected_potential = 0

    for t in sorted(gt_by_frame):
        gt_edges = gt_by_frame[t]
        rec_edges = rec_by_frame.get(t, [])
        if not gt_edges or not rec_edges:
            continue

        # Candidate rec nodes near each GT endpoint, per frame.
        near_parent = _neighbors(gt, rec, [e[0] for e in gt_edges],
                                 rec_nodes_by_frame.get(t - 1, []),
                                 matching_threshold)
        near_child = _neighbors(gt, rec, [e[1] for e in gt_edges],
                                rec_nodes_by_frame.get(t, []),
                                matching_threshold)

        costs: dict[tuple[int, int], float] = {}
        rec_index: dict[tuple, int] = {}
        for i, (gu, gv) in enumerate(gt_edges):
            parents = near_parent.get(gu, {})
            if not parents:
                continue
            for rv, dv in near_child.get(gv, {}).items():
                redge = rec_edge_of_child.get(rv)
                if redge is None:
                    continue
                ru = redge[0]
                du = parents.get(ru)
                if du is None:
                    continue
                j = rec_index.setdefault(redge, len(rec_index))
                costs[(i, j)] = du + dv

        if not costs:
            continue

        m_matches, n_unmatched = _solve_assignment(
            costs, len(gt_edges), len(rec_index), matching_threshold
        )
        index_to_rec = {j: e for e, j in rec_index.items()}
        for i, j in m_matches:
            matches.append((gt_edges[i], index_to_rec[j]))
        unselected_potential += n_unmatched

    return matches, unselected_potential


def _edges_by_child_frame(graph: nx.DiGraph) -> dict[int, list[tuple]]:
    out: dict[int, list[tuple]] = defaultdict(list)
    for u, v in graph.edges():
        out[graph.nodes[v]["t"]].append((u, v))
    return out


def _nodes_by_frame(graph: nx.DiGraph) -> dict[int, list]:
    out: dict[int, list] = defaultdict(list)
    for n, d in graph.nodes(data=True):
        out[d["t"]].append(n)
    return out


def _neighbors(gt, rec, gt_nodes, rec_nodes, threshold) -> dict:
    """For each GT node, the rec nodes within threshold and their distances."""
    if len(rec_nodes) == 0 or len(gt_nodes) == 0:
        return {}
    gt_nodes = list(dict.fromkeys(gt_nodes))
    rec_pos = np.array([_pos(rec, n) for n in rec_nodes], dtype=float)
    gt_pos = np.array([_pos(gt, n) for n in gt_nodes], dtype=float)
    tree = cKDTree(rec_pos)
    out: dict = {}
    for i, idxs in enumerate(tree.query_ball_point(gt_pos, threshold)):
        if not idxs:
            continue
        idxs = np.asarray(idxs)
        dists = np.linalg.norm(rec_pos[idxs] - gt_pos[i], axis=1)
        out[gt_nodes[i]] = {
            rec_nodes[j]: float(d) for j, d in zip(idxs, dists, strict=True)
        }
    return out


def _solve_assignment(costs, n_gt, n_rec, matching_threshold):
    """Assignment with both sides allowed to stay unmatched.

    ``no_match_cost`` follows linajea: ``2 * matching_threshold + 1``, i.e.
    just above the largest possible cost of a legal match, so a pair is only
    matched when doing so beats leaving both unmatched.
    """
    no_match_cost = 2 * matching_threshold + 1
    big = 10 * (no_match_cost + 1)

    size = n_gt + n_rec
    mat = np.full((size, size), big, dtype=float)
    for (i, j), c in costs.items():
        mat[i, j] = c
    for i in range(n_gt):
        mat[i, n_rec + i] = no_match_cost
    for j in range(n_rec):
        mat[n_gt + j, j] = no_match_cost
    mat[n_gt:, n_rec:] = 0.0

    rows, cols = linear_sum_assignment(mat)
    matches = [
        (i, j) for i, j in zip(rows, cols, strict=True)
        if i < n_gt and j < n_rec and mat[i, j] < big
    ]
    # rec edges that were candidates but went unmatched
    candidate_rec = {j for (_, j) in costs}
    matched_rec = {j for _, j in matches}
    return matches, len(candidate_rec - matched_rec)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def evaluate(
    gt: nx.DiGraph,
    rec: nx.DiGraph,
    matching_threshold: float = MATCHING_THRESHOLD_UM,
    sparse: bool = True,
) -> LinajeaReport:
    """Score ``rec`` against ``gt`` using the linajea error definitions."""
    edge_matches, unselected = match_edges(gt, rec, matching_threshold)

    gt_to_rec = dict(edge_matches)
    rec_to_gt = {r: g for g, r in edge_matches}

    rep = LinajeaReport()
    rep.gt_edges = gt.number_of_edges()
    rep.rec_edges = rec.number_of_edges()
    rep.matched_edges = len(edge_matches)
    rep.fn_edges = rep.gt_edges - rep.matched_edges
    rep.fp_edges = unselected if sparse else rep.rec_edges - rep.matched_edges

    gt_parents = [n for n, d in gt.out_degree() if d == 2]
    rec_parents = [n for n, d in rec.out_degree() if d == 2]
    rep.gt_divisions = len(gt_parents)
    rep.rec_divisions = len(rec_parents)

    _identity_switches(gt, gt_to_rec, rep)
    _fp_divisions(gt, rec, rec_parents, rec_to_gt, sparse, rep)
    _fn_divisions(gt, gt_parents, gt_to_rec, rep)
    return rep


def _identity_switches(gt, gt_to_rec, rep):
    """A GT cell whose incoming and outgoing edges map to different rec cells.

    Division nodes are skipped, as in linajea.
    """
    switches = []
    for cell in gt.nodes():
        succ = list(gt.successors(cell))
        pred = list(gt.predecessors(cell))
        if len(succ) != 1 or len(pred) != 1:
            continue  # ignore divisions and track ends
        prev_match = gt_to_rec.get((pred[0], cell))
        next_match = gt_to_rec.get((cell, succ[0]))
        if prev_match is None or next_match is None:
            continue
        # prev_match = (rec_parent, rec_cell), next_match = (rec_cell, rec_child)
        if prev_match[1] != next_match[0]:
            switches.append(cell)
    rep.identity_switches = len(switches)
    rep.details["identity_switch_nodes"] = switches


def _fp_divisions(gt, rec, rec_parents, rec_to_gt, sparse, rep):
    """Predicted divisions that the matched GT lineage does not make.

    With ``sparse=True`` a predicted division is only judged when it can be
    anchored to annotated GT; one sitting entirely in unannotated territory is
    skipped, counting as neither TP nor FP.
    """
    fp_nodes, skipped = [], []
    for parent in rec_parents:
        child_edges = [(parent, c) for c in rec.successors(parent)]
        matched = [rec_to_gt.get(e) for e in child_edges]
        # GT parent implied by each matched child edge
        gt_parents_seen = [m[0] if m is not None else None for m in matched]

        if all(m is None for m in matched) and sparse:
            preds = list(rec.predecessors(parent))
            if not preds:
                skipped.append(parent)
                continue
            prev_match = rec_to_gt.get((preds[0], parent))
            if prev_match is None:
                skipped.append(parent)
                continue
            # linajea also skips when the division falls after the end of the
            # matched GT track. Its check indexes a tuple where a node is
            # expected, so we implement the documented intent instead.
            if gt.out_degree(prev_match[1]) == 0:
                skipped.append(parent)
                continue

        if not _same_gt_parent(gt_parents_seen):
            fp_nodes.append(parent)

    rep.fp_divisions = len(fp_nodes)
    rep.skipped_divisions_unannotated = len(skipped)
    rep.details["fp_division_nodes"] = fp_nodes


def _same_gt_parent(seen) -> bool:
    """True when both child edges matched GT edges from the same GT parent."""
    if len(seen) != 2:
        return False
    a, b = seen
    return a is not None and b is not None and a == b


def _fn_divisions(gt, gt_parents, gt_to_rec, rep):
    """GT divisions the reconstruction failed to make, split by failure mode.

    Categories follow linajea: no connections; one unconnected child;
    unconnected parent.
    """
    no_conn, one_child, unconn_parent, tp = [], [], [], []
    for parent in gt_parents:
        children = list(gt.successors(parent))
        c1 = gt_to_rec.get((parent, children[0]))
        c2 = gt_to_rec.get((parent, children[1]))
        # rec node standing in for the GT parent, as seen from each child edge
        c1_s = c1[0] if c1 is not None else None
        c2_s = c2[0] if c2 is not None else None

        preds = list(gt.predecessors(parent))
        prev_match = gt_to_rec.get((preds[0], parent)) if preds else None
        prev_s = prev_match[1] if prev_match is not None else None

        children_joined = c1_s is not None and c1_s == c2_s

        if not preds:
            # no GT parent edge: only the two child edges are informative
            (tp if children_joined else no_conn).append(parent)
            continue

        if children_joined:
            if prev_s is not None and prev_s == c1_s:
                tp.append(parent)
            else:
                unconn_parent.append(parent)
        elif prev_s is not None and (prev_s == c1_s or prev_s == c2_s):
            one_child.append(parent)
        else:
            no_conn.append(parent)

    rep.fn_divisions_no_connection = len(no_conn)
    rep.fn_divisions_one_unconnected_child = len(one_child)
    rep.fn_divisions_unconnected_parent = len(unconn_parent)
    rep.fn_divisions = len(no_conn) + len(one_child) + len(unconn_parent)
    rep.tp_divisions = len(tp)
    rep.details["fn_division_nodes"] = {
        "no_connection": no_conn,
        "one_unconnected_child": one_child,
        "unconnected_parent": unconn_parent,
    }
