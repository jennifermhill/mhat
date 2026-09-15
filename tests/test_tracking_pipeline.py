"""End-to-end smoke test for the candidate-graph → ILP pipeline.

This is the contract that pins MHAT to its motile version. The repo subclasses
four motile extension points that are not documented API:

  - DivisionCost            relies on motile scanning __dict__ for Weight instances
  - EdgePairs               relies on Variable.__getitem__ returning an ilpy.Variable,
                            so that `pair * 2 + a * -1 + b * -1 <= 0` builds an
                            ilpy.Expression rather than a Python bool
  - CurvatureCost           an edge-pair cost keyed on the custom EdgePairs variable
  - LeavesScaledNodeSelection  overrides apply() while reusing NodeSelection.__init__

A single solve exercises all four. If a motile upgrade silently breaks one of
them, the ILP would keep running and quietly return the wrong answer — these
assertions are what turns that into a test failure.

The pipeline steps below mirror scripts/04_tracking/run_tracking.py.
"""

from __future__ import annotations

import motile
import networkx as nx
import numpy as np

from mhat.tracking import create_multihypo_graph, utils
from mhat.tracking.solve_with_motile import solve_with_motile


def build_candidate_graph(fragments, raw, merge_history, config, scale):
    """Mirror of run_tracking.py's graph construction, minus all the IO."""
    max_node_id = int(np.max(fragments))

    merge_history = create_multihypo_graph.normalize_costs(merge_history)
    merge_history = create_multihypo_graph.renumber_merge_history(
        merge_history, max_node_id
    )

    all_cand_graph = None
    all_exclusion_sets: list = []
    for timepoint in range(fragments.shape[0]):
        cand_graph, exclusion_sets = create_multihypo_graph.nodes_from_fragments(
            fragments[timepoint],
            merge_history[merge_history[:, 4] == timepoint],
            min_cost=config["min_merge_cost"],
            max_cost=config["max_merge_cost"],
            raw_img=raw[timepoint],
            size_threshold=config["size_threshold"],
            timepoint=timepoint,
            scale=scale,
        )
        if all_cand_graph is None:
            all_cand_graph = cand_graph
            all_exclusion_sets = exclusion_sets
        else:
            all_cand_graph = nx.compose(all_cand_graph, cand_graph)
            all_exclusion_sets.extend(exclusion_sets)

    utils.add_cand_edges(
        all_cand_graph,
        config["max_edge_distance"],
        max_children=config["max_children"],
    )
    all_cand_graph = utils.add_hyperedges(
        all_cand_graph,
        divisions=config["divisions"],
    )

    img_shape = fragments.shape
    img_shape_scaled = [img_shape[0]] + [
        int(img_shape[i] * scale[i]) for i in range(1, len(img_shape))
    ]
    utils.add_appear_ignore_attr(all_cand_graph)
    utils.add_disappear(all_cand_graph, img_shape_scaled)

    track_graph = motile.TrackGraph(all_cand_graph, frame_attribute="time")
    utils.add_drift_dist_attr(track_graph, drift=0)
    utils.add_area_diff_attr(track_graph)
    utils.add_intensity_diff_attr(track_graph)
    utils.add_division_attr(track_graph)

    return track_graph, all_exclusion_sets


def test_segmentation_run_without_agglomeration(
    synthetic_fragments, synthetic_raw, tracking_config, scale, tmp_path
):
    """A movie with no merge history at all goes through the normal path.

    This is the segmentation-run-with-merges-skipped case, and also what waterz
    emits for any frame whose objects never touch (cellpose produces no merges
    on 30 of the 48 Fluo-C2DL-MSC frames): `merge_history.csv` is header-only,
    so every frame's slice is empty. run_tracking.py does not branch for it --
    the helpers keep the array 2-D, normalize_costs is a no-op, and
    nodes_from_fragments falls back to the leaf fragments per frame. This used
    to raise "'NoneType' object has no attribute 'nodes'" because the graph was
    only ever seeded from inside the merge loop.
    """
    merge_csv = tmp_path / "merge_history.csv"
    merge_csv.write_text("a,b,c,cost,timepoint\n")

    merge_history = create_multihypo_graph.load_merge_history(merge_csv)
    assert merge_history.shape == (0, 5), "column indexing needs the second axis"

    # The three whole-movie steps run_tracking.py applies before the frame loop.
    merge_history = merge_history[merge_history[:, 4] < synthetic_fragments.shape[0]]
    merge_history = create_multihypo_graph.normalize_costs(merge_history)
    merge_history = create_multihypo_graph.renumber_merge_history(
        merge_history, int(np.max(synthetic_fragments))
    )

    for timepoint in range(synthetic_fragments.shape[0]):
        cand_graph, exclusion_sets = create_multihypo_graph.nodes_from_fragments(
            synthetic_fragments[timepoint],
            merge_history[merge_history[:, 4] == timepoint],
            min_cost=tracking_config["min_merge_cost"],
            max_cost=tracking_config["max_merge_cost"],
            raw_img=synthetic_raw[timepoint],
            size_threshold=tracking_config["size_threshold"],
            timepoint=timepoint,
            scale=scale,
        )
        assert cand_graph.number_of_nodes() > 0
        assert exclusion_sets == []
        for _, data in cand_graph.nodes(data=True):
            assert data["time"] == timepoint
            # The attributes the ILP's cohesion/adhesion costs read must exist
            # even here, or solving raises KeyError.
            assert data["cohesion"] == 1.0
            assert data["adhesion"] == 1.0
            assert data["num_leaves"] == 1


def test_candidate_graph_structure(
    synthetic_fragments, synthetic_raw, synthetic_merge_history, tracking_config,
    scale, ndim,
):
    """The graph carries the multi-hypothesis structure the ILP needs."""
    track_graph, exclusion_sets = build_candidate_graph(
        synthetic_fragments, synthetic_raw, synthetic_merge_history,
        tracking_config, scale,
    )

    real_nodes = [n for n, d in track_graph.nodes.items() if "time" in d]
    assert len(real_nodes) >= 4 * synthetic_fragments.shape[0], (
        "expected at least the four fragments per frame to survive size filtering"
    )

    # Every real node must carry the attributes the costs read, or the ILP would
    # silently price them as missing. The per-axis position scalars follow the
    # data's rank: a 2D node has y and x and must NOT have a z, since a phantom
    # z is exactly what makes a 2D geff declare an axis with no backing property.
    position_keys = ("z", "y", "x") if ndim == 3 else ("y", "x")
    for node in real_nodes:
        attrs = track_graph.nodes[node]
        for key in ("time", "centroid", "area", "intensity",
                    "cohesion", "adhesion", "num_leaves", *position_keys):
            assert key in attrs, f"node {node} missing {key!r}"
        assert len(attrs["centroid"]) == ndim
        if ndim == 2:
            assert "z" not in attrs, f"2D node {node} has a phantom z"

    # Agglomeration must produce parent/child conflicts, otherwise the
    # ExclusiveNodes constraint below is vacuous and the test proves nothing.
    assert exclusion_sets, "expected merge hypotheses to create exclusion sets"
    assert any(len(s) > 1 for s in exclusion_sets)

    # divisions=True must produce hyperedges, which is what makes EdgePairs and
    # CurvatureCost reachable at all. Note motile.TrackGraph absorbs the
    # hypernodes that add_hyperedges puts in the nx graph and re-expresses them
    # as ((u,), (v1, v2)) edge tuples, so they are not in .nodes any more.
    hyperedges = [e for e in track_graph.edges if track_graph.is_hyperedge(e)]
    assert hyperedges, "expected division hyperedges to be added"
    for us, vs in hyperedges:
        assert len(us) == 1 and len(vs) == 2, (
            f"division hyperedge {(us, vs)} is not the expected 1->2 shape"
        )


def test_edges_carry_all_cost_attributes(
    synthetic_fragments, synthetic_raw, synthetic_merge_history, tracking_config, scale
):
    """Each cost in solve_with_motile reads an edge attribute; all must exist."""
    track_graph, _ = build_candidate_graph(
        synthetic_fragments, synthetic_raw, synthetic_merge_history,
        tracking_config, scale,
    )

    assert track_graph.edges, "candidate graph has no edges"
    for edge in track_graph.edges:
        attrs = track_graph.edges[edge]
        for key in ("drift_dist", "area_diff", "intensity_diff", "is_division"):
            assert key in attrs, f"edge {edge} missing {key!r}"
        assert np.isfinite(attrs["drift_dist"])


def test_solve_produces_valid_tracks(
    synthetic_fragments, synthetic_raw, synthetic_merge_history, tracking_config, scale
):
    """The full solve: non-empty, respects exclusions, links across time.

    This is the assertion that actually exercises DivisionCost, EdgePairs,
    CurvatureCost and LeavesScaledNodeSelection together.
    """
    track_graph, exclusion_sets = build_candidate_graph(
        synthetic_fragments, synthetic_raw, synthetic_merge_history,
        tracking_config, scale,
    )

    solution = solve_with_motile(tracking_config, track_graph, exclusion_sets)

    assert solution.number_of_nodes() > 0, (
        "empty solution — the ILP found 'select nothing' optimal, which means the "
        "node costs are no longer being applied"
    )
    assert solution.number_of_edges() > 0, "solution has no links between frames"

    # The exclusion constraint must actually hold: a merged parent and either of
    # its children can never both be selected.
    selected = set(solution.nodes)
    for exclusion_set in exclusion_sets:
        both = selected.intersection(exclusion_set)
        assert len(both) <= 1, (
            f"ExclusiveNodes violated: {sorted(both)} all selected from {exclusion_set}"
        )

    # Every selected edge must move forward exactly one frame.
    for u, v in solution.edges:
        t_u = solution.nodes[u]["time"]
        t_v = solution.nodes[v]["time"]
        assert t_v == t_u + 1, f"edge {u}->{v} spans {t_u}->{t_v}, not one frame"

    # Tracking is only meaningful if the solution spans more than one timepoint.
    times = {d["time"] for _, d in solution.nodes(data=True)}
    assert len(times) > 1, f"solution confined to a single frame: {times}"


def test_solution_is_deterministic(
    synthetic_fragments, synthetic_raw, synthetic_merge_history, tracking_config, scale
):
    """Same inputs, same solution — guards against solver-dependent drift."""
    solutions = []
    for _ in range(2):
        track_graph, exclusion_sets = build_candidate_graph(
            synthetic_fragments.copy(), synthetic_raw,
            synthetic_merge_history.copy(), tracking_config, scale,
        )
        solutions.append(
            solve_with_motile(tracking_config, track_graph, exclusion_sets)
        )

    assert set(solutions[0].nodes) == set(solutions[1].nodes)
    assert set(solutions[0].edges) == set(solutions[1].edges)
