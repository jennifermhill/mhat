"""The candidate-graph -> ILP contract that pins MHAT to its motile version.

The repo subclasses four motile extension points that are not documented API:

  - DivisionCost            relies on motile scanning __dict__ for Weight instances
  - EdgePairs               relies on Variable.__getitem__ returning an ilpy.Variable,
                            so that `pair * 2 + a * -1 + b * -1 <= 0` builds an
                            ilpy.Expression rather than a Python bool
  - CurvatureCost           an edge-pair cost keyed on the custom EdgePairs variable
  - LeavesScaledNodeSelection  overrides apply() while reusing NodeSelection.__init__

A single solve exercises all four. If a motile upgrade silently broke one of
them, the ILP would keep running and quietly return a plausible wrong answer --
an empty solution, or one that selects a merged parent together with its
children. The assertions below are what turns that into a failure. Loud
breakages (missing attributes, shape errors) are deliberately not tested; see
tests/TESTING_POLICY.md.

The graph construction mirrors scripts/04_tracking/run_tracking.py.
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


def test_solve_produces_valid_tracks(
    synthetic_fragments, synthetic_raw, synthetic_merge_history, tracking_config, scale
):
    """The full solve: non-empty, respects exclusions, links across time.

    This is the assertion that actually exercises DivisionCost, EdgePairs,
    CurvatureCost and LeavesScaledNodeSelection together.
    """
    track_graph, exclusion_sets = build_candidate_graph(
        synthetic_fragments,
        synthetic_raw,
        synthetic_merge_history,
        tracking_config,
        scale,
    )

    # Preconditions that keep the assertions below from being vacuous: without
    # parent/child conflicts ExclusiveNodes has nothing to constrain, and
    # without division hyperedges EdgePairs and CurvatureCost are never built.
    # motile.TrackGraph absorbs the hypernodes add_hyperedges puts in the nx
    # graph and re-expresses them as ((u,), (v1, v2)) edge tuples.
    assert any(len(s) > 1 for s in exclusion_sets), (
        "expected merge hypotheses to create exclusion sets"
    )
    assert any(track_graph.is_hyperedge(e) for e in track_graph.edges), (
        "expected division hyperedges to be added"
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
