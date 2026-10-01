"""z-scored cost features (``normalize_features = "zscore"``), the negative control.

A sign slip or a mean/std mix-up in the weight/constant rewrite would still
solve and still score, just as a different (and wrongly labelled) control. This
pins the costs the ILP actually sees to hand-computed values of
``weight * (x - mean) / std``, num_leaves-scaled on nodes.
"""

from __future__ import annotations

import motile
import networkx as nx
import numpy as np
from motile.variables import EdgeSelected, NodeSelected

from mhat.tracking.solve_with_motile import add_costs
from mhat.tracking.utils import zscore_cost_params

S = np.sqrt(1.5)  # |z| of the outer two of three evenly spaced values


def test_zscored_costs_match_hand_computed_values():
    g = nx.DiGraph()
    # cohesion (0.2, 1.0, 0.6) -> z (-S, S, 0); adhesion (0.5, 1.0, 0.0) -> z (0, S, -S)
    g.add_node(1, time=0, cohesion=0.2, adhesion=0.5, num_leaves=1)
    g.add_node(2, time=1, cohesion=1.0, adhesion=1.0, num_leaves=2)
    g.add_node(3, time=1, cohesion=0.6, adhesion=0.0, num_leaves=1)
    # drift (2, 6) and area (0.1, 0.5) -> z (-1, 1) each
    g.add_edge(1, 2, drift_dist=2.0, area_diff=0.1, intensity_diff=0.0)
    g.add_edge(1, 3, drift_dist=6.0, area_diff=0.5, intensity_diff=0.0)
    track_graph = motile.TrackGraph(g, frame_attribute="time")

    config = {
        "drift_weight": 1.0, "drift_constant": 0.0,
        "area_weight": 1.0, "area_constant": 0.0,
        "intensity_weight": 0.0, "intensity_constant": 0.0,
        "curvature_weight": 0.0, "curvature_constant": 0.0,
        "cohesion_weight": -1.0, "cohesion_constant": 0.0,
        "adhesion_weight": -1.0, "adhesion_constant": 0.0,
        "appear_constant": 1.0, "disappear_constant": 1.0,
    }
    effective, _ = zscore_cost_params(config, track_graph)
    # Non-features pass through untouched.
    assert effective["appear_constant"] == 1.0
    assert effective["disappear_constant"] == 1.0

    solver = motile.Solver(track_graph)
    add_costs(solver, effective)
    costs = solver.costs
    edge_vars = solver.get_variables(EdgeSelected)
    node_vars = solver.get_variables(NodeSelected)

    # Edges: +1 * z_drift + 1 * z_area
    assert np.isclose(costs[edge_vars[(1, 2)]], -2.0)
    assert np.isclose(costs[edge_vars[(1, 3)]], 2.0)
    # Nodes: num_leaves * (-1 * z_cohesion - 1 * z_adhesion)
    assert np.isclose(costs[node_vars[1]], S)
    assert np.isclose(costs[node_vars[2]], 2 * (-S - S))
    assert np.isclose(costs[node_vars[3]], S)
