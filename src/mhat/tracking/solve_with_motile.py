import numpy as np
import motile
from mhat.tracking.edge_pairs import CurvatureCost
from mhat.tracking.leaves_scaled_costs import LeavesScaledNodeSelection
from mhat.tracking.utils import to_nx_graph


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

    # Compute curvature values from edge pairs
    curvature_values = []
    for node in track_graph.nodes:
        in_edges = list(track_graph.prev_edges[node])
        out_edges = list(track_graph.next_edges[node])
        for in_edge in in_edges:
            in_offset = np.array(track_graph.nodes[in_edge[1]]["centroid"]) - np.array(track_graph.nodes[in_edge[0]]["centroid"])
            for out_edge in out_edges:
                out_offset = np.array(track_graph.nodes[out_edge[1]]["centroid"]) - np.array(track_graph.nodes[out_edge[0]]["centroid"])
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


def add_costs(solver, config, force_all=False, no_merges=False):
    """Add ILP cost terms to the solver.

    Two inclusion regimes, selected by `force_all`:

    - Runtime (`force_all=False`, the default): a cost is ablated by setting BOTH
      its weight and constant to 0. A cost whose weight and constant are both 0 is
      not added to the solver at all, so it adds no variables/constraints and
      cannot affect the solution (this avoids phantom zero-cost terms changing the
      solver's tie-breaking). This is the unified "0 weight + 0 constant = ablated"
      convention; the legacy `ablate_*` flags are ignored on this path.

      `base_edge_constant` (default 0) adds a constant-only per-edge selection cost
      (weight 0). It is used for the "- All" condition -- where every feature cost
      is zeroed -- to provide a tunable negative offset against appear/disappear so
      the ILP still selects a non-empty solution. It is added whenever nonzero.

    - Fit (`force_all=True`): used by SSVM weight fitting, where every weight and
      constant starts at 0. The 0/0 rule would skip every cost, leaving nothing to
      fit, so instead each feature cost is added unconditionally EXCEPT those
      explicitly excluded via their `ablate_*` flag (`ablate_drift`, `ablate_area`,
      `ablate_intensity`, `ablate_curvature`, `ablate_cohesion_adhesion`). The
      flags exist here only to drop features from a fit for cost/perf reasons
      (e.g. curvature dominates solve time on NC281-sparse), not as a runtime
      ablation mechanism. `base_edge_constant` is a fixed (non-learnable) inference
      offset and is never added on this path.

    Appear/disappear costs are always added on both paths.

    `no_merges=True` force-skips cohesion/adhesion on every path: in no-merge mode
    the segmentation has no merge hierarchy, so those node attributes are never
    computed and referencing them would fail.
    """
    def _include(ablate_key, weight_key, const_key):
        """Decide whether to add a feature cost under the active regime."""
        if force_all:
            return not config.get(ablate_key, False)
        return config.get(weight_key, 0) != 0 or config.get(const_key, 0) != 0

    if not force_all:
        base_edge_constant = config.get("base_edge_constant", 0.0)
        if base_edge_constant != 0.0:
            # Constant-only per-edge selection incentive for the "- All" condition.
            solver.add_cost(
                motile.costs.EdgeSelection(
                    weight=0.0,
                    attribute="drift_dist",
                    constant=base_edge_constant,
                ),
                name="base_edge",
            )

    if _include("ablate_drift", "drift_weight", "drift_constant"):
        solver.add_cost(
            motile.costs.EdgeSelection(
                weight=config["drift_weight"],
                attribute="drift_dist",
                constant=config["drift_constant"],
            ),
            name="drift",
        )
    else:
        print("Skipping drift cost")

    if _include("ablate_area", "area_weight", "area_constant"):
        solver.add_cost(
            motile.costs.EdgeSelection(
                weight=config["area_weight"],
                attribute="area_diff",
                constant=config["area_constant"],
            ),
            name="area",
        )
    else:
        print("Skipping area cost")

    if _include("ablate_intensity", "intensity_weight", "intensity_constant"):
        solver.add_cost(
            motile.costs.EdgeSelection(
                weight=config["intensity_weight"],
                attribute="intensity_diff",
                constant=config["intensity_constant"],
            ),
            name="intensity",
        )
    else:
        print("Skipping intensity cost")

    if _include("ablate_curvature", "curvature_weight", "curvature_constant"):
        solver.add_cost(
            CurvatureCost(
                weight=config["curvature_weight"],
                position_attribute="centroid",
                constant=config["curvature_constant"],
            ),
            name="curvature",
        )
    else:
        print("Skipping curvature cost")

    # cohesion/adhesion share a single fit-time exclusion flag, but on the runtime
    # path each is included independently by its own weight/constant. In no-merge
    # mode the segmentation has no merge hierarchy, so the cohesion/adhesion node
    # attributes are never computed -- skip both regardless of weights/flags.
    if no_merges:
        add_cohesion = add_adhesion = False
        print("Skipping cohesion/adhesion costs (no-merge mode)")
    elif force_all:
        add_cohesion = add_adhesion = not config.get("ablate_cohesion_adhesion", False)
    else:
        add_cohesion = config.get("cohesion_weight", 0) != 0 or config.get("cohesion_constant", 0) != 0
        add_adhesion = config.get("adhesion_weight", 0) != 0 or config.get("adhesion_constant", 0) != 0

    if add_cohesion:
        solver.add_cost(
            LeavesScaledNodeSelection(
                weight=config["cohesion_weight"],
                attribute="cohesion",
                constant=config["cohesion_constant"],
            ),
            name="cohesion",
        )
    else:
        print("Skipping cohesion cost")

    if add_adhesion:
        solver.add_cost(
            LeavesScaledNodeSelection(
                weight=config["adhesion_weight"],
                attribute="adhesion",
                constant=config["adhesion_constant"],
            ),
            name="adhesion",
        )
    else:
        print("Skipping adhesion cost")

    solver.add_cost(
        motile.costs.Appear(
            constant=config["appear_constant"], ignore_attribute="ignore_appear"
        )
    )
    solver.add_cost(
        motile.costs.Disappear(
            constant=config["disappear_constant"], ignore_attribute="ignore_disappear"
        )
    )


def solve_with_motile(config, graph, exclusion_sets, no_merges=False):
    """Set up and solve the network flow problem.

    Args:
        graph (motile.TrackGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    solver = motile.Solver(graph)

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(1))

    add_costs(solver, config, no_merges=no_merges)

    report_graph_statistics(config, graph)

    if config.get("stats_only", False):
        print("stats_only=true; exiting before ILP solve.")
        return None

    solver.add_constraint(motile.constraints.ExclusiveNodes(exclusion_sets))

    solver.solve()
    solution_graph = to_nx_graph(solver.get_selected_subgraph())
    return solution_graph
