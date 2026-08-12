import motile
from mhat.tracking.division_cost import DivisionCost
from mhat.tracking.edge_pairs import CurvatureCost
from mhat.tracking.leaves_scaled_costs import LeavesScaledNodeSelection
from mhat.tracking.utils import to_nx_graph, report_graph_statistics


def add_costs(solver, config, no_merges=False):
    """Add all ILP cost terms to the solver.

    A cost is ablated by setting BOTH its weight and constant to 0: a cost whose
    weight and constant are both 0 is not added to the solver at all, so it adds
    no variables/constraints and cannot affect the solution (this avoids phantom
    zero-cost terms changing the solver's tie-breaking). Appear/disappear costs
    are always added.

    `base_edge_constant` (default 0) adds a constant-only per-edge selection cost
    (weight 0). It is used for the "- All" condition -- where every feature cost
    is zeroed -- to provide a tunable negative offset against appear/disappear so
    the ILP still selects a non-empty solution. It is added whenever nonzero.
    """
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

    if config.get("drift_weight", 0) != 0 or config.get("drift_constant", 0) != 0:
        solver.add_cost(
            motile.costs.EdgeSelection(
                weight=config["drift_weight"],
                attribute="drift_dist",
                constant=config["drift_constant"],
            ),
            name="drift",
        )
    else:
        print("Skipping drift cost (weight=0, constant=0)")

    if config.get("area_weight", 0) != 0 or config.get("area_constant", 0) != 0:
        solver.add_cost(
            motile.costs.EdgeSelection(
                weight=config["area_weight"],
                attribute="area_diff",
                constant=config["area_constant"],
            ),
            name="area",
        )
    else:
        print("Skipping area cost (weight=0, constant=0)")

    if config.get("intensity_weight", 0) != 0 or config.get("intensity_constant", 0) != 0:
        solver.add_cost(
            motile.costs.EdgeSelection(
                weight=config["intensity_weight"],
                attribute="intensity_diff",
                constant=config["intensity_constant"],
            ),
            name="intensity",
        )
    else:
        print("Skipping intensity cost (weight=0, constant=0)")

    if config.get("division_weight", 0) != 0:
        solver.add_cost(
            DivisionCost(weight=config["division_weight"]),
            name="division",
        )
    else:
        print("Skipping division cost (weight=0)")

    if config.get("curvature_weight", 0) != 0 or config.get("curvature_constant", 0) != 0:
        solver.add_cost(
            CurvatureCost(
                weight=config["curvature_weight"],
                position_attribute="centroid",
                constant=config["curvature_constant"],
            ),
            name="curvature",
        )
    else:
        print("Skipping curvature cost (weight=0, constant=0)")

    if no_merges:
        print("Skipping cohesion/adhesion costs (no-merge mode)")
    else:
        if config.get("cohesion_weight", 0) != 0 or config.get("cohesion_constant", 0) != 0:
            solver.add_cost(
                LeavesScaledNodeSelection(
                    weight=config["cohesion_weight"],
                    attribute="cohesion",
                    constant=config["cohesion_constant"],
                ),
                name="cohesion",
            )
        else:
            print("Skipping cohesion cost (weight=0, constant=0)")
        if config.get("adhesion_weight", 0) != 0 or config.get("adhesion_constant", 0) != 0:
            solver.add_cost(
                LeavesScaledNodeSelection(
                    weight=config["adhesion_weight"],
                    attribute="adhesion",
                    constant=config["adhesion_constant"],
                ),
                name="adhesion",
            )
        else:
            print("Skipping adhesion cost (weight=0, constant=0)")

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

    solver.add_constraint(motile.constraints.ExclusiveNodes(exclusion_sets))

    solver.solve()
    solution_graph = to_nx_graph(solver.get_selected_subgraph())
    return solution_graph
