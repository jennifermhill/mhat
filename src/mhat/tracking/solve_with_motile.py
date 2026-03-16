import motile
from mhat.tracking.edge_pairs import CurvatureCost
from mhat.tracking.utils import to_nx_graph, scale_by_leaves


def solve_with_motile(config, graph, exclusion_sets):
    """Set up and solve the network flow problem.

    Args:
        graph (motile.TrackGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    solver = motile.Solver(graph)

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(1))

    if config["drift_weight"] != 0:
        solver.add_cost(
            motile.costs.EdgeSelection(
                weight=config["drift_weight"],
                attribute="drift_dist",
                constant=config["drift_constant"],
            ),
            name="drift",
        )
    else:
        print("Skipping drift cost (weight=0)")

    if config["area_weight"] != 0:
        solver.add_cost(
            motile.costs.EdgeSelection(
                weight=config["area_weight"],
                attribute="area_diff",
                constant=config["area_constant"],
            ),
            name="area",
        )
    else:
        print("Skipping area cost (weight=0)")

    if config["intensity_weight"] != 0:
        solver.add_cost(
            motile.costs.EdgeSelection(
                weight=config["intensity_weight"],
                attribute="intensity_diff",
                constant=config["intensity_constant"],
            ),
            name="intensity",
        )
    else:
        print("Skipping intensity cost (weight=0)")

    if config["curvature_weight"] != 0:
        solver.add_cost(
            CurvatureCost(
                weight=config["curvature_weight"],
                position_attribute="centroid",
                constant=config["curvature_constant"],
            ),
            name="curvature",
        )
    else:
        print("Skipping curvature cost (weight=0)")

    solver.add_cost(
        motile.costs.NodeSelection(
            weight=config["cohesion_weight"],
            attribute="cohesion",
            constant=config["cohesion_constant"],
        ),
        name="cohesion",
    )

    solver.add_cost(
        motile.costs.NodeSelection(
            weight=config["adhesion_weight"],
            attribute="adhesion",
            constant=config["adhesion_constant"],
        ),
        name="adhesion",
    )

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

    scale_by_leaves(solver)

    solver.add_constraint(motile.constraints.ExclusiveNodes(exclusion_sets))

    solver.solve()
    solution_graph = to_nx_graph(solver.get_selected_subgraph())
    return solution_graph
