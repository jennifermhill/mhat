import os
import warnings

import motile
import numpy as np
from mhat.tracking.division_cost import DivisionCost
from mhat.tracking.edge_pairs import CurvatureCost
from mhat.tracking.leaves_scaled_costs import LeavesScaledNodeSelection
from mhat.tracking.utils import to_nx_graph, report_graph_statistics


def gurobi_license_status():
    """Say whether ilpy's default backend choice will pick Gurobi or SCIP.

    ``solver.solve()`` uses ``ilpy.Preference.Any``, which picks Gurobi only
    when a full license is available and otherwise falls back to SCIP without
    printing anything. The size-limited license bundled with the pip
    ``gurobipy`` wheel (``LicenseID == 0``) counts as no license. This
    mirrors that rule (``ilpy.solver_backends.create_solver_backend`` in
    ilpy 0.6) with gurobipy's public API, so the fallback can be reported
    before a long SCIP solve starts rather than discovered from its runtime.

    Returns:
        (uses_gurobi, reason): reason is None when a full license was found.
    """
    try:
        import gurobipy as gp
    except ImportError:
        return False, "gurobipy is not installed"
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            if int(env.getParam("LicenseID")) != 0:
                return True, None
    except gp.GurobiError as e:
        return False, f"no usable Gurobi license was found ({e})"
    return False, "only the size-limited license bundled with pip gurobipy was found"


def report_solver_backend():
    """Print which ILP backend will run, warning when ilpy falls back to SCIP
    although gurobipy is installed (the user evidently wanted Gurobi)."""
    uses_gurobi, reason = gurobi_license_status()
    if uses_gurobi:
        print("ILP solver: Gurobi (full license found)")
    elif reason == "gurobipy is not installed":
        print("ILP solver: SCIP (gurobipy is not installed)")
    else:
        warnings.warn(
            f"Gurobi is installed but {reason}, so ilpy is solving with SCIP. "
            "The solution is still optimal, but SCIP is much slower on real "
            "tracking problems, and equal-cost ties can break differently than "
            "under Gurobi. To use Gurobi, make a full license visible to "
            "gurobipy (GRB_LICENSE_FILE, ~/gurobi.lic, or `module load gurobi` "
            "on a cluster); academic licenses are free from gurobi.com.",
            stacklevel=2,
        )


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

    if config.get("division_weight", 0) != 0:
        solver.add_cost(
            DivisionCost(weight=config["division_weight"]),
            name="division",
        )
    else:
        print("Skipping division cost (weight=0)")

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
        LeavesScaledNodeSelection(
            weight=config["cohesion_weight"],
            attribute="cohesion",
            constant=config["cohesion_constant"],
        ),
        name="cohesion",
    )

    solver.add_cost(
        LeavesScaledNodeSelection(
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

    if config.get("verbose", False):
        report_graph_statistics(config, graph)

    solver.add_constraint(motile.constraints.ExclusiveNodes(exclusion_sets))

    num_threads = 16 if (os.cpu_count() or 1) >= 16 else 1
    report_solver_backend()
    solver.solve(num_threads=num_threads, verbose=True)
    solution_graph = to_nx_graph(solver.get_selected_subgraph())
    return solution_graph
