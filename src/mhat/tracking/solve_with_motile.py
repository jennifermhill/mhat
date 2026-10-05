import os
import warnings

import motile
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
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    solver = motile.Solver(graph)

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(1))

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

    if force_all:
        include_division = config.get("divisions", False)
    else:
        include_division = config.get("division_weight", 0) != 0
    if include_division:
        solver.add_cost(
            DivisionCost(weight=config.get("division_weight", 0.0)),
            name="division",
        )
    else:
        print("Skipping division cost")

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

    # Fit path only: if NO feature cost is active (the no-features "None" / "- All"
    # baseline), the only remaining terms are appear/disappear -- both positive --
    # so the ILP's optimum is the empty solution and the SSVM has nothing to fit
    # linking against. Add a learnable, feature-agnostic per-edge selection cost so
    # linking has a lever to learn (the fit analogue of the runtime
    # base_edge_constant). The attribute is 0 on every edge, so the learned weight
    # is inert and only the learned constant (-> base_edge_constant) matters.
    if force_all:
        feature_active = (
            _include("ablate_drift", "drift_weight", "drift_constant")
            or _include("ablate_area", "area_weight", "area_constant")
            or _include("ablate_intensity", "intensity_weight", "intensity_constant")
            or _include("ablate_curvature", "curvature_weight", "curvature_constant")
            or add_cohesion
            or add_adhesion
        )
        if not feature_active:
            for edge in solver.graph.edges:
                solver.graph.edges[edge]["base_edge"] = 0.0
            solver.add_cost(
                motile.costs.EdgeSelection(
                    weight=0.0, attribute="base_edge", constant=0.0
                ),
                name="base_edge",
            )
            print(
                "No feature costs active; added learnable base_edge cost so the "
                "SSVM fit does not collapse to an empty solution"
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

    num_threads = 16 if (os.cpu_count() or 1) >= 16 else 1
    report_solver_backend()
    solver.solve(num_threads=num_threads, verbose=True)
    solution_graph = to_nx_graph(solver.get_selected_subgraph())
    return solution_graph
