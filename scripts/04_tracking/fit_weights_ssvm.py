"""Fit ILP weights via structured SVM (motile.Solver.fit_weights).

Reads the same TOML schema as run_tracking.py (plus `ssvm_reg`, `ssvm_max_iter`,
`iou_threshold`). Builds the multi-hypothesis candidate graph, annotates GT
labels by IoU with Hungarian assignment per frame, then fits weights using
structsvm's bundle method. Writes a learned-weights TOML and runs a final
solve with the learned weights so downstream evaluation can compare against
the hand-tuned baseline.

Note: GT must already be in geff format at <gt_data_dir>/correct_tracks.zarr.
Run evaluate_tracks.py once on any prior tracking output to trigger the
CTC→geff conversion if not yet present.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import motile
import toml

from mhat.tracking import utils
from mhat.tracking.gt_annotation import annotate_gt_on_candidate_graph
from mhat.tracking.pipeline import (  # noqa: F401 -- load_gt is re-exported for the sweep scripts
    build_track_graph,
    load_gt,
    resolve_input_dirs,
    write_tracking_outputs,
)
from mhat.tracking.solve_with_motile import add_costs, report_solver_backend
from mhat.tracking.utils import report_graph_statistics


def configure_logging(output_dir):
    """Surface structsvm bundle-method convergence output to a logfile only.

    Attaches the handler directly rather than via ``logging.basicConfig``, which
    is a no-op once the root logger has handlers — a driver that fits many
    subsets in one process would otherwise funnel every run into the first
    run's logfile.
    """
    log_dir = output_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"fit_weights_ssvm_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s: %(message)s"))
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)
    logging.getLogger("structsvm").setLevel(logging.INFO)
    return log_path


# `load_gt` lives in mhat.tracking.pipeline (rank-agnostic: it builds the
# funtracks name map from the GT store's own axes) and is imported above; the
# sweep scripts still import it from here.


# Map (cost_name, weight_attr_name) tuples in solver.weights to the TOML keys
# used in tracking_config.toml. Cost class names without explicit `name=` in
# add_cost are the class names themselves ("Appear", "Disappear").
_LEARNED_WEIGHT_TO_TOML = {
    ("drift", "weight"): "drift_weight",
    ("drift", "constant"): "drift_constant",
    ("area", "weight"): "area_weight",
    ("area", "constant"): "area_constant",
    ("intensity", "weight"): "intensity_weight",
    ("intensity", "constant"): "intensity_constant",
    ("curvature", "weight"): "curvature_weight",
    ("curvature", "constant"): "curvature_constant",
    ("cohesion", "weight"): "cohesion_weight",
    ("cohesion", "constant"): "cohesion_constant",
    ("adhesion", "weight"): "adhesion_weight",
    ("adhesion", "constant"): "adhesion_constant",
    # DivisionCost exposes a weight only, so there is no division constant to map.
    ("division", "weight"): "division_weight",
    ("Appear", "constant"): "appear_constant",
    ("Disappear", "constant"): "disappear_constant",
    # Only the no-features baseline adds this (see add_costs). Its learned constant
    # transfers to the runtime base_edge (weight=0 there too, so it stays constant-
    # only); the inert weight on the all-zero attribute is intentionally not mapped.
    ("base_edge", "constant"): "base_edge_constant",
}


def build_learned_config(input_config, solver):
    """Copy of the input config with weight/constant fields replaced by learned values."""
    learned_config = dict(input_config)
    for (cost_name, var_name), weight in solver.weights._weights_by_name.items():
        toml_key = _LEARNED_WEIGHT_TO_TOML.get((cost_name, var_name))
        if toml_key is not None:
            learned_config[toml_key] = float(weight.value)
    return learned_config


def write_learned_config(input_config, solver, learned_path):
    """Write a tracking config with weight/constant fields replaced by learned values."""
    learned_config = build_learned_config(input_config, solver)
    with open(learned_path, "w") as f:
        toml.dump(learned_config, f)
    return learned_config


def fit_and_solve(config, raw_dir, seg_dir, flow_dirs, gt_data_dir, output_dir):
    configure_logging(output_dir)

    # Persist input config (pre-fit) for reproducibility
    with open(output_dir / "config.toml", "w") as f:
        toml.dump(config, f)

    track_graph, fragments, merge_history, exclusion_sets, scale, axes = build_track_graph(
        config, raw_dir, seg_dir, flow_dirs
    )

    # Mirror run_tracking.py: a segmentation with no merge history (e.g. cellpose
    # in skip-merges mode) is fit in no-merge mode, where cohesion/adhesion node
    # attributes don't exist and must be skipped (see add_costs).
    no_merges = len(merge_history) == 0
    if no_merges:
        print("No merge history found. Fitting in no-merge (fragments-only) mode; "
              "cohesion/adhesion costs will be skipped.")

    print("Loading GT...")
    gt_graph, gt_seg = load_gt(gt_data_dir, scale)

    print("Annotating GT on candidate graph...")
    stats = annotate_gt_on_candidate_graph(
        track_graph,
        fragments,
        gt_graph,
        gt_seg,
        merge_history,
        iogt_threshold=config.get("iogt_threshold", 0.5),
    )
    print("GT annotation stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return fit_and_solve_on_graph(
        config,
        track_graph,
        exclusion_sets,
        fragments,
        merge_history,
        scale,
        axes,
        output_dir,
        no_merges=no_merges,
    )


def fit_and_solve_on_graph(
    config,
    fit_graph,
    fit_exclusion_sets,
    fragments,
    merge_history,
    scale,
    axes,
    output_dir,
    no_merges=False,
    solve_graph=None,
    solve_exclusion_sets=None,
):
    """Fit weights on `fit_graph`, then solve and write the tracking outputs.

    `fit_graph` must already carry `gt_selected` on its nodes and edges.

    `solve_graph` lets the final solve run on a different (larger) graph than the
    fit — used by the GT-amount experiment, where arm A fits on a graph reduced
    to the annotated region but still wants predictions over the whole field of
    view, so the outputs stay comparable across subset sizes. Defaults to
    `fit_graph`.

    Returns a summary dict describing the fit (also useful as fit_summary.json).
    """
    print("\nBuilding solver and adding constraints/costs...")
    solver = motile.Solver(fit_graph)
    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(1))
    # force_all=True: weights/constants start at 0 here, so the runtime
    # "0 weight + 0 constant = ablated" rule would skip every cost and leave
    # nothing to fit. Add every feature cost except those excluded via ablate_*.
    add_costs(solver, config, force_all=True, no_merges=no_merges)
    # ExclusiveNodes must be added before fit_weights — the loss-augmented ILP
    # in SoftMarginLoss copies solver.constraints at construction time.
    solver.add_constraint(motile.constraints.ExclusiveNodes(fit_exclusion_sets))

    report_graph_statistics(config, fit_graph)

    # The fit's loss-augmented ILPs and the final solve below all use ilpy's
    # Preference.Any, so this one report covers every solve in this function.
    report_solver_backend()
    print("\nFitting weights via SSVM (this may take a while)...")
    # ssvm_standardize / ssvm_hamming_weight selected two exploratory fitting
    # variants, removed 2026-10-08 (see ssvm_results.md). Refuse them rather than
    # silently running the stock fit in their place.
    removed = [k for k in ("ssvm_standardize", "ssvm_hamming_weight") if config.get(k)]
    if removed:
        raise ValueError(
            f"{removed} selected fitting variants that were removed on 2026-10-08 "
            f"(fit_weights_standardized / fit_weights_hamming_weighted). Recover "
            f"them from git history to rerun those experiments."
        )
    # Use stock motile fit_weights — ε converges from above to ≈0 with the
    # post-2026-05-15 ilpy. No post-hoc adjustments needed; see CLAUDE.md.
    solver.fit_weights(
        gt_attribute="gt_selected",
        regularizer_weight=config.get("ssvm_reg", 0.1),
        max_iterations=config.get("ssvm_max_iter", 100),
        eps=config.get("ssvm_eps", 1e-6),
    )

    print("\nLearned weights:")
    print(solver.weights)

    learned_path = output_dir / "learned_weights.toml"
    learned_config = write_learned_config(config, solver, learned_path)
    print(f"Wrote learned weights to {learned_path}")

    print("\nSolving with learned weights...")
    if solve_graph is not None and solve_graph is not fit_graph:
        # Arm A: fit on the reduced graph, predict on the full one. Rebuild the
        # same cost set (same ablate_* flags via force_all) with the learned
        # values, so the only difference from the fit solver is the graph.
        solve_solver = motile.Solver(solve_graph)
        solve_solver.add_constraint(motile.constraints.MaxParents(1))
        solve_solver.add_constraint(motile.constraints.MaxChildren(1))
        add_costs(solve_solver, learned_config, force_all=True, no_merges=no_merges)
        solve_solver.add_constraint(
            motile.constraints.ExclusiveNodes(
                fit_exclusion_sets if solve_exclusion_sets is None else solve_exclusion_sets
            )
        )
        solve_solver.solve()
        solution_graph = utils.to_nx_graph(solve_solver.get_selected_subgraph())
    else:
        solver.solve()
        solution_graph = utils.to_nx_graph(solver.get_selected_subgraph())

    # `gt_selected` is a fitting label, not a prediction. Drop it before writing:
    # it is meaningless in the output, and for unlabeled candidates it is None,
    # which geff cannot serialize (it infers a dtype from the first value).
    for _, node_data in solution_graph.nodes(data=True):
        node_data.pop("gt_selected", None)
    for _, _, edge_data in solution_graph.edges(data=True):
        edge_data.pop("gt_selected", None)

    n_solution_nodes = solution_graph.number_of_nodes()
    if n_solution_nodes == 0:
        # Degenerate fit: nothing selected. Downstream geff/traccuracy cannot
        # handle an empty prediction, so bail out before writing outputs.
        print("WARNING: learned weights give an EMPTY solution; skipping output write.")
        return {
            "empty_solution": True,
            "n_fit_nodes": len(fit_graph.nodes),
            "n_fit_edges": len(fit_graph.edges),
            "n_solution_nodes": 0,
            "n_solution_edges": 0,
            "learned_weights": {
                k: learned_config[k]
                for k in _LEARNED_WEIGHT_TO_TOML.values()
                if k in learned_config
            },
        }

    print("Saving results...")
    # Same writer as run_tracking.py: node-id-labeled pred_seg.zarr and a geff
    # whose axis names come from the data's own metadata (2D or 3D).
    write_tracking_outputs(solution_graph, fragments, merge_history, scale, axes, output_dir)

    return {
        "empty_solution": False,
        "n_fit_nodes": len(fit_graph.nodes),
        "n_fit_edges": len(fit_graph.edges),
        "n_solution_nodes": n_solution_nodes,
        "n_solution_edges": solution_graph.number_of_edges(),
        "learned_weights": {
            k: learned_config[k] for k in _LEARNED_WEIGHT_TO_TOML.values() if k in learned_config
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    output_base_dir = Path(config["output_base_dir"])
    dataset = config["dataset"]
    experiment = config["experiment"]
    assert output_base_dir.is_dir()

    raw_dir, seg_dir, flow_dirs, gt_data_dir = resolve_input_dirs(config)

    if not config.get("exp_uid"):
        config["exp_uid"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = output_base_dir / "tracking" / experiment / dataset / config.get("output_name", "ssvm_fit")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving fit results to {output_dir}")

    fit_and_solve(config, raw_dir, seg_dir, flow_dirs, gt_data_dir, output_dir)
