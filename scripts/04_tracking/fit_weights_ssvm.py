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

import geff
import motile
import numpy as np
import structsvm as ssvm
import toml
import zarr

from funtracks.import_export import import_from_geff
from mhat.evaluation.evaluate_tracking import remap_seg_to_track_ids
from mhat.tracking import utils
from mhat.tracking.gt_annotation import annotate_gt_on_candidate_graph
from mhat.tracking.pipeline import build_track_graph
from mhat.tracking.solve_with_motile import add_costs, report_graph_statistics
from motile.variables import EdgeSelected, NodeSelected
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids


class TolerantBundleMethod(ssvm.BundleMethod):
    """UNUSED LEGACY (pre-ilpy-fix).

    BundleMethod whose convergence test treats only |ε| ≤ eps as converged.
    The base class exits on any ε ≤ eps (line 120 of structsvm/bundle_method.py),
    which includes large-magnitude negative ε. Before the 2026-05-15 ilpy update,
    a bug in ilpy's QP solver caused ε to converge to a non-zero negative fixed
    point — the base class would exit immediately on that as "convergence" when
    it was actually a spurious-cut signal. This subclass kept iterating in that
    case until max_iterations so stronger regularizers could be explored.

    With the fixed ilpy, ε converges monotonically from above to ≈0 and the
    base class exits correctly. This subclass is no longer wired into
    fit_and_solve; kept here for reference / in case the ilpy bug recurs.
    """

    def optimize(self, max_iterations=None):
        ssvm_logger = logging.getLogger("structsvm")
        w = np.zeros((self._dims,), dtype=np.float64)
        min_value = np.inf
        t = 0
        while max_iterations is None or t < max_iterations:
            t += 1
            ssvm_logger.info("----------------- iteration %d", t)
            w_tm1 = w
            L_w_tm1, a_t = self._value_gradient_callback(w_tm1)
            min_value = min(
                min_value, L_w_tm1 + 0.5 * self._lambda * np.dot(w_tm1, w_tm1)
            )
            b_t = L_w_tm1 - np.dot(w_tm1, a_t)
            self._add_hyperplane(a_t, b_t)
            w, min_lower = self._find_min_lower_bound()
            eps_t = min_value - min_lower
            ssvm_logger.info("          ε   is: %f", eps_t)
            if abs(eps_t) <= self._eps:
                ssvm_logger.info("converged (|ε| ≤ eps)")
                break
            if eps_t < 0:
                ssvm_logger.warning("ε < 0 (%f) — continuing; cut may be spurious", eps_t)
        return w


def fit_weights_standardized(
    solver,
    gt_attribute,
    regularizer_weight,
    max_iterations,
    eps,
):
    """UNUSED LEGACY (pre-ilpy-fix).

    Per-feature-standardized variant of motile.Solver.fit_weights. Each column
    of the feature matrix is divided by its std before being passed to
    structsvm's BundleMethod, then the returned weights are inverse-scaled so
    that the cost `features @ weights` is identical to what would have been
    computed without the rescaling. Mathematically a pure reconditioning of
    the QP — the function being optimized is unchanged.

    This was developed to mitigate an ill-conditioned QP that caused large-
    magnitude negative ε plateaus before the 2026-05-15 ilpy fix. Uses
    `TolerantBundleMethod` internally so it can keep iterating past spurious
    negative ε events. With the fixed ilpy, stock `solver.fit_weights()`
    converges directly and this helper is no longer wired into fit_and_solve;
    kept here for reference / in case the underlying conditioning issue
    resurfaces.

    Returns the optimal weights in the *original* (unscaled) space.
    """
    features = solver.features.to_ndarray()
    mask = np.zeros((solver.num_variables,), dtype=np.float32)
    ground_truth = np.zeros((solver.num_variables,), dtype=np.float32)

    for node, index in solver.get_variables(NodeSelected).items():
        gt = solver.graph.nodes[node].get(gt_attribute, None)
        if gt is not None:
            mask[index] = 1.0
            ground_truth[index] = gt
    for edge, index in solver.get_variables(EdgeSelected).items():
        gt = solver.graph.edges[edge].get(gt_attribute, None)
        if gt is not None:
            mask[index] = 1.0
            ground_truth[index] = gt

    feature_stds = features.std(axis=0)
    # Degenerate (all-zero) columns: leave scale at 1 — weight has no effect.
    safe_stds = np.where(feature_stds > 0, feature_stds, 1.0)
    features_scaled = features / safe_stds[np.newaxis, :]

    logger = logging.getLogger(__name__)
    weight_names = list(solver.weights._weights_by_name.keys())
    logger.info("Per-feature standardization:")
    for name, s in zip(weight_names, feature_stds):
        logger.info(f"  {str(name):<35} std={s:.4g}")

    loss = ssvm.SoftMarginLoss(
        solver.constraints,
        features_scaled.T,  # TODO: motile/ssvm.py:63 has the same transpose
        ground_truth,
        ssvm.HammingCosts(ground_truth, mask),
    )
    bundle = TolerantBundleMethod(
        loss.value_and_gradient,
        dims=features.shape[1],
        regularizer_weight=regularizer_weight,
        eps=eps,
    )
    w_scaled = bundle.optimize(max_iterations)
    w_original = w_scaled / safe_stds

    logger.info("Weights (scaled space → original space):")
    for name, ws, wo in zip(weight_names, w_scaled, w_original):
        logger.info(f"  {str(name):<35} scaled={ws:+.4g}   original={wo:+.4g}")

    return w_original


def configure_logging(output_dir):
    """Surface structsvm bundle-method convergence output to a logfile only."""
    log_dir = output_dir
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"fit_weights_ssvm_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler])
    logging.getLogger("structsvm").setLevel(logging.INFO)


def load_gt(gt_data_dir, scale):
    """Load CTC GT (already in geff format) and remap seg labels to track_ids."""
    gt_tracks_path = gt_data_dir / "correct_tracks.zarr"
    gt_seg_path = gt_data_dir / "correct_seg.zarr"
    if not gt_tracks_path.is_dir():
        raise FileNotFoundError(
            f"GT tracks not found at {gt_tracks_path}. Run evaluate_tracks.py "
            "once on a prior tracking result to trigger CTC→geff conversion."
        )
    name_map = {"time": "time", "x": "x", "y": "y", "z": "z", "id": "track_id"}
    gt_tracks = import_from_geff(
        gt_tracks_path,
        name_map,
        segmentation_path=gt_seg_path if gt_seg_path.is_dir() else None,
        scale=scale,
    )
    if gt_tracks.segmentation is None:
        raise RuntimeError("GT segmentation not found; cannot compute IoU matches.")
    gt_seg = remap_seg_to_track_ids(gt_tracks_path, gt_tracks.graph, gt_tracks.segmentation)
    return gt_tracks.graph, gt_seg


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
    ("Appear", "constant"): "appear_constant",
    ("Disappear", "constant"): "disappear_constant",
}


def write_learned_config(input_config, solver, learned_path):
    """Write a tracking config with weight/constant fields replaced by learned values."""
    learned_config = dict(input_config)
    for (cost_name, var_name), weight in solver.weights._weights_by_name.items():
        toml_key = _LEARNED_WEIGHT_TO_TOML.get((cost_name, var_name))
        if toml_key is not None:
            learned_config[toml_key] = float(weight.value)
    with open(learned_path, "w") as f:
        toml.dump(learned_config, f)


def get_solution_seg(fragments, merge_history, solution_graph):
    """Same logic as run_tracking.get_solution_seg — duplicated to avoid script-to-script imports."""
    merge_dict = {}
    for merge in merge_history:
        a, b, c, _cost, _tp = merge
        a, b, c = int(a), int(b), int(c)
        children = [a, b]
        if a in merge_dict:
            children.extend(merge_dict[a])
        if b in merge_dict:
            children.extend(merge_dict[b])
        merge_dict[c] = children
    frag_ids = set(np.unique(fragments).tolist()) - {0}

    # Build a lookup table mapping each leaf fragment id -> owning solution node id,
    # then apply it in a single vectorized pass. Merged/intermediate ids are all
    # > max(fragments) (see renumber_merge_history), so they never index into the
    # volume and only leaf slots are needed.
    max_frag_id = int(fragments.max())
    lookup = np.zeros(max_frag_id + 1, dtype=fragments.dtype)

    for node in solution_graph.nodes():
        if node in merge_dict:
            children = merge_dict[node]
        else:
            assert node in frag_ids, f"Node {node} not in merge dict or frag ids"
            children = [node]
        for child in children:
            if child > max_frag_id:
                continue  # intermediate/merged id, never present in the volume
            # Each leaf fragment may be claimed by at most one selected node (the ILP
            # ExclusiveNodes invariant). This preserves the original per-fragment
            # assertion as an O(children) check instead of a full-volume scan.
            assert lookup[child] == 0, (
                f"Child {child} fragment already assigned to node {lookup[child]}, "
                f"cannot reassign to {node}"
            )
            lookup[child] = node

    # Single O(n_pixels) vectorized remap.
    return lookup[fragments]


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

    print("\nBuilding solver and adding constraints/costs...")
    solver = motile.Solver(track_graph)
    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(1))
    # force_all=True: weights/constants start at 0 here, so the runtime
    # "0 weight + 0 constant = ablated" rule would skip every cost and leave
    # nothing to fit. Add every feature cost except those excluded via ablate_*.
    add_costs(solver, config, force_all=True, no_merges=no_merges)
    # ExclusiveNodes must be added before fit_weights — the loss-augmented ILP
    # in SoftMarginLoss copies solver.constraints at construction time.
    solver.add_constraint(motile.constraints.ExclusiveNodes(exclusion_sets))

    report_graph_statistics(config, track_graph)

    print("\nFitting weights via SSVM (this may take a while)...")
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
    write_learned_config(config, solver, learned_path)
    print(f"Wrote learned weights to {learned_path}")

    print("\nSolving with learned weights...")
    solver.solve()
    solution_graph = utils.to_nx_graph(solver.get_selected_subgraph())

    print("Saving results...")
    solution_seg = get_solution_seg(fragments, merge_history, solution_graph)
    assign_tracklet_ids(solution_graph)

    output_seg_path = output_dir / "pred_seg.zarr"
    output_zarr_root = zarr.open(
        output_seg_path, mode="a", shape=fragments.shape, chunks=(1, 1, 512, 512), dtype=np.uint32
    )
    if axes is not None:
        output_zarr_root.attrs["axes"] = axes
    output_zarr_root[:] = solution_seg

    output_filepath_geff = output_dir / "pred_tracks.zarr"
    metadata = geff.GeffMetadata(
        directed=True,
        related_objects=[{"type": "labels", "path": "../pred_seg.zarr", "label_prop": "label"}],
        node_props_metadata={},
        edge_props_metadata={},
    )
    geff.write(
        solution_graph,
        output_filepath_geff,
        axis_names=["time", "z", "y", "x"],
        axis_types=["time", "space", "space", "space"],
        axis_scales=scale,
        metadata=metadata,
        overwrite=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    raw_base_dir = Path(config["raw_base_dir"])
    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    dataset = config["dataset"]
    experiment = config["experiment"]
    assert raw_base_dir.is_dir()
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    raw_dir = raw_base_dir / experiment / f"{dataset}.zarr"
    assert raw_dir.is_dir(), f"Raw data directory {raw_dir} is missing"

    seg_dir = input_base_dir / "segmentation" / experiment / dataset / config["seg_result"]
    assert seg_dir.is_dir(), f"Segmentation data directory {seg_dir} is missing"

    flow_result = config.get("flow_result", None)
    if flow_result is not None:
        if config["use_lk"]:
            flow_dir_3d = input_base_dir / "opticalflow" / experiment / dataset / "opticalflow_lucaskanade" / flow_result
            flow_dir_2d = None
        else:
            flow_dir_2d = input_base_dir / "opticalflow" / experiment / dataset / "opticalflow_2d" / flow_result
            flow_dir_3d = input_base_dir / "opticalflow" / experiment / dataset / "opticalflow_3d" / flow_result
            if not flow_dir_2d.is_dir():
                flow_dir_2d = None
        flow_dirs = {"2d": flow_dir_2d, "3d": flow_dir_3d}
    else:
        flow_dirs = {"2d": None, "3d": None}

    gt_data_dir = input_base_dir / "tracking" / experiment / dataset

    if not config.get("exp_uid"):
        config["exp_uid"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = output_base_dir / "tracking" / experiment / dataset / config.get("output_name", "ssvm_fit")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving fit results to {output_dir}")

    fit_and_solve(config, raw_dir, seg_dir, flow_dirs, gt_data_dir, output_dir)
