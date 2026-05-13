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
import toml
import zarr

from funtracks.import_export import import_from_geff
from mhat.evaluation.evaluate_tracking import remap_seg_to_track_ids
from mhat.tracking import utils
from mhat.tracking.gt_annotation import annotate_gt_on_candidate_graph
from mhat.tracking.pipeline import build_track_graph
from mhat.tracking.solve_with_motile import add_costs, report_graph_statistics
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids


def configure_logging():
    """Surface structsvm bundle-method convergence output."""
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
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
    solution_seg = np.zeros_like(fragments)
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
    for node in solution_graph.nodes():
        if node in merge_dict:
            children = merge_dict[node]
        else:
            assert node in frag_ids, f"Node {node} not in merge dict or frag ids"
            children = [node]
        for child in children:
            solution_seg[fragments == child] = node
    return solution_seg


def fit_and_solve(config, raw_dir, seg_dir, flow_dirs, gt_data_dir, output_dir):
    configure_logging()

    # Persist input config (pre-fit) for reproducibility
    with open(output_dir / "config.toml", "w") as f:
        toml.dump(config, f)

    track_graph, fragments, merge_history, exclusion_sets, scale, axes = build_track_graph(
        config, raw_dir, seg_dir, flow_dirs
    )

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
    add_costs(solver, config)
    # ExclusiveNodes must be added before fit_weights — the loss-augmented ILP
    # in SoftMarginLoss copies solver.constraints at construction time.
    solver.add_constraint(motile.constraints.ExclusiveNodes(exclusion_sets))

    report_graph_statistics(config, track_graph)

    print("\nFitting weights via SSVM (this may take a while)...")
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

    output_dir = output_base_dir / "tracking" / experiment / dataset / "ssvm_fit"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving fit results to {output_dir}")

    fit_and_solve(config, raw_dir, seg_dir, flow_dirs, gt_data_dir, output_dir)
