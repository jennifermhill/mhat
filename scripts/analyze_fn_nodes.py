"""Analyze unmatched GT nodes to determine why they are FN.

For each GT node without a pred match, determines if:
1. No segmentation candidate exists at that location (segmentation miss)
2. A candidate exists but the ILP didn't select it (solver miss)
3. A candidate was selected but the matcher didn't match it (matcher miss)

Usage: conda run -n mhat-sandbox python scripts/analyze_fn_nodes.py scripts/05_evaluation/eval_config.toml
"""
import argparse
from pathlib import Path

import geff
import numpy as np
import toml
import zarr
from numpy import linalg
from scipy.ndimage import label as cc_label

from traccuracy import TrackingGraph, run_metrics
import traccuracy.matchers as matchers

from funtracks.import_export import import_from_geff
from mhat.evaluation.evaluate_tracking import remap_seg_to_track_ids, matchers_dict

name_map = {"time": "time", "x": "x", "y": "y", "z": "z", "id": "track_id"}


def analyze_fn_nodes(config, gt_data_dir, pred_data_dir):
    # Load pred metadata for scale
    (_, metadata) = geff.read(pred_data_dir / "pred_tracks.zarr")
    axes = metadata.axes
    scale = [a.scale for a in axes if a.scale is not None] if axes else [1.0, 1.0, 1.0, 1.0]

    # Load GT
    gt_seg_path = gt_data_dir / "correct_seg.zarr"
    gt_seg_path = gt_seg_path if gt_seg_path.exists() else None
    gt_tracks = import_from_geff(gt_data_dir / "correct_tracks.zarr", name_map,
                                  segmentation_path=gt_seg_path, scale=scale)
    gt_seg = None
    if gt_tracks.segmentation is not None:
        gt_seg = remap_seg_to_track_ids(
            gt_data_dir / "correct_tracks.zarr", gt_tracks.graph, gt_tracks.segmentation)
    gt_tg = TrackingGraph(graph=gt_tracks.graph, frame_key="time", label_key="track_id",
                          location_keys="pos", segmentation=gt_seg)

    # Load pred (solution)
    pred_seg_path = pred_data_dir / "pred_seg.zarr"
    pred_seg_path = pred_seg_path if pred_seg_path.exists() else None
    pred_tracks = import_from_geff(pred_data_dir / "pred_tracks.zarr", name_map,
                                    segmentation_path=pred_seg_path, scale=scale)
    pred_seg = None
    if pred_tracks.segmentation is not None:
        pred_seg = remap_seg_to_track_ids(
            pred_data_dir / "pred_tracks.zarr", pred_tracks.graph, pred_tracks.segmentation)
    pred_tg = TrackingGraph(graph=pred_tracks.graph, frame_key="time", label_key="track_id",
                            location_keys="pos", segmentation=pred_seg)

    # Run matcher
    matcher_name = config.get("matcher", "ctc")
    threshold = config.get("threshold", config.get("match_threshold", None))
    matcher_fn = matchers_dict[matcher_name]
    kwargs = {"threshold": threshold} if threshold is not None else {}
    matcher_obj = matcher_fn(**kwargs)
    matched = matcher_obj.compute_mapping(gt_tg, pred_tg)

    # Load the full candidate segmentation (all hypotheses, not just solution)
    seg_dir = Path(config["input_base_dir"]) / "segmentation" / config["experiment"] / config["dataset"]
    run_config = toml.load(pred_data_dir / "config.toml")
    seg_result = run_config["seg_result"]
    seg_zarr_path = seg_dir / seg_result / "data.zarr"
    seg_zarr_root = zarr.open(seg_zarr_path)
    fragments = seg_zarr_root["fragments"][:]

    # Load solution segmentation
    solution_seg = zarr.open(pred_data_dir / "pred_seg.zarr")[:]

    # Find unmatched GT nodes
    gt_graph = matched.gt_graph.graph
    pred_graph = matched.pred_graph.graph

    fn_nodes = []
    for node in gt_graph.nodes():
        if matched.get_gt_pred_match(node) is None:
            fn_nodes.append(node)

    print(f"\nTotal GT nodes: {gt_graph.number_of_nodes()}")
    print(f"Unmatched GT nodes (FN): {len(fn_nodes)}")

    # Precompute fragment and solution centroids per timepoint (in scaled coords)
    from scipy.ndimage import center_of_mass as com
    scale_spatial = np.array(scale[1:])

    frag_centroids = {}  # {(t, label): scaled_pos}
    sol_centroids = {}
    for t in range(fragments.shape[0]):
        frag_labels_t = set(np.unique(fragments[t])) - {0}
        for lab in frag_labels_t:
            c = np.array(com(fragments[t] == lab)) * scale_spatial
            frag_centroids[(t, int(lab))] = c

        sol_labels_t = set(np.unique(solution_seg[t])) - {0}
        for lab in sol_labels_t:
            c = np.array(com(solution_seg[t] == lab)) * scale_spatial
            sol_centroids[(t, int(lab))] = c

    def nearest_centroid(centroids, t, pos):
        """Find nearest centroid at timepoint t, return (label, distance)."""
        best_lab, best_dist = None, float('inf')
        for (ct, lab), cpos in centroids.items():
            if ct != t:
                continue
            d = linalg.norm(cpos - pos)
            if d < best_dist:
                best_dist = d
                best_lab = lab
        return best_lab, best_dist

    # Classify each FN node
    seg_miss = []       # No nearby candidate in segmentation
    solver_miss = []    # Candidate nearby but not in solution
    matcher_miss = []   # In solution but matcher didn't match

    for node in fn_nodes:
        ndata = gt_graph.nodes[node]
        t = ndata["time"]
        pos = np.array(ndata["pos"])  # scaled position

        # Find nearest pred node in the solution graph
        nearest_pred_dist = float('inf')
        nearest_pred_node = None
        for pnode in pred_graph.nodes():
            pdata = pred_graph.nodes[pnode]
            if pdata["time"] != t:
                continue
            ppos = np.array(pdata["pos"])
            d = linalg.norm(ppos - pos)
            if d < nearest_pred_dist:
                nearest_pred_dist = d
                nearest_pred_node = pnode

        # Find nearest fragment centroid and nearest solution centroid
        nearest_frag_lab, nearest_frag_dist = nearest_centroid(frag_centroids, t, pos)
        nearest_sol_lab, nearest_sol_dist = nearest_centroid(sol_centroids, t, pos)

        info = {
            "node": node, "time": t, "pos": pos,
            "nearest_pred_node": nearest_pred_node,
            "nearest_pred_dist": nearest_pred_dist,
            "nearest_frag_lab": nearest_frag_lab,
            "nearest_frag_dist": nearest_frag_dist,
            "nearest_sol_lab": nearest_sol_lab,
            "nearest_sol_dist": nearest_sol_dist,
        }

        # Classify: use nearest solution centroid distance to decide
        # If a solution object is nearby, it's a matcher miss
        # If no solution but a fragment is nearby, it's a solver miss
        # If no fragment nearby, it's a segmentation miss
        match_threshold = 20.0  # max distance to consider "nearby"

        if nearest_sol_dist < match_threshold:
            matcher_miss.append(info)
        elif nearest_frag_dist < match_threshold:
            solver_miss.append(info)
        else:
            seg_miss.append(info)

    # Print all FN nodes with their GT data
    print(f"\n=== All FN GT Nodes ({len(fn_nodes)}) ===")
    print(f"{'Node':<8} {'Time':<6} {'Track ID':<10} {'Position':<40} {'Other Attrs'}")
    print("-" * 110)
    for node in fn_nodes:
        ndata = dict(gt_graph.nodes[node])
        t = ndata.pop("time", "?")
        track_id = ndata.pop("track_id", "?")
        pos = ndata.pop("pos", None)
        pos_str = np.array2string(np.asarray(pos), precision=1) if pos is not None else "?"
        # Remove internal/redundant keys
        for key in list(ndata.keys()):
            if key in ("label", "segmentation_id"):
                ndata.pop(key)
        other = ", ".join(f"{k}={v}" for k, v in ndata.items()) if ndata else ""
        print(f"{node:<8} {t:<6} {track_id:<10} {pos_str:<40} {other}")

    print(f"\n=== FN Node Classification ===")
    print(f"  Segmentation miss (no candidate at GT location): {len(seg_miss)}")
    print(f"  Solver miss (candidate exists, not selected):    {len(solver_miss)}")
    print(f"  Matcher miss (selected, not matched):            {len(matcher_miss)}")

    def _print_section(title, items):
        if not items:
            return
        print(f"\n--- {title} ---")
        print(f"  {'Node':<8} {'t':<4} {'Track':<8} {'Frag Dist':<12} {'Sol Dist':<12} {'Pred Dist':<12} {'Nearest Frag':<14} {'Nearest Sol':<14} {'Nearest Pred'}")
        print(f"  {'-'*100}")
        for info in items:
            node = info['node']
            track_id = gt_graph.nodes[node].get("track_id", "?")
            print(f"  {node:<8} {info['time']:<4} {track_id:<8} "
                  f"{info['nearest_frag_dist']:>10.1f}  "
                  f"{info['nearest_sol_dist']:>10.1f}  "
                  f"{info['nearest_pred_dist']:>10.1f}  "
                  f"{str(info['nearest_frag_lab']):<14} "
                  f"{str(info['nearest_sol_lab']):<14} "
                  f"{str(info['nearest_pred_node'])}")

    _print_section(f"Segmentation misses (nearest frag >= {match_threshold})", seg_miss)
    _print_section(f"Solver misses (frag < {match_threshold}, sol >= {match_threshold})", solver_miss)
    _print_section(f"Matcher misses (sol < {match_threshold})", matcher_miss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config["input_base_dir"])
    experiment = config["experiment"]
    dataset = config["dataset"]

    gt_data_dir = input_base_dir / "tracking" / experiment / dataset
    pred_data_dir = input_base_dir / "tracking" / experiment / dataset / "test_run"

    print(f"Analyzing FN nodes for {experiment}/{dataset}")
    analyze_fn_nodes(config, gt_data_dir, pred_data_dir)
