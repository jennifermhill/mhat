"""Analyze properties of missed (FN) vs matched (TP) GT edges.

Compares edge attributes between true positive and false negative GT edges
to understand why certain edges are missed by the tracker.

Usage: conda run -n mhat-sandbox python scripts/analyze_fn_edges.py scripts/05_evaluation/eval_config.toml
"""
import argparse
from pathlib import Path

import geff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import toml
from numpy import linalg

from traccuracy import TrackingGraph, run_metrics
import traccuracy.matchers as matchers
import traccuracy.metrics as metrics

from funtracks.import_export import import_from_geff
from mhat.evaluation.evaluate_tracking import remap_seg_to_track_ids, matchers_dict

name_map = {"time": "time", "x": "x", "y": "y", "z": "z", "id": "track_id"}


def load_graphs(config, gt_data_dir, pred_data_dir):
    """Load GT and pred graphs, run matcher, return matched object and raw pred graph."""
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

    # Load pred
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
    matcher_name = config.get("matcher", "point")
    threshold = config.get("threshold", config.get("match_threshold", None))
    matcher_fn = matchers_dict[matcher_name]
    kwargs = {"threshold": threshold} if threshold is not None else {}
    matcher_obj = matcher_fn(**kwargs)
    matched = matcher_obj.compute_mapping(gt_tg, pred_tg)

    # Also load raw pred graph with edge attributes
    (raw_pred_graph, _) = geff.read(pred_data_dir / "pred_tracks.zarr")

    return matched, raw_pred_graph, scale


def analyze_edges(matched, raw_pred_graph, pred_data_dir, config):
    """Classify GT edges as TP/FN and compute attributes for each."""
    gt_graph = matched.gt_graph
    pred_graph = matched.pred_graph

    tp_edges = []  # (gt_u, gt_v, pred_u, pred_v)
    fn_edges_both_matched = []  # (gt_u, gt_v, pred_u, pred_v) — both nodes matched but edge missing
    fn_edges_node_unmatched = []  # (gt_u, gt_v) — at least one node not matched

    for gt_u, gt_v in gt_graph.graph.edges():
        pred_u = matched.get_gt_pred_match(gt_u)
        pred_v = matched.get_gt_pred_match(gt_v)

        if pred_u is None or pred_v is None:
            fn_edges_node_unmatched.append((gt_u, gt_v))
            continue

        if pred_graph.graph.has_edge(pred_u, pred_v):
            tp_edges.append((gt_u, gt_v, pred_u, pred_v))
        else:
            fn_edges_both_matched.append((gt_u, gt_v, pred_u, pred_v))

    # Split FN edges by whether they were in the candidate graph
    cand_edges_path = pred_data_dir / "candidate_edges.npy"
    if cand_edges_path.exists():
        cand_edges = np.load(cand_edges_path)
        cand_edge_set = set(map(tuple, cand_edges))
        fn_in_cand = [e for e in fn_edges_both_matched if (e[2], e[3]) in cand_edge_set]
        fn_not_in_cand = [e for e in fn_edges_both_matched if (e[2], e[3]) not in cand_edge_set]
    else:
        fn_in_cand = fn_edges_both_matched
        fn_not_in_cand = []

    print(f"\nEdge Classification:")
    print(f"  TP edges: {len(tp_edges)}")
    print(f"  FN edges (both nodes matched, edge missing): {len(fn_edges_both_matched)}")
    print(f"    - Were candidates but ILP rejected: {len(fn_in_cand)}")
    print(f"    - Never candidates (graph construction): {len(fn_not_in_cand)}")
    print(f"  FN edges (node unmatched): {len(fn_edges_node_unmatched)}")
    print(f"  Total GT edges: {len(tp_edges) + len(fn_edges_both_matched) + len(fn_edges_node_unmatched)}")

    # Collect attributes for TP edges (from raw pred graph which has edge attrs)
    tp_attrs = collect_tp_attrs(tp_edges, raw_pred_graph)

    # Collect attributes for FN edges (computed from pred node properties)
    fn_attrs = collect_fn_attrs(fn_edges_both_matched, raw_pred_graph)

    # Also collect GT node distances for both groups
    gt_graph_nx = gt_graph.graph
    tp_gt_dists = []
    for gt_u, gt_v, _, _ in tp_edges:
        pos_u = np.array(gt_graph_nx.nodes[gt_u]["pos"])
        pos_v = np.array(gt_graph_nx.nodes[gt_v]["pos"])
        tp_gt_dists.append(linalg.norm(pos_v - pos_u))

    fn_gt_dists = []
    for gt_u, gt_v, _, _ in fn_edges_both_matched:
        pos_u = np.array(gt_graph_nx.nodes[gt_u]["pos"])
        pos_v = np.array(gt_graph_nx.nodes[gt_v]["pos"])
        fn_gt_dists.append(linalg.norm(pos_v - pos_u))

    # Collect substitute attrs only for FN edges that WERE candidates (ILP rejected)
    fn_ilp_attrs = collect_fn_attrs(fn_in_cand, raw_pred_graph)
    # Use the tracking config that was actually used for this run
    tracking_config_path = pred_data_dir / "config.toml"
    run_config = toml.load(tracking_config_path) if tracking_config_path.exists() else config
    substitute_attrs = collect_substitute_attrs(fn_in_cand, raw_pred_graph, run_config)

    tp_pred_edges = {(pu, pv) for _, _, pu, pv in tp_edges}
    return tp_attrs, fn_attrs, np.array(tp_gt_dists), np.array(fn_gt_dists), fn_edges_both_matched, substitute_attrs, fn_ilp_attrs, run_config, tp_pred_edges


def collect_tp_attrs(tp_edges, raw_pred_graph):
    """Collect edge attributes for TP edges from the raw pred graph."""
    attrs = {"drift_dist": [], "area_diff": [], "intensity_diff": [],
             "node_dist": [], "flow_mag": []}
    for _, _, pred_u, pred_v in tp_edges:
        if raw_pred_graph.has_edge(pred_u, pred_v):
            edata = raw_pred_graph[pred_u][pred_v]
            attrs["drift_dist"].append(edata.get("drift_dist", np.nan))
            attrs["area_diff"].append(edata.get("area_diff", np.nan))
            attrs["intensity_diff"].append(edata.get("intensity_diff", np.nan))
        else:
            attrs["drift_dist"].append(np.nan)
            attrs["area_diff"].append(np.nan)
            attrs["intensity_diff"].append(np.nan)

        # Compute node distance and flow magnitude from node attrs
        nu = raw_pred_graph.nodes[pred_u]
        nv = raw_pred_graph.nodes[pred_v]
        pos_u = np.array([nu["z"], nu["y"], nu["x"]])
        pos_v = np.array([nv["z"], nv["y"], nv["x"]])
        attrs["node_dist"].append(linalg.norm(pos_v - pos_u))
        if "flow" in nu:
            attrs["flow_mag"].append(linalg.norm(nu["flow"]))
        else:
            attrs["flow_mag"].append(np.nan)

    return {k: np.array(v) for k, v in attrs.items()}


def collect_substitute_attrs(fn_edges, raw_pred_graph, config):
    """For each FN edge (pred_u, pred_v), collect attrs of the outgoing edge from pred_u
    and the incoming edge to pred_v that the solution picked instead. Also compute ILP costs."""
    outgoing = {"drift_dist": [], "area_diff": [], "intensity_diff": [],
                "node_dist": [], "flow_mag": [], "edge_cost": [], "target_node_cost": []}
    incoming = {"drift_dist": [], "area_diff": [], "intensity_diff": [],
                "node_dist": [], "flow_mag": [], "edge_cost": [], "source_node_cost": []}
    fn_costs = {"edge_cost": [], "target_node_cost": [], "source_node_cost": []}
    n_no_outgoing = 0
    n_no_incoming = 0

    for _, _, pred_u, pred_v in fn_edges:
        # FN edge costs (computed from node attributes since edge may not exist in solution)
        nu = raw_pred_graph.nodes[pred_u]
        nv = raw_pred_graph.nodes[pred_v]
        # Compute drift_dist for FN edge
        pos_u = np.array([nu["z"], nu["y"], nu["x"]])
        pos_v = np.array([nv["z"], nv["y"], nv["x"]])
        flow_u = np.array(nu.get("flow", [0, 0, 0]))
        fn_drift = linalg.norm(pos_u + flow_u - pos_v)
        fn_edge_cost = config.get("drift_weight", 0) * fn_drift + config.get("drift_constant", 0)
        fn_costs["edge_cost"].append(fn_edge_cost)
        fn_costs["target_node_cost"].append(compute_node_ilp_cost(config, raw_pred_graph, pred_v))
        fn_costs["source_node_cost"].append(compute_node_ilp_cost(config, raw_pred_graph, pred_u))

        # Outgoing edge from pred_u in solution
        out_edges = list(raw_pred_graph.successors(pred_u))
        if out_edges:
            succ = out_edges[0]
            _append_edge_attrs(outgoing, raw_pred_graph, pred_u, succ)
            outgoing["edge_cost"].append(compute_edge_ilp_cost(config, raw_pred_graph, pred_u, succ))
            outgoing["target_node_cost"].append(compute_node_ilp_cost(config, raw_pred_graph, succ))
        else:
            n_no_outgoing += 1

        # Incoming edge to pred_v in solution
        in_edges = list(raw_pred_graph.predecessors(pred_v))
        if in_edges:
            pred = in_edges[0]
            _append_edge_attrs(incoming, raw_pred_graph, pred, pred_v)
            incoming["edge_cost"].append(compute_edge_ilp_cost(config, raw_pred_graph, pred, pred_v))
            incoming["source_node_cost"].append(compute_node_ilp_cost(config, raw_pred_graph, pred))
        else:
            n_no_incoming += 1

    print(f"\nSubstitute edge analysis ({len(fn_edges)} FN edges):")
    print(f"  pred_u has no outgoing edge (track ends): {n_no_outgoing}/{len(fn_edges)}")
    print(f"  pred_v has no incoming edge (track starts): {n_no_incoming}/{len(fn_edges)}")

    return {
        "outgoing": {k: np.array(v) for k, v in outgoing.items()},
        "incoming": {k: np.array(v) for k, v in incoming.items()},
        "fn_costs": {k: np.array(v) for k, v in fn_costs.items()},
        "n_no_outgoing": n_no_outgoing,
        "n_no_incoming": n_no_incoming,
    }


def compute_edge_ilp_cost(config, graph, u, v):
    """Compute the total ILP edge cost for edge (u, v) using config weights."""
    edata = graph[u][v] if graph.has_edge(u, v) else {}
    cost = 0.0
    for attr, w_key, c_key in [
        ("drift_dist", "drift_weight", "drift_constant"),
        ("area_diff", "area_weight", "area_constant"),
        ("intensity_diff", "intensity_weight", "intensity_constant"),
    ]:
        w = config.get(w_key, 0.0)
        c = config.get(c_key, 0.0)
        if w != 0 and attr in edata:
            cost += w * edata[attr] + c
    return cost


def compute_node_ilp_cost(config, graph, node):
    """Compute the total ILP node cost for a node using config weights."""
    ndata = graph.nodes[node]
    cost = 0.0
    for attr, w_key, c_key in [
        ("cohesion", "cohesion_weight", "cohesion_constant"),
        ("adhesion", "adhesion_weight", "adhesion_constant"),
    ]:
        w = config.get(w_key, 0.0)
        c = config.get(c_key, 0.0)
        if w != 0 and attr in ndata:
            cost += (w * ndata[attr] + c) * ndata.get("num_leaves", 1)
    return cost


def _append_edge_attrs(attrs, graph, u, v):
    """Append edge attributes for edge (u, v) to the attrs dict."""
    nu = graph.nodes[u]
    nv = graph.nodes[v]
    edata = graph[u][v] if graph.has_edge(u, v) else {}

    attrs["drift_dist"].append(edata.get("drift_dist", np.nan))
    attrs["area_diff"].append(edata.get("area_diff", np.nan))
    attrs["intensity_diff"].append(edata.get("intensity_diff", np.nan))

    pos_u = np.array([nu["z"], nu["y"], nu["x"]])
    pos_v = np.array([nv["z"], nv["y"], nv["x"]])
    attrs["node_dist"].append(linalg.norm(pos_v - pos_u))

    if "flow" in nu:
        attrs["flow_mag"].append(linalg.norm(nu["flow"]))
    else:
        attrs["flow_mag"].append(np.nan)


def collect_fn_attrs(fn_edges, raw_pred_graph):
    """Compute edge attributes for FN edges from pred node properties."""
    attrs = {"node_dist": [], "area_diff": [], "intensity_diff": [],
             "flow_mag": [], "drift_dist": []}
    for _, _, pred_u, pred_v in fn_edges:
        nu = raw_pred_graph.nodes[pred_u]
        nv = raw_pred_graph.nodes[pred_v]
        pos_u = np.array([nu["z"], nu["y"], nu["x"]])
        pos_v = np.array([nv["z"], nv["y"], nv["x"]])

        attrs["node_dist"].append(linalg.norm(pos_v - pos_u))

        area_u, area_v = nu.get("area", 0), nv.get("area", 0)
        mean_area = (area_u + area_v) / 2
        attrs["area_diff"].append(abs(area_u - area_v) / mean_area if mean_area > 0 else 0)

        attrs["intensity_diff"].append(abs(nu.get("intensity", 0) - nv.get("intensity", 0)))

        if "flow" in nu:
            flow_u = np.array(nu["flow"])
            attrs["flow_mag"].append(linalg.norm(flow_u))
            attrs["drift_dist"].append(linalg.norm(pos_u + flow_u - pos_v))
        else:
            attrs["flow_mag"].append(np.nan)
            attrs["drift_dist"].append(np.nan)

    return {k: np.array(v) for k, v in attrs.items()}


def plot_histograms(tp_attrs, fn_ilp_attrs, substitute_attrs, run_config,
                    output_dir, raw_pred_graph, tp_pred_edges):
    """Save histogram PNGs comparing correct vs incorrect vs substitute edges."""
    ATTR_SPECS = [
        ("drift_dist", "drift_weight", "drift_constant"),
        ("area_diff", "area_weight", "area_constant"),
        ("intensity_diff", "intensity_weight", "intensity_constant"),
    ]

    # Collect attributes for all incorrect edges (in solution but not TP)
    incorrect_attrs = {a: [] for a, _, _ in ATTR_SPECS}
    for u, v in raw_pred_graph.edges():
        if (u, v) in tp_pred_edges:
            continue
        edata = raw_pred_graph[u][v]
        nu = raw_pred_graph.nodes[u]
        nv = raw_pred_graph.nodes[v]
        incorrect_attrs["drift_dist"].append(edata.get("drift_dist", np.nan))
        incorrect_attrs["area_diff"].append(edata.get("area_diff", np.nan))
        incorrect_attrs["intensity_diff"].append(edata.get("intensity_diff", np.nan))
    incorrect_attrs = {k: np.array(v) for k, v in incorrect_attrs.items()}

    output_dir.mkdir(parents=True, exist_ok=True)

    for attr, w_key, c_key in ATTR_SPECS:
        weight = run_config.get(w_key, 0.0)
        constant = run_config.get(c_key, 0.0)

        correct = np.concatenate([tp_attrs[attr], fn_ilp_attrs[attr]])
        correct = correct[~np.isnan(correct)]

        sub = substitute_attrs["outgoing"].get(attr, np.array([]))
        sub = sub[~np.isnan(sub)] if len(sub) > 0 else sub

        incorr = incorrect_attrs[attr]
        incorr = incorr[~np.isnan(incorr)] if len(incorr) > 0 else incorr

        if len(correct) == 0:
            continue

        # --- Histogram 1: Attribute value ---
        arrays = [correct, incorr, sub]
        all_vals = np.concatenate([a for a in arrays if len(a) > 0])
        bins = np.histogram_bin_edges(all_vals, bins=30)
        bin_width = bins[1] - bins[0]
        bw = bin_width * 0.3

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(bins[:-1], np.histogram(correct, bins=bins)[0],
               width=bw, align='edge', alpha=0.8,
               label=f"Correct (n={len(correct)})", color="steelblue")
        if len(incorr) > 0:
            ax.bar(bins[:-1] + bw, np.histogram(incorr, bins=bins)[0],
                   width=bw, align='edge', alpha=0.8,
                   label=f"All incorrect (n={len(incorr)})", color="gray")
        if len(sub) > 0:
            ax.bar(bins[:-1] + 2 * bw, np.histogram(sub, bins=bins)[0],
                   width=bw, align='edge', alpha=0.8,
                   label=f"Substitute (n={len(sub)})", color="salmon")

        if weight != 0:
            breakeven_attr = -constant / weight
            ax.axvline(breakeven_attr, color="black", linestyle="--",
                       linewidth=1.5, label=f"Break-even ({breakeven_attr:.1f})")

        ax.set_xlabel(attr)
        ax.set_ylabel("# of edges")
        ax.set_title(f"{attr} distribution: correct vs incorrect vs substitute")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"hist_{attr}_value.png", dpi=150)
        plt.close(fig)

        # --- Histogram 2: Cost value (skip if weight==0) ---
        if weight == 0:
            continue

        correct_cost = weight * correct + constant
        sub_cost = weight * sub + constant
        incorr_cost = weight * incorr + constant

        arrays_c = [correct_cost, incorr_cost, sub_cost]
        all_costs = np.concatenate([a for a in arrays_c if len(a) > 0])
        bins_c = np.histogram_bin_edges(all_costs, bins=30)
        bin_width_c = bins_c[1] - bins_c[0]
        bw_c = bin_width_c * 0.3

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(bins_c[:-1], np.histogram(correct_cost, bins=bins_c)[0],
               width=bw_c, align='edge', alpha=0.8,
               label=f"Correct (n={len(correct_cost)})", color="steelblue")
        if len(incorr_cost) > 0:
            ax.bar(bins_c[:-1] + bw_c, np.histogram(incorr_cost, bins=bins_c)[0],
                   width=bw_c, align='edge', alpha=0.8,
                   label=f"All incorrect (n={len(incorr_cost)})", color="gray")
        if len(sub_cost) > 0:
            ax.bar(bins_c[:-1] + 2 * bw_c, np.histogram(sub_cost, bins=bins_c)[0],
                   width=bw_c, align='edge', alpha=0.8,
                   label=f"Substitute (n={len(sub_cost)})", color="salmon")

        ax.axvline(0, color="black", linestyle="--",
                   linewidth=1.5, label="Break-even (cost=0)")

        ax.set_xlabel(f"{attr} cost (w={weight}, c={constant})")
        ax.set_ylabel("# of edges")
        ax.set_title(f"{attr} cost distribution: correct vs incorrect vs substitute")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"hist_{attr}_cost.png", dpi=150)
        plt.close(fig)

    print(f"\nHistograms saved to {output_dir}")


def print_comparison(tp_attrs, fn_attrs, tp_gt_dists, fn_gt_dists, fn_edges_both_matched, pred_data_dir, substitute_attrs, fn_ilp_attrs):
    """Print comparison table of TP vs FN edge attributes."""
    print(f"\n{'='*90}")
    print(f"Edge Attribute Comparison: TP ({len(tp_gt_dists)}) vs FN ({len(fn_gt_dists)}) edges")
    print(f"{'='*90}")
    header = f"{'Attribute':<20} {'TP Mean':>10} {'TP Std':>10} {'TP Med':>10} {'FN Mean':>10} {'FN Std':>10} {'FN Med':>10}"
    print(header)
    print("-" * 90)

    # GT node distances
    print(f"{'GT node dist':<20} {tp_gt_dists.mean():>10.3f} {tp_gt_dists.std():>10.3f} {np.median(tp_gt_dists):>10.3f} "
          f"{fn_gt_dists.mean():>10.3f} {fn_gt_dists.std():>10.3f} {np.median(fn_gt_dists):>10.3f}")

    # Pred node distances
    if "node_dist" in tp_attrs and "node_dist" in fn_attrs:
        tp_nd = tp_attrs["node_dist"]
        fn_nd = fn_attrs["node_dist"]
        print(f"{'Pred node dist':<20} {tp_nd.mean():>10.3f} {tp_nd.std():>10.3f} {np.median(tp_nd):>10.3f} "
              f"{fn_nd.mean():>10.3f} {fn_nd.std():>10.3f} {np.median(fn_nd):>10.3f}")

    # Drift dist
    if "drift_dist" in tp_attrs and "drift_dist" in fn_attrs:
        tp_dd = tp_attrs["drift_dist"][~np.isnan(tp_attrs["drift_dist"])]
        fn_dd = fn_attrs["drift_dist"][~np.isnan(fn_attrs["drift_dist"])]
        if len(tp_dd) > 0 and len(fn_dd) > 0:
            print(f"{'Drift dist':<20} {tp_dd.mean():>10.3f} {tp_dd.std():>10.3f} {np.median(tp_dd):>10.3f} "
                  f"{fn_dd.mean():>10.3f} {fn_dd.std():>10.3f} {np.median(fn_dd):>10.3f}")

    # Area diff
    for attr in ["area_diff", "intensity_diff"]:
        if attr in tp_attrs and attr in fn_attrs:
            tp_a = tp_attrs[attr][~np.isnan(tp_attrs[attr])]
            fn_a = fn_attrs[attr][~np.isnan(fn_attrs[attr])]
            if len(tp_a) > 0 and len(fn_a) > 0:
                print(f"{attr:<20} {tp_a.mean():>10.3f} {tp_a.std():>10.3f} {np.median(tp_a):>10.3f} "
                      f"{fn_a.mean():>10.3f} {fn_a.std():>10.3f} {np.median(fn_a):>10.3f}")

    # Flow magnitude
    if "flow_mag" in tp_attrs and "flow_mag" in fn_attrs:
        tp_fm = tp_attrs["flow_mag"][~np.isnan(tp_attrs["flow_mag"])]
        fn_fm = fn_attrs["flow_mag"][~np.isnan(fn_attrs["flow_mag"])]
        if len(tp_fm) > 0 and len(fn_fm) > 0:
            print(f"{'Flow magnitude':<20} {tp_fm.mean():>10.3f} {tp_fm.std():>10.3f} {np.median(tp_fm):>10.3f} "
                  f"{fn_fm.mean():>10.3f} {fn_fm.std():>10.3f} {np.median(fn_fm):>10.3f}")

    print("=" * 90)


    # Print substitute edge comparison (only for ILP-rejected FN edges)
    out = substitute_attrs["outgoing"]
    inc = substitute_attrs["incoming"]
    n_out = len(out["node_dist"])
    n_inc = len(inc["node_dist"])
    n_ilp_rejected = n_out + substitute_attrs["n_no_outgoing"]
    if n_out > 0 or n_inc > 0:
        print(f"\n{'='*100}")
        print(f"Substitute Edges: what the solution picked INSTEAD of the {n_ilp_rejected} ILP-rejected FN edges")
        print(f"{'='*100}")
        header = f"{'Attribute':<20} {'FN Mean':>10} {'Out Mean':>10} {'Out Std':>10} {'Out Med':>10} {'In Mean':>10} {'In Std':>10} {'In Med':>10}"
        print(header)
        print("-" * 100)

        for attr in ["node_dist", "drift_dist", "area_diff", "intensity_diff", "flow_mag"]:
            fn_vals = fn_ilp_attrs.get(attr, np.array([]))
            fn_vals = fn_vals[~np.isnan(fn_vals)] if len(fn_vals) > 0 else fn_vals
            out_vals = out.get(attr, np.array([]))
            out_vals = out_vals[~np.isnan(out_vals)] if len(out_vals) > 0 else out_vals
            in_vals = inc.get(attr, np.array([]))
            in_vals = in_vals[~np.isnan(in_vals)] if len(in_vals) > 0 else in_vals

            fn_m = f"{fn_vals.mean():.3f}" if len(fn_vals) > 0 else "N/A"
            out_m = f"{out_vals.mean():.3f}" if len(out_vals) > 0 else "N/A"
            out_s = f"{out_vals.std():.3f}" if len(out_vals) > 0 else "N/A"
            out_d = f"{np.median(out_vals):.3f}" if len(out_vals) > 0 else "N/A"
            in_m = f"{in_vals.mean():.3f}" if len(in_vals) > 0 else "N/A"
            in_s = f"{in_vals.std():.3f}" if len(in_vals) > 0 else "N/A"
            in_d = f"{np.median(in_vals):.3f}" if len(in_vals) > 0 else "N/A"

            print(f"{attr:<20} {fn_m:>10} {out_m:>10} {out_s:>10} {out_d:>10} {in_m:>10} {in_s:>10} {in_d:>10}")

        print("=" * 100)
        print(f"  pred_u has no outgoing edge (track ends): {substitute_attrs['n_no_outgoing']}/{len(fn_edges_both_matched)}")
        print(f"  pred_v has no incoming edge (track starts): {substitute_attrs['n_no_incoming']}/{len(fn_edges_both_matched)}")

    # Print ILP cost comparison
    fn_c = substitute_attrs.get("fn_costs", {})
    out_c = substitute_attrs["outgoing"]
    inc_c = substitute_attrs["incoming"]
    if "edge_cost" in fn_c and "edge_cost" in out_c:
        fn_ec = fn_c["edge_cost"]
        out_ec = out_c.get("edge_cost", np.array([]))
        inc_ec = inc_c.get("edge_cost", np.array([]))
        fn_tnc = fn_c.get("target_node_cost", np.array([]))
        fn_snc = fn_c.get("source_node_cost", np.array([]))
        out_tnc = out_c.get("target_node_cost", np.array([]))
        inc_snc = inc_c.get("source_node_cost", np.array([]))

        print(f"\n{'='*100}")
        print(f"ILP Cost Comparison: FN edge (correct) vs Substitute edges")
        print(f"{'='*100}")
        print(f"{'Cost component':<25} {'FN Mean':>12} {'FN Med':>12} {'Sub Out Mean':>12} {'Sub Out Med':>12} {'Sub In Mean':>12} {'Sub In Med':>12}")
        print("-" * 100)

        def _fmt(arr):
            if len(arr) == 0: return ("N/A", "N/A")
            return (f"{arr.mean():.1f}", f"{np.median(arr):.1f}")

        fm, fmd = _fmt(fn_ec)
        om, omd = _fmt(out_ec)
        im, imd = _fmt(inc_ec)
        print(f"{'Edge cost':<25} {fm:>12} {fmd:>12} {om:>12} {omd:>12} {im:>12} {imd:>12}")

        fm, fmd = _fmt(fn_tnc)
        om, omd = _fmt(out_tnc)
        print(f"{'Target node cost':<25} {fm:>12} {fmd:>12} {om:>12} {omd:>12} {'—':>12} {'—':>12}")

        fm, fmd = _fmt(fn_snc)
        im, imd = _fmt(inc_snc)
        print(f"{'Source node cost':<25} {fm:>12} {fmd:>12} {'—':>12} {'—':>12} {im:>12} {imd:>12}")

        # Total cost: edge + target node (for outgoing comparison)
        if len(fn_ec) > 0 and len(out_ec) > 0 and len(fn_tnc) > 0 and len(out_tnc) > 0:
            fn_total_out = fn_ec + fn_tnc
            out_total = out_ec + out_tnc
            print(f"{'Edge+target (out)':<25} {fn_total_out.mean():>12.1f} {np.median(fn_total_out):>12.1f} {out_total.mean():>12.1f} {np.median(out_total):>12.1f} {'—':>12} {'—':>12}")

        # Total cost: edge + source node (for incoming comparison)
        if len(fn_ec) > 0 and len(inc_ec) > 0 and len(fn_snc) > 0 and len(inc_snc) > 0:
            fn_total_in = fn_ec + fn_snc
            inc_total = inc_ec + inc_snc
            print(f"{'Edge+source (in)':<25} {fn_total_in.mean():>12.1f} {np.median(fn_total_in):>12.1f} {'—':>12} {'—':>12} {inc_total.mean():>12.1f} {np.median(inc_total):>12.1f}")

        print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config["input_base_dir"])
    experiment = config["experiment"]
    dataset = config["dataset"]

    gt_data_dir = input_base_dir / "tracking" / experiment / dataset
    # run_tracking.py writes to .../<exp_uid>/; "test_run" is only the legacy default.
    track_result = config.get("track_result", config.get("exp_uid", "test_run"))
    pred_data_dir = input_base_dir / "tracking" / experiment / dataset / track_result

    print(f"Analyzing FN edges for {experiment}/{dataset}")
    print(f"GT: {gt_data_dir}")
    print(f"Pred: {pred_data_dir}")

    matched, raw_pred_graph, scale = load_graphs(config, gt_data_dir, pred_data_dir)
    tp_attrs, fn_attrs, tp_gt_dists, fn_gt_dists, fn_edges_both_matched, substitute_attrs, fn_ilp_attrs, run_config, tp_pred_edges = analyze_edges(matched, raw_pred_graph, pred_data_dir, config)
    print_comparison(tp_attrs, fn_attrs, tp_gt_dists, fn_gt_dists, fn_edges_both_matched, pred_data_dir, substitute_attrs, fn_ilp_attrs)

    # Save histograms to evaluation output folder
    output_base_dir = Path(config.get("output_base_dir", config["input_base_dir"]))
    track_result = config.get("track_result", run_config.get("exp_uid", "unknown"))
    eval_output_dir = output_base_dir / "evaluation" / experiment / dataset / track_result
    plot_histograms(tp_attrs, fn_ilp_attrs, substitute_attrs, run_config,
                    eval_output_dir, raw_pred_graph, tp_pred_edges)
