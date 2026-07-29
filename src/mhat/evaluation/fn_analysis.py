"""Diagnostics explaining *why* the tracker misses ground truth nodes and edges.

Two analyses live here, both driven by an already-computed traccuracy
``Matched`` object so they cost no extra matching work when run as part of
``evaluate_tracks.py``:

- :func:`analyze_fn_edges` classifies every GT edge as TP or FN, splits the FN
  edges into "was a candidate but the ILP rejected it" vs "never a candidate",
  and compares the attributes/ILP costs of the missed edges against the edges
  the solution picked instead.
- :func:`analyze_fn_nodes` classifies every unmatched GT node as a
  segmentation miss (no candidate fragment nearby), a solver miss (a candidate
  fragment exists but was not selected) or a matcher miss (a solution object is
  right there but the matcher did not pair it with the GT node).
"""

from pathlib import Path

import geff
import numpy as np
import zarr
from numpy import linalg

from mhat.evaluation._report import Report

# Edge attributes and the config keys holding their ILP weight/constant.
EDGE_ATTR_SPECS = [
    ("drift_dist", "drift_weight", "drift_constant"),
    ("area_diff", "area_weight", "area_constant"),
    ("intensity_diff", "intensity_weight", "intensity_constant"),
]

NODE_ATTR_SPECS = [
    ("cohesion", "cohesion_weight", "cohesion_constant"),
    ("adhesion", "adhesion_weight", "adhesion_constant"),
]


def spatial_axis_names(metadata, default=("z", "y", "x")):
    """Spatial axis names (in array order) from geff metadata."""
    axes = getattr(metadata, "axes", None)
    if not axes:
        return list(default)
    names = [a.name for a in axes if getattr(a, "type", None) != "time"]
    return names or list(default)


def _node_pos(ndata, names):
    """Position of a raw geff node as an array in spatial-axis order."""
    return np.array([float(ndata[n]) for n in names])


def _stats(arr):
    """(mean, std, median) of the finite entries, or None if there are none."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean()), float(arr.std()), float(np.median(arr))


def _summarize(attrs):
    """{attr: {mean, std, median, n}} for a dict of attribute arrays."""
    out = {}
    for key, values in attrs.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        out[key] = {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "median": float(np.median(finite)),
            "n": int(finite.size),
        }
    return out


# ---------------------------------------------------------------------------
# FN edges
# ---------------------------------------------------------------------------


def classify_gt_edges(matched, pred_data_dir):
    """Split GT edges into TP / FN-with-both-nodes-matched / FN-node-unmatched.

    Returns a dict with the three lists plus the candidate-graph split of the
    "both nodes matched" FN edges (requires ``candidate_edges.npy``, written by
    ``run_tracking.py``).
    """
    gt_graph = matched.gt_graph.graph
    pred_graph = matched.pred_graph.graph

    tp_edges = []  # (gt_u, gt_v, pred_u, pred_v)
    fn_edges_both_matched = []  # both endpoints matched, but no pred edge
    fn_edges_node_unmatched = []  # (gt_u, gt_v), at least one endpoint unmatched

    for gt_u, gt_v in gt_graph.edges():
        pred_u = matched.get_gt_pred_match(gt_u)
        pred_v = matched.get_gt_pred_match(gt_v)

        if pred_u is None or pred_v is None:
            fn_edges_node_unmatched.append((gt_u, gt_v))
        elif pred_graph.has_edge(pred_u, pred_v):
            tp_edges.append((gt_u, gt_v, pred_u, pred_v))
        else:
            fn_edges_both_matched.append((gt_u, gt_v, pred_u, pred_v))

    # Split the FN edges by whether they were ever in the candidate graph.
    cand_edges_path = Path(pred_data_dir) / "candidate_edges.npy"
    if cand_edges_path.exists():
        cand_edges = np.load(cand_edges_path)
        cand_edge_set = set(map(tuple, cand_edges))
        fn_in_cand = [e for e in fn_edges_both_matched if (e[2], e[3]) in cand_edge_set]
        fn_not_in_cand = [e for e in fn_edges_both_matched if (e[2], e[3]) not in cand_edge_set]
        have_candidates = True
    else:
        fn_in_cand = fn_edges_both_matched
        fn_not_in_cand = []
        have_candidates = False

    return {
        "tp_edges": tp_edges,
        "fn_edges_both_matched": fn_edges_both_matched,
        "fn_edges_node_unmatched": fn_edges_node_unmatched,
        "fn_in_cand": fn_in_cand,
        "fn_not_in_cand": fn_not_in_cand,
        "have_candidates": have_candidates,
    }


def collect_tp_attrs(tp_edges, raw_pred_graph, names):
    """Edge attributes of TP edges, read off the solution graph."""
    attrs = {"drift_dist": [], "area_diff": [], "intensity_diff": [],
             "node_dist": [], "flow_mag": []}
    for _, _, pred_u, pred_v in tp_edges:
        edata = raw_pred_graph[pred_u][pred_v] if raw_pred_graph.has_edge(pred_u, pred_v) else {}
        for attr in ("drift_dist", "area_diff", "intensity_diff"):
            attrs[attr].append(edata.get(attr, np.nan))

        nu = raw_pred_graph.nodes[pred_u]
        nv = raw_pred_graph.nodes[pred_v]
        attrs["node_dist"].append(linalg.norm(_node_pos(nv, names) - _node_pos(nu, names)))
        attrs["flow_mag"].append(linalg.norm(nu["flow"]) if "flow" in nu else np.nan)

    return {k: np.array(v, dtype=float) for k, v in attrs.items()}


def collect_fn_attrs(fn_edges, raw_pred_graph, names):
    """Edge attributes of FN edges, recomputed from the pred node properties.

    The edge does not exist in the solution graph, so drift/area/intensity are
    reconstructed the same way the candidate graph would have computed them.
    """
    attrs = {"node_dist": [], "area_diff": [], "intensity_diff": [],
             "flow_mag": [], "drift_dist": []}
    for _, _, pred_u, pred_v in fn_edges:
        nu = raw_pred_graph.nodes[pred_u]
        nv = raw_pred_graph.nodes[pred_v]
        pos_u = _node_pos(nu, names)
        pos_v = _node_pos(nv, names)

        attrs["node_dist"].append(linalg.norm(pos_v - pos_u))

        if "area" in nu and "area" in nv:
            mean_area = (nu["area"] + nv["area"]) / 2
            attrs["area_diff"].append(
                abs(nu["area"] - nv["area"]) / mean_area if mean_area > 0 else 0
            )
        else:
            attrs["area_diff"].append(np.nan)

        if "intensity" in nu and "intensity" in nv:
            attrs["intensity_diff"].append(abs(nu["intensity"] - nv["intensity"]))
        else:
            attrs["intensity_diff"].append(np.nan)

        if "flow" in nu:
            flow_u = np.array(nu["flow"], dtype=float)
            attrs["flow_mag"].append(linalg.norm(flow_u))
            attrs["drift_dist"].append(linalg.norm(pos_u + flow_u - pos_v))
        else:
            attrs["flow_mag"].append(np.nan)
            attrs["drift_dist"].append(linalg.norm(pos_v - pos_u))

    return {k: np.array(v, dtype=float) for k, v in attrs.items()}


def compute_edge_ilp_cost(run_config, graph, u, v):
    """Total ILP edge cost for edge (u, v) under the tracking config weights."""
    edata = graph[u][v] if graph.has_edge(u, v) else {}
    cost = 0.0
    for attr, w_key, c_key in EDGE_ATTR_SPECS:
        w = run_config.get(w_key, 0.0)
        c = run_config.get(c_key, 0.0)
        if w != 0 and attr in edata:
            cost += w * edata[attr] + c
    return cost


def compute_node_ilp_cost(run_config, graph, node):
    """Total ILP node cost for a node under the tracking config weights."""
    ndata = graph.nodes[node]
    cost = 0.0
    for attr, w_key, c_key in NODE_ATTR_SPECS:
        w = run_config.get(w_key, 0.0)
        c = run_config.get(c_key, 0.0)
        if w != 0 and attr in ndata:
            cost += (w * ndata[attr] + c) * ndata.get("num_leaves", 1)
    return cost


def _append_edge_attrs(attrs, graph, u, v, names):
    edata = graph[u][v] if graph.has_edge(u, v) else {}
    for attr in ("drift_dist", "area_diff", "intensity_diff"):
        attrs[attr].append(edata.get(attr, np.nan))

    nu = graph.nodes[u]
    nv = graph.nodes[v]
    attrs["node_dist"].append(linalg.norm(_node_pos(nv, names) - _node_pos(nu, names)))
    attrs["flow_mag"].append(linalg.norm(nu["flow"]) if "flow" in nu else np.nan)


def collect_substitute_attrs(fn_edges, raw_pred_graph, run_config, names, report,
                             have_candidates=True):
    """What the solution picked *instead* of each ILP-rejected FN edge.

    For every FN edge (pred_u, pred_v) this collects the outgoing edge actually
    taken from pred_u and the incoming edge actually taken into pred_v, along
    with the ILP costs of both, so the two can be compared directly.
    """
    outgoing = {"drift_dist": [], "area_diff": [], "intensity_diff": [],
                "node_dist": [], "flow_mag": [], "edge_cost": [], "target_node_cost": []}
    incoming = {"drift_dist": [], "area_diff": [], "intensity_diff": [],
                "node_dist": [], "flow_mag": [], "edge_cost": [], "source_node_cost": []}
    fn_costs = {"edge_cost": [], "target_node_cost": [], "source_node_cost": []}
    n_no_outgoing = 0
    n_no_incoming = 0

    for _, _, pred_u, pred_v in fn_edges:
        # Cost the FN edge would have had (its attributes are recomputed since
        # the edge is absent from the solution graph).
        nu = raw_pred_graph.nodes[pred_u]
        nv = raw_pred_graph.nodes[pred_v]
        pos_u = _node_pos(nu, names)
        pos_v = _node_pos(nv, names)
        flow_u = np.array(nu.get("flow", np.zeros(len(names))), dtype=float)
        fn_drift = linalg.norm(pos_u + flow_u - pos_v)
        fn_edge_cost = run_config.get("drift_weight", 0) * fn_drift + run_config.get("drift_constant", 0)
        fn_costs["edge_cost"].append(fn_edge_cost)
        fn_costs["target_node_cost"].append(compute_node_ilp_cost(run_config, raw_pred_graph, pred_v))
        fn_costs["source_node_cost"].append(compute_node_ilp_cost(run_config, raw_pred_graph, pred_u))

        out_edges = list(raw_pred_graph.successors(pred_u))
        if out_edges:
            succ = out_edges[0]
            _append_edge_attrs(outgoing, raw_pred_graph, pred_u, succ, names)
            outgoing["edge_cost"].append(compute_edge_ilp_cost(run_config, raw_pred_graph, pred_u, succ))
            outgoing["target_node_cost"].append(compute_node_ilp_cost(run_config, raw_pred_graph, succ))
        else:
            n_no_outgoing += 1

        in_edges = list(raw_pred_graph.predecessors(pred_v))
        if in_edges:
            pred = in_edges[0]
            _append_edge_attrs(incoming, raw_pred_graph, pred, pred_v, names)
            incoming["edge_cost"].append(compute_edge_ilp_cost(run_config, raw_pred_graph, pred, pred_v))
            incoming["source_node_cost"].append(compute_node_ilp_cost(run_config, raw_pred_graph, pred))
        else:
            n_no_incoming += 1

    label = "ILP-rejected FN edges" if have_candidates else "FN edges"
    report.line("")
    report.line(f"Substitute edge analysis ({len(fn_edges)} {label}):")
    report.line(f"  pred_u has no outgoing edge (track ends): {n_no_outgoing}/{len(fn_edges)}")
    report.line(f"  pred_v has no incoming edge (track starts): {n_no_incoming}/{len(fn_edges)}")

    return {
        "outgoing": {k: np.array(v, dtype=float) for k, v in outgoing.items()},
        "incoming": {k: np.array(v, dtype=float) for k, v in incoming.items()},
        "fn_costs": {k: np.array(v, dtype=float) for k, v in fn_costs.items()},
        "n_no_outgoing": n_no_outgoing,
        "n_no_incoming": n_no_incoming,
    }


def _print_attr_comparison(report, tp_attrs, fn_attrs, tp_gt_dists, fn_gt_dists):
    report.line("")
    report.line("=" * 90)
    report.line(f"Edge Attribute Comparison: TP ({len(tp_gt_dists)}) vs FN ({len(fn_gt_dists)}) edges")
    report.line("=" * 90)
    report.line(f"{'Attribute':<20} {'TP Mean':>10} {'TP Std':>10} {'TP Med':>10} "
                f"{'FN Mean':>10} {'FN Std':>10} {'FN Med':>10}")
    report.line("-" * 90)

    def row(label, tp_vals, fn_vals):
        tp_s = _stats(tp_vals)
        fn_s = _stats(fn_vals)
        if tp_s is None or fn_s is None:
            return
        report.line(f"{label:<20} {tp_s[0]:>10.3f} {tp_s[1]:>10.3f} {tp_s[2]:>10.3f} "
                    f"{fn_s[0]:>10.3f} {fn_s[1]:>10.3f} {fn_s[2]:>10.3f}")

    row("GT node dist", tp_gt_dists, fn_gt_dists)
    row("Pred node dist", tp_attrs.get("node_dist", []), fn_attrs.get("node_dist", []))
    row("Drift dist", tp_attrs.get("drift_dist", []), fn_attrs.get("drift_dist", []))
    for attr in ("area_diff", "intensity_diff"):
        row(attr, tp_attrs.get(attr, []), fn_attrs.get(attr, []))
    row("Flow magnitude", tp_attrs.get("flow_mag", []), fn_attrs.get("flow_mag", []))
    report.line("=" * 90)


def _print_substitute_comparison(report, substitute_attrs, fn_ilp_attrs, n_fn_both_matched,
                                 have_candidates=True):
    out = substitute_attrs["outgoing"]
    inc = substitute_attrs["incoming"]
    n_out = len(out["node_dist"])
    n_inc = len(inc["node_dist"])
    if n_out == 0 and n_inc == 0:
        return

    n_considered = n_out + substitute_attrs["n_no_outgoing"]
    label = "ILP-rejected FN edges" if have_candidates else "FN edges"
    report.line("")
    report.line("=" * 100)
    report.line(f"Substitute Edges: what the solution picked INSTEAD of the {n_considered} {label}")
    report.line("=" * 100)
    report.line(f"{'Attribute':<20} {'FN Mean':>10} {'Out Mean':>10} {'Out Std':>10} {'Out Med':>10} "
                f"{'In Mean':>10} {'In Std':>10} {'In Med':>10}")
    report.line("-" * 100)

    def fmt(value):
        return f"{value:.3f}" if value is not None else "N/A"

    for attr in ("node_dist", "drift_dist", "area_diff", "intensity_diff", "flow_mag"):
        fn_s = _stats(fn_ilp_attrs.get(attr, []))
        out_s = _stats(out.get(attr, []))
        in_s = _stats(inc.get(attr, []))
        report.line(
            f"{attr:<20} {fmt(fn_s[0] if fn_s else None):>10} "
            f"{fmt(out_s[0] if out_s else None):>10} {fmt(out_s[1] if out_s else None):>10} "
            f"{fmt(out_s[2] if out_s else None):>10} "
            f"{fmt(in_s[0] if in_s else None):>10} {fmt(in_s[1] if in_s else None):>10} "
            f"{fmt(in_s[2] if in_s else None):>10}"
        )

    report.line("=" * 100)
    report.line(f"  pred_u has no outgoing edge (track ends): {substitute_attrs['n_no_outgoing']}/{n_fn_both_matched}")
    report.line(f"  pred_v has no incoming edge (track starts): {substitute_attrs['n_no_incoming']}/{n_fn_both_matched}")


def _print_cost_comparison(report, substitute_attrs):
    fn_c = substitute_attrs["fn_costs"]
    out_c = substitute_attrs["outgoing"]
    inc_c = substitute_attrs["incoming"]
    if len(fn_c.get("edge_cost", [])) == 0 or len(out_c.get("edge_cost", [])) == 0:
        return

    fn_ec = fn_c["edge_cost"]
    out_ec = out_c.get("edge_cost", np.array([]))
    inc_ec = inc_c.get("edge_cost", np.array([]))
    fn_tnc = fn_c.get("target_node_cost", np.array([]))
    fn_snc = fn_c.get("source_node_cost", np.array([]))
    out_tnc = out_c.get("target_node_cost", np.array([]))
    inc_snc = inc_c.get("source_node_cost", np.array([]))

    report.line("")
    report.line("=" * 100)
    report.line("ILP Cost Comparison: FN edge (correct) vs Substitute edges")
    report.line("=" * 100)
    report.line(f"{'Cost component':<25} {'FN Mean':>12} {'FN Med':>12} {'Sub Out Mean':>12} "
                f"{'Sub Out Med':>12} {'Sub In Mean':>12} {'Sub In Med':>12}")
    report.line("-" * 100)

    def _fmt(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return ("N/A", "N/A")
        return (f"{arr.mean():.1f}", f"{np.median(arr):.1f}")

    fm, fmd = _fmt(fn_ec)
    om, omd = _fmt(out_ec)
    im, imd = _fmt(inc_ec)
    report.line(f"{'Edge cost':<25} {fm:>12} {fmd:>12} {om:>12} {omd:>12} {im:>12} {imd:>12}")

    fm, fmd = _fmt(fn_tnc)
    om, omd = _fmt(out_tnc)
    report.line(f"{'Target node cost':<25} {fm:>12} {fmd:>12} {om:>12} {omd:>12} {'-':>12} {'-':>12}")

    fm, fmd = _fmt(fn_snc)
    im, imd = _fmt(inc_snc)
    report.line(f"{'Source node cost':<25} {fm:>12} {fmd:>12} {'-':>12} {'-':>12} {im:>12} {imd:>12}")

    if len(fn_ec) and len(out_ec) and len(fn_tnc) and len(out_tnc):
        fn_total_out = fn_ec + fn_tnc
        out_total = out_ec + out_tnc
        report.line(f"{'Edge+target (out)':<25} {fn_total_out.mean():>12.1f} {np.median(fn_total_out):>12.1f} "
                    f"{out_total.mean():>12.1f} {np.median(out_total):>12.1f} {'-':>12} {'-':>12}")

    if len(fn_ec) and len(inc_ec) and len(fn_snc) and len(inc_snc):
        fn_total_in = fn_ec + fn_snc
        inc_total = inc_ec + inc_snc
        report.line(f"{'Edge+source (in)':<25} {fn_total_in.mean():>12.1f} {np.median(fn_total_in):>12.1f} "
                    f"{'-':>12} {'-':>12} {inc_total.mean():>12.1f} {np.median(inc_total):>12.1f}")

    report.line("=" * 100)


def plot_edge_histograms(tp_attrs, fn_ilp_attrs, substitute_attrs, run_config,
                         output_dir, raw_pred_graph, tp_pred_edges):
    """Save per-attribute histograms of correct vs incorrect vs substitute edges."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Attributes of every solution edge that is not a TP.
    incorrect_attrs = {attr: [] for attr, _, _ in EDGE_ATTR_SPECS}
    for u, v in raw_pred_graph.edges():
        if (u, v) in tp_pred_edges:
            continue
        edata = raw_pred_graph[u][v]
        for attr, _, _ in EDGE_ATTR_SPECS:
            incorrect_attrs[attr].append(edata.get(attr, np.nan))
    incorrect_attrs = {k: np.array(v, dtype=float) for k, v in incorrect_attrs.items()}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for attr, w_key, c_key in EDGE_ATTR_SPECS:
        weight = run_config.get(w_key, 0.0)
        constant = run_config.get(c_key, 0.0)

        correct = np.concatenate([tp_attrs.get(attr, np.array([])),
                                  fn_ilp_attrs.get(attr, np.array([]))])
        correct = correct[np.isfinite(correct)]
        if correct.size == 0:
            continue

        sub = substitute_attrs["outgoing"].get(attr, np.array([]))
        sub = sub[np.isfinite(sub)]
        incorr = incorrect_attrs.get(attr, np.array([]))
        incorr = incorr[np.isfinite(incorr)]

        # --- Histogram 1: attribute value ---
        all_vals = np.concatenate([a for a in (correct, incorr, sub) if a.size > 0])
        bins = np.histogram_bin_edges(all_vals, bins=30)
        bw = (bins[1] - bins[0]) * 0.3

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(bins[:-1], np.histogram(correct, bins=bins)[0], width=bw, align="edge",
               alpha=0.8, label=f"Correct (n={len(correct)})", color="steelblue")
        if incorr.size:
            ax.bar(bins[:-1] + bw, np.histogram(incorr, bins=bins)[0], width=bw, align="edge",
                   alpha=0.8, label=f"All incorrect (n={len(incorr)})", color="gray")
        if sub.size:
            ax.bar(bins[:-1] + 2 * bw, np.histogram(sub, bins=bins)[0], width=bw, align="edge",
                   alpha=0.8, label=f"Substitute (n={len(sub)})", color="salmon")

        if weight != 0:
            breakeven_attr = -constant / weight
            ax.axvline(breakeven_attr, color="black", linestyle="--", linewidth=1.5,
                       label=f"Break-even ({breakeven_attr:.1f})")

        ax.set_xlabel(attr)
        ax.set_ylabel("# of edges")
        ax.set_title(f"{attr} distribution: correct vs incorrect vs substitute")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"hist_{attr}_value.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

        # --- Histogram 2: cost value (only meaningful for active attributes) ---
        if weight == 0:
            continue

        correct_cost = weight * correct + constant
        sub_cost = weight * sub + constant
        incorr_cost = weight * incorr + constant

        all_costs = np.concatenate([a for a in (correct_cost, incorr_cost, sub_cost) if a.size > 0])
        bins_c = np.histogram_bin_edges(all_costs, bins=30)
        bw_c = (bins_c[1] - bins_c[0]) * 0.3

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(bins_c[:-1], np.histogram(correct_cost, bins=bins_c)[0], width=bw_c, align="edge",
               alpha=0.8, label=f"Correct (n={len(correct_cost)})", color="steelblue")
        if incorr_cost.size:
            ax.bar(bins_c[:-1] + bw_c, np.histogram(incorr_cost, bins=bins_c)[0], width=bw_c,
                   align="edge", alpha=0.8, label=f"All incorrect (n={len(incorr_cost)})", color="gray")
        if sub_cost.size:
            ax.bar(bins_c[:-1] + 2 * bw_c, np.histogram(sub_cost, bins=bins_c)[0], width=bw_c,
                   align="edge", alpha=0.8, label=f"Substitute (n={len(sub_cost)})", color="salmon")

        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Break-even (cost=0)")
        ax.set_xlabel(f"{attr} cost (w={weight}, c={constant})")
        ax.set_ylabel("# of edges")
        ax.set_title(f"{attr} cost distribution: correct vs incorrect vs substitute")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"hist_{attr}_cost.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

    return written


def analyze_fn_edges(matched, pred_data_dir, run_config, output_dir=None, plots=True,
                     echo=True):
    """Compare missed (FN) GT edges against matched (TP) and substitute edges.

    Args:
        matched: traccuracy ``Matched`` object for this run.
        pred_data_dir (Path): Tracking result directory (holds ``pred_tracks.zarr``,
            and optionally ``candidate_edges.npy``).
        run_config (dict): The tracking config used for the run, for ILP costs.
        output_dir (Path | None): Where to write ``fn_edges_report.txt`` and the
            histogram PNGs. Nothing is written if None.
        plots (bool): Write the histogram PNGs.
        echo (bool): Print the report to stdout as it is built.

    Returns:
        dict: Machine-readable summary of the classification and attribute stats.
    """
    pred_data_dir = Path(pred_data_dir)
    raw_pred_graph, metadata = geff.read(pred_data_dir / "pred_tracks.zarr")
    names = spatial_axis_names(metadata)

    report = Report("FN edge analysis", echo=echo)

    classes = classify_gt_edges(matched, pred_data_dir)
    tp_edges = classes["tp_edges"]
    fn_both = classes["fn_edges_both_matched"]
    fn_unmatched = classes["fn_edges_node_unmatched"]
    fn_in_cand = classes["fn_in_cand"]
    fn_not_in_cand = classes["fn_not_in_cand"]
    total_gt_edges = len(tp_edges) + len(fn_both) + len(fn_unmatched)

    report.line("")
    report.line("Edge Classification:")
    report.line(f"  TP edges: {len(tp_edges)}")
    report.line(f"  FN edges (both nodes matched, edge missing): {len(fn_both)}")
    if classes["have_candidates"]:
        report.line(f"    - Were candidates but ILP rejected: {len(fn_in_cand)}")
        report.line(f"    - Never candidates (graph construction): {len(fn_not_in_cand)}")
    else:
        report.line("    - candidate_edges.npy not found; cannot split ILP-rejected "
                    "vs never-a-candidate")
    report.line(f"  FN edges (node unmatched): {len(fn_unmatched)}")
    report.line(f"  Total GT edges: {total_gt_edges}")

    tp_attrs = collect_tp_attrs(tp_edges, raw_pred_graph, names)
    fn_attrs = collect_fn_attrs(fn_both, raw_pred_graph, names)
    fn_ilp_attrs = collect_fn_attrs(fn_in_cand, raw_pred_graph, names)

    gt_graph = matched.gt_graph.graph
    tp_gt_dists = np.array([
        linalg.norm(np.array(gt_graph.nodes[v]["pos"]) - np.array(gt_graph.nodes[u]["pos"]))
        for u, v, _, _ in tp_edges
    ])
    fn_gt_dists = np.array([
        linalg.norm(np.array(gt_graph.nodes[v]["pos"]) - np.array(gt_graph.nodes[u]["pos"]))
        for u, v, _, _ in fn_both
    ])

    substitute_attrs = collect_substitute_attrs(fn_in_cand, raw_pred_graph, run_config,
                                                names, report,
                                                have_candidates=classes["have_candidates"])

    _print_attr_comparison(report, tp_attrs, fn_attrs, tp_gt_dists, fn_gt_dists)
    _print_substitute_comparison(report, substitute_attrs, fn_ilp_attrs, len(fn_both),
                                 have_candidates=classes["have_candidates"])
    _print_cost_comparison(report, substitute_attrs)

    summary = {
        "n_tp_edges": len(tp_edges),
        "n_fn_edges_both_matched": len(fn_both),
        "n_fn_edges_ilp_rejected": len(fn_in_cand) if classes["have_candidates"] else None,
        "n_fn_edges_never_candidate": len(fn_not_in_cand) if classes["have_candidates"] else None,
        "n_fn_edges_node_unmatched": len(fn_unmatched),
        "n_gt_edges": total_gt_edges,
        "n_fn_track_ends": substitute_attrs["n_no_outgoing"],
        "n_fn_track_starts": substitute_attrs["n_no_incoming"],
        "tp_attrs": _summarize(tp_attrs),
        "fn_attrs": _summarize(fn_attrs),
        "substitute_outgoing": _summarize(substitute_attrs["outgoing"]),
        "substitute_incoming": _summarize(substitute_attrs["incoming"]),
        "fn_edge_costs": _summarize(substitute_attrs["fn_costs"]),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        report.write(output_dir / "fn_edges_report.txt")
        if plots:
            tp_pred_edges = {(pu, pv) for _, _, pu, pv in tp_edges}
            summary["plots"] = plot_edge_histograms(
                tp_attrs, fn_ilp_attrs, substitute_attrs, run_config,
                output_dir, raw_pred_graph, tp_pred_edges,
            )
            report.line("")
            report.line(f"Histograms saved to {output_dir}")
            report.write(output_dir / "fn_edges_report.txt")

    return summary


# ---------------------------------------------------------------------------
# FN nodes
# ---------------------------------------------------------------------------


def _label_centroids(frame, scale_spatial):
    """Centroids (in world units) of every non-zero label in a single frame.

    Returns (labels, centroids) with centroids shaped (n_labels, ndim).
    """
    from scipy.ndimage import center_of_mass

    frame = np.asarray(frame)
    labels = np.unique(frame)
    labels = labels[labels != 0]
    if labels.size == 0:
        return labels, np.empty((0, frame.ndim))
    centroids = center_of_mass(np.ones_like(frame, dtype=np.uint8), labels=frame,
                               index=labels)
    return labels, np.asarray(centroids, dtype=float) * np.asarray(scale_spatial, dtype=float)


def _nearest(labels, centroids, pos):
    """Nearest (label, distance) to pos, or (None, inf) if there are none."""
    if len(labels) == 0:
        return None, float("inf")
    dists = linalg.norm(centroids - np.asarray(pos, dtype=float), axis=1)
    idx = int(np.argmin(dists))
    return int(labels[idx]), float(dists[idx])


def analyze_fn_nodes(matched, config, pred_data_dir, run_config, output_dir=None,
                     proximity_threshold=None, max_listed=None, echo=True):
    """Classify unmatched GT nodes as segmentation / solver / matcher misses.

    For each GT node with no predicted match, this looks for the nearest
    candidate fragment (the full segmentation hypothesis set) and the nearest
    object in the solution segmentation:

    - a solution object nearby   -> matcher miss (selected, but not matched)
    - only a fragment nearby     -> solver miss (candidate existed, not selected)
    - neither nearby             -> segmentation miss

    Args:
        matched: traccuracy ``Matched`` object for this run.
        config (dict): Evaluation config (for ``input_base_dir``/``experiment``/``dataset``).
        pred_data_dir (Path): Tracking result directory.
        run_config (dict): Tracking config of the run (for ``seg_result``).
        output_dir (Path | None): Where to write ``fn_nodes_report.txt``.
        proximity_threshold (float | None): Max distance (world units) for an
            object to count as "at the GT location". Defaults to the eval config's
            ``fn_proximity_threshold``, else ``match_threshold``, else 20.
        max_listed (int | None): Cap on the number of FN nodes listed individually.
        echo (bool): Print the report to stdout as it is built.

    Returns:
        dict: Counts per class plus the per-node details.
    """
    pred_data_dir = Path(pred_data_dir)
    if proximity_threshold is None:
        proximity_threshold = float(
            config.get("fn_proximity_threshold",
                       config.get("match_threshold", config.get("threshold", 20.0)) or 20.0)
        )

    report = Report("FN node analysis", echo=echo)

    gt_graph = matched.gt_graph.graph
    pred_graph = matched.pred_graph.graph

    fn_nodes = [n for n in gt_graph.nodes() if matched.get_gt_pred_match(n) is None]

    report.line("")
    report.line(f"Total GT nodes: {gt_graph.number_of_nodes()}")
    report.line(f"Unmatched GT nodes (FN): {len(fn_nodes)}")
    report.line(f"Proximity threshold: {proximity_threshold} (world units)")

    if not fn_nodes:
        report.line("No unmatched GT nodes - nothing to classify.")
        if output_dir is not None:
            report.write(Path(output_dir) / "fn_nodes_report.txt")
        return {"n_gt_nodes": gt_graph.number_of_nodes(), "n_fn_nodes": 0,
                "n_segmentation_miss": 0, "n_solver_miss": 0, "n_matcher_miss": 0,
                "n_unclassified": 0, "proximity_threshold": proximity_threshold,
                "fn_nodes": []}

    # Only the timepoints holding FN nodes need their centroids computed.
    fn_by_time = {}
    for node in fn_nodes:
        fn_by_time.setdefault(int(gt_graph.nodes[node]["time"]), []).append(node)
    timepoints = sorted(fn_by_time)

    # Candidate fragments (all hypotheses, not just the selected ones).
    fragments = None
    seg_result = run_config.get("seg_result")
    if seg_result:
        seg_zarr_path = (Path(config["input_base_dir"]) / "segmentation" /
                         config["experiment"] / config["dataset"] / seg_result / "data.zarr")
        try:
            fragments = zarr.open(str(seg_zarr_path), mode="r")["fragments"]
        except (KeyError, FileNotFoundError, ValueError) as e:
            report.line(f"Could not open candidate fragments at {seg_zarr_path}: {e}")
    else:
        report.line("No seg_result in the tracking config; skipping fragment lookup "
                    "(solver vs segmentation misses cannot be separated).")

    # Solution segmentation.
    solution_seg = None
    pred_seg_path = pred_data_dir / "pred_seg.zarr"
    if pred_seg_path.exists():
        solution_seg = zarr.open(str(pred_seg_path), mode="r")
    else:
        report.line(f"No {pred_seg_path}; falling back to solution node positions "
                    "for the matcher-miss test.")

    scale_spatial = None
    if solution_seg is not None or fragments is not None:
        arr = solution_seg if solution_seg is not None else fragments
        ndim = arr.ndim - 1  # drop time
        pos_len = len(gt_graph.nodes[fn_nodes[0]]["pos"])
        # The GT positions are already in world units; recover the spatial scale
        # from the geff axes of the prediction.
        _, metadata = geff.read(pred_data_dir / "pred_tracks.zarr")
        axes_scale = [a.scale if a.scale is not None else 1.0 for a in (metadata.axes or [])]
        scale_spatial = axes_scale[-ndim:] if len(axes_scale) >= ndim else [1.0] * ndim
        if pos_len != ndim:
            report.line(f"GT positions are {pos_len}D but segmentation is {ndim}D; "
                        "skipping segmentation-based classification.")
            fragments = solution_seg = None

    # Predicted (solution graph) node positions per timepoint.
    pred_pos_by_time = {}
    for pnode, pdata in pred_graph.nodes(data=True):
        pred_pos_by_time.setdefault(int(pdata["time"]), []).append((pnode, np.array(pdata["pos"])))

    seg_miss, solver_miss, matcher_miss, unclassified = [], [], [], []

    for t in timepoints:
        frag_labels, frag_centroids = (np.array([]), np.empty((0, 1)))
        sol_labels, sol_centroids = (np.array([]), np.empty((0, 1)))
        if fragments is not None and t < fragments.shape[0]:
            frag_labels, frag_centroids = _label_centroids(fragments[t], scale_spatial)
        if solution_seg is not None and t < solution_seg.shape[0]:
            sol_labels, sol_centroids = _label_centroids(solution_seg[t], scale_spatial)

        pred_entries = pred_pos_by_time.get(t, [])
        pred_ids = [p[0] for p in pred_entries]
        pred_positions = np.array([p[1] for p in pred_entries]) if pred_entries else np.empty((0, 1))

        for node in fn_by_time[t]:
            pos = np.array(gt_graph.nodes[node]["pos"], dtype=float)

            if len(pred_ids):
                dists = linalg.norm(pred_positions - pos, axis=1)
                idx = int(np.argmin(dists))
                nearest_pred_node, nearest_pred_dist = pred_ids[idx], float(dists[idx])
            else:
                nearest_pred_node, nearest_pred_dist = None, float("inf")

            nearest_frag_lab, nearest_frag_dist = _nearest(frag_labels, frag_centroids, pos)
            nearest_sol_lab, nearest_sol_dist = _nearest(sol_labels, sol_centroids, pos)

            # Without a solution segmentation, use the solution graph positions.
            sol_dist = nearest_sol_dist if solution_seg is not None else nearest_pred_dist

            info = {
                "node": node,
                "time": t,
                "track_id": gt_graph.nodes[node].get("track_id"),
                "pos": [float(p) for p in pos],
                "nearest_pred_node": nearest_pred_node,
                "nearest_pred_dist": nearest_pred_dist,
                "nearest_frag_label": nearest_frag_lab,
                "nearest_frag_dist": nearest_frag_dist,
                "nearest_sol_label": nearest_sol_lab,
                "nearest_sol_dist": nearest_sol_dist,
            }

            if sol_dist < proximity_threshold:
                info["classification"] = "matcher_miss"
                matcher_miss.append(info)
            elif fragments is None:
                # Without the candidate fragments we cannot tell a segmentation
                # miss from a solver miss.
                info["classification"] = "unclassified"
                unclassified.append(info)
            elif nearest_frag_dist < proximity_threshold:
                info["classification"] = "solver_miss"
                solver_miss.append(info)
            else:
                info["classification"] = "segmentation_miss"
                seg_miss.append(info)

    report.line("")
    report.line("=== FN Node Classification ===")
    report.line(f"  Segmentation miss (no candidate at GT location): {len(seg_miss)}")
    report.line(f"  Solver miss (candidate exists, not selected):    {len(solver_miss)}")
    report.line(f"  Matcher miss (selected, not matched):            {len(matcher_miss)}")
    if unclassified:
        report.line(f"  Unclassified (no candidate fragments available):  {len(unclassified)}")

    def _section(title, items):
        if not items:
            return
        report.line("")
        report.line(f"--- {title} ---")
        report.line(f"  {'Node':<10} {'t':<5} {'Track':<8} {'Frag Dist':>10} {'Sol Dist':>10} "
                    f"{'Pred Dist':>10}  {'Frag':<10} {'Sol':<10} {'Pred'}")
        report.line("  " + "-" * 100)
        listed = items if max_listed is None else items[:max_listed]
        for info in listed:
            report.line(
                f"  {str(info['node']):<10} {info['time']:<5} {str(info['track_id']):<8} "
                f"{info['nearest_frag_dist']:>10.1f} {info['nearest_sol_dist']:>10.1f} "
                f"{info['nearest_pred_dist']:>10.1f}  "
                f"{str(info['nearest_frag_label']):<10} {str(info['nearest_sol_label']):<10} "
                f"{str(info['nearest_pred_node'])}"
            )
        if max_listed is not None and len(items) > max_listed:
            report.line(f"  ... {len(items) - max_listed} more")

    _section(f"Segmentation misses (nearest fragment >= {proximity_threshold})", seg_miss)
    _section(f"Solver misses (fragment < {proximity_threshold}, solution >= {proximity_threshold})", solver_miss)
    _section(f"Matcher misses (solution object < {proximity_threshold})", matcher_miss)
    _section("Unclassified (candidate fragments unavailable)", unclassified)

    if output_dir is not None:
        report.write(Path(output_dir) / "fn_nodes_report.txt")

    return {
        "n_gt_nodes": gt_graph.number_of_nodes(),
        "n_fn_nodes": len(fn_nodes),
        "n_segmentation_miss": len(seg_miss),
        "n_solver_miss": len(solver_miss),
        "n_matcher_miss": len(matcher_miss),
        "n_unclassified": len(unclassified),
        "proximity_threshold": proximity_threshold,
        "fragments_available": fragments is not None,
        "solution_seg_available": solution_seg is not None,
        "fn_nodes": seg_miss + solver_miss + matcher_miss + unclassified,
    }
