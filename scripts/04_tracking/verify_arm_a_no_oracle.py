"""Prove that an arm-A fit uses no knowledge of the unannotated ground truth.

The stage-2 driver loads the COMPLETE GT and passes a kept-label list, relying on
`assign_gt_labels` to touch only the kept columns. That is an audit claim about code
paths. This script tests it the strong way instead: it derives the fit graph a second
time from the **materialized subset GT alone** — a directory containing only the
annotated tracks, from which the hidden ones are simply absent — and checks the two
agree exactly.

    dense path   load full correct_tracks.zarr, pass keep_gt_labels = the subset's
                 labels (this is what run_gt_amount_experiment.py does)
    subset path  load gt_subsets/n<N>_s<seed>/, pass keep_gt_labels = every label it
                 contains. The unannotated tracks are not on disk, so no code path
                 could consult them even in principle.

If the surviving node set, every `gt_selected` node label, and every `gt_selected`
edge label match, then the dense GT contributed nothing to the fit: same graph, same
labels, same regularizer implies the same learned weights.

Note the two runs use different `track_id` numbering (funtracks renumbers on load, so
a reloaded subset has its own label space). The comparison is therefore over candidate
node ids, which are shared, never over GT track ids.

Usage:
    python scripts/04_tracking/verify_arm_a_no_oracle.py <gt_amount_config.toml> \
        [--subset n050_s0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import toml

from fit_weights_ssvm import load_gt
from mhat.tracking.gt_annotation import assign_gt_labels, compute_gt_overlaps
from mhat.tracking.gt_subsets import copy_track_graph, prune_exclusion_sets, remove_nodes
from mhat.tracking.pipeline import build_track_graph, resolve_input_dirs


def build_arm_a(base_graph, overlaps, gt_graph, keep_labels, iogt, criterion, exclusion_sets):
    """Annotate, mask, and apply arm-A surgery. Returns (graph, exclusion_sets, stats)."""
    graph = copy_track_graph(base_graph)
    stats, masked = assign_gt_labels(
        graph,
        overlaps,
        gt_graph,
        iogt_threshold=iogt,
        keep_gt_labels=keep_labels,
        overlap_criterion=criterion,
    )
    remove_nodes(graph, masked)
    return graph, prune_exclusion_sets(exclusion_sets, graph), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--subset", default="n050_s0", help="gt_subsets/<name> to verify")
    args = parser.parse_args()

    config = toml.load(args.config)
    iogt = config.get("iogt_threshold", 0.5)
    criterion = config.get("mask_overlap_criterion", "any")
    raw_dir, seg_dir, flow_dirs, gt_data_dir = resolve_input_dirs(config)

    output_base_dir = Path(config["output_base_dir"])
    train_root = output_base_dir / "tracking" / config["experiment"] / config["dataset"]
    subset_dir = train_root / config.get("gt_subsets_dirname", "gt_subsets") / args.subset
    if not (subset_dir / "subset.json").is_file():
        raise SystemExit(f"no materialized subset at {subset_dir} (run --materialize-only first)")
    info = json.loads((subset_dir / "subset.json").read_text())
    print(f"subset {args.subset}: {info['n_tracks']}/{info['n_tracks_total']} tracks, "
          f"{info['n_gt_nodes_kept']} GT nodes\n")

    print("Building candidate graph (shared by both paths)...")
    base_graph, fragments, merge_history, exclusion_sets, scale, _axes = build_track_graph(
        config, raw_dir, seg_dir, flow_dirs
    )

    # ---- dense path: full GT + kept-label list -----------------------------
    print("DENSE path: loading the complete GT...")
    dense_graph_gt, dense_seg = load_gt(gt_data_dir, scale)
    dense_overlaps = compute_gt_overlaps(
        base_graph, fragments, dense_graph_gt, dense_seg, merge_history
    )
    keep_labels = set(info["kept_track_ids"])
    dense_fit, dense_excl, dense_stats = build_arm_a(
        base_graph, dense_overlaps, dense_graph_gt, keep_labels, iogt, criterion, exclusion_sets
    )

    # ---- subset path: only the annotated tracks exist on disk --------------
    print("SUBSET path: loading only the annotated tracks...")
    sub_graph_gt, sub_seg = load_gt(subset_dir, scale)
    sub_overlaps = compute_gt_overlaps(
        base_graph, fragments, sub_graph_gt, sub_seg, merge_history
    )
    # every track present is annotated, so keep them all — derivable from this dir alone
    sub_keep = {int(d["track_id"]) for _, d in sub_graph_gt.nodes(data=True)}
    sub_fit, sub_excl, sub_stats = build_arm_a(
        base_graph, sub_overlaps, sub_graph_gt, sub_keep, iogt, criterion, exclusion_sets
    )

    # ---- compare -----------------------------------------------------------
    print("\n--- comparison (by candidate node id; GT track ids are NOT comparable) ---")
    ok = True

    dense_nodes, sub_nodes = set(dense_fit.nodes), set(sub_fit.nodes)
    same_nodes = dense_nodes == sub_nodes
    ok &= same_nodes
    print(f"surviving node set      {len(dense_nodes)} vs {len(sub_nodes)}  "
          f"{'MATCH' if same_nodes else 'DIFFER (+%d/-%d)' % (len(sub_nodes - dense_nodes), len(dense_nodes - sub_nodes))}")

    dense_edges, sub_edges = set(dense_fit.edges), set(sub_fit.edges)
    same_edges = dense_edges == sub_edges
    ok &= same_edges
    print(f"surviving edge set      {len(dense_edges)} vs {len(sub_edges)}  "
          f"{'MATCH' if same_edges else 'DIFFER'}")

    if same_nodes:
        node_diff = [n for n in dense_nodes
                     if dense_fit.nodes[n].get("gt_selected") != sub_fit.nodes[n].get("gt_selected")]
        ok &= not node_diff
        print(f"node gt_selected labels {len(dense_nodes)} compared        "
              f"{'MATCH' if not node_diff else f'DIFFER at {len(node_diff)}'}")
    if same_edges:
        edge_diff = [e for e in dense_edges
                     if dense_fit.edges[e].get("gt_selected") != sub_fit.edges[e].get("gt_selected")]
        ok &= not edge_diff
        print(f"edge gt_selected labels {len(dense_edges)} compared        "
              f"{'MATCH' if not edge_diff else f'DIFFER at {len(edge_diff)}'}")

    same_excl = sorted(map(sorted, dense_excl)) == sorted(map(sorted, sub_excl))
    ok &= same_excl
    print(f"exclusion sets          {len(dense_excl)} vs {len(sub_excl)}  "
          f"{'MATCH' if same_excl else 'DIFFER'}")

    for key in ("n_nodes_pos", "n_nodes_neg", "n_edges_pos", "n_edges_neg"):
        a, b = dense_stats[key], sub_stats[key]
        ok &= a == b
        print(f"  {key:<16}{a:>8} vs {b:<8}{'ok' if a == b else 'DIFFER'}")

    print(
        "\nRESULT: "
        + (
            "PASS — the fit is identical whether or not the unannotated GT is on disk,\n"
            "        so it contributed nothing to the fitted weights."
            if ok
            else "FAIL — the dense GT changes the fit. Arm A is still an oracle."
        )
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
