"""Prove that a dense-crop (arm C) fit uses no knowledge of the GT outside the crop.

The crop arm's sibling gate is `verify_arm_a_no_oracle.py`; this is the same test for
the other protocol. The stage-2 driver loads the COMPLETE GT and derives both the
annotated-label set and the candidate mask from a box, relying on the box being the
only thing that decides either. That is an audit claim about code paths. This script
tests it the strong way: it derives the fit graph a second time from the
**materialized crop GT alone** — a directory in which the outside-the-box objects are
simply absent — and checks the two agree exactly.

    dense path   load the full correct_tracks.zarr, compute keep_gt_labels from the
                 box (this is what run_gt_amount_experiment.py does)
    crop path    load gt_crops/f<FFF>_s<seed>/, and keep every label it contains. The
                 outside-the-box objects are not on disk, so no code path could
                 consult them even in principle.

If the surviving node set, every `gt_selected` node label and every `gt_selected` edge
label match, the outside-the-crop GT contributed nothing: same graph, same labels, same
regularizer implies the same learned weights.

Two extra checks the sparse gate has no analogue for, because they are the properties
that make a *dense* label of 0 honest:

    contained/annotated pairing  no surviving candidate overlaps a GT track the crop
                                 did not annotate (so no `gt_selected = 0` is a lie)
    negatives survive            the crop actually keeps negative candidates, which is
                                 the entire reason this arm exists

Note the two runs use different `track_id` numbering (funtracks renumbers on load, and
a crop can split a track that leaves and re-enters). The comparison is therefore over
candidate node ids, which are shared, never over GT track ids.

Usage:
    python scripts/04_tracking/verify_crop_no_oracle.py <gt_amount_config.toml> \
        [--crop f250_s0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import toml

from fit_weights_ssvm import load_gt
from mhat.tracking.gt_annotation import assign_gt_labels, compute_gt_overlaps
from mhat.tracking.gt_crops import (
    CropBox,
    assert_no_unannotated_overlap,
    build_gt_voxel_index,
    compute_node_bboxes,
    gt_track_labels_in_crop,
    nodes_outside_crop,
)
from mhat.tracking.gt_subsets import copy_track_graph, prune_exclusion_sets, remove_nodes
from mhat.tracking.pipeline import build_track_graph, resolve_input_dirs


def build_crop_fit(
    base_graph, overlaps, gt_graph, keep_labels, outside, iogt, criterion, exclusion_sets
):
    """Annotate with dense in-crop labels, drop the out-of-crop candidates."""
    graph = copy_track_graph(base_graph)
    stats, masked = assign_gt_labels(
        graph,
        overlaps,
        gt_graph,
        iogt_threshold=iogt,
        keep_gt_labels=keep_labels,
        overlap_criterion=criterion,
        mask_untouched=False,
        extra_masked_nodes=outside,
    )
    assert masked == set(outside), "the crop mask is not exactly the out-of-box set"
    remove_nodes(graph, masked)
    return graph, prune_exclusion_sets(exclusion_sets, graph), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument(
        "--crop", default=None, help="gt_crops/<name> to verify, e.g. b0821_s1 or f250_s0"
    )
    args = parser.parse_args()

    config = toml.load(args.config)
    iogt = config.get("iogt_threshold", 0.5)
    criterion = config.get("mask_overlap_criterion", "any")
    membership = config.get("crop_membership", "contained")
    raw_dir, seg_dir, flow_dirs, gt_data_dir = resolve_input_dirs(config)

    output_base_dir = Path(config["output_base_dir"])
    train_root = output_base_dir / "tracking" / config["experiment"] / config["dataset"]
    crops_root = train_root / config.get("gt_crops_dirname", "gt_crops")
    if not crops_root.is_dir():
        raise SystemExit(f"no materialized crops at {crops_root} (run --materialize-only first)")

    name = args.crop
    if name is None:
        available = sorted(
            p.name for p in crops_root.iterdir() if p.is_dir() and (p / "subset.json").is_file()
        )
        if not available:
            raise SystemExit(f"no materialized crops under {crops_root}")
        name = available[len(available) // 2]  # a mid-sized one by name order
        print(f"no --crop given; verifying {name} (of {len(available)} available)")
    crop_dir = crops_root / name
    if not (crop_dir / "subset.json").is_file():
        raise SystemExit(f"no materialized crop at {crop_dir}")

    info = json.loads((crop_dir / "subset.json").read_text())
    box = CropBox.from_dict(info["crop"])
    print(
        f"crop {name}: {box.describe()} ({box.volume_fraction * 100:.1f}% of the volume), "
        f"{info['n_gt_nodes_kept']}/{info['n_gt_nodes_total']} GT nodes, "
        f"{info['n_tracks']}/{info['n_tracks_total']} tracks\n"
    )

    print("Building candidate graph (shared by both paths)...")
    base_graph, fragments, merge_history, exclusion_sets, scale, _axes = build_track_graph(
        config, raw_dir, seg_dir, flow_dirs
    )

    # The candidate mask is pure geometry — identical on both paths by construction,
    # which is exactly the property being demonstrated, so it is computed once.
    node_bboxes = None
    if membership == "contained":
        from mhat.tracking.gt_annotation import build_node_to_fragments

        node_bboxes = compute_node_bboxes(
            base_graph, fragments, build_node_to_fragments(base_graph, merge_history)
        )
    outside = nodes_outside_crop(
        base_graph, box, node_bboxes, membership=membership, scale=scale
    )
    print(
        f"crop excludes {len(outside)}/{len(base_graph.nodes)} candidates "
        f"(membership={membership!r}) — decided from the box alone, no GT consulted"
    )

    # ---- dense path: the complete GT is on disk -----------------------------
    print("\nDENSE path: loading the complete GT...")
    dense_gt, dense_seg = load_gt(gt_data_dir, scale)
    dense_overlaps = compute_gt_overlaps(base_graph, fragments, dense_gt, dense_seg, merge_history)
    dense_keep = gt_track_labels_in_crop(build_gt_voxel_index(dense_seg), box)
    dense_fit, dense_excl, dense_stats = build_crop_fit(
        base_graph, dense_overlaps, dense_gt, dense_keep, outside, iogt, criterion, exclusion_sets
    )

    # ---- crop path: only the in-box objects exist on disk -------------------
    print("CROP path: loading only the annotated (in-box) objects...")
    crop_gt, crop_seg = load_gt(crop_dir, scale)
    crop_overlaps = compute_gt_overlaps(base_graph, fragments, crop_gt, crop_seg, merge_history)
    # Everything present is annotated — derivable from this directory alone.
    crop_keep = {int(d["track_id"]) for _, d in crop_gt.nodes(data=True)}
    crop_fit, crop_excl, crop_stats = build_crop_fit(
        base_graph, crop_overlaps, crop_gt, crop_keep, outside, iogt, criterion, exclusion_sets
    )

    # ---- compare ------------------------------------------------------------
    print("\n--- comparison (by candidate node id; GT track ids are NOT comparable) ---")
    ok = True

    dense_nodes, crop_nodes = set(dense_fit.nodes), set(crop_fit.nodes)
    same_nodes = dense_nodes == crop_nodes
    ok &= same_nodes
    print(
        f"surviving node set      {len(dense_nodes)} vs {len(crop_nodes)}  "
        + ("MATCH" if same_nodes else f"DIFFER (+{len(crop_nodes - dense_nodes)}/"
           f"-{len(dense_nodes - crop_nodes)})")
    )

    dense_edges, crop_edges = set(dense_fit.edges), set(crop_fit.edges)
    same_edges = dense_edges == crop_edges
    ok &= same_edges
    print(f"surviving edge set      {len(dense_edges)} vs {len(crop_edges)}  "
          f"{'MATCH' if same_edges else 'DIFFER'}")

    if same_nodes:
        node_diff = [
            n for n in dense_nodes
            if dense_fit.nodes[n].get("gt_selected") != crop_fit.nodes[n].get("gt_selected")
        ]
        ok &= not node_diff
        print(f"node gt_selected labels {len(dense_nodes)} compared        "
              f"{'MATCH' if not node_diff else f'DIFFER at {len(node_diff)}'}")
    if same_edges:
        edge_diff = [
            e for e in dense_edges
            if dense_fit.edges[e].get("gt_selected") != crop_fit.edges[e].get("gt_selected")
        ]
        ok &= not edge_diff
        print(f"edge gt_selected labels {len(dense_edges)} compared        "
              f"{'MATCH' if not edge_diff else f'DIFFER at {len(edge_diff)}'}")

    same_excl = sorted(map(sorted, dense_excl)) == sorted(map(sorted, crop_excl))
    ok &= same_excl
    print(f"exclusion sets          {len(dense_excl)} vs {len(crop_excl)}  "
          f"{'MATCH' if same_excl else 'DIFFER'}")

    for key in ("n_nodes_pos", "n_nodes_neg", "n_edges_pos", "n_edges_neg"):
        a, b = dense_stats[key], crop_stats[key]
        ok &= a == b
        print(f"  {key:<16}{a:>8} vs {b:<8}{'ok' if a == b else 'DIFFER'}")

    # ---- the two dense-crop-specific properties ----------------------------
    print("\n--- dense-crop properties ---")
    try:
        assert_no_unannotated_overlap(dense_nodes, dense_overlaps, dense_keep)
        print("no surviving candidate overlaps an unannotated GT track   ok")
    except AssertionError as exc:
        ok = False
        print(f"contained/annotated pairing BROKEN: {exc}")

    if dense_stats["n_nodes_neg"] > 0:
        print(
            f"negatives kept inside the crop: {dense_stats['n_nodes_neg']}"
            f" (positives {dense_stats['n_nodes_pos']})                  ok"
        )
    else:
        ok = False
        print("NO negative candidates survived — this is not a dense annotation")

    print(
        "\nRESULT: "
        + (
            "PASS — the fit is identical whether or not the outside-the-crop GT is on\n"
            "        disk, and every in-crop negative is genuinely supported by the\n"
            "        annotation. Arm C is not an oracle."
            if ok
            else "FAIL — the out-of-crop GT changes the fit, or an in-crop negative is\n"
                 "        unsupported. Arm C is an oracle as implemented."
        )
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
