"""Visual sanity check for GT annotation on the candidate graph.

Loads the candidate graph the same way fit_weights_ssvm.py does, runs
annotate_gt_on_candidate_graph, then opens napari with three layers:
- raw: the raw image data
- gt_seg: the (remapped) GT segmentation, labels = track_id
- matched_cand_seg: per-frame, each pixel labeled with the candidate ID of
  the gt_selected=1 candidate that owns it (0 elsewhere)

If the matched_cand layer aligns spatially with gt_seg, the GT→candidate
mapping is sane. Otherwise something is off (frame index, scale, etc.) and
SSVM weights learned on top would be garbage.

Reads the same TOML schema as fit_weights_ssvm.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import toml
import zarr

from mhat.tracking.gt_annotation import (
    assign_gt_labels,
    build_node_to_fragments,
    compute_gt_overlaps,
)
from mhat.tracking.pipeline import build_track_graph, load_gt, resolve_input_dirs


def build_matched_cand_seg(track_graph, fragments, merge_history, nodes=None):
    """Per-pixel image where matched candidate IDs are painted (others 0).

    Matched candidates come from gt_selected=1 nodes. Conflict-resolved
    matching guarantees no two matched candidates share fragments, so the
    result is a valid label image (no overwrite ambiguity).

    Pass `nodes` to paint an explicit node set instead (e.g. the candidates a GT
    subset hides). Those may share fragments, so the image is for eyeballing
    coverage only, not a valid one-label-per-object segmentation.
    """
    node_to_fragments = build_node_to_fragments(track_graph, merge_history)
    matched_seg = np.zeros_like(fragments)
    selected = track_graph.nodes.items() if nodes is None else (
        (n, track_graph.nodes[n]) for n in nodes
    )
    for node, data in selected:
        if nodes is None and data.get("gt_selected") != 1:
            continue
        t = data.get("time")
        if t is None:
            continue
        frag_ids = node_to_fragments[node]
        mask = np.isin(fragments[t], frag_ids)
        matched_seg[t][mask] = node
    return matched_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--subset",
        default=None,
        help="path to a gt_subsets/<n>_<seed>/subset.json — restricts supervision to "
        "that subset and adds a `masked_cand` layer showing the hidden candidates",
    )
    parser.add_argument("--criterion", choices=("any", "iogt"), default=None)
    args = parser.parse_args()
    config = toml.load(args.config)

    # Same directory resolution as fit_weights_ssvm.py, including the
    # rank-dependent flow policy (2D data needs only a 2D flow) and the
    # `gt_data_dir` override. The old Lucas-Kanade branch (`use_lk`) is gone:
    # the fit never supported it and LK flow is dead code.
    raw_dir, seg_dir, flow_dirs, gt_data_dir = resolve_input_dirs(config)

    track_graph, fragments, merge_history, _exclusion_sets, scale, _axes = build_track_graph(
        config, raw_dir, seg_dir, flow_dirs
    )

    print("Loading GT...")
    gt_graph, gt_seg = load_gt(gt_data_dir, scale)

    print("Annotating GT on candidate graph...")
    keep_labels = None
    subset_info = None
    if args.subset:
        subset_info = json.loads(Path(args.subset).read_text())
        keep_labels = set(subset_info["kept_track_ids"])
        print(
            f"  restricted to subset {Path(args.subset).parent.name}: "
            f"{len(keep_labels)}/{subset_info['n_tracks_total']} GT tracks"
        )

    overlaps = compute_gt_overlaps(track_graph, fragments, gt_graph, gt_seg, merge_history)
    stats, masked_nodes = assign_gt_labels(
        track_graph,
        overlaps,
        gt_graph,
        iogt_threshold=config.get("iogt_threshold", 0.5),
        keep_gt_labels=keep_labels,
        overlap_criterion=(
            args.criterion
            or (subset_info or {}).get("overlap_criterion")
            or config.get("mask_overlap_criterion", "any")
        ),
    )
    print("GT annotation stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("Building matched-candidate seg image...")
    matched_seg = build_matched_cand_seg(track_graph, fragments, merge_history)
    masked_seg = (
        build_matched_cand_seg(track_graph, fragments, merge_history, nodes=masked_nodes)
        if masked_nodes
        else None
    )

    print("Loading raw image for visualization...")
    raw_img = zarr.open(raw_dir)[:, 0, ...]

    import napari

    viewer = napari.Viewer()
    viewer.add_image(np.asarray(raw_img), name="raw", scale=scale)
    viewer.add_labels(np.asarray(gt_seg).astype(np.int32), name="gt_seg", scale=scale)
    viewer.add_labels(matched_seg.astype(np.int32), name="matched_cand", scale=scale)
    if masked_seg is not None:
        viewer.add_labels(masked_seg.astype(np.int32), name="masked_cand", scale=scale)
    napari.run()
