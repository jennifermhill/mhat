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
from pathlib import Path

import numpy as np
import toml
import zarr

from funtracks.import_export import import_from_geff
from mhat.evaluation.evaluate_tracking import remap_seg_to_track_ids
from mhat.tracking.gt_annotation import (
    annotate_gt_on_candidate_graph,
    build_node_to_fragments,
)
from mhat.tracking.pipeline import build_track_graph


def load_gt(gt_data_dir, scale):
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
        raise RuntimeError("GT segmentation not found.")
    gt_seg = remap_seg_to_track_ids(gt_tracks_path, gt_tracks.graph, gt_tracks.segmentation)
    return gt_tracks.graph, gt_seg


def build_matched_cand_seg(track_graph, fragments, merge_history):
    """Per-pixel image where matched candidate IDs are painted (others 0).

    Matched candidates come from gt_selected=1 nodes. Conflict-resolved
    matching guarantees no two matched candidates share fragments, so the
    result is a valid label image (no overwrite ambiguity).
    """
    node_to_fragments = build_node_to_fragments(track_graph, merge_history)
    matched_seg = np.zeros_like(fragments)
    for node, data in track_graph.nodes.items():
        if data.get("gt_selected") != 1:
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
    args = parser.parse_args()
    config = toml.load(args.config)

    raw_base_dir = Path(config["raw_base_dir"])
    input_base_dir = Path(config["input_base_dir"])
    experiment = config["experiment"]
    dataset = config["dataset"]

    raw_dir = raw_base_dir / experiment / f"{dataset}.zarr"
    seg_dir = input_base_dir / "segmentation" / experiment / dataset / config["seg_result"]
    gt_data_dir = input_base_dir / "tracking" / experiment / dataset

    flow_result = config.get("flow_result", None)
    if flow_result is not None:
        if config["use_lk"]:
            flow_dir_3d = (
                input_base_dir
                / "opticalflow"
                / experiment
                / dataset
                / "opticalflow_lucaskanade"
                / flow_result
            )
            flow_dir_2d = None
        else:
            flow_dir_2d = (
                input_base_dir
                / "opticalflow"
                / experiment
                / dataset
                / "opticalflow_2d"
                / flow_result
            )
            flow_dir_3d = (
                input_base_dir
                / "opticalflow"
                / experiment
                / dataset
                / "opticalflow_3d"
                / flow_result
            )
            if not flow_dir_2d.is_dir():
                flow_dir_2d = None
        flow_dirs = {"2d": flow_dir_2d, "3d": flow_dir_3d}
    else:
        flow_dirs = {"2d": None, "3d": None}

    track_graph, fragments, merge_history, _exclusion_sets, scale, _axes = build_track_graph(
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

    print("Building matched-candidate seg image...")
    matched_seg = build_matched_cand_seg(track_graph, fragments, merge_history)

    print("Loading raw image for visualization...")
    raw_img = zarr.open(raw_dir)[:, 0, ...]

    import napari

    viewer = napari.Viewer()
    viewer.add_image(np.asarray(raw_img), name="raw", scale=scale)
    viewer.add_labels(np.asarray(gt_seg).astype(np.int32), name="gt_seg", scale=scale)
    viewer.add_labels(matched_seg.astype(np.int32), name="matched_cand", scale=scale)
    napari.run()
