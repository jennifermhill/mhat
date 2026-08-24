"""Overlay GT points and the matched / unmatched predicted segmentation in napari.

Runs the same matcher the evaluation uses, then splits the predicted
segmentation into the objects that matched a GT node and the ones that did not,
so the two classes can be toggled and compared against the GT points:

    GT points            yellow   all ground truth detections
    GT points (FN)       red      GT detections with no predicted match (hidden by default)
    pred matched         blue     predicted objects that matched a GT node
    pred unmatched       magenta  predicted objects with no GT match (the FPs)

The label layers are lazy (dask over the zarr), so full-size predictions do not
have to fit in memory. Labels present in the segmentation but absent from the
tracks graph are dropped from both layers.

Each GT point is drawn only in its own timepoint, and copied onto the nearby z
slices (see ``--z-spread``) so it stays visible while scrolling through z.

Usage:
    conda run -n mhat2 --no-capture-output python scripts/05_evaluation/view_matched_seg_napari.py scripts/05_evaluation/eval_config.toml
"""
import argparse
from pathlib import Path

import dask.array as da
import geff
import napari
import numpy as np
import toml
import zarr
from napari.utils.colormaps import DirectLabelColormap

from mhat.evaluation.diagnostics import load_run_config
from mhat.evaluation.evaluate_tracking import match_tracking

GT_COLOR = "yellow"
GT_FN_COLOR = "red"
MATCHED_COLOR = (0.15, 0.55, 1.0, 1.0)
UNMATCHED_COLOR = (1.0, 0.2, 0.8, 1.0)


def spatial_scale(pred_seg_zarr, pred_tracks_path):
    """Voxel -> world scale for the segmentation, from its axes or the geff."""
    axes = pred_seg_zarr.attrs.get("axes")
    if axes:
        return [float(a.get("scale") or 1.0) for a in axes]
    (_, metadata) = geff.read(pred_tracks_path)
    if metadata.axes:
        return [float(a.scale) if a.scale is not None else 1.0 for a in metadata.axes]
    return [1.0] * pred_seg_zarr.ndim


def label_luts(pred_tracks_path, matched_nodes):
    """Two lookup tables splitting segmentation labels into matched / unmatched."""
    raw_graph, metadata = geff.read(pred_tracks_path)

    label_prop = "label"
    if metadata.related_objects:
        for ro in metadata.related_objects:
            if ro.type == "labels" and ro.label_prop:
                label_prop = ro.label_prop
                break

    label_of = {node: int(data.get(label_prop, node))
                for node, data in raw_graph.nodes(data=True)}
    if not label_of:
        raise ValueError(f"No nodes found in {pred_tracks_path}")

    max_label = max(label_of.values())
    lut_matched = np.zeros(max_label + 1, dtype=np.uint32)
    lut_unmatched = np.zeros(max_label + 1, dtype=np.uint32)
    for node, label in label_of.items():
        if node in matched_nodes:
            lut_matched[label] = label
        else:
            lut_unmatched[label] = label

    return lut_matched, lut_unmatched, max_label


def apply_lut(seg, lut, max_label):
    """Lazily remap a label array through a LUT, dropping out-of-range labels."""
    def _block(block):
        return lut[np.where(block <= max_label, block, 0)]

    return da.map_blocks(_block, seg, dtype=np.uint32)


def spread_over_z(points, z_scale, n_z, spread):
    """Copy each point onto the z slices within ``spread`` world units of it.

    napari's ``out_of_slice_display`` fades a point into every non-displayed
    dimension, time included, so a point at t=5 with size 6 shows up across
    t=2-8. Points are therefore placed exactly (sharp in time) and only the z
    spread is reproduced by hand. The nearest slice is always included so a
    small spread can never hide a point.
    """
    if points.size == 0 or points.shape[1] != 4:  # no z axis to spread over
        return points

    slice_centers = np.arange(n_z) * z_scale
    copies = []
    for point in points:
        nearest = int(np.argmin(np.abs(slice_centers - point[1])))
        indices = set(np.flatnonzero(np.abs(slice_centers - point[1]) <= spread).tolist())
        indices.add(nearest)
        for k in sorted(indices):
            copy = point.copy()
            copy[1] = slice_centers[k]
            copies.append(copy)
    return np.array(copies)


def link_size_slider(layer):
    """Make the GUI point-size control resize the whole layer.

    The slider (and the number box next to it) sets ``current_size``, which
    napari applies only to *selected* points and to points added afterwards --
    so with nothing selected it looks like the control does nothing. Mirroring
    it onto ``size`` resizes every point, which is what you want for a layer
    that is being viewed rather than edited.
    """
    def _apply(event=None):
        if layer.size.size and not np.allclose(layer.size, layer.current_size):
            layer.size = layer.current_size

    layer.events.current_size.connect(_apply)
    return layer


def gt_points(matched, start_frame, end_frame):
    """(all_points, fn_points) as (n, ndim+1) arrays of [t, *world position]."""
    gt_graph = matched.gt_graph.graph
    all_points, fn_points = [], []
    for node, data in gt_graph.nodes(data=True):
        t = int(data["time"])
        if t < start_frame or (end_frame is not None and t >= end_frame):
            continue
        point = [t - start_frame, *[float(p) for p in data["pos"]]]
        all_points.append(point)
        if matched.get_gt_pred_match(node) is None:
            fn_points.append(point)
    return np.array(all_points), np.array(fn_points)


def find_raw(bases, experiment, dataset):
    """The raw zarr for this dataset, or None if it isn't reachable from here.

    The tracking config's ``raw_base_dir`` often points at the cluster, so the
    sibling ``data`` directory of ``input_base_dir`` is tried as well.
    """
    for base in bases:
        if not base:
            continue
        path = Path(base) / experiment / f"{dataset}.zarr"
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Evaluation config (same one evaluate_tracks.py uses)")
    parser.add_argument("-s", "--start-frame", type=int, default=0)
    parser.add_argument("-e", "--end-frame", type=int, default=None)
    parser.add_argument("--point-size", type=float, default=6.0,
                        help="GT point diameter in world units.")
    parser.add_argument("--z-spread", type=float, default=None,
                        help="Show each GT point on every z slice within this many "
                             "world units of it (default: half the point size). "
                             "0 shows it only on its own slice; points are always "
                             "confined to their own timepoint.")
    parser.add_argument("--match-threshold", type=float, default=None,
                        help="Override the config's match_threshold for this view only "
                             "(point matcher; world units). The config file is not modified.")
    parser.add_argument("--no-raw", action="store_true", help="Skip the raw image layer.")
    parser.add_argument("--raw-base-dir", default=None,
                        help="Where to look for <experiment>/<dataset>.zarr. Defaults to "
                             "the configs' raw_base_dir, then input_base_dir's sibling "
                             "'data' directory.")
    args = parser.parse_args()
    config = toml.load(args.config)

    if args.match_threshold is not None:
        config["match_threshold"] = args.match_threshold
        if "threshold" in config:  # this key wins in build_matcher, keep them in sync
            config["threshold"] = args.match_threshold

    input_base_dir = Path(config["input_base_dir"])
    experiment = config["experiment"]
    dataset = config["dataset"]
    track_result = config["track_result"]

    gt_data_dir = input_base_dir / "tracking" / experiment / dataset
    pred_data_dir = gt_data_dir / track_result
    pred_seg_path = pred_data_dir / "pred_seg.zarr"
    pred_tracks_path = pred_data_dir / "pred_tracks.zarr"
    if not pred_seg_path.exists():
        raise SystemExit(f"No predicted segmentation at {pred_seg_path} - nothing to split.")

    run_config = load_run_config(pred_data_dir)
    start, end = args.start_frame, args.end_frame

    print(f"Matching {experiment}/{dataset}/{track_result} "
          f"(matcher={config.get('matcher', 'point')})...")
    matched = match_tracking(config, gt_data_dir, pred_data_dir)

    matched_nodes = {
        pred for pred in (matched.get_gt_pred_match(n)
                          for n in matched.gt_graph.graph.nodes())
        if pred is not None
    }
    n_pred = matched.pred_graph.graph.number_of_nodes()
    print(f"  {len(matched_nodes)} of {n_pred} predicted nodes matched a GT node")

    pred_seg_zarr = zarr.open(str(pred_seg_path), mode="r")
    scale = spatial_scale(pred_seg_zarr, pred_tracks_path)
    lut_matched, lut_unmatched, max_label = label_luts(pred_tracks_path, matched_nodes)

    seg = da.from_zarr(str(pred_seg_path))[start:end]
    seg_matched = apply_lut(seg, lut_matched, max_label)
    seg_unmatched = apply_lut(seg, lut_unmatched, max_label)

    all_points, fn_points = gt_points(matched, start, end)
    print(f"  {len(all_points)} GT points in frames [{start}, {end}), "
          f"{len(fn_points)} of them unmatched")

    z_spread = args.point_size / 2 if args.z_spread is None else args.z_spread
    if len(all_points) and all_points.shape[1] == 4:
        n_z = seg.shape[1]
        all_points = spread_over_z(all_points, scale[1], n_z, z_spread)
        fn_points = spread_over_z(fn_points, scale[1], n_z, z_spread)
        print(f"  spread over z slices within {z_spread} world units: "
              f"{len(all_points)} point markers")

    viewer = napari.Viewer()

    raw_bases = [args.raw_base_dir, config.get("raw_base_dir"),
                 run_config.get("raw_base_dir"), input_base_dir.parent / "data"]
    raw_path = None if args.no_raw else find_raw(raw_bases, experiment, dataset)
    if raw_path is not None:
        raw = da.from_zarr(str(raw_path))[start:end]
        if raw.ndim == seg.ndim + 1:  # (t, c, z, y, x) -> first channel
            raw = raw[:, 0]
        viewer.add_image(raw, name="raw", scale=scale, blending="additive")
    elif not args.no_raw:
        print("  raw data not reachable from here; skipping the raw layer")

    viewer.add_labels(
        seg_unmatched, name="pred unmatched (FP)", scale=scale, opacity=0.6,
        colormap=DirectLabelColormap(color_dict={None: UNMATCHED_COLOR, 0: (0, 0, 0, 0)}),
    )
    viewer.add_labels(
        seg_matched, name="pred matched (TP)", scale=scale, opacity=0.6,
        colormap=DirectLabelColormap(color_dict={None: MATCHED_COLOR, 0: (0, 0, 0, 0)}),
    )
    if len(fn_points):
        link_size_slider(viewer.add_points(
            fn_points, name="GT points (FN)", face_color=GT_FN_COLOR,
            size=args.point_size, visible=False,
        ))
    if len(all_points):
        link_size_slider(viewer.add_points(
            all_points, name="GT points", face_color=GT_COLOR, size=args.point_size,
        ))

    napari.run()
