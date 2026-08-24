from pathlib import Path

import napari
import numpy as np
import zarr
from funtracks.data_model import Tracks
from mhat.tracking.tracks_io import load_tracks_from_csv, read_gt_tracks
from mhat.tracking.utils import (
    relabel_segmentation,
)
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids
from motile_tracker.application_menus import MainApp
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


def crop_tracks(tracks, start_frame, end_frame, frame_attr="time"):
    if start_frame is None:
        if end_frame is None:
            return tracks
        start_frame = 0
    if end_frame is None:
        end_frame = max([data[frame_attr] for node, data in tracks.nodes(data=True)])
    keep_nodes = [
        node
        for node, data in tracks.nodes(data=True)
        if data[frame_attr] >= start_frame and data[frame_attr] < end_frame
    ]
    subgraph = tracks.subgraph(keep_nodes)
    if start_frame is not None:
        for node in subgraph.nodes():
            subgraph.nodes[node][frame_attr] -= start_frame
    return subgraph


def view_run(
    gt_data_path: Path,
    pred_data_path,
    experiment: str,
    run_name: str,
    start_frame=None,
    end_frame=None,
):
    zarr_name = "data.zarr"
    raw_group = "phase"
    gt_seg_group = "mask"
    input_seg_group = "fragments"
    pred_seg_group = f"{experiment}_pred_mask"

    gt_tracks_name = "gt_tracks.csv"
    pred_tracks_name = f"{experiment}/pred_tracks.csv"
    gt_zarr_root = zarr.open(gt_data_path / zarr_name)
    pred_zarr_root = zarr.open(pred_data_path / zarr_name)
    raw_data = gt_zarr_root[raw_group][start_frame:end_frame]
    pred_seg = pred_zarr_root[pred_seg_group][start_frame:end_frame]
    fragments = gt_zarr_root[input_seg_group][start_frame:end_frame]

    pred_tracks = load_tracks_from_csv(pred_data_path / pred_tracks_name)
    pred_tracks = crop_tracks(pred_tracks, start_frame, end_frame)
    assign_tracklet_ids(pred_tracks)

    pred_seg = relabel_segmentation(pred_tracks, pred_seg)

    run = Tracks(
        graph=pred_tracks,
        segmentation=np.expand_dims(pred_seg, axis=1),
        pos_attr="pos",
    )

    viewer = napari.Viewer()
    viewer.add_image(raw_data, name="phase")
    viewer.add_labels(fragments, name="fragments")

    if gt_seg_group in gt_zarr_root:
        gt_seg = gt_zarr_root[gt_seg_group][start_frame:end_frame]
        gt_tracks = read_gt_tracks(
            gt_data_path / zarr_name, gt_data_path / gt_tracks_name
        )
        gt_tracks = crop_tracks(gt_tracks, start_frame, end_frame)
        assign_tracklet_ids(gt_tracks)
        viewer.add_labels(gt_seg, name="gt_seg")

    widget = MainApp(viewer)
    viewer.window.add_dock_widget(widget, name="Motile Widget")
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.tracks_list.add_tracks(run, name=experiment)
    napari.run()
