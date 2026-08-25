import argparse
from pathlib import Path
import napari
import toml
import zarr
from typing import Any

from funtracks.import_export import import_from_geff
from motile_tracker.application_menus import MainApp
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from mhat.utils import get_axes_metadata



def main(config, ground_truth: bool = False):

    experiment = config["experiment"]
    dataset = config["dataset"]
    dataset_fl2 = dataset.replace("nuclei", "cells") if "nuclei" in dataset else None
    exp_uid = config["exp_uid"]
    seg_result = config["seg_result"]
    flow_result = config.get("flow_result", "")

    # Base directories come from the tracking config that produced this run, so
    # the viewer works on any machine. run_tracking.py reads these same three
    # keys and writes a copy of the config next to its output.
    raw_base_dir = Path(config["raw_base_dir"])
    experiments_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])

    seg_base_dir = experiments_base_dir / "segmentation"
    flow_base_dir = experiments_base_dir / "opticalflow"
    tracking_base_dir = output_base_dir / "tracking"

    raw_cells_zarr_path = Path(raw_base_dir / experiment / f"{dataset}.zarr")
    raw_cells_fl2_zarr_path = Path(raw_base_dir / experiment / f"{dataset_fl2}.zarr") if dataset_fl2 else None
    frag_zarr_path = Path(seg_base_dir / experiment / dataset / seg_result / 'data.zarr')
    flow_2d_zarr_path = Path(flow_base_dir / experiment / dataset / "opticalflow_2d" / flow_result / "flow.zarr") 
    flow_3d_zarr_path = Path(flow_base_dir / experiment / dataset / "opticalflow_3d" / flow_result / "flow.zarr") 

    if all(path.exists() is False for path in [raw_cells_zarr_path, frag_zarr_path, flow_2d_zarr_path, flow_3d_zarr_path]):
        raise FileNotFoundError("None of the required data paths exist. Exiting.")
    
    viewer = napari.Viewer()

    if raw_cells_zarr_path.exists():
        raw_cells = zarr.open(raw_cells_zarr_path, mode='r')
        axes = get_axes_metadata(raw_cells)
        scale = [axis["scale"] for axis in axes]
        raw_cells = raw_cells[:, 0, ...]
        print(f"Raw shape: {raw_cells.shape}, dtype: {raw_cells.dtype}, scale: {scale}")
        viewer.add_image(raw_cells, name='raw', colormap='gray', blending='additive', scale=scale)
    else:
        print(f"Warning: Raw cells path {raw_cells_zarr_path} does not exist.")
        print("Defaulting scale to [1.0, 1.0, 1.0, 1.0]")
        scale = [1.0, 1.0, 1.0, 1.0]

    if raw_cells_fl2_zarr_path and raw_cells_fl2_zarr_path.exists():
        raw_cells_fl2 = zarr.open(raw_cells_fl2_zarr_path, mode='r')
        raw_cells_fl2 = raw_cells_fl2[:, 0, ...]
        print(f"Raw FL2 shape: {raw_cells_fl2.shape}, dtype: {raw_cells_fl2.dtype}")
        viewer.add_image(raw_cells_fl2, name='raw_FL2', colormap='magenta', blending='additive', scale=scale)

    # if frag_zarr_path.exists():
    #     fragments = zarr.open(frag_zarr_path, mode='r')
    #     fragments = fragments['fragments'][:, ...]
    #     print(f"Fragments shape: {fragments.shape}, dtype: {fragments.dtype}")
    #     viewer.add_labels(fragments, name='fragments', scale=scale)
    # else:
    #     print(f"Warning: Fragments path {frag_zarr_path} does not exist.")

    # if flow_2d_zarr_path.exists():
    #     flow_2d = zarr.open(flow_2d_zarr_path, mode='r')
    #     flow_2d = flow_2d['flow_frames_XY'][:, ...]
    #     print(f"2D Flow shape: {flow_2d.shape}, dtype: {flow_2d.dtype}")
    #     viewer.add_image(flow_2d, name='flow_2d', blending='additive', scale=scale)
    # else:
    #     print(f"Warning: 2D flow path {flow_2d_zarr_path} does not exist.")

    # if flow_3d_zarr_path.exists():
    #     flow_3d = zarr.open(flow_3d_zarr_path, mode='r')
    #     flow_3d_xy = flow_3d['flow_frames_XY'][:, ...]
    #     print(f"3D Flow XY shape: {flow_3d_xy.shape}, dtype: {flow_3d_xy.dtype}")
    #     viewer.add_image(flow_3d_xy, name='flow_3d_xy', blending='additive', scale=scale)

    #     flow_3d_z = flow_3d['flow_frames_Z'][:, ...]
    #     print(f"3D Flow Z shape: {flow_3d_z.shape}, dtype: {flow_3d_z.dtype}")
    #     viewer.add_image(flow_3d_z, name='flow_3d_z', colormap='berlin', blending='additive', scale=scale)
    # else:
    #     print(f"Warning: 3D flow path {flow_3d_zarr_path} does not exist.")

    # Load tracking data if it exists
    if ground_truth:
        print("Visualizing ground truth tracks.")
        track_data_zarr_path = Path(tracking_base_dir / experiment / dataset / 'correct_tracks.zarr')
        track_seg_zarr_path = Path(tracking_base_dir / experiment / dataset / 'correct_seg.zarr')
    else:
        print("Visualizing predicted tracks.")
        track_data_zarr_path = Path(tracking_base_dir / experiment / dataset / exp_uid / 'pred_tracks.zarr')
        track_seg_zarr_path = Path(tracking_base_dir / experiment / dataset / exp_uid / 'pred_seg.zarr')
    if not track_seg_zarr_path.exists():
        track_seg_zarr_path = None
    gt_seg_zarr_path = Path(tracking_base_dir / experiment / dataset / 'correct_seg.zarr')

    # Add the MainApp widget first
    widget = MainApp(viewer)
    viewer.window.add_dock_widget(widget)
    
    if track_data_zarr_path.exists():
        print(f"Loading tracks from {track_data_zarr_path}")
        
        # Load tracks using import_from_geff
        try:
            node_name_map = {
                "time": "time",
                "x": "x", 
                "y": "y",
                "z": "z",
                "id": "track_id",    # track_id stays constant across frames
            }

            tracks = import_from_geff(
                track_data_zarr_path,
                node_name_map=node_name_map,
                segmentation_path=track_seg_zarr_path,
                scale=scale,
            )

            # Add tracks to the TracksViewer
            tracks_viewer = TracksViewer.get_instance(viewer)
            tracks_viewer.tracks_list.add_tracks(tracks, dataset)
            print(f"Successfully loaded tracks: {tracks}")
            if tracks_viewer.tracking_layers.tracks_layer is not None:
                tracks_viewer.tracking_layers.tracks_layer.tail_length = 4
                tracks_viewer.tracking_layers.tracks_layer.tail_width = 2.0
                tracks_viewer.tracking_layers.points_layer.visible = False

            if gt_seg_zarr_path.exists():
                gt_seg = zarr.open(gt_seg_zarr_path, mode='r')
                print(f"Successfully loaded segmentation for tracks from {track_seg_zarr_path}")
                viewer.add_labels(gt_seg[:, ...], name='GT Segmentation', opacity=0.5, scale=scale)

        except Exception as e:
            print(f"Failed to load tracks: {e}")
    else:
        print(f"Warning: Track data path {track_data_zarr_path} does not exist.")

    napari.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="View tracking results in napari. Pass the config.toml that "
                    "run_tracking.py wrote next to the results you want to see."
    )
    parser.add_argument("config", help="path to a tracking run's config.toml")
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="show the ground truth tracks instead of the predictions",
    )
    args = parser.parse_args()
    track_config = toml.load(args.config)
    main(track_config, ground_truth=args.ground_truth)