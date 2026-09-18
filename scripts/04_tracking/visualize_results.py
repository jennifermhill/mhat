import argparse
from pathlib import Path
import napari
import toml
import zarr
import dask.array as da
from typing import Any

from funtracks.import_export import import_from_geff
from motile_tracker.application_menus import MainApp
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from mhat.evaluation.evaluate_tracking import (
    read_name_map_and_scale,
    remap_seg_to_track_ids_lazy,
)
from mhat.utils import get_axes_metadata


def main(config, ground_truth: bool = False, compute: bool = False):

    experiment = config["experiment"]
    dataset = config["dataset"]
    exp_uid = config["exp_uid"]

    # Base directories come from the tracking config that produced this run, so
    # the viewer works on any machine. run_tracking.py reads these same three
    # keys and writes a copy of the config next to its output.
    raw_base_dir = Path(config["raw_base_dir"])
    output_base_dir = Path(config["output_base_dir"])

    tracking_base_dir = output_base_dir / "tracking"

    raw_cells_zarr_path = Path(raw_base_dir / experiment / f"{dataset}.zarr")

    if not raw_cells_zarr_path.exists():
        raise FileNotFoundError(f"Raw cells path {raw_cells_zarr_path} does not exist. Exiting.")
    
    viewer = napari.Viewer()

    if raw_cells_zarr_path.exists():
        axes = get_axes_metadata(zarr.open(raw_cells_zarr_path, mode='r'))
        scale = [axis["scale"] for axis in axes]
        scale[0] = 1.0  # Set time axis scale to 1.0 for visualization
        raw_cells = da.from_zarr(raw_cells_zarr_path)[:, 0, ...]
        print(f"Raw shape: {raw_cells.shape}, dtype: {raw_cells.dtype}, scale: {scale}")
        if compute:
            raw_cells = raw_cells.compute()
        viewer.add_image(raw_cells, name='raw', colormap='gray', blending='additive', scale=scale)
    else:
        print(f"Warning: Raw cells path {raw_cells_zarr_path} does not exist.")
        print("Defaulting scale to [1.0, 1.0, 1.0, 1.0]")
        scale = [1.0, 1.0, 1.0, 1.0]

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

    # Add the MainApp widget first
    widget = MainApp(viewer)
    viewer.window.add_dock_widget(widget)
    
    if track_data_zarr_path.exists():
        print(f"Loading tracks from {track_data_zarr_path}")
        
        # Load tracks using import_from_geff
        try:
            # Built from the store's own axes rather than hardcoded to 3D --
            # funtracks infers dimensionality from this map, so a 3D map on a
            # 2D file silently declares an axis that is not there. See
            # read_name_map_and_scale for the whole story.
            name_map, _ = read_name_map_and_scale(track_data_zarr_path)
            node_name_map = {
                **name_map,
                "id": "track_id",    # track_id stays constant across frames
            }

            tracks = import_from_geff(
                track_data_zarr_path,
                node_name_map=node_name_map,
                segmentation_path=track_seg_zarr_path if compute else None,
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

            # Load segmentation outside of track import to avoid memory issues.
            # The store is labelled by node id, so relabel to track_id to colour
            # each track consistently over time, as the compute=True path does.
            # This stays lazy: only the slice on screen is read.
            if not compute and track_seg_zarr_path is not None:
                track_seg = remap_seg_to_track_ids_lazy(tracks.graph, track_seg_zarr_path)
                print(f"Successfully loaded segmentation for tracks from {track_seg_zarr_path}")
                viewer.add_labels(track_seg, name='Track Segmentation', opacity=0.5, scale=scale)

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
    parser.add_argument(
        "--compute",
        action="store_true",
        help="load the arrays into memory instead of viewing them lazily",
    )
    args = parser.parse_args()
    track_config = toml.load(args.config)
    main(track_config, ground_truth=args.ground_truth, compute=args.compute)