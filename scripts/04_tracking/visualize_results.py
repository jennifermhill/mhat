import argparse
from pathlib import Path
import napari
import zarr
import dask.array as da
from typing import Any

from funtracks.import_export import import_from_geff
from motile_tracker.application_menus import MainApp
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from mhat.dataset import Dataset, absolute_path
from mhat.evaluation.evaluate_tracking import (
    read_name_map_and_scale,
    remap_seg_to_track_ids_lazy,
)
from mhat.utils import get_axes_metadata


def main(
    run_dir: Path,
    raw_path: Path | None = None,
    gt_tracks: Path | None = None,
    gt_seg: Path | None = None,
    compute: bool = False,
):
    # The output directory is found from where the run sits, so this works on
    # any machine; raw comes from raw.toml unless --raw overrides it.
    run_dir = absolute_path(run_dir)
    if run_dir.name == "config.toml":
        run_dir = run_dir.parent
    ds = Dataset.from_run_dir(run_dir, "tracking")
    raw_cells_zarr_path = raw_path or ds.raw_path

    if gt_tracks is not None:
        print("Visualizing ground truth tracks.")
        track_data_zarr_path = Path(gt_tracks)
        track_seg_zarr_path = Path(gt_seg) if gt_seg is not None else None
    else:
        print("Visualizing predicted tracks.")
        track_data_zarr_path = run_dir / 'pred_tracks.zarr'
        track_seg_zarr_path = run_dir / 'pred_seg.zarr'
    if track_seg_zarr_path is not None and not track_seg_zarr_path.exists():
        track_seg_zarr_path = None

    viewer = napari.Viewer()

    if raw_cells_zarr_path is not None and Path(raw_cells_zarr_path).exists():
        axes = get_axes_metadata(zarr.open(raw_cells_zarr_path, mode='r'))
        scale = [axis["scale"] for axis in axes]
        scale[0] = 1.0  # Set time axis scale to 1.0 for visualization
        raw_cells = da.from_zarr(raw_cells_zarr_path)[:, 0, ...]
        print(f"Raw shape: {raw_cells.shape}, dtype: {raw_cells.dtype}, scale: {scale}")
        if compute:
            raw_cells = raw_cells.compute()
        viewer.add_image(raw_cells, name='raw', colormap='gray', blending='additive', scale=scale)
    else:
        # The tracks carry the same axes as the raw movie.
        _, scale = read_name_map_and_scale(track_data_zarr_path)
        scale[0] = 1.0
        print(f"Raw data not found at {raw_cells_zarr_path}; pass --raw to show it.")
        print(f"Using the tracks' own scale: {scale}")

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
            tracks_viewer.tracks_list.add_tracks(tracks, ds.name)
            print(f"Successfully loaded tracks: {tracks}")
            if tracks_viewer.tracking_layers.tracks_layer is not None:
                tracks_viewer.tracking_layers.tracks_layer.tail_length = 4
                tracks_viewer.tracking_layers.tracks_layer.tail_width = 2.0
                tracks_viewer.tracking_layers.points_layer.visible = False

            # Load segmentation outside of track import to avoid memory issues.
            # The store is labeled by node id, so relabel to track_id to color
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
        description="View tracking results in napari: one run's predictions, or "
                    "ground truth over that run's raw movie."
    )
    parser.add_argument(
        "run", type=Path,
        help="a tracking run, <output_dir>/tracking/<uid> (or its config.toml)",
    )
    parser.add_argument(
        "--raw", type=Path, default=None,
        help="raw movie to show, if the path in raw.toml is not valid on this machine",
    )
    parser.add_argument(
        "--gt-tracks", type=Path, default=None,
        help="show these ground-truth tracks (geff) instead of the predictions",
    )
    parser.add_argument(
        "--gt-seg", type=Path, default=None,
        help="ground-truth segmentation to show with --gt-tracks",
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="load the arrays into memory instead of viewing them lazily",
    )
    args = parser.parse_args()
    main(args.run, raw_path=args.raw, gt_tracks=args.gt_tracks, gt_seg=args.gt_seg,
         compute=args.compute)