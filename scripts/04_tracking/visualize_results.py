from pathlib import Path
import napari
import zarr
from typing import Any

from funtracks.import_export.import_from_geff import import_from_geff
from motile_tracker.application_menus import MainApp
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer



def main(dataset_name, exp_number, exp_name):

    data_path = Path('Y:\\jennifer\\mhat\\data')
    experiment_path = Path('Y:\\jennifer\\mhat\\experiments')
    # data_path = Path('/Volumes/sgrolab/jennifer/mhat/data')
    # experiment_path = Path('/Volumes/sgrolab/jennifer/mhat/experiments')
    # data_path = Path('/groups/sgro/sgrolab/jennifer/mhat/data')
    # experiment_path = Path('/groups/sgro/sgrolab/jennifer/mhat/experiments')

    raw_cells_zarr_path = data_path / dataset_name / f"{exp_number:02d}_nuclei.zarr"
    raw_rocks_zarr_path = data_path / dataset_name / f"{exp_number:02d}_rocks.zarr"
    frag_zarr_path = experiment_path / 'segmentation' / dataset_name / f"no_{exp_number:02d}_nuclei" / 'data.zarr'
    rocks_seg_zarr_path = experiment_path / 'segmentation' / dataset_name / f"{exp_number:02d}_rocks" / 'data.zarr'
    track_seg_zarr_path = experiment_path / 'tracking' / dataset_name / f"{exp_number:02d}_nuclei" / exp_name / 'pred_seg.zarr'

    if all(path.exists() is False for path in [raw_cells_zarr_path, raw_rocks_zarr_path, frag_zarr_path, rocks_seg_zarr_path, track_seg_zarr_path]):
        raise FileNotFoundError("None of the required data paths exist. Exiting.")
    
    viewer = napari.Viewer()

    if raw_cells_zarr_path.exists():
        raw_cells = zarr.open(raw_cells_zarr_path, mode='r')
        axes = raw_cells.attrs.get("axes", None)
        if axes is not None:
            axes = [axis for axis in axes if axis.get("name") != "channel"]
            scale = [axis["scale"] for axis in axes if axis.get("scale") is not None]
        else:
            scale = [1.0, 1.0, 1.0, 1.0]
        raw_cells = raw_cells[:, 0, ...]
        print(f"Raw shape: {raw_cells.shape}, dtype: {raw_cells.dtype}")
        viewer.add_image(raw_cells, name='raw', colormap='gray', blending='additive', scale=scale)
    else:
        print(f"Warning: Raw cells path {raw_cells_zarr_path} does not exist.")
        print("Defaulting scale to [1.0, 1.0, 1.0, 1.0]")
        scale = [1.0, 1.0, 1.0, 1.0]

    # if raw_rocks_zarr_path.exists():
    #     raw_rocks = zarr.open(raw_rocks_zarr_path, mode='r')
    #     raw_rocks = raw_rocks[:, 0, ...]
    #     print(f"Rocks shape: {raw_rocks.shape}, dtype: {raw_rocks.dtype}")
    #     viewer.add_image(raw_rocks, name='rocks', colormap='gray', blending='additive', scale=scale)
    # else:
    #     print(f"Warning: Raw rocks path {raw_rocks_zarr_path} does not exist.")

    if frag_zarr_path.exists():
        fragments = zarr.open(frag_zarr_path, mode='r')
        fragments = fragments['fragments'][:, ...]
        print(f"Fragments shape: {fragments.shape}, dtype: {fragments.dtype}")
        viewer.add_labels(fragments, name='fragments', scale=scale)
    else:
        print(f"Warning: Fragments path {frag_zarr_path} does not exist.")

    # if rocks_seg_zarr_path.exists():
    #     rocks_seg = zarr.open(rocks_seg_zarr_path, mode='r')
    #     rocks_seg = rocks_seg['segmentations'][:, ...]
    #     print(f"Rocks segmentation shape: {rocks_seg.shape}, dtype: {rocks_seg.dtype}")
    #     viewer.add_labels(rocks_seg, name='rocks_seg', opacity=0.5, scale=scale)
    # else:
    #     print(f"Warning: Rocks segmentation path {rocks_seg_zarr_path} does not exist.")

    if track_seg_zarr_path.exists():
        track_seg = zarr.open(track_seg_zarr_path, mode='r')
        track_seg = track_seg[:]
        print(f"Tracked segmentation shape: {track_seg.shape}, dtype: {track_seg.dtype}")
        viewer.add_labels(track_seg, name='track_seg', opacity=0.5, scale=scale)
    else:
        print(f"Warning: Track segmentation path {track_seg_zarr_path} does not exist.")

    # Load tracking data if it exists
    track_data_zarr_path = experiment_path / 'tracking' / dataset_name / f"{exp_number:02d}_nuclei" / exp_name / 'pred_tracks.zarr'
    
    # Add the MainApp widget first
    widget = MainApp(viewer)
    viewer.window.add_dock_widget(widget)
    
    if track_data_zarr_path.exists():
        print(f"Loading tracks from {track_data_zarr_path}")
        
        # Load tracks using import_from_geff
        try:
            name_map = {
                "time": "time",
                "x": "x", 
                "y": "y",
                "z": "z",
                "id": "label",
            }
            
            tracks = import_from_geff(
                track_data_zarr_path,
                name_map,
                segmentation_path=None,
                scale=scale,
            )
            # Add tracks to the TracksViewer
            tracks_viewer = TracksViewer.get_instance(viewer)
            tracks_viewer.tracks_list.add_tracks(tracks, exp_name)
            print(f"Successfully loaded tracks: {tracks}")

            
        except Exception as e:
            print(f"Failed to load tracks: {e}")
    else:
        print(f"Warning: Track data path {track_data_zarr_path} does not exist.")

    napari.run()

if __name__ == '__main__':
    main(dataset_name="NC281-Fl2mSiH2B", 
         exp_number=3, 
         exp_name="2025-12-15_15-37-21",
         )