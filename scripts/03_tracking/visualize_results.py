from pathlib import Path
import napari
import zarr
import networkx as nx
import pandas as pd
import numpy as np
from typing import Any

from motile_toolbox.candidate_graph import NodeAttr
from motile_tracker.import_export.load_tracks import tracks_from_df
from motile_tracker.application_menus import MainApp
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer



def main(dataset_name, exp_number, exp_name):

    data_path = Path('Y:\\jennifer\\mhat\\data')
    experiment_path = Path('Y:\\jennifer\\mhat\\experiments')

    raw_cells_zarr_path = data_path / dataset_name / f"{exp_number:02d}_cells.zarr"
    raw_rocks_zarr_path = data_path / dataset_name / f"{exp_number:02d}_rocks.zarr"
    frag_zarr_path = experiment_path / 'segmentation' / dataset_name / f"{exp_number:02d}_cells" / 'data.zarr'
    rocks_seg_zarr_path = experiment_path / 'segmentation' / dataset_name / f"{exp_number:02d}_rocks" / 'data.zarr'
    track_seg_zarr_path = experiment_path / 'tracking' / dataset_name / f"{exp_number:02d}_cells" / exp_name / 'pred_seg.zarr'

    # Check if all paths exist
    for path in [raw_cells_zarr_path, raw_rocks_zarr_path, frag_zarr_path, rocks_seg_zarr_path, track_seg_zarr_path]:
        if not path.exists():
            print(f"Warning: Path {path} does not exist.")
            path = None

    if all(path is None for path in [raw_cells_zarr_path, raw_rocks_zarr_path, frag_zarr_path, rocks_seg_zarr_path, track_seg_zarr_path]):
        raise FileNotFoundError("None of the required data paths exist. Exiting.")
    
    viewer = napari.Viewer()

    if raw_cells_zarr_path is not None:
        raw_cells = zarr.open(raw_cells_zarr_path, mode='r')
        raw_cells = raw_cells[:, 0, ...]
        print(f"Raw shape: {raw_cells.shape}, dtype: {raw_cells.dtype}")
        viewer.add_image(raw_cells, name='raw', colormap='purple', blending='additive')

    if raw_rocks_zarr_path is not None:
        raw_rocks = zarr.open(raw_rocks_zarr_path, mode='r')
        raw_rocks = raw_rocks[:, 0, ...]
        print(f"Rocks shape: {raw_rocks.shape}, dtype: {raw_rocks.dtype}")
        viewer.add_image(raw_rocks, name='rocks', colormap='gray', blending='additive')

    if frag_zarr_path is not None:
        fragments = zarr.open(frag_zarr_path, mode='r')
        fragments = fragments['fragments'][:, ...]
        print(f"Fragments shape: {fragments.shape}, dtype: {fragments.dtype}")
        viewer.add_labels(fragments, name='fragments')

    if rocks_seg_zarr_path is not None:
        rocks_seg = zarr.open(rocks_seg_zarr_path, mode='r')
        rocks_seg = rocks_seg['segmentations'][:, ...]
        print(f"Rocks segmentation shape: {rocks_seg.shape}, dtype: {rocks_seg.dtype}")
        viewer.add_labels(rocks_seg, name='rocks_seg', opacity=0.5)

    if track_seg_zarr_path is not None:
        track_seg = zarr.open(track_seg_zarr_path, mode='r')
        track_seg = track_seg[:]
        print(f"Tracked segmentation shape: {track_seg.shape}, dtype: {track_seg.dtype}")
        viewer.add_labels(track_seg, name='track_seg', opacity=0.5)

    widget = MainApp(viewer)
    viewer.window.add_dock_widget(widget)

    napari.run()

if __name__ == '__main__':
    main(dataset_name="nc281-spiAmSG", 
         exp_number=1, 
         exp_name="2025-10-28_17-42-55",
         )