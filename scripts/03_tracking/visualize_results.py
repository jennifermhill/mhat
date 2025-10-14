from pathlib import Path
import napari
import zarr
import csv
import numpy as np
import networkx as nx

import motile_plugin
from mhat.tracking.tracks_io import load_tracks_from_csv
from mhat.tracking.utils import (
    relabel_segmentation,
)
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids

def load_tracks_as_array(csv_path):
    """Load tracks from CSV and convert to napari format."""
    tracks_data = []
    parent_graph = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = int(float(row['id']))
            timepoint = int(float(row['time']))  # Convert float to int
            x = int(round(float(row['x'])))
            y = int(round(float(row['y'])))
            z = int(round(float(row['z'])))
            parent_id = row['parent_id']
            
            # Convert to napari format: [track_id, t, z, y, x]
            tracks_data.append([track_id, timepoint, z, y, x])
            
            # Build parent graph
            if parent_id and parent_id != '' and parent_id != '-1' and parent_id != 'nan':
                try:
                    parent_graph[track_id] = int(float(parent_id))
                except ValueError:
                    parent_graph[track_id] = None
            else:
                parent_graph[track_id] = None
    
    # Convert to numpy array
    tracks_array = np.array(tracks_data, dtype=int)
    
    return tracks_array, parent_graph


def main(raw_zarr_path, track_zarr_path):
    raw = zarr.open(raw_zarr_path, mode='r')
    pred = zarr.open(track_zarr_path, mode='r')

    raw = raw[:, 0, ...]
    pred_seg = pred['2025-10-13_15-49-56_pred_mask'][:]

    print(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")
    print(f"Tracks shape: {pred_seg.shape}, dtype: {pred_seg.dtype}")

    csv_path = Path(track_zarr_path).parent / 'pred_tracks.csv'
    tracks, parent_graph = load_tracks_as_array(csv_path)

    pred_tracks = load_tracks_from_csv(csv_path)

    run = motile_plugin.data_model.Tracks(
        segmentation=np.expand_dims(pred_seg, axis=1),
        graph=pred_tracks,
        pos_attr="pos",
    )

    viewer = napari.Viewer()
    viewer.add_image(raw, name='raw')
    viewer.add_labels(pred_seg, name='pred_seg', opacity=0.5)
    viewer.add_tracks(tracks, name='pred_tracks')
    napari.run()

if __name__ == '__main__':
    raw_zarr_path = '/groups/sgro/sgrolab/jennifer/mhat/data/mixin63/02_test_data.zarr'
    track_zarr_path = '/groups/sgro/sgrolab/jennifer/mhat/experiments/tracking/mixin63/03_test_data/2025-10-13_15-49-56/data.zarr'
    main(raw_zarr_path, track_zarr_path)