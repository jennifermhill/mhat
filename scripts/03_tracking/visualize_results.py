from pathlib import Path
import napari
import zarr
import pandas as pd

from motile_tracker.import_export.load_tracks import tracks_from_df
from motile_tracker.application_menus import MainApp
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


def main(raw_zarr_path, track_data_path, exp_name):

    pred_tracks_path = track_data_path / exp_name / 'pred_tracks.csv'
    pred_seg_path = track_data_path / exp_name / 'pred_seg.zarr'

    raw = zarr.open(raw_zarr_path, mode='r')
    raw = raw[:, 0, ...]
    print(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")

    # load the segmentation data if available
    try:
        pred_seg = zarr.open(pred_seg_path, mode='r')
        pred_seg = pred_seg[:]
        print(f"Segmentation shape: {pred_seg.shape}, dtype: {pred_seg.dtype}")
    except:
        pred_seg = None
        print(f"Segmentation not found at {pred_seg_path}, loading only raw image and tracks.")

    # load the tracks data
    csvfile = pred_tracks_path
    selected_columns = {
        "time": "time",
        "y": "y",
        "x": "x",
        "id": "id",
        "parent_id": "parent_id",
        "seg_id": "id",
    }

    df = pd.read_csv(csvfile)

    for feature, column in selected_columns.items():
        df[feature] = df[column]

    tracks = tracks_from_df(
        df=df,
        scale=[1, 1, 1, 1],
    )

    viewer = napari.Viewer()
    viewer.add_image(raw, name='raw')
    if pred_seg is not None:
        viewer.add_labels(pred_seg, name='pred_seg', opacity=0.5)
    widget = MainApp(viewer)
    viewer.window.add_dock_widget(widget)
    TracksViewer.get_instance(viewer).tracks_list.add_tracks(tracks, "example")

    napari.run()

if __name__ == '__main__':
    # raw_zarr_path = Path('/groups/sgro/sgrolab/jennifer/mhat/data/mixin63/02_test_data.zarr')
    # track_data_path = Path('/groups/sgro/sgrolab/jennifer/mhat/experiments/tracking/mixin63/03_test_data')
    raw_zarr_path = Path('Y:\\jennifer\\mhat\\data\\mixin63\\01_test_data.zarr')
    track_data_path = Path('Y:\\jennifer\\mhat\\experiments\\tracking\\mixin63\\01_test_data')
    exp_name = "2025-10-16_11-04-32"
    main(raw_zarr_path, track_data_path, exp_name)