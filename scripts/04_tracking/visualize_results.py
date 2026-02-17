from pathlib import Path
import napari
import toml
import zarr
from typing import Any

from funtracks.import_export import import_from_geff
from motile_tracker.application_menus import MainApp
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer



def main(config, compute: bool = False):

    experiment = config["experiment"]
    dataset = config["dataset"]
    exp_uid = config["exp_uid"]
    seg_result = config["seg_result"]
    flow_result = config.get("flow_result", "")

    if Path("Y:\\").exists():
        input_base_dir = Path("Y:\\jennifer\\mhat\\data")
        seg_base_dir = Path("Y:\\jennifer\\mhat\\experiments\\segmentation")
        flow_base_dir = Path("Y:\\jennifer\\mhat\\experiments\\opticalflow")
        tracking_base_dir = Path("Y:\\jennifer\\mhat\\experiments\\tracking")
    elif Path("/groups/sgro/sgrolab").exists():
        input_base_dir = Path("/groups/sgro/sgrolab/jennifer/mhat/data")
        seg_base_dir = Path("/groups/sgro/sgrolab/jennifer/mhat/experiments/segmentation")
        flow_base_dir = Path("/groups/sgro/sgrolab/jennifer/mhat/experiments/opticalflow")
        tracking_base_dir = Path("/groups/sgro/sgrolab/jennifer/mhat/experiments/tracking")
    else:
        input_base_dir = Path("/Volumes/sgro/sgrolab/jennifer/mhat/data")
        seg_base_dir = Path("/Volumes/sgro/sgrolab/jennifer/mhat/experiments/segmentation")
        flow_base_dir = Path("/Volumes/sgro/sgrolab/jennifer/mhat/experiments/opticalflow")
        tracking_base_dir = Path("/Volumes/sgro/sgrolab/jennifer/mhat/experiments/tracking")

    raw_cells_zarr_path = Path(input_base_dir / experiment / f"{dataset}.zarr")
    frag_zarr_path = Path(seg_base_dir / experiment / dataset / seg_result / 'data.zarr')
    flow_2d_zarr_path = Path(flow_base_dir / experiment / dataset / "opticalflow_2d" / flow_result / "flow.zarr") 
    flow_3d_zarr_path = Path(flow_base_dir / experiment / dataset / "opticalflow_3d" / flow_result / "flow.zarr") 
    track_seg_zarr_path = Path(tracking_base_dir / experiment / dataset / exp_uid / 'pred_seg.zarr')

    if all(path.exists() is False for path in [raw_cells_zarr_path, frag_zarr_path, flow_2d_zarr_path, flow_3d_zarr_path, track_seg_zarr_path]):
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

    if flow_2d_zarr_path.exists():
        flow_2d = zarr.open(flow_2d_zarr_path, mode='r')
        flow_2d = flow_2d['flow_frames_XY'][:, ...]
        print(f"2D Flow shape: {flow_2d.shape}, dtype: {flow_2d.dtype}")
        viewer.add_image(flow_2d, name='flow_2d', blending='additive', scale=scale)
    else:
        print(f"Warning: 2D flow path {flow_2d_zarr_path} does not exist.")

    if flow_3d_zarr_path.exists():
        flow_3d = zarr.open(flow_3d_zarr_path, mode='r')
        flow_3d_xy = flow_3d['flow_frames_XY'][:, ...]
        print(f"3D Flow XY shape: {flow_3d_xy.shape}, dtype: {flow_3d_xy.dtype}")
        viewer.add_image(flow_3d_xy, name='flow_3d_xy', blending='additive', scale=scale)

        flow_3d_z = flow_3d['flow_frames_Z'][:, ...]
        print(f"3D Flow Z shape: {flow_3d_z.shape}, dtype: {flow_3d_z.dtype}")
        viewer.add_image(flow_3d_z, name='flow_3d_z', colormap='berlin', blending='additive', scale=scale)
    else:
        print(f"Warning: 3D flow path {flow_3d_zarr_path} does not exist.")

    # Load tracking data if it exists
    track_data_zarr_path = Path(tracking_base_dir / experiment / dataset / exp_uid / 'pred_tracks.zarr')
    gt_track_data_zarr_path = Path(tracking_base_dir / experiment / dataset / exp_uid / 'correct_tracks.zarr')

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
            tracks_viewer.tracks_list.add_tracks(tracks, dataset)
            print(f"Successfully loaded tracks: {tracks}")
            if tracks_viewer.tracking_layers.tracks_layer is not None:
                tracks_viewer.tracking_layers.tracks_layer.tail_length = 4
                tracks_viewer.tracking_layers.tracks_layer.tail_width = 1.0
                tracks_viewer.tracking_layers.points_layer.visible = False

        except Exception as e:
            print(f"Failed to load tracks: {e}")
    else:
        print(f"Warning: Track data path {track_data_zarr_path} does not exist.")

    if gt_track_data_zarr_path.exists():
        print(f"Loading GT tracks from {gt_track_data_zarr_path}")
        
        # Load GT tracks using import_from_geff
        try:
            name_map = {
                "time": "time",
                "x": "x", 
                "y": "y",
                "z": "z",
                "id": "track_id",
            }
            
            gt_tracks = import_from_geff(
                gt_track_data_zarr_path,
                name_map,
                segmentation_path=None,
                scale=scale,
            )
            # Add GT tracks to the TracksViewer
            tracks_viewer = TracksViewer.get_instance(viewer)
            tracks_viewer.tracks_list.add_tracks(gt_tracks, "GT_" + dataset)
            print(f"Successfully loaded GT tracks: {gt_tracks}")

            if tracks_viewer.tracking_layers.tracks_layer is not None:
                tracks_viewer.tracking_layers.tracks_layer.tail_length = 4
                tracks_viewer.tracking_layers.tracks_layer.tail_width = 1.0
                tracks_viewer.tracking_layers.points_layer.visible = False

        except Exception as e:
            print(f"Failed to load GT tracks: {e}")
    else:
        print("No ground truth tracks available.")

    if track_seg_zarr_path.exists():
        track_seg = zarr.open(track_seg_zarr_path, mode='r')
        track_seg = track_seg[:]
        print(f"Tracked segmentation shape: {track_seg.shape}, dtype: {track_seg.dtype}")
        viewer.add_labels(track_seg, name='track_seg', opacity=0.25, scale=scale)
    else:
        print(f"Warning: Track segmentation path {track_seg_zarr_path} does not exist.")
      
    napari.run()

if __name__ == '__main__':
    path_to_config = "Y:\\jennifer\\mhat\\experiments\\tracking\\NC281-Fl2mSiH2B\\03_nuclei\\2026-02-13_13-45-36\\config.toml"
    track_config = toml.load(path_to_config)
    main(track_config, compute=True)