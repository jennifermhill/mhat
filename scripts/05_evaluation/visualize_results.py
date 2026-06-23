from pathlib import Path
import toml

import napari
import zarr
import numpy as np
import dask.array as da
from funtracks.import_export import import_from_geff
from traccuracy import TrackingGraph
from divisualisation import Divisualisation

# import_from_geff(..., scale=scale) multiplies node positions into world
# units. divis displays the raw image with z stretched by z_scale (x/y stay in
# pixel units) but renders the tracks *unscaled* at their raw coordinate values.
# So to make the tracks line up with the displayed image we convert x/y back to
# pixel units (divide by their scale) but leave z in world units, since the
# image's displayed z (z_pixel * z_scale) already equals the world-unit z.
# spatial_scale is the position part of the zarr `scale` (no time axis), in the
# same axis order as the stored "pos" array, i.e. [z_scale, y_scale, x_scale]
# (or [y_scale, x_scale] in 2D).
def to_divis_coords(tracking_graph, spatial_scale, pos_key="pos"):
    spatial_scale = np.asarray(spatial_scale, dtype=float)
    for _, data in tracking_graph.graph.nodes(data=True):
        pos = np.asarray(data[pos_key], dtype=float)
        divisor = spatial_scale[-len(pos):].copy()
        if len(pos) >= 3:
            divisor[-3] = 1.0  # leave z in world units; divis doesn't scale tracks
        data[pos_key] = pos / divisor


# divis reads per-node props "t", "x", "y", "z" directly off the networkx
# nodes (see divisualisation.visualize_edge_errors). funtracks stores the
# frame under "time" and position as a composite "pos" array, so copy them
# into the names divis expects.
def add_divis_node_props(tracking_graph, frame_key="time", pos_key="pos"):
    for _, data in tracking_graph.graph.nodes(data=True):
        data["t"] = data[frame_key]
        pos = data[pos_key]
        data["x"] = pos[-1]
        data["y"] = pos[-2]
        if len(pos) >= 3:
            data["z"] = pos[-3]

def main(config, compute: bool = False):

    experiment = config["experiment"]
    dataset = config["dataset"]
    track_result = config["exp_uid"]

    if Path("Y:\\").exists():
        raw_data_dir = Path("Y:\\jennifer\\mhat\\data")
        input_base_dir = Path("Y:\\jennifer\\mhat\\experiments\\evaluation")
    elif Path("/Volumes/sgrolab").exists():
        raw_data_dir = Path("/Volumes/sgrolab/jennifer/mhat/data")
        input_base_dir = Path("/Volumes/sgrolab/jennifer/mhat/experiments/")
    else:
        raw_data_dir = Path("/groups/sgro/sgrolab/jennifer/mhat/data")
        input_base_dir = Path("/groups/sgro/sgrolab/jennifer/mhat/experiments/")

    path_to_raw = Path(raw_data_dir / experiment / f"{dataset}.zarr")
    gt_data_dir = input_base_dir / "tracking" / experiment / dataset
    pred_data_dir = input_base_dir / "tracking" / experiment / dataset / track_result

    zarr_img = zarr.open(path_to_raw, mode='r')
    img = da.from_zarr(path_to_raw)
    if compute:
        img = img.compute()
    img = img[:, 0, ...] # Remove channel dimension
    print(f"Raw image shape: {img.shape}")
    
    axes = zarr_img.attrs.get("axes", None)
    if axes is not None:
        axes = [axis for axis in axes if axis.get("name") != "channel"]
        scale = [axis["scale"] for axis in axes if axis.get("scale") is not None]
        z_scale = scale[1]
    else:
        scale = [1.0, 1.0, 1.0, 1.0]
        z_scale = 1.0

    name_map = {
        "time": "time", # required key name "time" mapped to existing node prop "time"
        "x": "x", 
        "y": "y",
        "z": "z",
        "id": "track_id",    # track_id stays constant across frames
    }
    
    print(f"Loading gt tracks from {gt_data_dir / 'correct_tracks.zarr'}")
    gt_seg_path = gt_data_dir / "correct_seg.zarr"
    gt_seg_path = gt_seg_path if gt_seg_path.exists() else None
    gt_tracks = import_from_geff(
        gt_data_dir / "correct_tracks.zarr",
        name_map,
        segmentation_path=gt_seg_path,
        scale=scale,
    )
    
    gt_graph = TrackingGraph(
        graph=gt_tracks.graph,
        frame_key="time", # matches with geff name mapping
        label_key="track_id",
        location_keys="pos",
        segmentation=gt_tracks.segmentation,
    )

    print(f"Loading pred tracks from {pred_data_dir / 'pred_tracks.zarr'}")
    pred_seg_path = pred_data_dir / "pred_seg.zarr"
    pred_seg_path = pred_seg_path if pred_seg_path.exists() else None
    pred_tracks = import_from_geff(
        pred_data_dir / "pred_tracks.zarr",
        name_map,
        segmentation_path=pred_seg_path,
        scale=scale,
    )

    pred_graph = TrackingGraph(
        graph=pred_tracks.graph,
        frame_key="time",
        label_key="track_id",
        location_keys="pos",
        segmentation=pred_tracks.segmentation,
    )

    gt_masks = gt_tracks.segmentation
    pred_masks = pred_tracks.segmentation

    # scale is [time, z, y, x]; drop the time axis to get the spatial scale
    # that was applied to the node positions on import.
    spatial_scale = scale[1:]
    to_divis_coords(gt_graph, spatial_scale)
    to_divis_coords(pred_graph, spatial_scale)

    add_divis_node_props(gt_graph)
    add_divis_node_props(pred_graph)

    v = napari.current_viewer()
    if v is not None:
        v.close()
    v = napari.Viewer()
    for layer in v.layers:
        v.layers.remove(layer)
    v.theme = "dark"

    divis = Divisualisation(
        z_scale=z_scale,
        time_scale=1,
    )

    v = divis.visualize_gt(
        v,
        x=img,
        masks=gt_masks, # frame key is "time", needs to be "t"
        # networkx graph at traccuracy.TrackingGraph.graph
        gt_graph=gt_graph.graph, 
        pred_graph=pred_graph.graph,
    )

    v = divis.visualize_edge_errors(
        viewer=v,
        gt_graph=gt_graph,
        pred_graph=pred_graph,
        masks_original=gt_masks, # frame key is "time", needs to be "t"
        masks_tracked=pred_masks, # frame key is "time", needs to be "t"
    )

    v.dims.set_current_step(0, 190)
    v.camera.angles = (27.919484296382873, -49.86671510905139, -35.8190766165135)
    v.camera.perspective = 27

    divis.render(v, name="divisualisation_3d", steps=96)

if __name__ == "__main__":
    path_to_config = "/Volumes/sgrolab/jennifer/mhat/experiments/evaluation/NC281-sparse-label/01_nuclei_denoised/2026-04-29_11-36-23_R1/tracking_config.toml"
    config = toml.load(path_to_config)
    main(config, compute=False)