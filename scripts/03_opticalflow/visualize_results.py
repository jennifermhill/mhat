from pathlib import Path
import numpy as np
import toml
import napari
import zarr
import dask.array as da
from mhat.opticalflow.visualization import generate_flow_frames

def main(config, compute: bool = False):
    experiment = config["experiment"]
    dataset = config["dataset"]
    exp_uid = config["exp_uid"]

    if Path("Y:\\").exists():
        input_base_dir = Path("Y:\\jennifer\\mhat\\data")
        output_base_dir = Path("Y:\\jennifer\\mhat\\experiments\\opticalflow")
    else:
        input_base_dir = Path("/groups/sgro/sgrolab/jennifer/mhat/data")
        output_base_dir = Path("/groups/sgro/sgrolab/jennifer/mhat/experiments/opticalflow")

    path_to_raw = Path(input_base_dir / experiment / f"{dataset}.zarr")
    path_to_2d = Path(output_base_dir / experiment / dataset / "opticalflow_2d" / exp_uid / "flow.zarr")
    path_to_3d = Path(output_base_dir / experiment / dataset / "opticalflow_3d" / exp_uid / "flow.zarr")

    viewer = napari.Viewer()

    if path_to_raw.exists():
        raw_img = da.from_zarr(path_to_raw)
        if compute:
            raw_img = raw_img.compute()
        raw_img = raw_img[:, 0, ...] # Remove channel dimension
        print(f"Raw image shape: {raw_img.shape}")
        viewer.add_image(raw_img, name="Raw Image")
    else:
        print(f"Raw data not found at {path_to_raw}")

    if path_to_2d.exists():
        flow_zarr = zarr.open(path_to_2d, mode='a')

        path_to_flow_frames_2d = path_to_2d / "flow_frames_XY"
        if not path_to_flow_frames_2d.exists(): 
            print(f"Flow frames path does not exist. Creating at: {path_to_2d / 'flow_frames_XY'}")
            flow_zarr = zarr.open(path_to_2d, mode='a')
            generate_flow_frames(flow_zarr, scale_factor=1, color_wheel=True)

        flow_frames_2d = da.from_zarr(path_to_flow_frames_2d)
        # if flow_frames_2d_zarr.nchunks_initialized == 0:
        #     print(f"Flow frames dataset is empty (no chunks initialized). Regenerating...")
        #     del flow_zarr['flow_frames_XY']
        #     flow_zarr = zarr.open(path_to_2d, mode='a')
        #     generate_flow_frames(flow_zarr, scale_factor=1, color_wheel=True)

        if compute:
            flow_frames_2d = flow_frames_2d.compute()
        print(f"2D Flow frames shape: {flow_frames_2d.shape}")
        viewer.add_image(flow_frames_2d, name="2D Flow Frames", blending='additive')
    else:
        print(f"2D optical flow data not found at {path_to_2d}")

    if path_to_3d.exists():
        flow_zarr = zarr.open(path_to_3d, mode='a')

        path_to_flow_frames_3d_XY = path_to_3d / "flow_frames_XY"
        if not path_to_flow_frames_3d_XY.exists(): 
            print(f"Flow frames path does not exist. Creating at: {path_to_3d / 'flow_frames_XY'}")
            generate_flow_frames(flow_zarr, scale_factor=1, color_wheel=True)
        flow_frames_3d_XY = da.from_zarr(path_to_flow_frames_3d_XY)
        # if flow_frames_3d_XY.nchunks_initialized == 0:
        #     print(f"3D Flow frames dataset is empty (no chunks initialized). Regenerating...")
        #     del flow_zarr['flow_frames_XY']
        #     generate_flow_frames(flow_zarr, scale_factor=1, color_wheel=True)

        if compute:
            flow_frames_3d_XY = flow_frames_3d_XY.compute()
        print(f"3D Flow frames XY shape: {flow_frames_3d_XY.shape}")
        viewer.add_image(flow_frames_3d_XY, name="3D Flow Frames (XY component)", blending='additive')

        path_to_flow_frames_3d_Z = path_to_3d / "flow_frames_Z"
        if not path_to_flow_frames_3d_Z.exists():
            print(f"Flow frames Z path does not exist. Creating at: {path_to_flow_frames_3d_Z}")
            flow_raw = da.from_zarr(path_to_3d / "flow_raw")
            T, Z, Y, X, _ = flow_raw.shape
            flow_zarr.create_dataset('flow_frames_Z', shape=(T, Z, Y, X), chunks=(1, Z, Y, X), dtype=np.float32)
            flow_zarr['flow_frames_Z'][:] = flow_raw[..., 2]
        flow_frames_3d_Z = da.from_zarr(path_to_3d / "flow_frames_Z")
        # if flow_frames_3d_Z.nchunks_initialized == 0:
        #     print(f"3D Flow frames Z dataset is empty (no chunks initialized). Regenerating...")
        #     del flow_zarr['flow_frames_Z']
        #     flow_raw = da.from_zarr(path_to_3d / "flow_raw")
        #     T, Z, Y, X, _ = flow_raw.shape
        #     flow_zarr.create_dataset('flow_frames_Z', shape=(T, Z, Y, X), chunks=(1, Z, Y, X), dtype=np.float32)
        #     flow_zarr['flow_frames_Z'][:] = flow_raw[..., 2]

        if compute:
            flow_frames_3d_Z = flow_frames_3d_Z.compute()
        print(f"3D Flow frames Z shape: {flow_frames_3d_Z.shape}")
        viewer.add_image(flow_frames_3d_Z, name="3D Flow Frames (Z component)", colormap='berlin', blending='additive')
    else:
        print(f"3D optical flow data not found at {path_to_3d}")

    napari.run()


if __name__ == "__main__":
    path_to_config = "Y:\\jennifer\\mhat\\experiments\\opticalflow\\NC281-Fl2mSiH2B\\01_nuclei\\opticalflow_2d\\2026-01-22_12-51-55\\config.toml"
    config = toml.load(path_to_config)
    main(config, compute=True)