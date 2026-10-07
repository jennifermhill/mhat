import argparse
from pathlib import Path
import numpy as np
import napari
import zarr
import dask.array as da

from mhat.dataset import Dataset, absolute_path
from mhat.opticalflow.utils import open_flow_raw
from mhat.utils import get_axes_metadata

def main(run_dir: Path, raw_path: Path | None = None, compute: bool = False,
         scale_factor: float = 1.0):
    # The output directory is found from where the run sits, so this works on
    # any machine; raw comes from raw.toml unless --raw overrides it. Either
    # flow kind's run directory shows both the 2D and 3D flow of that uid.
    run_dir = absolute_path(run_dir)
    if run_dir.name == "config.toml":
        run_dir = run_dir.parent
    ds = Dataset.from_run_dir(run_dir, "opticalflow")
    exp_uid = run_dir.name
    path_to_raw = raw_path or ds.raw_path
    path_to_2d = ds.flow_dir(exp_uid, "2d") / "flow.zarr"
    path_to_3d = ds.flow_dir(exp_uid, "3d") / "flow.zarr"

    viewer = napari.Viewer()

    # The flow stores carry the raw movie's axes, so the scale is available
    # even when the raw movie is not.
    scale = None
    for path in (path_to_2d, path_to_3d):
        if path.exists():
            flow_raw = zarr.open(path, mode='r')["flow_raw"]
            scale = [axis["scale"] for axis in get_axes_metadata(flow_raw)]
            break

    if path_to_raw is not None and Path(path_to_raw).exists():
        zarr_img = zarr.open(path_to_raw, mode='r')
        raw_img = da.from_zarr(path_to_raw)
        axes = get_axes_metadata(zarr_img)
        scale = [axis["scale"] for axis in axes]
        if compute:
            raw_img = raw_img.compute()
        raw_img = raw_img[:, 0, ...] # Remove channel dimension
        print(f"Raw image shape: {raw_img.shape}")
        viewer.add_image(raw_img, name="Raw Image", scale=scale)
    else:
        print(f"Raw data not found at {path_to_raw}; pass --raw to show it.")

    if path_to_2d.exists():
        flow_zarr = zarr.open(path_to_2d, mode='a')

        path_to_flow_frames_2d = path_to_2d / "flow_frames_XY"
        if not path_to_flow_frames_2d.exists():
            from mhat.opticalflow.visualization import generate_flow_frames

            print(f"Flow frames path does not exist. Creating at: {path_to_2d / 'flow_frames_XY'}")
            flow_zarr = zarr.open(path_to_2d, mode='a')
            generate_flow_frames(flow_zarr, scale_factor=scale_factor, color_wheel=True)

        flow_frames_2d = da.from_zarr(path_to_flow_frames_2d)

        if compute:
            flow_frames_2d = flow_frames_2d.compute()
        print(f"2D Flow frames shape: {flow_frames_2d.shape}")
        viewer.add_image(flow_frames_2d, name="2D Flow Frames", blending='additive', scale=scale)
    else:
        print(f"2D optical flow data not found at {path_to_2d}")

    if path_to_3d.exists():
        flow_zarr = zarr.open(path_to_3d, mode='a')

        path_to_flow_frames_3d_XY = path_to_3d / "flow_frames_XY"
        if not path_to_flow_frames_3d_XY.exists():
            from mhat.opticalflow.visualization import generate_flow_frames

            print(f"Flow frames path does not exist. Creating at: {path_to_3d / 'flow_frames_XY'}")
            generate_flow_frames(flow_zarr, scale_factor=scale_factor, color_wheel=True)
        flow_frames_3d_XY = da.from_zarr(path_to_flow_frames_3d_XY)

        if compute:
            flow_frames_3d_XY = flow_frames_3d_XY.compute()
        print(f"3D Flow frames XY shape: {flow_frames_3d_XY.shape}")
        viewer.add_image(flow_frames_3d_XY, name="3D Flow Frames (XY component)", blending='additive', scale=scale)

        path_to_flow_frames_3d_Z = path_to_3d / "flow_frames_Z"
        if not path_to_flow_frames_3d_Z.exists():
            print(f"Flow frames Z path does not exist. Creating at: {path_to_flow_frames_3d_Z}")
            flow_raw = open_flow_raw(flow_zarr, dask=True)  # (vz, vy, vx), legacy stores included
            T, Z, Y, X, _ = flow_raw.shape
            flow_zarr.create_dataset('flow_frames_Z', shape=(T, Z, Y, X), chunks=(1, Z, Y, X), dtype=np.float32)
            flow_zarr['flow_frames_Z'][:] = flow_raw[..., 0]  # vz is component 0
        flow_frames_3d_Z = da.from_zarr(path_to_flow_frames_3d_Z)

        if compute:
            flow_frames_3d_Z = flow_frames_3d_Z.compute()
        print(f"3D Flow frames Z shape: {flow_frames_3d_Z.shape}")
        viewer.add_image(flow_frames_3d_Z, name="3D Flow Frames (Z component)", colormap='berlin', blending='additive', scale=scale)
    else:
        print(f"3D optical flow data not found at {path_to_3d}")

    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View optical flow results in napari: one run's 2D and 3D flow."
    )
    parser.add_argument(
        "run", type=Path,
        help="a flow run, <output_dir>/opticalflow/<2d|3d>/<uid> (or its config.toml)",
    )
    parser.add_argument(
        "--raw", type=Path, default=None,
        help="raw movie to show, if the path in raw.toml is not valid on this machine",
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="load the flow arrays into memory instead of viewing them lazily",
    )
    parser.add_argument(
        "--scale-factor", type=float, default=1.0,
        help="scale factor applied to the flow vectors when displaying (default: 1.0)",
    )
    args = parser.parse_args()
    main(args.run, raw_path=args.raw, compute=args.compute,
         scale_factor=args.scale_factor)
