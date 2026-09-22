import argparse
import toml
from pathlib import Path
import datetime

import dask.array as da
import numpy as np
import zarr
from tqdm import tqdm

from mhat.opticalflow.utils import create_flow_store, frame_average, rename_flow_uid
from mhat.opticalflow.farneback import compute_farneback_flow_2d, compute_farneback_flow_3d
from mhat.opticalflow.visualization import generate_flow_frames
from mhat.utils import get_axes_metadata


def calculate_flow(config, zarr_path: Path, output_dir: Path, do_3d: bool = False):
    '''
    config: dictionary of configuration parameters
    zarr_path: path to zarr directory containing .zarray
    output_dir: directory to save output optical flow zarr
    '''
    zarr_root = zarr.open(zarr_path, mode='r')
    axes = get_axes_metadata(zarr_root)

    zarr_img = da.from_zarr(zarr_path)
    print(f"Zarr image shape: {zarr_img.shape}")

    # Raw is (t, c, *spatial): (t, c, z, y, x) for a 3D movie, (t, c, y, x) for
    # a 2D one. Everything below is written as (t, *spatial, components).
    T = zarr_img.shape[0]
    spatial_shape = zarr_img.shape[2:]
    ndim = len(spatial_shape)
    zarr_img = zarr_img[:, 0, ...]  # remove channel dimension

    # One chunk per displayed plane: 1 along every axis before the last two.
    plane_chunks = (1,) * (ndim - 1) + spatial_shape[-2:]

    if do_3d:
        assert ndim == 3, (
            f"3D optical flow needs a 3D movie, but {zarr_path} is {ndim}D. "
            "Set do_3d = false for 2D data; 2D flow is all it has."
        )
        flow_function = compute_farneback_flow_3d
    else:
        flow_function = compute_farneback_flow_2d

    # flow_raw is (t, *spatial, components) with the components in axis order:
    # (vz, vy, vx) for 3D flow, (vy, vx) for 2D flow (per z slice on a 3D
    # movie), and that order recorded on the array.
    output_zarr = create_flow_store(
        output_dir / 'flow.zarr', T, spatial_shape, plane_chunks, axes,
        do_3d=do_3d,
    )
    
    flow = flow_function(config, zarr_img, output_zarr)
    
    frame_averaging = config['hyperparams'].get('frame_averaging', 0)
    if frame_averaging > 0:
        frame_average(flow_zarr=output_zarr, frame_avg=frame_averaging)

    generate_flow_frames(flow_zarr=output_zarr, color_wheel=True)

    if do_3d:
        output_zarr.create_dataset('flow_frames_Z', shape=(T, *spatial_shape), chunks=plane_chunks, dtype=np.float32)
        output_zarr['flow_frames_Z'][:] = flow[..., 0]  # vz is component 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config['input_base_dir'])
    output_base_dir = Path(config['output_base_dir'])
    experiment: str = config['experiment']
    dataset: str = config['dataset']
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    data_dir = input_base_dir / experiment / f"{dataset}.zarr"
    print(f"Loading data from {data_dir}")
    assert data_dir.is_dir()

    current_datetime = datetime.datetime.now()
    exp_uid = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    config["exp_uid"] = exp_uid

    # Check for rerun uid
    rerun_uid = config.get("rerun_uid", "")

    if config["2d"]["do_2d"]:
        print("Calculating 2D optical flow...")

        output_dir = output_base_dir / experiment / dataset / "opticalflow_2d" / exp_uid
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving 2D optical flow results to {output_dir}")

        config_filepath = output_dir / "config.toml"
        with open(config_filepath, 'w') as config_file:
            toml.dump(config, config_file)

        calculate_flow(config["2d"], data_dir, output_dir, do_3d=False)
    elif rerun_uid:
        prev_flow_dir = output_base_dir / experiment / dataset / "opticalflow_2d" / rerun_uid
        rename_flow_uid(prev_flow_dir, exp_uid)

    if config["3d"]["do_3d"]:
        print("Calculating 3D optical flow...")
        output_dir = output_base_dir / experiment / dataset / "opticalflow_3d" / exp_uid
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving 3D optical flow results to {output_dir}")

        config_filepath = output_dir / "config.toml"
        with open(config_filepath, 'w') as config_file:
            toml.dump(config, config_file)

        calculate_flow(config["3d"], data_dir, output_dir, do_3d=True)
    elif rerun_uid:
        prev_flow_dir = output_base_dir / experiment / dataset / "opticalflow_3d" / rerun_uid
        rename_flow_uid(prev_flow_dir, exp_uid)

    