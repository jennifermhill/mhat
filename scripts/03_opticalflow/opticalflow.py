import argparse
import toml
from pathlib import Path
import datetime

import dask.array as da
import numpy as np
import zarr
from tqdm import tqdm

from mhat.opticalflow.utils import rename_flow_uid, frame_average
from mhat.opticalflow.farneback import compute_farneback_flow_2d, compute_farneback_flow_3d
from mhat.opticalflow.lucaskanade import compute_lucaskanade_flow_3d
from mhat.opticalflow.visualization import generate_flow_frames
from mhat.utils import get_axes_metadata


def calculate_flow(config, zarr_path: Path, output_dir: Path, do_3d: bool = False, do_lk: bool = False):
    '''
    config: dictionary of configuration parameters
    zarr_path: path to zarr directory containing .zarray
    output_dir: directory to save output optical flow zarr
    '''
    zarr_root = zarr.open(zarr_path, mode='r')
    axes = get_axes_metadata(zarr_root)

    zarr_img = da.from_zarr(zarr_path)
    print(f"Zarr image shape: {zarr_img.shape}")

    T, C, Z, Y, X = zarr_img.shape
    zarr_img = zarr_img[:, 0, ...]  # remove channel dimension

    output_zarr = zarr.open(output_dir / 'flow.zarr', 'w')

    if do_3d:
        output_zarr.create_dataset('flow_raw', shape=(T, Z, Y, X, 3), chunks=(1, 1, Y, X, 3), dtype=np.float32)
        output_zarr.create_dataset('confidence', shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype=np.float32)
        flow_function = compute_farneback_flow_3d
    elif do_lk:
        output_zarr.create_dataset('flow_raw', shape=(T, Z, Y, X, 3), chunks=(1, 1, Y, X, 3), dtype=np.float32)
        output_zarr.create_dataset('confidence', shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype=np.float32)
        flow_function = compute_lucaskanade_flow_3d
    else:
        output_zarr.create_dataset('flow_raw', shape=(T, Z, Y, X, 2), chunks=(1, 1, Y, X, 2), dtype=np.float32)
        flow_function = compute_farneback_flow_2d

    output_zarr['flow_raw'].attrs["axes"] = axes
    
    flow = flow_function(config, zarr_img, output_zarr)
    
    frame_averaging = config['hyperparams'].get('frame_averaging', 0)
    if frame_averaging > 0:
        frame_average(flow_zarr=output_zarr, frame_avg=frame_averaging)

    generate_flow_frames(flow_zarr=output_zarr, color_wheel=True)

    if do_3d or do_lk:
        output_zarr.create_dataset('flow_frames_Z', shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype=np.float32)
        output_zarr['flow_frames_Z'][:] = flow[..., 2]


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

    if config["lucaskanade"]["do_lucaskanade"]:
        print("Calculating 3D Lucas-Kanade optical flow...")
        output_dir = output_base_dir / experiment / dataset / "opticalflow_lucaskanade" / exp_uid
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving 3D Lucas-Kanade optical flow results to {output_dir}")

        config_filepath = output_dir / "config.toml"
        with open(config_filepath, 'w') as config_file:
            toml.dump(config, config_file)

        calculate_flow(config["lucaskanade"], data_dir, output_dir, do_lk=True)

    