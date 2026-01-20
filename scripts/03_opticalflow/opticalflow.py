import argparse
import toml
from pathlib import Path
import datetime

import dask.array as da
import numpy as np
import zarr
import cv2
from tqdm import tqdm

from mhat.opticalflow.utils import rename_flow_uid
from mhat.opticalflow.farneback import compute_farneback_flow_2d, compute_farneback_flow_3d
from mhat.opticalflow.visualization import generate_flow_frames


def calculate_flow_2d(config, zarr_path: Path, output_dir: Path):
    '''
    config: dictionary of configuration parameters
    zarr_path: path to zarr directory containing .zarray
    output_dir: directory to save output optical flow zarr
    '''
    config_filepath = output_dir / "config.toml"
    with open(config_filepath, 'w') as config_file:
        toml.dump(config, config_file)

    zarr_img = da.from_zarr(zarr_path)
    print(f"Zarr image shape: {zarr_img.shape}")

    ds_factor = config.get('downsample_factor', 1)

    T, Z, Y, X = zarr_img.shape

    output_zarr = zarr.open(output_dir / 'flow.zarr', 'w')
    output_zarr.create_dataset('flow_raw', shape=(T-1, Z, Y, X, 2), chunks=(1, Z, Y, X, 2), dtype=np.float32)
    
    img_ds = zarr_img[:, :, ::ds_factor, ::ds_factor]
    
    for z_slice in range(Z):
        print(f"Computing flow for Z slice {z_slice}/{Z-1}")
        frames = img_ds[:, z_slice, :, :]

        flow = compute_farneback_flow_2d(config, frames)
        
        if config['frame_averaging'] > 0:
            for i in range(1, T-1):
                start_idx = max(0, i - config['frame_averaging'] // 2)
                end_idx = min(T-1, i + config['frame_averaging'] // 2 + 1)
                flow[i, z_slice, ...] = np.mean(output_zarr['flow_raw'][start_idx:end_idx, z_slice, ...], axis=0)
        output_zarr['flow_raw'][:, z_slice, :, :] = flow[:, z_slice, :, :]

    generate_flow_frames(flow_zarr=output_zarr, color_wheel=True)

def calculate_flow_3d(config, zarr_path: Path, output_dir: Path):
    '''
    config: dictionary of configuration parameters
    zarr_path: path to zarr directory containing .zarray
    output_dir: directory to save output optical flow zarr
    '''
    config_filepath = output_dir / "config.toml"
    with open(config_filepath, 'w') as config_file:
        toml.dump(config, config_file)

    zarr_img = da.from_zarr(zarr_path)
    print(f"Zarr image shape: {zarr_img.shape}")

    ds_factor = config.get('downsample_factor', [1, 1, 1])

    # T = 20 # for now

    frames = zarr_img[:, 32:132, 530:920, 400:750]
    frames_ds = frames[:, ::ds_factor[0], ::ds_factor[1], ::ds_factor[2]]

    T, Z, Y, X = frames_ds.shape

    output_zarr = zarr.open(output_dir / 'flow.zarr', 'w')
    output_zarr.create_dataset('flow_raw', shape=(T-1, Z, Y, X, 3), chunks=(1, Z, Y, X, 3), dtype=np.float32)
    output_zarr.create_dataset('confidence', shape=(T-1, Z, Y, X), chunks=(1, Z, Y, X), dtype=np.float32)

    flow, confidence = compute_farneback_flow_3d(config, frames_ds, output_zarr)

    if config['frame_averaging'] > 0:
        for i in range(1, T-1):
            start_idx = max(0, i - config['frame_averaging'] // 2)
            end_idx = min(T-1, i + config['frame_averaging'] // 2 + 1)
            flow[i, ...] = np.mean(output_zarr['flow_raw'][start_idx:end_idx, ...], axis=0)
        output_zarr['flow_raw'][:] = flow

    generate_flow_frames(output_zarr, color_wheel=True)

    output_zarr.create_dataset('flow_frames_Z', shape=(T-1, Z, Y, X), chunks=(1, Z, Y, X), dtype=np.float32)
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

    output_dir = output_base_dir / experiment / dataset / "opticalflow_2d" / exp_uid
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing output to {output_dir}")

    if config["2d"]["do_2d"]:
        print("Calculating 2D optical flow...")
        calculate_flow_2d(config["2d"], data_dir, output_dir)
    else:
        print(f"Renaming previous flow run {rerun_uid} to current experiment uid {exp_uid}")
        prev_flow_dir = output_base_dir / experiment / dataset / "opticalflow_2d" / rerun_uid
        rename_flow_uid(prev_flow_dir, exp_uid)

    if config["3d"]["do_3d"]:
        print("Calculating 3D optical flow...")
        calculate_flow_3d(config["3d"], data_dir, output_dir)
    else:
        print(f"Renaming previous flow run {rerun_uid} to current experiment uid {exp_uid}")
        prev_flow_dir = output_base_dir / experiment / dataset / "opticalflow_3d" / rerun_uid
        rename_flow_uid(prev_flow_dir, exp_uid)

    