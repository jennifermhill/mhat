import argparse
import toml
from pathlib import Path
import datetime

import dask.array as da
import numpy as np
import zarr
from tqdm import tqdm

from opticalflow3D.helpers.farneback_functions import farneback_3d
from mhat.opticalflow.utils import enhance_contrast_AHE
from mhat.opticalflow.visualization import generate_flow_frames

def compute_farneback_flow(config, frames, output_zarr):
    print("Normalizing all frames...")
    max_val = frames.max()
    min_val = frames.min()
    frames_norm = ((frames - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    prev_frame = frames_norm[0].compute()
    if config['enhance_contrast']:
        prev_frame = enhance_contrast_AHE(prev_frame)

    prev_flow = None

    for i in tqdm(range(1, frames_norm.shape[0]), desc="Computing optical flow"):

        curr_frame = frames_norm[i].compute()
        if config['enhance_contrast']:
            curr_frame = enhance_contrast_AHE(curr_frame)

        vx, vy, vz, confidence = farneback_3d(
            prev_frame,
            curr_frame,
            iters=config['iterations'],
            num_levels=config['levels'],
            scale=config['pyr_scale'],
            spatial_size=config['poly_n'],
            sigma_k=config['poly_sigma'],
            filter_type=config['filter_type'],
            filter_size=config['winsize'],
            presmoothing=config['presmoothing']
        )

        vx_np = vx.cpu().numpy()
        vy_np = vy.cpu().numpy()
        vz_np = vz.cpu().numpy()
        confidence_np = confidence.cpu().numpy()
        flow = np.stack((vx_np, vy_np, vz_np), axis=-1)

        if prev_flow is not None and config['temporal_smoothing'] is not None:
            alpha = config['temporal_smoothing']
            flow = alpha * flow + (1 - alpha) * prev_flow

        output_zarr['flow_raw'][i-1] = flow.astype(np.float32)
        output_zarr['confidence'][i-1] = confidence_np.astype(np.float32)

        prev_frame = curr_frame
        prev_flow = flow

    return output_zarr['flow_raw'], output_zarr['confidence']


def main(config, zarr_path: Path, output_dir: Path):
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

    frames = zarr_img[:, config['channel'], 32:132, 530:920, 400:750]
    frames_ds = frames[:, ::ds_factor[0], ::ds_factor[1], ::ds_factor[2]]

    T, Z, Y, X = frames_ds.shape

    output_zarr = zarr.open(output_dir / 'flow.zarr', 'w')
    output_zarr.create_dataset('flow_raw', shape=(T-1, Z, Y, X, 3), chunks=(1, Z, Y, X, 3), dtype=np.float32)
    output_zarr.create_dataset('confidence', shape=(T-1, Z, Y, X), chunks=(1, Z, Y, X), dtype=np.float32)

    flow, confidence = compute_farneback_flow(config, frames_ds, output_zarr)

    output_zarr['confidence'][:] = confidence

    if config['frame_averaging'] > 0:
        for i in range(1, T-1):
            start_idx = max(0, i - config['frame_averaging'] // 2)
            end_idx = min(T-1, i + config['frame_averaging'] // 2 + 1)
            flow[i, ...] = np.mean(output_zarr['flow_raw'][start_idx:end_idx, ...], axis=0)
        output_zarr['flow_raw'][:] = flow

    generate_flow_frames(output_zarr, color_wheel=config.get('color_wheel', False))

    output_zarr.create_dataset('flow_frames_Z', shape=(T-1, Z, Y, X), chunks=(1, Z, Y, X), dtype=np.float32)
    output_zarr['flow_frames_Z'][:] = flow[..., 2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    data_dir = Path(config['data_dir'])
    dataset: str = config['dataset']
    zarr_path = data_dir / dataset / f"{dataset}.zarr"
    assert zarr_path.exists(), f"Zarr path does not exist: {zarr_path}"

    # Check if zarr directory contains .zarray
    if not (zarr_path / '.zarray').exists():
        zarr_path = zarr_path / '0' / '0'

    current_datetime = datetime.datetime.now()
    exp_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    config["exp_name"] = exp_name

    output_dir = data_dir / dataset / "opticalflow_3d" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing output to {output_dir}")

    main(config, zarr_path, output_dir)