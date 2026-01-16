import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

from opticalflow3D.helpers.farneback_functions import farneback_3d
from mhat.opticalflow.utils import enhance_contrast_AHE


def compute_farneback_flow_2d(config, frames, output_zarr):
    print("Normalizing all frames...")
    max_val = frames.max()
    min_val = frames.min()
    frames_norm = ((frames - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    T, Z, Y, X = output_zarr['flow_raw'].shape

    prev_frame = frames_norm[0].compute()
    if config["enhance_contrast"]:
        prev_frame = enhance_contrast_AHE(prev_frame)

    prev_flow = None

    ds_factor = config.get('downsample_factor', 1)

    for i in tqdm(range(1, frames_norm.shape[0]), desc="Computing optical flow"):

        curr_frame = frames_norm[i].compute()
        if config["enhance_contrast"]:
            curr_frame = enhance_contrast_AHE(curr_frame)

        flow = cv2.calcOpticalFlowFarneback(
            prev_frame,
            curr_frame,
            None,
            pyr_scale=config['pyr_scale'],
            levels=config['levels'],
            winsize=config['winsize'],
            iterations=config['iterations'],
            poly_n=config['poly_n'],
            poly_sigma=config['poly_sigma'],
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        if prev_flow is not None and config['temporal_smoothing'] is True:
            alpha = config['temporal_smoothing_sigma']
            flow = alpha * flow + (1 - alpha) * prev_flow

        if ds_factor > 1:
            flow_upsampled = np.zeros((Y, X, 2), dtype=np.float32)

            # Zoom factors for Y, X, and channels
            zoom_factors = (ds_factor, ds_factor, 1)

            flow_upsampled = zoom(flow, zoom_factors, order=1)  # order=1 is bilinear

            # Scale the flow magnitudes by the zoom factor
            flow_upsampled[..., 0] *= ds_factor  # vy
            flow_upsampled[..., 1] *= ds_factor  # vx

            flow = flow_upsampled

        output_zarr['flow_raw'][i-1, ...] = flow.astype(np.float32)
        prev_frame = curr_frame

        prev_flow = flow

    return output_zarr['flow_raw']  # return the computed flow dataset


def compute_farneback_flow_3d(config, frames, output_zarr):
    print("Normalizing all frames...")
    max_val = frames.max()
    min_val = frames.min()
    frames_norm = ((frames - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    T, Z, Y, X = output_zarr['flow_raw'].shape

    prev_frame = frames_norm[0].compute()
    if config['enhance_contrast']:
        prev_frame = enhance_contrast_AHE(prev_frame)

    prev_flow = None

    ds_factor = config.get('downsample_factor', [1, 1, 1])

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

        if any(f > 1 for f in ds_factor):
            # Zoom each component separately
            flow_upsampled = np.zeros((Z, Y, X, 3), dtype=np.float32)
            
            # Zoom factors: (Z_factor, Y_factor, X_factor, 1 for channels)
            zoom_factors = (ds_factor[0], ds_factor[1], ds_factor[2], 1)
            
            flow_upsampled = zoom(flow, zoom_factors, order=1)  # order=1 is bilinear
            
            # Scale the flow magnitudes by the zoom factors
            flow_upsampled[..., 0] *= ds_factor[0]  # vz
            flow_upsampled[..., 1] *= ds_factor[1]  # vy
            flow_upsampled[..., 2] *= ds_factor[2]  # vx
            
            flow = flow_upsampled

        output_zarr['flow_raw'][i-1] = flow.astype(np.float32)
        output_zarr['confidence'][i-1] = confidence_np.astype(np.float32)

        prev_frame = curr_frame
        prev_flow = flow

    return output_zarr['flow_raw'], output_zarr['confidence']