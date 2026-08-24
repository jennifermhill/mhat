import cv2
import dask.array as da
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

from mhat.opticalflow.utils import enhance_contrast_AHE

# opticalflow3D is imported inside compute_farneback_flow_3d, not here. It is an
# optional dependency (the `flow3d` extra) that pulls in torch, and the 2D path
# below is pure OpenCV — importing it at module scope would make torch a hard
# requirement of the whole core install.


def _import_farneback_3d():
    try:
        from opticalflow3D.helpers.farneback_functions import farneback_3d
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "3D Farneback optical flow requires the 'flow3d' extra, which is not "
            "installed. Install it with:\n"
            '    pip install -e ".[flow3d]"\n'
            "Note this is the PyTorch fork at "
            "https://github.com/jennifermhill/opticalflow3d, not the cupy/CUDA "
            "package of the same name on PyPI. The 2D Farneback path "
            "(compute_farneback_flow_2d) needs none of this."
        ) from exc
    return farneback_3d


def compute_farneback_flow_2d(config, zarr_img, output_zarr):
    T, Z, Y, X = zarr_img.shape

    # Check if Y, or X dims are odd, and pad if so
    pad_y = 0
    pad_x = 0
    if Y % 2 != 0:
        pad_y = 1
    if X % 2 != 0:
        pad_x = 1
    if pad_y != 0 or pad_x != 0:
        zarr_img = da.pad(zarr_img, ((0,0),(0,0),(0,pad_y),(0,pad_x)), mode='edge')

    ds_factor = config.get('downsample_factor', 1)

    frames_ds = zarr_img[:, :, ::ds_factor, ::ds_factor]

    print("Normalizing all frames...")
    max_val = frames_ds.max()
    min_val = frames_ds.min()
    frames_norm = ((frames_ds - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    for z_slice in range(Z):
        print(f"Computing flow for Z slice {z_slice}/{Z-1}")
        frames_slice = frames_norm[:, z_slice, :, :]    
        prev_frame = frames_slice[0].compute()
        if config['hyperparams']["enhance_contrast"]:
            prev_frame = enhance_contrast_AHE(prev_frame)

        prev_flow = None

        for i in tqdm(range(1, frames_slice.shape[0]), desc="Computing optical flow"):

            curr_frame = frames_slice[i].compute()
            if config['hyperparams']["enhance_contrast"]:
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

            if prev_flow is not None and config['hyperparams']['temporal_smoothing'] is True:
                alpha = config['hyperparams']['temporal_smoothing_sigma']
                flow = alpha * flow + (1 - alpha) * prev_flow

            prev_frame = curr_frame
            prev_flow = flow

            if ds_factor > 1:
                flow_upsampled = np.zeros((Y, X, 2), dtype=np.float32)

                # Zoom factors for Y, X, and channels
                zoom_factors = (ds_factor, ds_factor, 1)

                flow_upsampled = zoom(flow, zoom_factors, order=1)  # order=1 is bilinear

                # Scale the flow magnitudes by the zoom factor
                flow_upsampled[..., 0] *= ds_factor  # vx
                flow_upsampled[..., 1] *= ds_factor  # vy

                flow = flow_upsampled
            
            # Unpad if necessary
            if pad_y != 0 or pad_x != 0:
                flow = flow[:Y, :X, :]

            output_zarr['flow_raw'][i-1, z_slice, ...] = flow.astype(np.float32)

    # Add empty flow for the last frame (since we compute flow between pairs of frames)
    output_zarr['flow_raw'][-1, ...] = np.zeros((Z, Y, X, 2), dtype=np.float32)

    return output_zarr['flow_raw']  # return the computed flow dataset


def compute_farneback_flow_3d(config, zarr_img, output_zarr):
    farneback_3d = _import_farneback_3d()

    T, Z, Y, X = zarr_img.shape

    # # Check if Z dim is large enough for 3D optical flow
    min_z_size = (int(config['pyr_scale'][0])^(int(config['levels']) - 1)) * 3  # heuristic minimum size
    if Z < min_z_size:
        pad_z = min_z_size - Z
    else:
        pad_z = 0

    # Check if Z, Y, or X dims are odd, and pad if so
    pad_y = 0
    pad_x = 0
    if Z % 2 != 0 and pad_z == 0:
        pad_z = 1
    if Y % 2 != 0:
        pad_y = 1
    if X % 2 != 0:
        pad_x = 1
    if pad_z != 0 or pad_y != 0 or pad_x != 0:
        zarr_img = da.pad(zarr_img, ((0,0),(0,pad_z),(0,pad_y),(0,pad_x)), mode='edge')

    ds_factor = config.get('downsample_factor', [1, 1, 1])

    frames_ds = zarr_img[:, ::ds_factor[0], ::ds_factor[1], ::ds_factor[2]]

    print("Normalizing all frames...")
    max_val = frames_ds.max()
    min_val = frames_ds.min()
    frames_norm = ((frames_ds - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    prev_frame = frames_norm[0].compute()
    if config['hyperparams']['enhance_contrast']:
        prev_frame = enhance_contrast_AHE(prev_frame)

    prev_flow = None

    for i in tqdm(range(1, frames_norm.shape[0]), desc="Computing optical flow"):

        curr_frame = frames_norm[i].compute()
        if config['hyperparams']['enhance_contrast']:
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

        if prev_flow is not None and config['hyperparams']['temporal_smoothing'] is True:
            alpha = config['hyperparams']['temporal_smoothing_sigma']
            flow = alpha * flow + (1 - alpha) * prev_flow
        
        prev_frame = curr_frame
        prev_flow = flow

        if any(f > 1 for f in ds_factor):
            # Zoom each component separately
            flow_upsampled = np.zeros((Z, Y, X, 3), dtype=np.float32)
            confidence_upsampled = np.zeros((Z, Y, X), dtype=np.float32)
            
            # Zoom factors: (Z_factor, Y_factor, X_factor, 1 for channels)
            zoom_factors = (ds_factor[0], ds_factor[1], ds_factor[2], 1)
            
            flow_upsampled = zoom(flow, zoom_factors, order=1)  # order=1 is bilinear
            confidence_upsampled = zoom(confidence_np, zoom_factors, order=1)  # order=1 is bilinear
            
            # Scale the flow magnitudes by the zoom factors
            flow_upsampled[..., 0] *= ds_factor[0]  # vz
            flow_upsampled[..., 1] *= ds_factor[1]  # vy
            flow_upsampled[..., 2] *= ds_factor[2]  # vx
            
            flow = flow_upsampled
        
        # Unpad if necessary
        if pad_z != 0 or pad_y != 0 or pad_x != 0:
            flow = flow[:Z, :Y, :X, :]
            confidence_np = confidence_np[:Z, :Y, :X]

        output_zarr['flow_raw'][i-1] = flow.astype(np.float32)
        output_zarr['confidence'][i-1] = confidence_np.astype(np.float32)

    # Add empty flow and confidence for the last frame (since we compute flow between pairs of frames)
    output_zarr['flow_raw'][-1] = np.zeros((Z, Y, X, 3), dtype=np.float32)
    output_zarr['confidence'][-1] = np.zeros((Z, Y, X), dtype=np.float32)

    return output_zarr['flow_raw']