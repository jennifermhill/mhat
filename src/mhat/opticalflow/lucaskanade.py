import cv2
import dask.array as da
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

from opticalflow3D.helpers.pyrlk_functions import pyrlk_3d
from mhat.opticalflow.utils import enhance_contrast_AHE

def compute_lucaskanade_flow_3d(config, zarr_img, output_zarr):
    T, Z, Y, X = zarr_img.shape

    # # Check if Z dim is large enough for 3D optical flow
    # min_z_size = (int(config['pyr_scale'])^(int(config['levels']) - 1)) * 3  # heuristic minimum size
    # if Z < min_z_size:
    #     pad_z = min_z_size - Z
    # else:
    #     pad_z = 0

    # # Check if Z, Y, or X dims are odd, and pad if so
    # pad_y = 0
    # pad_x = 0
    # if Z % 2 != 0 and pad_z == 0:
    #     pad_z = 1
    # if Y % 2 != 0:
    #     pad_y = 1
    # if X % 2 != 0:
    #     pad_x = 1
    # if pad_z != 0 or pad_y != 0 or pad_x != 0:
    #     print(f"Padding Z by {pad_z}, Y by {pad_y} and X by {pad_x} to make dimensions even for optical flow calculation.")
    #     zarr_img = da.pad(zarr_img, ((0,0),(0,pad_z),(0,pad_y),(0,pad_x)), mode='edge')
    #     # T, Z, Y, X = zarr_img.shape
    #     print(f"Padded zarr image shape: {zarr_img.shape}")

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

        vx, vy, vz = pyrlk_3d(
            prev_frame,
            curr_frame,
            iters=config['iterations'],
            num_levels=config['levels'],
            scale=config['pyr_scale'],
            tau=config['tau'],
            alpha=config['alpha'],
            filter_type=config['filter_type'],
            filter_size=config['winsize'],
            presmoothing=config['presmoothing']
        )

        vx_np = vx.cpu().numpy()
        vy_np = vy.cpu().numpy()
        vz_np = vz.cpu().numpy()
        flow = np.stack((vx_np, vy_np, vz_np), axis=-1)

        if prev_flow is not None and config['hyperparams']['temporal_smoothing'] is not None:
            alpha = config['hyperparams']['temporal_smoothing_sigma']
            flow = alpha * flow + (1 - alpha) * prev_flow
        
        prev_frame = curr_frame
        prev_flow = flow

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
        
        # # Unpad if necessary
        # if pad_z != 0 or pad_y != 0 or pad_x != 0:
        #     flow = flow[:Z - pad_z, :Y - pad_y, :X - pad_x, :]
        #     print(f"Unpadded flow shape: {flow.shape}")

        output_zarr['flow_raw'][i-1] = flow.astype(np.float32)

    return output_zarr['flow_raw']