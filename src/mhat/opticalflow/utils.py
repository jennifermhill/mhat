import numpy as np
from pathlib import Path
from skimage.exposure import equalize_adapthist


def enhance_contrast_AHE(frame):
    if frame.ndim == 2:
        kernel_size = [8, 8]
    elif frame.ndim == 3:
        kernel_size = [3, 8, 8]
    else:
        raise ValueError("Input frame must be 2D or 3D grayscale image for AHE.")
    frame = equalize_adapthist(frame, kernel_size=kernel_size, clip_limit=0.01)
    frame = (frame * 255).astype(np.uint8)  # Convert to 8-bit image
    return frame


def rename_flow_uid(rerun_uid: Path, exp_uid: str):
    if rerun_uid.exists():
        new_path = rerun_uid.parent / exp_uid
        rerun_uid.rename(new_path)
        print(f"Renamed {rerun_uid} to {new_path}")
    else:
        print(f"Warning: Previous flow directory {rerun_uid} does not exist")
        

def frame_average(flow_zarr, frame_avg):
    print("Applying frame averaging to flow data...")
    for i in range(0, flow_zarr['flow_raw'].shape[0]-1):
        start_idx = max(0, i - frame_avg // 2)
        end_idx = min(flow_zarr['flow_raw'].shape[0], i + frame_avg // 2 + 1)
        flow_zarr['flow_raw'][i, ...] = np.mean(flow_zarr['flow_raw'][start_idx:end_idx, ...], axis=0)
    return flow_zarr['flow_raw']