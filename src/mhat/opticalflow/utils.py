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
    print(f"Renaming previous flow uid to current experiment uid: {exp_uid}")
    
    if rerun_uid.exists():
        new_path = rerun_uid.parent / exp_uid
        rerun_uid.rename(new_path)
        print(f"Renamed {rerun_uid} to {new_path}")
    else:
        print(f"Warning: Previous flow directory {rerun_uid} does not exist")