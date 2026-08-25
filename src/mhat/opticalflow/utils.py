from collections import deque

import numpy as np
import toml
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

        config_filepath = new_path / "config.toml"
        if config_filepath.exists():
            config = toml.load(config_filepath)
            config["exp_uid"] = exp_uid
            with open(config_filepath, 'w') as config_file:
                toml.dump(config, config_file)
    else:
        print(f"Warning: Previous flow directory {rerun_uid} does not exist")
        

def frame_average(flow_zarr, frame_avg):
    """Smooth flow over time with a centered sliding window of frame_avg frames.
    """
    print("Applying frame averaging to flow data...")
    flow = flow_zarr['flow_raw']
    n_real_frames = flow.shape[0] - 1  # last frame is the zero placeholder
    half_window = frame_avg // 2

    # Pre-averaging copies of the previous `half_window` frames.
    previous_frames = deque(maxlen=half_window)
    for i in range(0, n_real_frames):
        end_idx = min(n_real_frames, i + half_window + 1)
        current_frame = flow[i]  # not yet overwritten, so still the raw flow
        # Frames after i are untouched, so they can be read straight from disk.
        window = list(previous_frames) + [current_frame] + [flow[j] for j in range(i + 1, end_idx)]
        previous_frames.append(current_frame)
        flow[i, ...] = np.mean(window, axis=0)
    return flow
