from collections import deque

import numpy as np
import toml
import torch
import torch.nn.functional as F
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


def get_device(device=None):
    """Resolve device: use provided, or auto-detect CUDA, or fall back to CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def correlate1d_gpu(tensor, kernel, axis):
    """
    GPU equivalent of scipy.ndimage.correlate1d with mode='nearest'.

    Convolves `tensor` with 1D `kernel` along `axis` using replicate padding
    (equivalent to scipy's nearest mode).

    Args:
        tensor: N-dimensional torch tensor on device
        kernel: 1D torch tensor (the filter)
        axis: int, the axis along which to convolve

    Returns:
        Convolved tensor, same shape as input
    """
    ndim = tensor.ndim
    # Move target axis to last position
    tensor = tensor.moveaxis(axis, -1)
    orig_shape = tensor.shape
    L = orig_shape[-1]
    batch = tensor.numel() // L

    # Reshape to (batch, 1, length) for conv1d
    tensor = tensor.reshape(batch, 1, L)

    # Replicate padding (equivalent to scipy mode='nearest')
    klen = kernel.shape[0]
    pad_left = klen // 2
    pad_right = klen - 1 - pad_left
    tensor = F.pad(tensor, (pad_left, pad_right), mode='replicate')

    # kernel shape for conv1d: (out_channels, in_channels, kernel_size) = (1, 1, K)
    kernel_conv = kernel.flip(0).reshape(1, 1, -1)
    # Note: correlate1d does correlation (no kernel flip), but conv1d does convolution
    # (with flip). So we pre-flip the kernel to get correlation behavior.
    # Actually: conv1d computes cross-correlation by default in PyTorch, NOT convolution.
    # So we should NOT flip. Let me correct:
    kernel_conv = kernel.reshape(1, 1, -1)

    result = F.conv1d(tensor, kernel_conv)
    result = result.reshape(orig_shape)
    result = result.moveaxis(-1, axis)
    return result