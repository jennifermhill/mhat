import warnings
from collections import deque
from pathlib import Path

import dask.array as da
import numpy as np
import toml
import zarr
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


# ---------------------------------------------------------------------------
# Flow store layout
#
# ``flow_raw`` is (t, *spatial, components) with the components in the same
# order as the spatial axes: (vz, vy, vx) for 3D flow, (vy, vx) for 2D. That is
# the order every other vector in the repo uses (centroids, scale,
# downsample_factor, neighborhood), so a flow vector lines up with a position
# vector component for component. The store records the order under
# ``flow_raw.attrs["channel_order"]``.
#
# Stores written before 2026-09-21 are x-first -- OpenCV's (dx, dy) for 2D and
# a matching (vx, vy, vz) stack for 3D -- and carry no marker. open_flow_raw
# reverses their last axis on read so they keep working unchanged.
#
# TODO(publish): remove the legacy path before the code is published --
# LEGACY_FLOW_CHANNEL_ORDER, _ReversedChannels, the missing-marker branches in
# flow_channel_order and open_flow_raw, and the legacy test in
# tests/test_farneback.py. Every flow store in the data tree is being
# regenerated after the waterz fix, so by then none should be legacy; make the
# missing marker an error instead.
# ---------------------------------------------------------------------------

FLOW_CHANNEL_ORDER = {3: ["z", "y", "x"], 2: ["y", "x"]}
LEGACY_FLOW_CHANNEL_ORDER = {3: ["x", "y", "z"], 2: ["x", "y"]}


def create_flow_store(path, n_frames, spatial_shape, plane_chunks, axes,
                      with_confidence):
    """Create ``flow.zarr`` with an empty ``flow_raw`` (and ``confidence`` for 3D).

    ``flow_raw`` gets one component per spatial axis, in axis order, and both
    the ``axes`` metadata and the ``channel_order`` marker that readers use.
    """
    ndim = len(spatial_shape)
    root = zarr.open(path, mode="w")
    flow_raw = root.create_dataset(
        "flow_raw",
        shape=(n_frames, *spatial_shape, ndim),
        chunks=(*plane_chunks, ndim),
        dtype=np.float32,
    )
    flow_raw.attrs["axes"] = axes
    flow_raw.attrs["channel_order"] = FLOW_CHANNEL_ORDER[ndim]
    if with_confidence:
        root.create_dataset(
            "confidence",
            shape=(n_frames, *spatial_shape),
            chunks=plane_chunks,
            dtype=np.float32,
        )
    return root


def flow_channel_order(flow_raw):
    """Component order of a ``flow_raw`` array, e.g. ``["z", "y", "x"]``.

    A store without the ``channel_order`` marker predates the z-first layout
    and is x-first (legacy; see the module note above).
    """
    n_components = flow_raw.shape[-1]
    order = flow_raw.attrs.get("channel_order")
    if order is None:
        warnings.warn(
            f"flow store {getattr(flow_raw.store, 'path', flow_raw.store)} has no "
            f"'channel_order' marker, so it is a legacy x-first store; reading it "
            f"with the components reversed. Regenerate it to drop this path.",
            stacklevel=3,
        )
        return LEGACY_FLOW_CHANNEL_ORDER[n_components]
    order = list(order)
    assert len(order) == n_components, (
        f"channel_order {order} does not match {n_components} flow components"
    )
    return order


class _ReversedChannels:
    """Read-only view of a legacy flow array with its last axis reversed.

    Reads one index at a time, like the zarr array it wraps, so a movie is
    never pulled into memory. Only indexing that leaves the component axis
    alone is allowed -- select the component after reading.
    """

    def __init__(self, arr):
        self._arr = arr

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def attrs(self):
        return self._arr.attrs

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, key):
        key_tuple = key if isinstance(key, tuple) else (key,)
        if Ellipsis in key_tuple or len(key_tuple) >= self.ndim:
            raise IndexError(
                "legacy flow store: index the component axis after reading, "
                "not in the same subscript"
            )
        return np.asarray(self._arr[key])[..., ::-1]


def open_flow_raw(flow_root, dask=False):
    """``flow_raw`` from a ``flow.zarr`` group or path, components in axis order.

    Native stores come back as the zarr array itself (or a dask array with
    ``dask=True``); legacy x-first stores come back reversed on read so every
    caller sees ``(vz, vy, vx)`` / ``(vy, vx)`` regardless of the file's age.
    """
    if isinstance(flow_root, (str, Path)):
        flow_root = zarr.open(flow_root, mode="r")
    flow_raw = flow_root["flow_raw"]
    n_components = flow_raw.shape[-1]
    order = flow_channel_order(flow_raw)
    if order == FLOW_CHANNEL_ORDER[n_components]:
        return da.from_zarr(flow_raw) if dask else flow_raw
    assert order == LEGACY_FLOW_CHANNEL_ORDER[n_components], (
        f"unsupported flow channel_order {order}"
    )
    if dask:
        return da.from_zarr(flow_raw)[..., ::-1]
    return _ReversedChannels(flow_raw)
