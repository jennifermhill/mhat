import numpy as np


def _offset_slices(shape, offset):
    """The two aligned windows an affinity edge compares.

    An affinity edge with offset vector ``offset`` pairs each voxel with the
    voxel ``offset`` away from it. ``near`` is the window of voxels that have
    such a partner; ``far`` is the window of their partners, and is also where
    the result is written: the affinity of the edge (v - offset, v) lands at
    v. That is the layout waterz reads (channel d at voxel v is the edge to
    the previous voxel along axis d), so the array can be handed to waterz as
    is. Both windows are clipped to the array on whichever side the offset
    runs off, and the first plane along the offset's axis, which has no
    previous voxel, is left at 0.

    Built one axis at a time from ``offset``, so this works for a 2D
    ``(dy, dx)`` neighborhood exactly as it does for a 3D ``(dz, dy, dx)`` one.
    """
    near = tuple(
        slice(max(0, -d), min(size, size - d)) for size, d in zip(shape, offset)
    )
    far = tuple(slice(max(0, d), min(size, size + d)) for size, d in zip(shape, offset))
    return near, far


def compute_affinities(seg: np.ndarray, nhood: list):
    """Per-edge label-agreement affinities for a segmentation frame.

    Args:
        seg: A label frame, (z, y, x) or (y, x).
        nhood: One offset vector per affinity edge, each as long as ``seg.ndim``.

    Returns:
        An (n_edges, *seg.shape) array, 1 at v where v and v - offset are both
        foreground and carry the same label (waterz's layout, see
        ``_offset_slices``).
    """
    nhood = np.array(nhood)

    shape = seg.shape
    n_edges = nhood.shape[0]
    assert nhood.shape[1] == seg.ndim, (
        f"neighborhood offsets have {nhood.shape[1]} components but the frame "
        f"is {seg.ndim}D: {nhood.tolist()}"
    )
    affinity = np.zeros((n_edges,) + shape, dtype=np.int32)

    for e in range(n_edges):
        near, far = _offset_slices(shape, nhood[e])
        affinity[(e, *far)] = (seg[near] == seg[far]) * (seg[near] > 0) * (seg[far] > 0)

    return affinity


def compute_fluorescent_affinities(raw: np.ndarray, nhood: list):
    """Per-edge intensity-difference affinities for a raw frame.

    Args:
        raw: An intensity frame, (z, y, x) or (y, x).
        nhood: One offset vector per affinity edge, each as long as ``raw.ndim``.

    Returns:
        An (n_edges, *raw.shape) float32 array with |I(v) - I(v - offset)| at
        v (waterz's layout, see ``_offset_slices``). Note this is a
        *dissimilarity*; ``create_seg_hypotheses.py`` normalizes and inverts it
        before handing it to waterz.
    """
    nhood = np.array(nhood)

    shape = raw.shape
    n_edges = nhood.shape[0]
    assert nhood.shape[1] == raw.ndim, (
        f"neighborhood offsets have {nhood.shape[1]} components but the frame "
        f"is {raw.ndim}D: {nhood.tolist()}"
    )

    affinity = np.zeros((n_edges,) + shape, dtype=np.float32)

    for e in range(n_edges):
        near, far = _offset_slices(shape, nhood[e])
        # Convert to float to avoid overflow on unsigned integer inputs.
        region1 = raw[near].astype(np.float32)
        region2 = raw[far].astype(np.float32)
        affinity[(e, *far)] = np.abs(region1 - region2)

    return affinity
