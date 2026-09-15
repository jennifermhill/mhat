import csv
from pathlib import Path

import numpy as np

# waterz reads affinity channel 0 as the z offset, 1 as y and 2 as x. This is
# not documented by waterz and was established empirically here: split a volume
# in half along one axis, raise exactly one affinity channel, and the pair
# merges only when that channel matches the split axis. The result was a clean
# diagonal over all nine (split axis, channel) combinations.
#
# Note this is NOT the order the 3D configs in this repo use: they ship
# ``neighborhood = [[0,0,1], [0,1,0], [1,0,0]]``, whose (dz, dy, dx) offsets put
# the *x* affinity in channel 0 and z in channel 2 -- so on 3D data waterz has
# been receiving z and x swapped. Fixing that would change every 3D
# segmentation ever produced here, so it is deliberately left alone; see the
# "2D Support" section of CLAUDE.md. It matters below only because the 2D path
# is new and has no results to preserve, so it maps its channels correctly.

# waterz's affinity channel for each spatial axis.
WATERZ_AXIS_CHANNEL = {"z": 0, "y": 1, "x": 2}


def pad_2d_for_waterz(affs: np.ndarray, fragments: np.ndarray, neighborhood):
    """Lift a 2D frame into the single-slice volume waterz requires.

    ``waterz.agglomerate``'s C++ contract is strictly 3D: (3, depth, height,
    width) float32 affinities and a 3D uint64 fragment volume. It is the one
    component of this pipeline that genuinely cannot do 2D, so it is the one
    place a dummy dimension is allowed to exist -- introduced immediately
    before the call and squeezed away immediately after. The merge history
    needs no adjustment at all, because it is label-based and labels do not
    know what rank they came from.

    Each 2D affinity channel is placed in the waterz slot for the axis its
    offset steps along, so the caller's neighborhood ordering does not matter.
    The axial channel is left at 0.0: with a single slice no voxel has a z
    neighbour, so waterz never reads it, and 0.0 is the minimum affinity after
    the inversion in create_seg_hypotheses in any case.

    Args:
        affs: (n_edges, height, width) affinities.
        fragments: (height, width) fragment labels.
        neighborhood: The (dy, dx) offset of each affinity channel, as passed
            to ``compute_fluorescent_affinities``.

    Returns:
        ((3, 1, height, width) float32 affinities, (1, height, width) uint64
        fragments), ready to hand to ``waterz.agglomerate``.
    """
    assert fragments.ndim == 2, f"expected a 2D frame, got {fragments.shape}"
    n_edges, height, width = affs.shape
    neighborhood = np.asarray(neighborhood)
    assert neighborhood.shape == (n_edges, 2), (
        f"neighborhood {neighborhood.shape} does not describe {n_edges} 2D "
        "affinity channels"
    )
    assert fragments.shape == (height, width), (
        f"affinities {affs.shape} and fragments {fragments.shape} disagree"
    )

    affs_3d = np.zeros((3, 1, height, width), dtype=np.float32)
    for edge, offset in enumerate(neighborhood):
        stepped = np.flatnonzero(offset)
        assert len(stepped) == 1, (
            "waterz has one affinity channel per axis, so every neighborhood "
            f"offset must step along exactly one axis; got {offset.tolist()}"
        )
        axis_name = ("y", "x")[int(stepped[0])]
        affs_3d[WATERZ_AXIS_CHANNEL[axis_name], 0] = affs[edge]

    fragments_3d = fragments[np.newaxis].astype(np.uint64)
    return affs_3d, fragments_3d


def agglomerate_frame(affs, fragments, thresholds, neighborhood=None, **kwargs):
    """``waterz.agglomerate`` on one frame, of either rank.

    3D frames are passed straight through, byte for byte as before. 2D frames
    are padded with :func:`pad_2d_for_waterz`, agglomerated, and squeezed back
    to 2D, so the caller only ever sees its own rank.

    Args:
        affs: (n_edges, *frame_shape) float32 affinities.
        fragments: (*frame_shape) fragment labels.
        thresholds: Merge thresholds, passed through to waterz.
        neighborhood: Required for 2D frames -- it says which axis each
            affinity channel belongs to. Ignored for 3D.
        **kwargs: Passed through to ``waterz.agglomerate``.

    Returns:
        (segmentation, merge_history) for the first threshold, with the
        segmentation at the same rank as ``fragments``.
    """
    import waterz  # type: ignore  # optional [waterz] extra, Linux only

    ndim = fragments.ndim
    if ndim == 3:
        segmentation, merge_history = next(waterz.agglomerate(
            affs=affs.astype(np.float32),
            fragments=fragments.astype(np.uint64),
            thresholds=thresholds,
            return_merge_history=True,
            **kwargs,
        ))
        return segmentation, merge_history

    assert ndim == 2, f"expected a 2D or 3D frame, got {fragments.shape}"
    assert neighborhood is not None, (
        "2D agglomeration needs the neighborhood to know which axis each "
        "affinity channel belongs to"
    )
    affs_3d, fragments_3d = pad_2d_for_waterz(affs, fragments, neighborhood)
    segmentation, merge_history = next(waterz.agglomerate(
        affs=affs_3d,
        fragments=fragments_3d,
        thresholds=thresholds,
        return_merge_history=True,
        **kwargs,
    ))
    return segmentation[0], merge_history


def agglomerate(fragments: np.ndarray, merge_history: Path, threshold: float):
    """Takes fragments and a merge history, and merges the fragments up until
    the threshold. Returns a new numpy array that is a "final" segmentation.

    Args:
        fragments (np.ndarray): Fragments that have already been run through
            waterz
        merge_history (Path): Result of running waterz on fragments with pretty
            high threshold
        threshold (float): The cost threshold to merge until. If it is bigger
            than the largest in the merge history, nothing more will merge.

    Returns:
        np.ndarray: A segmentation coming from merging the fragments according
            to the merge history
    """
    with open(merge_history) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["a"])
            b = int(row["b"])
            c = int(row["c"])
            # Support both "cost" (new) and "score" (old) column names
            cost = float(row.get("cost", row.get("score", 0.0)))
            if cost <= threshold:
                fragments[fragments == a] = c
                fragments[fragments == b] = c

    return fragments
