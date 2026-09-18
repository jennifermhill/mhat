import csv
from pathlib import Path

import numpy as np

# waterz's affinity contract, established empirically (2026-09-15; the sign probe
# and nine-combination channel diagonal are described in the "2D Support > Known
# hazards" section of CLAUDE.md): channel 0 is the z edge, 1 is y, 2 is x, and
# channel d at voxel v is the edge between v and the *previous* voxel along axis
# d, i.e. (v - e_d, v).
#
# Since 2026-09-17 the repo computes affinities directly in that layout:
# ``compute_fluorescent_affinities`` writes the edge (v - offset, v) at v, and
# the seg configs list the neighborhood offsets in z, y, x order, which
# :func:`check_waterz_neighborhood` enforces. A 3D frame is therefore handed to
# waterz exactly as stored, and a 2D frame only needs the dummy slice from
# :func:`pad_2d_for_waterz`. Before that date the affinities were written at the
# near voxel and the configs listed x first, so every earlier 3D segmentation
# fed waterz z and x swapped and every face one voxel off. Those segmentations
# are not reproducible with this code. From the fix onward every segmentation
# carries ``waterz_convention`` in its config.toml and zarr attrs; a
# segmentation without it is from before.

# waterz's affinity channel for each spatial axis.
WATERZ_AXIS_CHANNEL = {"z": 0, "y": 1, "x": 2}

# The default neighborhood per rank: a +1 step along z, y, x (y, x in 2D), in
# waterz's channel order. Larger steps are allowed; the order is not.
WATERZ_NEIGHBORHOOD = {
    3: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    2: [[1, 0], [0, 1]],
}

# Recorded on every segmentation this module produces, so pre- and post-fix
# runs can be told apart on disk.
WATERZ_CONVENTION = (
    "affinities stored in waterz layout: z,y,x channels, edge (v - e, v) at v "
    "(fixed 2026-09-17)"
)


def check_waterz_neighborhood(neighborhood, ndim):
    """Refuse a neighborhood whose channels are not in waterz's axis order.

    The affinities are handed to waterz as stored, so the config's offset
    order *is* waterz's channel order: channel c must step along axis c (z,
    y, x; y, x in 2D) and nothing else. The step size is free -- a larger
    offset compares intensities further apart -- but it must be positive, so
    that the value lands at the voxel waterz reads it from (see
    ``_offset_slices`` in ``affinities.py``). Configs from before 2026-09-17
    list x first; that order would silently swap z and x again, hence a hard
    check rather than a warning.
    """
    axis_names = ("z", "y", "x")[3 - ndim :]
    assert ndim in (2, 3), f"expected a 2D or 3D frame, got ndim={ndim}"
    got = [[int(d) for d in offset] for offset in neighborhood]
    assert len(got) == ndim and all(len(offset) == ndim for offset in got), (
        f"a {ndim}D neighborhood needs {ndim} offsets of {ndim} components, one "
        f"per axis; got {got}"
    )
    for channel, offset in enumerate(got):
        stepped = [axis for axis, d in enumerate(offset) if d != 0]
        assert stepped == [channel] and offset[channel] > 0, (
            f"affinity channel {channel} must be a positive step along "
            f"{axis_names[channel]} only (waterz's z,y,x channel order, e.g. "
            f"{WATERZ_NEIGHBORHOOD[ndim]}); got {offset}. Configs from before "
            "2026-09-17 list x first and must be reordered."
        )


def pad_2d_for_waterz(affs: np.ndarray, fragments: np.ndarray):
    """Lift a 2D frame into the single-slice volume waterz requires.

    ``waterz.agglomerate``'s C++ contract is strictly 3D: (3, depth, height,
    width) float32 affinities and a 3D uint64 fragment volume. It is the one
    component of this pipeline that genuinely cannot do 2D, so it is the one
    place a dummy dimension is allowed to exist -- introduced immediately
    before the call and squeezed away immediately after. The merge history
    needs no adjustment at all, because it is label-based and labels do not
    know what rank they came from.

    The two 2D channels (y, x) go into waterz's y and x slots. The z slot is
    left at 0.0: with a single slice no voxel has a z neighbour, so waterz
    never reads it.

    Args:
        affs: (2, height, width) affinities in waterz layout, channel 0 = y.
        fragments: (height, width) fragment labels.

    Returns:
        ((3, 1, height, width) float32 affinities, (1, height, width) uint64
        fragments), ready to hand to ``waterz.agglomerate``.
    """
    assert fragments.ndim == 2, f"expected a 2D frame, got {fragments.shape}"
    n_edges, height, width = affs.shape
    assert n_edges == 2, f"expected the (y, x) affinity channels, got {affs.shape}"
    assert fragments.shape == (height, width), (
        f"affinities {affs.shape} and fragments {fragments.shape} disagree"
    )

    affs_3d = np.zeros((3, 1, height, width), dtype=np.float32)
    affs_3d[WATERZ_AXIS_CHANNEL["y"] :, 0] = affs
    fragments_3d = fragments[np.newaxis].astype(np.uint64)
    return affs_3d, fragments_3d


def agglomerate_frame(affs, fragments, thresholds, neighborhood, **kwargs):
    """``waterz.agglomerate`` on one frame, of either rank.

    3D frames are passed straight through. 2D frames are padded with
    :func:`pad_2d_for_waterz`, agglomerated, and squeezed back to 2D, so the
    caller only ever sees its own rank.

    Args:
        affs: (n_edges, *frame_shape) float32 affinities in waterz layout.
        fragments: (*frame_shape) fragment labels.
        thresholds: Merge thresholds, passed through to waterz.
        neighborhood: The offset of each affinity channel, checked against
            :data:`WATERZ_NEIGHBORHOOD` for this rank.
        **kwargs: Passed through to ``waterz.agglomerate``.

    Returns:
        (segmentation, merge_history) for the first threshold, with the
        segmentation at the same rank as ``fragments``.
    """
    import waterz  # type: ignore  # optional [waterz] extra, Linux only

    ndim = fragments.ndim
    check_waterz_neighborhood(neighborhood, ndim)
    if ndim == 3:
        affs_wz = affs.astype(np.float32)
        fragments_wz = fragments.astype(np.uint64)
    else:
        affs_wz, fragments_wz = pad_2d_for_waterz(affs, fragments)

    segmentation, merge_history = next(
        waterz.agglomerate(
            affs=affs_wz,
            fragments=fragments_wz,
            thresholds=thresholds,
            return_merge_history=True,
            **kwargs,
        )
    )
    if ndim == 2:
        segmentation = segmentation[0]
    return segmentation, merge_history


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
