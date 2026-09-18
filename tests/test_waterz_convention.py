"""Pins the affinity layout handed to waterz, fixed 2026-09-17.

waterz reads affinity channel 0 as z, 1 as y, 2 as x, and reads channel d at
voxel v as the edge between v and the *previous* voxel along axis d. The repo
now computes affinities in exactly that layout, so they go to waterz as stored.
Before the fix they were written at the near voxel with x first, which waterz
silently took as z and x swapped and every face one voxel off. Both mistakes
give plausible segmentations rather than errors, which is why the layout is
pinned here at the numpy level (no waterz needed, so this runs on Windows).

The reference is the hand manipulation that was validated on the cluster before
the fix went into the code: take the old near-voxel, x-first affinities, permute
the channels to z, y, x, and roll each one +1 voxel along its own axis
(``configs/experiments/waterz_shift_refit/build_arms.py`` in the data tree, arm
``chanorder_zyx_shift``). The first plane of each channel is the one place the
two differ (the arms zeroed it; the new code never writes it, so it is also 0
before normalization), and waterz never reads it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mhat.segmentation.affinities import compute_fluorescent_affinities
from mhat.segmentation.agglomerate import (
    WATERZ_AXIS_CHANNEL,
    WATERZ_NEIGHBORHOOD,
    check_waterz_neighborhood,
)

NHOOD_3D = WATERZ_NEIGHBORHOOD[3]
OLD_NHOOD_3D = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]  # what every pre-fix 3D config shipped


def _old_near_voxel_affinities(raw, nhood):
    """The pre-fix computation: |I(v) - I(v + offset)| written at v."""
    out = np.zeros((len(nhood), *raw.shape), dtype=np.float32)
    for e, offset in enumerate(nhood):
        near = tuple(slice(0, s - d) for s, d in zip(raw.shape, offset, strict=True))
        far = tuple(slice(d, s) for s, d in zip(raw.shape, offset, strict=True))
        out[(e, *near)] = np.abs(raw[near].astype(np.float32) - raw[far])
    return out


def _build_arms_recipe(old_affs, old_nhood):
    """The build_arms.py arm chanorder_zyx_shift recipe.

    Permute channels to z, y, x, roll each +1 along its axis, zero plane 0.
    """
    perm = [2, 1, 0]
    frame = old_affs[perm].copy()
    for c, offset in enumerate(old_nhood[i] for i in perm):
        axis = int(np.flatnonzero(offset)[0])
        rolled = np.roll(frame[c], 1, axis=axis)
        first = [slice(None)] * 3
        first[axis] = 0
        rolled[tuple(first)] = 0.0
        frame[c] = rolled
    return frame


def test_3d_matches_the_validated_hand_conversion():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 1000, size=(5, 7, 9)).astype(np.uint16)

    new = compute_fluorescent_affinities(raw, NHOOD_3D)
    want = _build_arms_recipe(
        _old_near_voxel_affinities(raw, OLD_NHOOD_3D), OLD_NHOOD_3D
    )

    assert new.shape == (3, 5, 7, 9) and new.dtype == np.float32
    np.testing.assert_array_equal(new, want)


@pytest.mark.parametrize("axis, axis_name", [(0, "z"), (1, "y"), (2, "x")])
def test_step_edge_lands_where_waterz_reads_it(axis, axis_name):
    """A step between planes 7 and 8 must reach waterz at plane 8 of its axis.

    waterz scores the 7|8 face from the value it finds at plane 8 (the sign
    probe of 2026-09-15). Every other channel and every other plane must be
    untouched, which is what catches a channel landing on the wrong axis.
    """
    shape = (16, 16, 16)
    raw = np.zeros(shape, dtype=np.float32)
    hi = [slice(None)] * 3
    hi[axis] = slice(8, None)
    raw[tuple(hi)] = 1.0

    affs = compute_fluorescent_affinities(raw, NHOOD_3D)

    expected = np.zeros((3, *shape), dtype=np.float32)
    at_8 = [slice(None)] * 3
    at_8[axis] = 8
    expected[(WATERZ_AXIS_CHANNEL[axis_name], *at_8)] = 1.0
    np.testing.assert_array_equal(affs, expected)


@pytest.mark.parametrize("ndim", [2, 3])
def test_waterz_neighborhood_is_accepted(ndim):
    check_waterz_neighborhood(WATERZ_NEIGHBORHOOD[ndim], ndim)
    # TOML floats and tuples must not trip the comparison.
    check_waterz_neighborhood(
        [tuple(float(d) for d in o) for o in WATERZ_NEIGHBORHOOD[ndim]], ndim
    )
    # Only the axis order is fixed; the step size is the user's choice.
    check_waterz_neighborhood(
        [[3 * d for d in o] for o in WATERZ_NEIGHBORHOOD[ndim]], ndim
    )
    mixed = {3: [[2, 0, 0], [0, 1, 0], [0, 0, 4]], 2: [[2, 0], [0, 4]]}
    check_waterz_neighborhood(mixed[ndim], ndim)


@pytest.mark.parametrize(
    "neighborhood, ndim",
    [
        (OLD_NHOOD_3D, 3),  # the pre-fix x-first order
        ([[0, 1], [1, 0]], 2),  # the pre-fix 2D order
        ([[1, 0, 0], [0, 1, 0]], 3),  # a channel missing
        ([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], 3),  # wrong direction
        ([[1, 1, 0], [0, 1, 0], [0, 0, 1]], 3),  # diagonal step
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2),  # wrong rank
    ],
)
def test_other_neighborhoods_are_rejected(neighborhood, ndim):
    with pytest.raises(AssertionError):
        check_waterz_neighborhood(neighborhood, ndim)
