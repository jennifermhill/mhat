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
(``scratch_configs/experiments/waterz_shift_refit/build_arms.py`` in the data tree, arm
``chanorder_zyx_shift``). The first plane of each channel is the one place the
two differ (the arms zeroed it; the new code never writes it, so it is also 0
before normalization), and waterz never reads it.

The second test guards the other half of the fix: ``check_waterz_neighborhood``
must refuse the neighborhoods that would silently reproduce the bug, since an
old config run unchanged would otherwise produce a plausible segmentation with
the channels swapped again.
"""

from __future__ import annotations

import numpy as np
import pytest

from mhat.segmentation.affinities import compute_fluorescent_affinities
from mhat.segmentation.agglomerate import (
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


@pytest.mark.parametrize(
    "neighborhood, ndim",
    [
        (OLD_NHOOD_3D, 3),  # the pre-fix x-first order: z and x swapped again
        ([[0, 1], [1, 0]], 2),  # the pre-fix 2D order
        ([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], 3),  # wrong direction: faces shift back
        ([[1, 1, 0], [0, 1, 0], [0, 0, 1]], 3),  # diagonal step read as the z channel
    ],
)
def test_bug_reproducing_neighborhoods_are_rejected(neighborhood, ndim):
    """Each of these would agglomerate without error and be wrong."""
    with pytest.raises(AssertionError):
        check_waterz_neighborhood(neighborhood, ndim)
