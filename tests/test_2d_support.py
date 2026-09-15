"""The conventions 2D support introduced that are silent when wrong.

Most of 2D support is a consistent generalization of 3D code (positions of
rank ndim, one affinity slice per neighborhood offset, and so on) and the
ndim-parameterized fixtures in conftest already run the tracking and geff
smoke tests at both ranks. What they do not cover is the handful of ordering
conventions that only exist because of 2D, each of which produces plausible
numbers rather than an error if it is reversed:

  * flow channels are x-first while positions are z-first, and a 2-channel
    flow on 3D data must land on (y, x), not (z, y);
  * affinity slices are now built from the neighborhood offsets, so a 2D
    neighborhood has to mean what a per-pixel definition says it means;
  * the CTC evaluation boundary compares a (T, Y, X) prediction against 2D
    ground truth and derives the geff axes from the prediction;
  * waterz is strictly 3D, and its channel order (z, y, x) is the reverse of
    the neighborhood order the configs use, so the padding that lifts a 2D
    frame into a single slice has to route each channel to the right slot.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import tifffile
import zarr
from funtracks.import_export import import_from_geff

from mhat.evaluation.evaluate_tracking import (
    compute_ctc_seg,
    read_name_map_and_scale,
)
from mhat.evaluation.from_ctc_to_geff import from_ctc_to_geff
from mhat.segmentation.affinities import (
    compute_affinities,
    compute_fluorescent_affinities,
)
from mhat.segmentation.agglomerate import (
    WATERZ_AXIS_CHANNEL,
    agglomerate_frame,
    pad_2d_for_waterz,
)
from mhat.tracking.utils import flow_to_position_order

# The 2D neighborhood a 2D seg config ships: channel 0 steps in x, channel 1 in y.
NHOOD_2D = [[0, 1], [1, 0]]

# Distinct values throughout so a swapped pair cannot coincidentally pass.
VX, VY, VZ = 2.0, 3.0, 5.0
SZ, SY, SX = 7.0, 11.0, 13.0


@pytest.mark.parametrize(
    "vec, scale, offset, expected",
    [
        ((VX, VY, VZ), [SZ, SY, SX], 0, (VZ * SZ, VY * SY, VX * SX)),
        ((VX, VY), [SY, SX], 0, (VY * SY, VX * SX)),
        # The trap: 2 flow channels against 3 position axes. offset=1 says the
        # flow does not cover z, which keeps the y flow off the z axis (a naive
        # zip(centroid, vec[::-1]) would give (VY * SZ, VX * SY)).
        ((VX, VY), [SZ, SY, SX], 1, (VY * SY, VX * SX)),
    ],
    ids=["3d", "2d", "2d-flow-on-3d-data"],
)
def test_flow_to_position_order(vec, scale, offset, expected):
    assert flow_to_position_order(vec, scale, offset=offset) == expected


def _brute_force(frame, nhood, fn):
    """Reference implementation: visit every pixel and its offset partner."""
    nhood = np.asarray(nhood)
    out = np.zeros((len(nhood), *frame.shape), dtype=np.float64)
    for e, offset in enumerate(nhood):
        for index in itertools.product(*(range(s) for s in frame.shape)):
            partner = tuple(i + d for i, d in zip(index, offset, strict=True))
            if any(
                p < 0 or p >= s for p, s in zip(partner, frame.shape, strict=True)
            ):
                continue
            out[(e, *index)] = fn(frame[index], frame[partner])
    return out


@pytest.mark.parametrize(
    "compute, make_frame, definition, dtype",
    [
        (
            compute_affinities,
            lambda rng: rng.integers(0, 4, size=(7, 5), dtype=np.int32),
            lambda a, b: float(a == b and a > 0 and b > 0),
            np.int32,
        ),
        (
            compute_fluorescent_affinities,
            lambda rng: rng.integers(0, 1000, size=(7, 5)).astype(np.uint16),
            lambda a, b: abs(float(a) - float(b)),
            np.float32,
        ),
    ],
    ids=["label", "fluorescent"],
)
def test_affinities_2d_match_brute_force(compute, make_frame, definition, dtype):
    frame = make_frame(np.random.default_rng(11))

    got = compute(frame, NHOOD_2D)
    want = _brute_force(frame, NHOOD_2D, definition).astype(dtype)

    assert got.shape == (2, 7, 5)
    assert np.array_equal(got, want)


# A minimal 2D CTC sequence: two solid squares per frame, each drifting one
# pixel per frame in x, labelled by tracklet id.
T, Y, X = 3, 16, 16
OBJECTS = {1: (2, 6), 2: (10, 14)}


def _ctc_frame(t):
    frame = np.zeros((Y, X), dtype=np.uint16)
    for tracklet_id, (lo, hi) in OBJECTS.items():
        frame[lo:hi, lo + t : hi + t] = tracklet_id
    return frame


def test_ctc_2d_evaluation_boundary(tmp_path):
    """2D CTC ground truth in, 2D geff out, and SEG scored on a 2D prediction.

    Both ends of this are where a dummy-z prediction would fail: the geff would
    declare an axis no node property backs, and compute_ctc_seg would compare a
    (1, Y, X) slice against a (Y, X) reference.
    """
    tra_dir = tmp_path / "01_GT" / "TRA"
    seg_dir = tmp_path / "01_GT" / "SEG"
    tra_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)
    for t in range(T):
        tifffile.imwrite(tra_dir / f"man_track{t:03d}.tif", _ctc_frame(t))
        tifffile.imwrite(seg_dir / f"man_seg{t:03d}.tif", _ctc_frame(t))
    # track_id start end parent; two orphan rows, so np.loadtxt gives a 2D table.
    (tra_dir / "man_track.txt").write_text(
        "\n".join(f"{tracklet_id} 0 {T - 1} 0" for tracklet_id in OBJECTS) + "\n"
    )

    geff_path = tmp_path / "correct_tracks.zarr"
    seg_path = tmp_path / "correct_seg.zarr"
    from_ctc_to_geff(
        ctc_path=tra_dir,
        geff_path=geff_path,
        segmentation_store=seg_path,
        axes=None,
        overwrite=True,
    )

    # Three axes, no z, and a segmentation of the source frames' rank.
    name_map, scale = read_name_map_and_scale(geff_path)
    assert name_map["pos"] == ["y", "x"]
    assert len(scale) == 3
    assert zarr.open(str(seg_path), mode="r").shape == (T, Y, X)

    tracks = import_from_geff(
        geff_path, node_name_map=name_map, segmentation_path=None, scale=scale
    )
    assert tracks.graph.number_of_nodes() == T * len(OBJECTS)
    for _, attrs in tracks.graph.nodes(data=True):
        assert len(attrs["pos"]) == 2, f"expected a 2-component pos, got {attrs['pos']}"

    # SEG on a (T, Y, X) prediction that covers one of the two objects (with a
    # different label value, since the Jaccard is label-agnostic) is 0.5 --
    # not 1.0, and not None from a shape mismatch.
    pred_seg_path = tmp_path / "pred_seg.zarr"
    pred = zarr.open(
        str(pred_seg_path),
        mode="w",
        shape=(T, Y, X),
        chunks=(1, Y, X),
        dtype=np.uint32,
    )
    for t in range(T):
        pred[t] = np.where(_ctc_frame(t) == 1, 101, 0)
    assert compute_ctc_seg(seg_dir, pred_seg_path) == pytest.approx(0.5)


def test_waterz_2d_padding_and_squeeze():
    """A 2D frame goes into waterz on the right channels and comes back 2D.

    waterz reads channel 0 as z, 1 as y, 2 as x (see the note in
    agglomerate.py), the reverse of the config's neighborhood order, so a naive
    "prepend a dummy slice" would put the x affinities in the z slot -- where,
    with a single slice, they would be ignored entirely.
    """
    pytest.importorskip(
        "waterz",
        reason="waterz not installed — it is the optional [waterz] extra (Linux only)",
    )
    height, width = 16, 16
    affs = np.empty((2, height, width), dtype=np.float32)
    affs[0] = 0.25  # the x-offset channel
    affs[1] = 0.75  # the y-offset channel
    fragments = np.zeros((height, width), dtype=np.uint32)
    fragments[:8, :8] = 1
    fragments[:8, 8:] = 2
    fragments[8:, :8] = 3
    fragments[8:, 8:] = 4

    affs_3d, fragments_3d = pad_2d_for_waterz(affs, fragments, NHOOD_2D)
    assert affs_3d.shape == (3, 1, height, width)
    assert fragments_3d.shape == (1, height, width)
    assert np.all(affs_3d[WATERZ_AXIS_CHANNEL["x"], 0] == 0.25)
    assert np.all(affs_3d[WATERZ_AXIS_CHANNEL["y"], 0] == 0.75)
    # With one slice nothing has a z neighbour, so the axial channel stays at
    # the minimum affinity rather than encouraging anything.
    assert np.all(affs_3d[WATERZ_AXIS_CHANNEL["z"]] == 0.0)

    segmentation, merge_history = agglomerate_frame(
        affs=affs, fragments=fragments, thresholds=[0.5], neighborhood=NHOOD_2D
    )
    assert segmentation.shape == fragments.shape, (
        "the dummy z slice leaked out of the padding helper"
    )
    assert merge_history, "expected at least one merge at threshold 0.5"
