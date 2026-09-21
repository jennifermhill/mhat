"""Regression tests for the Farneback wrappers and the flow store layout.

Everything pinned here fails silently rather than loudly if it regresses:

* the Z-size guard used ``^`` (XOR) for a power; a minimum that is too *large*
  pads more replicated edge slices than necessary and quietly degrades the z
  flow (too small crashes on its own);
* the warning that padding dominates the real slices is, if it fails to fire,
  silent by definition;
* flow components are stored in axis order, ``(vz, vy, vx)`` / ``(vy, vx)``
  (switched from x-first on 2026-09-21). OpenCV hands back ``(dx, dy)`` and
  ``farneback_3d`` hands back three separate arrays, so a wrong order at either
  producer, a wrong per-axis factor when upsampling an anisotropic downsample,
  or a legacy x-first store read as-is all give finite, plausible flow fields
  with the components swapped.

The expected padding minima were established by running
``opticalflow3D.helpers.farneback_functions.farneback_3d`` on synthetic volumes
of increasing Z and recording the first size that did not raise.
"""

import warnings

import numpy as np
import pytest
import zarr

from mhat.opticalflow.farneback import min_size_for_farneback_3d
from mhat.opticalflow.utils import (
    FLOW_CHANNEL_ORDER,
    create_flow_store,
    open_flow_raw,
)

UNIT_AXES = [{"name": n, "scale": 1.0} for n in ("time", "z", "y", "x")]


def flow_config_3d(poly_n=3, downsample_factor=(1, 1, 1)):
    return {
        "pyr_scale": [0.5, 0.5, 0.5],
        "levels": 3,
        "poly_n": poly_n,
        "poly_sigma": [0.37, 1.2, 1.2],
        "iterations": 1,
        "winsize": [5, 15, 15],
        "filter_type": "gaussian",
        "presmoothing": 0.1,
        "downsample_factor": list(downsample_factor),
        "hyperparams": {
            "enhance_contrast": False,
            "temporal_smoothing": False,
            "temporal_smoothing_sigma": 0.7,
        },
    }


def flow_config_2d(downsample_factor=1):
    return {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 1,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "downsample_factor": downsample_factor,
        "hyperparams": {
            "enhance_contrast": False,
            "temporal_smoothing": False,
            "temporal_smoothing_sigma": 0.7,
        },
    }


def run_flow(compute, config, movie, tmp_path):
    """Run a compute_farneback_flow_* on an in-memory movie via the real store."""
    import dask.array as da

    T, *spatial = movie.shape
    ndim = len(spatial)
    out = create_flow_store(
        tmp_path / "flow.zarr",
        T,
        tuple(spatial),
        (1,) * (ndim - 1) + tuple(spatial[-2:]),
        UNIT_AXES[: ndim + 1],
        with_confidence=(ndim == 3),
    )
    compute(config, da.from_array(movie, chunks=-1), out)
    return out


# --------------------------------------------------------------------------- padding


@pytest.mark.parametrize(
    "scale, levels, poly_n, expected",
    [
        (0.8, 8, 5, 14),  # primary_nk_cells 3D flow config; XOR guard said 21
        (0.5, 3, 3, 6),  # current flow_config.toml; XOR guard said 6 by luck
        (0.5, 3, 5, 11),  # same pyramid, larger kernel; XOR guard still said 6
    ],
)
def test_min_size_matches_measured_minimum(scale, levels, poly_n, expected):
    assert min_size_for_farneback_3d(scale, levels, poly_n) == expected


def test_compute_3d_warns_when_padding_dominates(tmp_path):
    """4 real slices, poly_n 5 -> minimum 11 -> 7 padded slices: must warn.

    Also confirms the output is unpadded back to the real Z, which is the
    other thing that would go wrong quietly (a flow computed on the padded
    volume but sliced from the wrong end would not raise).
    """
    pytest.importorskip("opticalflow3D")
    from mhat.opticalflow.farneback import compute_farneback_flow_3d

    movie = np.random.default_rng(0).random((2, 4, 16, 16), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = run_flow(
            compute_farneback_flow_3d, flow_config_3d(poly_n=5), movie, tmp_path
        )

    padding_warnings = [w for w in caught if "replicated edge slices" in str(w.message)]
    assert padding_warnings, "no padding warning with 7 padded slices on 4 real ones"
    assert "smaller poly_n" in str(padding_warnings[0].message)

    assert out["flow_raw"].shape == (2, 4, 16, 16, 3)
    assert np.isfinite(out["flow_raw"][0]).all()
    assert not out["flow_raw"][-1].any()


# ------------------------------------------------------------------- component order

# Distinct primes so any swap or mis-scaling is a different number.
VX, VY, VZ, CONF = 2.0, 3.0, 5.0, 0.25


def test_3d_components_are_axis_ordered_and_rescaled_per_axis(tmp_path, monkeypatch):
    """(vz, vy, vx) in the store, each times its own axis' downsample factor.

    ``farneback_3d`` is replaced by a fake returning constant components on
    the downsampled grid, so this pins MHAT's stacking and upsampling
    arithmetic and nothing about opticalflow3D's estimate. The factor is
    anisotropic because an isotropic one hides a swapped pairing.
    """
    torch = pytest.importorskip("torch")
    farneback_functions = pytest.importorskip(
        "opticalflow3D.helpers.farneback_functions"
    )
    from mhat.opticalflow.farneback import compute_farneback_flow_3d

    ds_z, ds_y, ds_x = 1, 2, 4
    T, Z, Y, X = 2, 6, 16, 32  # Z = 6 clears the poly_n 3 pyramid minimum
    seen = []

    def fake_farneback_3d(prev, curr, **kwargs):
        seen.append(tuple(prev.shape))
        const = lambda v: torch.full(tuple(prev.shape), v, dtype=torch.float32)  # noqa: E731
        return const(VX), const(VY), const(VZ), const(CONF)  # opticalflow3D's order

    monkeypatch.setattr(farneback_functions, "farneback_3d", fake_farneback_3d)

    movie = np.random.default_rng(0).random((T, Z, Y, X), dtype=np.float32)
    out = run_flow(
        compute_farneback_flow_3d,
        flow_config_3d(downsample_factor=(ds_z, ds_y, ds_x)),
        movie,
        tmp_path,
    )

    # The fake must have been fed the downsampled grid, or the test is vacuous.
    assert seen == [(Z // ds_z, Y // ds_y, X // ds_x)]

    assert out["flow_raw"].attrs["channel_order"] == ["z", "y", "x"]
    flow = out["flow_raw"][0]
    assert flow.shape == (Z, Y, X, 3)
    np.testing.assert_allclose(flow[..., 0], VZ * ds_z, err_msg="vz (component 0)")
    np.testing.assert_allclose(flow[..., 1], VY * ds_y, err_msg="vy (component 1)")
    np.testing.assert_allclose(flow[..., 2], VX * ds_x, err_msg="vx (component 2)")
    np.testing.assert_allclose(out["confidence"][0], CONF)

    # And the reader hands the same thing back, unreversed.
    np.testing.assert_array_equal(open_flow_raw(out)[0], flow)


def test_2d_components_are_axis_ordered(tmp_path, monkeypatch):
    """OpenCV returns (dx, dy); the store must hold (vy, vx)."""
    import cv2

    from mhat.opticalflow.farneback import compute_farneback_flow_2d

    def fake_calc(prev, curr, *args, **kwargs):
        out = np.empty((*prev.shape, 2), dtype=np.float32)
        out[..., 0], out[..., 1] = VX, VY  # OpenCV's (dx, dy)
        return out

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", fake_calc)

    movie = np.random.default_rng(0).random((2, 16, 16), dtype=np.float32)
    out = run_flow(compute_farneback_flow_2d, flow_config_2d(), movie, tmp_path)

    assert out["flow_raw"].attrs["channel_order"] == ["y", "x"]
    flow = out["flow_raw"][0]
    np.testing.assert_allclose(flow[..., 0], VY, err_msg="vy (component 0)")
    np.testing.assert_allclose(flow[..., 1], VX, err_msg="vx (component 1)")


# TODO(publish): delete along with the legacy read path in
# mhat.opticalflow.utils once every flow store has been regenerated.
@pytest.mark.parametrize("ndim", [2, 3], ids=["2d", "3d"])
def test_legacy_store_is_reversed_on_read(tmp_path, ndim):
    """A store with no ``channel_order`` marker is x-first and comes back
    reversed, eagerly and as dask, so that x and z are not silently swapped.
    """
    spatial = (4, 8, 8)[-ndim:]
    legacy = np.empty((2, *spatial, ndim), dtype=np.float32)
    xfirst = (VX, VY, VZ)[:ndim]
    for c, v in enumerate(xfirst):
        legacy[..., c] = v

    root = zarr.open(str(tmp_path / "legacy.zarr"), mode="w")
    root.create_dataset("flow_raw", data=legacy)  # no marker: pre-2026-09-21

    with pytest.warns(UserWarning, match="legacy x-first"):
        eager = open_flow_raw(root)
    with pytest.warns(UserWarning, match="legacy x-first"):
        lazy = open_flow_raw(root, dask=True)

    expected = np.empty_like(legacy)
    for c, v in enumerate(reversed(xfirst)):  # axis order
        expected[..., c] = v
    assert eager.shape == legacy.shape
    np.testing.assert_array_equal(eager[0], expected[0])
    np.testing.assert_array_equal(np.asarray(lazy), expected)
    assert FLOW_CHANNEL_ORDER[ndim] == (["y", "x"] if ndim == 2 else ["z", "y", "x"])

    # Selecting a component in the same subscript would defeat the reversal.
    with pytest.raises(IndexError):
        eager[0, ..., 0]
