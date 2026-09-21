"""Pin the Z-padding rule for the 3D Farneback pyramid (fixed 2026-09-21).

``min_size_for_farneback_3d`` replaced a guard that used ``^`` (XOR) for a power
and was right only by coincidence. The two failure modes worth a test are
silent: a minimum that is too *large* pads more replicated edge slices than
necessary and quietly degrades the z flow (too small crashes on its own), and
a warning that fails to fire when padding dominates is, by definition, silent.

The expected minima were established by running
``opticalflow3D.helpers.farneback_functions.farneback_3d`` on synthetic volumes
of increasing Z and recording the first size that did not raise.
"""

import warnings

import numpy as np
import pytest

from mhat.opticalflow.farneback import min_size_for_farneback_3d


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
    other thing that would go wrong quietly (a (T, 11, Y, X) flow written into
    a (T, 4, Y, X) store would raise; a flow computed on the padded volume but
    sliced from the wrong end would not).
    """
    pytest.importorskip("opticalflow3D")
    import dask.array as da
    import zarr

    from mhat.opticalflow.farneback import compute_farneback_flow_3d

    config = {
        "pyr_scale": [0.5, 0.5, 0.5],
        "levels": 3,
        "poly_n": 5,
        "poly_sigma": [0.37, 1.2, 1.2],
        "iterations": 1,
        "winsize": [5, 15, 15],
        "filter_type": "gaussian",
        "presmoothing": 0.1,
        "downsample_factor": [1, 1, 1],
        "hyperparams": {
            "enhance_contrast": False,
            "temporal_smoothing": False,
            "temporal_smoothing_sigma": 0.7,
        },
    }
    T, Z, Y, X = 2, 4, 16, 16
    rng = np.random.default_rng(0)
    movie = da.from_array(rng.random((T, Z, Y, X), dtype=np.float32), chunks=-1)
    out = zarr.open_group(str(tmp_path / "flow.zarr"), mode="w")
    out.create_dataset("flow_raw", shape=(T, Z, Y, X, 3), dtype=np.float32)
    out.create_dataset("confidence", shape=(T, Z, Y, X), dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_farneback_flow_3d(config, movie, out)

    padding_warnings = [w for w in caught if "replicated edge slices" in str(w.message)]
    assert padding_warnings, "no padding warning with 7 padded slices on 4 real ones"
    assert "smaller poly_n" in str(padding_warnings[0].message)

    assert out["flow_raw"].shape == (T, Z, Y, X, 3)
    assert np.isfinite(out["flow_raw"][0]).all()
    assert not out["flow_raw"][-1].any()
