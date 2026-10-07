"""The guards in the output-directory layout that would otherwise fail silently.

- Flow selection: on 3D data a 2D flow, when present, supplies the in-plane
  drift components. Dropping it changes every drift cost without any error.
- Raw binding: tracking reads segmentation runs by uid, so runs made from
  another movie in the same output directory would be used without complaint
  whenever the shapes match.
- Existing runs: tracking into one would leave its evaluations (and any plots
  saved there) attributed to a result they did not come from.
"""

import numpy as np
import pytest
import zarr

from mhat.dataset import Dataset

SEG_UID = "seg"
FLOW_UID = "flow"


def make_dataset(tmp_path, ndim, flow_kinds):
    ds = Dataset(output_dir=tmp_path)
    root = zarr.open(ds.seg_dir(SEG_UID) / "data.zarr", mode="w")
    root.create_dataset("fragments", data=np.zeros((2,) + (4,) * ndim, np.uint32))
    for kind in flow_kinds:
        ds.flow_dir(FLOW_UID, kind).mkdir(parents=True)
    return ds


@pytest.mark.parametrize(
    "ndim, present, expected",
    [
        (3, ["2d", "3d"], {"2d": True, "3d": True}),
        (3, ["3d"], {"2d": False, "3d": True}),
        # A 3D flow on disk is ignored for 2D data.
        (2, ["2d", "3d"], {"2d": True, "3d": False}),
    ],
)
def test_flow_selection(tmp_path, ndim, present, expected):
    ds = make_dataset(tmp_path, ndim, present)
    dirs = ds.flow_dirs(FLOW_UID, SEG_UID)
    for kind, used in expected.items():
        assert dirs[kind] == (ds.flow_dir(FLOW_UID, kind) if used else None)


@pytest.mark.parametrize("ndim, present", [(3, ["2d"]), (2, ["3d"])])
def test_missing_required_flow_raises(tmp_path, ndim, present):
    ds = make_dataset(tmp_path, ndim, present)
    with pytest.raises(FileNotFoundError):
        ds.flow_dirs(FLOW_UID, SEG_UID)


def test_output_dir_is_bound_to_one_raw_movie(tmp_path):
    raw_a, raw_b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    for raw in (raw_a, raw_b):
        zarr.open(raw, mode="w", shape=(2, 1, 4, 4), dtype=np.uint8)
    output_dir = tmp_path / "out"

    Dataset(output_dir, raw_a).bind_raw()  # first stage records movie a
    Dataset(output_dir, raw_a).bind_raw()  # same movie: fine
    with pytest.raises(ValueError, match="raw movie"):
        Dataset(output_dir, raw_b).bind_raw()


def test_existing_tracking_run_is_never_reused(tmp_path):
    ds = Dataset(output_dir=tmp_path)
    track_dir = ds.prepare_run_dir("tracking", "run")
    (track_dir / "pred_tracks.zarr").mkdir()
    ds.new_eval_dir("run", "official", ["ctc"])

    with pytest.raises(FileExistsError):
        ds.prepare_run_dir("tracking", "run")
    assert len(ds.evals("run")) == 1  # refused, and nothing removed
