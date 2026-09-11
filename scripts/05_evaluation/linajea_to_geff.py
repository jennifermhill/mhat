"""Convert linajea-style ground truth track files to the repo's geff format.

Source of the data
------------------
Malin-Mayor et al. 2023, "Automated reconstruction of whole-embryo cell
lineages by learning from sparse annotations" (linajea). The drosophila
(120828) ground truth is published at

    https://janelia.figshare.com/articles/dataset/24937092   (gt_tracks.zip)

and contains four tab-separated files with the columns

    time  z  y  x  cell_id  parent_id  track_id

``track_id`` is ``-1`` in every published row, so tracklet ids are derived
here (see ``assign_tracklets``). ``parent_id`` is ``-1`` for a track start.

Coordinate convention (important)
---------------------------------
The ``z`` column is **not** a voxel index: it is a world coordinate expressed
in units of the *xy* pixel, i.e. ``z_voxel = z_column / anisotropy`` where
``anisotropy = voxel_size_z / voxel_size_x`` (5.0 for this dataset:
2.03 um / 0.406 um). The ``y`` and ``x`` columns are already voxel indices.
Verified empirically: the CTC ``Fluo-N3DL-DRO`` TRA markers coincide with
these points to a median nearest-neighbour distance of 0.0 voxels once z is
divided by 5.

Output positions are written in **micrometers**, matching every other geff in
this repo (``from_ctc_to_geff`` scales its centroids by the axis scale, and
``funtracks.import_from_geff`` does not re-scale on read).

Time convention
---------------
Times are those of the original 120828 sequence. The Cell Tracking Challenge
``Fluo-N3DL-DRO`` sequences are 50-frame excerpts of it:

    CTC 01 frame 0  ==  linajea time 5
    CTC 02 frame 0  ==  linajea time 115

Use ``--t-min/--t-max`` with ``--rebase-time`` to emit GT on a CTC frame
numbering (e.g. ``--t-min 5 --t-max 54 --rebase-time`` for sequence 01).
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import geff
import numpy as np
from geff.core_io import write_arrays
from geff_spec import Axis, GeffMetadata

COLUMNS = ("time", "z", "y", "x", "cell_id", "parent_id", "track_id")

# Fluo-N3DL-DRO / 120828 acquisition parameters, taken from the axis metadata
# on data/Fluo-N3DL-DRO/01_nuclei.zarr.
DEFAULT_VOXEL_SIZE = (2.03, 0.406, 0.406)  # um per voxel, (z, y, x)
DEFAULT_TIME_SCALE = 30.0  # seconds per frame


def load_linajea_tracks(path: Path) -> np.ndarray:
    """Read a linajea track file into an (N, 7) int array."""
    arr = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if arr.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path} has {arr.shape[1]} columns, expected {len(COLUMNS)}: {COLUMNS}"
        )
    return arr


def crop_time(arr: np.ndarray, t_min: int | None, t_max: int | None) -> np.ndarray:
    """Keep only rows whose time falls in [t_min, t_max] (inclusive)."""
    keep = np.ones(len(arr), dtype=bool)
    if t_min is not None:
        keep &= arr[:, 0] >= t_min
    if t_max is not None:
        keep &= arr[:, 0] <= t_max
    return arr[keep]


def assign_tracklets(
    cell_ids: np.ndarray, parent_ids: np.ndarray
) -> tuple[np.ndarray, int]:
    """Assign a CTC-style tracklet id to every node.

    A tracklet breaks at every division, matching the semantics of the
    ``track_id`` property written by ``from_ctc_to_geff``: a new tracklet
    starts at a node that has no parent present, or whose parent has more
    than one child.

    Returns the per-node tracklet ids and the number of divisions found.
    """
    present = set(cell_ids.tolist())
    children: dict[int, list[int]] = defaultdict(list)
    for cid, pid in zip(cell_ids.tolist(), parent_ids.tolist(), strict=True):
        if pid in present:
            children[pid].append(cid)

    n_divisions = sum(1 for kids in children.values() if len(kids) > 1)

    # Roots: no parent at all, or a parent that time-cropping removed.
    roots = [
        cid
        for cid, pid in zip(cell_ids.tolist(), parent_ids.tolist(), strict=True)
        if pid not in present
    ]

    tracklet_of: dict[int, int] = {}
    next_id = 1
    for root in roots:
        stack = [(root, next_id)]
        next_id += 1
        while stack:
            node, tid = stack.pop()
            tracklet_of[node] = tid
            kids = children.get(node, ())
            if len(kids) == 1:
                stack.append((kids[0], tid))  # tracklet continues
            else:
                for kid in kids:  # division: each child starts a new tracklet
                    stack.append((kid, next_id))
                    next_id += 1

    missing = present - set(tracklet_of)
    if missing:
        raise ValueError(
            f"{len(missing)} nodes were not reachable from any track start; "
            "the lineage forest is malformed (cycle or duplicate cell_id?)"
        )
    track_ids = np.array([tracklet_of[c] for c in cell_ids.tolist()], dtype=np.int64)
    return track_ids, n_divisions


def linajea_to_geff(
    tracks_path: Path,
    geff_path: Path,
    voxel_size: tuple[float, float, float] = DEFAULT_VOXEL_SIZE,
    time_scale: float = DEFAULT_TIME_SCALE,
    z_anisotropy: float | None = None,
    t_min: int | None = None,
    t_max: int | None = None,
    rebase_time: bool = False,
    overwrite: bool = False,
) -> None:
    """Convert one linajea track file to a geff zarr.

    Args:
        tracks_path: The linajea ``*.txt`` track file.
        geff_path: Output path; forced to a ``.zarr`` suffix.
        voxel_size: Micrometers per voxel as ``(z, y, x)``.
        time_scale: Seconds per frame, recorded on the time axis.
        z_anisotropy: Divisor turning the ``z`` column into a voxel index.
            Defaults to ``voxel_size[0] / voxel_size[2]``.
        t_min: Drop rows before this time (inclusive bound), if given.
        t_max: Drop rows after this time (inclusive bound), if given.
        rebase_time: Shift times so the earliest kept frame becomes 0.
        overwrite: Replace ``geff_path`` if it already exists.
    """
    tracks_path = Path(tracks_path)
    geff_path = Path(geff_path).with_suffix(".zarr")

    if geff_path.exists():
        if not overwrite:
            raise FileExistsError(f"{geff_path} already exists (pass --overwrite)")
        shutil.rmtree(geff_path)

    if z_anisotropy is None:
        z_anisotropy = voxel_size[0] / voxel_size[2]

    arr = load_linajea_tracks(tracks_path)
    n_all = len(arr)
    arr = crop_time(arr, t_min, t_max)
    if len(arr) == 0:
        raise ValueError(
            f"no rows left in {tracks_path} after cropping to time [{t_min}, {t_max}]"
        )

    source_t_min = int(arr[:, 0].min())
    source_t_max = int(arr[:, 0].max())
    time_offset = source_t_min if rebase_time else 0
    times = arr[:, 0] - time_offset

    cell_ids = arr[:, 4]
    parent_ids = arr[:, 5]
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError(f"cell_id is not unique in {tracks_path}")

    track_ids, n_divisions = assign_tracklets(cell_ids, parent_ids)

    # z column is in xy-pixel world units -> voxel index -> micrometers.
    z_um = (arr[:, 1] / z_anisotropy) * voxel_size[0]
    y_um = arr[:, 2] * voxel_size[1]
    x_um = arr[:, 3] * voxel_size[2]

    present = set(cell_ids.tolist())
    edges = np.array(
        [
            (pid, cid)  # forward in time (parent -> child)
            for cid, pid in zip(cell_ids.tolist(), parent_ids.tolist(), strict=True)
            if pid in present
        ],
        dtype=np.uint64,
    ).reshape(-1, 2)

    axes = [
        Axis(name="time", type="time", scale=time_scale, unit="second"),
        Axis(name="z", type="space", scale=voxel_size[0], unit="micrometer"),
        Axis(name="y", type="space", scale=voxel_size[1], unit="micrometer"),
        Axis(name="x", type="space", scale=voxel_size[2], unit="micrometer"),
    ]

    node_props = {
        "time": times,
        "track_id": track_ids,
        "z": z_um,
        "y": y_um,
        "x": x_um,
        # Keep the published identifiers so rows can be traced back to the
        # figshare files after tracklet renumbering.
        "cell_id": cell_ids,
    }

    write_arrays(
        geff_store=geff_path,
        node_ids=cell_ids.astype(np.uint64),
        node_props={
            name: {"values": np.asarray(values), "missing": None}
            for name, values in node_props.items()
        },
        edge_ids=edges,
        edge_props={},
        metadata=GeffMetadata(
            geff_version=geff.__version__,
            axes=axes,
            directed=True,
            node_props_metadata={},
            edge_props_metadata={},
            track_node_props={"tracklet": "track_id"},
        ),
        zarr_format=2,
    )

    print(f"{tracks_path.name} -> {geff_path}")
    print(f"  rows {n_all} -> {len(arr)} kept")
    print(
        f"  time {int(times.min())}..{int(times.max())} "
        f"(source {source_t_min}..{source_t_max}, offset {time_offset})"
    )
    print(
        f"  nodes {len(cell_ids)}  edges {len(edges)}  "
        f"tracklets {len(np.unique(track_ids))}  divisions {n_divisions}"
    )
    print(
        f"  z {z_um.min():.2f}..{z_um.max():.2f} um, "
        f"y {y_um.min():.2f}..{y_um.max():.2f} um, "
        f"x {x_um.min():.2f}..{x_um.max():.2f} um"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tracks_txt", type=Path, help="linajea track file")
    parser.add_argument("output_geff", type=Path, help="output geff zarr path")
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=list(DEFAULT_VOXEL_SIZE),
        metavar=("Z", "Y", "X"),
        help="micrometers per voxel",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=DEFAULT_TIME_SCALE,
        help="seconds per frame",
    )
    parser.add_argument(
        "--z-anisotropy",
        type=float,
        default=None,
        help="divisor for the z column (default voxel_z/voxel_x)",
    )
    parser.add_argument(
        "--t-min", type=int, default=None, help="drop times below this (inclusive)"
    )
    parser.add_argument(
        "--t-max", type=int, default=None, help="drop times above this (inclusive)"
    )
    parser.add_argument(
        "--rebase-time",
        action="store_true",
        help="shift the earliest kept frame to 0",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    linajea_to_geff(
        tracks_path=args.tracks_txt,
        geff_path=args.output_geff,
        voxel_size=tuple(args.voxel_size),
        time_scale=args.time_scale,
        z_anisotropy=args.z_anisotropy,
        t_min=args.t_min,
        t_max=args.t_max,
        rebase_time=args.rebase_time,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
