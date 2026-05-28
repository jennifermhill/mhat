"""Convert MTrackJ Points CSV (sparse 2D ground truth) to geff format.

The MTrackJ Points export has one row per annotated point, with columns:
    Nr, TID, PID, x [micron], y [micron], t [min], ...

TID groups points into tracks. There is no parent/child track relationship
in the points file, so each TID becomes an independent track. The CSV stores
x/y in microns and t in minutes; we convert t to a frame index by dividing
by the acquisition interval (frame_interval_min). z is missing; we set it
to 0 since the raw data is only a few z slices and every cell is in every
slice.

Output node props x/y/z are in world units (microns) to match the
convention used by pred_tracks and from_ctc_to_geff.
"""

import argparse
import shutil
from pathlib import Path

import geff
import numpy as np
import pandas as pd
from geff.core_io import write_arrays
from geff_spec import Axis, GeffMetadata


def from_mtrackj_to_geff(
    points_csv: Path,
    geff_path: Path,
    frame_interval_min: float = 1.0,
    z_value: float = 0.0,
    overwrite: bool = False,
) -> None:
    points_csv = Path(points_csv)
    geff_path = Path(geff_path).with_suffix(".zarr")

    if not points_csv.exists():
        raise FileNotFoundError(f"Points CSV {points_csv} does not exist")

    if geff_path.exists() and not overwrite:
        raise FileExistsError(f"GEFF file {geff_path} already exists")
    if geff_path.exists() and overwrite:
        shutil.rmtree(geff_path)

    df = pd.read_csv(points_csv)

    # The Nr column is unique across the file; use it as the node id.
    node_ids = df["Nr"].to_numpy(dtype=np.uint64)
    track_ids = df["TID"].to_numpy(dtype=np.int64)

    # Convert t [min] to frame index. Check exact divisibility so we fail
    # loudly if frame_interval_min is set wrong for this CSV.
    t_min = df["t [min]"].to_numpy(dtype=np.float64)
    frame_idx_float = t_min / frame_interval_min
    if not np.allclose(frame_idx_float, np.round(frame_idx_float)):
        raise ValueError(
            f"t values in CSV are not integer multiples of frame_interval_min="
            f"{frame_interval_min}. Got t values like {sorted(set(t_min))[:10]}."
        )
    times = np.round(frame_idx_float).astype(np.int64)

    xs = df["x [micron]"].to_numpy(dtype=np.float64)
    ys = df["y [micron]"].to_numpy(dtype=np.float64)
    zs = np.full(len(df), z_value, dtype=np.float64)

    # Build edges: connect consecutive points within each track in time order.
    edges: list[tuple[int, int]] = []
    for _tid, group in df.sort_values(["TID", "t [min]"]).groupby("TID"):
        nrs = group["Nr"].to_numpy()
        for i in range(len(nrs) - 1):
            edges.append((int(nrs[i]), int(nrs[i + 1])))
    edge_ids = np.asarray(edges, dtype=np.uint64)

    # Axes mirror the raw zarr metadata for primary_nk_cells (time in frames,
    # z scale 2.0 micron/pixel, y/x scale 0.5365853970565678 micron/pixel).
    # GT positions are stored in microns (world units), matching pred_tracks.
    axes = [
        Axis(name="time", type="time", scale=1.0),
        Axis(name="z", type="space", scale=2.0),
        Axis(name="y", type="space", scale=0.5365853970565678),
        Axis(name="x", type="space", scale=0.5365853970565678),
    ]

    node_props = {
        "time": times,
        "z": zs,
        "y": ys,
        "x": xs,
        "track_id": track_ids,
    }

    write_arrays(
        geff_store=geff_path,
        node_ids=node_ids,
        node_props={
            name: {"values": values, "missing": None}
            for name, values in node_props.items()
        },
        edge_ids=edge_ids,
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

    print(
        f"Wrote {len(node_ids)} nodes, {len(edge_ids)} edges, "
        f"{df['TID'].nunique()} tracks to {geff_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points_csv", type=Path)
    parser.add_argument("geff_path", type=Path)
    parser.add_argument(
        "--frame-interval-min",
        type=float,
        default=1.0,
        help="Acquisition interval in minutes (t [min] is divided by this to get frame index)",
    )
    parser.add_argument("--z-value", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from_mtrackj_to_geff(
        args.points_csv,
        args.geff_path,
        frame_interval_min=args.frame_interval_min,
        z_value=args.z_value,
        overwrite=args.overwrite,
    )
