"""Convert a Cell Tracking Challenge image sequence (tXXX.tif) to the flat
(t, c, *spatial) zarr used by the rest of the pipeline.

The CTC folders hold one TIFF per timepoint: a (z, y, x) stack for a 3D dataset
or a (y, x) image for a 2D one. This writes a single zarr array with a singleton
channel axis and an "axes" attribute matching the other raw zarrs in data/ (see
scripts/create_zarr.py) -- (t, c, z, y, x) with four axes for 3D data, or
(t, c, y, x) with three for 2D. The rank follows the TIFFs; nothing is padded.

Voxel size is read from the TIFF resolution tags when available (CTC exports put
z spacing in private tag 65001) and can be overridden on the command line.

Example:
    python scripts/ctc_to_zarr.py ctc/Fluo-N3DL-DRO/01 data/Fluo-N3DL-DRO/01_nuclei.zarr \
        --time-scale 30 --time-unit second
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
import zarr
from tqdm import tqdm

# TIFF ResolutionUnit values -> micrometers per unit
_UNIT_TO_UM = {1: None, 2: 25400.0, 3: 10000.0}  # none, inch, centimeter

# Private tag some CTC exports use for z resolution, same rational form as
# XResolution/YResolution.
_Z_RESOLUTION_TAG = 65001


def _rational_to_um(value, unit_um):
    """Convert a TIFF (numerator, denominator) resolution to micrometers/pixel."""
    if value is None or unit_um is None:
        return None
    numerator, denominator = value
    if not numerator or not denominator:
        return None
    pixels_per_unit = numerator / denominator
    if pixels_per_unit <= 0:
        return None
    return unit_um / pixels_per_unit


def get_voxel_size_from_tiff(tiff_path):
    """Read (z, y, x) voxel size in micrometers from a TIFF's resolution tags.

    Returns None for any axis whose tags are missing or unusable.
    """
    with tiff.TiffFile(tiff_path) as tif:
        tags = tif.pages[0].tags
        unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 3
        unit_um = _UNIT_TO_UM.get(int(unit))

        def tag_value(key):
            return tags[key].value if key in tags else None

        x = _rational_to_um(tag_value("XResolution"), unit_um)
        y = _rational_to_um(tag_value("YResolution"), unit_um)
        z = _rational_to_um(tag_value(_Z_RESOLUTION_TAG), unit_um)
    return z, y, x


def build_axes(voxel_size, time_scale, time_unit):
    """Axes metadata for a (t, c, *spatial) raw zarr.

    One space axis per ``voxel_size`` entry, named from the tail of (z, y, x),
    so a 2-tuple describes a 2D movie and a 3-tuple a 3D one.
    """
    names = ("z", "y", "x")[-len(voxel_size):]
    return [
        dict(name="time", type="time", unit=time_unit, scale=float(time_scale)),
        dict(name="channel", type="channel", scale=1.0),
    ] + [
        dict(name=name, type="space", unit="micrometer", scale=float(size))
        for name, size in zip(names, voxel_size, strict=True)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="CTC sequence dir of tXXX.tif")
    parser.add_argument("output_zarr", type=Path, help="Output .zarr path")
    parser.add_argument(
        "--time-scale", type=float, default=1.0, help="Frame interval (default 1.0)"
    )
    parser.add_argument(
        "--time-unit", default="second", help="Unit for --time-scale (default second)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs="+",
        metavar="SIZE",
        help="Voxel size in micrometers, overriding the TIFF resolution tags. "
             "Z Y X for a 3D sequence, Y X for a 2D one",
    )
    parser.add_argument(
        "--dtype", default=None, help="Output dtype (default: the TIFF dtype)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output zarr"
    )
    args = parser.parse_args()

    tiff_paths = sorted(args.input_dir.glob("t*.tif"))
    if not tiff_paths:
        raise SystemExit(f"No t*.tif files found in {args.input_dir}")
    print(f"Found {len(tiff_paths)} TIFFs in {args.input_dir}")

    with tiff.TiffFile(tiff_paths[0]) as tif:
        series = tif.series[0]
        shape = tuple(series.shape)
        src_dtype = series.dtype
    if len(shape) not in (2, 3):
        raise SystemExit(
            f"Expected (z, y, x) or (y, x) TIFFs, got shape {shape}"
        )

    T = len(tiff_paths)
    ndim = len(shape)
    axis_labels = ("z", "y", "x")[-ndim:]
    dtype = np.dtype(args.dtype) if args.dtype else src_dtype

    # The TIFF tags are read as (z, y, x); a 2D sequence uses the tail of that.
    voxel_size = args.voxel_size or get_voxel_size_from_tiff(tiff_paths[0])[-ndim:]
    if len(voxel_size) != ndim:
        raise SystemExit(
            f"--voxel-size takes {ndim} values for these {ndim}D TIFFs "
            f"({' '.join(n.upper() for n in axis_labels)}), got {len(voxel_size)}"
        )
    if any(v is None for v in voxel_size):
        raise SystemExit(
            f"Could not read voxel size from {tiff_paths[0].name} "
            f"(got {', '.join(axis_labels)} = {voxel_size}); pass --voxel-size "
            f"{' '.join(n.upper() for n in axis_labels)}"
        )
    axes = build_axes(voxel_size, args.time_scale, args.time_unit)

    # One chunk per displayed plane: singleton along everything but the last two.
    chunks = (1, 1) + (1,) * (ndim - 2) + shape[-2:]

    print(f"Shape (t, c, {', '.join(axis_labels)}): {(T, 1, *shape)}, dtype: {dtype}")
    print(f"Voxel size ({', '.join(axis_labels)}) um: "
          f"{tuple(float(v) for v in voxel_size)}")

    args.output_zarr.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open(
        str(args.output_zarr),
        mode="w" if args.overwrite else "a",
        shape=(T, 1, *shape),
        chunks=chunks,
        dtype=dtype,
    )
    out.attrs["axes"] = axes

    for tp, tiff_path in enumerate(tqdm(tiff_paths, desc="Writing timepoints")):
        frame = tiff.imread(tiff_path)
        if frame.shape != shape:
            raise SystemExit(
                f"{tiff_path.name} has shape {frame.shape}, expected {shape}"
            )
        out[tp, 0] = frame.astype(dtype, copy=False)

    print(f"Wrote {args.output_zarr}")


if __name__ == "__main__":
    main()
