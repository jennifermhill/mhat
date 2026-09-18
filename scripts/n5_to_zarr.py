"""Crop a time range out of an N5 volume into the flat 5D zarr used by the pipeline.

Written for the 120828 Drosophila sequence published with linajea
(Malin-Mayor et al. 2023), whose N5 is 4D ``(x, y, z, t)`` in N5's F-order —
zarr's ``N5Store`` presents that as C-order ``(t, z, y, x)``. The output adds
the singleton channel axis and the ``axes`` attribute the rest of the pipeline
expects, matching ``scripts/ctc_to_zarr.py``.

Frames are re-based so the first kept frame becomes 0. Keep the source range
in mind when pairing with ground truth: the matching GT crop must use the same
``--t-min/--t-max`` with ``linajea_to_geff.py --rebase-time``.

Example:
    python scripts/n5_to_zarr.py \
        data/Fluo-N3DL-DRO/120828_full/120828.n5 raw \
        data/Fluo-N3DL-DRO/03_nuclei_div.zarr \
        --t-min 261 --t-max 310 \
        --voxel-size 2.03 0.406 0.406 --time-scale 30
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm


def build_axes(voxel_size, time_scale, time_unit):
    """Axes metadata for a (t, c, *spatial) raw zarr.

    One space axis per ``voxel_size`` entry, named from the tail of (z, y, x),
    so a 2-tuple describes a 2D movie and a 3-tuple a 3D one.
    """
    names = ("z", "y", "x")[-len(voxel_size):]
    return [
        {"name": "time", "type": "time", "unit": time_unit,
         "scale": float(time_scale)},
        {"name": "channel", "type": "channel", "scale": 1.0},
    ] + [
        {"name": name, "type": "space", "unit": "micrometer", "scale": float(size)}
        for name, size in zip(names, voxel_size, strict=True)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_n5", type=Path, help="Path to the .n5 container")
    parser.add_argument("dataset", help="Dataset inside the container, e.g. 'raw'")
    parser.add_argument("output_zarr", type=Path, help="Output .zarr path")
    parser.add_argument("--t-min", type=int, required=True, help="first frame (incl)")
    parser.add_argument("--t-max", type=int, required=True, help="last frame (incl)")
    parser.add_argument(
        "--voxel-size", type=float, nargs="+", metavar="SIZE", required=True,
        help="Voxel size in micrometers: Z Y X for a 3D volume, Y X for a 2D one",
    )
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--time-unit", default="second")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # N5Store is deprecated in zarr 2 and gone in zarr 3; if this stops working,
    # read the blocks with the `n5py` package instead.
    from zarr.n5 import N5Store

    src = zarr.open(store=N5Store(str(args.input_n5)), mode="r")[args.dataset]
    if src.ndim not in (3, 4):
        raise SystemExit(
            f"expected a (t, z, y, x) or (t, y, x) dataset, got {src.shape}"
        )
    if src.ndim - 1 != len(args.voxel_size):
        raise SystemExit(
            f"--voxel-size has {len(args.voxel_size)} values but the source is "
            f"{src.ndim - 1}D: {src.shape}"
        )

    n_t = src.shape[0]
    if not 0 <= args.t_min <= args.t_max < n_t:
        raise SystemExit(
            f"--t-min/--t-max {args.t_min}..{args.t_max} outside "
            f"source range 0..{n_t - 1}"
        )

    T = args.t_max - args.t_min + 1
    spatial_shape = src.shape[1:]
    # One chunk per displayed plane: singleton along everything but the last two.
    chunks = (1, 1) + (1,) * (len(spatial_shape) - 2) + spatial_shape[-2:]
    print(f"source {src.shape} {src.dtype}, chunks {src.chunks}")
    print(f"cropping t {args.t_min}..{args.t_max} -> {T} frames, rebased to 0..{T-1}")

    out = zarr.open(
        str(args.output_zarr),
        mode="w" if args.overwrite else "w-",
        shape=(T, 1, *spatial_shape),
        chunks=chunks,
        dtype=src.dtype,
        compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
    )
    out.attrs["axes"] = build_axes(args.voxel_size, args.time_scale, args.time_unit)

    # Read in source-chunk-aligned groups so each compressed block is
    # decompressed once rather than once per frame.
    step = src.chunks[0]
    start = args.t_min - (args.t_min % step)
    for block_start in tqdm(
        range(start, args.t_max + 1, step), desc="time blocks", unit="blk"
    ):
        lo = max(block_start, args.t_min)
        hi = min(block_start + step, args.t_max + 1)
        data = src[lo:hi]
        for i, t in enumerate(range(lo, hi)):
            out[t - args.t_min, 0] = data[i]

    nonzero = np.count_nonzero(out[0, 0])
    print(f"wrote {args.output_zarr}")
    print(f"  shape {out.shape} chunks {out.chunks} dtype {out.dtype}")
    print(f"  frame 0 nonzero fraction {nonzero / int(np.prod(spatial_shape)):.3f}")


if __name__ == "__main__":
    main()
