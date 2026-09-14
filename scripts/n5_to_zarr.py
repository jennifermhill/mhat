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
    z, y, x = voxel_size
    return [
        {"name": "time", "type": "time", "unit": time_unit,
         "scale": float(time_scale)},
        {"name": "channel", "type": "channel", "scale": 1.0},
        {"name": "z", "type": "space", "unit": "micrometer", "scale": float(z)},
        {"name": "y", "type": "space", "unit": "micrometer", "scale": float(y)},
        {"name": "x", "type": "space", "unit": "micrometer", "scale": float(x)},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_n5", type=Path, help="Path to the .n5 container")
    parser.add_argument("dataset", help="Dataset inside the container, e.g. 'raw'")
    parser.add_argument("output_zarr", type=Path, help="Output .zarr path")
    parser.add_argument("--t-min", type=int, required=True, help="first frame (incl)")
    parser.add_argument("--t-max", type=int, required=True, help="last frame (incl)")
    parser.add_argument(
        "--voxel-size", type=float, nargs=3, metavar=("Z", "Y", "X"), required=True,
        help="Voxel size in micrometers",
    )
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--time-unit", default="second")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # N5Store is deprecated in zarr 2 and gone in zarr 3; if this stops working,
    # read the blocks with the `n5py` package instead.
    from zarr.n5 import N5Store

    src = zarr.open(store=N5Store(str(args.input_n5)), mode="r")[args.dataset]
    if src.ndim != 4:
        raise SystemExit(f"expected a 4D (t, z, y, x) dataset, got {src.shape}")

    n_t = src.shape[0]
    if not 0 <= args.t_min <= args.t_max < n_t:
        raise SystemExit(
            f"--t-min/--t-max {args.t_min}..{args.t_max} outside "
            f"source range 0..{n_t - 1}"
        )

    T = args.t_max - args.t_min + 1
    Z, Y, X = src.shape[1:]
    print(f"source {src.shape} {src.dtype}, chunks {src.chunks}")
    print(f"cropping t {args.t_min}..{args.t_max} -> {T} frames, rebased to 0..{T-1}")

    out = zarr.open(
        str(args.output_zarr),
        mode="w" if args.overwrite else "w-",
        shape=(T, 1, Z, Y, X),
        chunks=(1, 1, 1, Y, X),
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
    print(f"  frame 0 nonzero fraction {nonzero / (Z * Y * X):.3f}")


if __name__ == "__main__":
    main()
