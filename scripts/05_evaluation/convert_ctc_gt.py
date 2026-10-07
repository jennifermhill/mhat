"""Convert Cell Tracking Challenge tracking ground truth to the geff evaluation reads.

Reads a CTC ``TRA`` folder (``man_track*.tif`` markers plus ``man_track.txt``)
and writes two stores into an output directory of your choosing:

- ``correct_tracks.zarr``: the tracks as geff, positions in world units.
- ``correct_seg.zarr``: the markers relabeled by geff node id (the CTC matcher
  needs it).

Point evaluation at them with ``gt_tracks`` / ``gt_seg``. The evaluation
directory's default label is the output directory's name, so name it after the
ground truth (e.g. ``.../gt/official``).

The axes, and with them the world-unit scale applied to every position, come
from the raw movie the ground truth annotates. Existing stores are never
replaced unless ``--overwrite`` is given: hand-made ground truth may be the
only copy of those annotations.

Example:
    python scripts/05_evaluation/convert_ctc_gt.py ctc/Fluo-C3DL-MDA231/01_GT/TRA \
        gt/Fluo-C3DL-MDA231/01_cells/official --raw data/Fluo-C3DL-MDA231/01_cells.zarr
"""

import argparse
from pathlib import Path

import zarr
from geff_spec import Axis

from mhat.evaluation.from_ctc_to_geff import from_ctc_to_geff
from mhat.utils import get_axes_metadata


def axes_from_raw(raw_path: Path) -> list[Axis]:
    """geff axes (name, type, scale) for the raw movie, channel axis dropped."""
    axes = get_axes_metadata(zarr.open(raw_path, mode="r"))
    return [
        Axis(
            name=axis["name"],
            type=axis.get("type", "time" if i == 0 else "space"),
            scale=axis["scale"],
        )
        for i, axis in enumerate(axes)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Convert a CTC TRA folder to correct_tracks/correct_seg.zarr."
    )
    parser.add_argument(
        "tra_dir", type=Path, help="CTC TRA folder (man_track*.tif, man_track.txt)"
    )
    parser.add_argument(
        "output_dir", type=Path, help="directory to write the two stores into"
    )
    parser.add_argument(
        "--raw", type=Path, required=True,
        help="raw movie zarr the ground truth annotates (source of axes and scale)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace correct_tracks.zarr / correct_seg.zarr if they already exist",
    )
    args = parser.parse_args()

    tracks_path = args.output_dir / "correct_tracks.zarr"
    seg_path = args.output_dir / "correct_seg.zarr"
    existing = [p for p in (tracks_path, seg_path) if p.exists()]
    if existing:
        if not args.overwrite:
            raise SystemExit(
                "Refusing to replace existing ground truth:\n"
                + "".join(f"  {p}\n" for p in existing)
                + "Pass --overwrite if you are sure; these may be the only copy."
            )
        print("WARNING: replacing existing ground truth:")
        for p in existing:
            print(f"  {p}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    from_ctc_to_geff(
        ctc_path=args.tra_dir,
        geff_path=tracks_path,
        segmentation_store=seg_path,
        axes=axes_from_raw(args.raw),
        overwrite=args.overwrite,
    )
    print(f"Wrote {tracks_path} and {seg_path}")


if __name__ == "__main__":
    main()
