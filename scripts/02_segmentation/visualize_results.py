import argparse
from pathlib import Path

import napari
import toml
import zarr
import dask.array as da

from mhat.utils import get_axes_metadata


def main(config, compute=False):
    # Base directories come from the seg config that produced this run, so the
    # viewer works on any machine. create_seg_hypotheses.py reads these same
    # keys and writes a copy of the config next to its output.
    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    experiment = config["experiment"]
    dataset = config["dataset"]
    exp_uid = config["exp_uid"]

    raw_zarr_path = input_base_dir / experiment / f"{dataset}.zarr"
    seg_zarr_path = output_base_dir / experiment / dataset / exp_uid / "data.zarr"

    raw = da.from_zarr(raw_zarr_path)
    raw = raw[:, 0, ...]

    seg = zarr.open(seg_zarr_path, mode='r')
    affinities = da.from_zarr(seg['affinities'])[:, ...]
    fragments = da.from_zarr(seg['fragments'])[:, ...]
    segmentations = da.from_zarr(seg['segmentations'])[:, ...]
    axes = get_axes_metadata(seg['affinities'])
    scale = [axis["scale"] for axis in axes]

    print(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")
    print(f"Affinities shape: {affinities.shape}, dtype: {affinities.dtype}")
    print(f"Fragments shape: {fragments.shape}, dtype: {fragments.dtype}")
    print(f"Segmentations shape: {segmentations.shape}, dtype: {segmentations.dtype}")

    if compute:
        raw = raw.compute()
        affinities = affinities.compute()
        fragments = fragments.compute()
        segmentations = segmentations.compute()

    affinities = 1.0 - affinities

    viewer = napari.Viewer()
    viewer.add_image(raw, name='raw', scale=scale)
    viewer.add_image(affinities, name='affinities', channel_axis=1, contrast_limits=[0, 1], scale=scale)
    viewer.add_labels(fragments, name='fragments', scale=scale)
    viewer.add_labels(segmentations, name='segmentations', opacity=0.5, scale=scale)

    napari.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="View segmentation hypotheses in napari: raw, affinities, "
                    "fragments and agglomerated segmentations. Pass the config.toml "
                    "that create_seg_hypotheses.py wrote next to the results."
    )
    parser.add_argument("config", help="path to a segmentation run's config.toml")
    parser.add_argument(
        "--compute",
        action="store_true",
        help="load the arrays into memory instead of viewing them lazily",
    )
    args = parser.parse_args()
    main(toml.load(args.config), compute=args.compute)
