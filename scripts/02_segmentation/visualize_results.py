import argparse
from pathlib import Path

import napari
import zarr
import dask.array as da

from mhat.dataset import Dataset, absolute_path
from mhat.utils import get_axes_metadata


def main(run_dir: Path, raw_path: Path | None = None, compute=False):
    # The output directory is found from where the run sits, so this works on
    # any machine; raw comes from raw.toml unless --raw overrides it.
    run_dir = absolute_path(run_dir)
    if run_dir.name == "config.toml":
        run_dir = run_dir.parent
    ds = Dataset.from_run_dir(run_dir, "segmentation")
    raw_path = raw_path or ds.raw_path
    seg_zarr_path = run_dir / "data.zarr"

    seg = zarr.open(seg_zarr_path, mode='r')
    affinities = da.from_zarr(seg['affinities'])[:, ...]
    fragments = da.from_zarr(seg['fragments'])[:, ...]
    if 'segmentations' in seg:
        segmentations = da.from_zarr(seg['segmentations'])[:, ...]
    axes = get_axes_metadata(seg['affinities'])
    scale = [axis["scale"] for axis in axes]

    if raw_path is not None and Path(raw_path).exists():
        raw = da.from_zarr(raw_path)[:, 0, ...]
        print(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")
    else:
        print(f"Raw data not found at {raw_path}; pass --raw to show it.")
        raw = None
    print(f"Affinities shape: {affinities.shape}, dtype: {affinities.dtype}")
    print(f"Fragments shape: {fragments.shape}, dtype: {fragments.dtype}")
    if 'segmentations' in seg:
        print(f"Segmentations shape: {segmentations.shape}, dtype: {segmentations.dtype}")

    if compute:
        raw = raw.compute() if raw is not None else None
        affinities = affinities.compute()
        fragments = fragments.compute()
        if 'segmentations' in seg:
            segmentations = segmentations.compute()

    affinities = 1.0 - affinities

    viewer = napari.Viewer()
    if raw is not None:
        viewer.add_image(raw, name='raw', scale=scale)
    viewer.add_image(affinities, name='affinities', channel_axis=1, contrast_limits=[0, 1], scale=scale)
    viewer.add_labels(fragments, name='fragments', scale=scale)
    if 'segmentations' in seg:
        viewer.add_labels(segmentations, name='segmentations', opacity=0.5, scale=scale)

    napari.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="View segmentation hypotheses in napari: raw, affinities, "
                    "fragments and agglomerated segmentations."
    )
    parser.add_argument(
        "run", type=Path,
        help="a segmentation run, <output_dir>/segmentation/<uid> (or its config.toml)",
    )
    parser.add_argument(
        "--raw", type=Path, default=None,
        help="raw movie to show, if the path in raw.toml is not valid on this machine",
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="load the arrays into memory instead of viewing them lazily",
    )
    args = parser.parse_args()
    main(args.run, raw_path=args.raw, compute=args.compute)
