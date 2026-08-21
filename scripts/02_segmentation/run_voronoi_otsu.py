import argparse
from pathlib import Path

import numpy as np
from mhat.data import (
    RawDataZarr,
    SegmentationZarr,
    add_data_args,
    add_segmentation_args,
)
from mhat.utils import get_experiment_metadata
from mhat.segmentation.voronoi_otsu import voronoi_otsu_labeling


def segment_data(
    raw_zarr: RawDataZarr,
    seg_zarr: SegmentationZarr,
    spot_sigma: int,
    outline_sigma: int,
    skip_if_present=True,
    exp_metadata=None,
):
    """Run voronoi otsu segmentation on all fluorescent channels for all fovs in the
    raw dataset, and save results in the segmentation zarr.

    Args:
        raw_zarr (RawDataZarr): zarr containing the raw data to be segmented. Will run
            segmentation on all fovs and fluorescent channels present.
        seg_zarr (SegmentationZarr): zarr in which to store the raw data. Should have
            write capability (be opened with write-enabled mode).
        spot_sigma (int): Voronoi otsu hyperparameter that controls how much blurring
            before local maxima are extracted for watershed seeds. Lower -> more seeds.
        outline_sigma (int): Voronoi otsu hyperparameter that controls how much blurring
            before thresholding. Lower -> tigher (but noisier) segmentations.
        skip_if_present (bool, optional): Skip computation if seg array already exists.
            Defaults to True.
        exp_metadata (dict, optional): Dictionary of experimental metadata to save in
            results zarr array attributes. Defaults to None.
    """
    fluorescent_prefixes = [
        "mCherry",
        "RFP",
        "YFP",
        "GFP",
        "CFP",
    ]
    for fov in raw_zarr.get_fovs():
        for channel in raw_zarr.get_channels(fov):
            # skip non-fluorescent channels
            if not any(prefix in channel for prefix in fluorescent_prefixes):
                continue
            if seg_zarr.has_segmentation(fov, channel) and skip_if_present:
                print(
                    f"Result already present for {seg_zarr.dataset_name} "
                    f"{fov} {channel}, skipping."
                )
                continue
            raw_data = raw_zarr.get_data(fov, channel)
            seg_arr = seg_zarr.get_data(
                fov, channel, shape=raw_data.shape, dtype="uint16"
            )
            if exp_metadata:
                seg_arr.attrs.update(**exp_metadata)
            for i, frame in enumerate(raw_data):
                labeling = voronoi_otsu_labeling(
                    frame.squeeze(), spot_sigma=spot_sigma, outline_sigma=outline_sigma
                )
                seg_arr[i, 0] = labeling.astype(np.uint16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_segmentation_args(parser)
    parser.add_argument("--cluster", type=bool, default=False)
    parser.add_argument("--overwrite", type=bool, default=False)
    args = parser.parse_args()

    spot_sigma = 2
    outline_sigma = 1

    if not args.result_name:
        result_name = f"voronoi_otsu_{spot_sigma}_{outline_sigma}"
    else:
        result_name = args.result_name

    if args.cluster:
        if args.data_base_path.startswith("/Volumes"):
            args.data_base_path = args.data_base_path.replace("/Volumes", "/nrs")
        if args.segmentation_base_path.startswith("/Volumes"):
            args.segmentation_base_path = args.segmentation_base_path.replace(
                "/Volumes", "/nrs"
            )
    base_path = Path(args.data_base_path)
    if not args.dataset_name:
        datasets = [s.name for s in base_path.iterdir() if s.is_dir()]
        print(datasets)
    else:
        datasets = [args.dataset_name]
    if args.overwrite:
        mode = "w"
    else:
        mode = "a"

    for ds_name in datasets:
        print(f"Running voronoi otsu for {ds_name} for all fovs and channels.")
        raw_zarr = RawDataZarr(base_path, ds_name, mode="r", store_type="flat")
        # make dataset dir if necessary
        seg_dataset_path = Path(args.segmentation_base_path) / ds_name
        if not seg_dataset_path.exists():
            print(f"Making output directory at {seg_dataset_path}")
            seg_dataset_path.mkdir()

        seg_zarr = SegmentationZarr(
            args.segmentation_base_path, ds_name, result_name, mode=mode
        )

        segment_data(
            raw_zarr,
            seg_zarr,
            spot_sigma=spot_sigma,
            outline_sigma=outline_sigma,
            skip_if_present=not args.overwrite,
            exp_metadata=get_experiment_metadata(),
        )
