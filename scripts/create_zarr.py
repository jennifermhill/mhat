import argparse
import os

import numpy as np
import zarr
from tqdm import tqdm

from mhat.utils import get_axes_from_ome_xml, check_uint16_safe


def main(zarr_path, output_dir, datasets):
    """Split the channels of an OME-zarr into one zarr per dataset.

    Args:
        zarr_path: path to the source OME-zarr.
        output_dir: directory to write ``<dataset>.zarr`` into.
        datasets: mapping of output dataset name to source channel index,
            e.g. ``{"01_nuclei": 0, "01_cells": 1}``.
    """
    root = zarr.open(zarr_path, mode='r')
    img = root['0']['0']

    axes = get_axes_from_ome_xml(zarr_path)

    T, _, Z, Y, X = img.shape
    for dataset in datasets.keys():
        dataset_zarr_path = os.path.join(output_dir, f"{dataset}.zarr")
        os.makedirs(dataset_zarr_path, exist_ok=True)
        print(f"Creating Zarr dataset for {dataset} at {dataset_zarr_path}")
        dataset_zarr = zarr.open(dataset_zarr_path, 
                                 mode="a", 
                                 shape=(T, 1, Z, Y, X), 
                                 chunks=(1, 1, 1, Y, X),
                                 dtype=np.uint16,
                                 )
        dataset_zarr.attrs["axes"] = axes
        for tp in tqdm(range(T)):
            frame = img[tp, datasets[dataset]]
            check_uint16_safe(frame, tp)
            dataset_zarr[tp, 0] = frame

def _parse_datasets(values):
    """Turn ["01_nuclei=0", "01_cells=1"] into {"01_nuclei": 0, "01_cells": 1}."""
    datasets = {}
    for item in values:
        name, _, index = item.partition("=")
        if not name or not index.isdigit():
            raise argparse.ArgumentTypeError(
                f"expected NAME=CHANNEL_INDEX, got {item!r}"
            )
        datasets[name] = int(index)
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the channels of an OME-zarr into one zarr per dataset."
    )
    parser.add_argument("zarr_path", help="path to the source OME-zarr")
    parser.add_argument("output_dir", help="directory to write <dataset>.zarr into")
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="NAME=CHANNEL",
        help="output dataset name and its source channel index, "
             "e.g. 01_nuclei=0 01_cells=1",
    )
    args = parser.parse_args()
    main(args.zarr_path, args.output_dir, _parse_datasets(args.datasets))