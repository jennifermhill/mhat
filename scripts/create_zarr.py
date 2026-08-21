import os

import numpy as np
import zarr
from tqdm import tqdm

from mhat.utils import get_axes_from_ome_xml


def main(zarr_path):
    
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
            dataset_zarr[tp, 0] = frame

if __name__ == "__main__":
    zarr_path = "/groups/sgro/sgrolab/jennifer/cryolite/Natalie/260228InterfaceTransferMirrorB1-DB.zarr"
    output_dir = "/groups/sgro/sgrolab/jennifer/mhat/data/InterfaceTransferMirrorB1-DB"
    datasets = {"01_nuclei": 0, "01_cells": 1}
    main(zarr_path)