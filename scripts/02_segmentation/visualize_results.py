from pathlib import Path
import napari
import zarr
import csv
import numpy as np
import networkx as nx


def main(raw_zarr_path, seg_zarr_path):
    raw = zarr.open(raw_zarr_path, mode='r')
    seg = zarr.open(seg_zarr_path, mode='r')
    raw = raw[:, 0, ...]

    affinities = seg['affinities'][:, ...]
    fragments = seg['fragments'][:, ...]
    segmentations = seg['segmentations'][:, ...]

    print(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")
    print(f"Affinities shape: {affinities.shape}, dtype: {affinities.dtype}")
    print(f"Fragments shape: {fragments.shape}, dtype: {fragments.dtype}")
    print(f"Segmentations shape: {segmentations.shape}, dtype: {segmentations.dtype}")

    viewer = napari.Viewer()
    viewer.add_image(raw, name='raw')
    viewer.add_image(affinities, name='affinities', channel_axis=1, contrast_limits=[0, 1])
    viewer.add_labels(fragments, name='fragments')
    viewer.add_labels(segmentations, name='segmentations', opacity=0.5)

    napari.run()

if __name__ == '__main__':
    raw_zarr_path = '/groups/sgro/sgrolab/jennifer/mhat/data/nc281-spiAmSG_agg/02_cells.zarr'
    seg_zarr_path = '/groups/sgro/sgrolab/jennifer/mhat/experiments/segmentation/nc281-spiAmSG_agg/02_cells/data.zarr'
    # raw_zarr_path = '/Volumes/sgrolab/jennifer/mhat/data/nc281-spiAmSG/01_cells.zarr'
    # seg_zarr_path = '/Volumes/sgrolab/jennifer/mhat/experiments/segmentation/nc281-spiAmSG/01_cells/data.zarr'
    main(raw_zarr_path, seg_zarr_path)
