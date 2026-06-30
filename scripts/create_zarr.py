import zarr
import os
import napari
import tifffile as tiff
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as et


def get_voxel_dims_from_XML(xml_file, res_lvl=0):
    XML_tree = et.parse(xml_file)
    XML_root = XML_tree.getroot()
    
    namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    
    images = XML_root.findall('ome:Image', namespaces)
    if res_lvl >= len(images):
        res_lvl = 0
    
    pixels = images[res_lvl].find('ome:Pixels', namespaces)
    
    pixel_size_x = float(pixels.get('PhysicalSizeX'))
    pixel_size_y = float(pixels.get('PhysicalSizeY'))
    pixel_size_z = float(pixels.get('PhysicalSizeZ'))
    return pixel_size_x, pixel_size_y, pixel_size_z

def get_axes(zarr_path):
    ome_metadata_path = Path(zarr_path) / "OME" / "METADATA.ome.xml"
    if ome_metadata_path.exists():
        pixel_size_x, pixel_size_y, pixel_size_z = get_voxel_dims_from_XML(ome_metadata_path)
        axes = [
            dict(name='time', type='time', unit='second', scale=1.0),
            dict(name='channel', type='channel', scale=1.0),
            dict(name='z', type='space', unit='micrometer', scale=pixel_size_z),
            dict(name='y', type='space', unit='micrometer', scale=pixel_size_y),
            dict(name='x', type='space', unit='micrometer', scale=pixel_size_x),
        ]
    else:
        axes = None
    return axes


def main(zarr_path):
    
    root = zarr.open(zarr_path, mode='r')
    img = root['0']['0'] 

    axes = get_axes(zarr_path)

    T, _, Z, Y, X = img.shape
    for dataset in datasets.keys():
        dataset_zarr_path = os.path.join(output_dir, f"{dataset}.zarr")
        os.makedirs(dataset_zarr_path, exist_ok=True)
        print(f"Creating Zarr dataset for {dataset} at {dataset_zarr_path}")
        dataset_zarr = zarr.open(dataset_zarr_path, mode="a", shape=(T, 1, Z, Y, X), chunks=(1, 1, 1, Y, X))
        dataset_zarr.attrs["axes"] = axes
        for tp in tqdm(range(T)):
            frame = img[tp, datasets[dataset]]
            dataset_zarr[tp, 0] = frame

if __name__ == "__main__":
    zarr_path = "/groups/sgro/sgrolab/jennifer/cryolite/Natalie/260228InterfaceTransferMirrorB1-DB.zarr"
    output_dir = "/groups/sgro/sgrolab/jennifer/mhat/data/InterfaceTransferMirrorB1-DB"
    datasets = {"01_nuclei": 0, "01_cells": 1}
    main(zarr_path)