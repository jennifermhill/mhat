import argparse
import csv
from pathlib import Path

import numpy as np
import toml
import waterz # type: ignore
import zarr
import torch
import geff
import skimage
import networkx as nx

from mhat.segmentation.threshold_labeling import voronoi_otsu_labeling, voronoi_mean_labeling, voronoi_li_labeling
from mhat.segmentation.cellpose import segment_with_cellpose
from mhat.segmentation.threshold_labeling import segment_cells_from_nuclei_frame
from mhat.tracking import utils

def get_axes_metadata(zarr_root):
    axes = zarr_root.attrs.get("axes", None)
    if axes is not None:
        axes = [axis for axis in axes if axis.get("name") != "channel"]
    else:
        # Default axes metadata
        axes = [
            dict(name='time', type='time', unit='second', scale=1.0),
            dict(name='z', type='space', unit='micrometer', scale=1.0),
            dict(name='y', type='space', unit='micrometer', scale=1.0),
            dict(name='x', type='space', unit='micrometer', scale=1.0),
        ]
    return axes

def get_nuclei_segmentation(data_zarr: Path, output_root, config):
    zarr_root = zarr.open(data_zarr, "r")
    axes = get_axes_metadata(zarr_root)

    raw_data = zarr_root

    T, C, Z, Y, X = raw_data.shape

    output_root.create_dataset(
        "nuclei", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype=np.uint64, overwrite=True
    )
    output_root['nuclei'].attrs["axes"] = axes

    seg_method = config["seg_method"]
    if seg_method == 'cellpose':
        # Check for cuda availability
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU for Cellpose.")
            gpu = True
        else:
            print("CUDA is not available. Using CPU for Cellpose.")
            gpu = False
        cellpose_kwargs = config.get("cellpose_params", {})

    max_node_id = 0
    for tp in range(T):
        print(f"Processing frame {tp}")
        frame = raw_data[tp, 0]
        if seg_method == 'cellpose':
            labels = segment_with_cellpose(frame, gpu=gpu, **cellpose_kwargs)
        elif seg_method == 'mean':
            labels = voronoi_mean_labeling(frame, spot_sigma=config["spot_sigma"], outline_sigma=config["outline_sigma"])
        elif seg_method == 'li':
            labels = voronoi_li_labeling(frame, spot_sigma=config["spot_sigma"], outline_sigma=config["outline_sigma"])
        else:
            labels = voronoi_otsu_labeling(frame, spot_sigma=config["spot_sigma"], outline_sigma=config["outline_sigma"])

        # Offset labels so node ids are globally unique across timepoints.
        # Keep max_node_id monotonic so an empty frame can't reset the offset
        # and cause id collisions with an earlier frame.
        labels[labels != 0] += max_node_id
        max_node_id = max(max_node_id, int(labels.max()))

        output_root['nuclei'][tp] = labels

def get_cell_segmentation(cells_data_zarr: Path, output_root):
    """Segment cells from the nuclei seeds, writing one frame at a time to disk.

    Reads the raw cell channel and the cached nuclei labels per timepoint and
    writes the result into ``output_root['cells']`` (labels match the nuclei
    labels), so the full cell volume is never held in memory.
    """
    cells_zarr = zarr.open(cells_data_zarr, "r")
    axes = get_axes_metadata(cells_zarr)
    T, C, Z, Y, X = cells_zarr.shape

    nuclei_seg = output_root["nuclei"]

    output_root.create_dataset(
        "cells", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype=np.uint64, overwrite=True
    )
    output_root["cells"].attrs["axes"] = axes

    for tp in range(T):
        print(f"Segmenting cells for frame {tp}")
        cell_img = cells_zarr[tp, 0]
        nuclei_labels_img = nuclei_seg[tp]
        output_root["cells"][tp] = segment_cells_from_nuclei_frame(cell_img, nuclei_labels_img)


def nodes_from_segmentation(
    segmentation: np.ndarray,
    raw_img: np.ndarray | None = None,
    size_threshold: int | None = None,
    tp: int = 0,
    scale: list[float] = [1.0, 1.0, 1.0, 1.0]
) -> nx.DiGraph:
    """Extract candidate nodes from a segmentation.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (z, y, x).

        size_threshold (int): A minimum area for candidate nodes. Nodes smaller
            than this area will not be added to the graph.

        tp (int, optional): The timepoint to assign to the nodes. Defaults to 0.

        scale (list[float], optional): The scaling factors for each axis of
            the fragments array. Defaults to [1.0, 1.0, 1.0, 1.0].

    Returns:
        nx.DiGraph: A candidate graph with only nodes.
    """
    cand_graph = nx.DiGraph()
    # for t in range(len(segmentation)):
    #     seg_frame = segmentation[t]
    props = skimage.measure.regionprops(segmentation)
    for regionprop in props:
        if size_threshold and regionprop.area < size_threshold:
            continue
        node_id = int(regionprop.label)
        region = segmentation == node_id
        centroid = (float(regionprop.centroid[0] * scale[1]),
                    float(regionprop.centroid[1] * scale[2]),
                    float(regionprop.centroid[2] * scale[3]))
        region_raw = raw_img[region]
        intensity = np.mean(region_raw)
        attrs = {
            "time": int(tp),
            "x": centroid[2],
            "y": centroid[1],
            "z": centroid[0],
            "centroid": centroid,
            "label": node_id,
            "area": regionprop.area,
            "intensity": intensity,
        }
        cand_graph.add_node(node_id, **attrs)

    return cand_graph

def generate_graph(data_zarr: Path, seg_root):
    raw_root = zarr.open(data_zarr, "r")
    for tp in range(seg_root["nuclei"].shape[0]):
        print(f"Processing timepoint {tp}")
        raw_img = raw_root[tp, 0]
        nuclei_seg = seg_root["nuclei"][tp]

        cand_graph = nodes_from_segmentation(
            nuclei_seg, raw_img=raw_img, tp=tp, scale=[1.0, 1.0, 1.0, 1.0]
        )

        if tp == 0:
            all_cand_graph = cand_graph
        else:
            all_cand_graph = nx.compose(all_cand_graph, cand_graph)

    return all_cand_graph

def save_graph(data_dir: Path, cand_graph: nx.DiGraph, output_store):
    axes = get_axes_metadata(zarr.open(data_dir, "r"))
    if axes is not None:
        for axis in axes:
            if axis["scale"] is None:
                axis["scale"] = 1.0
            else:
                axis["scale"] = float(axis["scale"])
        scale = [axis["scale"] for axis in axes if "scale" in axis]
    else:
        scale = [1.0, 1.0, 1.0, 1.0]

    # Save tracks to geff file format
    metadata = geff.GeffMetadata(directed=True, 
                                 related_objects=[{
                                     "type": "labels", 
                                     "path": "../seg.zarr/nuclei",
                                     "label_prop": "label"
                                 }],
                                 node_props_metadata={},
                                 edge_props_metadata={},
                                 )
    geff.write(cand_graph,
               output_store,
               axis_names=["time", "z", "y", "x"],
               axis_types=["time", "space", "space", "space"],
               axis_scales=scale,
               metadata=metadata,
               overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    experiment: str = config["experiment"]
    dataset: str = config["dataset"]
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    nuclei_data_dir = input_base_dir / experiment / f"nuclei.zarr"
    print(f"Loading data from {nuclei_data_dir}")
    assert nuclei_data_dir.is_dir()

    cells_data_dir = input_base_dir / experiment / f"cells.zarr"
    print(f"Loading data from {cells_data_dir}")
    assert cells_data_dir.is_dir()

    output_dir = output_base_dir / experiment / "jingyi_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "config.toml", "w") as config_file:
        toml.dump(config, config_file)

    seg_root = zarr.open(output_dir / "seg.zarr", "a")
    if "nuclei" not in seg_root or config["overwrite"]:
        get_nuclei_segmentation(nuclei_data_dir, seg_root, config["seg_params"])

    if "cells" not in seg_root or config["overwrite"]:
        get_cell_segmentation(cells_data_dir, seg_root)

    cand_graph = generate_graph(nuclei_data_dir, seg_root)

    raw_cells = zarr.open(cells_data_dir, "r")
    utils.add_camp_signal_attr(cand_graph, raw_cells, seg_root["cells"], channel=0)

    save_graph(nuclei_data_dir, cand_graph, output_dir / "graph.zarr")

