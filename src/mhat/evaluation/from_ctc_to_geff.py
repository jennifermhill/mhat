import geff
import os
import shutil
from pathlib import Path
from typing import Literal

import tifffile
from skimage.measure import regionprops

import warnings

import numpy as np
import zarr
from zarr.storage import StoreLike

import geff
from geff.core_io import write_arrays
from geff_spec import Axis, GeffMetadata, RelatedObject

from mhat.utils import seg_chunks

# Node ids run to one per object per frame, which outgrows the uint16 the CTC
# marker tifs use; uint32 also matches the dtype run_tracking writes.
SEG_DTYPE = np.uint32


def from_ctc_to_geff(
    ctc_path: Path,
    geff_path: Path,
    segmentation_store: StoreLike | None = None,
    axes: list[Axis] | None = None,
    tczyx: bool = False,
    overwrite: bool = False,
    zarr_format: Literal[2, 3] = 2,
) -> None:
    """
    Convert a CTC file to a GEFF file.
    Adapted from geff but modified to save with zarr extension, use "time" as frame key,
    and track_id as the track node property.

    The exported segmentation is labelled by **graph node id**, matching
    ``pred_seg.zarr`` from ``run_tracking.py``. That is the invariant funtracks
    relies on throughout (``Tracks.get_pixels`` resolves a node's pixels with
    ``segmentation[time] == node``), so ground truth and predictions load and
    remap through exactly the same path. The CTC tracklet id is kept separately
    as the ``track_id`` node property, and ``label`` mirrors the node id so
    ``related_objects`` has a property to point at.

    Args:
        ctc_path: The path to the CTC file.
        geff_path: The path to the GEFF file.
        segmentation_store: The path or store to save the segmentation to.
                            If not provided, it won't be exported.
        axes: The axes for the GEFF file. If not provided, it will be inferred from the data 
              (assuming "time" is the first axis and "y", "x" are the spatial axes).
        tczyx: Expand data to make it (T, C, Z, Y, X) otherwise it's (T,) + Frame shape.
        overwrite: Whether to overwrite the GEFF file if it already exists.
        zarr_format (Literal[2, 3]): The zarr specification to use when writing the zarr.
            Defaults to 2.
    """
    ctc_path = Path(ctc_path)
    geff_path = Path(geff_path).with_suffix(".zarr")

    if not ctc_path.exists():
        raise FileNotFoundError(f"CTC file {ctc_path} does not exist")

    for tracks_file in ["man_track.txt", "res_track.txt"]:
        tracks_file_path = ctc_path / tracks_file
        if tracks_file_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"Tracks file {ctc_path}/man_track.txt or {ctc_path}/res_track.txt does not exist"
        )

    if geff_path.exists() and not overwrite:
        raise FileExistsError(f"GEFF file {geff_path} already exists")

    if geff_path.exists() and overwrite:
        shutil.rmtree(geff_path)

    tracks: dict[int, list[int]] = {}

    edges = []
    node_props: dict[str, list[int | float]] = {
        "id": [],
        "label": [],
        "track_id": [],
        "time": [],
        "x": [],
        "y": [],
    }

    segm_array = None
    node_id = 1

    sorted_files = sorted(ctc_path.glob("*.tif"))
    n_1_padding: tuple[int, ...] = ()
    expand_dims: tuple[None, ...] | slice = slice(None)

    for t, filepath in enumerate(sorted_files):
        frame = tifffile.imread(filepath)

        # A 3D frame needs somewhere to put its z centroids. This is keyed off
        # the frame's rank alone: it used to sit inside the segmentation-store
        # branch below, so converting 3D markers without exporting a
        # segmentation raised a KeyError from the centroid loop.
        if t == 0 and frame.ndim == 3:
            node_props["z"] = []

        if segmentation_store is not None and segm_array is None:
            # created in first iteration
            if tczyx:
                n_1_padding = (1,) * (5 - frame.ndim - 1)  # forcing data to be (T, C, Z, Y, X)
                expand_dims = (np.newaxis,) * len(n_1_padding)

            # Warning is triggered when the zarr_format argument is ignored by zarr 2
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*ignoring keyword argument.*zarr_format.*",
                    category=UserWarning,
                )
                segm_array = zarr.open_array(
                    segmentation_store,
                    shape=(len(sorted_files), *n_1_padding, *frame.shape),
                    chunks=seg_chunks((*n_1_padding, *frame.shape)),
                    dtype=SEG_DTYPE,
                    mode="w" if overwrite else "w-",
                    zarr_format=zarr_format,
                )

        # Maps this frame's CTC tracklet ids onto the node ids assigned below.
        # Entries left at 0 stay background.
        frame_lut = (
            np.zeros(int(frame.max()) + 1, dtype=SEG_DTYPE)
            if segm_array is not None
            else None
        )

        for obj in regionprops(frame):
            tracklet_id = obj.label # Real seg_ids and tracklet_ids from man_track.txt
            node_props["id"].append(node_id)
            node_props["label"].append(node_id)
            node_props["track_id"].append(tracklet_id)
            node_props["time"].append(t)
            # using y,x for 2d and z,y,x for 3d
            for c, v in zip(("x", "y", "z"), obj.centroid[::-1], strict=False):
                node_props[c].append(v)

            if tracklet_id not in tracks:
                tracks[tracklet_id] = []

            tracks[tracklet_id].append(node_id)
            if frame_lut is not None:
                frame_lut[tracklet_id] = node_id
            node_id += 1

        if segm_array is not None:
            segm_array[t] = frame_lut[frame][expand_dims]

    if len(node_props["id"]) == 0:
        raise ValueError(f"No nodes found in the CTC directory {ctc_path}")

    for _node_ids in tracks.values():
        # connect simple-paths of each track
        for i in range(len(_node_ids) - 1):
            # forward in time (parent -> child)
            edges.append((_node_ids[i], _node_ids[i + 1]))

    tracks_table = np.loadtxt(tracks_file_path, dtype=int)

    # removing orphan tracklets
    tracks_table = tracks_table[tracks_table[:, -1] > 0]

    for row in tracks_table:
        child_track_id = row[0]
        parent_track_id = row[-1]
        child_node_id = tracks[child_track_id][0]
        parent_node_id = tracks[parent_track_id][-1]
        # forward in time (parent -> child)
        edges.append((parent_node_id, child_node_id))

    if axes is None:
        axis_names = [
            Axis(name="time", type="time"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ]
        scale = {"time": 1.0, "y": 1.0, "x": 1.0}
        if "z" in node_props:
            axis_names.insert(1, Axis(name="z", type="space"))
            scale["z"] = 1.0
    else:
        axis_names = axes
        scale = {a.name: a.scale for a in axes if a.scale is not None}

    for coord in ("x", "y", "z"):
        if coord in node_props and coord in scale:
            node_props[coord] = [
                v * scale[coord] for v in node_props[coord]
            ]

    node_ids = np.asarray(node_props.pop("id"), dtype="uint")

    rel_objs = None
    if segmentation_store is not None:
        # Record related object metadata for segmentation
        seg_path = None
        if isinstance(segmentation_store, Path) or isinstance(segmentation_store, str):
            seg_path = segmentation_store
        else:
            try:
                if zarr.__version__.startswith("2"):
                    seg_path = segmentation_store.path  # type: ignore
                else:
                    seg_path = segmentation_store.root  # type: ignore
            except AttributeError:
                warnings.warn(
                    "Cannot determine path to segmentation_store for related objects metadata",
                    stacklevel=2,
                )

        if seg_path is not None:
            rel_path = os.path.relpath(seg_path, geff_path)
            rel_objs = [RelatedObject(type="labels", path=rel_path, node_prop="label")]
    write_arrays(
        geff_store=geff_path,
        node_ids=node_ids,
        node_props={
            name: {"values": np.asarray(values), "missing": None}
            for name, values in node_props.items()
        },
        edge_ids=np.asarray(edges, dtype=node_ids.dtype),
        edge_props={},
        metadata=GeffMetadata(
            geff_version=geff.__version__,
            axes=axis_names,
            directed=True,
            node_props_metadata={},
            edge_props_metadata={},
            track_node_props={"tracklet": "track_id"},
            related_objects=rel_objs,
        ),
        zarr_format=zarr_format,
    )