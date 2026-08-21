"""Small helpers shared across the pipeline scripts."""

import statistics
import xml.etree.ElementTree as et
from pathlib import Path

import git


def get_axes_metadata(zarr_root):
    """Return the OME-style axes metadata for a zarr group or array.

    The "channel" axis is dropped, since the pipeline arrays written downstream
    are single-channel. If the zarr has no axes metadata, unit-scale time/z/y/x
    axes are returned as a default.

    Every returned axis is guaranteed to carry a float "scale", so callers can
    build a scale list without length surprises: a missing or null scale means
    "unknown" and becomes 1.0, matching the default axes above. The axis dicts
    are copies, so coercing them does not touch the zarr's cached attributes.
    """
    axes = zarr_root.attrs.get("axes", None)
    if axes is not None:
        axes = [dict(axis) for axis in axes if axis.get("name") != "channel"]
        for axis in axes:
            scale = axis.get("scale")
            axis["scale"] = 1.0 if scale is None else float(scale)
    else:
        # Default axes metadata
        axes = [
            dict(name='time', type='time', unit='second', scale=1.0),
            dict(name='z', type='space', unit='micrometer', scale=1.0),
            dict(name='y', type='space', unit='micrometer', scale=1.0),
            dict(name='x', type='space', unit='micrometer', scale=1.0),
        ]
    return axes


def get_experiment_metadata():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return {"git_hash": sha}


OME_NAMESPACES = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

# OME time units, expressed in seconds.
TIME_UNIT_SECONDS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "µs": 1e-6,
    "us": 1e-6,
    "microsecond": 1e-6,
    "min": 60.0,
    "minute": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
}


def _get_pixels_element(xml_file, res_lvl=0):
    XML_tree = et.parse(xml_file)
    XML_root = XML_tree.getroot()

    images = XML_root.findall('ome:Image', OME_NAMESPACES)
    if res_lvl >= len(images):
        res_lvl = 0

    return images[res_lvl].find('ome:Pixels', OME_NAMESPACES)


def get_voxel_dims_from_XML(xml_file, res_lvl=0):
    """Return (x, y, z) voxel size in micrometers from an OME-XML file."""
    pixels = _get_pixels_element(xml_file, res_lvl)

    pixel_size_x = float(pixels.get('PhysicalSizeX'))
    pixel_size_y = float(pixels.get('PhysicalSizeY'))
    pixel_size_z = float(pixels.get('PhysicalSizeZ'))
    return pixel_size_x, pixel_size_y, pixel_size_z


def get_time_increment_from_XML(xml_file, res_lvl=0):
    """Return the frame interval in seconds from an OME-XML file, or None.

    Prefers the Pixels/@TimeIncrement attribute. Reads the per-plane DeltaT 
    timestamps if the Pixels/@TimeIncrement attribute is missing. 
    Returns None if no time metadata is found.
    """
    pixels = _get_pixels_element(xml_file, res_lvl)

    increment = pixels.get('TimeIncrement')
    if increment is not None:
        unit = pixels.get('TimeIncrementUnit', 's')
        return float(increment) * TIME_UNIT_SECONDS.get(unit, 1.0)

    # No TimeIncrement: derive the interval from the plane timestamps.
    delta_t_by_timepoint = {}
    for plane in pixels.findall('ome:Plane', OME_NAMESPACES):
        delta_t, the_t = plane.get('DeltaT'), plane.get('TheT')
        if delta_t is None or the_t is None:
            continue
        seconds = float(delta_t) * TIME_UNIT_SECONDS.get(plane.get('DeltaTUnit', 's'), 1.0)
        # A timepoint spans many z/channel planes; the first one starts it.
        the_t = int(the_t)
        if the_t not in delta_t_by_timepoint or seconds < delta_t_by_timepoint[the_t]:
            delta_t_by_timepoint[the_t] = seconds

    timepoints = sorted(delta_t_by_timepoint)
    if len(timepoints) < 2:
        return None

    gaps = [
        delta_t_by_timepoint[later] - delta_t_by_timepoint[earlier]
        for earlier, later in zip(timepoints, timepoints[1:])
    ]
    return statistics.median(gaps)


def get_axes_from_ome_xml(zarr_path, res_lvl=0):
    """Build axes metadata for a raw OME-zarr from its OME-XML sidecar.

    Returns None if the zarr has no OME metadata to read. Unlike
    get_axes_metadata, this reads the acquisition metadata rather than axes
    previously written by this pipeline, so it is what raw data conversion
    should use. The channel axis is kept here, since the array being described
    still has one.
    """
    ome_metadata_path = Path(zarr_path) / "OME" / "METADATA.ome.xml"
    if not ome_metadata_path.exists():
        return None

    pixel_size_x, pixel_size_y, pixel_size_z = get_voxel_dims_from_XML(ome_metadata_path, res_lvl)

    time_increment = get_time_increment_from_XML(ome_metadata_path, res_lvl)
    if time_increment is None:
        print(f"No time metadata in {ome_metadata_path}, defaulting time scale to 1.0")
        time_increment = 1.0

    return [
        dict(name='time', type='time', unit='second', scale=time_increment),
        dict(name='channel', type='channel', scale=1.0),
        dict(name='z', type='space', unit='micrometer', scale=pixel_size_z),
        dict(name='y', type='space', unit='micrometer', scale=pixel_size_y),
        dict(name='x', type='space', unit='micrometer', scale=pixel_size_x),
    ]
