"""Small helpers shared across the pipeline scripts."""

import statistics
import xml.etree.ElementTree as et
from collections.abc import Mapping
from pathlib import Path

import numpy as np

# gitpython is imported inside get_experiment_metadata rather than here. That is
# its only user, and a module-level import would make gitpython a hard runtime
# dependency of every stage — this module also provides get_axes_metadata, which
# run_tracking.py imports.


# Spatial axis names in array order. A 2D movie uses the last two.
SPATIAL_NAMES = ("z", "y", "x")


def position_names(ndim: int) -> list[str]:
    """Per-axis position names for data of this rank: (z, y, x) or (y, x).

    This is the array-level answer to "how many spatial dimensions", used where
    only an array is in hand. ``spatial_axis_names`` is the zarr-metadata
    answer to the same question; the two always agree.
    """
    assert 2 <= ndim <= 3, f"only 2D and 3D data are supported, got ndim={ndim}"
    return list(SPATIAL_NAMES[-ndim:])


def _axis_field(axis, field, default=None):
    """Read a field from an axis, whether it is a dict or a geff ``Axis``.

    ``get_axes_metadata`` returns dicts (that is how they are stored in the
    zarr attrs) while ``geff.read`` returns pydantic objects, and the same
    questions get asked on both sides of the geff boundary.
    """
    if isinstance(axis, Mapping):
        return axis.get(field, default)
    return getattr(axis, field, default)


def spatial_axis_names(axes) -> list[str]:
    """The names of the space axes, in array order.

    ``["z", "y", "x"]`` for a 3D movie, ``["y", "x"]`` for a 2D one. This is
    the single place the pipeline asks a dataset how many spatial dimensions it
    has -- everything downstream (node position attributes, geff axis names,
    the funtracks name map) follows from it rather than assuming 3D.
    """
    names = [
        _axis_field(axis, "name") for axis in axes
        if _axis_field(axis, "type") == "space"
    ]
    if names:
        return names
    # Axes written without a "type" (none this pipeline produces, but a
    # hand-edited store might): everything that is not the time axis is spatial.
    return [
        _axis_field(axis, "name") for axis in axes
        if _axis_field(axis, "name") != "time"
    ]


def seg_chunks(spatial_shape, tile=512):
    """Chunk shape for a (t, *spatial) label array: one tile of one slice.

    Chunking a whole frame together would mean reading the entire frame
    (hundreds of MB on a large 3D dataset) to display a single slice, which
    defeats viewing a segmentation lazily. The two trailing axes are the ones
    displayed, so they are tiled and everything before them is chunked at 1.

    Used by ``run_tracking.py`` for ``pred_seg.zarr``, by
    ``create_seg_hypotheses.py`` for the stage-02 arrays, and by
    ``from_ctc_to_geff`` for ``correct_seg.zarr``, so the three stay consistent.
    """
    n = len(spatial_shape)
    return (1, *(
        min(tile, size) if i >= n - 2 else 1
        for i, size in enumerate(spatial_shape)
    ))


def default_axes(ndim: int) -> list[dict]:
    """Unit-scale time + space axes for data of this rank."""
    return [dict(name='time', type='time', unit='second', scale=1.0)] + [
        dict(name=name, type='space', unit='micrometer', scale=1.0)
        for name in position_names(ndim)
    ]


def get_axes_metadata(zarr_root):
    """Return the OME-style axes metadata for a zarr group or array.

    The "channel" axis is dropped, since the pipeline arrays written downstream
    are single-channel. If the zarr has no axes metadata, unit-scale time+space
    axes are returned as a default, with as many space axes as the array has
    (so a 2D movie does not get handed a phantom z).

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
        # Rank comes from the array rather than being assumed to be 4. A raw
        # zarr still carries its channel axis at this point, which the axes
        # above drop, so it contributes one extra dimension.
        ndim = getattr(zarr_root, "ndim", 4)
        spatial_ndim = ndim - 1
        if spatial_ndim > 3:
            spatial_ndim -= 1  # (t, c, z, y, x): drop the channel axis too
        axes = default_axes(spatial_ndim)
    return axes


def get_experiment_metadata():
    """Record the current git commit, for stamping experiment outputs.

    Requires gitpython, which is an optional dependency — install it with
    ``pip install gitpython`` if you need this.
    """
    try:
        import git
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "get_experiment_metadata requires gitpython, which is not installed. "
            "Install it with:\n    pip install gitpython"
        ) from exc

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
    """Return (x, y, z) voxel size in micrometers from an OME-XML file.

    ``z`` is None for a 2D acquisition, which omits PhysicalSizeZ. Callers
    decide what that means for them; ``get_axes_from_ome_xml`` below drops the
    z axis entirely rather than inventing a spacing for it.
    """
    pixels = _get_pixels_element(xml_file, res_lvl)

    pixel_size_x = float(pixels.get('PhysicalSizeX'))
    pixel_size_y = float(pixels.get('PhysicalSizeY'))
    pixel_size_z = pixels.get('PhysicalSizeZ')
    pixel_size_z = None if pixel_size_z is None else float(pixel_size_z)
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

    axes = [
        dict(name='time', type='time', unit='second', scale=time_increment),
        dict(name='channel', type='channel', scale=1.0),
    ]
    # A 2D acquisition has no PhysicalSizeZ. Emit four axes rather than a z axis
    # with a made-up spacing: the array it describes has no z either.
    if pixel_size_z is not None:
        axes.append(
            dict(name='z', type='space', unit='micrometer', scale=pixel_size_z)
        )
    axes.append(dict(name='y', type='space', unit='micrometer', scale=pixel_size_y))
    axes.append(dict(name='x', type='space', unit='micrometer', scale=pixel_size_x))
    return axes


def check_uint16_safe(frame, tp=None):
    """Guard the uint16 cast, which numpy performs silently.

    uint8/uint16 sources fit by construction and cost nothing to clear. Wider or
    signed integers, and floats, may still hold perfectly good 16-bit counts, so
    those are judged on their actual values rather than rejected outright.

    `tp` names the timepoint in the error message when checking frame by frame;
    leave it unset to check a whole array at once.
    """
    if np.can_cast(frame.dtype, np.uint16):
        return

    what = "Data" if tp is None else f"Frame {tp}"
    info = np.iinfo(np.uint16)
    fmin, fmax = float(frame.min()), float(frame.max())
    assert info.min <= fmin and fmax <= info.max, (
        f"{what} has values [{fmin}, {fmax}], outside the uint16 range "
        f"[{info.min}, {info.max}] -- the cast would wrap. Rescale the source first."
    )
    assert not np.issubdtype(frame.dtype, np.floating) or np.array_equal(
        frame, np.rint(frame)
    ), (
        f"{what} is {frame.dtype} holding non-integer values in "
        f"[{fmin}, {fmax}] -- the cast would truncate them (to all zeros, if this "
        "is normalized data). Rescale to integer counts first."
    )