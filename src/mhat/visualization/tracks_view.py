"""Load a raw movie and one set of tracks into a napari viewer.

Shared by ``scripts/04_tracking/visualize_results.py`` (interactive viewing) and
``scripts/04_tracking/render_tracks_video.py`` (video rendering), so the two
show the same thing. The tracks go through motile_tracker's ``TracksViewer``,
which displays one set of tracks at a time -- a run's predictions *or* a ground
truth, never both. Either is named by one path (see ``resolve_tracks``).
"""

from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import geff
import napari
import zarr
from funtracks.import_export import import_from_geff
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer

from mhat.dataset import SOLUTION_STAGES, Dataset, absolute_path
from mhat.evaluation.evaluate_tracking import (
    read_name_map_and_scale,
    remap_seg_to_track_ids_lazy,
)
from mhat.utils import get_axes_metadata

PRED_TRACKS, PRED_SEG = "pred_tracks.zarr", "pred_seg.zarr"
GT_TRACKS, GT_SEG = "correct_tracks.zarr", "correct_seg.zarr"


@dataclass(frozen=True)
class TracksSource:
    """One set of tracks to show, and where its raw movie comes from.

    Attributes:
        tracks_path: The geff tracks store.
        seg_path: Its segmentation, or None if there is none.
        home: The folder the tracks belong to: the run directory for
            predictions, the ground truth's folder for a ground truth.
        dataset: The output directory holding them, or None for a ground
            truth that sits outside every output directory.
        is_gt: Whether these are ground-truth tracks.
    """

    tracks_path: Path
    seg_path: Path | None
    home: Path
    dataset: Dataset | None
    is_gt: bool

    @property
    def name(self) -> str:
        """Label for the tracks in the viewer."""
        if not self.is_gt:
            return self.dataset.name
        prefix = f"{self.dataset.name} " if self.dataset is not None else ""
        return f"{prefix}GT {self.home.name}"


def resolve_tracks(path: str | Path, seg_path: str | Path | None = None):
    """What to show, from one path.

    ``path`` is either a tracking or ssvm run (its directory or
    ``config.toml``), whose predictions are shown, or a ground truth: a geff
    tracks store, or a folder holding ``correct_tracks.zarr`` (e.g.
    ``<output_dir>/gt/ctc``). The output directory, and with it the raw movie
    in ``raw.toml``, is found by walking up from there, so a ground truth needs
    no tracking run beside it.

    The segmentation defaults to the store beside the tracks (``pred_seg.zarr``
    or ``correct_seg.zarr``); ``seg_path`` overrides it. A default that does not
    exist is None.

    Raises:
        ValueError: If ``path`` is neither a run nor a ground truth.
    """
    path = absolute_path(path)
    if path.name in ("config.toml", PRED_TRACKS):
        path = path.parent  # the run itself
    if (path / PRED_TRACKS).exists():
        # A run: also checks it sits in a tracking or ssvm stage directory.
        dataset = Dataset.from_run_dir(path, SOLUTION_STAGES)
        home, is_gt = path, False
        tracks_path, default_seg = home / PRED_TRACKS, home / PRED_SEG
    elif (path / GT_TRACKS).exists() or (path.suffix == ".zarr" and path.exists()):
        home = path if (path / GT_TRACKS).exists() else path.parent
        tracks_path = home / GT_TRACKS if home == path else path
        default_seg, is_gt = home / GT_SEG, True
        dataset = Dataset.containing(home)
    else:
        raise ValueError(
            f"{path} is neither a tracking run (no {PRED_TRACKS}) nor a ground "
            f"truth (a tracks store, or a folder holding {GT_TRACKS})"
        )

    if seg_path is not None:
        seg = absolute_path(seg_path)
        if not seg.exists():
            raise FileNotFoundError(f"Segmentation {seg} does not exist")
    else:
        seg = default_seg if default_seg.exists() else None
    return TracksSource(tracks_path, seg, home, dataset, is_gt)


def _geff_axes(tracks_path: Path) -> list[dict]:
    """The geff store's axes as dicts shaped like ``get_axes_metadata``'s."""
    axes = geff.GeffMetadata.read(tracks_path).axes
    _, scale = read_name_map_and_scale(tracks_path)
    if axes is None:
        names, types = ["time", "z", "y", "x"], ["time", "space", "space", "space"]
        return [
            {"name": n, "type": t, "unit": None, "scale": s}
            for n, t, s in zip(names, types, scale, strict=True)
        ]
    return [
        {"name": a.name, "type": a.type, "unit": a.unit, "scale": s}
        for a, s in zip(axes, scale, strict=True)
    ]


def load_raw(
    viewer: napari.Viewer,
    raw_path: Path | None,
    tracks_path: Path,
    channel: int = 0,
    compute: bool = False,
) -> tuple[napari.layers.Image | None, list[float], list[dict]]:
    """Add one channel of the raw movie as an image layer.

    Without a raw movie, nothing is added and the scale comes from the tracks,
    which carry the same axes.

    Returns:
        (layer or None, display scale, axes). The display scale has its time
        entry set to 1.0; ``axes`` keeps the real time scale and units.
    """
    if raw_path is not None and Path(raw_path).exists():
        axes = get_axes_metadata(zarr.open(raw_path, mode="r"))
        scale = [axis["scale"] for axis in axes]
        scale[0] = 1.0  # Set time axis scale to 1.0 for visualization
        raw = da.from_zarr(raw_path)[:, channel, ...]
        print(f"Raw shape: {raw.shape}, dtype: {raw.dtype}, scale: {scale}")
        if compute:
            raw = raw.compute()
        layer = viewer.add_image(
            raw, name="raw", colormap="gray", blending="additive", scale=scale
        )
        return layer, scale, axes

    axes = _geff_axes(tracks_path)
    scale = [axis["scale"] for axis in axes]
    scale[0] = 1.0
    print(f"Raw data not found at {raw_path}; pass --raw to show it.")
    print(f"Using the tracks' own scale: {scale}")
    return None, scale, axes


def load_tracks(
    viewer: napari.Viewer,
    tracks_path: Path,
    seg_path: Path | None,
    scale: list[float],
    name: str,
    compute: bool = False,
):
    """Add a geff tracks store to the viewer through motile_tracker, plus its
    segmentation relabeled by track id.

    Returns:
        (tracks_viewer, tracks, seg_layer or None).
    """
    print(f"Loading tracks from {tracks_path}")
    # Built from the store's own axes rather than hardcoded to 3D --
    # funtracks infers dimensionality from this map, so a 3D map on a
    # 2D file silently declares an axis that is not there. See
    # read_name_map_and_scale for the whole story.
    name_map, _ = read_name_map_and_scale(tracks_path)
    node_name_map = {
        **name_map,
        "id": "track_id",  # track_id stays constant across frames
    }
    tracks = import_from_geff(
        tracks_path,
        node_name_map=node_name_map,
        segmentation_path=seg_path if compute else None,
        scale=scale,
    )

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.tracks_list.add_tracks(tracks, name)
    print(f"Successfully loaded tracks: {tracks}")

    # Load segmentation outside of track import to avoid memory issues.
    # The store is labeled by node id, so relabel to track_id to color
    # each track consistently over time, as the compute=True path does.
    # This stays lazy: only the slice on screen is read.
    seg_layer = tracks_viewer.tracking_layers.seg_layer if compute else None
    if not compute and seg_path is not None:
        track_seg = remap_seg_to_track_ids_lazy(tracks.graph, seg_path)
        print(f"Successfully loaded segmentation for tracks from {seg_path}")
        seg_layer = viewer.add_labels(
            track_seg, name="Track Segmentation", opacity=0.5, scale=scale
        )
    return tracks_viewer, tracks, seg_layer
