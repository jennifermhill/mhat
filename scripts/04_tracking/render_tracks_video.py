"""Render a video of one set of tracks over its raw movie.

The tracks are named by one path: a tracking or ssvm run, whose predictions
are shown, or a ground truth (e.g. ``<output_dir>/gt/ctc`` or its
``correct_tracks.zarr``). The raw movie comes from the ``raw.toml`` of the
output directory holding them, so a ground truth needs no run beside it. They
are loaded exactly as ``visualize_results.py`` loads them (through
motile_tracker's ``TracksViewer``, with the segmentation relabeled by track id
so each cell is colored like its tail), and the video steps through time in
one of two modes:

- ``3d``: 3D view, raw rendered as an attenuated MIP (3D data only)
- ``2d``: 2D view, at a fixed ``z`` for 3D data; the only mode for 2D data

TracksViewer shows one set of tracks at a time, so predictions and GT are two
separate videos.

To show the raw movie alone (with the timestamp and scale bar), name an output
directory (or its ``raw.toml``) instead of tracks, or give only ``--raw``.

More raw movies (another zarr, or another channel of the same one) are
overlaid on the first with additive blending through ``[[extra_raw]]`` tables
in the config. Each must have the raw movie's shape and scale, and is displayed
by its own settings; with any extra, the first raw defaults to green and the
extras to magenta, cyan, yellow.

Frames are captured one by one with ``viewer.screenshot()`` rather than with
napari-animation, which deep-copies the viewer state to interpolate between
keyframes and crashes on TracksViewer's backing objects. Each frame is written
to a PNG as soon as it is captured, so an interrupted render resumes at the
first missing frame. The frames directory records the display settings that
produced it, and a rerun with different settings refuses to reuse its frames
(``--overwrite`` clears that directory and nothing else).

Videos go next to the tracks they show: ``<run>/videos/`` for predictions,
``<gt folder>/videos/`` (e.g. ``<output_dir>/gt/ctc/videos/``) for a ground
truth, ``<output_dir>/videos/raw_*.mp4`` for the raw movie alone. A raw movie
given only by ``--raw`` belongs to no output directory, so it needs ``-o``.

Usage:
    python scripts/04_tracking/render_tracks_video.py <tracks> --setup
    python scripts/04_tracking/render_tracks_video.py --config render.toml
    python scripts/04_tracking/render_tracks_video.py <tracks> --end 3 -o test.mp4
    python scripts/04_tracking/render_tracks_video.py <output_dir> --raw <zarr>
    python scripts/04_tracking/render_tracks_video.py --raw <zarr> -o raw.mp4
    python scripts/04_tracking/render_tracks_video.py <tracks> --extra-raw @1 --setup

``--setup`` opens the loaded viewer without rendering. Adjust the view by hand,
then press Shift-S (or close the window) to print the display settings as TOML,
ready to paste into the config.

Example TOML config (keep these under scratch_configs/visualization/):

    tracks = "Y:/jennifer/mhat/experiments/<exp>/<dataset>/tracking/<uid>"
    # tracks = "Y:/jennifer/mhat/experiments/<exp>/<dataset>/gt/ctc"  # a GT
    # tracks = "Y:/jennifer/mhat/experiments/<exp>/<dataset>"  # raw alone
    raw_path = "Y:/jennifer/mhat/data/<dataset>.zarr"  # if raw.toml's path is
                                                        # not valid here (--raw)
    # seg_path = "..."       # default: pred_seg.zarr / correct_seg.zarr beside
                             # the tracks (--seg)
    mode = "3d"              # "3d", "2d" or "auto" (3d for 3D data)
    # z = 10                 # slice for mode = "2d" on 3D data
    start = 0                # frame range [start, end); default whole movie
    # end = 30
    fps = 4
    quality = 9              # 0-10, passed to imageio's ffmpeg writer
    screenshot_scale = 2.0
    canvas_size = [1000, 1200]   # [height, width]
    tail_length = 1000       # frames; default the whole movie
    tail_width = 2.0
    show_tracks = true       # false: seg colored by track id, no tails
    show_points = false
    scale_bar = true
    # scale_bar_length = 50.0  # fixed length in the spatial unit (e.g. um);
                               # default: napari picks one from the zoom
    # time_unit = "min"      # force one unit ("s", "min" or "h"); default fits
                             # the movie, e.g. 7hr:15min for 30 x 15 min
    timestamp_font_size = 18     # as in the setup viewer; the render scales
    scale_bar_font_size = 20     # both by screenshot_scale to keep proportions

    [raw]
    contrast_limits = [0, 255]   # default: p1/p99.5 of a few sampled frames
    colormap = "gray"
    gamma = 1.0
    rendering = "attenuated_mip"  # 3d mode only
    attenuation = 0.05

    [[extra_raw]]                # optional, repeatable: overlaid on the raw
    path = "Y:/jennifer/mhat/data/<dataset>_ch2.zarr"
    channel = 0
    colormap = "magenta"         # the [raw] keys, same defaults
    contrast_limits = [0, 255]

    [seg]
    visible = true
    opacity = 0.5
    contour = 0              # 2d mode only

    [camera]
    angles = [0.0, 0.0, 90.0]
    zoom = 1.0
    center = [0.0, 100.0, 100.0]
    perspective = 0.0
"""

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import imageio.v3 as iio
import napari
import numpy as np
import toml
from mhat.dataset import RAW_RECORD, Dataset, absolute_path
from mhat.visualization.timestamps import time_formatter
from mhat.visualization.tracks_view import (
    load_raw,
    load_tracks,
    resolve_tracks,
)
from mhat.visualization.video_frames import (
    even_crop,
    prepare_frames_dir,
    write_video,
)
from tqdm import tqdm

RAW_KEYS = {"contrast_limits", "colormap", "gamma", "rendering", "attenuation"}
EXTRA_RAW_KEYS = RAW_KEYS | {"path", "channel"}
EXTRA_RAW_COLORMAPS = ("magenta", "cyan", "yellow")
SEG_KEYS = {"visible", "opacity", "contour", "rendering"}
CAMERA_KEYS = {"angles", "zoom", "center", "perspective"}

# Settings that change what a frame looks like. A frames directory rendered
# with other values for any of these is not reused; the frame range, fps,
# quality and output path only affect which frames are assembled, and how.
FRAME_SETTINGS = (
    "tracks",
    "raw_path",
    "seg_path",
    "mode",
    "z",
    "channel",
    "screenshot_scale",
    "canvas_size",
    "tail_length",
    "tail_width",
    "show_tracks",
    "show_points",
    "scale_bar",
    "scale_bar_length",
    "time_unit",
    "timestamp_font_size",
    "scale_bar_font_size",
    "raw_display",
    "extra_raw",
    "seg",
    "camera",
)


@dataclass
class RenderConfig:
    tracks: str | None = None
    raw_path: str | None = None
    seg_path: str | None = None
    output: str | None = None
    mode: str = "auto"
    z: int | None = None
    channel: int = 0
    start: int = 0
    end: int | None = None
    fps: int = 4
    quality: float = 9
    screenshot_scale: float = 2.0
    canvas_size: list[int] = field(default_factory=lambda: [1000, 1200])
    tail_length: int | None = None
    tail_width: float = 2.0
    show_tracks: bool = True
    show_points: bool = False
    scale_bar: bool = True
    scale_bar_length: float | None = None
    time_unit: str | None = None
    timestamp_font_size: float = 18
    scale_bar_font_size: float = 20
    # The [raw] table of the TOML: how the raw movie is displayed.
    raw_display: dict = field(default_factory=dict)
    # The [[extra_raw]] tables: more raw movies overlaid on the first.
    extra_raw: list[dict] = field(default_factory=list)
    seg: dict = field(default_factory=dict)
    camera: dict = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str | Path) -> "RenderConfig":
        data = toml.load(path)
        if "raw" in data:
            if not isinstance(data["raw"], dict):
                raise ValueError(
                    f"{path}: `raw` is the [raw] display table; the raw movie's "
                    f"path is `raw_path`"
                )
            data["raw_display"] = data.pop("raw")
        if "raw_display" in data and not isinstance(data["raw_display"], dict):
            raise ValueError(f"{path}: use a [raw] table, not raw_display")
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        cfg = cls(**data)
        cfg.check_tables(path)
        return cfg

    def check_tables(self, source) -> None:
        for table, allowed, name in (
            (self.raw_display, RAW_KEYS, "raw"),
            (self.seg, SEG_KEYS, "seg"),
            (self.camera, CAMERA_KEYS, "camera"),
        ):
            unknown = set(table) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown keys in [{name}] of {source}: {sorted(unknown)} "
                    f"(allowed: {sorted(allowed)})"
                )
        for i, extra in enumerate(self.extra_raw):
            unknown = set(extra) - EXTRA_RAW_KEYS
            if unknown:
                raise ValueError(
                    f"Unknown keys in [[extra_raw]] #{i + 1} of {source}: "
                    f"{sorted(unknown)} (allowed: {sorted(EXTRA_RAW_KEYS)})"
                )
            if "path" not in extra:
                raise ValueError(f"[[extra_raw]] #{i + 1} of {source} has no path")

    def override_from_args(self, args: argparse.Namespace) -> None:
        """Override config values with explicitly given CLI flags."""
        if args.raw is not None:
            self.raw_path = str(args.raw)
        if args.seg is not None:
            self.seg_path = str(args.seg)
        for spec in args.extra_raw or []:
            self.extra_raw.append(parse_extra_raw(spec))
        for name in (
            "tracks",
            "output",
            "mode",
            "z",
            "channel",
            "start",
            "end",
            "fps",
            "quality",
            "screenshot_scale",
            "tail_length",
            "time_unit",
            "scale_bar_length",
        ):
            value = getattr(args, name)
            if value is not None:
                setattr(self, name, str(value) if isinstance(value, Path) else value)

    def frame_settings(self) -> dict:
        settings = {k: v for k, v in asdict(self).items() if k in FRAME_SETTINGS}
        if not settings["extra_raw"]:
            # Absent, so frames rendered before extra_raw existed are reused.
            del settings["extra_raw"]
        return settings


def resolve_mode(cfg: RenderConfig, n_spatial: int) -> str:
    """Pick the view mode from the config and the data's rank."""
    mode = cfg.mode
    if mode == "auto":
        mode = "3d" if n_spatial == 3 else "2d"
    if mode not in ("3d", "2d"):
        raise ValueError(f"mode must be '3d', '2d' or 'auto', got {mode!r}")
    if mode == "3d" and n_spatial != 3:
        raise ValueError("mode = '3d' needs 3D data; this movie is 2D")
    if mode == "2d" and n_spatial == 3 and cfg.z is None:
        raise ValueError("mode = '2d' on 3D data needs a z slice (--z)")
    return mode


def auto_contrast_limits(layer, n_samples: int = 5) -> list[float]:
    """p1/p99.5 over a few evenly spaced frames, so every frame shares one
    contrast and the video doesn't flicker.
    """
    data = layer.data
    n_time = data.shape[0]
    frames = np.unique(np.linspace(0, n_time - 1, min(n_samples, n_time)).astype(int))
    values = np.concatenate([np.asarray(data[t]).ravel() for t in frames])
    low, high = np.percentile(values, [1, 99.5])
    if high <= low:
        high = low + 1
    return [float(low), float(high)]


@dataclass
class Scene:
    """The layers a render touches, and what the data looks like."""

    viewer: napari.Viewer
    raw_layer: object
    seg_layer: object
    tracks_layer: object
    points_layer: object
    axes: list[dict]
    n_spatial: int
    n_time: int
    raw_only: bool = False
    # One layer per [[extra_raw]] table, in the config's order.
    extra_raw_layers: list = field(default_factory=list)


def named_output_dir(path: str | Path) -> Dataset | None:
    """The output directory ``path`` names (it, or its ``raw.toml``), or None
    if ``path`` is something else, such as a run or a ground truth.
    """
    path = absolute_path(path)
    if path.name == RAW_RECORD:
        path = path.parent
    if not (path / RAW_RECORD).is_file():
        return None
    return Dataset(output_dir=path, raw_path=Dataset(path).recorded_raw_path())


def parse_extra_raw(spec: str) -> dict:
    """An ``--extra-raw`` value, ``PATH[@CHANNEL]``, as an [[extra_raw]] table.
    An empty PATH (``@1``) means the first raw movie's zarr, filled in at load.
    """
    path, sep, channel = spec.rpartition("@")
    if not sep or not channel.isdigit():
        path, channel = spec, None
    extra = {"path": path or None}
    if channel is not None:
        extra["channel"] = int(channel)
    return extra


def load_extra_raw(
    viewer: napari.Viewer,
    cfg: RenderConfig,
    raw_layer,
    axes: list[dict],
    raw_path,
) -> list:
    """Add the [[extra_raw]] movies, checked against the first raw movie so
    they overlay it voxel for voxel. An extra without a path is another
    channel of ``raw_path``, the first raw movie's zarr.
    """
    if not cfg.extra_raw:
        return []
    if raw_layer is None:
        raise ValueError("[[extra_raw]] needs the raw movie to overlay; pass --raw")
    layers = []
    for i, extra in enumerate(cfg.extra_raw):
        if extra.get("path") is None:
            extra["path"] = str(raw_path)  # setup prints it
        path = absolute_path(extra["path"])
        if not path.exists():
            raise FileNotFoundError(f"[[extra_raw]] #{i + 1}: {path} does not exist")
        layer, _, extra_axes = load_raw(viewer, path, None, extra.get("channel", 0))
        if layer.data.shape != raw_layer.data.shape:
            raise ValueError(
                f"[[extra_raw]] #{i + 1} ({path}) has shape {layer.data.shape}, "
                f"the raw movie {raw_layer.data.shape}"
            )
        scales = [a["scale"] for a in extra_axes[1:]]
        if not np.allclose(scales, [a["scale"] for a in axes[1:]]):
            raise ValueError(
                f"[[extra_raw]] #{i + 1} ({path}) has spatial scale {scales}, "
                f"the raw movie {[a['scale'] for a in axes[1:]]}"
            )
        layer.name = f"raw {i + 2}"
        layers.append(layer)
    return layers


def load_scene(cfg: RenderConfig) -> tuple[Scene, Path | None]:
    """Build the viewer: raw plus one set of tracks, or raw alone.

    Returns:
        (scene, the folder whose ``videos/`` the output goes in by default, or
        None for a raw movie given only by ``raw_path``).
    """
    if cfg.tracks is None and cfg.raw_path is None:
        raise ValueError(
            "Nothing to show: give tracks or an output directory (positional "
            "argument or `tracks` in the config), or a raw movie (--raw)"
        )
    dataset = named_output_dir(cfg.tracks) if cfg.tracks is not None else None
    if cfg.tracks is None or dataset is not None:
        return load_raw_scene(cfg, dataset)

    source = resolve_tracks(cfg.tracks, cfg.seg_path)
    raw_path = cfg.raw_path or (
        source.dataset.raw_path if source.dataset is not None else None
    )
    if raw_path is None:
        print(
            f"{source.tracks_path} is not inside an output directory, so there "
            f"is no raw.toml to find the raw movie in; pass --raw to show it."
        )

    viewer = napari.Viewer()
    raw_layer, scale, axes = load_raw(viewer, raw_path, source.tracks_path, cfg.channel)
    # Before the tracks, so the seg and tails draw on top of every channel.
    extra_raw_layers = load_extra_raw(viewer, cfg, raw_layer, axes, raw_path)
    tracks_viewer, tracks, seg_layer = load_tracks(
        viewer, source.tracks_path, source.seg_path, scale, source.name
    )

    if seg_layer is not None:
        n_time = seg_layer.data.shape[0]
    elif raw_layer is not None:
        n_time = raw_layer.data.shape[0]
    else:
        n_time = 1 + max(t for _, t in tracks.graph.nodes(data="time"))
    scene = Scene(
        viewer=viewer,
        raw_layer=raw_layer,
        seg_layer=seg_layer,
        tracks_layer=tracks_viewer.tracking_layers.tracks_layer,
        points_layer=tracks_viewer.tracking_layers.points_layer,
        axes=axes,
        n_spatial=len(axes) - 1,
        n_time=n_time,
        extra_raw_layers=extra_raw_layers,
    )
    return scene, source.home


def load_raw_scene(
    cfg: RenderConfig, dataset: Dataset | None
) -> tuple[Scene, Path | None]:
    """Build the viewer with the raw movie alone; no TracksViewer is opened."""
    if cfg.seg_path is not None:
        raise ValueError("--seg needs tracks to color it by; none were given")
    raw_path = cfg.raw_path or dataset.raw_path
    if raw_path is None or not Path(raw_path).exists():
        where = f" (from {dataset.output_dir / RAW_RECORD})" if dataset else ""
        raise FileNotFoundError(
            f"Raw movie {raw_path}{where} does not exist here; pass --raw"
        )
    viewer = napari.Viewer()
    raw_layer, _, axes = load_raw(viewer, raw_path, None, cfg.channel)
    extra_raw_layers = load_extra_raw(viewer, cfg, raw_layer, axes, raw_path)
    scene = Scene(
        viewer=viewer,
        raw_layer=raw_layer,
        seg_layer=None,
        tracks_layer=None,
        points_layer=None,
        axes=axes,
        n_spatial=len(axes) - 1,
        n_time=raw_layer.data.shape[0],
        raw_only=True,
        extra_raw_layers=extra_raw_layers,
    )
    return scene, dataset.output_dir if dataset is not None else None


def apply_display(scene: Scene, cfg: RenderConfig, mode: str) -> None:
    """Set the view from the config. ``read_display`` is its inverse."""
    viewer = scene.viewer
    viewer.dims.ndisplay = 3 if mode == "3d" else 2
    if mode == "2d" and scene.n_spatial == 3:
        # The dims are in world units, not voxels: TracksViewer's layers have
        # scale 1 (geff positions are world units already), so napari steps z
        # in units of 1, and a voxel index must go through the z scale.
        viewer.dims.set_point(1, cfg.z * scene.axes[1]["scale"])

    if scene.raw_layer is not None:
        default_colormap = "green" if cfg.extra_raw else "gray"
        _apply_raw(scene.raw_layer, cfg.raw_display, mode, default_colormap, "raw")
    extras = zip(scene.extra_raw_layers, cfg.extra_raw, strict=True)
    for i, (layer, extra) in enumerate(extras):
        default_colormap = EXTRA_RAW_COLORMAPS[i % len(EXTRA_RAW_COLORMAPS)]
        _apply_raw(layer, extra, mode, default_colormap, layer.name)
    if scene.seg_layer is not None:
        _apply_seg(scene.seg_layer, cfg, mode)

    if scene.tracks_layer is not None:
        scene.tracks_layer.tail_length = cfg.tail_length or scene.n_time
        scene.tracks_layer.tail_width = cfg.tail_width
        scene.tracks_layer.visible = cfg.show_tracks
    if scene.points_layer is not None:
        scene.points_layer.visible = cfg.show_points

    viewer.scale_bar.visible = cfg.scale_bar
    # None lets napari pick a length from the zoom, which can differ between
    # the setup viewer and the render (the render zooms in by screenshot_scale).
    viewer.scale_bar.length = cfg.scale_bar_length
    set_units(viewer, scene.axes)
    viewer.scale_bar.font_size = cfg.scale_bar_font_size

    viewer.text_overlay.visible = True
    viewer.text_overlay.position = "top_left"
    viewer.text_overlay.font_size = cfg.timestamp_font_size

    for key in ("angles", "center", "zoom", "perspective"):
        if key in cfg.camera:
            value = cfg.camera[key]
            setattr(
                viewer.camera, key, tuple(value) if isinstance(value, list) else value
            )


def _apply_raw(
    raw, settings: dict, mode: str, default_colormap: str, label: str
) -> None:
    """Display one raw layer by its config table ([raw] or an [[extra_raw]])."""
    if "contrast_limits" not in settings:
        # Recorded in the config, so the frames' settings capture it.
        settings["contrast_limits"] = auto_contrast_limits(raw)
        print(f"{label} contrast limits (p1/p99.5): {settings['contrast_limits']}")
    raw.contrast_limits = tuple(settings["contrast_limits"])
    raw.colormap = settings.get("colormap", default_colormap)
    raw.gamma = settings.get("gamma", 1.0)
    if mode == "3d":
        # Labels layers only support translucent/iso_categorical, so this
        # is set on the image layer alone.
        raw.rendering = settings.get("rendering", "attenuated_mip")
        if "attenuation" in settings:
            raw.attenuation = settings["attenuation"]


def _apply_seg(seg, cfg: RenderConfig, mode: str) -> None:
    seg.visible = cfg.seg.get("visible", True)
    seg.opacity = cfg.seg.get("opacity", 0.5)
    if mode == "2d":
        seg.contour = cfg.seg.get("contour", 0)  # napari draws contours in 2D only
    if mode == "3d" and "rendering" in cfg.seg:
        seg.rendering = cfg.seg["rendering"]


def set_units(viewer: napari.Viewer, axes: list[dict]) -> None:
    """Give the scale bar the data's spatial unit.

    napari >= 0.8 takes it from the layers' units, which must agree across
    every layer (TracksViewer's included) or the bar falls back to pixels;
    older napari reads ``scale_bar.unit``.

    Only the spatial axes get a unit. The time axis keeps napari's default:
    given a real time unit, napari's Tracks layer slices at the time converted
    to seconds (frame 1 of an 80 min/frame movie becomes time 60), so every
    track shows at once and its tail fades out within a frame or two.
    """
    spatial = [axis.get("unit") for axis in axes[1:]]
    if any(unit is None for unit in spatial):
        return
    layers = list(viewer.layers)
    if layers and all(hasattr(layer, "units") for layer in layers):
        for layer in layers:
            layer.units = (layer.units[0], *spatial[-(layer.ndim - 1) :])
    else:
        viewer.scale_bar.unit = spatial[-1]


def fit_to_canvas(viewer: napari.Viewer, size: list[int]) -> None:
    """Zoom and center so the data fills a canvas of ``size`` (physical
    pixels), keeping the camera angles. Used when the config sets no zoom, so
    the framing doesn't depend on the window the viewer happened to open in.
    """
    angles = viewer.camera.angles
    with viewer.window._qt_viewer.resize_canvas(tuple(size), 1.0):
        viewer.reset_view()
        fitted = viewer.camera.zoom, viewer.camera.center
    viewer.camera.angles = angles
    viewer.camera.zoom, viewer.camera.center = fitted


def canvas_size(viewer: napari.Viewer) -> list[int]:
    """The canvas size in physical pixels, [height, width], in the form
    ``viewer.screenshot(size=...)`` takes it.
    """
    qt_viewer = viewer.window._qt_viewer
    h, w = qt_viewer.canvas.size
    ratio = qt_viewer.devicePixelRatio()
    return [round(h * ratio), round(w * ratio)]


def read_display(scene: Scene, cfg: RenderConfig, canvas: list[int]) -> dict:
    """The current view as config keys, ready to paste into a TOML."""
    viewer = scene.viewer
    ndisplay = viewer.dims.ndisplay
    out: dict = {"mode": "3d" if ndisplay == 3 else "2d"}
    if ndisplay == 2 and scene.n_spatial == 3:
        out["z"] = round(viewer.dims.point[1] / scene.axes[1]["scale"])
    out["canvas_size"] = canvas
    if scene.tracks_layer is not None:
        out["tail_length"] = int(scene.tracks_layer.tail_length)
        out["tail_width"] = float(scene.tracks_layer.tail_width)
        out["show_tracks"] = bool(scene.tracks_layer.visible)
    if scene.points_layer is not None:
        out["show_points"] = bool(scene.points_layer.visible)
    out["scale_bar"] = bool(viewer.scale_bar.visible)
    if viewer.scale_bar.length is not None:
        out["scale_bar_length"] = float(viewer.scale_bar.length)
    # Not adjustable in the viewer; carried over so the pasted fragment keeps
    # them.
    out["timestamp_font_size"] = cfg.timestamp_font_size
    out["scale_bar_font_size"] = cfg.scale_bar_font_size

    if scene.raw_layer is not None:
        out["raw"] = _read_raw(scene.raw_layer, ndisplay)
    if scene.extra_raw_layers:
        out["extra_raw"] = [
            {"path": extra["path"], "channel": extra.get("channel", 0)}
            | _read_raw(layer, ndisplay)
            for layer, extra in zip(scene.extra_raw_layers, cfg.extra_raw, strict=True)
        ]
    if scene.seg_layer is not None:
        seg = scene.seg_layer
        out["seg"] = {"visible": bool(seg.visible), "opacity": float(seg.opacity)}
        if ndisplay == 2:
            out["seg"]["contour"] = int(seg.contour)
        else:
            out["seg"]["rendering"] = str(seg.rendering)
    cam = viewer.camera
    out["camera"] = {
        "angles": [float(v) for v in cam.angles],
        "zoom": float(cam.zoom),
        "center": [float(v) for v in cam.center],
        "perspective": float(cam.perspective),
    }
    return out


def _read_raw(raw, ndisplay: int) -> dict:
    out = {
        "contrast_limits": [float(v) for v in raw.contrast_limits],
        "colormap": raw.colormap.name,
        "gamma": float(raw.gamma),
    }
    if ndisplay == 3:
        out["rendering"] = str(raw.rendering)
        out["attenuation"] = float(raw.attenuation)
    return out


def run_setup(scene: Scene, cfg: RenderConfig) -> None:
    """Open the viewer for adjusting the view by hand; print the settings on
    Shift-S and when the window closes.
    """
    viewer = scene.viewer
    last_canvas = [list(cfg.canvas_size)]

    def print_settings():
        try:
            last_canvas[0] = canvas_size(viewer)
        except RuntimeError:
            pass  # window already gone; keep the last size seen
        fragment = toml.dumps(read_display(scene, cfg, last_canvas[0]))
        h, w = (round(v * cfg.screenshot_scale) for v in last_canvas[0])
        print("\n# --- display settings: paste into the render config ---")
        print(
            f"# frames come out {h} x {w} px "
            f"(canvas_size x screenshot_scale = {cfg.screenshot_scale})"
        )
        print(fragment)

    @viewer.bind_key("Shift-S", overwrite=True)
    def _print(_viewer):
        print_settings()

    print("Adjust the view, then press Shift-S or close the window to print it.")
    napari.run()
    print_settings()


def render(scene: Scene, cfg: RenderConfig, frames_dir: Path, output: Path) -> None:
    """Capture one PNG per frame (skipping those already on disk), then
    assemble the video.
    """
    viewer = scene.viewer
    end = scene.n_time if cfg.end is None else min(cfg.end, scene.n_time)
    frame_range = range(cfg.start, end)
    if not frame_range:
        raise ValueError(f"Empty frame range [{cfg.start}, {end})")
    if "zoom" not in cfg.camera:
        fit_to_canvas(viewer, cfg.canvas_size)
        if "center" in cfg.camera:
            viewer.camera.center = tuple(cfg.camera["center"])
    # Formatted for the whole movie, not the rendered range, so frames reused
    # from a render of another range carry the same format.
    time_label = time_formatter(scene.axes[0], scene.n_time, cfg.time_unit)
    # screenshot(scale=...) enlarges the canvas but draws text at its on-screen
    # pixel size, so the fonts are scaled up to keep the setup viewer's
    # proportions. The timestamp's is reapplied on every frame: TracksViewer
    # resets the overlay text and font size whenever its display mode changes.
    font_size = cfg.timestamp_font_size * cfg.screenshot_scale
    viewer.scale_bar.font_size = cfg.scale_bar_font_size * cfg.screenshot_scale

    frame_paths = []
    for t in tqdm(frame_range, desc="Rendering frames"):
        frame_path = frames_dir / f"frame_{t:04d}.png"
        frame_paths.append(frame_path)
        if frame_path.exists():
            continue  # captured by an earlier, interrupted run
        viewer.dims.set_point(0, t)  # time has scale 1 on every layer
        viewer.text_overlay.text = time_label(t)
        viewer.text_overlay.font_size = font_size
        # screenshot(scale=...) enlarges the canvas without zooming in, so
        # the content would come out smaller. Zoom in to match; napari
        # restores the zoom after the capture.
        zoom = viewer.camera.zoom
        viewer.camera.zoom = zoom * cfg.screenshot_scale
        try:
            frame = viewer.screenshot(
                size=tuple(cfg.canvas_size),
                scale=cfg.screenshot_scale,
                canvas_only=True,
                flash=False,
            )
        finally:
            viewer.camera.zoom = zoom
        iio.imwrite(frame_path, even_crop(frame))

    write_video(frame_paths, output, cfg.fps, cfg.quality)


def default_output(
    cfg: RenderConfig, mode: str, scene: Scene, home: Path | None
) -> Path:
    """``<run>/videos/`` for predictions, ``<gt folder>/videos/`` for a GT,
    ``<output_dir>/videos/raw_*`` for the raw movie alone.
    """
    if home is None:
        raise ValueError(
            "A raw movie given only by --raw belongs to no output directory; "
            "pass -o, or name the output directory instead"
        )
    stem = mode if not (mode == "2d" and scene.n_spatial == 3) else f"2d_z{cfg.z}"
    if scene.raw_only:
        stem = f"raw_{stem}"
    return home / "videos" / f"{stem}.mp4"


def parse_args() -> tuple[RenderConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Render a video of a run's predicted tracks, or a ground "
        "truth, over the raw movie; or of the raw movie alone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tracks",
        type=Path,
        nargs="?",
        default=None,
        help="a tracking run, <output_dir>/tracking/<uid>, or an SSVM fit, "
        "<output_dir>/ssvm/<uid> (or its config.toml); or a ground truth, e.g. "
        "<output_dir>/gt/ctc or its correct_tracks.zarr; or an output "
        "directory (or its raw.toml), to show its raw movie alone",
    )
    parser.add_argument("--config", type=Path, default=None, help="render TOML")
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="raw movie, if the path in raw.toml is not valid on this machine, "
        "or the tracks are outside any output directory; given alone, the "
        "raw movie is shown without tracks (needs -o)",
    )
    parser.add_argument(
        "--extra-raw",
        action="append",
        metavar="PATH[@CHANNEL]",
        help="another raw movie to overlay (repeatable, added after any "
        "[[extra_raw]] of the config); PATH@1 is channel 1 of that zarr, and "
        "@1 alone is channel 1 of the first raw movie's zarr",
    )
    parser.add_argument(
        "--seg",
        type=Path,
        default=None,
        help="segmentation, instead of the one beside the tracks",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["auto", "3d", "2d"], default=None)
    parser.add_argument("--z", type=int, default=None, help="slice for 2d mode")
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None, help="exclusive")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--quality", type=float, default=None)
    parser.add_argument("--screenshot-scale", type=float, default=None)
    parser.add_argument("--tail-length", type=int, default=None)
    parser.add_argument(
        "--scale-bar-length",
        type=float,
        default=None,
        help="fixed scale bar length in the spatial unit (e.g. 50 for 50 um); "
        "default: chosen by napari from the zoom",
    )
    parser.add_argument(
        "--time-unit",
        default=None,
        help="show the time in this one unit (s, min or h) instead of the "
        "format that fits the movie",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="open the viewer to set the view by hand and print it as TOML",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="clear frames rendered with other settings",
    )
    args = parser.parse_args()

    cfg = RenderConfig.from_toml(args.config) if args.config else RenderConfig()
    cfg.override_from_args(args)
    return cfg, args


def main():
    cfg, args = parse_args()
    scene, home = load_scene(cfg)
    mode = resolve_mode(cfg, scene.n_spatial)
    apply_display(scene, cfg, mode)

    if args.setup:
        run_setup(scene, cfg)
        return

    output = (
        absolute_path(Path(cfg.output))
        if cfg.output
        else default_output(cfg, mode, scene, home)
    )
    frames_dir = output.parent / f"{output.stem}_frames"
    prepare_frames_dir(frames_dir, cfg.frame_settings(), args.overwrite)
    render(scene, cfg, frames_dir, output)
    scene.viewer.close()


if __name__ == "__main__":
    main()
