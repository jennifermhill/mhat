"""Render a movie of confidence-masked quiver plots of one z slice of a 3D flow.

Each frame is a matplotlib quiver plot of the in-plane flow, with the arrows
colored by direction on a cyclic colormap and drawn to scale in the data's
units. A timestamp sits in the top-left corner, and a legend in another corner:
a color ring giving the color of every direction, and eight arrows at
``scale_bar_velocity`` showing how arrow length relates to speed.

Flow frame ``t`` is the motion from raw frame ``t`` to ``t + 1``, and is
labeled with frame ``t``'s time. The last flow frame is an all-zero placeholder;
its zero vectors draw no arrows, so it comes out as a frame with only the
timestamp and legend. That keeps the movie frame for frame with the raw movie
and its tracks videos (``render_tracks_video.py``, 4 fps by default, as here).

Scales and units come from the axes recorded in the output directory's
``raw.toml``, which is kept in step with the raw movie's metadata. The flow
store has its own copy, taken when the flow ran, which goes stale when the raw
metadata is corrected later; it is used only when ``raw.toml`` records no axes.
``xy_scale``, ``t_scale``, ``xyz_scale_unit`` and ``t_scale_unit`` override
either (e.g. ``t_scale = 0.5``, ``t_scale_unit = "min"`` for a 30 s interval,
so the legend reads um/min).

Each frame is written to a PNG as soon as it is drawn, so an interrupted render
resumes at the first missing frame. The frames directory records the settings
that produced it, and a rerun with different settings refuses to reuse its
frames (``--overwrite`` clears that directory and nothing else). The movie goes
in ``<flow run>/videos/`` unless ``-o`` says otherwise.

Usage:
    python scripts/03_opticalflow/make_quiver_movie.py movie_config.toml
    python scripts/03_opticalflow/make_quiver_movie.py test.toml --end 3 -o t.mp4

See ``movie_config_example.toml`` for the config keys.
"""

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import colorcet as cc
import dask.array as da
import imageio.v3 as iio
import numpy as np
import toml
import zarr
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Wedge
from mhat.dataset import RAW_RECORD, Dataset, absolute_path, require_dir
from mhat.opticalflow.utils import open_flow_raw
from mhat.utils import get_axes_metadata
from mhat.visualization.timestamps import time_formatter
from mhat.visualization.video_frames import (
    even_crop,
    prepare_frames_dir,
    write_video,
)
from tqdm import tqdm

COLORMAP = cc.m_CET_C8
DIRECTION_NORM = Normalize(vmin=-np.pi, vmax=np.pi)

# Unit names as axes metadata spell them, shortened for labels.
SHORT_UNITS = {
    "micrometer": "µm",
    "micrometers": "µm",
    "micron": "µm",
    "microns": "µm",
    "um": "µm",
    "nanometer": "nm",
    "nanometers": "nm",
    "millimeter": "mm",
    "millimeters": "mm",
    "second": "s",
    "seconds": "s",
    "sec": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "hr": "h",
}

LEGEND_CORNERS = ("bottom_right", "bottom_left", "top_right", "top_left")

RETIRED_KEYS = {
    "png_dpi": "replaced by `dpi`, which sets both the PNGs and the video",
    "z_scale": "no longer used",
}

# Settings that don't change what a frame looks like; every other field does.
NOT_FRAME_SETTINGS = {"output", "start", "end", "fps", "quality"}


@dataclass
class QuiverConfig:
    # IO
    output_dir: str | None = None
    raw_path: str | None = None
    flow_result: str | None = None
    flow_kind: str = "3d"
    output: str | None = None
    # Overrides for the flow store's axes metadata
    xy_scale: float | None = None
    t_scale: float | None = None
    xyz_scale_unit: str | None = None
    t_scale_unit: str | None = None
    # Thresholding
    confidence_percentile: float = 90
    # Visualization
    z_slice: int | None = None
    gap: int = 2
    quiver_scale: float = 1.0
    arrow_width: float = 0.3
    scale_bar_velocity: float = 5.0
    legend_corner: str = "bottom_right"
    timestamp_font_size: float = 18
    legend_font_size: float = 14
    time_unit: str | None = None
    fig_width: float = 10.0
    dpi: float = 150
    # Movie
    start: int = 0
    end: int | None = None
    fps: int = 4  # render_tracks_video.py's default, so the videos play in step
    quality: float = 9

    @classmethod
    def from_toml(cls, path: str | Path) -> "QuiverConfig":
        data = toml.load(path)
        Dataset.from_config(data)  # rejects the old layout's IO keys
        retired = {k: why for k, why in RETIRED_KEYS.items() if k in data}
        if retired:
            raise ValueError(f"Retired keys in {path}: {retired}")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        cfg = cls(**data)
        if cfg.flow_result is None:
            raise ValueError(f"{path}: flow_result is required")
        if cfg.z_slice is None:
            raise ValueError(f"{path}: z_slice is required")
        if cfg.legend_corner not in LEGEND_CORNERS:
            raise ValueError(
                f"{path}: legend_corner must be one of {LEGEND_CORNERS}, "
                f"got {cfg.legend_corner!r}"
            )
        return cfg

    def override_from_args(self, args: argparse.Namespace) -> None:
        for name in ("output", "start", "end", "fps", "quality"):
            value = getattr(args, name)
            if value is not None:
                setattr(self, name, str(value) if isinstance(value, Path) else value)

    def resolve_axes(self, axes: list[dict]) -> None:
        """Fill the scales and units not given in the config from ``axes``."""
        y_axis, x_axis = axes[-2], axes[-1]
        if self.xy_scale is None:
            if not np.isclose(y_axis["scale"], x_axis["scale"]):
                raise ValueError(
                    f"y and x scales differ ({y_axis['scale']} vs "
                    f"{x_axis['scale']}); the quiver plot assumes square pixels"
                )
            self.xy_scale = x_axis["scale"]
        if self.t_scale is None:
            self.t_scale = axes[0]["scale"]
        if self.xyz_scale_unit is None:
            self.xyz_scale_unit = x_axis.get("unit") or "px"
        if self.t_scale_unit is None:
            self.t_scale_unit = axes[0].get("unit") or "frame"

    def frame_settings(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in NOT_FRAME_SETTINGS}


def movie_axes(ds: Dataset, flow_raw) -> list[dict]:
    """The movie's axes, (t, *spatial): from ``raw.toml`` if it records them,
    else the flow store's copy.
    """
    record = ds.output_dir / RAW_RECORD
    axes = toml.load(record).get("axes") if record.is_file() else None
    if axes:
        source = record
    else:
        source = "the flow store"
        print(
            f"No axes recorded in {record}; using the flow store's copy, which "
            f"may predate a correction to the raw movie's metadata"
        )
        axes = get_axes_metadata(flow_raw)
    if len(axes) != flow_raw.ndim - 1:  # flow_raw is (t, *spatial, components)
        raise ValueError(
            f"Axes from {source} ({[a['name'] for a in axes]}) don't match the "
            f"flow's shape {flow_raw.shape}"
        )
    return axes


def short_unit(unit: str) -> str:
    return SHORT_UNITS.get(unit.lower(), unit)


def direction_colors(vy, vx) -> np.ndarray:
    """RGBA color of each direction ``(vy, vx)``, in image coordinates (y
    down). The data arrows and the legend both go through here, so the legend
    can't disagree with the arrows.
    """
    theta = np.arctan2(vy, vx)
    return COLORMAP(DIRECTION_NORM(np.ravel(theta)))


def draw_legend(ax, cfg: QuiverConfig, x_range, y_range) -> None:
    """Color ring plus eight arrows at ``scale_bar_velocity``, in a corner.

    Drawn in data units, so the legend arrows have exactly the length a data
    arrow of that speed has. The y axis is inverted, so a larger y is lower on
    screen; every angle and direction here is in those data coordinates.
    """
    arm = cfg.scale_bar_velocity / cfg.quiver_scale  # arrow length, data units
    r_inner, r_outer = 1.2 * arm, 1.45 * arm
    backing_radius = r_outer + 0.15 * arm
    margin = 0.03 * (x_range[1] - x_range[0])
    pad = 0.1 * arm  # between the ring and the label

    # The speed label, measured so the legend fits in its corner whichever of
    # the label and the ring is wider. It is positioned once the ring is.
    speed_unit = f"{short_unit(cfg.xyz_scale_unit)}/{short_unit(cfg.t_scale_unit)}"
    label = ax.text(
        0,
        0,
        f"{cfg.scale_bar_velocity:g} {speed_unit}",
        color="white",
        ha="center",
        va="top",
        fontsize=cfg.legend_font_size,
        zorder=6,
    )
    extent = label.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    data_per_px = (x_range[1] - x_range[0]) / ax.figure.bbox.width
    label_w, label_h = extent.width * data_per_px, extent.height * data_per_px
    half_w = max(backing_radius, label_w / 2)

    left = cfg.legend_corner.endswith("left")
    top = cfg.legend_corner.startswith("top")
    cx = x_range[0] + margin + half_w if left else x_range[1] - margin - half_w
    if top:
        cy = y_range[0] + margin + backing_radius
    else:
        cy = y_range[1] - margin - label_h - pad - backing_radius
    label.set_position((cx, cy + backing_radius + pad))  # below the ring

    # A dark disk under the legend keeps it readable over dense arrows.
    backing = PatchCollection(
        [Wedge((cx, cy), backing_radius, 0, 360)],
        facecolor="black",
        alpha=0.8,
        linewidths=0,
        zorder=3,
    )
    ax.add_collection(backing)

    # The ring: one wedge per direction bin, colored by its mid-angle. The
    # wedges overlap slightly so no seams show between them.
    n_bins = 72
    step = 360 / n_bins
    mids = np.deg2rad(np.arange(n_bins) * step)
    wedges = [
        Wedge(
            (cx, cy),
            r_outer,
            np.rad2deg(mid) - step / 2 - 0.3,
            np.rad2deg(mid) + step / 2 + 0.3,
            width=r_outer - r_inner,
        )
        for mid in mids
    ]
    ring = PatchCollection(wedges, linewidths=0, zorder=4)
    ring.set_facecolor(direction_colors(np.sin(mids), np.cos(mids)))
    ax.add_collection(ring)

    angles = np.deg2rad(np.arange(8) * 45)
    dx, dy = np.cos(angles), np.sin(angles)
    ax.quiver(
        np.full(8, cx),
        np.full(8, cy),
        dx * cfg.scale_bar_velocity,
        dy * cfg.scale_bar_velocity,
        color=direction_colors(dy, dx),
        scale=cfg.quiver_scale,
        angles="xy",
        scale_units="xy",
        units="xy",
        width=cfg.arrow_width,
        zorder=5,
    )


def render_frame(
    cfg: QuiverConfig,
    flow_frame: np.ndarray,
    conf_frame: np.ndarray,
    timestamp: str,
) -> np.ndarray:
    """One frame as an RGB array."""
    n_y, n_x = conf_frame.shape
    x = np.arange(n_x) * cfg.xy_scale
    y = np.arange(n_y) * cfg.xy_scale
    x_range, y_range = (x[0], x[-1]), (y[0], y[-1])
    width, height = x_range[1] - x_range[0], y_range[1] - y_range[0]

    # --- Confidence mask ---
    conf_mask = conf_frame > np.percentile(conf_frame, cfg.confidence_percentile)

    # Components are in axis order, (vz, vy, vx) for 3D flow, so the in-plane
    # pair is the last two. Masked and zero vectors become NaN (not drawn);
    # the rest are converted to physical units.
    to_physical = cfg.xy_scale / cfg.t_scale
    vy = flow_frame[..., -2].astype(float) * to_physical
    vx = flow_frame[..., -1].astype(float) * to_physical
    hidden = ~conf_mask | ((vx == 0) & (vy == 0))
    vx[hidden] = np.nan
    vy[hidden] = np.nan

    gap = cfg.gap
    grid_x, grid_y = np.meshgrid(x[::gap], y[::gap])
    vx_sub, vy_sub = vx[::gap, ::gap], vy[::gap, ::gap]

    # A bare Figure on an Agg canvas: no pyplot state, nothing to close.
    fig = Figure(
        figsize=(cfg.fig_width, cfg.fig_width * height / width),
        dpi=cfg.dpi,
        facecolor="black",
    )
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.quiver(
        grid_x,
        grid_y,
        vx_sub,
        vy_sub,
        color=direction_colors(vy_sub, vx_sub),
        scale=cfg.quiver_scale,
        angles="xy",
        scale_units="xy",
        units="xy",
        width=cfg.arrow_width,
    )

    draw_legend(ax, cfg, x_range, y_range)

    ax.text(
        0.02,
        0.98,
        timestamp,
        transform=ax.transAxes,
        color="white",
        ha="left",
        va="top",
        fontsize=cfg.timestamp_font_size,
        zorder=6,
    )

    ax.set_xlim(*x_range)
    ax.set_ylim(y_range[1], y_range[0])  # image orientation: y down
    ax.set_aspect("equal")
    ax.axis("off")

    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def default_output(cfg: QuiverConfig, flow_run: Path) -> Path:
    stem = (
        f"quiver_z{cfg.z_slice}_conf{cfg.confidence_percentile:g}"
        f"_gap{cfg.gap}_scale{cfg.quiver_scale:g}"
    )
    return flow_run / "videos" / f"{stem}.mp4"


def parse_args() -> tuple[QuiverConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Render a movie of confidence-masked quiver plots of one z "
        "slice of a flow run.",
    )
    parser.add_argument("config", type=Path, help="movie TOML")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None, help="exclusive")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--quality", type=float, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="clear frames rendered with other settings",
    )
    args = parser.parse_args()
    cfg = QuiverConfig.from_toml(args.config)
    cfg.override_from_args(args)
    return cfg, args


def main():
    cfg, args = parse_args()
    if cfg.flow_kind == "2d":
        raise ValueError(
            "2D flow visualization not implemented yet (needs confidence "
            "array). Please set flow_kind to '3d'."
        )
    if cfg.flow_kind != "3d":
        raise ValueError(f"flow_kind must be '2d' or '3d', got {cfg.flow_kind!r}")

    ds = Dataset.from_config(asdict(cfg))
    flow_run = ds.flow_dir(cfg.flow_result, cfg.flow_kind)
    data_dir = require_dir(flow_run / "flow.zarr", "Flow store")
    print(f"Loading data from {data_dir}")

    axes = movie_axes(ds, zarr.open(data_dir, mode="r")["flow_raw"])
    cfg.resolve_axes(axes)
    print(
        f"Scales: {cfg.xy_scale:g} {cfg.xyz_scale_unit}/px, "
        f"{cfg.t_scale:g} {cfg.t_scale_unit}/frame"
    )

    flow = open_flow_raw(data_dir, dask=True)  # (vz, vy, vx), legacy stores included
    conf = da.from_zarr(data_dir / "confidence")
    n_time = flow.shape[0]
    flow_slice = flow[:, cfg.z_slice]  # [T, Y, X, C]
    conf_slice = conf[:, cfg.z_slice]  # [T, Y, X]

    # Every frame, the trailing all-zero placeholder included (drawn without
    # arrows), so the movie has as many frames as the raw movie.
    end = n_time if cfg.end is None else min(cfg.end, n_time)
    frame_range = range(cfg.start, end)
    if not frame_range:
        raise ValueError(f"Empty frame range [{cfg.start}, {end})")
    # Formatted for the whole movie, not the rendered range, so frames reused
    # from a render of another range carry the same format.
    time_axis = {"scale": cfg.t_scale, "unit": cfg.t_scale_unit}
    time_label = time_formatter(time_axis, n_time, cfg.time_unit)

    output = (
        absolute_path(Path(cfg.output)) if cfg.output else default_output(cfg, flow_run)
    )
    frames_dir = output.parent / f"{output.stem}_frames"
    prepare_frames_dir(frames_dir, cfg.frame_settings(), args.overwrite)

    frame_paths = []
    for t in tqdm(frame_range, desc="Rendering frames"):
        frame_path = frames_dir / f"frame_{t:04d}.png"
        frame_paths.append(frame_path)
        if frame_path.exists():
            continue  # drawn by an earlier, interrupted run
        img = render_frame(
            cfg,
            flow_slice[t].compute(),
            conf_slice[t].compute(),
            time_label(t),
        )
        iio.imwrite(frame_path, even_crop(img))

    write_video(frame_paths, output, cfg.fps, cfg.quality)
    print(f"PNG frames in {frames_dir}")


if __name__ == "__main__":
    main()
