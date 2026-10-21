"""Render chosen frames of a track (or raw) view as PNGs for a paper figure.

The config names a ``render_tracks_video.py`` config, ``render_config``, and
the frame indices to render, ``frames``. Each frame is rendered as that
video's frame would be, except that:

- no frame has a timestamp;
- every frame has the scale bar, at the same place, but only the last one has
  its length written above it.

So the video and the stills share one set of view settings. The video config's
``start``, ``end``, ``fps``, ``quality`` and ``output`` don't apply here.
Any other render key in this config overrides the video config's (a table such
as ``[camera]`` key by key), e.g. a smaller ``scale_bar_font_size`` for print.

The PNGs are ``<output>/frame_{t:04d}.png``; ``output`` is a directory, by
default ``<run>/paper_frames/<this config's name>/`` (``<gt folder>/...`` for
a ground truth, ``<output_dir>/...`` for the raw movie alone). As with the
videos, the directory records the settings that produced it, and a rerun with
other settings, a different ``frames`` list included, needs ``--overwrite``.

Usage:
    python scripts/07_plotting/render_paper_frames.py --config frames.toml
    python scripts/07_plotting/render_paper_frames.py --config frames.toml --setup
    python scripts/07_plotting/render_paper_frames.py --config frames.toml \
        --frames 0 10 20 -o <dir> --overwrite

``--setup`` prints the view as ``render_tracks_video.py --setup`` does, to
paste into the video config.

Example TOML config (keep these under scratch_configs/visualization/):

    render_config = "tracks_seg_easy.toml"  # relative to this file's folder
    frames = [0, 10, 20]     # increasing; the scale bar's length is on frame 20
    # output = "<dir>"       # where the PNGs go
    # scale_bar_font_size = 12   # overrides of the video config, if any
"""

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import imageio.v3 as iio
import toml
from mhat.dataset import absolute_path
from mhat.visualization.video_frames import prepare_frames_dir
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "04_tracking"))
import render_tracks_video as rtv  # noqa: E402

# Keys of a render config that only a video uses.
VIDEO_KEYS = {"start", "end", "fps", "quality", "output"}
# The tables of a render config, by their TOML name.
TABLES = {"raw": "raw_display", "seg": "seg", "camera": "camera"}


def load_config(path: Path) -> tuple[rtv.RenderConfig, list[int], str | None]:
    """The video config with this config's overrides, the frames, and the
    output directory (None for the default).
    """
    data = toml.load(path)
    if "render_config" not in data:
        raise ValueError(f"{path}: no render_config (the video config to use)")
    render_path = Path(data.pop("render_config"))
    if not render_path.is_absolute():
        render_path = path.parent / render_path
    frames = data.pop("frames", [])
    output = data.pop("output", None)

    video_only = sorted(set(data) & VIDEO_KEYS)
    if video_only:
        raise ValueError(f"{path}: {video_only} only apply to a video")
    cfg = rtv.RenderConfig.from_toml(render_path)
    fields = set(rtv.RenderConfig.__dataclass_fields__)
    for key, value in data.items():
        if key in TABLES:
            if not isinstance(value, dict):
                raise ValueError(f"{path}: `{key}` must be a table")
            getattr(cfg, TABLES[key]).update(value)
        elif key in fields and key not in TABLES.values():
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown key in {path}: {key!r}")
    cfg.check_tables(path)
    return cfg, frames, output


def check_frames(frames: list[int], n_time: int) -> list[int]:
    if not frames:
        raise ValueError("No frames to render: set `frames` (or --frames)")
    if any(b <= a for a, b in pairwise(frames)):
        raise ValueError(f"frames must be increasing, got {frames}")
    if frames[0] < 0 or frames[-1] >= n_time:
        raise ValueError(f"frames {frames} outside the movie's 0..{n_time - 1}")
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render chosen frames as PNGs for a figure: no timestamp, "
        "the scale bar on every frame, its length on the last.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="frames TOML")
    parser.add_argument(
        "--frames", type=int, nargs="+", default=None, help="instead of `frames`"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="directory for the PNGs"
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
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = absolute_path(args.config)
    cfg, frames, output = load_config(config_path)
    if args.frames is not None:
        frames = args.frames
    if args.output is not None:
        output = args.output
    scene, home = rtv.load_scene(cfg)
    mode = rtv.resolve_mode(cfg, scene.n_spatial)
    rtv.apply_display(scene, cfg, mode)
    scene.viewer.text_overlay.visible = False

    if args.setup:
        rtv.run_setup(scene, cfg)
        return

    frames = check_frames(frames, scene.n_time)
    if output is not None:
        out_dir = absolute_path(output)
    elif home is not None:
        out_dir = home / "paper_frames" / config_path.stem
    else:
        raise ValueError(
            "A raw movie given only by raw_path belongs to no output directory; "
            "set output or pass -o"
        )
    prepare_frames_dir(
        out_dir, cfg.frame_settings() | {"frames": frames}, args.overwrite
    )
    scale_bar = rtv.prepare_capture(scene, cfg)
    for t in tqdm(frames, desc="Rendering frames"):
        frame_path = out_dir / f"frame_{t:04d}.png"
        if frame_path.exists():
            continue  # captured by an earlier, interrupted run
        frame = rtv.capture_frame(
            scene, cfg, t, None, scale_bar, scale_bar_text=t == frames[-1]
        )
        iio.imwrite(frame_path, frame)
    print(f"Wrote {len(frames)} frames to {out_dir}")
    scene.viewer.close()


if __name__ == "__main__":
    main()
