"""Render chosen frames of a track (or raw) view as PNGs for a paper figure.

Takes a ``render_tracks_video.py`` config plus ``frames``, the frame indices
to render, and renders exactly as the video would, except that:

- no frame has a timestamp;
- every frame has the scale bar, at the same place, but only the last one has
  its length written above it.

The PNGs are ``<output>/frame_{t:04d}.png``; ``output`` is a directory here,
by default ``<run>/paper_frames/<mode>/`` (``<gt folder>/paper_frames/...``
for a ground truth, ``<output_dir>/paper_frames/raw_<mode>/`` for the raw
movie alone). As with the videos, the directory records the settings that
produced it, and a rerun with other settings, a different ``frames`` list
included, needs ``--overwrite``.

Usage:
    python scripts/07_plotting/render_paper_frames.py --config frames.toml
    python scripts/07_plotting/render_paper_frames.py --config frames.toml --setup
    python scripts/07_plotting/render_paper_frames.py --config frames.toml \
        --frames 0 10 20 -o <dir> --overwrite

Example TOML config (keep these under scratch_configs/visualization/): any
render config, without the video-only keys, plus

    frames = [0, 10, 20]     # increasing; the scale bar's length is on frame 20
    # output = "<dir>"       # where the PNGs go
"""

import argparse
import sys
from dataclasses import dataclass, field
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
VIDEO_KEYS = ("start", "end", "fps", "quality")


@dataclass
class PaperFramesConfig(rtv.RenderConfig):
    frames: list[int] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: str | Path) -> "PaperFramesConfig":
        video_only = sorted(set(toml.load(path)) & set(VIDEO_KEYS))
        if video_only:
            raise ValueError(
                f"{path}: {video_only} only apply to a video; the frames "
                f"rendered are `frames`"
            )
        return super().from_toml(path)


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
    cfg = PaperFramesConfig.from_toml(args.config)
    if args.frames is not None:
        cfg.frames = args.frames
    if args.output is not None:
        cfg.output = str(args.output)
    scene, home = rtv.load_scene(cfg)
    mode = rtv.resolve_mode(cfg, scene.n_spatial)
    rtv.apply_display(scene, cfg, mode)
    scene.viewer.text_overlay.visible = False

    if args.setup:
        rtv.run_setup(scene, cfg)
        return

    frames = check_frames(cfg.frames, scene.n_time)
    if cfg.output is not None:
        out_dir = absolute_path(Path(cfg.output))
        if out_dir.suffix == ".mp4":
            raise ValueError(f"output is the directory for the PNGs, got {out_dir}")
    else:
        video = rtv.default_output(cfg, mode, scene, home)
        out_dir = video.parent.parent / "paper_frames" / video.stem
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
