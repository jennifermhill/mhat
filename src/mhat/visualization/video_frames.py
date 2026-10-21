"""Per-frame PNGs and the video assembled from them.

A render writes each frame to ``<frames_dir>/frame_{t:04d}.png`` as soon as it
is drawn, so an interrupted render resumes at the first missing frame. The
frames directory records the settings that produced it
(``render_settings.json``), and a rerun with other settings refuses to reuse
its frames.
"""

import json
import shutil
from pathlib import Path

import imageio.v2 as iio2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


def prepare_frames_dir(frames_dir: Path, settings: dict, overwrite: bool) -> None:
    """Make sure every PNG in ``frames_dir`` was rendered with ``settings``."""
    settings_path = frames_dir / "render_settings.json"
    if frames_dir.exists() and overwrite:
        # Only ever the frames directory this script made -- videos may sit
        # beside a hand-annotated GT store.
        if any(frames_dir.glob("frame_*.png")) and not settings_path.exists():
            raise RuntimeError(
                f"{frames_dir} has frames but no render_settings.json, so it "
                f"was not made by this script; not deleting it"
            )
        shutil.rmtree(frames_dir)
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        if previous != settings:
            changed = sorted(
                k
                for k in set(previous) | set(settings)
                if previous.get(k) != settings.get(k)
            )
            raise RuntimeError(
                f"{frames_dir} holds frames rendered with other settings "
                f"(changed: {changed}). Pass --overwrite to clear it, or "
                f"choose another --output."
            )
    elif frames_dir.exists() and any(frames_dir.glob("frame_*.png")):
        raise RuntimeError(
            f"{frames_dir} has frames but no render_settings.json; move it "
            f"away or choose another --output"
        )
    frames_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2))


def even_crop(frame: np.ndarray) -> np.ndarray:
    """Drop a trailing row/column so both dimensions are even, as H.264 in
    yuv420p needs.
    """
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


def write_video(
    frame_paths: list[Path], output: Path, fps: int, quality: float
) -> None:
    """Assemble the PNGs into ``output``, in the order given."""
    output.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=2: the frames are already even, so ffmpeg needn't resize.
    with iio2.get_writer(
        output, fps=fps, quality=quality, macro_block_size=2
    ) as writer:
        for frame_path in tqdm(frame_paths, desc="Writing video"):
            writer.append_data(iio.imread(frame_path)[..., :3])
    print(f"Wrote {output} ({len(frame_paths)} frames at {fps} fps)")
