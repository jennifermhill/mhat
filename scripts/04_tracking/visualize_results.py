import argparse
from pathlib import Path

import napari
from motile_tracker.application_menus import MainApp

from mhat.visualization.tracks_view import load_raw, load_tracks, resolve_tracks


def main(
    tracks: Path,
    raw_path: Path | None = None,
    seg_path: Path | None = None,
    compute: bool = False,
):
    # The output directory is found from where the tracks sit, so this works on
    # any machine; raw comes from raw.toml unless --raw overrides it.
    source = resolve_tracks(tracks, seg_path)
    recorded_raw = source.dataset.raw_path if source.dataset is not None else None
    raw_cells_zarr_path = raw_path or recorded_raw

    if source.is_gt:
        print("Visualizing ground truth tracks.")
    else:
        print("Visualizing predicted tracks.")

    viewer = napari.Viewer()

    _, scale, _ = load_raw(
        viewer, raw_cells_zarr_path, source.tracks_path, compute=compute
    )

    # Add the MainApp widget first
    widget = MainApp(viewer)
    viewer.window.add_dock_widget(widget)

    try:
        tracks_viewer, _, _ = load_tracks(
            viewer, source.tracks_path, source.seg_path, scale, source.name,
            compute=compute,
        )
        if tracks_viewer.tracking_layers.tracks_layer is not None:
            tracks_viewer.tracking_layers.tracks_layer.tail_length = 4
            tracks_viewer.tracking_layers.tracks_layer.tail_width = 2.0
            tracks_viewer.tracking_layers.points_layer.visible = False
    except Exception as e:
        print(f"Failed to load tracks: {e}")

    napari.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="View one set of tracks in napari over its raw movie: a "
                    "run's predictions, or a ground truth."
    )
    parser.add_argument(
        "tracks", type=Path,
        help="a tracking run, <output_dir>/tracking/<uid>, or an SSVM fit, "
             "<output_dir>/ssvm/<uid> (or its config.toml); or a ground truth, "
             "e.g. <output_dir>/gt/ctc or its correct_tracks.zarr",
    )
    parser.add_argument(
        "--raw", type=Path, default=None,
        help="raw movie to show, if the path in raw.toml is not valid on this "
             "machine, or the tracks are outside any output directory",
    )
    parser.add_argument(
        "--seg", type=Path, default=None,
        help="segmentation to show, instead of the one beside the tracks "
             "(pred_seg.zarr / correct_seg.zarr)",
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="load the arrays into memory instead of viewing them lazily",
    )
    args = parser.parse_args()
    main(args.tracks, raw_path=args.raw, seg_path=args.seg, compute=args.compute)
