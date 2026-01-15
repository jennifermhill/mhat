import argparse
import csv
from pathlib import Path

import napari
import numpy as np
import zarr


def agglomerate(fragments: np.ndarray, merge_history: Path, threshold: float):
    """Takes fragments and a merge history, and merges the fragments up until
    the threshold. Returns a new numpy array that is a "final" segmentation.

    Args:
        fragments (np.ndarray): Fragments that have already been run through
            waterz
        merge_history (Path): Result of running waterz on fragments with pretty
            high threshold
        threshold (float): The score threshold to merge until. If it is bigger
            than the largest in the merge history, nothing more will merge.

    Returns:
        np.ndarray: A segmentation coming from merging the fragments according
            to the merge history
    """
    with open(merge_history) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["a"])
            b = int(row["b"])
            c = int(row["c"])
            score = float(row["score"])
            if score <= threshold:
                fragments[fragments == a] = c
                fragments[fragments == b] = c

    return fragments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("seg_thresholds", nargs="+", type=float)
    args = parser.parse_args()

    seg_thresholds = args.seg_thresholds

    data_path = Path(args.data_path)
    zarr_path = data_path / "data.zarr"
    assert zarr_path.is_dir()
    zarr_root = zarr.open(zarr_path)
    raw_data = zarr_root["phase"][:]
    pred_affs = zarr_root["pred_affs"][:]
    pred_affs = np.mean(pred_affs, axis=0)
    fragments = zarr_root["fragments"][:]

    viewer = napari.Viewer()

    viewer.add_image(raw_data, name="phase")
    viewer.add_image(pred_affs, name="affinities")
    viewer.add_labels(fragments, name="fragments")

    merge_history = data_path / "merge_history.csv"
    for threshold in seg_thresholds:
        seg = agglomerate(fragments.copy(), merge_history, threshold)
        viewer.add_labels(seg, name=f"seg_{threshold}")

    viewer.grid.enabled = True
    viewer.grid.shape = (1, 3 + len(seg_thresholds))
    napari.run()
