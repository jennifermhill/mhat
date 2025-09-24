import argparse
import csv
from pathlib import Path

import numpy as np
import waterz # type: ignore
import zarr
from scipy.ndimage import label
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed

from mhat.segmentation.voronoi_otsu import voronoi_otsu_labeling
from mhat.segmentation.affinities import compute_affinities, compute_fluorescent_affinities

def generate_fragments(data_zarr: Path, output_zarr: Path):
    zarr_root = zarr.open(data_zarr, "r+")

    raw_data = zarr_root

    T, C, Z, Y, X = raw_data.shape

    output_root = zarr.open(output_zarr, "a")
    output_root.create_dataset(
        "fragments", shape=(T, C, Z, Y, X), chunks=(1, 1, 1, Y, X), dtype=np.uint64, overwrite=True
    )

    for channel in range(C):
        for tp in range(T):
            print(f"Processing channel {channel}, frame {tp}")
            frame = raw_data[tp, channel]
            labels = voronoi_otsu_labeling(frame, spot_sigma=0, outline_sigma=0.5)

            output_root['fragments'][tp, channel] = labels


def generate_fluorescent_affinities(data_zarr: Path, output_zarr: Path):
    zarr_root = zarr.open(data_zarr, "r+")

    raw_data = zarr_root

    T, C, Z, Y, X = raw_data.shape

    neighborhood = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    output_root = zarr.open(output_zarr, "a")
    output_root.create_dataset(
        "affinities", shape=(T, C, 3, Z, Y, X), chunks=(1, 1, 1, 1, Y, X), dtype=np.float32, overwrite=True
    )

    affinities = np.zeros((T, C, 3, Z, Y, X), dtype=np.float32)
    for channel in range(C):
        for tp in range(T):
            print(f"Processing channel {channel}, frame {tp}")
            frame = raw_data[tp, channel]
            affinities[tp, channel] = compute_fluorescent_affinities(frame, neighborhood)

    # Normalize affinities to range [0, 1] and invert
    max_val = np.max(affinities)
    min_val = np.min(affinities)
    if max_val > 0:
        affinities = (affinities - min_val) / (max_val - min_val)

    affinities = 1.0 - affinities
    output_root['affinities'][:] = affinities


def get_segmentation(zarr_path, thresholds, outfile):
    zarr_root = zarr.open(zarr_path, "a")
    fragments = zarr_root["fragments"][:]
    affinities = zarr_root["affinities"][:]

    T, C, Z, Y, X = fragments.shape

    output_root.create_dataset(
        "segmentations", shape=(len(thresholds), T, C, Z, Y, X), chunks=(1, 1, 1, 1, Y, X), dtype=np.uint64, overwrite=True
    )

    # Process each timepoint and channel separately
    all_merge_history = []
    
    for t in range(T):
        for c in range(C):
            print(f"Processing timepoint {t}, channel {c}")
            
            fragments_3d = fragments[t, c]  # Shape: (Z, Y, X)
            affinities_3d = affinities[t, c]  # Shape: (3, Z, Y, X)
            
            ws_affs = affinities_3d.astype(np.float32)
            
            generator = waterz.agglomerate(
                affs=ws_affs,
                fragments=fragments_3d,
                thresholds=thresholds,
                return_merge_history=True,
            )

            for threshold in thresholds:
                segmentation, merge_history = next(generator)

                output_root['segmentations'][thresholds.index(threshold), t, c] = segmentation
                
                for row in merge_history:
                    row['threshold'] = threshold
                    row['timepoint'] = t
                    row['channel'] = c
                    all_merge_history.append(row)

    # Write all merge history to file
    fields = ["a", "b", "c", "score", "threshold", "timepoint", "channel"]

    with open(outfile, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_merge_history:
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="zarr with input data")
    parser.add_argument("-op", "--output_path", default=None)
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite existing affinity predictions",
    )
    args = parser.parse_args()
    data_zarr = Path(args.data_path)
    if args.output_path is not None:
        output_zarr = Path(args.output_path)
    else:
        output_zarr = data_zarr

    output_root = zarr.open(output_zarr, "a")
    if "fragments" not in output_root or args.overwrite:
        generate_fragments(data_zarr, output_zarr)

    if "affinities" not in output_root or args.overwrite:
        generate_fluorescent_affinities(data_zarr, output_zarr)

    thresholds = [0, 0.001, 0.0015, 0.002, 0.0025, 1]

    merge_history_file = output_zarr.parent / "merge_history.csv"
    get_segmentation(output_zarr, thresholds, merge_history_file)

    