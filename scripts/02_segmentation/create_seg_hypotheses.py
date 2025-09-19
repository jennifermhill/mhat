import argparse
import csv
from pathlib import Path

import numpy as np
import waterz
import zarr
from scipy.ndimage import label
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed

from mhat.segmentation.voronoi_otsu import voronoi_otsu_labeling
from mhat.segmentation.affinities import compute_affinities

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
            labels = voronoi_otsu_labeling(frame, spot_sigma=2, outline_sigma=1)

            output_root['fragments'][tp, channel] = labels

def generate_affinities(output_zarr: Path):
    output_root = zarr.open(output_zarr, "a")
    raw_data = output_root['fragments']

    T, C, Z, Y, X = raw_data.shape

    neighborhood = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    output_root.create_dataset(
        "affinities", shape=(T, C, 3, Z, Y, X), chunks=(1, 1, 1, 1, Y, X), dtype=np.int32, overwrite=True
    )

    for channel in range(C):
        for tp in range(T):
            print(f"Processing channel {channel}, frame {tp}")
            frame = raw_data[tp, channel]
            affinities = compute_affinities(frame, neighborhood)
            output_root['affinities'][tp, channel] = affinities


def watershed_from_boundary_distance(
    boundary_distances, boundary_mask, id_offset=0, min_seed_distance=10
):
    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=boundary_mask
    )

    ret = (fragments.astype(np.uint64), n + id_offset)

    return ret


def watershed_from_affinities(
    affs, max_affinity_value=1.0, id_offset=0, min_seed_distance=3
):
    mean_affs = 0.5 * (affs[0] + affs[1])

    boundary_mask = mean_affs > 0.5 * max_affinity_value

    fragments = np.zeros(mean_affs.shape, dtype=np.uint64)

    for time in range(0, affs.shape[1]):
        boundary_distances = distance_transform_edt(boundary_mask[time])

        frags, id_offset = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask[time],
            id_offset=id_offset,
            min_seed_distance=min_seed_distance,
        )
        fragments[time] = frags

    return fragments

def get_segmentation(zarr_path, threshold, outfile):
    zarr_root = zarr.open(zarr_path, "a")
    fragments = zarr_root["fragments"][:]
    affinities = zarr_root["affinities"][:]
    thresholds = [threshold]

    # fragments = watershed_from_affinities(affinities)
    # zarr_root["fragments"] = fragments
    # zarr_root["fragments"].attrs["resolution"] = (1, 1, 1)

    ws_affs = np.stack(
        [np.zeros_like(affinities[0]), affinities[0], affinities[1]]
    ).astype(np.float32)

    generator = waterz.agglomerate(
        affs=ws_affs,
        fragments=fragments,
        thresholds=thresholds,
        return_merge_history=True,
    )

    segmentation, merge_history = next(generator)

    fields = ["a", "b", "c", "score"]

    with open(outfile, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in merge_history:
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
        generate_affinities(output_zarr)

    threshold = 0.5

    merge_history_file = output_zarr.parent / "merge_history.csv"
    get_segmentation(output_zarr, threshold, merge_history_file)

    