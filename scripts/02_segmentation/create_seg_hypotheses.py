import argparse
import csv
from pathlib import Path

import numpy as np
import toml
import waterz # type: ignore
import zarr
import torch
import datetime
from scipy.ndimage import label
# from scipy.ndimage.filters import maximum_filter
# from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed
from skimage.filters import gaussian

from mhat.segmentation.threshold_labeling import threshold_labeling
from mhat.segmentation.cellpose import segment_with_cellpose
from mhat.segmentation.affinities import compute_affinities, compute_fluorescent_affinities
from mhat.utils import get_axes_metadata

def generate_fragments(data_zarr: Path, output_root, config):
    zarr_root = zarr.open(data_zarr, "r+")
    axes = get_axes_metadata(zarr_root)

    raw_data = zarr_root

    T, C, Z, Y, X = raw_data.shape

    output_root.create_dataset(
        "fragments", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype=np.uint32, overwrite=True
    )
    output_root['fragments'].attrs["axes"] = axes

    seg_method = config["seg_method"]
    if seg_method == 'cellpose':
        # Check for cuda availability
        print("Using Cellpose for segmentation.")
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU for Cellpose.")
            gpu = True
        else:
            print("CUDA is not available. Using CPU for Cellpose.")
            gpu = False
        cellpose_kwargs = config.get("cellpose_params", {})
    else:
        print(f"Using '{seg_method}' threshold segmentation method.")

    for tp in range(T):
        print(f"Processing frame {tp}")
        frame = raw_data[tp, 0]
        if seg_method == 'cellpose':
            labels = segment_with_cellpose(frame, gpu=gpu, **cellpose_kwargs)
        else:
            labels = threshold_labeling(
                frame,
                method=seg_method,
                spot_sigma=config["spot_sigma"],
                outline_sigma=config["outline_sigma"],
                threshold=config.get("threshold"),
                mad_k=config.get("mad_k", 9.0),
            )
        
        if tp != 0:
            labels[labels != 0] += max_node_id

        max_node_id = np.max(labels)

        output_root['fragments'][tp] = labels

def generate_fluorescent_affinities(data_zarr: Path, output_root, config):
    zarr_root = zarr.open(data_zarr, "r+")
    axes = get_axes_metadata(zarr_root)

    raw_data = zarr_root

    T, C, Z, Y, X = raw_data.shape

    neighborhood = config["neighborhood"]

    output_root.create_dataset(
        "affinities", shape=(T, 3, Z, Y, X), chunks=(1, 1, 1, Y, X), dtype=np.float32, overwrite=True
    )
    output_root['affinities'].attrs["axes"] = axes

    affinities = np.zeros((T, 3, Z, Y, X), dtype=np.float32)
    for tp in range(T):
        print(f"Processing frame {tp}")
        frame = raw_data[tp, 0]
        if config["smoothing"] != 0:
            smoothing_sigma = tuple(config["smoothing"])
            frame = gaussian(frame, sigma=smoothing_sigma) 
        affinities[tp] = compute_fluorescent_affinities(frame, neighborhood)

    # Normalize affinities to [0, 1] and invert
    max_val = np.max(affinities)
    min_val = np.min(affinities)
    if max_val > 0:
        affinities = (affinities - min_val) / (max_val - min_val)
    
    affinities = 1.0 - affinities

    output_root['affinities'][:] = affinities

# def watershed_from_boundary_distance(
#     boundary_distances, boundary_mask, id_offset=0, min_seed_distance=50
# ):
#     max_filtered = maximum_filter(boundary_distances, min_seed_distance)
#     maxima = max_filtered == boundary_distances
#     seeds, n = label(maxima)

#     print(f"Found {n} fragments")

#     if n == 0:
#         return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

#     seeds[seeds != 0] += id_offset

#     fragments = watershed(
#         boundary_distances.max() - boundary_distances, seeds, mask=boundary_mask
#     )

#     ret = (fragments.astype(np.uint64), n + id_offset)

#     return ret


# def watershed_from_affinities(
#     affs, max_affinity_value=1.0, id_offset=0, min_seed_distance=3
# ):
#     mean_affs = 0.33 * (affs[:, 0] + affs[:, 1] + affs[:, 2])

#     boundary_mask = mean_affs > 0.5 * max_affinity_value

#     fragments = np.zeros(mean_affs.shape, dtype=np.uint64)

#     for time in range(0, affs.shape[0]):
#         boundary_distances = distance_transform_edt(boundary_mask[time])

#         frags, id_offset = watershed_from_boundary_distance(
#             boundary_distances,
#             boundary_mask[time],
#             id_offset=id_offset,
#             min_seed_distance=min_seed_distance,
#         )
#         fragments[time] = frags

#     return fragments

def get_segmentation(output_root, thresholds, outfile, waterz_params):
    affinities = output_root["affinities"][:].astype(np.float32)
    fragments = output_root["fragments"][:]

    axes = get_axes_metadata(output_root["fragments"])

    T, Z, Y, X = fragments.shape

    output_root.create_dataset(
        "segmentations", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype=np.uint32, overwrite=True
    )
    output_root['segmentations'].attrs["axes"] = axes

    # Process each timepoint and channel separately
    all_merge_history = []
    
    for t in range(T):
        print(f"Processing timepoint {t}")

        fragments_3d = fragments[t]  # Shape: (Z, Y, X)
        affinities_3d = affinities[t]  # Shape: (3, Z, Y, X)

        ws_frags = fragments_3d.astype(np.uint64)
        ws_affs = affinities_3d.astype(np.float32)

        score_func = waterz_params.get("scoring_function", None)
        if score_func == "random":
            score_func = "Random<RegionGraphType>"
        elif score_func == "symmetric":
            score_func = (
                "Divide<"
                "Subtract<MaxSize<RegionGraphType>, MinSize<RegionGraphType>>,"
                "Add<MaxSize<RegionGraphType>, MinSize<RegionGraphType>>"
                ">"
            )
        else:
            score_func = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"
        
        generator = waterz.agglomerate(
            affs=ws_affs,
            fragments=ws_frags,
            thresholds=thresholds,
            scoring_function=score_func,
            return_merge_history=True,
        )

        segmentation, merge_history = next(generator)

        output_root['segmentations'][t] = segmentation
        
        for row in merge_history:
            row['cost'] = row.pop('score')  # Rename 'score' to 'cost'
            row['timepoint'] = t
            all_merge_history.append(row)

    # Write all merge history to file
    fields = ["a", "b", "c", "cost", "timepoint"]


    with open(outfile, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_merge_history:
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    experiment: str = config["experiment"]
    dataset: str = config["dataset"]
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    data_dir = input_base_dir / experiment / f"{dataset}.zarr"
    print(f"Loading data from {data_dir}")
    assert data_dir.is_dir()

    exp_uid = config.get("exp_uid")
    if not exp_uid:
        exp_uid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config["exp_uid"] = exp_uid

    output_dir = output_base_dir / experiment / dataset / exp_uid
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "config.toml", "w") as config_file:
        toml.dump(config, config_file)

    output_root = zarr.open(output_dir / "data.zarr", "a")
    if "fragments" not in output_root or config["overwrite"]:
        generate_fragments(data_dir, output_root, config["seg_params"])

    if "affinities" not in output_root or config["overwrite"]:
        generate_fluorescent_affinities(data_dir, output_root, config["affinity_params"])

    merge_history_file = output_dir / "merge_history.csv"

    if config.get("skip_merges", False):
        print("Skipping waterz agglomeration (skip_merges=True). Writing empty merge history.")
        fields = ["a", "b", "c", "cost", "timepoint"]
        with open(merge_history_file, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
    else:
        threshold = config["merge_thresholds"]
        get_segmentation(output_root, threshold, merge_history_file, config["waterz_params"])
