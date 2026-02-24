import argparse
import csv
from pathlib import Path

import gunpowder as gp
import numpy as np
import waterz
import zarr
from mhat.segmentation import MtlsdModel
from scipy.ndimage import label
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed


def predict(checkpoint, data_zarr, phase_group, output_zarr, data_type="synthetic"):
    phase = gp.ArrayKey("PHASE")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    voxel_size = gp.Coordinate((1, 1, 1))
    size = gp.Coordinate((3, 400, 36))
    output_shape = gp.Coordinate((1, 400, 36))

    input_size = size * voxel_size
    output_size = output_shape * voxel_size

    scan_request = gp.BatchRequest()

    scan_request.add(phase, input_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(pred_affs, output_size)

    context = (input_size - output_size) / 2

    phase_array_specs = gp.ArraySpec(
        voxel_size=voxel_size, dtype=np.uint16, interpolatable=True
    )

    phase_source = gp.ZarrSource(
        data_zarr, {phase: phase_group}, {phase: phase_array_specs}
    ) + gp.Pad(key=phase, size=context)

    with gp.build(phase_source):
        total_input_roi = phase_source.spec[phase].roi
        # total_output_roi = phase_source.spec[phase].roi
        total_output_roi = phase_source.spec[phase].roi.grow(-context, -context)

    lsd_shape = total_output_roi.get_shape() / voxel_size
    aff_shape = total_output_roi.get_shape() / voxel_size

    # generating the zarr file for saving
    zarrfile = zarr.open(output_zarr, "a")

    # zarrfile.create_dataset('phase', shape= total_input_roi.get_shape() / voxel_size)
    zarrfile.create_dataset(
        "pred_lsds", shape=(6, lsd_shape[0], lsd_shape[1], lsd_shape[2]), overwrite=True
    )
    zarrfile.create_dataset(
        "pred_affs", shape=(2, aff_shape[0], aff_shape[1], aff_shape[2]), overwrite=True
    )

    in_channels = 3
    num_fmaps = 16
    fmap_inc_factor = 2
    downsample_factors = [(2, 2), (2, 2)]
    num_levels = len(downsample_factors) + 1
    kernel_size_down = [[(3, 3), (3, 3)]] * num_levels
    kernel_size_up = [[(3, 3), (3, 3)]] * (num_levels - 1)
    constant_upsample = True

    model = MtlsdModel(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        constant_upsample=constant_upsample,
    )

    model.eval()

    pipeline = phase_source

    if data_type == "synthetic":
        print("Scaling by 1/65535")
        scale_factor = 1 / 65535
    elif data_type == "real":
        print("Scaling by 1/3000")
        scale_factor = 1 / 3000
    else:
        raise ValueError(f"data type must be synthetic or real, got {data_type}")
    pipeline += gp.Normalize(phase, factor=scale_factor)

    # pipeline += gp.Unsqueeze([phase])

    pipeline += gp.Stack(1)

    pipeline += gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs={"input": phase},
        outputs={0: pred_lsds, 1: pred_affs},
    )

    # pipeline += gp.Squeeze([phase])

    pipeline += gp.Squeeze([phase, pred_lsds, pred_affs])

    dataset_names = {
        pred_lsds: "pred_lsds",
        pred_affs: "pred_affs",
    }

    pipeline += gp.ZarrWrite(
        dataset_names=dataset_names,
        output_dir=output_zarr.parent,
        output_filename=output_zarr.name,
    )

    pipeline += gp.Scan(scan_request)

    predict_request = gp.BatchRequest()

    predict_request[phase] = total_input_roi
    predict_request[pred_lsds] = total_output_roi
    predict_request[pred_affs] = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    # print(
    #     f"\tphase: {batch[phase].data}, \tPred LSDS: {batch[pred_lsds].data}, \tPred Affs: {batch[pred_affs].data}")

    # return batch[phase].data, batch[pred_lsds].data, batch[pred_affs].data


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
    affinities = zarr_root["pred_affs"][:]

    fragments = watershed_from_affinities(affinities)
    zarr_root["fragments"] = fragments
    zarr_root["fragments"].attrs["resolution"] = (1, 1, 1)
    thresholds = [threshold]

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

    fields = ["a", "b", "c", "cost"]

    with open(outfile, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in merge_history:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="zarr with input data")
    parser.add_argument("-op", "--output_path", default=None)
    parser.add_argument("-dg", "--data_group", default="phase")
    parser.add_argument(
        "-dt", "--data_type", default="synthetic", help="synthetic or real"
    )
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
    phase_file = args.data_group
    checkpoint = "/groups/funke/home/sistaa/code/SyMBac/model_checkpoint_100000"

    output_root = zarr.open(output_zarr)
    if "pred_affs" not in output_root or args.overwrite:
        predict(checkpoint, data_zarr, phase_file, output_zarr, args.data_type)

    threshold = 0.5

    merge_history_file = output_zarr.parent / "merge_history.csv"
    get_segmentation(output_zarr, threshold, merge_history_file)
