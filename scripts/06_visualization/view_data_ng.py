import argparse
import logging
from pathlib import Path

import neuroglancer as ng
import neuroglancer.cli as ngcli
from mhat.data import (
    RawDataZarr,
    SegmentationZarr,
    add_data_args,
    add_segmentation_args,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)


def visualize_data(
    viewer_context,
    mhat_zarr: RawDataZarr,
    fov: int,
):
    channels = mhat_zarr.get_channels(fov)
    print(f"visualizing data for channels {channels}")

    for channel in channels:
        data = mhat_zarr.get_data(fov=fov, channel=channel)
        layer = ng.LocalVolume(
            data=data,
            dimensions=ng.CoordinateSpace(
                names=["t", "c^", "y", "x"],
                units=["s", "", "nm", "nm"],
                scales=[1, 1, 1, 1],
            ),
            volume_type="image",
        )
        name = "_".join(
            (
                mhat_zarr.dataset_name,
                mhat_zarr._fov_key(fov),
                mhat_zarr._channel_key(channel),
            )
        )
        # compute shader normalization ranges from one time point
        target_time = data.shape[0] // 2
        shader_min = 0.8 * data[target_time].min()
        shader_max = 1.2 * data[target_time].max()
        viewer_context.layers[name] = ng.ImageLayer(
            source=layer,
            shader_controls={"normalized": {"range": [shader_min, shader_max]}},
        )


def visualize_segmentation(
    viewer_context,
    seg_zarr: SegmentationZarr,
    fov: int,
):
    channels = seg_zarr.get_channels(fov)
    print(f"visualizing segmentations for channels {channels}")
    for channel in channels:
        seg_data = seg_zarr.get_data(fov=fov, channel=channel)
        layer = ng.LocalVolume(
            data=seg_data,
            dimensions=ng.CoordinateSpace(
                names=["t", "c^", "y", "x"],
                units=["s", "", "nm", "nm"],
                scales=[1, 1, 1, 1],
            ),
            volume_type="segmentation",
        )
        seg_name = "_".join(
            (
                seg_zarr.dataset_name,
                seg_zarr.zarr_base_group,
                seg_zarr._fov_key(fov),
                seg_zarr._channel_key(channel),
            )
        )
        viewer_context.layers.append(name=seg_name, layer=layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ngcli.add_server_arguments(parser)
    add_data_args(parser)
    add_segmentation_args(parser)
    parser.add_argument("--cluster", type=bool, default=False)
    args = parser.parse_args()
    # ngcli.handle_server_arguments(args)
    ng.set_server_bind_address(bind_address="0.0.0.0")
    if args.cluster:
        if args.data_base_path.startswith("/Volumes"):
            args.data_base_path = args.data_base_path.replace("/Volumes", "/nrs")
        if args.segmentation_base_path.startswith("/Volumes"):
            args.segmentation_base_path = args.segmentation_base_path.replace(
                "/Volumes", "/nrs"
            )
    base_path = Path(args.data_base_path)

    if not args.dataset_name:
        datasets = [s.name for s in base_path.iterdir() if s.is_dir()]
        print(datasets)
    else:
        datasets = [args.dataset_name]

    for ds_name in datasets:
        viewer = ng.Viewer()
        raw_zarr = RawDataZarr(base_path, ds_name, mode="r", store_type="flat")
        if not args.fov:
            fovs = raw_zarr.get_fovs()
            if not fovs:
                print(f"no fovs for dataset {ds_name}, continuing")
                continue
            fov = raw_zarr.get_fovs()[0]
        else:
            fov = args.fov
        with viewer.txn() as s:
            visualize_data(
                s,
                raw_zarr,
                fov=fov,
            )
        try:
            print("trying to get segmentation")
            print(f"{args.segmentation_base_path}_{ds_name}_{args.result_name}")
            seg_zarr = SegmentationZarr(
                args.segmentation_base_path, ds_name, args.result_name, mode="r"
            )
            if not args.result_name:
                result_names = seg_zarr.get_result_names()
                if len(result_names) == 0:
                    print("No results in segmentation zarr. Skipping.")
                else:
                    # take first result
                    seg_zarr.set_result_name(result_names[0])
            has_seg = seg_zarr.has_segmentation(
                fov=fov,
            )
        except ValueError as e:
            print(f"value error {e}")
            has_seg = False

        if has_seg:
            with viewer.txn() as s:
                visualize_segmentation(
                    s,
                    seg_zarr,
                    fov=fov,
                )
        with viewer.txn() as s:
            s.layout = "yz"
        url = str(viewer)
        print(url)
        # webbrowser.open_new(url)

        print("Press ENTER to go to next dataset")
        input()
