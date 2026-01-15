import argparse
import logging
from pathlib import Path

import neuroglancer as ng
import neuroglancer.cli as ngcli
import numpy as np
import zarr

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)

rgb_shader_code = """
void main() {
    emitRGB(
        %f*vec3(
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)))
        );
}"""


def visualize_lsds(
    viewer_context,
    data,
    name,
):
    shader = rgb_shader_code % (1.0, 0, 1, 2)
    for i in range(0, data.shape[1], 3):
        end_channel = min(i + 3, data.shape[1])
        current_channels = data[:, i:end_channel]
        layer = ng.LocalVolume(
            data=current_channels,
            dimensions=ng.CoordinateSpace(
                names=["c^", "t", "y", "x"],
                units=["", "s", "nm", "nm"],
                scales=[1, 1, 1, 1],
            ),
            volume_type="image",
        )
        viewer_context.layers[name + f"_{i}-{end_channel}"] = ng.ImageLayer(
            source=layer,
            shader=shader,
        )


def visualize_image(
    viewer_context,
    data,
    name,
):
    layer = ng.LocalVolume(
        data=data,
        dimensions=ng.CoordinateSpace(
            names=["t", "y", "x"],
            units=["s", "nm", "nm"],
            scales=[1, 1, 1],
        ),
        volume_type="image",
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
    data,
    name,
):
    data = data.astype(np.uint64)
    layer = ng.LocalVolume(
        data=data,
        dimensions=ng.CoordinateSpace(
            names=["t", "y", "x"],
            units=["s", "nm", "nm"],
            scales=[1, 1, 1],
        ),
        volume_type="segmentation",
    )

    viewer_context.layers.append(name=name, layer=layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_zarr",
    )
    parser.add_argument("-g", "--groups", nargs="+")
    ngcli.add_server_arguments(parser)
    args = parser.parse_args()
    ng.set_server_bind_address(bind_address="0.0.0.0")
    base_path = Path(args.path_to_zarr)

    viewer = ng.Viewer()
    root = zarr.open(base_path)
    for group in args.groups:
        print(group)
        print(root.keys())
        data = root[group]
        print(data.shape)
        if group == "mask":
            with viewer.txn() as s:
                print(s)
                visualize_segmentation(s, data, group)
        elif group in ["gt_affs", "affs_weights", "pred_affs"]:
            affs_y = data[0]
            affs_x = data[1]
            with viewer.txn() as s:
                print(s)
                visualize_image(
                    s,
                    affs_x,
                    group + "_x",
                )
                visualize_image(
                    s,
                    affs_y,
                    group + "_y",
                )

        elif group in ["gt_lsds", "pred_lsds"]:
            with viewer.txn() as s:
                visualize_lsds(
                    s,
                    data,
                    group,
                )
        elif group == "phase":
            with viewer.txn() as s:
                print(s)
                visualize_image(s, data, group)
        else:
            raise ValueError(f"Couldn't visualize group {group}")
    url = str(viewer)
    print(url)
    input()
