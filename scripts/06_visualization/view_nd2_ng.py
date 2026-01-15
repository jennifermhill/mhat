import argparse
import logging
from pathlib import Path
import json

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
        5.0 * vec3(
            %f*toNormalized(getDataValue(1)),
            %f*toNormalized(getDataValue(2)),
            %f*toNormalized(getDataValue(3)))
        );
}"""

grey_shader_code = """
void main() {
    emitGrayscale(
        5.0 * toNormalized(getDataValue(%i))
    );
}"""
def visualize_phase(
    viewer_context,
    data,
    name,
):  
    print("visualizing phase")
    layer = ng.LocalVolume(
        data=data,
        dimensions=ng.CoordinateSpace(
            names=["t", "c^", "y", "x"],
            units=["s", "", "nm", "nm"],
            scales=[1, 1, 1, 1],
        ),
        volume_type="image",
    )
    print("Done adding phase")
    # compute shader normalization ranges from one time point
    target_time = data.shape[0] // 2
    shader_min = 0.8 * data[target_time, 0].min()
    shader_max = 1.2 * data[target_time, 0].max()
    print(shader_min, shader_max)

    shader = grey_shader_code % 0
    viewer_context.layers[name] = ng.ImageLayer(
        source=layer,
        shader=shader,
        shader_controls={"normalized": {"range": [shader_min, shader_max]}},
    )

def visualize_fluor(
    viewer_context,
    data,
    name,
):
    print("visualizing fluor")
    # compute shader normalization ranges from one time point
    target_time = data.shape[0] // 2
    shader_vals = [6.0, 1.0, 6.0]  # hard coding to save time
    # total_max = np.max(data[target_time, 1:])
    # print(total_max)
    # for channel in range(1, data.shape[1]):
    #     scale_factor = total_max / np.max(data[target_time, channel])
    #     shader_vals.append(scale_factor)
    #     shader_vals.append(channel)
        
    shader = rgb_shader_code % tuple(shader_vals) 
    print(shader)
    layer = ng.LocalVolume(
        data=data,
        dimensions=ng.CoordinateSpace(
            names=["t", "c^", "y", "x"],
            units=["s", "", "nm", "nm"],
            scales=[1, 1, 1, 1],
        ),
        volume_type="image",
    )
    viewer_context.layers[name] = ng.ImageLayer(
        source=layer,
        shader=shader,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_zarr")
    parser.add_argument("group")
    ngcli.add_server_arguments(parser)
    args = parser.parse_args()
    ng.set_server_bind_address(bind_address="0.0.0.0")
    base_path = Path(args.path_to_zarr)

    viewer = ng.Viewer()
    root = zarr.open(base_path)
    group = args.group
    data = root[group]
    print("loaded data")
    with viewer.txn() as s:
        visualize_phase(s, data, "phase")
        visualize_fluor(s, data, "fluor")
    url = str(viewer)
    print(url)
    input()
