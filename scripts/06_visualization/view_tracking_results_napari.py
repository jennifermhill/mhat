import argparse
import logging
from pathlib import Path

import toml
from mhat.visualization import napari_utils

# _themes["dark"].font_size = "18pt"
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_config")
    parser.add_argument("vid_name")
    parser.add_argument("-s", "--start_frame", type=int, default=None)
    parser.add_argument("-e", "--end_frame", type=int, default=None)
    args = parser.parse_args()
    config = toml.load(args.eval_config)
    vid_name = args.vid_name

    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    dataset: str = config["dataset"]
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    input_data_dir = input_base_dir / dataset / vid_name
    assert input_data_dir.is_dir()

    exp_name: str = config["exp_name"]

    out_data_dir = output_base_dir / dataset / vid_name
    out_exp_dir = out_data_dir / exp_name

    napari_utils.view_run(
        input_data_dir,
        out_data_dir,
        exp_name,
        run_name="GOST",
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
