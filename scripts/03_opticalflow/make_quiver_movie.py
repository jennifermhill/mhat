"""
Generate reliability-masked quiver plots for each frame and a single z slice and compile into a movie.

Uses matplotlib quiver rendering (matching example_analysis_script.ipynb style).
Slower than the OpenCV version (make_quiver_movie.py) but produces different visual output.

Usage:
    conda activate opticalflow3D
    python make_quiver_movie_mpl.py movie_config.toml
"""

import argparse

import dask.array as da
import numpy as np
import toml
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import colorcet as cc
import cv2


def main(config, data_dir: Path):

    xy_scale = config['xy_scale']
    t_scale = config['t_scale']
    t_start = config.get('t_start', 0)
    t_end = config.get('t_end', None)

    conf_per = config['confidence_percentile']

    z_slice = config['z_slice']
    gap = config['gap']

    fps = config['fps']

    output_path = data_dir.parent / f"quiver_movie_z{z_slice}_conf{conf_per}_gap{gap}.mp4"

    colormap = cc.m_CET_C8

    # Determine canvas size from first frame to set up consistent figure dimensions
    flow = da.from_zarr(data_dir / "flow_raw")
    conf = da.from_zarr(data_dir / "confidence")
    T, Z, Y, X, _ = flow.shape
    flow_slice = flow[:, z_slice, :, :, :] # [T, Y, X, C]
    conf_slice = conf[:, z_slice, :, :] # [T, Y, X]

    # Physical coordinate grids
    x = np.linspace(0, X - 1, X) * xy_scale
    y = np.linspace(0, Y - 1, Y) * xy_scale

    X_2d, Y_2d = np.meshgrid(x, y)

    # Render one frame to determine output image size
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.max(y), np.min(y))
    ax.set_aspect('equal')
    plt.axis('off')
    fig.set_facecolor("black")
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame_arr = np.asarray(buf)
    canvas_h, canvas_w = frame_arr.shape[:2]
    plt.close(fig)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))

    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)

    if t_end is None:
        t_end = T - 1

    for frame in range(t_start, t_end + 1):
        print(f'Processing frame {frame}...')

        flow_frame = flow_slice[frame, ...].compute() # [Y, X, C]
        conf_frame = conf_slice[frame, ...].compute() # [Y, X]

        # --- Confidence mask ---
        conf_thresh = np.percentile(conf_frame, conf_per)
        conf_mask = conf_frame > conf_thresh

        # --- Load velocity data ---
        # channel order is z, y, x if 3D flow, and y, x if 2D flow
        vx = flow_frame[..., -1].astype(float) # [Y, X]
        vy = flow_frame[..., -2].astype(float) # [Y, X]

        # Apply confidence mask and convert to physical units
        for v in (vx, vy):
            v *= conf_mask
            v[v == 0] = np.nan
            v *= xy_scale / t_scale

        # Direction for coloring
        theta = np.arctan2(vy, vx)

        # Subsample
        vx_sub = vx[::gap, ::gap]
        vy_sub = vy[::gap, ::gap]
        theta_sub = theta[::gap, ::gap]

        thetaPlot = norm(theta_sub).flatten()

        # --- Matplotlib quiver plot (matching notebook style) ---
        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(10)

        ax.quiver(
            X_2d[::gap, ::gap], Y_2d[::gap, ::gap],
            vx_sub, vy_sub,
            color=colormap(thetaPlot),
            scale=1/10, angles='xy', scale_units='xy', units='xy', headwidth=4,
        )

        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.max(y), np.min(y))
        ax.set_aspect('equal')
        plt.axis('off')
        fig.set_facecolor("black")

        # Render figure to array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # drop alpha
        img_bgr = img[:, :, ::-1]  # RGB -> BGR for OpenCV
        plt.close(fig)

        writer.write(img_bgr)

    writer.release()
    print(f'Movie saved to {output_path}')
    print('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    input_base_dir = Path(config['input_base_dir'])
    experiment: str = config['experiment']
    dataset: str = config['dataset']
    exp_uid: str = config['exp_uid']
    assert input_base_dir.is_dir()

    if config["flow_result"] == "2d":
        # data_dir = input_base_dir / "opticalflow" / experiment / dataset / "opticalflow_2d" / exp_uid / "flow.zarr"
        raise ValueError("2D flow visualization not implemented yet (needs confidence array). Please set flow_result to '3d' or 'lk'.")
    elif config["flow_result"] == "3d":
        data_dir = input_base_dir / "opticalflow" / experiment / dataset / "opticalflow_3d" / exp_uid / "flow.zarr"
    elif config["flow_result"] == "lk":
        data_dir = input_base_dir / "opticalflow" / experiment / dataset / "opticalflow_lucaskanade" / exp_uid / "flow.zarr"
    else:
        raise ValueError(f"Invalid flow_result value: {config['flow_result']}. Must be one of '2d', '3d', or 'lk'.")
    print(f"Loading data from {data_dir}")
    assert data_dir.is_dir()

    main(config, data_dir)