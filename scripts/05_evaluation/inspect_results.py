# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 09-tracking
#     language: python
#     name: python3
# ---

# %%
# %reload_ext autoreload
# %autoreload 2
from pathlib import Path

import pandas as pd
import toml
from itables import init_notebook_mode
from mhat.evaluation.eval_io import (
    get_cell_lengths,
    load_segmentation_results,
    load_tracking_results,
)

# %%
init_notebook_mode(all_interactive=True)


# %%
experiments = {"mhat-validation": "2024-08-15_23-13-04"}

# %%
config: dict[str, str] = toml.load("eval_config.toml")
exp_name = config["exp_name"]
config

# %%
results_base_dir = Path(config["output_base_dir"])
ds_dir = results_base_dir / config["dataset"]

# %%
per_frame_seg_results = load_segmentation_results(ds_dir, exp_name)
per_frame_seg_results

# %%
per_video_seg_results = (
    per_frame_seg_results.groupby("video", as_index=True).sum().drop(columns=["frame"])
)
per_video_seg_results


# %%
def normalize_results(df, norm_columns, gt_column):
    # Errors per 1000 GT nodes
    df = df.copy()
    for column in norm_columns:
        df[column] = df[column] / df[gt_column] * 1000
    return df


columns = ["TP", "FP", "FN", "merge", "split"]
norm_seg_results = normalize_results(per_video_seg_results.copy(), columns, "GT")
norm_seg_results

# %%
norm_seg_results.mean()

# %%
import matplotlib.pyplot as plt

plt.scatter(norm_seg_results["TP"], norm_seg_results["max_cell_length"])
plt.xlabel("True Positives Per 1000")
plt.ylabel("Max Cell Length")

# %%
input_base_dir = Path(config["input_base_dir"]) / config["dataset"]
cell_lengths_dict = get_cell_lengths(input_base_dir)
norm_seg_results["max_cell_length"] = pd.Series(cell_lengths_dict)
norm_seg_results

# %%
from mhat.visualization.napari_utils import view_run

# %%
video = 9
vid_dir = input_base_dir / str(video)
view_run(vid_dir, exp_name, "with_area")

# %%
tracking_results = load_tracking_results(ds_dir, exp_name)
tracking_results

# %%
tracking_results["tp_edges"] = (
    tracking_results["gt_edges"] - tracking_results["fn_edges"]
)

# %%
columns = ["tp_edges", "fn_edges", "fp_edges"]
norm_tra_results = normalize_results(tracking_results, columns, "gt_edges")
norm_tra_results[columns].mean() / 1000

# %%
tracking_results["gt_div"] = (
    tracking_results["tp_div_fb0"] + tracking_results["fn_div_fb0"]
)
tracking_results

# %%
column_templates = ["tp_div_fb{}", "fp_div_fb{}", "fn_div_fb{}"]
frame_buffers = [0, 1, 2]
all_columns = []
for buff in frame_buffers:
    all_columns.extend(map(lambda x: x.format(buff), column_templates))
all_columns


# %%
norm_div_results = normalize_results(tracking_results, all_columns, "gt_div")
norm_div_results[all_columns].mean() / 1000

# %%
div_results_per_buffer = {}

# %%

buff = 1
tracking_results["div_precision"] = tracking_results[f"tp_div_fb{buff}"] / (
    tracking_results[f"tp_div_fb{buff}"] + tracking_results[f"fp_div_fb{buff}"]
)
tracking_results["div_recall"] = tracking_results[f"tp_div_fb{buff}"] / (
    tracking_results[f"tp_div_fb{buff}"] + tracking_results[f"fn_div_fb{buff}"]
)
div_results_per_buffer[buff] = tracking_results[["div_precision", "div_recall"]].mean()

# %%
div_results_per_buffer

# %%

fig, ax = plt.subplots()
for metric in ["div_precision", "div_recall"]:
    x = [0, 1, 2]
    y = [div_results_per_buffer[buff][metric] for buff in x]
    ax.plot(x, y, label=metric)
    ax.legend()
    ax.set_xlabel("Frame Offset Allowed")
    ax.set_xticks(x)
fig

# %%
all_results = tracking_results.join(per_video_seg_results, on="video")
plt.scatter(all_results["div_f1_fb2"], all_results["max_cell_length"])
plt.xlabel("Division F1 Score at Allowed Frame Offset 2")
plt.ylabel("Max Cell Length")

# %%
all_results["div_f1_fb2"]

# %%
