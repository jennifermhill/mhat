"""MHAT vs CTC leaderboard top-3 (per metric) on Fluo-C3DL-MDA231 (02_cells / video 02).

Three-panel bar chart: one panel per CTC metric (TRA, DET, SEG), with the metric
named on that panel's y-axis. Each panel shows MHAT plus the three best CTC
competitors *for that metric*, ordered highest-to-lowest; the top-3 set therefore
differs between panels. MHAT is light blue and every competitor is gray, so the
x tick labels -- not color -- identify the methods.

Data provenance
---------------
* MHAT: loaded live from the 02_cells baseline eval. The baseline uid and eval
  directory come from an ablation TOML (default: merge_ablation.toml
  [mda231_02cells].conditions.baseline) so this figure tracks the same baseline
  as the other 02_cells figures. TRA/DET/SEG read from that run's
  track_metrics.json so a re-eval (e.g. the 2026-07-20 SEG fix) is picked up.
* CTC methods: public benchmark spreadsheets (snapshot 2025-08-15),
  CellTrackingBenchmark.xlsx / CellSegmentationBenchmark.xlsx. Method labels are
  the CTC website labels. Values are the published leaderboard scores, i.e. the
  mean over videos 01+02 (note: MHAT is 02_cells / video 02 only, so the two are
  not on identical sequences -- this is an informal positioning against the field).
  TRA is from the Cell Tracking Benchmark; DET from the Cell Segmentation
  Benchmark; SEG pooled across both (identical where they overlap).

Usage:
    python scripts/07_plotting/leaderboard_comparison_mda231.py \
        [config] [--dataset mda231_02cells] [--output foo.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml

# The MHAT baseline pointer is read from an ablation TOML (below); only the CTC
# leaderboard values stay hardcoded -- they're external published scores.
DEFAULT_CONFIG = "configs/evaluation/merge_ablation.toml"
DEFAULT_DATASET = "mda231_02cells"


def baseline_eval_dir(config_path, dataset):
    """(eval_dir, baseline_uid) for the dataset's conditions.baseline."""
    cfg = toml.load(config_path)[dataset]
    eval_dir = (
        Path(cfg["eval_base_dir"])
        / cfg["experiment"]
        / cfg["dataset_dir"]
    )
    return eval_dir, cfg["conditions"]["baseline"]["tracking_uid"]

# --- CTC top-3 competitors per metric, MDA231 leaderboard averages (mean over
# videos 01+02; snapshot 2025-08-15). Ordered highest -> lowest within each metric.
CTC_TOP3 = {
    "TRA": [("KIT-GE (3)", 0.8845), ("KTH-SE (1)", 0.8821), ("LEID-NL", 0.8798)],
    "DET": [("CALT-US (*)", 0.9106), ("KTH-SE (1)", 0.9074), ("KIT-GE (3)", 0.9044)],
    "SEG": [("KIT-GE (3)", 0.7096), ("CALT-US (*)", 0.7095), ("MU-US (3*)", 0.7044)],
}
METRICS = ["TRA", "DET", "SEG"]

# MHAT is highlighted in light blue; every competing method is the same gray.
# Methods are identified by the x tick labels, not by color.
MHAT_COLOR = "#56B4E9"   # sky blue (Wong colorblind-safe palette)
OTHER_COLOR = "#B0B0B0"  # gray


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                    help="ablation TOML providing the MHAT baseline uid")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset key whose conditions.baseline uid to use")
    ap.add_argument("--output", default=None, help="Override output PNG path")
    args = ap.parse_args()

    eval_dir, baseline_uid = baseline_eval_dir(args.config, args.dataset)
    mhat = json.loads((eval_dir / baseline_uid / "track_metrics.json")
                      .read_text())["CTCMetrics"]

    # Per panel: MHAT first, then that metric's top-3 competitors (already sorted).
    groups = {m: [("MHAT", mhat[m])] + CTC_TOP3[m] for m in METRICS}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(9, 4), sharey=True)
    for ax, m in zip(axes, METRICS):
        names = [name for name, _ in groups[m]]
        heights = [v for _, v in groups[m]]
        colors = [MHAT_COLOR if name == "MHAT" else OTHER_COLOR for name in names]

        x = np.arange(len(names))
        bars = ax.bar(x, heights, 0.72, color=colors)
        for bar, v in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
        for tick, name in zip(ax.get_xticklabels(), names):
            if name == "MHAT":
                tick.set_fontweight("bold")
        ax.set_ylabel(m, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="y", labelleft=True)

        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path(args.output or eval_dir / "leaderboard_comparison_mda231_02.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
