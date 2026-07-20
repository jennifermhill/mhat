"""MHAT vs CTC leaderboard top-3 (per metric) on Fluo-C3DL-MDA231 (02_cells / video 02).

Grouped bar chart: one x-group per CTC metric (TRA, DET, SEG). Each group shows
MHAT plus the three best CTC competitors *for that metric*, with the competitors
ordered highest-to-lowest. The top-3 set therefore differs between metrics.
Colors are fixed per method, so the legend maps color -> method across the whole
figure; bars touch within a group.

Data provenance
---------------
* MHAT: loaded live from the 02_cells baseline eval (exp_uid 2026-06-01_14-41-03),
  hand-tuned config transferred from 01_cells. TRA/DET/SEG read from
  track_metrics.json so a re-eval (e.g. the 2026-07-20 SEG fix) is picked up.
* CTC methods: public benchmark spreadsheets (snapshot 2025-08-15),
  CellTrackingBenchmark.xlsx / CellSegmentationBenchmark.xlsx. Method labels are
  the CTC website labels. Values are the published leaderboard scores, i.e. the
  mean over videos 01+02 (note: MHAT is 02_cells / video 02 only, so the two are
  not on identical sequences -- this is an informal positioning against the field).
  TRA is from the Cell Tracking Benchmark; DET from the Cell Segmentation
  Benchmark; SEG pooled across both (identical where they overlap).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- MHAT baseline eval (loaded live) -------------------------------------
MHAT_METRICS_JSON = Path(
    "Y:/jennifer/mhat/experiments/evaluation/Fluo-C3DL-MDA231/02_cells"
    "/2026-06-01_14-41-03/track_metrics.json"
)

# --- CTC top-3 competitors per metric, MDA231 leaderboard averages (mean over
# videos 01+02; snapshot 2025-08-15). Ordered highest -> lowest within each metric.
CTC_TOP3 = {
    "TRA": [("KIT-GE (3)", 0.8845), ("KTH-SE (1)", 0.8821), ("LEID-NL", 0.8798)],
    "DET": [("CALT-US (*)", 0.9106), ("KTH-SE (1)", 0.9074), ("KIT-GE (3)", 0.9044)],
    "SEG": [("KIT-GE (3)", 0.7096), ("CALT-US (*)", 0.7095), ("MU-US (3*)", 0.7044)],
}
METRICS = ["TRA", "DET", "SEG"]

# Fixed color per method (Wong colorblind-safe palette, project palette).
COLOR = {
    "MHAT":        "#0072B2",  # blue      (project MHAT color)
    "KTH-SE (1)":  "#E69F00",  # orange
    "KIT-GE (3)":  "#009E73",  # bluish green
    "LEID-NL":     "#CC79A7",  # reddish purple
    "CALT-US (*)": "#D55E00",  # vermillion
    "MU-US (3*)":  "#56B4E9",  # sky blue
}
# Legend order: MHAT first, then competitors as they first appear across metrics.
LEGEND_ORDER = ["MHAT", "KIT-GE (3)", "KTH-SE (1)", "LEID-NL", "CALT-US (*)", "MU-US (3*)"]


def main():
    mhat = json.loads(MHAT_METRICS_JSON.read_text())["CTCMetrics"]

    # Per group: MHAT first, then that metric's top-3 competitors (already sorted).
    groups = {m: [("MHAT", mhat[m])] + CTC_TOP3[m] for m in METRICS}
    n = 1 + 3  # bars per group

    x = np.arange(len(METRICS))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(9, 5))
    for slot in range(n):
        offset = (slot - (n - 1) / 2) * width
        xs = x + offset
        heights = [groups[m][slot][1] for m in METRICS]
        colors = [COLOR[groups[m][slot][0]] for m in METRICS]
        bars = ax.bar(xs, heights, width, color=colors,
                      edgecolor="white", linewidth=1.0)
        for bar, m in zip(bars, METRICS):
            name, v = groups[m][slot]
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                    rotation=90, ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0, 1.0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in LEGEND_ORDER]
    ax.legend(handles, LEGEND_ORDER, frameon=False, loc="upper center",
              ncol=len(LEGEND_ORDER), fontsize=8.5, bbox_to_anchor=(0.5, -0.08),
              columnspacing=1.2, handlelength=1.2, handletextpad=0.5)

    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path(
        "Y:/jennifer/mhat/experiments/evaluation/Fluo-C3DL-MDA231/02_cells"
        "/leaderboard_comparison_mda231_02.png"
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
