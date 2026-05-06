import matplotlib.pyplot as plt
import numpy as np

# Per-cost color (Wong palette, consistent across all addition/ablation plots)
COLOR = {
    "None": "#999999",        # grey
    "+ Cohesion": "#CC79A7",  # reddish purple
    "+ Volume": "#E69F00",    # orange
    "+ Intensity": "#56B4E9", # sky blue
    "+ Drift": "#009E73",     # bluish green
    "+ Curvature": "#D55E00", # vermillion
    "Full": "#0072B2",        # blue
}

# Best-of-3-sweep per addition bar from batch 2026-05-01_10-02-14:
#   + Cohesion+Adhesion: w20k
#   + Volume:            w1300
#   + Intensity:         w6
#   + Drift:             w100
#   + Curvature:         w100
#   Full = B2R1 (drift + cohesion + adhesion)
# None pinned far-left, Full pinned far-right; middle bars sorted by Track Fraction ascending.
rows = [
    # (label,        TE,    TF,    Purity)
    ("None",         0.320, 0.347, 0.350),
    ("+ Cohesion",   0.409, 0.433, 0.464),
    ("+ Volume",     0.553, 0.571, 0.689),
    ("+ Intensity",  0.598, 0.621, 0.689),
    ("+ Curvature",  0.824, 0.819, 0.919),
    ("+ Drift",      0.828, 0.839, 0.923),
    ("Full",         0.827, 0.838, 0.923),
]

conditions = [r[0] for r in rows]
te = [r[1] for r in rows]
tf = [r[2] for r in rows]
purity = [r[3] for r in rows]
colors = [COLOR[c] for c in conditions]

metrics = [
    ("Target Effectiveness", te),
    ("Track Fraction", tf),
    ("Track Purity", purity),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (title, values) in zip(axes, metrics):
    x = np.arange(len(conditions))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.axhline(y=values[-1], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

fig.suptitle(
    "NC28.1 sparse-label Solver Addition",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    "/groups/sgro/sgrolab/jennifer/mhat/experiments/evaluation/NC281-sparse-label/01_nuclei_denoised/solver_addition_results_nc281-sparse.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved solver_addition_results_nc281-sparse.png")
