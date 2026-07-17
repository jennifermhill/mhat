import matplotlib.pyplot as plt
import numpy as np

# Ablation values from batch 2026-05-01_10-02-14.
# Baseline is Full = B2R1 (drift_w=100, drift_c=-2000, cohesion -500/450, adhesion -100/50).
conditions = [
    "Baseline",
    "- Cohesion",
    "- Drift",
    "- All",
]

te = [0.827, 0.826, 0.407, 0.472]
tf = [0.838, 0.837, 0.435, 0.494]
purity = [0.923, 0.923, 0.453, 0.514]

metrics = [
    ("Target Effectiveness", te),
    ("Track Fraction", tf),
    ("Track Purity", purity),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Wong colorblind-friendly palette (per-concept consistent across figures)
colors = [
    "#0072B2",  # Baseline   — blue
    "#CC79A7",  # - Cohesion — reddish purple
    "#009E73",  # - Drift    — bluish green
    "#999999",  # - All      — grey
]

for ax, (title, values) in zip(axes, metrics):
    x = np.arange(len(conditions))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.axhline(y=values[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
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
    "NC28.1 sparse-label Solver Ablation",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    "/groups/sgro/sgrolab/jennifer/mhat/experiments/evaluation/NC281-sparse-label/01_nuclei_denoised/solver_ablation_results_nc281-sparse.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved solver_ablation_results_nc281-sparse.png")
