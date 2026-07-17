import matplotlib.pyplot as plt
import numpy as np

conditions = [
    "Baseline",
    "- Cohesion",
    "- Drift",
    "- All",
]

te = [0.662, 0.649, 0.302, 0.308]
tf = [0.706, 0.693, 0.341, 0.342]
er = [0.842, 0.831, 0.521, 0.527]

metrics = [
    ("Target Effectiveness", te),
    ("Track Fraction", tf),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

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
    "NC28.1-Fl2mSiH2B Solver Ablation",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    r"C:\Users\hillj\Documents\mhat\experiments\evaluation\NC281-Fl2mSiH2B\03_nuclei\solver_ablation_results_nc281.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved solver_ablation_results_nc281.png")
