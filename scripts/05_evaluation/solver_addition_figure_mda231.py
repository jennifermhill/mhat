import matplotlib.pyplot as plt
import numpy as np

conditions = [
    "None",
    "+ Coh/Adh",
    "+ Volume",
    "+ Intensity",
    "+ Drift",
    "Full",
]

tra = [0.797, 0.804, 0.826, 0.815, 0.842, 0.881]
det = [0.828, 0.838, 0.846, 0.834, 0.852, 0.886]
lnk = [0.570, 0.558, 0.680, 0.677, 0.763, 0.845]

metrics = [
    ("TRA", tra),
    ("DET", det),
    ("LNK", lnk),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors = ["#9C27B0", "#E91E63", "#FF9800", "#64B5F6", "#4CAF50", "#2196F3"]

for ax, (title, values) in zip(axes, metrics):
    x = np.arange(len(conditions))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
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
    "Fluo-C3DL-MDA231 Solver Addition",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    r"C:\Users\hillj\Documents\mhat\experiments\evaluation\Fluo-C3DL-MDA231\01_cells\solver_addition_results_mda231.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved solver_addition_results_mda231.png")
