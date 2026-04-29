import matplotlib.pyplot as plt
import numpy as np

conditions = [
    "Baseline",
    "- Intensity",
    "- Volume",
    "- Drift",
    "- Coh/Adh",
    "- All",
]

tra = [0.881, 0.879, 0.864, 0.863, 0.860, 0.788]
det = [0.886, 0.887, 0.874, 0.880, 0.864, 0.823]
lnk = [0.845, 0.821, 0.788, 0.737, 0.833, 0.531]

metrics = [
    ("TRA", tra),
    ("DET", det),
    ("LNK", lnk),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors = ["#2196F3", "#64B5F6", "#FF9800", "#E91E63", "#4CAF50", "#9C27B0"]

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
    "Fluo-C3DL-MDA231 Solver Ablation",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    r"C:\Users\hillj\Documents\mhat\experiments\evaluation\Fluo-C3DL-MDA231\01_cells\solver_ablation_results_mda231.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved solver_ablation_results_mda231.png")
