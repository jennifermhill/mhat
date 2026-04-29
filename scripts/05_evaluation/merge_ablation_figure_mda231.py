import matplotlib.pyplot as plt
import numpy as np

conditions = [
    "Baseline",
    "- Coh/Adh",
    "- Affinities",
    "- Merges",
]

tra = [0.881, 0.860, 0.856, 0.836]
det = [0.886, 0.864, 0.860, 0.841]
lnk = [0.845, 0.833, 0.827, 0.800]

metrics = [
    ("TRA", tra),
    ("DET", det),
    ("LNK", lnk),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors = ["#2196F3", "#64B5F6", "#FF9800", "#4CAF50"]

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
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Fluo-C3DL-MDA231 Merge Ablation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(
    r"C:\Users\hillj\Documents\mhat\experiments\evaluation\Fluo-C3DL-MDA231\01_cells\merge_ablation_results_mda231.png",
    dpi=150, bbox_inches="tight"
)
print("Saved merge_ablation_results_mda231.png")
