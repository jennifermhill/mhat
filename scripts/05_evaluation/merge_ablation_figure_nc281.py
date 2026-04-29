import matplotlib.pyplot as plt
import numpy as np

conditions = [
    "Baseline",
    "- Cohesion",
    "- Affinities",
    "- Merges",
]

te = [0.6616, 0.6486, 0.6443, 0.5987]
tf = [0.7057, 0.6934, 0.6930, 0.6448]
er = [0.8416, 0.8308, 0.8286, 0.8091]

metrics = [
    ("Target Effectiveness", te),
    ("Track Fraction", tf),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

colors = ["#2196F3", "#64B5F6", "#FF9800", "#4CAF50"]

for ax, (title, values) in zip(axes, metrics):
    x = np.arange(len(conditions))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.axhline(y=values[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("NC28.1-Fl2mSiH2B Merge Ablation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(
    r"C:\Users\hillj\Documents\mhat\experiments\evaluation\NC281-Fl2mSiH2B\03_nuclei\merge_ablation_results_nc281.png",
    dpi=150, bbox_inches="tight"
)
print("Saved merge_ablation_results_nc281.png")
