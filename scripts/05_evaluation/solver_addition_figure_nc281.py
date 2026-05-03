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

# + Curvature from nc281curvadd_2026-04-29_16-49-06 R3 (curv_w=100, curv_c=-800).
# Bars sorted by Track Fraction ascending.
rows = [
    # (label,        TE,    TF)
    ("None",         0.297, 0.336),
    ("+ Cohesion",   0.330, 0.371),
    ("+ Volume",     0.401, 0.457),
    ("+ Intensity",  0.427, 0.462),
    ("+ Curvature",  0.616, 0.649),
    ("+ Drift",      0.659, 0.705),
    ("Full",         0.662, 0.706),
]

conditions = [r[0] for r in rows]
te = [r[1] for r in rows]
tf = [r[2] for r in rows]
colors = [COLOR[c] for c in conditions]

metrics = [
    ("Target Effectiveness", te),
    ("Track Fraction", tf),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

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
    "NC28.1-Fl2mSiH2B Solver Addition",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    r"C:\Users\hillj\Documents\mhat\experiments\evaluation\NC281-Fl2mSiH2B\03_nuclei\solver_addition_results_nc281.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved solver_addition_results_nc281.png")
