import matplotlib.pyplot as plt
import numpy as np

# Per-cost color (Wong palette, consistent across all addition/ablation plots)
COLOR = {
    "None": "#999999",        # grey
    "+ Coh/Adh": "#CC79A7",   # reddish purple
    "+ Volume": "#E69F00",    # orange
    "+ Intensity": "#56B4E9", # sky blue
    "+ Drift": "#009E73",     # bluish green
    "+ Curvature": "#D55E00", # vermillion
    "Full": "#0072B2",        # blue
}

# Tuned values (merges=false):
# + Coh/Adh:    coh=2000, adh=-1000   (drift_c=-2000 baseline)
# + Volume:     area_w=1000, area_c=-1000
# + Intensity:  intensity_w=2, intensity_c=-1000
# + Drift:      drift_w=30, drift_c=-1000
# + Curvature:  curv_w=10, curv_c=-500
# Bars sorted by LNK ascending.
rows = [
    # (label,         TRA,   DET,   LNK)
    ("None",          0.788, 0.826, 0.512),
    ("+ Coh/Adh",     0.822, 0.855, 0.586),
    ("+ Intensity",   0.822, 0.839, 0.696),
    ("+ Volume",      0.857, 0.869, 0.771),
    ("+ Curvature",   0.853, 0.858, 0.815),
    ("+ Drift",       0.853, 0.858, 0.821),
    ("Full",          0.881, 0.886, 0.845),
]

conditions = [r[0] for r in rows]
tra = [r[1] for r in rows]
det = [r[2] for r in rows]
lnk = [r[3] for r in rows]
colors = [COLOR[c] for c in conditions]

metrics = [
    ("TRA", tra),
    ("DET", det),
    ("LNK", lnk),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

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
