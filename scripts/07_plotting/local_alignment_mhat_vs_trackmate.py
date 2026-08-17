"""Local alignment index over time: MHAT vs TrackMate.

Converted from scripts/for_jingyi/inspect_tracks.ipynb (the MHAT-vs-TrackMate
local alignment cell). Generates only that figure.

For each cell at each time point, the local alignment index is the mean cosine
similarity between that cell's velocity and the velocities of its K nearest
cells at the same time point:

    -1 (opposing) .. 0 (uncorrelated) .. +1 (moving in lockstep)

The figure overlays the per-frame mean for both tracking results. Only the means
are drawn -- the +/-1 std bands overlap too heavily to keep both legible, and the
spread is per-cell rather than per-frame.

Node positions are already in world micrometers for both results (the MHAT solver
output and the TrackMate->GEFF conversion each bake the voxel size into the
stored coordinates, and import_from_geff does not re-apply the GEFF `scale`
metadata to node positions). So positions are used as-is -- multiplying by
`scale` again would double-scale them, and because the scale is anisotropic
(z = 2.1 um vs x/y = 0.325 um) it would inflate z-motion ~6.5x relative to xy,
distorting the two results differently depending on how much z-jitter each has.
`scale[0]` (seconds per frame) is still used, to put the time axis in seconds.

`--frame-range` restricts the plot to a window of frames. The index is computed
per frame (neighbors are drawn from the same frame, velocity from the next one),
so restricting the range changes only which frames are drawn and averaged -- the
per-frame values are identical to those in the full-length figure.

Usage:
    python scripts/07_plotting/local_alignment_mhat_vs_trackmate.py [--k 4] [--output foo.png]
    python scripts/07_plotting/local_alignment_mhat_vs_trackmate.py --frame-range 0 99
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from funtracks.import_export import import_from_geff
from scipy.spatial import cKDTree

# Tracking results and raw data (defaults from the notebook).
MHAT_RESULT = Path(
    "Y:/jennifer/mhat/experiments/tracking/InterfaceTransferMirrorB1-DB/02_nuclei/2026-07-09_20-20-45"
)
TRACKMATE_RESULT = Path(
    "Y:/jennifer/mhat/experiments/tracking/InterfaceTransferMirrorB1-DB/02_nuclei/trackmate"
)
RAW_NUCLEI_ZARR = Path(
    "Y:/jennifer/mhat/data/InterfaceTransferMirrorB1-DB/02_nuclei.zarr"
)
DEFAULT_OUTPUT = MHAT_RESULT.parent / "local_alignment_mhat_vs_trackmate.png"

DEFAULT_K = 4  # number of nearest cells (by world distance) to average over

# MHAT first, in the same sky blue / gray as the competitor_vs_mhat bar figure.
MHAT_COLOR = "#56B4E9"
TRACKMATE_COLOR = "#B0B0B0"

# GEFF -> funtracks attribute names; "id" is the per-track id, constant in time.
NAME_MAP = {"time": "time", "x": "x", "y": "y", "z": "z", "id": "track_id"}


def load_scale(raw_nuclei_zarr):
    """Per-axis scale [t, z, y, x] from the raw data zarr's OME axes metadata."""
    raw_nuclei = zarr.open(raw_nuclei_zarr, mode="r")
    axes = raw_nuclei.attrs["axes"]
    if axes is None:
        return [1.0, 1.0, 1.0, 1.0]
    axes = [axis for axis in axes if axis.get("name") != "channel"]
    return [axis["scale"] for axis in axes if axis.get("scale") is not None]


def local_alignment_by_time(tracks, k):
    """Per-frame mean/std/count of the local alignment index for one tracks graph.

    Returns a DataFrame indexed by time with columns [mean, std, count].
    """
    def pos(node_id):
        return np.asarray(tracks.graph.nodes[node_id]["pos"], dtype=float)

    # Per-node forward velocity (world um/frame), averaged over successor edges.
    vel = {}
    for node_id in tracks.graph.nodes:
        succs = list(tracks.graph.successors(node_id))
        if not succs:
            continue  # last detection of a track -> no forward velocity
        p0 = pos(node_id)
        vel[node_id] = np.mean([pos(s) - p0 for s in succs], axis=0)

    nodes_by_time = defaultdict(list)
    for node_id in vel:
        nodes_by_time[tracks.graph.nodes[node_id]["time"]].append(node_id)

    records = []  # (time, local_alignment)
    for time, node_ids in sorted(nodes_by_time.items()):
        k_eff = min(k + 1, len(node_ids))  # +1 because the query returns self
        if k_eff < 2:
            continue  # need at least one neighbor besides the cell itself
        positions = np.array([pos(n) for n in node_ids])
        vels = np.array([vel[n] for n in node_ids])
        norms = np.linalg.norm(vels, axis=1)
        # Unit velocity vectors (zero for stationary cells, excluded below).
        unit = np.divide(vels, norms[:, None], out=np.zeros_like(vels),
                         where=norms[:, None] > 0)

        tree = cKDTree(positions)
        _, nn_idx = tree.query(positions, k=k_eff)  # column 0 is the cell itself
        for i in range(len(node_ids)):
            if norms[i] == 0:
                continue  # stationary cell has no defined direction
            neighbors = [j for j in nn_idx[i, 1:] if norms[j] > 0]
            if not neighbors:
                continue  # neighbors all stationary -> alignment undefined
            records.append((time, float(np.mean(unit[neighbors] @ unit[i]))))

    df = pd.DataFrame(records, columns=["time", "local_alignment"])
    return df.groupby("time")["local_alignment"].agg(["mean", "std", "count"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="number of nearest neighbors to average over")
    parser.add_argument("--mhat-result", type=Path, default=MHAT_RESULT,
                        help="MHAT tracking result directory")
    parser.add_argument("--trackmate-result", type=Path, default=TRACKMATE_RESULT,
                        help="TrackMate tracking result directory")
    parser.add_argument("--raw-nuclei-zarr", type=Path, default=RAW_NUCLEI_ZARR,
                        help="raw nuclei zarr, read for the axis scales")
    parser.add_argument("--frame-range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        default=None,
                        help="restrict the plot to frames FIRST..LAST inclusive")
    parser.add_argument("--output", type=Path, default=None, help="Override output PNG path")
    args = parser.parse_args()

    scale = load_scale(args.raw_nuclei_zarr)
    print(f"Scale: T: {scale[0]} seconds, Z: {scale[1]} um, "
          f"Y: {scale[2]} um, X: {scale[3]} um")

    # (label, tracking result dir, color) -- MHAT first so it leads the legend.
    datasets = [
        ("MHAT", args.mhat_result, MHAT_COLOR),
        ("TrackMate", args.trackmate_result, TRACKMATE_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, result_dir, color in datasets:
        tracks = import_from_geff(result_dir / "pred_tracks.zarr", NAME_MAP, scale=scale)
        avg = local_alignment_by_time(tracks, args.k)
        if args.frame_range is not None:
            first, last = args.frame_range
            avg = avg.loc[(avg.index >= first) & (avg.index <= last)]
            if avg.empty:
                raise ValueError(
                    f"{label}: no frames in range {first}..{last}"
                )
        t_seconds = avg.index.values * scale[0]  # scale[0] = seconds per frame
        ax.plot(t_seconds, avg["mean"], color=color, lw=2, label=label)
        print(f"{label}: overall mean local alignment = {avg['mean'].mean():.3f}")

    ax.axhline(0.0, color="gray", ls="--", lw=1)  # 0 = no net alignment
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Local alignment index")
    ax.legend()

    fig.tight_layout()
    out = Path(args.output or DEFAULT_OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
