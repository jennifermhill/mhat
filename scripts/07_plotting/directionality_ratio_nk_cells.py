"""Histogram of per-track directionality ratio for primary_nk_cells / 01_cells.

For each track, the directionality ratio is the net displacement from the
track's origin divided by the total distance the cell actually travelled:

    ratio = |p_last - p_first| / sum_i |p_{i+1} - p_i|

By the triangle inequality it is bounded in [0, 1]: 1.0 is a perfectly straight
path, and values near 0 mean the cell wandered back close to where it started.
Short tracks are very noisy (a 2-frame track is 1.0 by construction), so tracks
with fewer than --min-frames detections are excluded.

Node positions in pred_tracks.zarr are already in world micrometers -- the
pipeline bakes the voxel size into the centroid at graph-construction time
(nodes_from_segmentation in src/mhat/tracking/utils.py). So they are used
as-is; multiplying by the geff `scale` metadata again would double-scale them.
The geff metadata is read only for the spatial axis *names*.

The scale is strongly anisotropic for this dataset (z = 2.0 um over just 4
slices vs y/x = 0.5366 um), so a single voxel of z jitter in a centroid adds
~2 um of spurious path length and biases the ratio *down*. Pass `--axes y,x` to
compute the in-plane ratio instead and see how much of the result depends on z.
Measured on nkgraph_size120, the difference is small: 3D mean 0.4258 vs 2D
0.4275 over the same 1157 tracks.

The tracking run this targets has divisions = false and merges = false, and
solve_with_motile adds MaxParents(1) + MaxChildren(1), so every track is a
simple path and grouping nodes by the stored `track_id` recovers whole tracks.
`geff.read` is used rather than funtracks' `import_from_geff` because the
latter renumbers track_id.

Usage:
    python scripts/07_plotting/directionality_ratio_nk_cells.py
    python scripts/07_plotting/directionality_ratio_nk_cells.py --axes y,x
"""
import argparse
from collections import defaultdict
from pathlib import Path

import geff
import numpy as np
import pandas as pd
from numpy import linalg

from mhat.evaluation._report import Report
from mhat.evaluation.fn_analysis import spatial_axis_names

RESULT_DIR = Path(
    "Y:/jennifer/mhat/experiments/tracking/primary_nk_cells/01_cells/nkgraph_size120"
)
TRACKS_ZARR = RESULT_DIR / "pred_tracks.zarr"

MIN_FRAMES = 5  # tracks with fewer detections are too noisy to be meaningful
N_BINS = 40  # ~29 tracks/bin at the default filter, still well populated

MHAT_COLOR = "#56B4E9"  # sky blue, as in the other 07_plotting figures
STAT_COLOR = "#B0B0B0"


def bin_edges(n_bins=N_BINS):
    """``n_bins`` equal-width bins spanning the ratio's full [0, 1] range.

    linspace, not arange(0, 1 + w, w): float accumulation makes arange's edge
    count and final value unreliable.
    """
    return np.linspace(0.0, 1.0, n_bins + 1)


def load_track_positions(tracks_zarr, axes=None):
    """Group geff nodes by their stored ``track_id``.

    Returns (positions, times, info) where positions[track_id] is an
    (n_nodes, len(axes)) float array of world-um coordinates sorted by time,
    times[track_id] is the matching int time array, and info carries graph
    totals for the report.
    """
    graph, metadata = geff.read(tracks_zarr)
    axes = tuple(axes) if axes else tuple(spatial_axis_names(metadata))

    grouped = defaultdict(list)  # track_id -> [(time, [coords...]), ...]
    for _, data in graph.nodes(data=True):
        grouped[int(data["track_id"])].append(
            (int(data["time"]), [float(data[axis]) for axis in axes])
        )

    positions, times, n_gapped = {}, {}, 0
    for track_id, entries in grouped.items():
        track_times = np.array([t for t, _ in entries], dtype=int)
        track_pos = np.array([p for _, p in entries], dtype=float)
        order = np.argsort(track_times, kind="stable")
        track_times, track_pos = track_times[order], track_pos[order]

        # A repeated time means this track_id is not a simple path, and then
        # "total path length" is meaningless -- fail loudly rather than emit
        # a plausible-looking wrong number.
        if len(np.unique(track_times)) != len(track_times):
            raise ValueError(
                f"track_id {track_id} has multiple nodes at the same time point; "
                "it is not a simple path, so path length is undefined"
            )
        if np.any(np.diff(track_times) != 1):
            n_gapped += 1  # gap-closed edge: path length spans >1 frame

        positions[track_id] = track_pos
        times[track_id] = track_times

    info = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_tracks": len(positions),
        "n_gapped": n_gapped,
        "axes": axes,
    }
    return positions, times, info


def track_metrics(positions):
    """(net displacement from origin, total path length) in world um."""
    steps = linalg.norm(np.diff(positions, axis=0), axis=1)
    return float(linalg.norm(positions[-1] - positions[0])), float(steps.sum())


def build_track_table(positions, times, min_frames=MIN_FRAMES):
    """One row per track, including the ones the filter drops.

    Keeping every track behind an ``included`` flag means a single code path
    and leaves the dropped tracks inspectable in the CSV.
    """
    rows = []
    for track_id in sorted(positions):
        track_pos, track_times = positions[track_id], times[track_id]
        displacement, path_length = track_metrics(track_pos)
        rows.append(
            {
                "track_id": track_id,
                "n_frames": len(track_times),
                "t_start": int(track_times[0]),
                "t_end": int(track_times[-1]),
                "displacement_um": displacement,
                "path_length_um": path_length,
            }
        )
    df = pd.DataFrame(rows)

    # A track whose positions are all identical gives 0/0. Leaving it as 0.0
    # would plant a fake "maximally non-directional" spike in the first bin, so
    # make it NaN and exclude it explicitly instead.
    disp = df["displacement_um"].to_numpy()
    path = df["path_length_um"].to_numpy()
    ratio = np.divide(disp, path, out=np.full_like(disp, np.nan), where=path > 0.0)

    # displacement <= path_length always holds; anything past 1 is float
    # rounding at ~1e-16. A larger excursion means a real bug (unsorted times,
    # a branching track_id), so check before clipping it away.
    if np.isfinite(ratio).any():
        assert np.nanmax(ratio) <= 1.0 + 1e-9, (
            f"directionality ratio {np.nanmax(ratio)} exceeds 1 by more than "
            "float rounding -- the track ordering is likely wrong"
        )
    df["ratio"] = np.clip(ratio, 0.0, 1.0)
    df["included"] = (df["n_frames"] >= min_frames) & np.isfinite(ratio)
    return df


def plot_ratio_histogram(ratios, output_path, min_frames, axes, edges, run_name):
    """Histogram of the included tracks' directionality ratios."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean, median = float(np.mean(ratios)), float(np.median(ratios))
    dims = f"{len(axes)}D ({', '.join(axes)})"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=edges, color=MHAT_COLOR, edgecolor="white", linewidth=0.6)
    ax.axvline(median, color="#333333", ls="--", lw=1.5, label=f"median = {median:.3f}")
    ax.axvline(mean, color=STAT_COLOR, ls=":", lw=1.5, label=f"mean = {mean:.3f}")

    # Pin the axis to [0, 1] so the 20 bins fill it without matplotlib padding.
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("Directionality ratio (net displacement / path length)")
    ax.set_ylabel("Number of tracks")
    ax.set_title(
        f"NK cell tracks ({run_name}), {dims}\n"
        f"n = {len(ratios)} tracks with >= {min_frames} frames"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_report(df, counts, info, min_frames, tracks_zarr, axes, edges):
    """Summary statistics and the per-bin table, echoed and saved."""
    report = Report("Directionality ratio (net displacement / path length)")
    report.line(f"Tracks:    {tracks_zarr}")
    report.line(f"Axes used: {list(axes)}  ({len(axes)}D)")
    report.line(
        "Positions are already in world micrometers; the geff `scale` metadata "
        "was NOT re-applied."
    )
    report.line("")
    report.line(
        f"Graph: {info['n_nodes']} nodes, {info['n_edges']} edges, "
        f"{info['n_tracks']} tracks (grouped by stored track_id)"
    )
    report.line(f"Tracks with time gaps: {info['n_gapped']}")
    report.line("")

    n_short = int((df["n_frames"] < min_frames).sum())
    # ratio is NaN iff the track's total path length was 0
    n_zero_path = int((~np.isfinite(df["ratio"])).sum())
    kept = df[df["included"]]
    report.line(f"Filter: min_frames = {min_frames}")
    report.line(f"  tracks kept:                      {len(kept)}")
    report.line(f"  dropped (< {min_frames} frames):             {n_short}")
    report.line(f"  excluded (zero path length):      {n_zero_path}")
    report.line("")

    if kept.empty:
        report.line("No tracks passed the filter; nothing to summarize.")
        return report

    frames = kept["n_frames"].to_numpy()
    report.line(
        f"Frames per kept track: min {frames.min()}, "
        f"median {np.median(frames):.1f}, max {frames.max()}"
    )
    report.line("")

    ratios = kept["ratio"].to_numpy()
    report.line("Directionality ratio over the kept tracks:")
    for label, value in [
        ("n", float(len(ratios))),
        ("Mean", float(ratios.mean())),
        ("Std", float(ratios.std())),
        ("Min", float(ratios.min())),
        ("25%", float(np.percentile(ratios, 25))),
        ("Median", float(np.median(ratios))),
        ("75%", float(np.percentile(ratios, 75))),
        ("Max", float(ratios.max())),
    ]:
        report.line(f"  {label:>10}  {value:>10.4f}")
    report.line("")

    width = edges[1] - edges[0]
    report.line(f"Histogram ({len(counts)} bins of width {width:.4f}):")
    report.line(f"  {'bin':>16}  {'count':>7}  {'fraction':>9}")
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        report.line(
            f"  [{lo:.4f}, {hi:.4f})  {int(count):>7}  {count / counts.sum():>9.4f}"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-zarr", type=Path, default=TRACKS_ZARR,
                        help="pred_tracks.zarr of the tracking result")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where the PNG, report and CSV are written "
                             "(default: the tracking result dir being read)")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                        help="drop tracks with fewer detections than this")
    parser.add_argument("--bins", type=int, default=N_BINS,
                        help="number of equal-width bins spanning [0, 1]")
    parser.add_argument("--axes", type=str, default=None,
                        help="comma-separated spatial axes, e.g. 'y,x' for the "
                             "in-plane ratio (default: all, from geff metadata)")
    parser.add_argument("--no-csv", action="store_true",
                        help="skip writing the per-track CSV")
    args = parser.parse_args()

    axes = tuple(a.strip() for a in args.axes.split(",")) if args.axes else None
    positions, times, info = load_track_positions(args.tracks_zarr, axes=axes)
    axes = info["axes"]

    df = build_track_table(positions, times, min_frames=args.min_frames)
    # Default beside the run being read, so pointing --tracks-zarr at a different
    # result never overwrites another run's figure.
    run_dir = Path(args.tracks_zarr).parent
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = None
    if not args.no_csv:
        csv_path = output_dir / "directionality_ratio_per_track.csv"
        df.to_csv(csv_path, index=False, float_format="%.6f")

    edges = bin_edges(args.bins)
    ratios = df.loc[df["included"], "ratio"].to_numpy()
    counts, _ = np.histogram(ratios, bins=edges)
    # Clipping to [0, 1] above makes this exact: np.histogram would otherwise
    # silently drop a ratio of 1.0000000000000002 as out of range.
    assert counts.sum() == len(ratios), (
        f"{len(ratios) - counts.sum()} ratios fell outside the [0, 1] bins"
    )

    png_path = None
    if len(ratios):
        png_path = plot_ratio_histogram(
            ratios,
            output_dir / "directionality_ratio_hist.png",
            args.min_frames,
            axes,
            edges,
            run_dir.name,
        )

    report = build_report(
        df, counts, info, args.min_frames, args.tracks_zarr, axes, edges
    )
    report.line("")
    if png_path:
        report.line(f"Histogram saved to {png_path}")
    if csv_path:
        report.line(f"Per-track values saved to {csv_path}")
    report.write(output_dir / "directionality_ratio_report.txt")


if __name__ == "__main__":
    main()
