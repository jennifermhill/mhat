"""Convert a TrackMate spot-export CSV into the GEFF track format used by this
project, so TrackMate results can be evaluated with the standard pipeline
(evaluate_tracks.py).

TrackMate's "spots" CSV export has a 4-line header: one machine-readable header
row followed by three human-readable description/unit rows (skipped here). Each
remaining row is one spot with a unique spot ID, a TRACK_ID, a FRAME, and
POSITION_X/Y/Z in PIXEL units.

The pipeline's GEFF stores node positions in SCALED (physical) units, matching
the ground-truth `correct_tracks.zarr`. We reproduce that convention by
multiplying the pixel positions by the per-axis scale read from the GT metadata
(x,y by the xy scale, z by the z scale). Node key = TrackMate spot ID;
`track_id` node property = TrackMate TRACK_ID. Edges are reconstructed by linking,
within each track, each spot to the spot in the next occupied frame (TrackMate
tracks here are simple linear chains: no divisions, no frame gaps).

Point-matcher evaluation (NC281) needs only node positions + track_id + edges;
no segmentation is written, so `evaluate_tracking` runs with `pred_seg = None`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geff
import networkx as nx
import pandas as pd


def load_trackmate_spots(csv_path: str | Path) -> pd.DataFrame:
    """Read a TrackMate spots CSV, dropping the 3 description/unit header rows."""
    df = pd.read_csv(csv_path, skiprows=[1, 2, 3])
    required = {"ID", "TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_Z", "FRAME"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TrackMate CSV missing columns: {sorted(missing)}")
    # Ensure integer typing for ids/frames
    df = df.dropna(subset=["TRACK_ID"]).copy()
    df["ID"] = df["ID"].astype(int)
    df["TRACK_ID"] = df["TRACK_ID"].astype(int)
    df["FRAME"] = df["FRAME"].astype(int)
    return df


def build_graph(df: pd.DataFrame, scale: list[float]) -> nx.DiGraph:
    """Build a directed track graph in the pipeline's GEFF node convention.

    Args:
        df: TrackMate spots dataframe.
        scale: [t_scale, z_scale, y_scale, x_scale] used to convert pixel
            positions to the scaled units the GT/pred GEFF stores use.
    """
    _, z_scale, y_scale, x_scale = scale

    graph = nx.DiGraph()
    for row in df.itertuples(index=False):
        graph.add_node(
            int(row.ID),
            id=int(row.ID),
            time=int(row.FRAME),
            z=float(row.POSITION_Z) * z_scale,
            y=float(row.POSITION_Y) * y_scale,
            x=float(row.POSITION_X) * x_scale,
            track_id=int(row.TRACK_ID),
        )

    # Reconstruct edges: within each track, link consecutive-by-frame spots.
    n_edges = 0
    for track_id, sub in df.groupby("TRACK_ID"):
        ordered = sub.sort_values("FRAME")
        ids = ordered["ID"].tolist()
        frames = ordered["FRAME"].tolist()
        for (src, src_f), (dst, dst_f) in zip(
            zip(ids, frames), zip(ids[1:], frames[1:])
        ):
            if dst_f <= src_f:
                # Two spots in the same/earlier frame in one track would be a
                # division/merge; TrackMate's linear tracks shouldn't produce
                # these, but skip rather than create a bad edge if they do.
                print(
                    f"  WARNING: track {track_id} has non-increasing frames "
                    f"({src_f} -> {dst_f}); skipping edge {src}->{dst}"
                )
                continue
            graph.add_edge(int(src), int(dst))
            n_edges += 1

    print(f"Built graph: {graph.number_of_nodes()} nodes, {n_edges} edges, "
          f"{df['TRACK_ID'].nunique()} tracks")
    return graph


def read_scale(scale_source: Path) -> list[float]:
    """Read the per-axis scale ([t, z, y, x]) from an existing GEFF store's
    metadata. Any pipeline GEFF store works (the GT `correct_tracks.zarr`, or,
    when there is no GT, the pipeline's own `pred_tracks.zarr` for the dataset).
    """
    _, meta = geff.read(scale_source)
    scale = [a.scale if a.scale is not None else 1.0 for a in meta.axes]
    print(f"Using axis scale from {scale_source}: {scale}")
    return scale


def trackmate_to_geff(
    csv_path: str | Path,
    output_path: str | Path,
    scale_source: str | Path,
) -> None:
    df = load_trackmate_spots(csv_path)
    print(f"Loaded {len(df)} spots from {csv_path}")

    scale = read_scale(Path(scale_source))
    graph = build_graph(df, scale)

    metadata = geff.GeffMetadata(
        directed=True,
        node_props_metadata={},
        edge_props_metadata={},
    )
    geff.write(
        graph,
        Path(output_path),
        axis_names=["time", "z", "y", "x"],
        axis_types=["time", "space", "space", "space"],
        axis_scales=scale,
        metadata=metadata,
        overwrite=True,
    )
    print(f"Wrote GEFF tracks to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a TrackMate spots CSV to the pipeline GEFF format."
    )
    parser.add_argument("input_csv", help="Path to TrackMate spots CSV")
    parser.add_argument(
        "output_geff", help="Output pred_tracks.zarr path (GEFF store)"
    )
    parser.add_argument(
        "scale_source",
        help="Path to an existing GEFF store to read the axis scale from "
        "(GT correct_tracks.zarr, or the pipeline pred_tracks.zarr when no GT).",
    )
    args = parser.parse_args()

    trackmate_to_geff(args.input_csv, args.output_geff, args.scale_source)
    print("Done!")


if __name__ == "__main__":
    main()
