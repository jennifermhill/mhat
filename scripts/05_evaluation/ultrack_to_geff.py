"""Convert an Ultrack ``tracks.csv`` export into the GEFF track format used by
this project, so Ultrack results can be evaluated with the standard pipeline
(``evaluate_tracks.py``) against the ``primary_nk_cells`` ground truth.

TRACKS
------
Ultrack's ``tracks.csv`` has one row per detection with columns:
``track_id, t, z, y, x, id, parent_track_id, parent_id``. Unlike the TrackMate
export, edges are given *explicitly*: ``parent_id`` points to the ``id`` of the
detection in the previous timepoint (``-1`` marks a track start), and a parent
with more than one child encodes a division. We add one ``parent_id -> id`` edge
per row with ``parent_id != -1``, reproducing both linear links and divisions.

COORDINATE CONVENTION (why we scale ALL axes):
    The pipeline's ``pred_tracks.zarr`` / GT ``correct_tracks.zarr`` store node
    positions in WORLD units (micrometers): each axis is ``pixel_index *
    axis_scale``. Verified against the pred store, which ran on the SAME
    full-resolution 1518x1200x4-slice image as Ultrack:
        x_max 814.85 = 1518 * 0.5366   (xy scale)
        y_max 644.29 = 1200 * 0.5366
        z_max 6.00   = 3    * 2.0      (z scale; z pixels span slices 0-3)
    Ultrack's ``tracks.csv`` stores raw PIXEL indices on that same grid, so we
    multiply EVERY axis by its scale read from an existing pipeline GEFF store
    (z*z_scale, y*xy_scale, x*xy_scale), exactly like ``trackmate_to_geff.py``.
    True z is preserved (Ultrack z pixels {1,2} -> world z {2,4}); the current GT
    is still a placeholder plane at world z=3, harmless under match_threshold=10.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import geff
import networkx as nx
import pandas as pd
from geff_spec import GeffMetadata

REQUIRED_COLUMNS = {"track_id", "t", "z", "y", "x", "id", "parent_id"}


def load_ultrack_tracks(csv_path: str | Path) -> pd.DataFrame:
    """Read an Ultrack ``tracks.csv`` and validate its columns/integer typing."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Ultrack CSV missing columns: {sorted(missing)}")
    df = df.copy()
    for col in ("track_id", "t", "id", "parent_id"):
        df[col] = df[col].astype(int)

    if df["id"].duplicated().any():
        raise ValueError("Ultrack CSV has duplicate node ids; expected unique 'id'.")

    # Every non-root parent_id must reference an existing node id.
    ids = set(df["id"])
    dangling = df.loc[(df["parent_id"] != -1) & (~df["parent_id"].isin(ids))]
    if len(dangling):
        raise ValueError(
            f"{len(dangling)} rows have a parent_id not present in 'id' "
            "(broken lineage); refusing to build edges."
        )
    return df


def build_graph(df: pd.DataFrame, scale: list[float]) -> nx.DiGraph:
    """Build a directed track graph in the pipeline's GEFF node convention.

    Node key = ``id`` (unique per detection). ``label`` = ``id`` mirrors MHAT
    (node key == seg label) so the eval's seg relabeling lines up.

    Args:
        df: Ultrack tracks dataframe.
        scale: ``[t_scale, z_scale, y_scale, x_scale]`` from an existing pipeline
            GEFF store. Every spatial axis (z, y, x) pixel position is multiplied
            by its scale to convert Ultrack pixels to the pred/GT world frame,
            preserving true z.
    """
    _, z_scale, y_scale, x_scale = scale

    graph = nx.DiGraph()
    for row in df.itertuples(index=False):
        graph.add_node(
            int(row.id),
            id=int(row.id),
            label=int(row.id),
            time=int(row.t),
            z=float(row.z) * z_scale,
            y=float(row.y) * y_scale,
            x=float(row.x) * x_scale,
            track_id=int(row.track_id),
        )

    # Edges are explicit: parent_id -> id for every non-root node. A parent with
    # >1 child is a division and naturally yields multiple out-edges.
    n_edges = 0
    n_divisions = 0
    for parent_id, children in df.loc[df["parent_id"] != -1].groupby("parent_id"):
        if len(children) > 1:
            n_divisions += 1
        for child_id in children["id"]:
            graph.add_edge(int(parent_id), int(child_id))
            n_edges += 1

    print(
        f"Built graph: {graph.number_of_nodes()} nodes, {n_edges} edges, "
        f"{df['track_id'].nunique()} tracks, {n_divisions} divisions"
    )
    return graph


def read_scale(scale_source: Path) -> list[float]:
    """Read the per-axis scale ``[t, z, y, x]`` from an existing pipeline GEFF
    store's metadata (the GT ``correct_tracks.zarr`` or any ``pred_tracks.zarr``
    for the dataset — both share the same axis scales)."""
    _, meta = geff.read(scale_source)
    scale = [a.scale if a.scale is not None else 1.0 for a in meta.axes]
    print(f"Using axis scale from {scale_source}: {scale}")
    return scale


def ultrack_to_geff(
    csv_path: str | Path,
    output_path: str | Path,
    scale_source: str | Path,
) -> None:
    df = load_ultrack_tracks(csv_path)
    print(f"Loaded {len(df)} detections from {csv_path}")

    scale = read_scale(Path(scale_source))
    graph = build_graph(df, scale)

    output_path = Path(output_path)

    metadata = GeffMetadata(
        directed=True,
        node_props_metadata={},
        edge_props_metadata={},
    )
    geff.write(
        graph,
        output_path,
        axis_names=["time", "z", "y", "x"],
        axis_types=["time", "space", "space", "space"],
        axis_scales=scale,
        metadata=metadata,
        overwrite=True,
    )
    print(f"Wrote GEFF tracks to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an Ultrack tracks.csv to the pipeline GEFF format."
    )
    parser.add_argument("input_csv", help="Path to Ultrack tracks.csv")
    parser.add_argument(
        "output_geff",
        help="Output pred_tracks.zarr path (GEFF store). To evaluate, place it "
        "under experiments/tracking/<experiment>/<dataset>/<track_result>/ and "
        "set eval_config track_result=<track_result>.",
    )
    parser.add_argument(
        "scale_source",
        help="Path to an existing GEFF store to read the axis scale from "
        "(GT correct_tracks.zarr, or any pred_tracks.zarr for the dataset).",
    )
    args = parser.parse_args()

    ultrack_to_geff(
        args.input_csv, args.output_geff, args.scale_source
    )
    print("Done!")


if __name__ == "__main__":
    main()
