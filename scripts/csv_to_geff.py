"""
Script to load tracking data from a CSV file and export to GEFF format.

This script uses funtracks' built-in import/export functions for better
integration with the funtracks ecosystem.

The CSV file should have columns:
- id: Node ID
- time: Time point
- x, y, (z): Spatial coordinates
- parent_id: Parent node ID (-1 or empty for nodes with no parent)

Additional columns will be added as node attributes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from funtracks.import_export import export_to_geff, tracks_from_df


def csv_to_geff(
    csv_path: str | Path,
    output_path: str | Path,
    scale: list[float] | None = None,
    node_name_map: dict[str, str | list[str]] | None = None,
) -> None:
    """Load tracking data from CSV and export to GEFF format.

    Uses funtracks' tracks_from_df() for proper Tracks object creation.

    Args:
        csv_path: Path to the CSV file containing tracking data
        output_path: Path where the GEFF zarr will be saved
        scale: Scale factors for each axis [time, z, y, x] or [time, y, x]
               (default: all 1.0)
        node_name_map: Optional mapping from standard funtracks keys to CSV
            column names: {standard_key: column_name}.
            For example: {"time": "t", "pos": ["y", "x"], "seg_id": "label"}
            If None, column names are auto-detected.

    Required columns (standard names):
        - id: Node ID (integer)
        - time: Time point (integer)
        - x, y: Spatial coordinates (float/int)
        - z: z coordinate (for 3D data)
        - parent_id: Parent node ID (integer, -1 or empty for no parent)

    Optional columns:
        Any additional columns will be added as node attributes
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Loaded CSV file with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Drop the "correct" column since it doesn't follow funtracks/geff format
    if "correct" in df.columns:
        print("Dropping 'correct' column")
        df = df.drop(columns=["correct"])
 
    # Create Tracks object from DataFrame
    tracks = tracks_from_df(
        df=df,
        segmentation=None,
        scale=scale,
        node_name_map=node_name_map,
    )

    print(f"Created Tracks object:")
    print(f"  Nodes: {tracks.graph.number_of_nodes()}")
    print(f"  Edges: {tracks.graph.number_of_edges()}")
    print(f"  Dimensions: {tracks.ndim}D")

    # Export to GEFF format
    export_to_geff(
        tracks=tracks,
        directory=Path(output_path),
    )

    print(f"Successfully exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert tracking data from CSV to GEFF format using funtracks"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV file with tracking data",
    )
    parser.add_argument(
        "output_geff",
        type=str,
        help="Path to output GEFF zarr file (should end in .zarr)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=None,
        help="Scale factors for each axis. For 3D: [time, z, y, x]. For 2D: [time, y, x]",
    )

    args = parser.parse_args()

    print(f"Loading tracking data from {args.input_csv}...")
    csv_to_geff(
        csv_path=args.input_csv,
        output_path=args.output_geff,
        scale=args.scale,
    )

    print("Done!")


if __name__ == "__main__":
    main()
