"""Convert a TrackMate export into the GEFF track format used by this project,
so TrackMate results can be evaluated with the standard pipeline
(evaluate_tracks.py). Two inputs are accepted, chosen by file extension.

TrackMate XML (``.xml``, File > Save in the TrackMate GUI) -- preferred
-----------------------------------------------------------------------
The XML holds everything: spot positions in calibrated units, the image
calibration, explicit edges (so gaps, splits and merges survive), and, for
detectors that segment (Cellpose, StarDist, thresholding, label images), each
spot's outline as a polygon. Only spots in the *filtered* tracks are written --
TrackMate's tracking result; spots in no track are left out unless
``--include-untracked`` is given. Positions are converted to world units as
``position / pixel_size * scale``, i.e. through the pixel grid, so both a
calibrated and a pixel-unit TrackMate project land in the scale-source frame.

For 2D data the outlines are rasterized into ``pred_seg.zarr`` beside the
tracks (a pixel belongs to a spot when its center lies inside the polygon),
labeled by node id, so the CTC matcher can score the result. Node id = TrackMate
spot ID + 1 and ``track_id`` = TrackMate TRACK_ID + 1, because 0 is the
segmentation background and spot 0 / track 0 would otherwise vanish from the
mask; ``trackmate_id`` keeps the original spot ID.

TrackMate spots CSV (``.csv``) -- point matcher only
----------------------------------------------------
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
import xml.etree.ElementTree as ET
from pathlib import Path

import geff
import networkx as nx
import numpy as np
import pandas as pd
import zarr
from skimage.draw import polygon as fill_polygon


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


def read_axes(scale_source: Path) -> tuple[list[str], list[float]]:
    """Axis names and scales of an existing GEFF store (``time`` first)."""
    meta = geff.GeffMetadata.read(scale_source)
    names = [a.name for a in meta.axes]
    scale = [a.scale if a.scale is not None else 1.0 for a in meta.axes]
    print(f"Using axes {names} with scale {scale} from {scale_source}")
    return names, scale


def load_trackmate_xml(xml_path: str | Path, include_untracked: bool = False):
    """Spots, edges and image geometry from a TrackMate XML file.

    Returns:
        (spots, edges, image) -- ``spots`` maps spot ID to a dict with
        ``frame``, ``pos`` (calibrated x, y, z), ``track`` (TRACK_ID or None),
        ``radius`` and ``roi`` (an (n, 2) array of calibrated x, y offsets from
        the spot center, or None); ``edges`` is a list of (source, target) spot
        IDs; ``image`` holds ``width``, ``height``, ``nslices``, ``nframes`` and
        the pixel size ``(pixelwidth, pixelheight, voxeldepth)``.
    """
    root = ET.parse(xml_path).getroot()
    model = root.find("Model")
    image_el = root.find("Settings/ImageData")
    image = {
        key: int(image_el.get(key)) for key in ("width", "height", "nslices", "nframes")
    }
    image["pixel_size"] = tuple(
        float(image_el.get(key)) for key in ("pixelwidth", "pixelheight", "voxeldepth")
    )
    print(f"TrackMate {root.get('version')}, units {model.get('spatialunits')}, "
          f"image {image}")

    kept_tracks = {int(el.get("TRACK_ID")) for el in model.find("FilteredTracks")}
    spot_track, edges = {}, []
    for track in model.find("AllTracks"):
        track_id = int(track.get("TRACK_ID"))
        if track_id not in kept_tracks:
            continue
        for edge in track.findall("Edge"):
            src, dst = int(edge.get("SPOT_SOURCE_ID")), int(edge.get("SPOT_TARGET_ID"))
            edges.append((src, dst))
            spot_track[src] = spot_track[dst] = track_id

    spots = {}
    for spot in model.find("AllSpots").iter("Spot"):
        spot_id = int(spot.get("ID"))
        if spot_id not in spot_track and not include_untracked:
            continue
        n_points = int(float(spot.get("ROI_N_POINTS", 0)))
        roi = None
        if n_points:
            roi = np.array(spot.text.split(), dtype=float).reshape(n_points, 2)
        spots[spot_id] = {
            "frame": int(float(spot.get("FRAME"))),
            "pos": tuple(float(spot.get(f"POSITION_{a}")) for a in "XYZ"),
            "track": spot_track.get(spot_id),
            "radius": float(spot.get("RADIUS")),
            "roi": roi,
        }
    n_untracked = sum(s["track"] is None for s in spots.values())
    print(f"Loaded {len(spots)} spots ({n_untracked} in no track), {len(edges)} "
          f"edges, {len(kept_tracks)} filtered tracks from {xml_path}")
    return spots, edges, image


def build_graph_from_xml(spots, edges, image, axis_names, scale) -> nx.DiGraph:
    """Track graph in the pipeline's GEFF convention, positions in world units.

    Each calibrated position goes through the pixel grid (``/ pixel_size``) and
    then into the scale source's frame (``* scale``). Untracked spots (when
    kept) each get a track id of their own above the TrackMate ones.
    """
    pixel = dict(zip("xyz", image["pixel_size"]))
    world = dict(zip(axis_names[1:], scale[1:]))
    next_track = max((s["track"] for s in spots.values() if s["track"] is not None),
                     default=-1) + 1
    graph = nx.DiGraph()
    for spot_id, spot in sorted(spots.items()):
        track = spot["track"]
        if track is None:
            track, next_track = next_track, next_track + 1
        pos = dict(zip("xyz", spot["pos"]))
        attrs = {axis: pos[axis] / pixel[axis] * world[axis] for axis in axis_names[1:]}
        graph.add_node(
            spot_id + 1,
            time=spot["frame"],
            label=spot_id + 1,
            track_id=track + 1,
            trackmate_id=spot_id,
            **attrs,
        )
    graph.add_edges_from((src + 1, dst + 1) for src, dst in edges)
    n_div = sum(1 for n in graph if graph.out_degree(n) > 1)
    n_merge = sum(1 for n in graph if graph.in_degree(n) > 1)
    print(f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} "
          f"edges, {n_div} divisions, {n_merge} merges")
    return graph


def rasterize_rois(spots, image) -> np.ndarray:
    """A (t, y, x) uint32 label array, each spot filled with its node id.

    A pixel belongs to a spot when its center lies inside the spot's polygon
    (pixel (row, col) has its center at calibrated (col, row) * pixel size, the
    ImageJ convention TrackMate positions use). A spot without an outline is
    filled as a disc of its radius. Where two outlines overlap, the later spot
    (higher ID) wins; the overlap is reported.
    """
    px, py, _ = image["pixel_size"]
    shape = (image["height"], image["width"])
    seg = np.zeros((image["nframes"], *shape), dtype=np.uint32)
    n_disc = n_overlap = 0
    area_ratios = []
    for spot_id, spot in sorted(spots.items()):
        x, y, _ = spot["pos"]
        if spot["roi"] is not None:
            cols = (x + spot["roi"][:, 0]) / px
            rows = (y + spot["roi"][:, 1]) / py
        else:
            n_disc += 1
            angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            cols = (x + spot["radius"] * np.cos(angle)) / px
            rows = (y + spot["radius"] * np.sin(angle)) / py
        rr, cc = fill_polygon(rows, cols, shape)
        frame = seg[spot["frame"]]
        n_overlap += int(np.count_nonzero(frame[rr, cc]))
        frame[rr, cc] = spot_id + 1
        if spot["roi"] is not None:
            # Shoelace area of the polygon, in pixels, against what was filled.
            poly_area = 0.5 * abs(np.dot(cols, np.roll(rows, 1)) - np.dot(rows, np.roll(cols, 1)))
            area_ratios.append(len(rr) / poly_area if poly_area else np.nan)
    print(f"Rasterized {len(spots)} spots: {n_disc} as discs (no outline), "
          f"{n_overlap} pixels overwritten by an overlapping outline, filled / "
          f"polygon area median {np.nanmedian(area_ratios):.3f}")
    return seg


def trackmate_xml_to_geff(
    xml_path: str | Path,
    output_path: str | Path,
    scale_source: str | Path,
    include_untracked: bool = False,
) -> None:
    spots, edges, image = load_trackmate_xml(xml_path, include_untracked)
    axis_names, scale = read_axes(Path(scale_source))
    if axis_names[0] != "time" or not set(axis_names[1:]) <= {"z", "y", "x"}:
        raise ValueError(f"Unexpected axes {axis_names} in {scale_source}")
    graph = build_graph_from_xml(spots, edges, image, axis_names, scale)

    output_path = Path(output_path)
    geff.write(
        graph,
        output_path,
        axis_names=axis_names,
        axis_types=["time"] + ["space"] * (len(axis_names) - 1),
        axis_scales=scale,
        metadata=geff.GeffMetadata(
            directed=True, node_props_metadata={}, edge_props_metadata={}
        ),
        overwrite=True,
    )
    print(f"Wrote GEFF tracks to {output_path}")

    if len(axis_names) != 3:
        print("3D data: no segmentation written (TrackMate 3D outlines are meshes); "
              "evaluate with the point matcher.")
        return
    seg = rasterize_rois(spots, image)
    seg_path = output_path.parent / "pred_seg.zarr"
    store = zarr.open(
        str(seg_path), mode="w", shape=seg.shape, dtype=seg.dtype,
        chunks=(1, 512, 512),
    )
    store[:] = seg
    store.attrs["axes"] = [
        {"name": name, "scale": s, "type": "time" if name == "time" else "space"}
        for name, s in zip(axis_names, scale)
    ]
    print(f"Wrote segmentation to {seg_path}")


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
        description="Convert a TrackMate XML or spots CSV to the pipeline GEFF format."
    )
    parser.add_argument("input", help="TrackMate XML (.xml) or spots CSV (.csv)")
    parser.add_argument(
        "output_geff", help="Output pred_tracks.zarr path (GEFF store)"
    )
    parser.add_argument(
        "scale_source",
        help="Path to an existing GEFF store to read the axes and scale from "
        "(GT correct_tracks.zarr, or the pipeline pred_tracks.zarr when no GT).",
    )
    parser.add_argument(
        "--include-untracked", action="store_true",
        help="XML only: also write spots that belong to no track, one track each.",
    )
    args = parser.parse_args()

    if Path(args.input).suffix.lower() == ".xml":
        trackmate_xml_to_geff(
            args.input, args.output_geff, args.scale_source, args.include_untracked
        )
    else:
        trackmate_to_geff(args.input, args.output_geff, args.scale_source)
    print("Done!")


if __name__ == "__main__":
    main()
