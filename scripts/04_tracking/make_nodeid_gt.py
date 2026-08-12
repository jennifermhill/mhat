"""Write a node-id-labeled twin of a CTC-derived ground truth.

WHY THIS EXISTS
---------------
The GT-amount materializers assume `correct_seg.zarr` is labeled with geff **node
ids**, and say so in their comments:

  - `gt_subsets.write_gt_subset` builds `lut = np.zeros(seg.max() + 1)` and indexes
    it with the kept node ids;
  - `gt_crops.write_gt_crop` reads `np.unique(seg[box])` and looks those values up
    directly in `raw_graph.nodes`.

That holds for a GT whose geff declares `related_objects[0].label_prop = "seg_id"`
(or omits it, falling through to the node id). It does **not** hold for a GT written
by `from_ctc_to_geff`, which labels the voxels with the CTC **track_id** and declares
`label_prop = "track_id"`. On Fluo-C3DL-MDA231/01_cells that is 33 distinct labels
(max 51) against 364 node ids, so:

  - the subset LUT is 52 entries long and silently drops every kept node id above 51,
    writing an arbitrary subset segmentation;
  - the crop path reads track ids, finds them all "valid" (1..51 are real node ids),
    and keeps an unrelated set of nodes — then trips the driver's
    `n_gt_nodes_kept == n_gt_nodes_annotated` assertion.

Rather than teach both materializers to honour `label_prop` — which would be a real
behaviour change to the no-oracle-verified crop path — this writes a *twin* GT in the
label space they already document, and the sweep config points `gt_data_dir` at it.
The original GT directory is never touched.

WHAT IT WRITES
--------------
    <out_dir>/correct_seg.zarr    voxels relabeled (t, track_id) -> node id
    <out_dir>/correct_tracks.zarr the same graph, byte-identical node ids / time /
                                  track_id / x / y / z, plus an explicit
                                  `seg_id` = node id property, and
                                  related_objects.label_prop = "seg_id"

The (time, track_id) -> node mapping is a bijection on any CTC-derived GT: one object
per track per frame. That is asserted, not assumed.

Usage:
    python scripts/04_tracking/make_nodeid_gt.py \
        --gt-dir experiments/tracking/Fluo-C3DL-MDA231/01_cells \
        --out-dir experiments/tracking/Fluo-C3DL-MDA231/01_cells/gt_amount/gt_nodeid
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geff
import numpy as np
import zarr
from geff.core_io import write_arrays
from geff_spec import RelatedObject


def read_geff_arrays(path: Path):
    """Read node ids, node props and edge ids straight off the zarr store.

    Deliberately not `geff.read`: going through networkx and back would rebuild the
    property arrays, and the whole point of the twin is that everything except the
    label space is carried over untouched.
    """
    root = zarr.open(str(path), mode="r")
    node_ids = np.asarray(root["nodes/ids"][:])
    node_props = {
        name: np.asarray(root[f"nodes/props/{name}/values"][:])
        for name in root["nodes/props"]
    }
    edge_ids = np.asarray(root["edges/ids"][:])
    edge_props = {}
    if "edges/props" in root:
        edge_props = {
            name: np.asarray(root[f"edges/props/{name}/values"][:])
            for name in root["edges/props"]
        }
    return node_ids, node_props, edge_ids, edge_props


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True, help="dir holding correct_*.zarr")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_tracks = args.gt_dir / "correct_tracks.zarr"
    src_seg = args.gt_dir / "correct_seg.zarr"
    assert src_tracks.is_dir(), f"missing {src_tracks}"
    assert src_seg.is_dir(), f"missing {src_seg}"

    out_tracks = args.out_dir / "correct_tracks.zarr"
    out_seg_path = args.out_dir / "correct_seg.zarr"
    if args.out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    node_ids, node_props, edge_ids, edge_props = read_geff_arrays(src_tracks)
    metadata = geff.GeffMetadata.read(src_tracks)

    label_prop = "track_id"
    if metadata.related_objects:
        for ro in metadata.related_objects:
            if ro.type == "labels" and ro.label_prop:
                label_prop = ro.label_prop
                break
    print(f"Source label_prop = {label_prop!r}")
    if label_prop == "seg_id":
        print("Source is already node-id labeled; the twin is a straight copy of the label space.")

    times = np.asarray(node_props["time"]).astype(np.int64)
    labels = np.asarray(node_props[label_prop]).astype(np.int64)

    # One object per (frame, label) is what makes the relabeling well defined.
    pairs = list(zip(times.tolist(), labels.tolist()))
    assert len(set(pairs)) == len(pairs), (
        f"{len(pairs) - len(set(pairs))} duplicate (time, {label_prop}) pairs; the "
        "voxel -> node mapping is not a bijection and cannot be inverted"
    )

    # --- segmentation ---------------------------------------------------------
    seg_arr = zarr.open(str(src_seg), mode="r")
    seg = seg_arr[:]
    n_frames = seg.shape[0]
    # node ids reach 364 here but nothing guarantees the source dtype is wide enough
    # for them in general — pick a dtype from the ids, not from the source.
    out_dtype = np.min_scalar_type(int(node_ids.max()))
    out_seg = np.zeros(seg.shape, dtype=out_dtype)

    n_relabeled = 0
    empty_nodes = []
    for node, t, label in zip(node_ids.tolist(), times.tolist(), labels.tolist()):
        assert 0 <= t < n_frames, f"node {node} has time {t}, outside the segmentation"
        mask = seg[t] == label
        if not mask.any():
            empty_nodes.append(node)
            continue
        out_seg[t][mask] = node
        n_relabeled += 1

    print(f"Relabeled {n_relabeled}/{len(node_ids)} nodes")
    if empty_nodes:
        # Not fatal: a GT node whose marker owns no voxels is invisible to the
        # overlap computation either way. Report it so it is never a surprise.
        print(f"  {len(empty_nodes)} node(s) own no voxels: {empty_nodes[:10]}")

    out_zarr = zarr.open(
        str(out_seg_path),
        mode="w",
        shape=seg.shape,
        chunks=getattr(seg_arr, "chunks", None),
        dtype=out_dtype,
    )
    for key, value in dict(seg_arr.attrs).items():
        out_zarr.attrs[key] = value
    out_zarr[:] = out_seg
    print(f"Wrote {out_seg_path}  {seg.shape} {out_dtype}")

    # --- graph ----------------------------------------------------------------
    # seg_id IS the node id. Written as an explicit property so the label space is
    # declared in the file rather than inferred from a missing key.
    twin_props = dict(node_props)
    twin_props["seg_id"] = node_ids.astype(np.int64)

    twin_meta = metadata.model_copy(deep=True)
    twin_meta.related_objects = [
        RelatedObject(type="labels", path="../correct_seg.zarr", label_prop="seg_id")
    ]
    # node_props_metadata is keyed by property name; drop it and let write_arrays
    # rebuild it rather than hand-maintaining an entry for seg_id.
    twin_meta.node_props_metadata = {}
    twin_meta.edge_props_metadata = {}

    write_arrays(
        geff_store=out_tracks,
        node_ids=node_ids,
        node_props={
            name: {"values": values, "missing": None} for name, values in twin_props.items()
        },
        edge_ids=edge_ids,
        edge_props={
            name: {"values": values, "missing": None} for name, values in edge_props.items()
        },
        metadata=twin_meta,
        overwrite=True,
    )
    print(f"Wrote {out_tracks}")

    # --- G0 ------------------------------------------------------------------
    verify(src_tracks, out_tracks, out_seg_path, node_ids, times, labels, label_prop)


def verify(src_tracks, out_tracks, out_seg_path, node_ids, times, labels, label_prop) -> None:
    """G0: the twin round-trips, and its voxels are in node-id space."""
    src_graph, _ = geff.read(src_tracks)
    twin_graph, twin_meta = geff.read(out_tracks)

    assert twin_graph.number_of_nodes() == src_graph.number_of_nodes(), "node count changed"
    assert twin_graph.number_of_edges() == src_graph.number_of_edges(), "edge count changed"
    assert set(twin_graph.nodes) == set(src_graph.nodes), "node ids changed"
    assert set(twin_graph.edges) == set(src_graph.edges), "edges changed"
    for node in src_graph.nodes:
        src_data = src_graph.nodes[node]
        twin_data = twin_graph.nodes[node]
        for key in ("time", "track_id", "x", "y", "z"):
            if key in src_data:
                assert twin_data[key] == src_data[key], f"node {node} {key} changed"
        assert twin_data["seg_id"] == node, f"node {node} seg_id is not its own id"

    ro = twin_meta.related_objects[0]
    assert ro.label_prop == "seg_id", f"label_prop is {ro.label_prop!r}"
    assert "\\" not in ro.path, f"windows separator in related_objects path {ro.path!r}"

    twin_seg = zarr.open(str(out_seg_path), mode="r")[:]
    present = set(int(v) for v in np.unique(twin_seg) if v != 0)
    expected = {
        int(n)
        for n, t, label in zip(node_ids.tolist(), times.tolist(), labels.tolist())
        if int(n) in set(src_graph.nodes)
    }
    # Only nodes that actually own voxels can appear.
    assert present <= expected, f"twin seg holds {sorted(present - expected)[:10]} which are not nodes"
    missing = expected - present
    print(
        f"G0 OK: {twin_graph.number_of_nodes()} nodes, {twin_graph.number_of_edges()} edges, "
        f"{len(present)} labels in the twin segmentation"
        + (f" ({len(missing)} node(s) own no voxels)" if missing else "")
    )


if __name__ == "__main__":
    main()
