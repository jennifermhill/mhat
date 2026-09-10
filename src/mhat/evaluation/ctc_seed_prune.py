"""Restrict a tracking result to the cell lineages the CTC actually evaluates.

Three of the CTC 3D datasets -- Fluo-N3DL-DRO, Fluo-N3DL-TRIC and
Fluo-N3DL-TRIF -- annotate only a subset of the cells in the field of view, and
the Cell Tracking Benchmark counts *every* other tracked cell as an error. From
the DRO dataset note:

    Only the cells that form the developing nervous system will be used for the
    evaluation of the detection, segmentation, and tracking accuracy. Those
    cells can be identified in the first frame of the provided gold reference
    tracking annotation. Be aware that any other cells segmented and tracked
    will be considered as errors by the evaluation software used for the Cell
    Tracking Benchmark.

The frame-0 markers are provided with the challenge sequences precisely so that
they can be used this way, so seeding from them is required input rather than
ground-truth leakage -- what stays hidden is frames 1..T-1. Everything here
therefore reads exactly one file out of the ``TRA`` folder: the first frame.

The transformation is a forward lineage prune. Each frame-0 marker is matched to
at most one predicted object under the CTC detection criterion; the surviving
result is every node reachable from those seeds by following edges forward in
time, and nothing else. Note that this only removes false positives -- it cannot
recover a missed link, and in fact makes one more expensive, because a lineage
that breaks at frame k loses every node after k as well. That is the CTB
scoring behaviour, not an artefact of doing it here.

The CTC SEG measure must NOT be read off a pruned result. The segmentation
benchmark filters extra cells itself, and its reference annotation covers cells
the tracking benchmark excludes -- in Fluo-N3DL-DRO the GT numbering runs to 793
while ``TRA`` uses only 189 of those labels, and every label in ``01_GT/SEG``
falls in the gaps. Pruning zeroes exactly those cells, so SEG computed on the
pruned store collapses. Point ``seg_track_result`` in the eval config at the
unpruned run to keep SEG honest while TRA/DET/LNK come from the pruned one.
"""

import json
import shutil
from collections import deque
from pathlib import Path

import geff
import numpy as np
import tifffile
import zarr

# A predicted object detects a reference marker when it covers more than half
# of it (CTC detection criterion). At most one object can satisfy this, which is
# what makes the per-marker assignment unambiguous.
CTC_DETECTION_FRACTION = 0.5


def read_first_frame_markers(tra_dir):
    """Read the first frame of a CTC ``TRA`` folder.

    Returns (markers, path). Only the first frame is ever read -- the rest of
    the folder is the withheld part of the ground truth.
    """
    tra_dir = Path(tra_dir)
    frames = sorted(tra_dir.glob("man_track*.tif"))
    if not frames:
        raise FileNotFoundError(f"No man_track*.tif files in {tra_dir}")
    return tifffile.imread(frames[0]), frames[0]


def seed_from_markers(markers, pred_frame):
    """Match every frame-0 marker to at most one predicted label.

    Both arrays are on the same voxel grid: ``markers`` holds the small
    (~75-voxel) CTC track markers, ``pred_frame`` frame 0 of ``pred_seg.zarr``,
    which is labelled by graph node id.

    Only the marker voxels are ever touched, so the cost is set by the total
    marker volume (a few thousand voxels) rather than by the image, which for
    DRO is 96M voxels a frame.

    Returns:
        seeds: ``{marker_label: predicted_node_id}`` for every detected marker.
        unmatched: marker labels no predicted object covered by more than half
            -- a segmentation miss in frame 0, which costs the whole lineage.
        collisions: ``{predicted_node_id: [marker_label, ...]}`` for predicted
            objects that detect two or more markers. This is a genuine
            under-segmentation and is reported rather than repaired: merging two
            evaluated lineages into one is exactly the error CTC will charge.
    """
    if markers.shape != pred_frame.shape:
        raise ValueError(
            f"Marker frame {markers.shape} and predicted frame {pred_frame.shape} "
            "are on different grids; they must come from the same voxel space."
        )

    idx = np.nonzero(markers)
    marker_at = markers[idx]
    pred_at = np.asarray(pred_frame)[idx]

    seeds = {}
    unmatched = []
    for label in np.unique(marker_at):
        selected = marker_at == label
        area = int(selected.sum())
        covering = pred_at[selected]
        covering = covering[covering != 0]
        if covering.size:
            values, counts = np.unique(covering, return_counts=True)
            best = int(values[np.argmax(counts)])
            if int(counts.max()) > CTC_DETECTION_FRACTION * area:
                seeds[int(label)] = best
                continue
        unmatched.append(int(label))

    collisions = {}
    for marker_label, node in seeds.items():
        collisions.setdefault(node, []).append(marker_label)
    collisions = {n: sorted(m) for n, m in collisions.items() if len(m) > 1}

    return seeds, unmatched, collisions


def forward_lineage(graph, seeds):
    """Every node reachable from ``seeds`` by following edges forward in time.

    Divisions keep all children and merges are followed through, so the result
    is the full descendant set of the seeded objects. A node whose other parent
    was not seeded is still kept -- it descends from an evaluated cell -- but
    the edge from the unseeded parent is dropped with that parent.
    """
    keep = set()
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        if node in keep:
            continue
        keep.add(node)
        queue.extend(graph.successors(node))
    return keep


def lineage_survival(graph, seeds, n_frames):
    """How far forward each seeded lineage survives.

    A seeded lineage that stops before the last frame has a missing link
    somewhere, and everything downstream of it is gone from the prediction. This
    is the number to look at before reading a pruned TRA: it says how much of
    the evaluated ground truth the prediction can still cover at best.

    Returns ``{seed_node: last_frame_reached}`` and the count reaching the end.
    """
    last_frame = {}
    for seed in sorted(set(seeds)):
        reached = forward_lineage(graph, [seed])
        last_frame[seed] = max(int(graph.nodes[n]["time"]) for n in reached)
    complete = sum(1 for t in last_frame.values() if t >= n_frames - 1)
    return last_frame, complete


def write_pruned_segmentation(src_path, dst_path, keep, max_label):
    """Copy ``src_path`` to ``dst_path`` with every label outside ``keep`` zeroed.

    The node-id labelling convention is preserved (kept labels keep their value)
    so the pruned store still satisfies the invariant
    ``evaluate_tracking._check_labelled_by_node_id`` enforces. Frames are copied
    one at a time: a full DRO volume is ~19 GB as uint32, one frame is ~380 MB.
    """
    src = zarr.open(str(src_path), mode="r")
    lut = np.zeros(int(max_label) + 1, dtype=src.dtype)
    if keep:
        kept = np.fromiter(sorted(keep), dtype=np.int64, count=len(keep))
        lut[kept] = kept  # identity on kept labels, background everywhere else

    dst = zarr.open(
        str(dst_path), mode="w", shape=src.shape, chunks=src.chunks, dtype=src.dtype
    )
    dst.attrs.update(dict(src.attrs))
    for t in range(src.shape[0]):
        frame = np.asarray(src[t])
        if frame.size:
            highest = int(frame.max())
            if highest >= lut.size:
                raise ValueError(
                    f"Segmentation at {src_path} holds label {highest}, which is "
                    f"not a graph node id (largest is {lut.size - 1}). The "
                    "segmentation and its tracks store are out of step."
                )
        dst[t] = lut[frame]
    return dst


def write_pruned_tracks(graph, metadata, keep, dst_path):
    """Write the subgraph induced on ``keep`` as a geff store.

    ``track_id`` is reassigned over the pruned lineages rather than carried
    over, so the tracklet numbering describes the result that is actually being
    scored.
    """
    from motile_toolbox.visualization.napari_utils import assign_tracklet_ids

    pruned = graph.subgraph(keep).copy()
    # Returns (graph, track_ids, max_id); it is the returned graph that carries
    # the reassigned ids, so it must not be discarded.
    pruned, _, _ = assign_tracklet_ids(pruned)

    axes = metadata.axes or []
    pruned_metadata = geff.GeffMetadata(
        directed=True,
        related_objects=[
            {"type": "labels", "path": "../pred_seg.zarr", "node_prop": "label"}
        ],
        node_props_metadata={},
        edge_props_metadata={},
    )
    geff.write(
        pruned,
        dst_path,
        axis_names=[a.name for a in axes] or None,
        axis_types=[a.type for a in axes] or None,
        axis_scales=[a.scale for a in axes] or None,
        metadata=pruned_metadata,
        overwrite=True,
    )
    return pruned


def prune_to_ctc_seeds(pred_dir, tra_dir, out_dir, dry_run=False):
    """Prune a tracking result to the lineages seeded by the frame-0 CTC markers.

    The source ``config.toml`` is copied into ``out_dir`` so the pruned result
    keeps its provenance and the evaluation diagnostics can still resolve the
    segmentation stage it came from (``fn_analysis`` reads ``seg_result`` out of
    it to separate solver misses from segmentation misses).

    Args:
        pred_dir: tracking result holding ``pred_tracks.zarr`` and
            ``pred_seg.zarr``.
        tra_dir: the CTC ``TRA`` folder. Only its first frame is read.
        out_dir: where to write the pruned result. Laid out like ``pred_dir``,
            so it can be pointed at by ``track_result`` in an eval config.
        dry_run: do the matching and the prune and report on them, but write
            nothing. Use it to read the survival statistics before committing to
            the segmentation copy, which for DRO is ~19 GB.

    Returns the report dict, which is also written to ``ctc_prune_report.json``.
    """
    pred_dir, tra_dir, out_dir = Path(pred_dir), Path(tra_dir), Path(out_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    graph, metadata = geff.read(pred_dir / "pred_tracks.zarr")
    seg = zarr.open(str(pred_dir / "pred_seg.zarr"), mode="r")
    n_frames = int(seg.shape[0])

    markers, marker_path = read_first_frame_markers(tra_dir)
    seeds, unmatched, collisions = seed_from_markers(markers, np.asarray(seg[0]))

    seed_nodes = sorted(set(seeds.values()))
    missing = [n for n in seed_nodes if n not in graph]
    if missing:
        raise ValueError(
            f"Frame-0 labels {missing[:10]} are in pred_seg.zarr but not in "
            "pred_tracks.zarr; the two stores are out of step."
        )

    keep = forward_lineage(graph, seed_nodes)
    last_frame, complete = lineage_survival(graph, seed_nodes, n_frames)

    if dry_run:
        pruned = graph.subgraph(keep)
    else:
        pruned = write_pruned_tracks(
            graph, metadata, keep, out_dir / "pred_tracks.zarr"
        )
        write_pruned_segmentation(
            pred_dir / "pred_seg.zarr",
            out_dir / "pred_seg.zarr",
            keep,
            max_label=max(graph.nodes),
        )
        source_config = pred_dir / "config.toml"
        if source_config.is_file():
            shutil.copy2(source_config, out_dir / "config.toml")

    survival = np.array(sorted(last_frame.values())) if last_frame else np.array([])
    report = {
        "source": str(pred_dir),
        "marker_frame": str(marker_path),
        "n_frames": n_frames,
        "markers": {
            "total": int(len(seeds) + len(unmatched)),
            "detected": len(seeds),
            "undetected": len(unmatched),
            "undetected_labels": unmatched,
            "distinct_seed_objects": len(seed_nodes),
            "merged_seed_objects": len(collisions),
            "merged_marker_groups": {str(k): v for k, v in collisions.items()},
        },
        "graph": {
            "nodes_before": graph.number_of_nodes(),
            "nodes_after": pruned.number_of_nodes(),
            "edges_before": graph.number_of_edges(),
            "edges_after": pruned.number_of_edges(),
        },
        "lineage_survival": {
            "reaching_final_frame": int(complete),
            "of_seeded": len(seed_nodes),
            "last_frame_min": int(survival.min()) if survival.size else None,
            "last_frame_median": float(np.median(survival)) if survival.size else None,
            "last_frame_mean": float(survival.mean()) if survival.size else None,
            "per_seed_last_frame": {str(k): int(v) for k, v in last_frame.items()},
        },
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        with open(out_dir / "ctc_prune_report.json", "w") as handle:
            json.dump(report, handle, indent=2)
    return report


def format_report(report):
    """One-screen summary of :func:`prune_to_ctc_seeds`'s report."""
    markers = report["markers"]
    graph = report["graph"]
    survival = report["lineage_survival"]
    n_frames = report["n_frames"]
    undetected = ""
    if markers["undetected"]:
        undetected = f"  labels {markers['undetected_labels'][:10]}"
    merged = ""
    if markers["merged_seed_objects"]:
        merged = f"  ({markers['merged_seed_objects']} object(s) cover >1 marker)"

    lines = [
        f"Frame-0 markers:        {markers['total']}",
        f"  detected (>50% cover) {markers['detected']}",
        f"  undetected            {markers['undetected']}{undetected}",
        f"  distinct seed objects {markers['distinct_seed_objects']}{merged}",
        "",
        (
            f"Nodes: {graph['nodes_before']} -> {graph['nodes_after']}"
            f"   Edges: {graph['edges_before']} -> {graph['edges_after']}"
        ),
        "",
        (
            f"Seeded lineages reaching frame {n_frames - 1}: "
            f"{survival['reaching_final_frame']} / {survival['of_seeded']}"
        ),
    ]
    if survival["last_frame_median"] is not None:
        lines.append(
            f"  last frame reached: min {survival['last_frame_min']}, "
            f"median {survival['last_frame_median']:.0f}, "
            f"mean {survival['last_frame_mean']:.1f}"
        )
    if report["dry_run"]:
        lines.append("")
        lines.append("Dry run: nothing written.")
    return "\n".join(lines)
