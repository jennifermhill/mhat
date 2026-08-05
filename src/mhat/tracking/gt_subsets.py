"""Ground-truth subsetting for the "how much GT do you need?" experiment.

Holds raw data, segmentation and optical flow fixed and varies only how much GT
the SSVM fit can see. Two ways of hiding a GT track from the fit are supported:

- **Arm A — graph removal.** Candidate nodes touching no annotated track are
  deleted from the fit graph, together with their edges, and the exclusion sets
  are pruned. This is a *true* ignore. Keeping every track is a strict no-op, so
  the full-GT fit is reproduced exactly.
- **Arm B — `None` masking.** The same candidates keep their place in the graph
  but get ``gt_selected = None``, motile's documented "unlabeled" marker.

A third protocol, **arm C — dense crops**, lives in :mod:`gt_crops`: instead of
scattering the annotation thinner over the whole field of view, keep a smaller box
and annotate everything in it. Both arms here keep the competing hypotheses around
an annotated cell, but neither can label a candidate that overlaps *no* annotated
object — under sparse annotation that is indistinguishable from a cell the
annotator skipped. Inside a crop it is a supervised false positive. That is the
difference the third arm exists to measure.

Both arms here are realizable protocols: "does this candidate touch one of the
cells I annotated?" needs only the annotations in hand. (Before 2026-08-03 the rule was
"touches a *removed* track", which read the hidden GT — see the note in
``gt_annotation.assign_gt_labels``. Arm A under that rule was an oracle; under
the current rule it is simply "restrict fitting to the annotated neighbourhoods".)

Arm B is not equivalent to arm A. ``structsvm.HammingCosts`` zeroes the loss
coefficients for masked variables, but ``SoftMarginLoss`` still builds
``self._d = features @ ground_truth`` and ``a = <f, y'>`` from the *unmasked*
ground-truth vector, which is 0 in masked slots. A masked variable therefore
still enters the margin and the gradient as "GT says do not select" — a soft
negative. Comparing the two arms is the point of the experiment.

Label-space warning
-------------------
Three numbering systems are in play and they do not agree:

1. **geff node id** — what ``correct_seg.zarr`` is actually labelled with.
2. **raw geff ``track_id`` prop** — what the annotation tool wrote.
3. **funtracks-renumbered ``track_id``** — what ``import_from_geff`` produces
   and what ``gt_seg`` carries after ``remap_seg_to_track_ids``.

Sampling and masking happen in space 3 (it is what ``assign_gt_labels`` sees).
Zeroing the segmentation happens in space 1. Reloading a *materialized* masked
geff yields a fourth, different space-3 numbering — the masked pair is
internally self-consistent and drops straight into ``load_gt``, but every
cross-check against ``subset.json`` must go through geff node ids.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import geff
import motile
import networkx as nx
import numpy as np
import zarr


def subset_dirname(n_tracks: int, seed: int) -> str:
    """Directory name for a materialized GT subset."""
    return f"n{n_tracks:03d}_s{seed}"


# --- where runs live under a dataset directory ------------------------------
#
# A sweep's runs can either sit directly under ``<base>/tracking/<exp>/<dataset>/``
# (the original flat layout, and still the default) or be grouped in a
# subdirectory, e.g. ``gt_amount/stage2_fits/``. Grouping keeps a 60-run sweep from
# burying the handful of ordinary runs that share the dataset directory.
#
# Run *tokens* stay flat either way — they are dict keys, log-file stems and the
# CSV ``run`` column, and `parse_run_token` matches a single directory name. Only
# the path to a run and the ``exp_uid`` / ``track_result`` that addresses it carry
# the subdirectory. Both are consumed as ``<root> / <value>``, so a POSIX relative
# path works unchanged in every script that resolves one.


def run_dir(root: Path, subdir: str, token: str) -> Path:
    """Directory for one run under a dataset root, with or without grouping."""
    return root / subdir / token if subdir else root / token


def run_ref(subdir: str, token: str) -> str:
    """The ``exp_uid`` / ``track_result`` that addresses `token` from a dataset root.

    Forward slashes, so the value is identical on Windows and on the cluster — the
    configs are written on one and read on the other.
    """
    return f"{subdir}/{token}" if subdir else token


def iter_run_dirs(root: Path, subdir: str = ""):
    """Yield the run directories of a sweep, sorted, or nothing if none exist yet.

    Callers filter by `token_regex` / `parse_run_token` on ``.name``, so this must
    stay one level deep: the grouping directory holds runs, not more groups.
    """
    base = root / subdir if subdir else root
    if not base.is_dir():
        return
    yield from sorted(p for p in base.iterdir() if p.is_dir())


def run_token(arm: str, size, seed: int, prefix: str = "gta") -> str:
    """Canonical run name, used as fit output_name, test exp_uid and eval track_result.

    `size` is either the track count (int -> ``n050``) or an already-formed size
    token (str, e.g. ``gt_crops.fraction_token(0.25)`` -> ``f250``). Two annotation
    protocols therefore share one flat token namespace, so every downstream script
    can enumerate runs with a single regex.
    """
    size_token = f"n{int(size):03d}" if not isinstance(size, str) else size
    return f"{prefix}_{arm}_{size_token}_s{seed}"


#: Run tokens are ``<prefix>_<arm>_<size>_s<seed>``. Size is ``n<tracks>`` for the
#: sparse arms and the full-GT reference; for the crop arm it is ``f<fraction*1000>``
#: when crops were cut to a volume fraction, or ``b<gt_node_budget>`` when they were
#: grown to an annotation budget.
_TOKEN_PATTERN = (
    r"^{prefix}_(?P<arm>[A-Za-z]+)_"
    r"(?:n(?P<n>\d{{3}})|f(?P<f>\d{{3}})|b(?P<b>\d{{4}}))_s(?P<seed>\d+)$"
)


def token_regex(prefix: str = "gta") -> re.Pattern:
    """Compiled matcher for run tokens produced by `run_token` under `prefix`."""
    return re.compile(_TOKEN_PATTERN.format(prefix=re.escape(prefix)))


def parse_run_token(token: str, prefix: str = "gta") -> dict | None:
    """Decompose a run token, or None if it does not belong to this sweep.

    Returns ``{arm, seed, n_tracks, crop_fraction, crop_target_gt_nodes}``. Exactly
    one of the three size fields is set. `crop_fraction` is recovered from the
    3-digit token and so is only accurate to that; the authoritative values live in
    the run's ``fit_summary.json`` / ``subset.json``.
    """
    match = token_regex(prefix).match(token)
    if not match:
        return None
    return {
        "arm": match.group("arm"),
        "seed": int(match.group("seed")),
        "n_tracks": int(match.group("n")) if match.group("n") else None,
        "crop_fraction": int(match.group("f")) / 1000.0 if match.group("f") else None,
        "crop_target_gt_nodes": int(match.group("b")) if match.group("b") else None,
    }


def sample_track_subsets(
    gt_graph, sizes, seed: int, gt_label_key: str = "track_id"
) -> tuple[dict[int, set[int]], list[int]]:
    """Draw nested random subsets of GT track labels.

    A single permutation per seed, truncated at each size, so smaller subsets are
    strict subsets of larger ones. That makes the learning curve monotone-
    comparable within a seed instead of mixing independent draws.

    Returns:
        ({n_tracks: set of track labels}, full shuffled order).
    """
    all_labels = sorted({int(d[gt_label_key]) for _, d in gt_graph.nodes(data=True)})
    order = [int(x) for x in np.random.default_rng(seed).permutation(all_labels)]
    subsets = {}
    for n in sizes:
        if n > len(order):
            raise ValueError(f"Requested {n} tracks but GT only has {len(order)}")
        subsets[int(n)] = set(order[: int(n)])
    return subsets, order


def copy_track_graph(graph: motile.TrackGraph) -> motile.TrackGraph:
    """Deep-enough copy of a TrackGraph: node/edge attribute dicts are copied.

    Insertion order of ``nodes`` and ``edges`` is preserved. ``motile.Solver``
    builds its feature matrix by iterating those dicts, so their order fixes the
    ILP variable layout — reordering here would silently change the fit.
    """
    new = motile.TrackGraph(frame_attribute=graph.frame_attribute)
    new.nodes = {n: dict(d) for n, d in graph.nodes.items()}
    new.edges = {e: dict(d) for e, d in graph.edges.items()}
    new.prev_edges = defaultdict(list, {k: list(v) for k, v in graph.prev_edges.items()})
    new.next_edges = defaultdict(list, {k: list(v) for k, v in graph.next_edges.items()})
    new._graph_changed = True
    new._update_metadata()
    return new


def remove_nodes(graph: motile.TrackGraph, nodes) -> dict:
    """Delete `nodes` and their incident edges from `graph` in place (arm A).

    ``prev_edges``/``next_edges`` are rebuilt from scratch afterwards. They are
    not bookkeeping: ``MaxParents``, ``MaxChildren``, ``NodeAppear`` and
    ``NodeDisappear`` all look up ``edge_indicators[e]`` through them, so a stale
    entry is a hard KeyError when the solver is constructed.
    """
    nodes = set(nodes)
    if not nodes:
        return {"n_nodes_removed": 0, "n_edges_removed": 0}

    n_edges_before = len(graph.edges)
    for edge in list(graph.edges):
        if graph.is_hyperedge(edge):
            raise NotImplementedError(
                "arm-A graph removal is not defined for hyperedges; this config "
                "should have none (divisions=false, merges=false)"
            )
        u, v = edge
        if u in nodes or v in nodes:
            del graph.edges[edge]

    for node in nodes:
        graph.nodes.pop(node, None)

    graph.prev_edges = defaultdict(list)
    graph.next_edges = defaultdict(list)
    for edge in graph.edges:
        u, v = edge
        graph.prev_edges[v].append(edge)
        graph.next_edges[u].append(edge)

    graph._graph_changed = True
    graph._update_metadata()
    return {
        "n_nodes_removed": len(nodes),
        "n_edges_removed": n_edges_before - len(graph.edges),
    }


def prune_exclusion_sets(exclusion_sets, graph: motile.TrackGraph) -> list[list]:
    """Restrict exclusion sets to surviving nodes, dropping those left with < 2.

    Mirrors the ``len(set) > 1`` filter in create_multihypo_graph.
    """
    pruned = []
    for group in exclusion_sets:
        kept = [n for n in group if n in graph.nodes]
        if len(kept) > 1:
            pruned.append(kept)
    return pruned


def mask_nodes_none(graph: motile.TrackGraph, nodes, gt_attribute: str = "gt_selected") -> dict:
    """Mark `nodes` and every incident edge as unlabeled (arm B)."""
    nodes = set(nodes)
    if not nodes:
        return {"n_nodes_masked": 0, "n_edges_masked": 0}

    for node in nodes:
        graph.nodes[node][gt_attribute] = None

    n_edges_masked = 0
    for edge, data in graph.edges.items():
        if any(n in nodes for n in graph.nodes_of(edge)):
            data[gt_attribute] = None
            n_edges_masked += 1

    return {"n_nodes_masked": len(nodes), "n_edges_masked": n_edges_masked}


def assert_graph_consistent(graph: motile.TrackGraph) -> None:
    """Cheap structural check after surgery. Raises AssertionError on any dangling ref."""
    for edge in graph.edges:
        for node in graph.nodes_of(edge):
            assert node in graph.nodes, f"edge {edge} references missing node {node}"
    for node, edges in graph.prev_edges.items():
        assert node in graph.nodes or not edges, f"prev_edges holds missing node {node}"
        for edge in edges:
            assert edge in graph.edges, f"prev_edges[{node}] holds stale edge {edge}"
    for node, edges in graph.next_edges.items():
        assert node in graph.nodes or not edges, f"next_edges holds missing node {node}"
        for edge in edges:
            assert edge in graph.edges, f"next_edges[{node}] holds stale edge {edge}"


def count_orphan_positives(graph: motile.TrackGraph, frame_attribute: str = "time") -> int:
    """GT-positive nodes at t>0 with no incoming candidate edge.

    Arm A deletes the candidates belonging to removed tracks, so a kept track's
    node can lose predecessors it never should have linked to anyway. This count
    should equal the number of kept tracks starting after t=0; anything higher
    means real supervision was destroyed.
    """
    n = 0
    for node, data in graph.nodes.items():
        if data.get("gt_selected") != 1:
            continue
        if int(data[frame_attribute]) == 0:
            continue
        if not graph.prev_edges.get(node):
            n += 1
    return n


def _hash_node_ids(gt_tracks_path: Path) -> str:
    """md5 of the GT geff node id array, so a regenerated GT invalidates subsets loudly."""
    ids = np.asarray(zarr.open(gt_tracks_path, mode="r")["nodes"]["ids"][:])
    return hashlib.md5(np.ascontiguousarray(ids)).hexdigest()


def materialize_masked_gt(
    gt_data_dir: Path,
    out_dir: Path,
    kept_track_ids,
    ft_graph,
    scale,
    seed: int,
    n_tracks: int,
    n_tracks_total: int,
    overlap_criterion: str,
    shuffle_order=None,
) -> dict:
    """Write a reduced copy of the GT (geff graph + label volume) to `out_dir`.

    The result is a drop-in ``gt_data_dir``: it holds ``correct_tracks.zarr`` and
    ``correct_seg.zarr`` exactly like the full GT, so ``load_gt`` and
    ``evaluate_tracks.py`` can be pointed straight at it. This is what the
    follow-on hand-tuning phase scores against.

    Args:
        gt_data_dir: directory holding the full ``correct_tracks.zarr`` / ``correct_seg.zarr``.
        out_dir: destination directory (created).
        kept_track_ids: track labels to keep, in the funtracks-renumbered space.
        ft_graph: the funtracks graph from ``load_gt`` (node ids == geff node ids).
        scale: per-axis scale, for the geff metadata.

    Returns:
        The contents written to ``subset.json``.
    """
    kept_track_ids = {int(x) for x in kept_track_ids}
    gt_tracks_path = Path(gt_data_dir) / "correct_tracks.zarr"
    gt_seg_path = Path(gt_data_dir) / "correct_seg.zarr"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_graph, _meta = geff.read(gt_tracks_path)

    missing = [n for n in raw_graph.nodes if n not in ft_graph.nodes]
    if missing:
        raise RuntimeError(
            f"{len(missing)} geff nodes are absent from the funtracks graph "
            "(e.g. {missing[:5]}); node ids are not comparable, refusing to mask"
        )
    kept_node_ids = sorted(
        int(n) for n in raw_graph.nodes if int(ft_graph.nodes[n]["track_id"]) in kept_track_ids
    )
    masked = raw_graph.subgraph(kept_node_ids).copy()

    # correct_seg.zarr is labelled with geff node ids (the GT metadata names a
    # `seg_id` prop that does not exist, so remap_seg_to_track_ids falls through
    # to the node id). Write seg_id explicitly here so the masked copy does not
    # depend on that fallback — the value is identical either way.
    nx.set_node_attributes(masked, {n: int(n) for n in masked.nodes}, "seg_id")

    metadata = geff.GeffMetadata(
        directed=True,
        related_objects=[
            {"type": "labels", "path": "../correct_seg.zarr", "label_prop": "seg_id"}
        ],
        node_props_metadata={},
        edge_props_metadata={},
    )
    geff.write(
        masked,
        out_dir / "correct_tracks.zarr",
        axis_names=["time", "z", "y", "x"],
        axis_types=["time", "space", "space", "space"],
        axis_scales=list(scale),
        metadata=metadata,
        overwrite=True,
    )

    # Zero every voxel whose node id was dropped, via a lookup table.
    seg = zarr.open(gt_seg_path, mode="r")[:]
    lut = np.zeros(int(seg.max()) + 1, dtype=seg.dtype)
    keep_arr = np.asarray(kept_node_ids, dtype=np.int64)
    keep_arr = keep_arr[keep_arr < lut.size]
    lut[keep_arr] = keep_arr
    out_seg = zarr.open(
        out_dir / "correct_seg.zarr",
        mode="w",
        shape=seg.shape,
        chunks=(1,) + seg.shape[1:],
        dtype=seg.dtype,
    )
    out_seg[:] = lut[seg]

    info = {
        "seed": int(seed),
        "n_tracks": int(n_tracks),
        "n_tracks_total": int(n_tracks_total),
        "overlap_criterion": overlap_criterion,
        "source_gt_dir": str(gt_data_dir),
        "source_node_ids_md5": _hash_node_ids(gt_tracks_path),
        "kept_track_ids": sorted(kept_track_ids),
        "kept_raw_track_ids": sorted(
            {int(d["track_id"]) for n, d in masked.nodes(data=True) if "track_id" in d}
        ),
        "kept_geff_node_ids": kept_node_ids,
        "shuffle_order": list(shuffle_order) if shuffle_order is not None else None,
        "n_gt_nodes_kept": masked.number_of_nodes(),
        "n_gt_nodes_total": raw_graph.number_of_nodes(),
        "n_gt_edges_kept": masked.number_of_edges(),
        "n_gt_edges_total": raw_graph.number_of_edges(),
    }
    with open(out_dir / "subset.json", "w") as f:
        json.dump(info, f, indent=2)
    return info
