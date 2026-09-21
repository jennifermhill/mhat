"""Round-trip contract for the geff / funtracks boundary.

`run_tracking.py` writes tracks with `geff.write`; `evaluate_tracking.py` reads
them back with funtracks' `import_from_geff`. Two things at that boundary fail
silently rather than loudly, and those are what is pinned here:

  * funtracks infers dimensionality from the caller's name map rather than
    from the file, so a stale 3D map on a 2D geff is skipped in silence while
    `ndims` still says 4 -- hence the round trip runs at both ranks;
  * geff-spec 1.2.1 renamed `label_prop` to `node_prop` as separate fields, so
    a reader that only knows one of them quietly falls back to `track_id` and
    remaps the segmentation against the wrong property.
"""

from __future__ import annotations

import geff
import networkx as nx
import numpy as np
import pytest
from funtracks.import_export import import_from_geff

from mhat.evaluation.evaluate_tracking import read_name_map_and_scale

AXIS_NAMES = {3: ["time", "z", "y", "x"], 2: ["time", "y", "x"]}

# Two two-frame tracks. The 3D case is unchanged from before 2D support; the
# 2D case is the same tracks with z dropped.
POSITIONS = {
    3: [(1.0, 10.0, 8.0), (1.0, 10.0, 9.0), (1.0, 10.0, 24.0), (1.0, 10.0, 25.0)],
    2: [(10.0, 8.0), (10.0, 9.0), (10.0, 24.0), (10.0, 25.0)],
}


def unit_scale(ndim):
    return [1.0] * (ndim + 1)


def make_solution_graph(ndim):
    """Two two-frame tracks, shaped like what solve_with_motile returns."""
    names = AXIS_NAMES[ndim][1:]
    graph = nx.DiGraph()
    for node, (time, track_id, position) in enumerate(
        zip((0, 1, 0, 1), (1, 1, 2, 2), POSITIONS[ndim], strict=True), start=1
    ):
        attrs = {"time": time, "track_id": track_id, "label": node}
        attrs.update(dict(zip(names, position, strict=True)))
        graph.add_node(node, **attrs)
    graph.add_edges_from([(1, 2), (3, 4)])
    return graph


def write_tracks(graph, path, ndim, prop_field="node_prop"):
    """Byte-for-byte the geff.write call in run_tracking.py.

    `prop_field` exists so a test can also produce a pre-geff-spec-1.2.1 file,
    which used `label_prop`.
    """
    names = AXIS_NAMES[ndim]
    metadata = geff.GeffMetadata(
        directed=True,
        related_objects=[
            {"type": "labels", "path": "../pred_seg.zarr", prop_field: "label"}
        ],
        node_props_metadata={},
        edge_props_metadata={},
    )
    geff.write(
        graph,
        path,
        axis_names=names,
        axis_types=["time"] + ["space"] * (len(names) - 1),
        axis_scales=unit_scale(ndim),
        metadata=metadata,
        overwrite=True,
    )


@pytest.mark.parametrize("prop_field", ["node_prop", "label_prop"])
def test_seg_label_prop_is_readable_old_and_new(tmp_path, prop_field):
    """`evaluate_tracking` must find the seg-label property on files of both ages.

    The two fields are not aliases -- setting one leaves the other None -- so a
    plain rename in the writers would silently stop `remap_seg_to_track_ids`
    from finding the property on every pred_tracks.zarr already on disk. This
    pins the resolution expression that reader uses. Rank-independent, so 3D
    only.
    """
    path = tmp_path / f"tracks_{prop_field}.zarr"
    write_tracks(make_solution_graph(3), path, 3, prop_field=prop_field)

    _, metadata = geff.read(path)
    ro = metadata.related_objects[0]

    # Exactly the expression in evaluate_tracking.remap_seg_to_track_ids.
    resolved = ro.node_prop or ro.label_prop
    assert resolved == "label", (
        f"a file written with {prop_field!r} resolved to {resolved!r}; "
        "remap_seg_to_track_ids would fall back to 'track_id' and remap "
        "against the wrong property"
    )


@pytest.mark.parametrize("ndim", [2, 3], ids=["2d", "3d"])
def test_import_from_geff_roundtrip(tmp_path, ndim):
    """funtracks reads back the graph MHAT wrote, with positions under 'pos'.

    The name map is the one evaluate_tracking builds from the file itself, so
    this is the real call path rather than an approximation of it.
    """
    path = tmp_path / "correct_tracks.zarr"
    write_tracks(make_solution_graph(ndim), path, ndim)

    # The map must have exactly this file's spatial axes, in order. Adding
    # "seg_id" would make funtracks load the label array, cutting against
    # evaluate_tracking's own remap_seg_to_track_ids.
    name_map, read_scale = read_name_map_and_scale(path)
    assert name_map["time"] == "time"
    assert name_map["pos"] == AXIS_NAMES[ndim][1:]
    assert read_scale == unit_scale(ndim)
    assert "seg_id" not in name_map

    tracks = import_from_geff(
        path, node_name_map=name_map, segmentation_path=None, scale=read_scale
    )

    assert tracks.graph.number_of_nodes() == 4
    assert tracks.graph.number_of_edges() == 2

    # traccuracy is handed location_keys="pos", so funtracks must have collapsed
    # the per-axis properties into a single positional attribute of this rank.
    for _, attrs in tracks.graph.nodes(data=True):
        assert "pos" in attrs, f"expected 'pos' on every node, got {sorted(attrs)}"
        assert len(attrs["pos"]) == ndim
        assert "time" in attrs

    # Positions must survive the round trip unscaled (scale is 1.0 here).
    positions = sorted(
        tuple(np.asarray(d["pos"], dtype=float))
        for _, d in tracks.graph.nodes(data=True)
    )
    assert positions == sorted(POSITIONS[ndim])
