"""Round-trip contract for the geff / funtracks boundary, at both ranks.

`run_tracking.py` writes tracks with `geff.write`; `evaluate_tracking.py` and
`scripts/04_tracking/visualize_results.py` read them back with funtracks'
`import_from_geff`. Nothing else in the repo pins that boundary, and it is the
one most likely to move: funtracks renamed `name_map` -> `node_name_map`
between 1.4 and 1.8 (keeping the old spelling as a deprecated alias), and MHAT
only keeps working because the map is accepted under that name.

These tests exist so that swapping the vendored funtracks checkout for a stock
PyPI release fails loudly here rather than silently downstream. They run at
both ranks because funtracks infers dimensionality from the caller's name map
rather than from the file, so the map MHAT builds has to follow the file.
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


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def ndim(request):
    return request.param


@pytest.fixture
def scale(ndim):
    return [1.0] * (ndim + 1)


@pytest.fixture
def solution_graph(ndim):
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


def write_tracks(graph, path, ndim, scale, prop_field="node_prop"):
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
        axis_scales=scale,
        metadata=metadata,
        overwrite=True,
    )


def test_geff_write_then_read(solution_graph, tmp_path, ndim, scale):
    """geff.write produces something geff.read returns as (graph, metadata)."""
    path = tmp_path / "pred_tracks.zarr"
    write_tracks(solution_graph, path, ndim, scale)

    read_graph, metadata = geff.read(path)

    assert read_graph.number_of_nodes() == solution_graph.number_of_nodes()
    assert read_graph.number_of_edges() == solution_graph.number_of_edges()

    # read_name_map_and_scale derives `scale` from these axes and would fall
    # back to isotropic 3D if they went missing.
    assert metadata.axes is not None
    assert [a.name for a in metadata.axes] == AXIS_NAMES[ndim]
    assert [a.scale for a in metadata.axes] == scale

    # The related_objects entry is how the segmentation is located downstream.
    assert metadata.related_objects
    assert metadata.related_objects[0].node_prop == "label"


def test_seg_label_prop_is_readable_old_and_new(solution_graph, tmp_path, ndim, scale):
    """`evaluate_tracking` must find the seg-label property on files of both ages.

    geff-spec 1.2.1 renamed `label_prop` to `node_prop`. They are separate
    fields rather than aliases — setting one leaves the other None — so a plain
    rename in the writers would silently stop `remap_seg_to_track_ids` from
    finding the property on every pred_tracks.zarr already on disk, and it would
    quietly fall back to "track_id".

    This pins the resolution expression that reader uses.
    """
    for prop_field in ("node_prop", "label_prop"):
        path = tmp_path / f"tracks_{prop_field}.zarr"
        write_tracks(solution_graph, path, ndim, scale, prop_field=prop_field)

        _, metadata = geff.read(path)
        ro = metadata.related_objects[0]

        # Exactly the expression in evaluate_tracking.remap_seg_to_track_ids.
        resolved = ro.node_prop or ro.label_prop
        assert resolved == "label", (
            f"a file written with {prop_field!r} resolved to {resolved!r}; "
            "remap_seg_to_track_ids would fall back to 'track_id' and remap "
            "against the wrong property"
        )


def test_import_from_geff_roundtrip(solution_graph, tmp_path, ndim, scale):
    """funtracks reads back the graph MHAT wrote, with positions under 'pos'.

    The name map is the one evaluate_tracking builds from the file itself, so
    this is the real call path rather than an approximation of it.
    """
    path = tmp_path / "correct_tracks.zarr"
    write_tracks(solution_graph, path, ndim, scale)

    # The map must have exactly this file's spatial axes, in order: a stale 3D
    # map on a 2D file is skipped in silence by funtracks' rename loop while its
    # ndims still says 4. Adding "seg_id" would make funtracks load the label
    # array, cutting against evaluate_tracking's own remap_seg_to_track_ids.
    name_map, read_scale = read_name_map_and_scale(path)
    assert name_map["time"] == "time"
    assert name_map["pos"] == AXIS_NAMES[ndim][1:]
    assert read_scale == scale
    assert "seg_id" not in name_map

    tracks = import_from_geff(
        path, node_name_map=name_map, segmentation_path=None, scale=read_scale
    )

    assert tracks.graph is not None
    assert tracks.graph.number_of_nodes() == 4
    assert tracks.graph.number_of_edges() == 2

    # traccuracy is handed location_keys="pos", so funtracks must have collapsed
    # the per-axis properties into a single positional attribute of this rank.
    for _, attrs in tracks.graph.nodes(data=True):
        assert "pos" in attrs, f"expected 'pos' on every node, got {sorted(attrs)}"
        assert len(attrs["pos"]) == ndim
        assert "time" in attrs

    # Positions must survive the round trip unscaled (scale is 1.0 here).
    positions = sorted(tuple(np.asarray(d["pos"], dtype=float))
                       for _, d in tracks.graph.nodes(data=True))
    assert positions == sorted(POSITIONS[ndim])
