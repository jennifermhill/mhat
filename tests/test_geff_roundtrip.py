"""Round-trip contract for the geff / funtracks boundary.

`run_tracking.py` writes tracks with `geff.write`; `evaluate_tracking.py` and
`scripts/04_tracking/visualize_results.py` read them back with funtracks'
`import_from_geff`. Nothing else in the repo pins that boundary, and it is the
one most likely to move: funtracks renamed `name_map` -> `node_name_map`
between 1.4 and 1.8 (keeping the old spelling as a deprecated alias), and MHAT
only keeps working because it passes the map positionally.

These tests exist so that swapping the vendored funtracks checkout for a stock
PyPI release fails loudly here rather than silently downstream.
"""

from __future__ import annotations

import geff
import networkx as nx
import numpy as np
import pytest
from funtracks.import_export import import_from_geff

# The map evaluate_tracking.py and visualize_results.py both use.
NAME_MAP = {
    "time": "time",
    "x": "x",
    "y": "y",
    "z": "z",
    "id": "track_id",
}

SCALE = [1.0, 1.0, 1.0, 1.0]


@pytest.fixture
def solution_graph():
    """Two two-frame tracks, shaped like what solve_with_motile returns."""
    graph = nx.DiGraph()
    nodes = [
        (1, {"time": 0, "z": 1.0, "y": 10.0, "x": 8.0, "track_id": 1, "label": 1}),
        (2, {"time": 1, "z": 1.0, "y": 10.0, "x": 9.0, "track_id": 1, "label": 2}),
        (3, {"time": 0, "z": 1.0, "y": 10.0, "x": 24.0, "track_id": 2, "label": 3}),
        (4, {"time": 1, "z": 1.0, "y": 10.0, "x": 25.0, "track_id": 2, "label": 4}),
    ]
    graph.add_nodes_from(nodes)
    graph.add_edges_from([(1, 2), (3, 4)])
    return graph


def write_tracks(graph, path, prop_field="node_prop"):
    """Byte-for-byte the geff.write call in run_tracking.py.

    `prop_field` exists so a test can also produce a pre-geff-spec-1.2.1 file,
    which used `label_prop`.
    """
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
        axis_names=["time", "z", "y", "x"],
        axis_types=["time", "space", "space", "space"],
        axis_scales=SCALE,
        metadata=metadata,
        overwrite=True,
    )


def test_geff_write_then_read(solution_graph, tmp_path):
    """geff.write produces something geff.read returns as (graph, metadata)."""
    path = tmp_path / "pred_tracks.zarr"
    write_tracks(solution_graph, path)

    read_graph, metadata = geff.read(path)

    assert read_graph.number_of_nodes() == solution_graph.number_of_nodes()
    assert read_graph.number_of_edges() == solution_graph.number_of_edges()

    # evaluate_tracking.py derives `scale` from these axes and would fall back to
    # isotropic if they went missing.
    assert metadata.axes is not None
    assert [a.scale for a in metadata.axes] == SCALE

    # The related_objects entry is how the segmentation is located downstream.
    assert metadata.related_objects
    assert metadata.related_objects[0].node_prop == "label"


def test_seg_label_prop_is_readable_old_and_new(solution_graph, tmp_path):
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
        write_tracks(solution_graph, path, prop_field=prop_field)

        _, metadata = geff.read(path)
        ro = metadata.related_objects[0]

        # Exactly the expression in evaluate_tracking.remap_seg_to_track_ids.
        resolved = ro.node_prop or ro.label_prop
        assert resolved == "label", (
            f"a file written with {prop_field!r} resolved to {resolved!r}; "
            "remap_seg_to_track_ids would fall back to 'track_id' and remap "
            "against the wrong property"
        )


def test_import_from_geff_roundtrip(solution_graph, tmp_path):
    """funtracks reads back the graph MHAT wrote, with positions under 'pos'.

    NOTE the name map is passed positionally, exactly as the callers in
    src/mhat/evaluation/evaluate_tracking.py and
    scripts/04_tracking/visualize_results.py do. Under funtracks >=1.8 the
    second positional parameter is `node_name_map`; `name_map` survives only as
    a deprecated keyword alias. If a funtracks upgrade reorders these, this test
    is what catches it.
    """
    path = tmp_path / "correct_tracks.zarr"
    write_tracks(solution_graph, path)

    tracks = import_from_geff(path, NAME_MAP, segmentation_path=None, scale=SCALE)

    assert tracks.graph is not None
    assert tracks.graph.number_of_nodes() == 4
    assert tracks.graph.number_of_edges() == 2

    # traccuracy is handed location_keys="pos", so funtracks must have collapsed
    # z/y/x into a single positional attribute.
    for _, attrs in tracks.graph.nodes(data=True):
        assert "pos" in attrs, f"expected 'pos' on every node, got {sorted(attrs)}"
        assert len(attrs["pos"]) == 3
        assert "time" in attrs

    # Positions must survive the round trip unscaled (scale is 1.0 here).
    positions = sorted(tuple(np.asarray(d["pos"], dtype=float))
                       for _, d in tracks.graph.nodes(data=True))
    assert positions == sorted([
        (1.0, 10.0, 8.0), (1.0, 10.0, 9.0),
        (1.0, 10.0, 24.0), (1.0, 10.0, 25.0),
    ])


def test_import_from_geff_keyword_form(solution_graph, tmp_path):
    """The keyword spelling MHAT should migrate to gives an identical result.

    Step 4 of the migration switches the three call sites to `node_name_map=`.
    This asserts that change is a no-op before it is made.
    """
    path = tmp_path / "correct_tracks.zarr"
    write_tracks(solution_graph, path)

    positional = import_from_geff(path, NAME_MAP, segmentation_path=None, scale=SCALE)
    keyword = import_from_geff(
        path, node_name_map=NAME_MAP, segmentation_path=None, scale=SCALE
    )

    assert set(positional.graph.nodes) == set(keyword.graph.nodes)
    assert set(positional.graph.edges) == set(keyword.graph.edges)
