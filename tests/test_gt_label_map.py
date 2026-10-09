"""gt_drift averages the flow over each GT cell's region, found by its label.

Ground truth converted since 2026-09-18 labels its segmentation by node id and
declares ``node_prop = "label"``. Reading ``track_id`` instead (the old default)
looked up the wrong labels: most nodes fell back to centroid sampling, and the
rest averaged the flow over another cell's region, without an error.
"""

from types import SimpleNamespace

import networkx as nx
from geff_spec import RelatedObject

from mhat.evaluation.gt_drift import gt_label_map


def gt(related_objects):
    graph = nx.DiGraph()
    graph.add_node(7, label=7, track_id=1, seg_id=7)
    graph.add_node(8, label=8, track_id=1, seg_id=7)
    return graph, SimpleNamespace(related_objects=related_objects)


def test_node_prop_label_maps_to_node_ids():
    graph, meta = gt([RelatedObject(type="labels", path="../seg", node_prop="label")])
    assert gt_label_map(graph, meta) == {7: 7, 8: 8}


def test_declared_label_prop_is_used():
    graph, meta = gt([RelatedObject(type="labels", path="../seg", label_prop="track_id")])
    assert gt_label_map(graph, meta) == {7: 1, 8: 1}


def test_absent_label_prop_falls_back_to_node_ids():
    graph, meta = gt([RelatedObject(type="labels", path="../seg", label_prop="missing")])
    assert gt_label_map(graph, meta) == {7: 7, 8: 8}
