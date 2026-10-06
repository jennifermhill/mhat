"""Detection F1, division F1 and AOGM (``F1AOGMMetrics``) on a hand-built case.

These are the numbers tuning optimizes, and two of their rules are ours rather
than traccuracy's, each wrong in silence if reversed: a CTC non-split counts as
1 TP + (k - 1) FN detections, and divisions are classified on the one-to-one
part of a many-to-one matching.

GT: cell 1 divides into 2 and 3, which continue to 4 and 5. The prediction
divides correctly (11 -> 12, 13) but merges 4 and 5 into one node 14 and adds
an unconnected false positive 15.
"""

from __future__ import annotations

import networkx as nx
import pytest
from traccuracy import TrackingGraph
from traccuracy.matchers._matched import Matched
from traccuracy.metrics import CTCMetrics

from mhat.evaluation.evaluate_tracking import F1AOGMMetrics


def _graph(nodes, edges):
    g = nx.DiGraph()
    for node, (t, y, x) in nodes.items():
        g.add_node(node, time=t, pos=[y, x])
    g.add_edges_from(edges)
    return TrackingGraph(g, frame_key="time", label_key=None, location_keys="pos")


def _matched():
    gt = _graph(
        {1: (0, 5, 5), 2: (1, 0, 0), 3: (1, 10, 10), 4: (2, 0, 0), 5: (2, 10, 10)},
        [(1, 2), (1, 3), (2, 4), (3, 5)],
    )
    pred = _graph(
        {11: (0, 5, 5), 12: (1, 0, 0), 13: (1, 10, 10), 14: (2, 5, 5), 15: (2, 50, 50)},
        [(11, 12), (11, 13), (12, 14)],
    )
    mapping = [(1, 11), (2, 12), (3, 13), (4, 14), (5, 14)]
    return Matched(gt, pred, mapping, {"name": "hand"})


def test_f1_aogm_counts():
    r = F1AOGMMetrics().compute(_matched()).results
    # 3 one-to-one TPs + 1 for the non-split; its second GT cell is an FN.
    assert (r["Detection TP"], r["Detection FP"], r["Detection FN"]) == (4, 1, 1)
    assert r["Detection F1"] == pytest.approx(8 / 10)
    # The division survives dropping the non-split pairs.
    assert (r["Division TP"], r["Division FP"], r["Division FN"]) == (1, 0, 0)
    assert r["Division F1"] == 1.0
    # NS 1 x 5 + FP node 1 x 1 + FN edges (2->4, 3->5) 2 x 1.5.
    assert r["AOGM"] == pytest.approx(9.0)


def test_f1_aogm_after_ctc_metrics():
    """Same answer when ``ctc`` has already flagged the graphs, and the AOGM
    agrees with the one CTCMetrics reports."""
    matched = _matched()
    ctc = CTCMetrics().compute(matched).results
    r = F1AOGMMetrics().compute(matched).results
    assert r["AOGM"] == ctc["AOGM"] == pytest.approx(9.0)
    assert r["Detection F1"] == pytest.approx(8 / 10)
    assert r["Division F1"] == 1.0
