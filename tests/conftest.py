"""Synthetic fixtures for the tracking smoke tests.

Builds a tiny multi-hypothesis tracking problem in memory — no zarr, no disk, no
real microscopy data — so the candidate-graph and ILP code can be exercised in
about a second.

The scene is two blobs drifting in +x, one voxel per frame. Each blob is split
into two adjacent fragments that agglomerate into a parent, which is what gives
the graph its merge history, its cohesion/adhesion/num_leaves node attributes,
and its exclusion sets (a parent and its children can never be selected
together).
"""

from __future__ import annotations

import numpy as np
import pytest

# Scene geometry. Deliberately small: the ILP is solved for real in these tests.
T, Z, Y, X = 4, 3, 32, 32

# Per-frame fragment labels are 10*t + 1 .. 10*t + 4 so that labels are unique
# across time. nodes_from_fragments builds one graph per timepoint and
# run_tracking composes them with nx.compose, which would silently collapse
# nodes if two frames reused a label.
FRAGMENTS_PER_FRAME = 4


def _paint(volume, label, z0, z1, y0, y1, x0, x1):
    volume[z0:z1, y0:y1, x0:x1] = label


@pytest.fixture
def synthetic_fragments():
    """(T, Z, Y, X) uint32 fragment labels: two blobs, two fragments each."""
    fragments = np.zeros((T, Z, Y, X), dtype=np.uint32)

    for t in range(T):
        base = 10 * t
        # Blob A drifts from x=6; its two fragments are stacked in y.
        ax = 6 + t
        _paint(fragments[t], base + 1, 0, 2, 8, 12, ax, ax + 4)
        _paint(fragments[t], base + 2, 0, 2, 12, 16, ax, ax + 4)
        # Blob B drifts from x=22, far enough that it never links to A.
        bx = 22 + t
        _paint(fragments[t], base + 3, 0, 2, 8, 12, bx, bx + 4)
        _paint(fragments[t], base + 4, 0, 2, 12, 16, bx, bx + 4)

    return fragments


@pytest.fixture
def synthetic_raw(synthetic_fragments):
    """Raw intensities: blob A dimmer than blob B, plus deterministic noise."""
    rng = np.random.default_rng(0)
    raw = rng.normal(10.0, 1.0, size=synthetic_fragments.shape).astype(np.float32)

    for t in range(T):
        base = 10 * t
        for label, level in ((base + 1, 100.0), (base + 2, 105.0),
                             (base + 3, 150.0), (base + 4, 155.0)):
            raw[t][synthetic_fragments[t] == label] = level

    return raw


@pytest.fixture
def synthetic_merge_history():
    """Raw merge history rows: (a, b, c, cost, timepoint).

    Two merges per frame, one per blob. Costs vary across the whole array
    because normalize_costs divides by (max - min) and would produce NaNs if
    every cost were identical. Written in the same shape waterz emits, with `c`
    reusing `a`'s label — renumber_merge_history is what makes the ids unique.
    """
    rows = []
    for t in range(T):
        base = 10 * t
        rows.append([base + 1, base + 2, base + 1, 0.20 + 0.05 * t, t])
        rows.append([base + 3, base + 4, base + 3, 0.60 + 0.05 * t, t])
    return np.array(rows, dtype=float)


@pytest.fixture
def tracking_config():
    """ILP parameters chosen to produce a non-empty, non-trivial solution.

    The node costs are negative so detections are worth selecting, and the edge
    costs are negative for short links, so the optimum is "track both blobs
    across all four frames" rather than the degenerate empty solution.
    """
    return {
        # edge costs
        "drift_weight": 1.0,
        "drift_constant": -20.0,
        "area_weight": 1.0,
        "area_constant": -5.0,
        "intensity_weight": 1.0,
        "intensity_constant": -5.0,
        "curvature_weight": 1.0,
        "curvature_constant": -5.0,
        "division_weight": 2.0,
        # node costs
        "cohesion_weight": -1.0,
        "cohesion_constant": -5.0,
        "adhesion_weight": -1.0,
        "adhesion_constant": -5.0,
        "appear_constant": 2.0,
        "disappear_constant": 2.0,
        # graph construction
        "max_children": 3,
        "max_edge_distance": 10,
        "size_threshold": 10,
        "min_merge_cost": 0.0,
        "max_merge_cost": 1.0,
        "divisions": True,
        "verbose": False,
    }


@pytest.fixture
def scale():
    return [1.0, 1.0, 1.0, 1.0]
