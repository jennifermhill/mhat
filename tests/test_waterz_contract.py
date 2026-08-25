"""Contract test for the waterz agglomeration backend.

waterz is the `[waterz]` extra: Linux-only today, and pinned to a commit of
funkey/waterz rather than PyPI (PyPI's 0.9.4 predates Python 3.11, and tag
v0.9.6 ships a generated evaluate.cpp that includes the pre-3.11
"longintrepr.h"). The whole module is skipped when it is not installed, so a
core install is unaffected.

Two things make this test worth having rather than relying on a clean
`pip install`:

1. **Nothing is compiled until first call.** On current master the agglomerate
   path goes through `witty.compile_cython()`, and that call sits *inside*
   `waterz.agglomerate()` — not at module scope. So neither installing nor
   importing waterz proves the C++ toolchain works. Only calling it does.

2. **`create_seg_hypotheses.py` depends on the merge-history row format**, not
   just on the function existing. At create_seg_hypotheses.py:187
   it does `row['cost'] = row.pop('score')`, so a rename of that field upstream
   would produce a KeyError deep in stage 02 rather than anything obvious here.

Expect the first run to be slow — it compiles a Cython/C++ extension.
"""

from __future__ import annotations

import numpy as np
import pytest

waterz = pytest.importorskip(
    "waterz",
    reason="waterz not installed — it is the optional [waterz] extra (Linux only)",
)


@pytest.fixture
def synthetic_affinities():
    """A tiny volume of four touching fragments with high affinities between them.

    waterz wants affinities as (3, depth, height, width) float32 — the z, y and
    x channels — and fragments as uint64. Affinities are uniformly high so that
    under the default OneMinus<MeanAffinity> scoring every boundary scores low
    and the fragments agglomerate well below the threshold, guaranteeing a
    non-empty merge history.
    """
    depth, height, width = 4, 16, 16
    affs = np.full((3, depth, height, width), 0.9, dtype=np.float32)

    fragments = np.zeros((depth, height, width), dtype=np.uint64)
    fragments[:, :8, :8] = 1
    fragments[:, :8, 8:] = 2
    fragments[:, 8:, :8] = 3
    fragments[:, 8:, 8:] = 4

    return affs, fragments


def test_agglomerate_compiles_and_returns_merge_history(synthetic_affinities):
    """The exact call create_seg_hypotheses.py makes, on a toy volume.

    This is what actually exercises the witty JIT compile, so a broken C++
    toolchain or Boost install fails here rather than midway through a real
    segmentation run.
    """
    affs, fragments = synthetic_affinities

    generator = waterz.agglomerate(
        affs=affs,
        fragments=fragments,
        thresholds=[0.5],
        return_merge_history=True,
    )
    segmentation, merge_history = next(generator)

    assert segmentation.shape == fragments.shape
    assert segmentation.dtype == fragments.dtype

    # Four touching fragments at uniformly high affinity must merge into fewer
    # than four objects, otherwise the scoring never fired and the rest of this
    # test would be vacuous.
    assert len(np.unique(segmentation)) < 4
    assert merge_history, "expected at least one merge at threshold 0.5"


def test_merge_history_row_format(synthetic_affinities):
    """Rows must carry a/b/c/score — the field names stage 02 reads.

    `create_seg_hypotheses.py` renames 'score' to 'cost' and adds 'timepoint',
    producing the a,b,c,cost,timepoint CSV that
    `create_multihypo_graph.load_merge_history` parses. If waterz ever renames
    these, that CSV silently loses a column.
    """
    affs, fragments = synthetic_affinities

    generator = waterz.agglomerate(
        affs=affs,
        fragments=fragments,
        thresholds=[0.5],
        return_merge_history=True,
    )
    _, merge_history = next(generator)

    for row in merge_history:
        for key in ("a", "b", "c", "score"):
            assert key in row, (
                f"merge history row {row!r} is missing {key!r} — "
                "scripts/02_segmentation/create_seg_hypotheses.py reads these by name"
            )
        assert float(row["score"]) >= 0.0

    # Sorted ascending by score is assumed downstream: nodes_from_fragments
    # walks the history in order and requires costs to be non-decreasing.
    scores = [float(row["score"]) for row in merge_history]
    assert scores == sorted(scores), "merge history is not sorted by ascending score"
