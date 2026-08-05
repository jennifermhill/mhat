"""Dense-crop ground-truth reduction for the "how much GT do you need?" experiment.

The sibling module :mod:`gt_subsets` shrinks the annotation budget by hiding whole
GT *tracks* scattered across the full field of view — the annotation gets sparser
but never smaller. This module shrinks it the other way: keep a smaller *box* and
annotate everything inside it. The two describe genuinely different annotation
protocols and are expected to behave differently at the same node budget.

Why a crop is not just "another way to drop tracks"
---------------------------------------------------
Under sparse removal, a candidate touching none of the annotated tracks is
unlabeled: the annotator cannot tell background from a cell they did not annotate,
so asserting ``gt_selected = 0`` there would be supervision they never supplied
(arm A deletes those candidates, arm B marks them ``None``).

That does **not** leave the sparse arms short of negatives — they keep every
competing merge hypothesis around each annotated cell, since those all touch an
annotated track. On NC281-sparse `02_nuclei_denoised_train` arm A's negative
fraction is if anything *higher* than the full graph's (173/923 at N=50 vs
348/2896), because shrinking the GT enriches the survivors in multi-cell merged
hypotheses.

What they cannot supply is the other kind of negative: a candidate overlapping no
annotated object **at all** — a detection that is simply wrong. Those are exactly
what the "touches no annotated track" rule masks, so the sparse arms see precisely
zero of them at every budget (74 such candidates exist; arms A and B label 0 of
them, at N=100 and at N=12 alike). The fit learns to arbitrate between hypotheses
of cells it knows about, and never learns that a detection can be spurious.

Inside a crop the GT is dense, so a contained candidate matching nothing is a
genuine false positive and labelling it ``0`` is exactly what the annotator's work
supports. The crop's negative sample is therefore *unbiased with respect to the
GT* — selected by geometry, not by proximity to an annotated cell — and includes
the spurious detections in the proportion the solver meets them at inference. The
driver reports this directly as ``n_neg_no_gt_overlap``; it is 0 for every sparse
run by construction and non-zero for crops wherever the box covers a
false-positive-prone region.

The membership rule, and why this pairing is exact
--------------------------------------------------
Two different tests, deliberately not the same one:

- a **GT object is annotated** iff it intersects the box (shares ≥ 1 voxel);
- a **candidate enters the fit** iff its bounding box is *fully contained* in the box.

So any GT object overlapping a used candidate shares a voxel with a region that
lies entirely inside the box, hence intersects the box, hence is annotated. No
candidate can ever be labelled ``0`` because of a cell the annotator did not
annotate — the failure mode that made the pre-2026-08-03 sparse rule an oracle.
``assert_no_unannotated_overlap`` re-checks this on real data rather than trusting
the argument.

Nothing here reads the ground truth to decide the crop. The box comes from the
seed alone, and candidate membership is pure geometry, so the arm's masking
decision is *independent of the GT* — a stronger no-oracle property than arm A's
"does this candidate touch an annotated track?", which at least has to look at the
annotations.

Objects are never truncated. A GT object or candidate that straddles the border is
kept whole or dropped whole, so no attribute (area, intensity, IoGT) is computed
from a clipped mask.

Two ways to size a crop
-----------------------
- **By volume fraction** (``sample_crop_boxes``) — every seed cuts the same
  proportion of the field. Simple, but on clumpy ground truth it buys wildly
  different amounts of annotation: on NC281-sparse at 50% of the volume, seed 0
  caught 643 GT nodes and seed 1 caught 1780. That variance lands straight on the
  x-axis, where the whole point is to compare protocols at equal budget.
- **By annotation budget** (``sample_crop_boxes_for_budgets``) — grow the box from
  its seeded centre until it holds about N annotated GT nodes. Seeds then land on
  the same x (824 / 822 / 821 for a target of 821) and the crop *volume* absorbs
  the density difference instead (55% of the field for one seed, 27% for another).

Budget sizing is not an oracle. The search counts only the GT *inside* the box it
is considering — the annotator's own completed work — never the GT outside, and the
box's **position** still comes from the seed alone. Only its size adapts. "Annotate
outward from here until you have drawn N objects, then stop" is an ordinary way to
spend a fixed budget; what this deliberately does not do is move the box toward
denser ground truth.

Boundary transitions are charged normally
-----------------------------------------
A track that walks out of the box loses its successor candidate, so the fit sees a
selected node with no outgoing edge and pays ``disappear_constant`` for it — the
crop is treated as a standalone dataset, which is what cropping the raw data and
re-segmenting would have given. This is a real effect, not a bug: expect the
learned appear/disappear constants to drift downward as the crop shrinks, and read
them with that in mind. (``add_appear_ignore_attr`` / ``add_disappear`` still
exempt t=0 and t=T-1 as always; with an XY-only crop the time range is untouched,
so those stay correct.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geff
import networkx as nx
import numpy as np
import zarr
from scipy import ndimage as ndi

#: Volume axis order used throughout (fragments / gt_seg are (T, Z, Y, X)).
AXIS_NAMES = ("t", "z", "y", "x")


def fraction_token(fraction: float) -> str:
    """Three-digit run-token component for a crop fraction: 0.125 -> ``f125``.

    ``int(x * 1000 + 0.5)`` rather than ``round``: 0.0625 * 1000 is 62.5, and
    Python's banker's rounding would turn it into ``f062`` while 0.0635 also
    lands on 62 — the half-up form keeps neighbouring fractions distinct.
    """
    return f"f{int(float(fraction) * 1000 + 0.5):03d}"


def budget_token(target_gt_nodes: int) -> str:
    """Four-digit run-token component for an annotation budget: 798 -> ``b0798``."""
    target = int(target_gt_nodes)
    if not 0 < target < 10000:
        raise ValueError(f"crop GT-node budget {target} does not fit a 4-digit token")
    return f"b{target:04d}"


def crop_dirname(box_or_token, seed: int | None = None) -> str:
    """Directory name for a materialized crop GT, e.g. ``f125_s3`` or ``b0798_s3``.

    Accepts a `CropBox` (preferred — it knows its own seed and sizing mode) or a
    bare size token plus a seed.
    """
    if isinstance(box_or_token, CropBox):
        return f"{box_or_token.size_token}_s{box_or_token.seed}"
    return f"{box_or_token}_s{seed}"


@dataclass(frozen=True)
class CropBox:
    """A half-open box in (t, z, y, x) pixel coordinates.

    Sized one of two ways. ``target_gt_nodes is None`` means the box was cut to a
    requested *volume* fraction; otherwise it was grown until it contained roughly
    that many annotated GT nodes, and `fraction` is whatever volume that took.
    """

    fraction: float
    seed: int
    starts: tuple[int, int, int, int]
    stops: tuple[int, int, int, int]
    volume_shape: tuple[int, int, int, int]
    axes: tuple[str, ...]
    target_gt_nodes: int | None = None
    realized_gt_nodes: int | None = None

    @property
    def size_token(self) -> str:
        """Run-token component identifying this box's condition."""
        if self.target_gt_nodes is not None:
            return budget_token(self.target_gt_nodes)
        return fraction_token(self.fraction)

    @property
    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(a, b) for a, b in zip(self.starts, self.stops))

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(b - a for a, b in zip(self.starts, self.stops))

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.sizes, dtype=np.int64))

    @property
    def volume_fraction(self) -> float:
        """Realized fraction of the volume, which integer sizes make ≠ `fraction`."""
        return self.n_voxels / float(np.prod(self.volume_shape, dtype=np.int64))

    def contains_box(self, starts, stops) -> bool:
        """Is the half-open box [starts, stops) entirely inside this one?"""
        return all(
            s >= a and e <= b
            for s, e, a, b in zip(starts, stops, self.starts, self.stops)
        )

    def contains_point(self, point) -> bool:
        """Is a (t, z, y, x) pixel coordinate inside this box?"""
        return all(a <= p < b for p, a, b in zip(point, self.starts, self.stops))

    def to_dict(self) -> dict:
        return {
            "fraction": float(self.fraction),
            "seed": int(self.seed),
            "axes": list(self.axes),
            "starts": [int(v) for v in self.starts],
            "stops": [int(v) for v in self.stops],
            "sizes": [int(v) for v in self.sizes],
            "volume_shape": [int(v) for v in self.volume_shape],
            "volume_fraction_realized": self.volume_fraction,
            "target_gt_nodes": self.target_gt_nodes,
            "realized_gt_nodes": self.realized_gt_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CropBox:
        return cls(
            fraction=float(data["fraction"]),
            seed=int(data["seed"]),
            starts=tuple(int(v) for v in data["starts"]),
            stops=tuple(int(v) for v in data["stops"]),
            volume_shape=tuple(int(v) for v in data["volume_shape"]),
            axes=tuple(data["axes"]),
            target_gt_nodes=data.get("target_gt_nodes"),
            realized_gt_nodes=data.get("realized_gt_nodes"),
        )

    def describe(self) -> str:
        parts = []
        for name, a, b in zip(AXIS_NAMES, self.starts, self.stops):
            parts.append(f"{name}[{a}:{b}]")
        return " ".join(parts)


def sample_crop_boxes(
    volume_shape,
    fractions,
    seed: int,
    axes=("y", "x"),
) -> dict[float, CropBox]:
    """Draw nested random crop boxes, one per requested volume fraction.

    A single random centre per seed, shared by every fraction, so — exactly like
    ``sample_track_subsets`` — the smaller crops are strict subsets of the larger
    ones and the learning curve is monotone-comparable within a seed.

    Nesting survives the clamp to the volume: the admissible centre interval for a
    small box strictly contains that of a large one, so a centre clamped for the
    large box still leaves the small box inside it. Integer rounding can violate
    that by a pixel, so each box is additionally clamped into its predecessor,
    which makes the containment exact rather than almost-exact.

    Args:
        volume_shape: (T, Z, Y, X) of the fragments volume.
        fractions: target fractions of the total volume, e.g. [0.5, 0.25, 0.125].
        seed: RNG seed; also stored on each box.
        axes: which axes to shrink. The default ("y", "x") keeps the full time
            range and full z — an annotator choosing a smaller field of view —
            so track lengths, and therefore the link supervision per annotated
            cell, are unchanged.

    Returns:
        {fraction: CropBox}, one entry per requested fraction.
    """
    volume_shape = tuple(int(v) for v in volume_shape)
    if len(volume_shape) != 4:
        raise ValueError(f"expected a 4D (T, Z, Y, X) shape, got {volume_shape}")
    axes = tuple(axes)
    unknown = [a for a in axes if a not in AXIS_NAMES]
    if unknown:
        raise ValueError(f"unknown crop axes {unknown}; choose from {AXIS_NAMES}")
    if not axes:
        raise ValueError("crop_axes is empty; nothing would be cropped")

    fractions = [float(f) for f in fractions]
    for f in fractions:
        if not 0.0 < f <= 1.0:
            raise ValueError(f"crop fraction {f} is not in (0, 1]")
    tokens = {fraction_token(f) for f in fractions}
    if len(tokens) != len(set(fractions)):
        raise ValueError(
            f"crop fractions {sorted(set(fractions))} collide as 3-digit run tokens "
            f"({sorted(tokens)}); run names would not be unique"
        )

    centers = _centers_for_seed(volume_shape, axes, seed)

    boxes: dict[float, CropBox] = {}
    previous: CropBox | None = None
    for fraction in sorted(fractions, reverse=True):
        box = _box_for_fraction(volume_shape, fraction, centers, axes, seed, previous)
        boxes[fraction] = box
        previous = box

    return boxes


def _centers_for_seed(volume_shape, axes, seed: int) -> dict[str, float]:
    """One centre per cropped axis, drawn from the seed before any size is known.

    Shared by every box of a seed, which is what makes the family nested and what
    keeps crop placement independent of where the ground truth happens to be.
    """
    rng = np.random.default_rng(seed)
    return {a: float(rng.random()) * volume_shape[AXIS_NAMES.index(a)] for a in axes}


def _box_for_fraction(
    volume_shape,
    fraction: float,
    centers: dict[str, float],
    axes,
    seed: int,
    enclosing: CropBox | None = None,
    target_gt_nodes: int | None = None,
    realized_gt_nodes: int | None = None,
) -> CropBox:
    """Centred box covering `fraction` of the volume, clamped inside `enclosing`."""
    side_power = 1.0 / len(axes)
    side = fraction**side_power
    starts, stops = [], []
    for i, name in enumerate(AXIS_NAMES):
        extent = volume_shape[i]
        if name not in axes:
            starts.append(0)
            stops.append(extent)
            continue
        size = int(round(extent * side))
        size = max(1, min(extent, size))
        start = int(round(centers[name] - size / 2.0))
        start = max(0, min(extent - size, start))
        if enclosing is not None:
            lo = enclosing.starts[i]
            hi = enclosing.stops[i] - size
            if hi < lo:
                raise AssertionError(
                    f"fraction {fraction} needs {size}px on {name} but the "
                    f"enclosing crop is only {enclosing.sizes[i]}px wide"
                )
            start = max(lo, min(hi, start))
        starts.append(start)
        stops.append(start + size)
    box = CropBox(
        fraction=float(fraction),
        seed=int(seed),
        starts=tuple(starts),
        stops=tuple(stops),
        volume_shape=tuple(volume_shape),
        axes=tuple(axes),
        target_gt_nodes=target_gt_nodes,
        realized_gt_nodes=realized_gt_nodes,
    )
    if enclosing is not None:
        assert enclosing.contains_box(box.starts, box.stops), (
            f"crop nesting broken: {box.describe()} is not inside {enclosing.describe()}"
        )
    return box


@dataclass
class GtVoxelIndex:
    """Every annotated GT voxel as flat coordinate arrays, for fast box queries.

    Growing a crop to an annotation budget asks "how many GT nodes are in this box?"
    hundreds of times per seed. Answering that by slicing the label volume costs a
    pass over ~65M voxels each time; here the GT's own voxels (~10^6, since GT is
    sparse in a mostly-empty volume) are extracted once and every query is a
    boolean mask over that much smaller array.

    A "GT node" is one (frame, track label) pair — an object in a frame, which is
    the unit an annotator actually draws.
    """

    coords: np.ndarray  # (4, n_voxels) int32, rows ordered (t, z, y, x)
    labels: np.ndarray  # (n_voxels,) int64 track label
    node_key: np.ndarray  # (n_voxels,) int64, unique per (frame, label)

    def _mask_in(self, box: CropBox) -> np.ndarray:
        inside = np.ones(self.labels.shape, dtype=bool)
        for axis in range(4):
            lo, hi = box.starts[axis], box.stops[axis]
            if lo == 0 and hi >= box.volume_shape[axis]:
                continue  # uncropped axis: skip the comparison entirely
            row = self.coords[axis]
            inside &= (row >= lo) & (row < hi)
        return inside

    def count_nodes_in(self, box: CropBox) -> int:
        """Distinct (frame, track label) pairs with at least one voxel in `box`."""
        return int(np.unique(self.node_key[self._mask_in(box)]).size)

    def labels_in(self, box: CropBox) -> set[int]:
        """Track labels with at least one voxel in `box`."""
        return {int(v) for v in np.unique(self.labels[self._mask_in(box)])}


def build_gt_voxel_index(gt_seg: np.ndarray) -> GtVoxelIndex:
    """Extract the nonzero voxels of a track-id-remapped GT volume, once."""
    coords = np.nonzero(gt_seg)
    labels = np.asarray(gt_seg[coords], dtype=np.int64)
    stacked = np.stack([np.asarray(c, dtype=np.int32) for c in coords], axis=0)
    # Background is already excluded (nonzero), so labels are >= 1 and this is a
    # collision-free pairing of frame with label.
    max_label = int(labels.max()) if labels.size else 0
    node_key = stacked[0].astype(np.int64) * (max_label + 1) + labels
    return GtVoxelIndex(coords=stacked, labels=labels, node_key=node_key)


def grow_crop_to_gt_budget(
    gt_index: GtVoxelIndex,
    volume_shape,
    target_gt_nodes: int,
    seed: int,
    axes=("y", "x"),
    step: float = 0.05,
    max_refine: int = 20,
) -> tuple[float, int]:
    """Grow a seeded crop until it holds about `target_gt_nodes` annotated GT nodes.

    Returns ``(fraction, realized_gt_nodes)`` for the best-matching box.

    Why this is not an oracle: the search counts only the GT *inside* the box it is
    considering — the annotator's own completed work. It never consults the GT
    outside, and the crop's placement still comes from the seed alone. "Keep
    annotating outward from here until you have drawn N objects, then stop" is an
    ordinary way to spend a fixed annotation budget, and it is exactly what this
    computes. What it deliberately does *not* do is move the box toward denser
    ground truth; only the size adapts, never the position.

    The count is monotone non-decreasing in fraction, because the boxes form a
    nested family around one centre, so a coarse scan on `step`-sized increments
    brackets the target and a bisection refines it. Whichever of the bracketing
    pair lands closer to the target wins, so a target between two achievable counts
    resolves to the nearer one rather than always overshooting.
    """
    if target_gt_nodes <= 0:
        raise ValueError(f"crop GT-node budget must be positive, got {target_gt_nodes}")
    volume_shape = tuple(int(v) for v in volume_shape)
    centers = _centers_for_seed(volume_shape, axes, seed)

    def count_at(fraction: float) -> int:
        box = _box_for_fraction(volume_shape, fraction, centers, axes, seed)
        return gt_index.count_nodes_in(box)

    total = count_at(1.0)
    if target_gt_nodes > total:
        raise ValueError(
            f"crop GT-node budget {target_gt_nodes} exceeds the {total} GT nodes in "
            "the whole volume; no crop can reach it"
        )

    # Coarse scan: first step-grid fraction that reaches the target.
    lo_f, lo_count = 0.0, 0
    hi_f, hi_count = 1.0, total
    fraction = step
    while fraction < 1.0:
        count = count_at(fraction)
        if count >= target_gt_nodes:
            hi_f, hi_count = fraction, count
            break
        lo_f, lo_count = fraction, count
        fraction += step

    # Bisect for the smallest fraction still reaching the target.
    for _ in range(max_refine):
        if hi_f - lo_f < 1e-4:
            break
        mid = 0.5 * (lo_f + hi_f)
        count = count_at(mid)
        if count >= target_gt_nodes:
            hi_f, hi_count = mid, count
        else:
            lo_f, lo_count = mid, count

    # The bracketing pair straddles the target; take whichever is closer.
    if lo_f > 0.0 and abs(lo_count - target_gt_nodes) < abs(hi_count - target_gt_nodes):
        return lo_f, lo_count
    return hi_f, hi_count


def sample_crop_boxes_for_budgets(
    gt_index: GtVoxelIndex,
    volume_shape,
    targets,
    seed: int,
    axes=("y", "x"),
    step: float = 0.05,
) -> dict[int, CropBox]:
    """Nested crop boxes, one per annotated-GT-node budget.

    The fixed-fraction sibling (`sample_crop_boxes`) makes every seed cut the same
    *volume*, which on clumpy ground truth means wildly different annotation
    budgets — on NC281-sparse at 50% of the volume, seed 0 caught 643 GT nodes and
    seed 1 caught 1780. Sizing by budget instead puts every seed on the same x
    value, so seed variance stops being x-axis noise and the crop arm can be
    compared to the sparse arms point-for-point.

    Returns {target: CropBox}, each box carrying its realized count.
    """
    volume_shape = tuple(int(v) for v in volume_shape)
    axes = tuple(axes)
    targets = sorted({int(t) for t in targets}, reverse=True)
    tokens = {budget_token(t) for t in targets}
    if len(tokens) != len(targets):
        raise ValueError(f"crop budgets {targets} collide as run tokens ({sorted(tokens)})")

    centers = _centers_for_seed(volume_shape, axes, seed)
    boxes: dict[int, CropBox] = {}
    previous: CropBox | None = None
    for target in targets:
        fraction, realized = grow_crop_to_gt_budget(
            gt_index, volume_shape, target, seed, axes=axes, step=step
        )
        box = _box_for_fraction(
            volume_shape,
            fraction,
            centers,
            axes,
            seed,
            enclosing=previous,
            target_gt_nodes=target,
            realized_gt_nodes=realized,
        )
        # Nesting can clamp the box by a pixel, which can move the count off what
        # the search measured. Re-count so the recorded number is the truth.
        box = CropBox(
            fraction=box.fraction,
            seed=box.seed,
            starts=box.starts,
            stops=box.stops,
            volume_shape=box.volume_shape,
            axes=box.axes,
            target_gt_nodes=target,
            realized_gt_nodes=gt_index.count_nodes_in(box),
        )
        boxes[target] = box
        previous = box
    return boxes


def compute_node_bboxes(
    track_graph,
    fragments: np.ndarray,
    node_to_fragments: dict[int, list[int]],
    frame_key: str = "time",
) -> dict[int, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """Pixel-space bounding box of every candidate node, as (starts, stops) in (t,z,y,x).

    A merge-hypothesis node owns several leaf fragments, so its box is the union of
    theirs. One ``find_objects`` pass per frame gives every fragment's extent at
    once — far cheaper than materialising a boolean mask per candidate, which is
    what the overlap sweep in ``gt_annotation`` has to do.
    """
    nodes_by_time: dict[int, list[int]] = {}
    for node, data in track_graph.nodes.items():
        t = data.get(frame_key)
        if t is None:
            continue  # hypernode: no volume of its own
        nodes_by_time.setdefault(int(t), []).append(node)

    bboxes: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for t, nodes in nodes_by_time.items():
        frame = np.asarray(fragments[t])
        max_label = int(frame.max()) if frame.size else 0
        if max_label == 0:
            continue
        # find_objects wants a signed int type and indexes 1..max_label densely.
        objects = ndi.find_objects(frame.astype(np.int32, copy=False))
        for node in nodes:
            lo = [None, None, None]
            hi = [None, None, None]
            for frag_id in node_to_fragments.get(node, [int(node)]):
                frag_id = int(frag_id)
                if frag_id < 1 or frag_id > len(objects):
                    continue  # merged/intermediate id, never present in the volume
                found = objects[frag_id - 1]
                if found is None:
                    continue
                for axis, sl in enumerate(found):
                    lo[axis] = sl.start if lo[axis] is None else min(lo[axis], sl.start)
                    hi[axis] = sl.stop if hi[axis] is None else max(hi[axis], sl.stop)
            if lo[0] is None:
                raise RuntimeError(
                    f"candidate {node} at t={t} has no voxels in the fragment volume; "
                    "the candidate graph and the segmentation disagree"
                )
            bboxes[node] = (
                (t, int(lo[0]), int(lo[1]), int(lo[2])),
                (t + 1, int(hi[0]), int(hi[1]), int(hi[2])),
            )
    return bboxes


def nodes_outside_crop(
    track_graph,
    box: CropBox,
    node_bboxes: dict | None = None,
    membership: str = "contained",
    scale=None,
    frame_key: str = "time",
) -> set:
    """Candidates that the crop does NOT supervise — to be removed from the fit graph.

    Args:
        node_bboxes: output of ``compute_node_bboxes``; required for
            ``membership="contained"``.
        membership: ``"contained"`` (default) keeps a candidate only if its whole
            bounding box lies inside the crop. This is what makes the
            annotated-GT/used-candidate pairing exact (see the module docstring),
            at the cost of dropping the border-straddling candidates.
            ``"centroid"`` keeps any candidate whose centroid is inside; cheaper
            and keeps more candidates, but a border candidate can then overlap a
            GT object lying mostly outside, so use it only with
            ``assert_no_unannotated_overlap`` watching.
        scale: (t, z, y, x) voxel scale, required for ``membership="centroid"``
            because node centroids are in world units and the box is in pixels.

    Nothing here consults the ground truth: the decision is the crop box and the
    candidate's own geometry.
    """
    if membership not in ("contained", "centroid"):
        raise ValueError(f"membership must be 'contained' or 'centroid', got {membership!r}")
    if membership == "contained" and node_bboxes is None:
        raise ValueError("membership='contained' needs node_bboxes from compute_node_bboxes")
    if membership == "centroid":
        if scale is None:
            raise ValueError("membership='centroid' needs the voxel scale")
        zyx_scale = tuple(float(s) for s in list(scale)[1:4])

    outside = set()
    for node, data in track_graph.nodes.items():
        t = data.get(frame_key)
        if t is None:
            continue  # hypernodes carry no position; leave them to the caller
        if membership == "contained":
            found = node_bboxes.get(node)
            if found is None:
                outside.add(node)
                continue
            starts, stops = found
            if not box.contains_box(starts, stops):
                outside.add(node)
        else:
            centroid = data.get("centroid")
            if centroid is None:
                outside.add(node)
                continue
            # `centroid` is in world units; the box is in pixels.
            pixel = [int(np.floor(float(c) / s)) for c, s in zip(centroid, zyx_scale)]
            if not box.contains_point((int(t), *pixel)):
                outside.add(node)
    return outside


def gt_track_labels_in_crop(gt_index: GtVoxelIndex, box: CropBox) -> set[int]:
    """GT track labels annotated by this crop: those with ≥ 1 voxel inside the box.

    Build `gt_index` with `build_gt_voxel_index` from a GT volume already remapped
    to track ids (``remap_seg_to_track_ids``). This is the ``keep_gt_labels`` set
    handed to ``assign_gt_labels``.
    """
    return gt_index.labels_in(box)


def count_gt_nodes_in_crop(gt_index: GtVoxelIndex, box: CropBox) -> int:
    """Annotated GT *nodes*: distinct (frame, track label) pairs touching the box.

    This is the annotation-cost proxy the learning curve is plotted against — an
    annotator pays per object per frame, and it is the only budget measure the
    crop arm and the sparse arms share.
    """
    return gt_index.count_nodes_in(box)


def _crop_view(volume: np.ndarray, box: CropBox) -> np.ndarray:
    """Box slice of a (T, Z, Y, X) volume, tolerating a shorter time axis."""
    t0, t1 = box.starts[0], min(box.stops[0], volume.shape[0])
    if t1 <= t0:
        return volume[0:0]
    return volume[(slice(t0, t1), *box.slices[1:])]


def assert_no_unannotated_overlap(
    surviving_nodes,
    overlaps,
    keep_gt_labels: set[int],
) -> None:
    """Gate: no surviving candidate touches a GT track the crop did not annotate.

    This is the property that makes a dense-crop label of ``0`` honest. It follows
    from "candidate contained in the box" + "GT annotated iff it intersects the
    box", but the whole point of the experiment is that this kind of reasoning is
    easy to get wrong — so check it against the real overlap matrices instead.

    Raises:
        AssertionError naming the offending candidate and GT label.
    """
    surviving = set(surviving_nodes)
    for t, frame in overlaps.frames.items():
        touches = frame.iou > 0.0
        for row, cand in enumerate(frame.cand_nodes):
            if cand not in surviving:
                continue
            for col in np.flatnonzero(touches[row]):
                label = int(frame.gt_labels[int(col)])
                if label not in keep_gt_labels:
                    raise AssertionError(
                        f"candidate {cand} (t={t}) survives the crop but overlaps GT "
                        f"track {label}, which the crop did not annotate — labelling "
                        "it gt_selected=0 would be supervision the annotator could "
                        "not supply"
                    )


def materialize_cropped_gt(
    gt_data_dir: Path,
    out_dir: Path,
    box: CropBox,
    ft_graph,
    scale,
    n_gt_tracks_total: int,
    membership: str = "contained",
) -> dict:
    """Write the GT an annotator working in `box` would actually possess.

    A drop-in ``gt_data_dir``: ``correct_tracks.zarr`` + ``correct_seg.zarr``, so
    ``load_gt`` and ``evaluate_tracks.py`` point straight at it. Kept GT objects
    are written **whole**, never clipped to the box — an annotator who annotates a
    cell annotates all of it, and clipping would corrupt every area/IoGT the fit
    and the evaluation compute.

    A GT node is kept iff its segment has at least one voxel inside the box, which
    is the same test ``gt_track_labels_in_crop`` applies, just resolved per node
    rather than per track. A track that wanders out of the box therefore comes back
    truncated (possibly into several pieces) — that is the annotation, not a defect.

    Note ``correct_seg.zarr`` is labelled with **geff node ids**, while
    ``gt_track_labels_in_crop`` reads the track-id-remapped volume. Both are views
    of the same voxels, so the two label sets correspond exactly.
    """
    gt_tracks_path = Path(gt_data_dir) / "correct_tracks.zarr"
    gt_seg_path = Path(gt_data_dir) / "correct_seg.zarr"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_graph, _meta = geff.read(gt_tracks_path)
    seg = zarr.open(gt_seg_path, mode="r")[:]

    in_box = sorted(int(v) for v in np.unique(_crop_view(seg, box)) if int(v) != 0)
    # A handful of labels in correct_seg.zarr have no node in correct_tracks.zarr
    # (3 of them on NC281-sparse 02_nuclei_denoised_train). They are dropped, not
    # kept, because `remap_seg_to_track_ids` builds its output only from graph
    # nodes and so already renders them background — the fit never sees them, and a
    # materialized crop that kept them would not match the GT the fit was labeled
    # against. Reported rather than silent: a large count means the GT pair is
    # genuinely inconsistent, not just ragged at the edges.
    orphans = [n for n in in_box if n not in raw_graph.nodes]
    kept_node_ids = [n for n in in_box if n in raw_graph.nodes]
    masked = raw_graph.subgraph(kept_node_ids).copy()

    # Mirror materialize_masked_gt: write seg_id explicitly so the masked copy does
    # not rely on remap_seg_to_track_ids falling through to the node id.
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

    # Keep every retained object whole — zero only the objects that never touch the box.
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

    kept_track_ids = sorted(
        {int(ft_graph.nodes[n]["track_id"]) for n in kept_node_ids if n in ft_graph.nodes}
    )
    info = {
        "kind": "crop",
        "seed": int(box.seed),
        "crop": box.to_dict(),
        "membership": membership,
        "source_gt_dir": str(gt_data_dir),
        "kept_track_ids": kept_track_ids,
        "kept_geff_node_ids": kept_node_ids,
        "n_tracks": len(kept_track_ids),
        "n_tracks_total": int(n_gt_tracks_total),
        "n_gt_nodes_kept": masked.number_of_nodes(),
        "n_gt_nodes_total": raw_graph.number_of_nodes(),
        "n_gt_edges_kept": masked.number_of_edges(),
        "n_gt_edges_total": raw_graph.number_of_edges(),
        "n_orphan_seg_labels_dropped": len(orphans),
        "orphan_seg_labels_dropped": orphans[:20],
    }
    with open(out_dir / "subset.json", "w") as f:
        json.dump(info, f, indent=2)
    return info
