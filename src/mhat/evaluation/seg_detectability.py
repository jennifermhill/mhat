"""Can the segmentation *hypothesis set* detect each CTC ground-truth marker?

This is an upstream measurement. Every other evaluation in this package scores a
tracking *solution*; this one scores the candidate pool the ILP gets to choose
from, and so separates "the tracker made the wrong choice" from "the right
choice was never on the menu". That distinction is what the FN-node diagnostics
on Fluo-N3DL-DRO could not make: 91% of the false-negative GT nodes had no
prediction of any kind, and the contention analysis put the blame on objects
that simply are not in the segmentation.

The criterion is the CTC one, applied to a candidate rather than to a solution
object: a candidate ``S`` detects a reference marker ``R`` iff

    |R & S| > 0.5 * |R|

Because that threshold is strictly greater than half of ``R``, at most one
candidate per *exclusion set* can satisfy it, but candidates from different
levels of the merge tree overlap freely -- so the question here is only whether
**some** candidate does, not which one the solver would pick.

The candidate pool is reconstructed exactly as ``nodes_from_fragments`` builds
it, because a hypothesis that pool excludes is not available to the tracker no
matter how good the underlying fragments are:

* base fragments, and every merge-tree node whose (normalised) merge cost lies
  in ``[min_merge_cost, max_merge_cost)``;
* all of them filtered by ``size_threshold``.

Rather than materialising each hypothesis mask, overlaps are accumulated up the
merge tree: a merge node's intersection with ``R`` is the sum of its two
children's, since the children are disjoint by construction. That makes the
whole candidate pool measurable in one pass over each frame.

Three numbers are reported per marker, and their differences are the diagnosis:

``frag``      best fraction using base fragments alone.
``gated``     best over the pool the tracker actually sees. **This is the one
              that matters** -- ``gated`` recall is the ceiling on DET.
``ungated``   best over the entire merge tree, ignoring cost and size filters.

``frag << gated`` means fragments are over-segmented but agglomeration repairs
them. ``gated < ungated`` means the repair exists but ``min_merge_cost`` /
``max_merge_cost`` / ``size_threshold`` are throwing it away -- a config fix, not
a segmentation one. ``ungated`` still short of 0.5 is the real failure: no union
of fragments covers the marker, which means a fragment boundary runs through it
and only re-segmentation can help.

Recall alone is a weak test on this dataset, and reporting it alone would be
misleading. The CTC ``TRA`` markers are 75-voxel seeds, not nucleus outlines, so
"covers more than half the marker" is satisfied by any blob that swallows the
seed -- Fluo-N3DL-DRO already scores 0.9915 while its tracking falls apart. Two
further measurements say whether the detecting object is the *right* object:

``collisions``  markers whose best candidate is also some other marker's best
                candidate. One blob spanning two nuclei is exactly the
                ``MaxParents(1)`` contention that ends 64 of 71 DRO lineages:
                both predecessors reach for one successor and only one may have
                it. Every collision is a link the tracker cannot make regardless
                of cost.
``offset``      distance from the marker centroid to its candidate's centroid,
                in micrometres. GT flow-corrected drift between frames is ~1 um,
                so an object whose centre sits several micrometres off the
                nucleus makes the correct link look longer than a wrong one --
                the detection is nominally right and the geometry is not.

``over_merge`` markers whose smallest covering candidate exceeds
                ``OVER_MERGE_RATIO`` marker volumes, i.e. is too big to be one
                cell. This is the annotation-independent version of
                ``collisions``: a collision needs *two* annotated markers to
                claim one candidate, and with only ~4% of Fluo-N3DL-DRO objects
                annotated, almost every fusion pairs an annotated nucleus with
                an unannotated one and is missed. Over-merge needs one marker,
                because the fused blob is simply twice the size. Measured on the
                aniso5_fs1 break events, 93% of the fused candidates are base
                cellpose fragments rather than merge-tree nodes, so this is a
                segmentation defect and not an agglomeration one.

**Rank on ``over_merge_rate``**, with ``collision_rate`` secondary and recall as
the constraint that must not regress. Ranking on ``collision_rate`` alone
reorders the top of the table: on the DRO waves the two agree only to Spearman
0.75, and the collisions-winner ``aniso5_cp0_fs1`` falls to 12th on over-merge.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# CTC's detection rule. Strictly greater than, per the benchmark definition.
CTC_DETECTION_FRACTION = 0.5

# A detected marker whose smallest covering candidate exceeds this many marker
# volumes is fused with at least one neighbour. The DRO median is ~11 (= one
# nucleus, a 75-voxel marker inside a ~710-voxel nucleus), so ~2x that is the
# point where the candidate can no longer be a single cell.
#
# This is the annotation-independent companion to `collisions`. A collision
# needs TWO annotated markers to claim one candidate, but only ~4% of DRO
# objects are annotated, so nearly every fusion is between an annotated nucleus
# and an unannotated neighbour and stays invisible to it. The over-merge rate
# needs only one annotated marker, because the fused blob is simply twice the
# size. Rank segmentations on it; see the module docstring.
OVER_MERGE_RATIO = 20.0


def load_merge_history(path):
    """The merge history as ``(a, b, c, cost, timepoint)`` rows, cost-sorted.

    Deliberately a local copy of ``create_multihypo_graph.load_merge_history``
    rather than an import: this module has to stay readable against a *saved*
    segmentation run, including old ones written before the ``score`` ->
    ``cost`` column rename, without pulling in the tracking package.
    """
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            rows.append([
                int(row["a"]), int(row["b"]), int(row["c"]),
                float(row.get("cost", row.get("score", 0.0))),
                int(row["timepoint"]),
            ])
    if not rows:
        return np.empty((0, 5))
    return np.array(rows, dtype=np.float64)


def prepare_merge_history(merge_history, max_node_id, n_frames):
    """Normalise and renumber exactly as ``run_tracking`` does before building.

    Both steps are global over the frames being tracked, so they must be applied
    to the whole (truncated) history at once -- normalising per frame would
    rescale each frame's costs against a different range and silently change
    which merges the ``min_merge_cost``/``max_merge_cost`` gate admits.
    """
    merge_history = merge_history[merge_history[:, 4] < n_frames].copy()
    if len(merge_history) == 0:
        return merge_history

    costs = merge_history[:, 3]
    spread = costs.max() - costs.min()
    if spread > 0:
        merge_history[:, 3] = (costs - costs.min()) / spread
    else:
        merge_history[:, 3] = 0.0

    for idx in range(merge_history.shape[0]):
        max_node_id += 1
        c = merge_history[idx][2]
        merge_history[idx][2] = max_node_id
        later = merge_history[idx + 1:, :3]
        later[later == c] = max_node_id

    return merge_history


def marker_overlaps(gt_frame, frag_frame):
    """Voxel counts of every (marker, fragment) pair, plus marker sizes.

    Only the marker voxels are visited. CTC markers are a few hundred voxels
    each against ~96M in a DRO frame, so this is three orders of magnitude
    cheaper than pairing up the full volume.
    """
    mask = gt_frame > 0
    gt_vals = gt_frame[mask].astype(np.int64)
    frag_vals = frag_frame[mask].astype(np.int64)

    marker_sizes = dict(zip(*(x.tolist() for x in np.unique(gt_vals, return_counts=True))))

    # A single key per (marker, fragment) pair. `frag_vals` is a label image, so
    # its max bounds the fragment side of the key.
    stride = int(frag_vals.max()) + 1 if frag_vals.size else 1
    keys, counts = np.unique(gt_vals * stride + frag_vals, return_counts=True)

    overlaps = {}
    for key, count in zip(keys.tolist(), counts.tolist()):
        marker, frag = divmod(key, stride)
        if frag == 0:
            continue  # background: contributes to no candidate
        overlaps.setdefault(frag, {})[marker] = count

    return overlaps, marker_sizes


def coordinate_sums(frag_frame, n_ids):
    """Summed z, y, x voxel coordinates per fragment label.

    Accumulated a z-plane at a time so the in-plane coordinate grids stay at
    frame size rather than volume size -- materialising ``np.indices`` over a
    DRO volume would cost 2.3 GB to answer a question about a few thousand
    labels.
    """
    Z, Y, X = frag_frame.shape
    yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing="ij")
    yy = yy.ravel().astype(np.float64)
    xx = xx.ravel().astype(np.float64)

    sums = np.zeros((3, n_ids), dtype=np.float64)
    for z in range(Z):
        plane = frag_frame[z].ravel()
        counts = np.bincount(plane, minlength=n_ids)
        sums[0] += counts * z
        sums[1] += np.bincount(plane, weights=yy, minlength=n_ids)[:n_ids]
        sums[2] += np.bincount(plane, weights=xx, minlength=n_ids)[:n_ids]
    return sums


def marker_centroids(gt_frame):
    """Centroid of every marker, in voxel coordinates."""
    mask = gt_frame > 0
    coords = np.nonzero(mask)
    labels = gt_frame[mask].astype(np.int64)
    n = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n)
    out = {}
    sums = [np.bincount(labels, weights=axis.astype(np.float64), minlength=n)
            for axis in coords]
    for marker in np.nonzero(counts)[0]:
        out[int(marker)] = tuple(float(s[marker] / counts[marker]) for s in sums)
    return out


def frame_detectability(
    gt_frame, frag_frame, merges, size_threshold, min_cost, max_cost,
    scale=(1.0, 1.0, 1.0),
):
    """Best overlap fraction per marker, under each of the three candidate pools.

    ``merges`` are this frame's rows only, already normalised and renumbered.
    ``scale`` is the physical voxel size (z, y, x) used for the offset metric.
    """
    overlaps, marker_sizes = marker_overlaps(gt_frame, frag_frame)

    sizes = np.bincount(frag_frame.ravel())
    sizes[0] = 0  # background is not a candidate
    node_size = {int(i): int(sizes[i]) for i in np.nonzero(sizes)[0]}

    coord_sums = coordinate_sums(frag_frame, len(sizes))
    gt_centroids = marker_centroids(gt_frame)
    # Only nodes that can win are ever asked for a centroid, so carry the sums
    # up the merge tree lazily rather than for the whole forest.
    node_coord = {int(i): coord_sums[:, i].copy() for i in np.nonzero(sizes)[0]}

    best = {
        marker: {
            "frag": 0, "gated": 0, "ungated": 0,
            # The *tightest* candidate that satisfies the CTC criterion, not the
            # one with the largest raw overlap. The merge tree runs up to a
            # single top-level blob that contains every marker, so "largest
            # overlap" would name that blob for all of them and make the offset
            # and collision statistics meaningless. The smallest qualifying
            # object is the best case actually available to the tracker.
            "tight_node": 0, "tight_size": None,
        }
        for marker in marker_sizes
    }

    def offer(node, pools):
        """Record ``node``'s overlaps against whichever pools it belongs to."""
        size = node_size.get(node, 0)
        for marker, count in overlaps.get(node, {}).items():
            entry = best[marker]
            for pool in pools:
                if count > entry[pool]:
                    entry[pool] = count
            if (
                "gated" in pools
                and count > CTC_DETECTION_FRACTION * marker_sizes[marker]
                and (entry["tight_size"] is None or size < entry["tight_size"])
            ):
                entry["tight_node"] = node
                entry["tight_size"] = size

    for node in overlaps:
        big_enough = node_size.get(node, 0) >= size_threshold
        offer(node, ("frag", "gated", "ungated") if big_enough else ("frag", "ungated"))

    for a, b, c, cost, _ in merges:
        a, b, c = int(a), int(b), int(c)
        combined = dict(overlaps.get(a, {}))
        for marker, count in overlaps.get(b, {}).items():
            combined[marker] = combined.get(marker, 0) + count
        node_size[c] = node_size.get(a, 0) + node_size.get(b, 0)
        zero = np.zeros(3)
        node_coord[c] = node_coord.get(a, zero) + node_coord.get(b, zero)
        if not combined:
            continue
        overlaps[c] = combined
        pools = ["ungated"]
        if min_cost <= cost < max_cost and node_size[c] >= size_threshold:
            pools.append("gated")
        offer(c, pools)

    def offset(marker):
        """Physical distance from the marker's centre to its candidate's."""
        node = best[marker]["tight_node"]
        if not node or node not in node_coord or not node_size.get(node):
            return None
        centre = node_coord[node] / node_size[node]
        delta = np.asarray(centre) - np.asarray(gt_centroids[marker])
        return float(np.linalg.norm(delta * np.asarray(scale)))

    return {
        int(marker): {
            "size": int(size),
            "frag": best[marker]["frag"] / size,
            "gated": best[marker]["gated"] / size,
            "ungated": best[marker]["ungated"] / size,
            "tight_node": int(best[marker]["tight_node"]),
            "offset": offset(marker),
            # Volume of the tightest detecting candidate over the marker's.
            # The CTC markers are seeds, not nucleus outlines, so this is well
            # above 1 even when the segmentation is perfect. On DRO the marker
            # is 75 voxels and the SEG annotations put a nucleus at ~19 px
            # across in xy, i.e. ~3.8 planes at 2.03 um z, so an ellipsoid of
            # ~710 voxels -- a ratio near 10. Roughly 2x that is one blob
            # holding two nuclei.
            "tight_size_ratio": (
                best[marker]["tight_size"] / size
                if best[marker]["tight_size"] is not None else None
            ),
        }
        for marker, size in marker_sizes.items()
    }


def voxel_scale(fragments):
    """(z, y, x) voxel size in physical units from the zarr ``axes`` metadata.

    Falls back to voxels if the attribute is absent, in which case ``offset`` is
    still comparable across runs on the same data -- just not in micrometres.
    """
    axes = fragments.attrs.get("axes")
    if not axes:
        return (1.0, 1.0, 1.0)
    spatial = [ax for ax in axes if ax.get("type") == "space"]
    if len(spatial) != 3:
        return (1.0, 1.0, 1.0)
    return tuple(float(ax.get("scale", 1.0)) for ax in spatial)


def evaluate(
    seg_dir, gt_tra_dir, n_frames=None, size_threshold=20,
    min_cost=0.0, max_cost=1.0,
):
    """Per-frame, per-marker detectability for one segmentation run."""
    import tifffile
    import zarr

    fragments = zarr.open(Path(seg_dir) / "data.zarr", "r")["fragments"]
    scale = voxel_scale(fragments)
    print(f"  voxel scale (z, y, x) = {scale}", flush=True)
    total_frames = fragments.shape[0]
    n_frames = total_frames if n_frames is None else min(n_frames, total_frames)

    merge_path = Path(seg_dir) / "merge_history.csv"
    merge_history = (
        load_merge_history(merge_path) if merge_path.is_file() else np.empty((0, 5))
    )

    # Same max-id convention as run_tracking: the largest *fragment* id over the
    # frames in play, which need not be the largest id in the merge history.
    max_node_id = max(int(fragments[t].max()) for t in range(n_frames))
    merge_history = prepare_merge_history(merge_history, max_node_id, n_frames)

    gt_files = sorted(Path(gt_tra_dir).glob("man_track*.tif"))
    if len(gt_files) < n_frames:
        raise SystemExit(
            f"{gt_tra_dir} has {len(gt_files)} marker frames, need {n_frames}"
        )

    per_frame = []
    for t in range(n_frames):
        gt_frame = tifffile.imread(gt_files[t])
        frag_frame = np.asarray(fragments[t])
        if gt_frame.shape != frag_frame.shape:
            raise SystemExit(
                f"frame {t}: GT {gt_frame.shape} != fragments {frag_frame.shape}"
            )
        merges = merge_history[merge_history[:, 4] == t] if len(merge_history) else []
        per_frame.append(
            frame_detectability(
                gt_frame, frag_frame, merges, size_threshold, min_cost, max_cost,
                scale=scale,
            )
        )
        print(f"  frame {t}: {len(per_frame[-1])} markers", flush=True)

    return per_frame


def summarize(per_frame):
    """Recall under each pool, plus collisions, offsets, and where losses sit."""
    pools = ("frag", "gated", "ungated")
    totals = {pool: 0 for pool in pools}
    n = 0
    misses = []
    offsets = []
    size_ratios = []
    collisions = []

    for t, frame in enumerate(per_frame):
        # Which candidate each detected marker claimed. Two markers naming the
        # same node means one object is covering both nuclei, so the tracker can
        # link at most one of them forward -- see the module docstring.
        claims = {}
        for marker, entry in frame.items():
            n += 1
            for pool in pools:
                if entry[pool] > CTC_DETECTION_FRACTION:
                    totals[pool] += 1
            if entry["gated"] <= CTC_DETECTION_FRACTION:
                misses.append({"t": t, "marker": marker, **entry})
                continue
            if entry.get("offset") is not None:
                offsets.append(entry["offset"])
            if entry.get("tight_size_ratio") is not None:
                size_ratios.append(entry["tight_size_ratio"])
            claims.setdefault(entry["tight_node"], []).append(marker)

        for node, markers in claims.items():
            if len(markers) > 1:
                collisions.append({"t": t, "node": node, "markers": markers})

    n_collided = sum(len(c["markers"]) for c in collisions)
    offsets_arr = np.asarray(offsets) if offsets else np.zeros(0)
    ratios_arr = np.asarray(size_ratios) if size_ratios else np.zeros(0)
    n_over = int((ratios_arr > OVER_MERGE_RATIO).sum()) if size_ratios else 0

    return {
        "n_markers": n,
        "n_frames": len(per_frame),
        "recall": {pool: (totals[pool] / n if n else 0.0) for pool in pools},
        "detected": dict(totals),
        # A gated miss that `ungated` rescues is a filter problem; one it does
        # not rescue is a fragment-boundary problem. See the module docstring.
        "gated_misses_rescued_by_ungated": sum(
            1 for m in misses if m["ungated"] > CTC_DETECTION_FRACTION
        ),
        "n_collisions": len(collisions),
        "n_markers_collided": n_collided,
        "collision_rate": n_collided / n if n else 0.0,
        "offset_mean": float(offsets_arr.mean()) if offsets else None,
        "offset_median": float(np.median(offsets_arr)) if offsets else None,
        "offset_p90": float(np.percentile(offsets_arr, 90)) if offsets else None,
        # Against ~1 um of real per-frame drift, a candidate more than 2 um off
        # centre makes the correct link cost more than a wrong one.
        "offset_over_2um": int((offsets_arr > 2.0).sum()) if offsets else 0,
        "size_ratio_median": float(np.median(size_ratios)) if size_ratios else None,
        "size_ratio_p90": float(np.percentile(size_ratios, 90)) if size_ratios else None,
        # Over-merge: sees fusions with UNANNOTATED neighbours, which
        # `collision_rate` cannot. This is the ranking statistic.
        "over_merge_ratio": OVER_MERGE_RATIO,
        "n_over_merged": n_over,
        "over_merge_rate": n_over / len(size_ratios) if size_ratios else 0.0,
        "collisions": collisions,
        "misses": misses,
    }


def format_report(label, summary):
    lines = [
        f"Segmentation detectability: {label}",
        "=" * 78,
        f"GT markers over {summary['n_frames']} frames: {summary['n_markers']}",
        "",
        "Candidate pool                  detected    recall",
        "-" * 78,
    ]
    names = {
        "frag": "base fragments only",
        "gated": "tracker's pool (size+cost)",
        "ungated": "whole merge tree",
    }
    for pool in ("frag", "gated", "ungated"):
        lines.append(
            f"  {names[pool]:<30}{summary['detected'][pool]:>6}"
            f"{summary['recall'][pool]:>10.4f}"
        )

    lines += [
        "",
        "Is the detecting object the RIGHT object?",
        "-" * 78,
        f"  markers sharing a candidate with another marker: "
        f"{summary['n_markers_collided']} in {summary['n_collisions']} blobs"
        f"  ({summary['collision_rate']:.4f})",
        "      each one is a link MaxParents(1) cannot make at any cost",
    ]
    if summary["offset_median"] is not None:
        lines += [
            f"  candidate centroid offset (um): median "
            f"{summary['offset_median']:.2f}, mean {summary['offset_mean']:.2f}, "
            f"p90 {summary['offset_p90']:.2f}",
            f"  offset > 2 um (i.e. > 2x the real per-frame drift): "
            f"{summary['offset_over_2um']}",
            f"  candidate volume / marker volume: median "
            f"{summary['size_ratio_median']:.1f}, p90 {summary['size_ratio_p90']:.1f}"
            "   (~10 is one DRO nucleus)",
            f"  OVER-MERGED (> {summary['over_merge_ratio']:.0f}x marker volume, "
            f"i.e. fused with a neighbour): {summary['n_over_merged']} "
            f"({summary['over_merge_rate']:.4f})",
            "      unlike collisions above, this counts fusions with "
            "UNANNOTATED neighbours too -- rank on it",
        ]

    n_miss = len(summary["misses"])
    lines += [
        "",
        f"Undetectable by the tracker's pool: {n_miss}",
        f"  ... of which the ungated merge tree would rescue: "
        f"{summary['gated_misses_rescued_by_ungated']}"
        "   (filter problem, not segmentation)",
        f"  ... genuinely absent from every union of fragments: "
        f"{n_miss - summary['gated_misses_rescued_by_ungated']}",
    ]

    if n_miss:
        by_frame = {}
        for miss in summary["misses"]:
            by_frame[miss["t"]] = by_frame.get(miss["t"], 0) + 1
        lines += ["", "Misses per frame:"]
        lines += [f"  t={t:<4}{count}" for t, count in sorted(by_frame.items())]

        near = [m for m in summary["misses"] if m["gated"] > 0.3]
        lines += [
            "",
            f"Near misses (best gated fraction in (0.3, 0.5]): {len(near)}"
            "   -- these are the ones a small segmentation change can flip",
        ]
    return "\n".join(lines)


def write_results(out_dir, label, per_frame, summary):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = format_report(label, summary)
    (out_dir / "detectability_report.txt").write_text(report + "\n")
    with open(out_dir / "detectability.json", "w") as handle:
        json.dump(
            {
                "label": label,
                "summary": summary,
                # Keyed by frame index as a string; JSON has no int keys.
                "per_frame": [
                    {str(k): v for k, v in frame.items()} for frame in per_frame
                ],
            },
            handle,
        )
    return report
