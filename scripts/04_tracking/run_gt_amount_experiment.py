"""Fit SSVM weights from progressively smaller amounts of ground truth.

Answers "how much GT does the weight fit actually need?". Raw data, segmentation
and optical flow are identical for every run — only the annotation budget changes.
Three protocols for spending a smaller budget are compared:

    arm A   sparser GT: delete the unannotated candidates from the fit graph
            (a true ignore)                                     — gt_subsets.py
    arm B   sparser GT: mark them gt_selected=None
            (motile's documented sparse-GT API)                 — gt_subsets.py
    arm C   dense crop: annotate everything inside a smaller box, and fit only
            on candidates that lie entirely within it           — gt_crops.py

A and B thin the annotation out over the whole field of view; C spends the same
budget in one place. The difference is not how many negatives the fit gets — the
sparse arms keep every competing hypothesis around an annotated cell, and their
negative fraction actually *rises* as the GT shrinks — but which ones. A candidate
overlapping no annotated object at all cannot honestly be called a false positive
under sparse annotation (the annotator could not tell it from a cell they skipped),
so the sparse arms label exactly zero of those at every budget. Inside a crop they
are supervised normally. Watch the `neg_fp` column: 0 for every A/B run, non-zero
for crops. Compare arms at equal annotated-GT-node count, not equal track count.

The candidate graph, the GT, and the candidate<->GT overlap matrices are built
ONCE and reused across every run — that is the whole reason this is a driver
rather than N invocations of fit_weights_ssvm.py (~4 min of setup each).

Usage:
    python scripts/04_tracking/run_gt_amount_experiment.py <config.toml>
        [--dry-run]           annotate + mask every combination, assert invariants,
                              print the table, fit nothing
        [--self-test]         graph-surgery + crop-geometry unit check, needs no data
        [--materialize-only]  only write the reduced GT (subsets and crops)
        [--only TOKEN ...]    restrict to specific run tokens
        [--sizes N ...] [--seeds S ...] [--arms A B C]
        [--crop-fractions F ...]
        [--criterion any|iogt]
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import motile
import networkx as nx
import numpy as np
import toml

from fit_weights_ssvm import configure_logging, fit_and_solve_on_graph, load_gt
from mhat.tracking.gt_annotation import assign_gt_labels, compute_gt_overlaps
from mhat.tracking.gt_crops import (
    CropBox,
    assert_no_unannotated_overlap,
    build_gt_voxel_index,
    compute_node_bboxes,
    count_gt_nodes_in_crop,
    crop_dirname,
    gt_track_labels_in_crop,
    materialize_cropped_gt,
    nodes_outside_crop,
    sample_crop_boxes,
    sample_crop_boxes_for_budgets,
)
from mhat.tracking.gt_subsets import (
    assert_graph_consistent,
    copy_track_graph,
    count_orphan_positives,
    mask_nodes_none,
    materialize_masked_gt,
    prune_exclusion_sets,
    remove_nodes,
    run_dir,
    run_ref,
    run_token,
    sample_track_subsets,
    subset_dirname,
)
from mhat.tracking.pipeline import build_track_graph, resolve_input_dirs

# Keys that configure the sweep itself; stripped from the per-run tracking config.
_SWEEP_KEYS = (
    "sizes",
    "include_full",
    "seeds",
    "arms",
    "mask_overlap_criterion",
    "gt_subsets_dirname",
    "run_name_prefix",
    "ssvm_reg_normalize",
    "ssvm_reg_effective_target",
    "ssvm_reg_sweep",
    "crop_fractions",
    "crop_target_gt_nodes",
    "crop_growth_step",
    "crop_axes",
    "crop_membership",
    "gt_crops_dirname",
    # Where the sweep groups its runs — a property of the sweep, not of any one
    # run's tracking parameters.
    "runs_subdir",
    "test_runs_subdir",
    "regsweep_subdir",
    # How the sweep is scored and where stage 3 sends its solves — sweep-level, not
    # per-run tracking parameters.
    "eval_metrics",
    "eval_matcher",
    "eval_match_threshold",
    "ctc_gt",
    "test_ctc_gt",
    "selection_metric",
    "test_dataset",
    "test_template",
)

#: Arm C ("dense crop") is handled by a different code path from A/B throughout.
CROP_ARM = "C"


def count_labeled_variables(graph, gt_attribute: str = "gt_selected") -> int:
    """Number of solver variables that carry a GT label — motile's `sum(mask)`.

    `HammingCosts.set_scaling_factor(np.sum(mask))` divides the whole data term
    by this, while `BundleMethod` adds the regularizer outside that scaling, so
    the *effective* regularizer is `ssvm_reg * n_labeled`. Shrinking the GT
    therefore weakens regularization unless it is compensated — which is what
    `ssvm_reg_normalize` does. Note motile only masks NodeSelected/EdgeSelected;
    appear/disappear variables are never labeled and never counted.
    """
    n = sum(1 for d in graph.nodes.values() if d.get(gt_attribute) is not None)
    n += sum(1 for d in graph.edges.values() if d.get(gt_attribute) is not None)
    return n


def count_positive_edges(graph) -> int:
    return sum(1 for data in graph.edges.values() if data.get("gt_selected") == 1)


def find_candidates_touching_no_gt(overlaps) -> set:
    """Candidates that overlap no GT object anywhere in the *complete* ground truth.

    Diagnostic only — never used for supervision, since a subset annotator cannot
    compute it. It is what separates the two protocols in practice: these are the
    detections that are false positives with respect to the whole image, and they
    are exactly the supervision a sparse annotation cannot supply. Arm A masks every
    one of them (they touch no annotated track by definition); a crop keeps the ones
    inside its box, because there the annotation really does say "nothing here".
    """
    no_gt = set()
    for frame in overlaps.frames.values():
        touches_any = (frame.iou > 0.0).any(axis=1)
        for row, cand in enumerate(frame.cand_nodes):
            if not touches_any[row]:
                no_gt.add(cand)
    return no_gt


def build_jobs(sizes, seeds, arms, n_total, prefix, crop_boxes=None):
    """One job dict per run, full-GT reference first and only once.

    Each job carries `kind` ("full" / "tracks" / "crop") because the protocols are
    parameterized differently — track count, crop volume fraction, or crop
    annotation budget — and only the *realized* annotated-GT-node count puts them
    on one axis.

    `crop_boxes` is {seed: {condition: CropBox}}; the condition key is the volume
    fraction or the GT-node budget depending on how the crops were sized, and each
    box names its own run token, so this function needs to know neither.
    """
    jobs = []
    if n_total in sizes:
        jobs.append({"arm": "full", "kind": "full", "n_tracks": n_total, "seed": seeds[0]})
    for arm in arms:
        if arm == CROP_ARM:
            # Conditions are shared across seeds by construction, so the first
            # seed's keys define the sweep.
            conditions = sorted(crop_boxes[seeds[0]], reverse=True)
            for condition in conditions:
                for seed in seeds:
                    jobs.append(
                        {"arm": arm, "kind": "crop", "crop_condition": condition, "seed": seed}
                    )
        else:
            for n in sorted((s for s in sizes if s != n_total), reverse=True):
                for seed in seeds:
                    jobs.append({"arm": arm, "kind": "tracks", "n_tracks": n, "seed": seed})
    for job in jobs:
        if job["kind"] == "crop":
            size = crop_boxes[job["seed"]][job["crop_condition"]].size_token
        else:
            size = job["n_tracks"]
        job["token"] = run_token(job["arm"], size, job["seed"], prefix)
    return jobs


def run_self_test() -> None:
    """Exercise the graph surgery against a real motile.Solver. No data required."""
    print("Self-test: building a toy candidate graph...")
    n_frames, per_frame = 4, 3
    g = nx.DiGraph()
    for t in range(n_frames):
        for k in range(per_frame):
            g.add_node(
                t * per_frame + k + 1,
                time=t,
                area=10.0 + k,
                centroid=[float(t), 0.0, float(k), 0.0],
                cohesion=0.5,
                adhesion=0.5,
                num_leaves=1,
                ignore_appear=(t == 0),
                ignore_disappear=(t == n_frames - 1),
            )
    for t in range(n_frames - 1):
        for k in range(per_frame):
            for j in range(per_frame):
                g.add_edge(
                    t * per_frame + k + 1,
                    (t + 1) * per_frame + j + 1,
                    drift_dist=float(abs(k - j)),
                    area_diff=0.0,
                    intensity_diff=0.0,
                )
    graph = motile.TrackGraph(g, frame_attribute="time")
    exclusion_sets = [[1, 2, 3], [4, 5]]

    # copy independence + order preservation
    copy = copy_track_graph(graph)
    assert list(copy.nodes) == list(graph.nodes), "copy reordered nodes"
    assert list(copy.edges) == list(graph.edges), "copy reordered edges"
    copy.nodes[1]["gt_selected"] = 1
    assert "gt_selected" not in graph.nodes[1], "copy shares node attribute dicts"

    # arm B masking
    masked_b = copy_track_graph(graph)
    b_stats = mask_nodes_none(masked_b, {5})
    assert masked_b.nodes[5]["gt_selected"] is None
    assert b_stats["n_edges_masked"] == len(masked_b.prev_edges[5]) + len(masked_b.next_edges[5])

    # arm A surgery
    n_edges_before = len(graph.edges)
    stats = remove_nodes(copy, {5})
    assert 5 not in copy.nodes
    assert stats["n_edges_removed"] == n_edges_before - len(copy.edges)
    assert_graph_consistent(copy)
    assert copy.get_frames() == (0, n_frames)
    pruned = prune_exclusion_sets(exclusion_sets, copy)
    assert pruned == [[1, 2, 3]], f"unexpected pruned exclusion sets: {pruned}"

    # the real check: does a solver build and solve on the reduced graph?
    solver = motile.Solver(copy)
    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(1))
    solver.add_cost(
        motile.costs.EdgeSelection(weight=1.0, attribute="drift_dist", constant=-5.0)
    )
    solver.add_cost(motile.costs.Appear(constant=1.0, ignore_attribute="ignore_appear"))
    solver.add_cost(
        motile.costs.Disappear(constant=1.0, ignore_attribute="ignore_disappear")
    )
    solver.add_constraint(motile.constraints.ExclusiveNodes(pruned))
    solver.solve()
    selected = solver.get_selected_subgraph()
    assert len(selected.nodes) > 0, "toy solve produced an empty solution"
    print(
        f"Self-test OK — reduced graph {len(copy.nodes)} nodes / {len(copy.edges)} edges, "
        f"solved to {len(selected.nodes)} selected nodes."
    )
    run_crop_self_test()


def run_crop_self_test() -> None:
    """Exercise crop geometry: nesting, containment, bbox union. No data required."""
    print("\nSelf-test: crop geometry...")
    shape = (20, 25, 512, 512)
    fractions = [0.5, 0.25, 0.125, 0.0625]

    for seed in range(8):
        boxes = sample_crop_boxes(shape, fractions, seed=seed, axes=("y", "x"))
        assert set(boxes) == {float(f) for f in fractions}
        for box in boxes.values():
            # XY-only: the time and z axes must come through untouched, otherwise
            # track lengths would change and the arms stop being comparable.
            assert (box.starts[0], box.stops[0]) == (0, shape[0]), "time axis was cropped"
            assert (box.starts[1], box.stops[1]) == (0, shape[1]), "z axis was cropped"
            for i in (2, 3):
                assert 0 <= box.starts[i] < box.stops[i] <= shape[i], "box escaped the volume"
            assert abs(box.volume_fraction - box.fraction) < 0.02, (
                f"realized volume fraction {box.volume_fraction:.4f} is far from "
                f"the requested {box.fraction}"
            )
        ordered = sorted(fractions, reverse=True)
        for larger, smaller in zip(ordered, ordered[1:]):
            assert boxes[larger].contains_box(boxes[smaller].starts, boxes[smaller].stops), (
                f"seed {seed}: crop {smaller} is not nested inside {larger}"
            )

    # Different seeds must actually move the box, or every "replicate" is the same run.
    origins = {sample_crop_boxes(shape, [0.25], seed=s)[0.25].starts for s in range(8)}
    assert len(origins) > 1, "crop placement does not depend on the seed"

    # Bounding boxes: a merge node's box is the union of its leaf fragments'.
    fragments = np.zeros((1, 4, 10, 10), dtype=np.int32)
    fragments[0, 1, 2:4, 2:4] = 1
    fragments[0, 1, 6:8, 6:8] = 2
    toy = nx.DiGraph()
    for node in (1, 2, 3):
        toy.add_node(node, time=0)
    graph = motile.TrackGraph(toy, frame_attribute="time")
    bboxes = compute_node_bboxes(graph, fragments, {1: [1], 2: [2], 3: [1, 2]})
    assert bboxes[1] == ((0, 1, 2, 2), (1, 2, 4, 4)), bboxes[1]
    assert bboxes[2] == ((0, 1, 6, 6), (1, 2, 8, 8)), bboxes[2]
    assert bboxes[3] == ((0, 1, 2, 2), (1, 2, 8, 8)), bboxes[3]

    box = CropBox(
        fraction=0.25,
        seed=0,
        starts=(0, 0, 0, 0),
        stops=(1, 4, 5, 5),
        volume_shape=(1, 4, 10, 10),
        axes=("y", "x"),
    )
    outside = nodes_outside_crop(graph, box, bboxes, membership="contained")
    assert outside == {2, 3}, f"expected the border-straddling nodes to be dropped, got {outside}"

    # --- budget-sized crops --------------------------------------------------
    # A synthetic GT with a deliberately lopsided density: the left half of the
    # field carries four times as many objects as the right. A fixed-fraction crop
    # must therefore land on very different budgets depending on the seed, and a
    # budget-sized crop must not.
    gt = np.zeros((10, 4, 200, 200), dtype=np.int32)
    label = 0
    for gy in range(20):
        for gx in range(20):
            label += 1
            density_gx = gx if gx < 10 else gx * 2  # sparser on the right
            y, x = 5 + gy * 10, 5 + (density_gx % 20) * 10
            gt[:, 1:3, y : y + 4, x : x + 4] = label
    gt_index = build_gt_voxel_index(gt)
    shape = gt.shape

    # The index must agree with a plain slice-and-count, or every budget is wrong.
    for seed in range(3):
        probe = sample_crop_boxes(shape, [0.3], seed=seed, axes=("y", "x"))[0.3]
        direct = 0
        view = gt[(slice(*probe.starts[:1] + probe.stops[:1]), *probe.slices[1:])]
        for frame in view:
            direct += int(np.count_nonzero(np.unique(frame)))
        assert gt_index.count_nodes_in(probe) == direct, (
            f"voxel index counted {gt_index.count_nodes_in(probe)} GT nodes but a "
            f"direct slice counted {direct}"
        )

    targets = [400, 200, 100]
    realized_by_seed = {}
    for seed in range(6):
        boxes = sample_crop_boxes_for_budgets(
            gt_index, shape, targets, seed=seed, axes=("y", "x"), step=0.05
        )
        assert set(boxes) == set(targets)
        for target in targets:
            box = boxes[target]
            realized = gt_index.count_nodes_in(box)
            assert box.realized_gt_nodes == realized, "recorded count is stale"
            realized_by_seed.setdefault(target, []).append(realized)
            assert abs(realized - target) <= max(10, 0.1 * target), (
                f"seed {seed} target {target}: realized {realized} is not close"
            )
            assert box.size_token == f"b{target:04d}"
        ordered = sorted(targets, reverse=True)
        for larger, smaller in zip(ordered, ordered[1:]):
            assert boxes[larger].contains_box(boxes[smaller].starts, boxes[smaller].stops), (
                f"seed {seed}: budget crop {smaller} is not nested inside {larger}"
            )

    # The whole point: budget sizing must collapse the seed spread that fixed
    # fractions produce on the same clumpy GT.
    fixed = [
        gt_index.count_nodes_in(sample_crop_boxes(shape, [0.25], seed=s, axes=("y", "x"))[0.25])
        for s in range(6)
    ]
    fixed_spread = (max(fixed) - min(fixed)) / max(1, np.mean(fixed))
    budgeted = realized_by_seed[200]
    budget_spread = (max(budgeted) - min(budgeted)) / max(1, np.mean(budgeted))
    assert budget_spread < fixed_spread, (
        f"budget sizing did not reduce seed spread (budget {budget_spread:.2f} vs "
        f"fixed-fraction {fixed_spread:.2f})"
    )
    print(
        f"Crop self-test OK — nesting, seed sensitivity, bbox union, containment, "
        f"voxel-index counts, and budget sizing (seed spread {fixed_spread:.2f} "
        f"fixed-fraction -> {budget_spread:.2f} budgeted)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="sweep config TOML")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, help="run tokens to restrict to")
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument(
        "--crop-fractions",
        nargs="+",
        type=float,
        default=None,
        help=f"volume fractions for arm {CROP_ARM} (dense crops), e.g. 0.5 0.25 0.125",
    )
    parser.add_argument(
        "--crop-targets",
        nargs="+",
        type=int,
        default=None,
        help=f"annotated-GT-node budgets for arm {CROP_ARM}; each crop is grown until "
        "it holds about that many. Takes precedence over --crop-fractions",
    )
    parser.add_argument("--criterion", choices=("any", "iogt"), default=None)
    parser.add_argument(
        "--force-gt", action="store_true", help="rewrite masked GT subsets that already exist"
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.config:
        parser.error("config is required unless --self-test is given")

    config = toml.load(args.config)
    sizes = args.sizes or config["sizes"]
    seeds = args.seeds or config["seeds"]
    arms = args.arms or config["arms"]
    criterion = args.criterion or config.get("mask_overlap_criterion", "any")
    prefix = config.get("run_name_prefix", "gta")
    iogt_threshold = config.get("iogt_threshold", 0.5)
    crop_fractions = args.crop_fractions or config.get("crop_fractions", [])
    crop_targets = args.crop_targets or config.get("crop_target_gt_nodes", [])
    crop_growth_step = float(config.get("crop_growth_step", 0.05))
    crop_axes = tuple(config.get("crop_axes", ["y", "x"]))
    crop_membership = config.get("crop_membership", "contained")
    wants_crops = CROP_ARM in arms
    # Budget sizing takes precedence: it is the mode that makes the crop arm
    # comparable to the sparse arms, so if both are configured it is the one meant.
    crop_mode = "budget" if (crop_targets or crop_targets == "match_sizes") else "fraction"
    if wants_crops and not crop_targets and not crop_fractions:
        parser.error(
            f"arm {CROP_ARM} requested but neither crop_target_gt_nodes nor "
            "crop_fractions given (set one in the config, or pass "
            "--crop-targets / --crop-fractions)"
        )

    raw_dir, seg_dir, flow_dirs, gt_data_dir = resolve_input_dirs(config)
    output_base_dir = Path(config["output_base_dir"])
    dataset_out = output_base_dir / "tracking" / config["experiment"] / config["dataset"]
    gt_subsets_root = dataset_out / config.get("gt_subsets_dirname", "gt_subsets")
    gt_crops_root = dataset_out / config.get("gt_crops_dirname", "gt_crops")
    runs_subdir = config.get("runs_subdir", "")

    print("Building candidate graph (once)...")
    base_graph, fragments, merge_history, base_exclusion_sets, scale, axes = build_track_graph(
        config, raw_dir, seg_dir, flow_dirs
    )
    no_merges = len(merge_history) == 0
    print(f"  {len(base_graph.nodes)} candidate nodes, {len(base_graph.edges)} candidate edges")

    print("Loading GT (once)...")
    gt_tracks, gt_seg = load_gt(gt_data_dir, scale)

    print("Computing candidate<->GT overlaps (once)...")
    overlaps = compute_gt_overlaps(base_graph, fragments, gt_tracks, gt_seg, merge_history)

    all_labels = sorted({int(d["track_id"]) for _, d in gt_tracks.nodes(data=True)})
    n_total = len(all_labels)
    print(f"  GT has {n_total} tracks, {gt_tracks.number_of_nodes()} nodes")
    missing = set(all_labels) - overlaps.all_gt_labels
    if missing:
        print(f"  NOTE: {len(missing)} GT tracks have no voxels in correct_seg.zarr")

    # The full-GT size is whatever import_from_geff's renumbering produced, which
    # is NOT the raw geff track_id count (funtracks splits tracks at divisions).
    # Derive it rather than trusting a hand-written size, otherwise a
    # "nearly all" subset would silently masquerade as the full-GT reference run.
    requested = sorted({int(s) for s in sizes}, reverse=True)
    dropped = [s for s in requested if s >= n_total]
    sizes = [s for s in requested if s < n_total]
    if config.get("include_full", True):
        sizes = [n_total] + sizes
    if dropped:
        print(f"  NOTE: sizes {dropped} are >= the true GT size {n_total}; using {n_total} as full")
    print(f"  sizes: {sizes}")

    subsets, orders = {}, {}
    for seed in seeds:
        subsets[seed], orders[seed] = sample_track_subsets(gt_tracks, sizes, seed)

    # --- nesting invariant (cheap, always checked) ---------------------------
    for seed in seeds:
        for smaller, larger in zip(sorted(sizes), sorted(sizes)[1:]):
            assert subsets[seed][smaller] <= subsets[seed][larger], (
                f"seed {seed}: subset {smaller} is not contained in {larger}"
            )

    # --- materialize the reduced GT -----------------------------------------
    subset_info: dict[tuple[int, int], dict] = {}
    print(f"\nMaterializing GT subsets under {gt_subsets_root}")
    for seed in seeds:
        for n in sorted(sizes, reverse=True):
            if n == n_total and seed != seeds[0]:
                continue  # full GT is identical for every seed
            out = gt_subsets_root / subset_dirname(n, seed)
            if (out / "subset.json").is_file() and not args.force_gt:
                subset_info[(n, seed)] = json.loads((out / "subset.json").read_text())
                continue
            info = materialize_masked_gt(
                gt_data_dir,
                out,
                subsets[seed][n],
                gt_tracks,
                scale,
                seed=seed,
                n_tracks=n,
                n_tracks_total=n_total,
                overlap_criterion=criterion,
                shuffle_order=orders[seed],
            )
            subset_info[(n, seed)] = info
            print(
                f"  {subset_dirname(n, seed)}: {info['n_gt_nodes_kept']}/{info['n_gt_nodes_total']}"
                f" GT nodes, {info['n_gt_edges_kept']}/{info['n_gt_edges_total']} GT edges"
            )

    # --- crop boxes ----------------------------------------------------------
    # Placement depends on the seed alone — never on where the GT happens to be —
    # so no crop can be "lucky". Only the SIZE may adapt, and only in budget mode.
    crop_boxes: dict[int, dict] = {}
    node_bboxes = None
    if wants_crops:
        volume_shape = tuple(int(v) for v in fragments.shape)
        gt_index = build_gt_voxel_index(gt_seg)

        if crop_mode == "budget":
            targets = crop_targets
            if targets == "match_sizes":
                # One budget per track-subset size, equal to the mean annotated-GT-node
                # count those subsets actually cost across seeds. This is what puts arm
                # C's points directly on top of arm A/B's on the shared x-axis instead
                # of scattered between them.
                targets = []
                for n in sorted((s for s in sizes if s != n_total), reverse=True):
                    counts = [
                        subset_info[(n, s)]["n_gt_nodes_kept"]
                        for s in seeds
                        if (n, s) in subset_info
                    ]
                    if not counts:
                        parser.error(
                            f'crop_target_gt_nodes = "match_sizes" needs the track '
                            f"subsets for size {n} to exist; run --materialize-only first"
                        )
                    targets.append(int(round(sum(counts) / len(counts))))
                print(
                    f"\nCrop budgets from sizes {[s for s in sizes if s != n_total]}: "
                    f"{targets} annotated GT nodes (mean over {len(seeds)} seeds)"
                )
            targets = [int(t) for t in targets]
            # The budgets end up in the run tokens, and under "match_sizes" they
            # depend on which seeds took part. A fan-out that narrowed --seeds would
            # otherwise resolve different budgets, silently splitting the sweep into
            # two token namespaces that never appear on the same curve. Pin them on
            # first resolve and refuse to drift.
            budgets_path = gt_crops_root / "crop_budgets.json"
            if budgets_path.is_file() and not args.force_gt:
                pinned = json.loads(budgets_path.read_text())["targets"]
                if pinned != targets:
                    raise SystemExit(
                        f"crop budgets resolved to {targets} but {budgets_path} pins "
                        f"{pinned}.\nUnder crop_target_gt_nodes = \"match_sizes\" the "
                        "budgets are the mean over `seeds`, so narrowing --seeds "
                        "changes them and the run tokens with them.\nRe-run with the "
                        "config's full seed list, or pass --force-gt to re-pin."
                    )
                targets = pinned
            else:
                gt_crops_root.mkdir(parents=True, exist_ok=True)
                with open(budgets_path, "w") as f:
                    json.dump(
                        {"targets": targets, "seeds": list(seeds), "sizes": list(sizes)}, f, indent=2
                    )
            for seed in seeds:
                crop_boxes[seed] = sample_crop_boxes_for_budgets(
                    gt_index,
                    volume_shape,
                    targets,
                    seed=seed,
                    axes=crop_axes,
                    step=crop_growth_step,
                )
        else:
            for seed in seeds:
                crop_boxes[seed] = sample_crop_boxes(
                    volume_shape, crop_fractions, seed=seed, axes=crop_axes
                )

        print(
            f"\nCrop boxes (mode={crop_mode}, axes={list(crop_axes)}, "
            f"membership={crop_membership!r}):"
        )
        for seed in seeds:
            for condition in sorted(crop_boxes[seed], reverse=True):
                box = crop_boxes[seed][condition]
                budget = ""
                if box.target_gt_nodes is not None:
                    error = box.realized_gt_nodes - box.target_gt_nodes
                    budget = (
                        f"  {box.realized_gt_nodes} GT nodes "
                        f"(target {box.target_gt_nodes}, {error:+d})"
                    )
                print(
                    f"  {crop_dirname(box):<12} {box.describe()}"
                    f"  ({box.volume_fraction * 100:.1f}% of the volume){budget}"
                )
        print("Computing candidate bounding boxes (once)...")
        node_bboxes = compute_node_bboxes(base_graph, fragments, overlaps.node_to_fragments)

    crop_info: dict[tuple, dict] = {}
    if wants_crops:
        print(f"\nMaterializing GT crops under {gt_crops_root}")
        for seed in seeds:
            for condition in sorted(crop_boxes[seed], reverse=True):
                box = crop_boxes[seed][condition]
                out = gt_crops_root / crop_dirname(box)
                if (out / "subset.json").is_file() and not args.force_gt:
                    existing = json.loads((out / "subset.json").read_text())
                    # A crop dir is named by its condition and seed only, so changing
                    # crop_axes, the growth step or the volume shape would silently
                    # reuse a box drawn under the old settings and quietly invalidate
                    # the whole sweep.
                    wanted = box.to_dict()
                    found = existing.get("crop", {})
                    if [found.get("starts"), found.get("stops")] != [
                        wanted["starts"],
                        wanted["stops"],
                    ]:
                        raise SystemExit(
                            f"{out} holds a different crop box than this config asks for\n"
                            f"  on disk:  starts={found.get('starts')} stops={found.get('stops')}"
                            f" axes={found.get('axes')}\n"
                            f"  config:   starts={wanted['starts']} stops={wanted['stops']}"
                            f" axes={wanted['axes']}\n"
                            "Pass --force-gt to rewrite it, or use a fresh gt_crops_dirname."
                        )
                    crop_info[(condition, seed)] = existing
                    continue
                info = materialize_cropped_gt(
                    gt_data_dir,
                    out,
                    box,
                    gt_tracks,
                    scale,
                    n_gt_tracks_total=n_total,
                    membership=crop_membership,
                )
                crop_info[(condition, seed)] = info
                print(
                    f"  {crop_dirname(box)}: {info['n_gt_nodes_kept']}/"
                    f"{info['n_gt_nodes_total']} GT nodes across {info['n_tracks']} tracks, "
                    f"{info['n_gt_edges_kept']}/{info['n_gt_edges_total']} GT edges"
                )
    if args.materialize_only:
        return

    jobs = build_jobs(sizes, seeds, arms, n_total, prefix, crop_boxes)
    if args.only:
        jobs = [j for j in jobs if j["token"] in set(args.only)]
        if not jobs:
            parser.error(f"--only matched no runs; available tokens use prefix {prefix!r}")

    # The full-GT problem is fully labeled, so its labeled-variable count is just
    # the whole graph. This is the reference point for ssvm_reg normalization.
    reference_scale = len(base_graph.nodes) + len(base_graph.edges)

    full_reference = {}
    rows = []
    if config.get("ssvm_reg_effective_target"):
        reg_desc = f"effective reg pinned at {config['ssvm_reg_effective_target']}"
    elif config.get("ssvm_reg_normalize", False):
        reg_desc = f"effective reg normalized to full GT ({reference_scale} labeled vars)"
    else:
        reg_desc = f"ssvm_reg fixed at {config.get('ssvm_reg', 0.1)}"
    print(f"\n{len(jobs)} run(s), criterion={criterion!r}, {reg_desc}\n")
    header = (
        f"{'token':<22}{'kept':>6}{'gt_n':>7}{'masked':>8}{'pos_n':>7}{'neg_n':>8}"
        f"{'neg_fp':>8}{'pos_e':>7}{'fit_n':>8}{'fit_e':>8}{'orph':>6}{'lbl_var':>8}{'reg':>10}"
    )
    print(header)
    print("-" * len(header))
    # neg_fp = labeled negatives overlapping no GT at all — the supervision only a
    # densely annotated region can supply. Expect 0 for arms A and B, always.
    no_gt_nodes = find_candidates_touching_no_gt(overlaps)

    for job in jobs:
        arm, seed, token = job["arm"], job["seed"], job["token"]
        n_tracks = job.get("n_tracks")
        is_crop = job["kind"] == "crop"
        box = crop_boxes[seed][job["crop_condition"]] if is_crop else None
        fraction = box.fraction if is_crop else None
        crop_target = box.target_gt_nodes if is_crop else None

        if is_crop:
            # Annotated iff it intersects the box; used iff wholly inside it. That
            # pairing is what makes a dense `gt_selected = 0` honest — see gt_crops.
            keep_labels = gt_track_labels_in_crop(gt_index, box)
            outside = nodes_outside_crop(
                base_graph, box, node_bboxes, membership=crop_membership, scale=scale
            )
            gt_data_subdir = gt_crops_root / crop_dirname(box)
            n_gt_nodes_annotated = count_gt_nodes_in_crop(gt_index, box)
            # The fit is labeled against `gt_seg` (track-id space) while the train
            # eval scores against the materialized crop (geff node-id space). They
            # are two views of the same voxels, so they must agree — if they ever
            # drift, the train score stops being "measured on exactly what this run
            # annotated", which is the claim that makes it safe to select on.
            info = crop_info.get((job["crop_condition"], seed))
            if info is not None:
                assert info["n_gt_nodes_kept"] == n_gt_nodes_annotated, (
                    f"{token}: the fit sees {n_gt_nodes_annotated} annotated GT nodes "
                    f"but the materialized crop holds {info['n_gt_nodes_kept']}; the "
                    "fit and its train evaluation would be scored on different GT"
                )
        else:
            keep_labels = None if n_tracks == n_total else subsets[seed][n_tracks]
            outside = None
            gt_data_subdir = gt_subsets_root / subset_dirname(n_tracks, seed)
            info = subset_info.get((n_tracks, seed))
            n_gt_nodes_annotated = (
                info["n_gt_nodes_kept"] if info else gt_tracks.number_of_nodes()
            )

        graph = copy_track_graph(base_graph)
        stats, masked = assign_gt_labels(
            graph,
            overlaps,
            gt_tracks,
            iogt_threshold=iogt_threshold,
            keep_gt_labels=keep_labels,
            overlap_criterion=criterion,
            # A crop annotates its whole region, so "touches nothing annotated" is
            # NOT grounds for masking — those are the real negatives it buys.
            mask_untouched=not is_crop,
            extra_masked_nodes=outside,
        )
        pos_edges_before = count_positive_edges(graph)

        if arm == "full":
            assert not masked, "full-GT run masked candidates; keep_gt_labels handling is wrong"

        if is_crop:
            assert masked == outside, (
                f"{token}: crop masked {len(masked)} candidates but the box excludes "
                f"{len(outside)}; mask_untouched leaked into the crop path"
            )
            surviving = [n for n in base_graph.nodes if n not in masked]
            # The no-oracle gate. Cheap, and it is the one invariant that would let
            # unannotated ground truth back into the labels if the membership rule
            # or the annotated-label rule ever drifted apart.
            assert_no_unannotated_overlap(surviving, overlaps, keep_labels)

        if arm == "A" or is_crop:
            solve_graph = copy_track_graph(graph)
            solve_exclusion_sets = base_exclusion_sets
            remove_nodes(graph, masked)
            assert_graph_consistent(graph)
            fit_graph = graph
            fit_exclusion_sets = prune_exclusion_sets(base_exclusion_sets, graph)
            pos_edges_after = count_positive_edges(fit_graph)
            assert pos_edges_after == pos_edges_before, (
                f"{token}: node removal destroyed GT-positive edges "
                f"({pos_edges_before} -> {pos_edges_after}); a positive edge only "
                "exists between two matched candidates, and matching never selects "
                "a masked one, so both endpoints must survive"
            )
            for group in fit_exclusion_sets:
                assert all(node in fit_graph.nodes for node in group)
        else:
            mask_stats = mask_nodes_none(graph, masked)
            n_none_nodes = sum(
                1 for d in graph.nodes.values() if d.get("gt_selected", 0) is None
            )
            assert n_none_nodes == len(masked), (
                f"{token}: {n_none_nodes} unlabeled nodes but {len(masked)} masked"
            )
            assert not any(
                graph.nodes[node].get("gt_selected") == 1 for node in masked
            ), f"{token}: a masked node is still labelled positive"
            stats.update({f"arm_b_{k}": v for k, v in mask_stats.items()})
            fit_graph = graph
            fit_exclusion_sets = base_exclusion_sets
            solve_graph = None
            solve_exclusion_sets = None

        orphans = count_orphan_positives(fit_graph)
        n_labeled = count_labeled_variables(fit_graph)
        # Labeled negatives that overlap no GT anywhere: the "this detection is
        # simply wrong" supervision. Sparse arms have none by construction.
        n_neg_no_gt = sum(
            1
            for node in no_gt_nodes
            if node in fit_graph.nodes and fit_graph.nodes[node].get("gt_selected") == 0
        )
        base_reg = config.get("ssvm_reg", 0.1)
        reg_target = config.get("ssvm_reg_effective_target")
        if reg_target:
            # Hold the effective regularizer at an explicitly chosen value — e.g. the
            # regime that scored best on the TRAIN annotated subsets — so the curve
            # shows GT amount at a fixed, deliberately picked operating point.
            run_reg = float(reg_target) / n_labeled
        elif config.get("ssvm_reg_normalize", False):
            # Hold the EFFECTIVE regularizer (ssvm_reg * n_labeled) fixed at the
            # full-GT value, so shrinking the GT does not silently also weaken
            # regularization and confound the learning curve.
            run_reg = base_reg * reference_scale / n_labeled
        else:
            run_reg = base_reg

        row = {
            "token": token,
            "arm": arm,
            "kind": job["kind"],
            # For a crop this is the number of tracks the box happens to touch —
            # measured, not requested. The comparable budget across protocols is
            # n_gt_nodes_annotated (you pay per object per frame), which is what
            # the learning curve is plotted against.
            "n_tracks": n_tracks if not is_crop else stats["n_kept_gt_labels"],
            "n_gt_nodes_annotated": n_gt_nodes_annotated,
            "crop_fraction": fraction,
            # The requested budget in budget mode; None when crops were sized by
            # volume. Plotting groups arm-C seeds on this, since it is the one
            # value every seed of a condition shares exactly.
            "crop_target_gt_nodes": crop_target,
            "crop_membership": crop_membership if is_crop else None,
            "crop_box": box.to_dict() if is_crop else None,
            "seed": seed,
            "criterion": criterion,
            "n_fit_nodes": len(fit_graph.nodes),
            "n_fit_edges": len(fit_graph.edges),
            "n_labeled_variables": n_labeled,
            "ssvm_reg_used": run_reg,
            "ssvm_reg_effective": run_reg * n_labeled,
            "n_orphan_positives": orphans,
            "n_neg_no_gt_overlap": n_neg_no_gt,
            "gt_subset_dir": str(gt_data_subdir),
            **stats,
        }
        rows.append(row)
        print(
            f"{token:<22}{stats['n_kept_gt_labels']:>6}{n_gt_nodes_annotated:>7}{len(masked):>8}"
            f"{stats['n_nodes_pos']:>7}{stats['n_nodes_neg']:>8}{n_neg_no_gt:>8}"
            f"{stats['n_edges_pos']:>7}"
            f"{len(fit_graph.nodes):>8}{len(fit_graph.edges):>8}{orphans:>6}"
            f"{n_labeled:>8}{run_reg:>10.4f}"
        )

        if arm == "full":
            full_reference = dict(stats)
            full_reference["n_gt_nodes_annotated"] = n_gt_nodes_annotated
        elif full_reference:
            assert stats["n_nodes_pos"] <= full_reference["n_nodes_pos"], (
                f"{token}: more positives than the full-GT run"
            )
            assert n_gt_nodes_annotated <= full_reference["n_gt_nodes_annotated"], (
                f"{token}: {n_gt_nodes_annotated} annotated GT nodes exceeds the "
                f"full-GT run's {full_reference['n_gt_nodes_annotated']}"
            )
        if is_crop:
            # A dense crop must keep negatives — that is the whole reason it exists.
            # Arm A at a comparable budget has almost none, so a crop reporting ~0
            # means mask_untouched leaked in and the arms are measuring the same thing.
            assert stats["n_nodes_neg"] > 0, (
                f"{token}: crop kept no negative candidates; the dense-annotation "
                "path is not doing what it claims"
            )

        if args.dry_run:
            continue

        out_dir = run_dir(dataset_out, runs_subdir, token)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_config = {k: v for k, v in config.items() if k not in _SWEEP_KEYS}
        run_config["output_name"] = token
        run_config["exp_uid"] = run_ref(runs_subdir, token)
        run_config["gt_amount_arm"] = arm
        run_config["gt_amount_kind"] = job["kind"]
        run_config["gt_amount_n_tracks"] = int(row["n_tracks"])
        run_config["gt_amount_n_gt_nodes"] = int(n_gt_nodes_annotated)
        run_config["gt_amount_seed"] = int(seed)
        run_config["gt_amount_criterion"] = criterion
        if is_crop:
            run_config["gt_amount_crop_fraction"] = float(fraction)
            run_config["gt_amount_crop_membership"] = crop_membership
            run_config["gt_amount_crop_starts"] = [int(v) for v in box.starts]
            run_config["gt_amount_crop_stops"] = [int(v) for v in box.stops]
            if crop_target is not None:
                run_config["gt_amount_crop_target_gt_nodes"] = int(crop_target)
        run_config["ssvm_reg"] = run_reg
        with open(out_dir / "config.toml", "w") as f:
            toml.dump(run_config, f)

        configure_logging(out_dir)
        try:
            summary = fit_and_solve_on_graph(
                run_config,
                fit_graph,
                fit_exclusion_sets,
                fragments,
                merge_history,
                scale,
                axes,
                out_dir,
                no_merges=no_merges,
                solve_graph=solve_graph,
                solve_exclusion_sets=solve_exclusion_sets,
            )
            row.update(summary)
            row["status"] = "empty_solution" if summary.get("empty_solution") else "ok"
        except Exception:
            row["status"] = "fit_failed"
            row["traceback"] = traceback.format_exc()
            print(f"  {token} FAILED:\n{row['traceback']}")

        with open(out_dir / "fit_summary.json", "w") as f:
            json.dump(row, f, indent=2)

    # One file per invocation. With --only (the shape a cluster fan-out takes) every
    # job would otherwise write the same path concurrently and clobber the others.
    stem = "dry_run_summary" if args.dry_run else "run_summary"
    if args.only:
        stem += "_" + ("_".join(sorted(args.only)) if len(args.only) <= 3 else f"{len(args.only)}runs")
    summary_path = gt_subsets_root / f"{stem}.json"
    gt_subsets_root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {summary_path}")

    if not args.dry_run:
        bad = [r["token"] for r in rows if r.get("status") not in (None, "ok")]
        if bad:
            print(f"\n{len(bad)} run(s) did not complete cleanly: {', '.join(bad)}")


if __name__ == "__main__":
    main()
