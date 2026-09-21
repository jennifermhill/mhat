"""Smoke tests for the shared SSVM-branch pipeline helpers, at both ranks.

``mhat.tracking.pipeline`` is what fit_weights_ssvm.py, solve_many_weights.py,
sweep_ssvm_reg.py and the GT-amount driver build their candidate graph with,
load their ground truth with, and write their outputs with. run_tracking.py
does the same three things inline. These tests pin the two paths to each
other and to ``evaluate_tracking``'s readers, on 2D data as well as 3D, so a
rank assumption creeping back into the shared module fails here rather than
in a cluster run.

The scene is the conftest one (two drifting blobs, two fragments each),
written to disk in the layout stage 02 produces: raw as (t, c, *spatial),
fragments as (t, *spatial) with axes metadata, and a merge_history.csv.
"""

from __future__ import annotations

import csv
import shutil

import numpy as np
import pytest
import zarr

from conftest import T, frame_shape
from mhat.evaluation.evaluate_tracking import read_name_map_and_scale
from mhat.tracking import pipeline
from mhat.tracking.gt_annotation import (
    annotate_gt_on_candidate_graph,
    build_node_to_fragments,
)
from mhat.tracking.gt_crops import (
    CropBox,
    build_gt_voxel_index,
    compute_node_bboxes,
    nodes_outside_crop,
    sample_crop_boxes,
)
from mhat.tracking.solve_with_motile import solve_with_motile
from mhat.utils import default_axes, position_names, seg_chunks
from test_tracking_pipeline import build_candidate_graph


@pytest.fixture
def on_disk_scene(tmp_path, ndim, synthetic_fragments, synthetic_raw, synthetic_merge_history):
    """The synthetic scene as stage-02 output on disk: (raw_dir, seg_dir)."""
    raw_dir = tmp_path / "raw.zarr"
    raw = zarr.open(
        raw_dir, mode="w", shape=(T, 1, *frame_shape(ndim)), dtype=np.float32,
        chunks=(1, 1, *frame_shape(ndim)),
    )
    raw[:, 0] = synthetic_raw

    seg_dir = tmp_path / "seg"
    seg_dir.mkdir()
    root = zarr.open(seg_dir / "data.zarr", mode="a")
    fragments = root.create_dataset("fragments", data=synthetic_fragments)
    fragments.attrs["axes"] = default_axes(ndim)
    with open(seg_dir / "merge_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "c", "cost", "timepoint"])
        for a, b, c, cost, t in synthetic_merge_history:
            writer.writerow([int(a), int(b), int(c), float(cost), int(t)])
    return raw_dir, seg_dir


NO_FLOW = {"2d": None, "3d": None}


def _real_nodes(graph):
    return {n for n, d in graph.nodes.items() if "time" in d}


def test_build_track_graph_matches_inline_construction(
    on_disk_scene, synthetic_fragments, synthetic_raw, synthetic_merge_history,
    tracking_config, scale, ndim,
):
    """The shared builder and run_tracking.py's inline steps agree exactly."""
    raw_dir, seg_dir = on_disk_scene
    graph, fragments, merge_history, exclusion_sets, read_scale, axes = (
        pipeline.build_track_graph(tracking_config, raw_dir, seg_dir, NO_FLOW)
    )
    reference, reference_exclusions = build_candidate_graph(
        synthetic_fragments, synthetic_raw, synthetic_merge_history.copy(),
        tracking_config, scale,
    )

    assert read_scale == scale
    assert [a["name"] for a in axes] == ["time", *position_names(ndim)]
    assert fragments.shape == synthetic_fragments.shape
    assert _real_nodes(graph) == _real_nodes(reference)
    assert set(graph.edges) == set(reference.edges)
    assert sorted(map(sorted, exclusion_sets)) == sorted(map(sorted, reference_exclusions))
    for node in _real_nodes(graph):
        got, want = graph.nodes[node], reference.nodes[node]
        assert len(got["centroid"]) == ndim
        assert got["centroid"] == pytest.approx(want["centroid"])
        assert (got["cohesion"], got["adhesion"]) == (want["cohesion"], want["adhesion"])
        if ndim == 2:
            assert "z" not in got, f"2D node {node} carries a phantom z"
    for edge in graph.edges:
        assert graph.edges[edge]["drift_dist"] == pytest.approx(
            reference.edges[edge]["drift_dist"]
        )


def test_no_merge_history_builds_leaf_graph(on_disk_scene, tracking_config, ndim):
    """A header-only merge_history.csv (skip-merges / no touching objects)."""
    raw_dir, seg_dir = on_disk_scene
    (seg_dir / "merge_history.csv").write_text("a,b,c,cost,timepoint\n")
    graph, _, merge_history, exclusion_sets, _, _ = pipeline.build_track_graph(
        tracking_config, raw_dir, seg_dir, NO_FLOW
    )
    assert merge_history.shape == (0, 5)
    assert exclusion_sets == []
    nodes = _real_nodes(graph)
    assert len(nodes) == 4 * T
    for node in nodes:
        assert graph.nodes[node]["num_leaves"] == 1
        assert len(graph.nodes[node]["centroid"]) == ndim


def test_outputs_round_trip_as_ground_truth(on_disk_scene, tracking_config, tmp_path, ndim):
    """write_tracking_outputs -> read_name_map_and_scale / load_gt -> annotation.

    The written stores must declare exactly the data's axes, be chunked one
    tile of one slice, and be labelled by node id, or evaluate_tracking's
    remap refuses them. Re-reading them as a ground truth also exercises
    ``load_gt`` and ``annotate_gt_on_candidate_graph`` at this rank: a
    solution annotated against itself must label every selected node positive.
    """
    raw_dir, seg_dir = on_disk_scene
    graph, fragments, merge_history, exclusion_sets, scale, axes = (
        pipeline.build_track_graph(tracking_config, raw_dir, seg_dir, NO_FLOW)
    )
    solution = solve_with_motile(tracking_config, graph, exclusion_sets)
    assert solution.number_of_nodes() > 0

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    pipeline.write_tracking_outputs(solution, fragments, merge_history, scale, axes, out_dir)

    name_map, read_scale = read_name_map_and_scale(out_dir / "pred_tracks.zarr")
    assert name_map["pos"] == position_names(ndim)
    assert read_scale == scale
    seg = zarr.open(out_dir / "pred_seg.zarr", mode="r")
    assert seg.shape == fragments.shape
    assert seg.chunks == seg_chunks(fragments.shape[1:])
    assert set(int(v) for v in np.unique(seg[:])) - {0} == set(solution.nodes)

    # The same stores under the ground-truth names.
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    shutil.copytree(out_dir / "pred_tracks.zarr", gt_dir / "correct_tracks.zarr")
    shutil.copytree(out_dir / "pred_seg.zarr", gt_dir / "correct_seg.zarr")
    gt_graph, gt_seg = pipeline.load_gt(gt_dir, scale)
    assert gt_graph.number_of_nodes() == solution.number_of_nodes()
    assert gt_seg.shape == fragments.shape
    for _, data in gt_graph.nodes(data=True):
        assert len(data["pos"]) == ndim

    stats = annotate_gt_on_candidate_graph(graph, fragments, gt_graph, gt_seg, merge_history)
    assert stats["n_nodes_pos"] == solution.number_of_nodes()
    assert stats["n_edges_pos"] == solution.number_of_edges()
    assert stats["n_unmatched_gt_total"] == 0
    for node in solution.nodes:
        assert graph.nodes[node]["gt_selected"] == 1


def test_load_gt_refuses_a_rank_mismatch(on_disk_scene, tracking_config, tmp_path, ndim):
    """A 3D scale against a 2D GT (or vice versa) must fail, not truncate."""
    raw_dir, seg_dir = on_disk_scene
    graph, fragments, merge_history, exclusion_sets, scale, axes = (
        pipeline.build_track_graph(tracking_config, raw_dir, seg_dir, NO_FLOW)
    )
    solution = solve_with_motile(tracking_config, graph, exclusion_sets)
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    pipeline.write_tracking_outputs(solution, fragments, merge_history, scale, axes, gt_dir)
    (gt_dir / "pred_tracks.zarr").rename(gt_dir / "correct_tracks.zarr")
    (gt_dir / "pred_seg.zarr").rename(gt_dir / "correct_seg.zarr")
    with pytest.raises(AssertionError, match="not the same rank"):
        pipeline.load_gt(gt_dir, [1.0] * (6 - ndim))


@pytest.mark.parametrize("has_2d,has_3d", [(True, True), (True, False), (False, True)])
def test_flow_policy_follows_the_data_rank(tmp_path, ndim, synthetic_fragments, has_2d, has_3d):
    """3D data needs a 3D flow and takes the 2D one if present; 2D needs the 2D one."""
    input_base = tmp_path
    seg_dir = input_base / "segmentation" / "exp" / "ds" / "seg0"
    seg_dir.mkdir(parents=True)
    zarr.open(seg_dir / "data.zarr", mode="a").create_dataset(
        "fragments", data=synthetic_fragments
    )
    flow_base = input_base / "opticalflow" / "exp" / "ds"
    if has_2d:
        (flow_base / "opticalflow_2d" / "flow0").mkdir(parents=True)
    if has_3d:
        (flow_base / "opticalflow_3d" / "flow0").mkdir(parents=True)
    config = {
        "input_base_dir": str(input_base), "experiment": "exp", "dataset": "ds",
        "flow_result": "flow0",
    }

    if ndim == 3 and not has_3d:
        with pytest.raises(AssertionError, match="3D optical flow"):
            pipeline.resolve_flow_dirs(config, seg_dir)
        return
    if ndim == 2 and not has_2d:
        with pytest.raises(AssertionError, match="2D optical flow"):
            pipeline.resolve_flow_dirs(config, seg_dir)
        return

    flow_dirs = pipeline.resolve_flow_dirs(config, seg_dir)
    if ndim == 3:
        assert flow_dirs["3d"] == flow_base / "opticalflow_3d" / "flow0"
        assert flow_dirs["2d"] == (flow_base / "opticalflow_2d" / "flow0" if has_2d else None)
    else:
        assert flow_dirs["2d"] == flow_base / "opticalflow_2d" / "flow0"
        assert flow_dirs["3d"] is None, "2D data must never be handed a 3D flow"

    assert pipeline.resolve_flow_dirs({**config, "flow_result": None}, seg_dir) == NO_FLOW


def test_crop_geometry_follows_the_volume_rank(
    on_disk_scene, tracking_config, synthetic_fragments, ndim
):
    """gt_crops boxes, bounding boxes and the voxel index all take the volume's rank."""
    raw_dir, seg_dir = on_disk_scene
    graph, fragments, merge_history, _, scale, _ = pipeline.build_track_graph(
        tracking_config, raw_dir, seg_dir, NO_FLOW
    )
    n_axes = ndim + 1

    boxes = sample_crop_boxes(fragments.shape, [0.5, 0.25], seed=0, axes=("y", "x"))
    for box in boxes.values():
        assert len(box.starts) == n_axes
        assert box.axis_names == ("t", *position_names(ndim))
        # Only y and x shrink; time (and z) stay whole.
        for axis in range(n_axes - 2):
            assert (box.starts[axis], box.stops[axis]) == (0, fragments.shape[axis])
    assert boxes[0.5].contains_box(boxes[0.25].starts, boxes[0.25].stops)

    # Merge hypotheses own several leaf fragments; the box is the union of theirs.
    bboxes = compute_node_bboxes(
        graph, fragments, build_node_to_fragments(graph, merge_history)
    )
    assert bboxes, "no bounding boxes computed"
    for node, (starts, stops) in bboxes.items():
        assert len(starts) == len(stops) == n_axes
        assert stops[0] == starts[0] + 1 == graph.nodes[node]["time"] + 1

    # A box covering the whole volume leaves nothing outside, by either rule.
    whole = CropBox(
        fraction=1.0, seed=0, starts=(0,) * n_axes, stops=tuple(fragments.shape),
        volume_shape=tuple(fragments.shape), axes=("y", "x"),
    )
    assert nodes_outside_crop(graph, whole, bboxes, membership="contained") == set()
    assert nodes_outside_crop(graph, whole, membership="centroid", scale=scale) == set()

    index = build_gt_voxel_index(np.asarray(synthetic_fragments))
    assert index.coords.shape[0] == n_axes
    assert index.count_nodes_in(whole) == 4 * T
    assert index.labels_in(boxes[0.25]) < index.labels_in(whole)
