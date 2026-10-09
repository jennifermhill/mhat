"""The candidate-graph cache must not outlive the runs it was built from.

A run may be redone under the same uid after its directory is deleted. Keyed on
uids alone, the cache would then hand back a graph built from the old fragments
or flow, and every result tracked from it would be silently wrong.
"""

import importlib.util
import os
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "04_tracking" / "run_tracking.py"
spec = importlib.util.spec_from_file_location("run_tracking", SCRIPT)
run_tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_tracking)

CONFIG = {
    "seg_result": "seg",
    "flow_result": "flow",
    "size_threshold": 100,
    "max_edge_distance": 50,
    "max_children": 2,
    "divisions": True,
}


def cache_path(seg_dir, flow_dir, cache_dir):
    fingerprint = run_tracking.input_fingerprint(seg_dir, {"2d": None, "3d": flow_dir})
    _, pkl_path, _ = run_tracking.graph_cache_paths(CONFIG, cache_dir, False, fingerprint)
    return pkl_path


def test_redone_input_run_misses_the_cache(tmp_path):
    seg_dir, flow_dir = tmp_path / "seg", tmp_path / "flow"
    for run_dir in (seg_dir, flow_dir):
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("")
        os.utime(run_dir / "config.toml", (1_000_000, 1_000_000))
    before = cache_path(seg_dir, flow_dir, tmp_path / "cache")
    assert cache_path(seg_dir, flow_dir, tmp_path / "cache") == before

    for run_dir in (seg_dir, flow_dir):  # each input, redone under its uid
        os.utime(run_dir / "config.toml", (2_000_000, 2_000_000))
        assert cache_path(seg_dir, flow_dir, tmp_path / "cache") != before
        os.utime(run_dir / "config.toml", (1_000_000, 1_000_000))


def test_drift_distance_is_keyed_only_when_no_flow_exists():
    """With either flow, add_flow_dist_attr wins and drift_distance is unused.

    Keying on it anyway split the cache on a value that could not change the
    graph (2D data, which has 2D flow and no 3D flow). The opposite mistake is
    the dangerous one: two configs differing only in drift_distance would then
    share a digest, and the second would load a graph whose drift_dist came
    from the first's drift vector.
    """
    assert run_tracking.uses_drift_distance({"2d": None, "3d": None})
    for flow_dirs in (
        {"2d": Path("flow2d"), "3d": None},  # 2D data: the bug this pins
        {"2d": None, "3d": Path("flow3d")},
        {"2d": Path("flow2d"), "3d": Path("flow3d")},
    ):
        assert not run_tracking.uses_drift_distance(flow_dirs)


def test_drift_distance_changes_the_key_only_when_it_is_used():
    fingerprint = {"seg": 1_000_000}
    near = {**CONFIG, "drift_distance": [0.0, 3.0, 6.0]}
    far = {**CONFIG, "drift_distance": [0.0, 99.0, 99.0]}
    keyed = [run_tracking.graph_cache_key(c, True, fingerprint) for c in (near, far)]
    assert keyed[0] != keyed[1]
    unkeyed = [run_tracking.graph_cache_key(c, False, fingerprint) for c in (near, far)]
    assert unkeyed[0] == unkeyed[1]
    assert "drift_distance" not in unkeyed[0]
