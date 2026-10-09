"""collect_sweep finds a run's eval by exact name, never by prefix or by recency.

Picking the wrong directory would put another eval's numbers into a sweep's
results.csv without any error: a prefix match crosses into a sweep whose id
starts the same way, and an eval that crashed has no metrics yet.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "05_evaluation" / "collect_sweep.py"
spec = importlib.util.spec_from_file_location("collect_sweep", SCRIPT)
collect_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect_sweep)

TS = "2026-10-09_12-00-00"


def make_evals(run_dir, names_with_metrics):
    for name, complete in names_with_metrics:
        eval_dir = run_dir / "evaluation" / name
        eval_dir.mkdir(parents=True)
        if complete:
            (eval_dir / "track_metrics.json").write_text("{}")


def test_exact_complete_eval_is_found(tmp_path):
    make_evals(tmp_path, [
        (f"s1_ctc+track_overlap_{TS}", True),
        (f"s1_ctc+track_overlap_2026-10-09_13-00-00", False),  # crashed
        (f"s1_baseline_ctc+track_overlap_{TS}", True),  # another sweep's id
        (f"s1_ctc_{TS}", True),  # another metric set
    ])
    found = collect_sweep.find_eval(tmp_path, "s1", ["track_overlap", "ctc"])
    assert found.name == f"s1_ctc+track_overlap_{TS}"


def test_two_complete_matches_are_refused(tmp_path):
    make_evals(tmp_path, [(f"s1_ctc_{TS}", True), ("s1_ctc_2026-10-09_13-00-00", True)])
    with pytest.raises(SystemExit):
        collect_sweep.find_eval(tmp_path, "s1", ["ctc"])
