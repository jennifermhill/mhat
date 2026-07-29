"""Run the post-evaluation diagnostics for a tracking result.

``evaluate_tracks.py`` calls :func:`run_diagnostics` after computing the
metrics, so every evaluation also produces the FN-edge, FN-node and GT-drift
reports next to ``track_metrics.json``. Each analysis is independent and is
skipped (with the reason recorded) rather than failing the evaluation.

Config keys, all optional, in the eval config:

    [diagnostics]
    enabled = true        # master switch
    fn_edges = true
    fn_nodes = true
    gt_drift = true
    plots = true
    proximity_threshold = 20.0   # FN node "is there anything here?" radius
    max_listed = 500             # cap on individually listed FN nodes
"""

import json
import traceback
from pathlib import Path

import numpy as np
import toml

from mhat.evaluation.evaluate_tracking import match_tracking
from mhat.evaluation.fn_analysis import analyze_fn_edges, analyze_fn_nodes
from mhat.evaluation.gt_drift import compare_gt_drift

DIAGNOSTICS_SUBDIR = "diagnostics"


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_run_config(pred_data_dir):
    """The tracking config saved alongside the tracking result ({} if absent)."""
    config_path = Path(pred_data_dir) / "config.toml"
    if config_path.is_file():
        return toml.load(config_path)
    return {}


def run_diagnostics(config, gt_data_dir, pred_data_dir, output_dir, matched=None):
    """Run every enabled diagnostic, writing reports under ``output_dir/diagnostics``.

    Args:
        config (dict): Evaluation config.
        gt_data_dir (Path): Ground truth directory.
        pred_data_dir (Path): Tracking result directory.
        output_dir (Path): Evaluation output directory (the reports go in a
            ``diagnostics`` subfolder).
        matched: Existing traccuracy ``Matched`` object. Computed here if None,
            which re-runs the matcher.

    Returns:
        dict: Per-analysis summaries, also written to ``diagnostics_summary.json``.
    """
    diag_config = config.get("diagnostics", {}) or {}
    if not diag_config.get("enabled", True):
        print("Diagnostics disabled in the eval config; skipping.")
        return {}

    gt_data_dir = Path(gt_data_dir)
    pred_data_dir = Path(pred_data_dir)
    diag_dir = Path(output_dir) / DIAGNOSTICS_SUBDIR
    diag_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(pred_data_dir)
    plots = diag_config.get("plots", True)
    summary = {}

    needs_match = diag_config.get("fn_edges", True) or diag_config.get("fn_nodes", True)
    if matched is None and needs_match:
        print("Computing matching for diagnostics...")
        matched = match_tracking(config, gt_data_dir, pred_data_dir)

    def _run(name, fn):
        if not diag_config.get(name, True):
            return
        print(f"\nRunning {name} diagnostic...")
        try:
            summary[name] = fn()
        except Exception as e:  # a broken diagnostic must not lose the metrics
            traceback.print_exc()
            print(f"Diagnostic '{name}' failed: {e}")
            summary[name] = {"error": f"{type(e).__name__}: {e}"}

    _run("fn_edges", lambda: analyze_fn_edges(
        matched, pred_data_dir, run_config, output_dir=diag_dir, plots=plots,
    ))
    _run("fn_nodes", lambda: analyze_fn_nodes(
        matched, config, pred_data_dir, run_config, output_dir=diag_dir,
        proximity_threshold=diag_config.get("proximity_threshold"),
        max_listed=diag_config.get("max_listed"),
    ))
    _run("gt_drift", lambda: compare_gt_drift(
        config, gt_data_dir, pred_data_dir, run_config, output_dir=diag_dir, plots=plots,
    ))

    summary_path = diag_dir / "diagnostics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nDiagnostics written to {diag_dir}")

    return summary
