"""Post-hoc constant offset sweep on top of SSVM-learned weights.

Standard SSVM optimizes margins between gt=1 and gt=0 costs but doesn't push
absolute costs negative enough for inference to select anything. This script
sweeps a few additive offsets on the dominant edge constant (intensity_constant)
and reports tracking metrics for each. Picks up SSVM-learned weights from
`<exp_dir>/learned_weights.toml` as the starting point.

Usage:
    python scripts/04_tracking/sweep_post_ssvm_offsets.py <learned_weights.toml>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import toml


# 2D sweep: (intensity_constant_offset, drift_constant_offset).
# 1D intensity sweep plateaued at -1000; check if adding a drift offset shifts
# the plateau or helps LNK (still 0.12 below hand-tuned).
INTENSITY_OFFSETS = [-1000.0, -2000.0]
DRIFT_OFFSETS = [0.0, -500.0, -1000.0, -2000.0]


def run_tracking_and_eval(
    config: dict,
    out_subdir: str,
    base_dir: Path,
    experiment: str,
    dataset: str,
):
    """Write modified config, run tracking + eval, return CTC metrics dict."""
    tracking_root = base_dir / "tracking" / experiment / dataset
    out_dir = tracking_root / out_subdir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    # Write the input config to a temp location; run_tracking expects it as an arg
    cfg_path = out_dir / "input_config.toml"
    config["exp_uid"] = out_subdir
    with open(cfg_path, "w") as f:
        toml.dump(config, f)

    # run_tracking.py writes to .../tracking/<experiment>/<dataset>/<exp_uid>/,
    # and exp_uid is set to out_subdir above, so it lands directly in out_dir.
    track_cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_tracking.py"),
        str(cfg_path),
    ]
    print(f"  Running tracking with {out_subdir}...")
    proc = subprocess.run(track_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ERROR: tracking failed:\n{proc.stderr[-2000:]}")
        return None

    if not (out_dir / "pred_tracks.zarr").is_dir():
        print("  ERROR: no pred_tracks.zarr produced (likely empty solution).")
        # Still copy the input config so we have a record
        return None

    # Build a temporary eval config pointing at out_subdir
    eval_cfg = {
        "input_base_dir": str(base_dir).replace("\\", "/") + "/",
        "output_base_dir": str(base_dir).replace("\\", "/") + "/",
        "experiment": experiment,
        "dataset": dataset,
        "track_result": out_subdir,
        "metrics": ["ctc"],
        "matcher": "ctc",
    }
    eval_cfg_path = out_dir / "eval_config.toml"
    with open(eval_cfg_path, "w") as f:
        toml.dump(eval_cfg, f)

    eval_cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "05_evaluation" / "evaluate_tracks.py"),
        str(eval_cfg_path),
    ]
    print(f"  Running eval...")
    proc = subprocess.run(eval_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ERROR: eval failed:\n{proc.stderr[-2000:]}")
        return None

    metrics_path = base_dir / "evaluation" / experiment / dataset / out_subdir / "track_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    return metrics.get("CTCMetrics")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("learned_weights")
    args = parser.parse_args()

    base_config = toml.load(args.learned_weights)
    base_dir = Path(base_config["output_base_dir"])
    experiment = base_config["experiment"]
    dataset = base_config["dataset"]

    start_intensity = base_config["intensity_constant"]
    start_drift = base_config["drift_constant"]
    print(f"Starting intensity_constant: {start_intensity}")
    print(f"Starting drift_constant:     {start_drift}")

    results = []
    for i_off in INTENSITY_OFFSETS:
        for d_off in DRIFT_OFFSETS:
            cfg = deepcopy(base_config)
            cfg["intensity_constant"] = start_intensity + i_off
            cfg["drift_constant"] = start_drift + d_off
            out_subdir = f"ssvm_off_i{int(i_off)}_d{int(d_off)}"
            print(
                f"\n=== intensity_offset={i_off}, drift_offset={d_off} "
                f"→ intensity_const={cfg['intensity_constant']:.2f}, "
                f"drift_const={cfg['drift_constant']:.2f} ==="
            )
            metrics = run_tracking_and_eval(cfg, out_subdir, base_dir, experiment, dataset)
            results.append((i_off, d_off, metrics))

    print("\n" + "=" * 80)
    print(
        f"{'i_offset':>10} {'d_offset':>10} {'TRA':>8} {'DET':>8} {'LNK':>8} "
        f"{'fp':>5} {'fn':>5}"
    )
    print("=" * 80)
    for i_off, d_off, m in results:
        if m is None:
            print(f"{i_off:>10.0f} {d_off:>10.0f}  FAIL/EMPTY")
        else:
            print(
                f"{i_off:>10.0f} {d_off:>10.0f} {m['TRA']:>8.4f} {m['DET']:>8.4f} "
                f"{m['LNK']:>8.4f} {m['fp_nodes']:>5} {m['fn_nodes']:>5}"
            )


if __name__ == "__main__":
    main()
