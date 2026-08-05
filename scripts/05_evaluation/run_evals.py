"""Run evaluate_tracks.py over many eval configs in one process.

Same per-run behaviour as evaluate_tracks.py — including the `gt_data_dir`
override — but pays the import cost once instead of once per config. A run that
fails is reported and skipped rather than aborting the batch.

Usage:
    python scripts/05_evaluation/run_evals.py --config-dir <dir-of-eval-tomls>
    python scripts/05_evaluation/run_evals.py a_eval.toml b_eval.toml [--only TOKEN ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import traceback
from pathlib import Path

import toml

from evaluate_tracks import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None, help="track_result values to keep")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    paths = list(args.configs)
    if args.config_dir:
        paths += [Path(p) for p in sorted(glob.glob(str(args.config_dir / "*.toml")))]
    if not paths:
        parser.error("no eval configs given")

    failures = []
    done = 0
    for path in sorted(set(paths)):
        config = toml.load(path)
        track_result = config["track_result"]
        if args.only and track_result not in set(args.only):
            continue

        input_base_dir = Path(config["input_base_dir"])
        output_base_dir = Path(config["output_base_dir"])
        experiment = config["experiment"]
        dataset = config["dataset"]

        if config.get("gt_data_dir"):
            gt_data_dir = Path(config["gt_data_dir"])
        else:
            gt_data_dir = input_base_dir / "tracking" / experiment / dataset
        pred_data_dir = input_base_dir / "tracking" / experiment / dataset / track_result
        output_dir = output_base_dir / "evaluation" / experiment / dataset / track_result

        if args.skip_existing and (output_dir / "track_metrics.json").is_file():
            print(f"{track_result}: metrics already present, skipping")
            continue
        # The run directory exists as soon as a config is written, so check for the
        # prediction itself — an empty solution leaves the directory but no geff.
        if not (pred_data_dir / "pred_tracks.zarr").is_dir():
            print(f"{track_result}: no pred_tracks.zarr in {pred_data_dir}, skipping")
            failures.append((track_result, "no prediction (empty solution?)"))
            continue

        print(f"\n=== {track_result} ===")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            track_metrics = run_evaluation(config, gt_data_dir, pred_data_dir)
            with open(output_dir / "track_metrics.json", "w") as f:
                json.dump(track_metrics, f)
            shutil.copy2(path, output_dir / "eval_config.toml")
            tracking_config = pred_data_dir / "config.toml"
            if tracking_config.is_file():
                shutil.copy2(tracking_config, output_dir / "tracking_config.toml")
            overlap = track_metrics.get("TrackOverlapMetrics", {})
            if overlap:
                print(
                    f"  TE={overlap.get('target_effectiveness'):.4f} "
                    f"purity={overlap.get('track_purity'):.4f}"
                )
            done += 1
        except Exception:
            print(f"  {track_result} FAILED:\n{traceback.format_exc()}")
            failures.append((track_result, "exception"))

    print(f"\nEvaluated {done} run(s)")
    if failures:
        print(f"{len(failures)} failure(s):")
        for token, why in failures:
            print(f"  {token}: {why}")


if __name__ == "__main__":
    main()
