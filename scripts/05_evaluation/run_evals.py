"""Run evaluate_tracks.py over many eval configs in one process.

Same per-run behavior as evaluate_tracks.py (each eval nests inside its run,
diagnostics included) but pays the import cost once instead of once per config.
A run that fails is reported and skipped rather than aborting the batch.

``--skip-existing`` skips a run that already holds an eval of the same name and
metric set (``<eval_name or GT label>_<metrics>_<timestamp>``, found the way
``collect_sweep.py`` finds it).

Usage:
    python scripts/05_evaluation/run_evals.py --config-dir <dir-of-eval-tomls>
    python scripts/05_evaluation/run_evals.py a_eval.toml b_eval.toml [--only TOKEN ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import traceback
from pathlib import Path

import toml

from collect_sweep import find_eval
from evaluate_tracks import evaluate

from mhat.dataset import Dataset, absolute_path

DEFAULT_METRICS = ["basic", "track_overlap"]


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

        ds = Dataset.from_config(config)
        run_dir = ds.solution_dir(track_result, config.get("track_stage", "tracking"))
        if args.skip_existing:
            label = config.get("eval_name") or absolute_path(config["gt_tracks"]).parent.name
            existing = find_eval(run_dir, label, config.get("metrics", DEFAULT_METRICS))
            if existing is not None:
                print(f"{track_result}: already evaluated ({existing.name}), skipping")
                continue
        # The run directory exists as soon as a config is written, so check for the
        # prediction itself — an empty solution leaves the directory but no geff.
        if not (run_dir / "pred_tracks.zarr").is_dir():
            print(f"{track_result}: no pred_tracks.zarr in {run_dir}, skipping")
            failures.append((track_result, "no prediction (empty solution?)"))
            continue

        print(f"\n=== {track_result} ===")
        try:
            eval_dir = evaluate(config)
            with open(eval_dir / "track_metrics.json") as f:
                overlap = json.load(f).get("TrackOverlapMetrics", {})
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
