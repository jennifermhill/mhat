"""Prune a tracking result to the lineages the CTC evaluates, ready for eval.

For Fluo-N3DL-DRO / TRIC / TRIF the Cell Tracking Benchmark scores only the
cells seeded by the first frame of the gold reference, and counts every other
tracked cell as an error. This reads that one frame, matches each marker to a
predicted object, keeps everything descended from those objects, and writes the
result as a new tracking run beside the source, ``<track_result><suffix>``, that
``evaluate_tracks.py`` can be pointed at unchanged. Its ``config.toml`` is the
source run's, plus ``pruned_from``.

Takes the same eval config as ``evaluate_tracks.py`` -- it needs ``output_dir``,
``track_result`` (``track_stage`` if not ``"tracking"``) and ``ctc_tra_dir``, the
CTC ``TRA`` marker folder:

    python scripts/05_evaluation/prune_to_ctc_seeds.py <eval_config.toml> --dry-run
    python scripts/05_evaluation/prune_to_ctc_seeds.py <eval_config.toml>

``--dry-run`` reports the matching and the lineage survival without writing
anything; read it first, because the survival count is the ceiling on what a
pruned TRA can reach. Then set ``track_result`` to the pruned result and
``seg_track_result`` to the unpruned one before running the evaluation.
"""

import argparse
from pathlib import Path

import toml

from mhat.dataset import Dataset, require_dir
from mhat.evaluation.ctc_seed_prune import format_report, prune_to_ctc_seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="eval config TOML (the one evaluate_tracks.py takes)"
    )
    parser.add_argument(
        "--suffix",
        default="_ctcseeded",
        help="appended to track_result to name the pruned result (default: _ctcseeded)",
    )
    parser.add_argument(
        "--track-result",
        default=None,
        help="prune this run instead of the config's `track_result`. Lets one "
             "eval config (whose track_result already names the *pruned* result) "
             "drive the prune of its source run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the matching and lineage survival without writing anything",
    )
    args = parser.parse_args()
    config = toml.load(args.config)

    ds = Dataset.from_config(config)
    stage = config.get("track_stage", "tracking")
    track_result = args.track_result or config["track_result"]
    pred_dir = require_dir(ds.solution_dir(track_result, stage), "Source run")
    tra_dir = require_dir(Path(config["ctc_tra_dir"]), "CTC TRA folder")

    out_name = f"{track_result}{args.suffix}"
    out_dir = ds.check_new_run(stage, out_name)

    print(f"Source:  {pred_dir}")
    print(f"Markers: {tra_dir} (first frame only)")
    print(f"Output:  {out_dir}{' [dry run]' if args.dry_run else ''}")
    print()

    report = prune_to_ctc_seeds(pred_dir, tra_dir, out_dir, dry_run=args.dry_run)
    print(format_report(report))

    if not args.dry_run:
        # The pruned run is a run of its own: say where it came from.
        config_path = out_dir / "config.toml"
        run_config = toml.load(config_path) if config_path.is_file() else {}
        run_config["exp_uid"] = out_name
        run_config["pruned_from"] = track_result
        with open(config_path, "w") as handle:
            toml.dump(run_config, handle)

        print()
        print("To evaluate the pruned result, in the eval config set:")
        print(f'    track_result = "{out_name}"')
        print(f'    seg_track_result = "{Path(track_result).name}"')
        print("    # SEG must stay on the unpruned run")
