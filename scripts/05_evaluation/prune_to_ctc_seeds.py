"""Prune a tracking result to the lineages the CTC evaluates, ready for eval.

For Fluo-N3DL-DRO / TRIC / TRIF the Cell Tracking Benchmark scores only the
cells seeded by the first frame of the gold reference, and counts every other
tracked cell as an error. This reads that one frame, matches each marker to a
predicted object, keeps everything descended from those objects, and writes the
result as a sibling tracking result that ``evaluate_tracks.py`` can be pointed
at unchanged.

Takes the same eval config as ``evaluate_tracks.py`` -- it needs
``input_base_dir``, ``experiment``, ``dataset``, ``track_result`` and
``ctc_gt``:

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

    input_base_dir = Path(config["input_base_dir"])
    experiment = config["experiment"]
    dataset = config["dataset"]
    track_result = args.track_result or config["track_result"]
    ctc_gt = config.get("ctc_gt", "01_GT")

    data_dir = input_base_dir / "tracking" / experiment / dataset
    pred_dir = data_dir / track_result
    assert pred_dir.is_dir(), f"Pred data dir {pred_dir} is missing"

    tra_dir = data_dir / ctc_gt / "TRA"
    assert tra_dir.is_dir(), f"CTC TRA dir {tra_dir} is missing"

    out_name = f"{track_result}{args.suffix}"
    out_dir = data_dir / out_name

    print(f"Source:  {pred_dir}")
    print(f"Markers: {tra_dir} (first frame only)")
    print(f"Output:  {out_dir}{' [dry run]' if args.dry_run else ''}")
    print()

    report = prune_to_ctc_seeds(pred_dir, tra_dir, out_dir, dry_run=args.dry_run)
    print(format_report(report))

    if not args.dry_run:
        print()
        print("To evaluate the pruned result, in the eval config set:")
        print(f'    track_result = "{out_name}"')
        print(f'    seg_track_result = "{track_result}"')
        print("    # SEG must stay on the unpruned run")
