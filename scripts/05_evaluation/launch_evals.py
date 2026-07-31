"""Re-evaluate already-completed tracking runs on LSF, without re-tracking them.

``launch_sweep.py`` always submits a tracking job with its eval chained behind
it, which is wrong when the tracking outputs are fine and only the *ground truth*
changed. This script submits eval-only jobs for a list of existing ``exp_uid``s
and writes a ``launch_sweep``-compatible ``manifest.toml``, so
``collect_sweep.py`` reads the result with no changes.

A re-eval is described by a spec TOML::

    reeval_id = "corrgt0730"
    experiment = "primary_nk_cells"
    dataset = "01_cells"
    all_runs = true            # or: uids = ["full", "no_drift", ...]

    [lsf]
    env = "mhat2"
    queue = "local"
    eval_slots = 1
    eval_walltime = "0:30"

    [eval]
    input_base_dir = "/groups/sgro/sgrolab/jennifer/mhat/experiments/"
    output_base_dir = "/groups/sgro/sgrolab/jennifer/mhat/experiments/"
    metrics = [ "basic", "track_overlap",]
    matcher = "point"
    match_threshold = 10

``all_runs = true`` discovers every subdirectory of the dataset's tracking
directory that holds a ``pred_tracks.zarr``.

**``evaluate_tracks.py`` writes ``track_metrics.json`` in place**, so re-running
an eval destroys the previous metrics for that ``exp_uid``. That is the point
here, but it means the old numbers only survive wherever they were written down.

Usage, from the repo root on the cluster::

    conda run -n mhat2 python scripts/05_evaluation/launch_evals.py \\
        configs/sweeps/nk_cells_reeval_corrgt.toml [--dry-run] [--only UID ...]

Read back the results with ``scripts/05_evaluation/collect_sweep.py``.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import toml

REPO = Path(__file__).resolve().parents[2]

# bsub() and the eval command string already exist in the sweep launcher; import
# them so the two paths cannot drift apart. Its main() is guarded, so importing
# it runs nothing.
sys.path.insert(0, str(REPO / "scripts" / "04_tracking"))
from launch_sweep import EVAL_INNER, bsub, check_no_windows_paths  # noqa: E402


def validate_spec(spec):
    for key in ("reeval_id", "experiment", "dataset"):
        if key not in spec:
            raise SystemExit(f"spec is missing required key {key!r}")

    eval_stub = spec.get("eval")
    if not eval_stub:
        raise SystemExit("spec is missing the [eval] config stub")
    check_no_windows_paths(eval_stub, "[eval]")
    for key in ("input_base_dir", "output_base_dir"):
        if key not in eval_stub:
            raise SystemExit(f"[eval] is missing {key!r}")
    # Same trap as in launch_sweep: omitting `matcher` silently falls back to
    # PointMatcher and produces wrong metrics with no error (CLAUDE.md).
    if "matcher" not in eval_stub:
        raise SystemExit('[eval] must set `matcher` explicitly (e.g. matcher = "ctc")')
    if eval_stub["matcher"] == "ctc":
        for key in ("match_threshold", "threshold"):
            if key in eval_stub:
                raise SystemExit(f"[eval] must not set {key!r} with the ctc matcher")
    elif eval_stub["matcher"] == "point" and not (
        "match_threshold" in eval_stub or "threshold" in eval_stub
    ):
        raise SystemExit('[eval] matcher = "point" requires match_threshold')

    if "uids" not in spec and not spec.get("all_runs"):
        raise SystemExit("spec must set either `uids = [...]` or `all_runs = true`")


def discover_uids(track_dir):
    """Every subdirectory holding a pred_tracks.zarr, i.e. every finished run."""
    return sorted(
        path.name
        for path in track_dir.iterdir()
        if path.is_dir() and (path / "pred_tracks.zarr").is_dir()
    )


def build_configs(spec, reeval_dir, track_dir, uids):
    """Write one eval config per uid; return the manifest run records."""
    eval_stub = spec["eval"]
    eval_dir = reeval_dir / "eval_configs"
    eval_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for uid in uids:
        eval_config = copy.deepcopy(eval_stub)
        eval_config["experiment"] = spec["experiment"]
        eval_config["dataset"] = spec["dataset"]
        eval_config["track_result"] = uid
        eval_path = eval_dir / f"{uid}.toml"
        with open(eval_path, "w") as handle:
            toml.dump(eval_config, handle)

        # `effective` is what collect_sweep.py's edge report reads to work out
        # which parameters varied across runs. These runs predate the sweep
        # manifests, so recover it from each run's own saved tracking config.
        # Externally-produced results (e.g. ultrack) have none -- that is fine,
        # they just contribute no parameter values.
        track_config_path = track_dir / uid / "config.toml"
        if track_config_path.is_file():
            effective = {
                key: value
                for key, value in toml.load(track_config_path).items()
                if key != "exp_uid"
            }
        else:
            effective = {}

        records.append(
            {
                "label": uid,
                "condition": spec.get("condition", ""),
                "exp_uid": uid,
                "overrides": {},
                "condition_deltas": {},
                "effective": effective,
                "eval_config": str(eval_path.relative_to(REPO)),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("spec", help="path to the re-eval spec TOML")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write eval configs and print bsub commands, do not submit",
    )
    parser.add_argument("--only", nargs="+", default=None, help="subset of exp_uids to submit")
    args = parser.parse_args()

    spec = toml.load(args.spec)
    validate_spec(spec)

    lsf = spec.get("lsf", {})
    env = lsf.get("env", "mhat2")
    queue = lsf.get("queue", "local")
    eval_slots = lsf.get("eval_slots", 1)
    eval_walltime = lsf.get("eval_walltime", "0:30")

    base_dir = Path(spec["eval"]["input_base_dir"])
    track_dir = base_dir / "tracking" / spec["experiment"] / spec["dataset"]
    if not track_dir.is_dir():
        raise SystemExit(f"tracking dir {track_dir} does not exist")

    uids = spec.get("uids") or discover_uids(track_dir)
    if not uids:
        raise SystemExit(f"no runs with a pred_tracks.zarr found under {track_dir}")
    missing = [uid for uid in uids if not (track_dir / uid / "pred_tracks.zarr").is_dir()]
    if missing:
        raise SystemExit(f"no pred_tracks.zarr for: {missing}")

    reeval_dir = track_dir / "reevals" / spec["reeval_id"]
    log_dir = reeval_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    records = build_configs(spec, reeval_dir, track_dir, uids)
    print(f"Wrote {len(records)} eval configs to {reeval_dir}")

    selected = records
    if args.only:
        known = {record["label"] for record in records}
        unknown = [label for label in args.only if label not in known]
        if unknown:
            raise SystemExit(f"unknown exp_uids: {unknown}")
        selected = [record for record in records if record["label"] in args.only]

    for record in selected:
        label = record["label"]
        print(f"[{label}]")
        record["eval_job_id"] = bsub(
            f"{spec['reeval_id']}_evl_{label}",
            eval_slots, eval_walltime, queue,
            str(log_dir / f"eval_{label}"),
            EVAL_INNER.format(threads=eval_slots, env=env, cfg=record["eval_config"]),
            dry_run=args.dry_run,
        ) or ""

    if args.dry_run:
        print(f"\n(dry run) {len(selected)} runs; no jobs submitted, no manifest written.")
        return

    manifest_path = reeval_dir / "manifest.toml"
    # Preserve job ids from earlier partial submissions (repeated --only).
    if manifest_path.is_file():
        previous = {run["label"]: run for run in toml.load(manifest_path).get("runs", [])}
        for record in records:
            if not record.get("eval_job_id"):
                record["eval_job_id"] = previous.get(record["label"], {}).get("eval_job_id", "")
    else:
        for record in records:
            record.setdefault("eval_job_id", "")

    manifest = {
        "sweep_id": spec["reeval_id"],
        "experiment": spec["experiment"],
        "dataset": spec["dataset"],
        "spec": str(Path(args.spec).resolve()),
        "queue": queue,
        "eval_slots": eval_slots,
        "eval_walltime": eval_walltime,
        "eval_base_dir": spec["eval"]["output_base_dir"],
        "runs": records,
    }
    with open(manifest_path, "w") as handle:
        toml.dump(manifest, handle)
    print(f"\nSubmitted {len(selected)} eval jobs.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
