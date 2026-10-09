"""Re-evaluate already-completed tracking runs on LSF, without re-tracking them.

``launch_sweep.py`` always submits a tracking job with its eval chained behind
it, which is wrong when the tracking outputs are fine and only the *ground truth*
or the metric set changed. This script submits eval-only jobs for a list of
existing runs and writes a manifest that ``collect_sweep.py`` reads unchanged.

Every re-eval is a new eval nested inside the run it scores, named
``<reeval_id>_<metrics>_<timestamp>`` (``eval_name = <reeval_id>``), so nothing
earlier is overwritten. The re-eval's configs, logs and manifest live in
``<output_dir>/<stage>/sweeps/<reeval_id>/``, beside the tracking sweeps, so a
re-eval id can never reuse a sweep's.

A re-eval is described by a spec TOML::

    reeval_id = "corrgt0730"
    output_dir = "/groups/sgro/sgrolab/jennifer/mhat/experiments/primary_nk_cells/01_cells"
    stage = "tracking"         # or "ssvm" to score SSVM fits; default "tracking"
    all_runs = true            # or: uids = ["full", "sweeps/s1/runs/a", ...]
                               # or: sweep = "<sweep_id>" (every run of that sweep)

    [lsf]
    env = "mhat-cluster"
    queue = "local"
    eval_slots = 1
    eval_walltime = "0:30"

    [eval]
    gt_tracks = ".../primary_nk_cells/01_cells/gt/manual/correct_tracks.zarr"
    metrics = [ "basic", "track_overlap",]
    matcher = "point"
    match_threshold = 10

``all_runs = true`` takes every ungrouped run of the stage that holds a
``pred_tracks.zarr``; runs inside sweeps are named with ``uids`` or ``sweep``.

For the sparsely annotated CTC datasets (Fluo-N3DL-DRO / TRIC / TRIF), adding::

    prune_to_ctc_seeds = true
    prune_suffix = "_ctcseeded"   # optional, this is the default

and ``ctc_tra_dir`` (the CTC ``TRA`` marker folder) to ``[eval]`` puts a
``prune_to_ctc_seeds.py`` job in front of each eval and points the eval at the
pruned run, a sibling ``<uid><suffix>``, so ``TRA``/``DET``/``LNK`` mean
something. ``seg_track_result`` is set back to the unpruned run, because the SEG
reference annotates cells the tracking benchmark excludes. Use
``--metric-set ctc`` when collecting.

Usage, from the repo root on the cluster::

    conda run -n mhat-cluster python scripts/05_evaluation/launch_evals.py \\
        scratch_configs/sweeps/nk_cells_reeval_corrgt.toml [--dry-run] [--only UID ...]

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
from launch_sweep import (  # noqa: E402
    EVAL_INNER,
    FILLED_EVAL_KEYS,
    bsub,
    check_no_old_keys,
    check_no_windows_paths,
)

# Prune each run to the CTC-evaluated lineages before scoring it. Only used when
# the spec sets `prune_to_ctc_seeds`; see the module docstring.
PRUNE_INNER = (
    "conda run -n {env} --no-capture-output "
    "python -u scripts/05_evaluation/prune_to_ctc_seeds.py {cfg} "
    "--track-result {uid} --suffix {suffix}"
)
STAGES = ("tracking", "ssvm")


def validate_spec(spec):
    for key in ("reeval_id", "output_dir"):
        if key not in spec:
            raise SystemExit(f"spec is missing required key {key!r}")
    check_no_old_keys(spec, "the spec")
    if spec.get("stage", "tracking") not in STAGES:
        raise SystemExit(f"stage must be one of {STAGES}")

    eval_stub = spec.get("eval")
    if not eval_stub:
        raise SystemExit("spec is missing the [eval] config stub")
    check_no_old_keys(eval_stub, "[eval]")
    check_no_windows_paths(eval_stub, "[eval]")
    filled = [key for key in FILLED_EVAL_KEYS if key in eval_stub]
    if filled:
        raise SystemExit(f"[eval] must not set {filled}: the launcher fills them per run")
    if "gt_tracks" not in eval_stub:
        raise SystemExit("[eval] is missing gt_tracks")
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
    if spec.get("prune_to_ctc_seeds") and "ctc_tra_dir" not in eval_stub:
        raise SystemExit("prune_to_ctc_seeds needs ctc_tra_dir (the CTC TRA folder) in [eval]")

    chosen = [key for key in ("uids", "all_runs", "sweep") if spec.get(key)]
    if len(chosen) != 1:
        raise SystemExit("spec must set exactly one of `uids`, `all_runs = true`, `sweep`")


def finished(run_dir):
    """A run that wrote its prediction (an empty solution leaves no geff)."""
    return (run_dir / "pred_tracks.zarr").is_dir()


def select_uids(spec, stage_dir):
    if spec.get("uids"):
        return list(spec["uids"])
    if spec.get("all_runs"):
        return sorted(
            p.name for p in stage_dir.iterdir()
            if p.is_dir() and (p / "config.toml").is_file() and finished(p)
        )
    runs_dir = stage_dir / "sweeps" / spec["sweep"] / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"sweep {spec['sweep']!r} has no runs/ in {runs_dir.parent}")
    return sorted(
        f"sweeps/{spec['sweep']}/runs/{p.name}" for p in runs_dir.iterdir()
        if p.is_dir() and finished(p)
    )


def file_stem(uid):
    """A uid as a single file or job name (grouped uids contain slashes)."""
    return uid.replace("/", "__")


def _job_path(path):
    """Path to hand a bsub inner command, which runs with ``cwd=REPO``.

    Relative when it can be, so the submitted command stays readable, but the
    experiments tree does not have to live under the repo -- from a worktree it
    generally does not -- in which case the absolute path is what works.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def build_configs(spec, reeval_dir, stage_dir, uids):
    """Write one eval config per uid; return the manifest run records."""
    eval_stub = spec["eval"]
    eval_dir = reeval_dir / "eval_configs"
    eval_dir.mkdir(parents=True, exist_ok=True)

    prune = spec.get("prune_to_ctc_seeds", False)
    suffix = spec.get("prune_suffix", "_ctcseeded")

    records = []
    for uid in uids:
        scored = f"{uid}{suffix}" if prune else uid
        eval_config = copy.deepcopy(eval_stub)
        eval_config["output_dir"] = spec["output_dir"]
        eval_config["track_result"] = scored
        eval_config["track_stage"] = spec.get("stage", "tracking")
        eval_config["eval_name"] = spec["reeval_id"]
        if prune:
            # SEG must stay on the unpruned run: pruning zeroes exactly the cells
            # the SEG reference annotates. See mhat.evaluation.ctc_seed_prune.
            # evaluate_tracks resolves it as a sibling of the scored run.
            eval_config["seg_track_result"] = uid.rsplit("/", 1)[-1]
        eval_path = eval_dir / f"{file_stem(uid)}.toml"
        with open(eval_path, "w") as handle:
            toml.dump(eval_config, handle)

        # `effective` is what collect_sweep.py's edge report reads to work out
        # which parameters varied across runs; recover it from each run's own
        # saved config. External results (e.g. ultrack) have none -- they just
        # contribute no parameter values.
        track_config_path = stage_dir / uid / "config.toml"
        effective = (
            {k: v for k, v in toml.load(track_config_path).items() if k != "exp_uid"}
            if track_config_path.is_file() else {}
        )
        records.append({
            "label": uid,
            "condition": spec.get("condition", ""),
            # The run the eval scores -- under pruning, the pruned sibling.
            "exp_uid": scored,
            "source_uid": uid,
            "overrides": {},
            "condition_deltas": {},
            "effective": effective,
            "eval_config": _job_path(eval_path),
        })
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
    parser.add_argument("--only", nargs="+", default=None, help="subset of uids to submit")
    args = parser.parse_args()

    spec = toml.load(args.spec)
    validate_spec(spec)

    lsf = spec.get("lsf", {})
    env = lsf.get("env", "mhat-cluster")
    queue = lsf.get("queue", "local")
    eval_slots = lsf.get("eval_slots", 1)
    eval_walltime = lsf.get("eval_walltime", "0:30")

    stage = spec.get("stage", "tracking")
    stage_dir = Path(spec["output_dir"]) / stage
    if not stage_dir.is_dir():
        raise SystemExit(f"{stage} directory {stage_dir} does not exist")

    uids = select_uids(spec, stage_dir)
    if not uids:
        raise SystemExit(f"no finished runs selected under {stage_dir}")
    missing = [uid for uid in uids if not finished(stage_dir / uid)]
    if missing:
        raise SystemExit(f"no pred_tracks.zarr for: {missing}")

    reeval_dir = stage_dir / "sweeps" / spec["reeval_id"]
    manifest_path = reeval_dir / "manifest.toml"
    previous = toml.load(manifest_path) if manifest_path.is_file() else None
    if previous is not None and previous.get("kind") != "reeval":
        raise SystemExit(f"{reeval_dir} belongs to a tracking sweep; pick another reeval_id")
    log_dir = reeval_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    records = build_configs(spec, reeval_dir, stage_dir, uids)
    print(f"Wrote {len(records)} eval configs to {reeval_dir}")

    selected = records
    if args.only:
        known = {record["label"] for record in records}
        unknown = [label for label in args.only if label not in known]
        if unknown:
            raise SystemExit(f"unknown uids: {unknown}")
        selected = [record for record in records if record["label"] in args.only]

    prune = spec.get("prune_to_ctc_seeds", False)
    suffix = spec.get("prune_suffix", "_ctcseeded")
    prune_slots = lsf.get("prune_slots", 4)
    prune_walltime = lsf.get("prune_walltime", "4:00")

    for record in selected:
        label, stem = record["label"], file_stem(record["label"])
        print(f"[{label}]")
        dep_job = None
        if prune:
            # ended(), not done(): the prune is cheap to re-read but the eval
            # should still fire if the job exits nonzero after writing output,
            # matching how launch_sweep chains tracking -> eval.
            dep_job = record["prune_job_id"] = bsub(
                f"{spec['reeval_id']}_prn_{stem}",
                prune_slots, prune_walltime, queue,
                str(log_dir / f"prune_{stem}"),
                PRUNE_INNER.format(
                    env=env, cfg=record["eval_config"],
                    uid=record["source_uid"], suffix=suffix,
                ),
                dry_run=args.dry_run,
            ) or ""
            # Empty only on a dry run, where bsub returns no id; do not emit a
            # malformed `-w ended()`.
            dep_job = dep_job or None
        record["eval_job_id"] = bsub(
            f"{spec['reeval_id']}_evl_{stem}",
            eval_slots, eval_walltime, queue,
            str(log_dir / f"eval_{stem}"),
            EVAL_INNER.format(threads=eval_slots, env=env, cfg=record["eval_config"]),
            dep_job=dep_job,
            dry_run=args.dry_run,
        ) or ""

    if args.dry_run:
        print(f"\n(dry run) {len(selected)} runs; no jobs submitted, no manifest written.")
        return

    # Preserve job ids from earlier partial submissions (repeated --only).
    earlier = {run["label"]: run for run in (previous or {}).get("runs", [])}
    for record in records:
        for key in ("eval_job_id", "prune_job_id") if prune else ("eval_job_id",):
            if not record.get(key):
                record[key] = earlier.get(record["label"], {}).get(key, "")

    eval_stub = spec["eval"]
    manifest = {
        "sweep_id": spec["reeval_id"],
        "kind": "reeval",
        "output_dir": spec["output_dir"],
        "stage": stage,
        # How collect_sweep.py finds each run's eval: <eval_name>_<metrics>_<time>.
        "eval_name": spec["reeval_id"],
        "metrics": eval_stub.get("metrics", ["basic", "track_overlap"]),
        "gt": {key: eval_stub[key] for key in ("gt_tracks", "gt_seg", "ctc_seg_dir")
               if key in eval_stub},
        "spec": str(Path(args.spec).resolve()),
        "queue": queue,
        "eval_slots": eval_slots,
        "eval_walltime": eval_walltime,
        "runs": records,
    }
    with open(manifest_path, "w") as handle:
        toml.dump(manifest, handle)
    print(f"\nSubmitted {len(selected)} eval jobs.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
