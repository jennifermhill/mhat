"""Generate and submit a coordinate-wise parameter sweep to LSF.

A sweep is described by a single spec TOML (see ``configs/sweeps/``). The spec
holds one ``[base]`` tracking config, one ``[eval]`` stub, optional
``[conditions.<name>]`` tables, and a list of ``[[runs]]``, each naming a
``label``, the ``condition`` it belongs to, and a small ``[runs.overrides]``
table. Parameters are layered ``base`` -> ``conditions[condition]`` ->
``overrides``, so a condition states once which costs it zeroes and each run only
states the one or two values it is sweeping. For each run this script:

  1. writes ``<sweep>/track_configs/<label>.toml`` = the layered config, with
     ``exp_uid = "<sweep_id>_<label>"`` (``run_tracking.py`` honours an explicit
     ``exp_uid`` instead of minting a timestamp),
  2. writes ``<sweep>/eval_configs/<label>.toml`` = eval stub with
     ``track_result`` pointing at that ``exp_uid``,
  3. submits the tracking job, then the eval job gated on ``ended(<track job>)``
     -- ``ended()`` rather than ``done()`` because tracking sometimes exits 120
     during interpreter shutdown *after* writing valid output,
  4. records everything in ``<sweep>/manifest.toml``.

Everything (configs, logs, manifest) is written under the shared filesystem, not
node-local scratch, so the compute nodes can read it.

Usage, from the repo root on the cluster::

    conda run -n mhat2 python scripts/04_tracking/launch_sweep.py \\
        configs/sweeps/mda231_01cells_stageA.toml [--dry-run] [--only LABEL ...]

Read back the results with ``scripts/05_evaluation/collect_sweep.py``.
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path

import toml

REPO = Path(__file__).resolve().parents[2]

JOB_ID_RE = re.compile(r"Job <(\d+)>")

# Structural / IO parameters that a sweep must never vary: changing them makes
# runs incomparable (or violates the "do not tune" rule in CLAUDE.md).
PROTECTED_KEYS = {
    "raw_base_dir",
    "input_base_dir",
    "output_base_dir",
    "experiment",
    "dataset",
    "exp_uid",
    "size_threshold",
    "max_edge_distance",
    "max_children",
    "merges",
    "divisions",
    "min_merge_cost",
    "max_merge_cost",
}

# Override keys that legitimately do not appear in a baseline config.
OPTIONAL_KEYS = {
    "base_edge_constant",
    "z_flow_conf_threshold",
    "z_flow_min_pass_pixels",
    "max_timepoints",
}

TRACK_INNER = (
    "export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} "
    "OPENBLAS_NUM_THREADS={threads} && "
    "module load gurobi && conda run -n {env} --no-capture-output "
    "python -u scripts/04_tracking/run_tracking.py {cfg}"
)
EVAL_INNER = (
    "export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} "
    "OPENBLAS_NUM_THREADS={threads} && "
    "conda run -n {env} --no-capture-output "
    "python -u scripts/05_evaluation/evaluate_tracks.py {cfg}"
)


def check_no_windows_paths(config, where):
    """Windows drive letters in a config create literal ``C:``/``Y:`` dirs on Linux."""
    for key, value in config.items():
        if isinstance(value, str) and re.match(r"^[A-Za-z]:[\\/]", value):
            raise SystemExit(f"{where}: {key} = {value!r} is a Windows path")


def validate_spec(spec, allow_protected=False):
    for key in ("sweep_id", "experiment", "dataset"):
        if key not in spec:
            raise SystemExit(f"spec is missing required key {key!r}")
    base = spec.get("base")
    if not base:
        raise SystemExit("spec is missing the [base] tracking config")
    if "exp_uid" in base:
        raise SystemExit("[base] must not set exp_uid -- it is generated per run")
    check_no_windows_paths(base, "[base]")

    eval_stub = spec.get("eval")
    if not eval_stub:
        raise SystemExit("spec is missing the [eval] config stub")
    check_no_windows_paths(eval_stub, "[eval]")
    # Omitting `matcher` silently falls back to PointMatcher and produces wrong
    # (lower) CTC numbers with no error -- see the phantom-regression note in
    # CLAUDE.md. Require it explicitly.
    if "matcher" not in eval_stub:
        raise SystemExit("[eval] must set `matcher` explicitly (e.g. matcher = \"ctc\")")
    if eval_stub["matcher"] == "ctc":
        for key in ("match_threshold", "threshold"):
            if key in eval_stub:
                raise SystemExit(f"[eval] must not set {key!r} with the ctc matcher")
    elif eval_stub["matcher"] == "point" and not (
        "match_threshold" in eval_stub or "threshold" in eval_stub
    ):
        raise SystemExit("[eval] matcher = \"point\" requires match_threshold")

    conditions = spec.get("conditions", {})
    runs = spec.get("runs") or []
    if not runs:
        raise SystemExit("spec has no [[runs]]")
    labels = [run.get("label") for run in runs]
    if None in labels:
        raise SystemExit("every [[runs]] entry needs a `label`")
    dupes = {label for label in labels if labels.count(label) > 1}
    if dupes:
        raise SystemExit(f"duplicate run labels: {sorted(dupes)}")

    def check_keys(keys, where):
        for key in keys:
            if key in PROTECTED_KEYS and not allow_protected:
                raise SystemExit(
                    f"{where} sets protected key {key!r} "
                    "(pass --allow-protected if this is deliberate)"
                )
            if key not in base and key not in OPTIONAL_KEYS:
                raise SystemExit(
                    f"{where} sets {key!r}, which is neither in [base] nor a "
                    "known optional key -- typo?"
                )

    for name, deltas in conditions.items():
        check_keys(deltas, f"[conditions.{name}]")

    for run in runs:
        condition = run.get("condition")
        if condition and condition not in conditions:
            raise SystemExit(
                f"run {run['label']!r} names condition {condition!r}, which has no "
                f"[conditions.{condition}] table"
            )
        # A run with no overrides is legitimate: it reproduces its condition's
        # reference point, which is how a sweep anchors its own comparisons.
        check_keys(run.get("overrides", {}), f"run {run['label']!r}")


def build_configs(spec, sweep_dir):
    """Write per-run tracking and eval configs; return the run records."""
    base = spec["base"]
    eval_stub = spec["eval"]
    conditions = spec.get("conditions", {})
    sweep_id = spec["sweep_id"]

    track_dir = sweep_dir / "track_configs"
    eval_dir = sweep_dir / "eval_configs"
    track_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for run in spec["runs"]:
        label = run["label"]
        condition = run.get("condition", "")
        overrides = run.get("overrides", {})
        exp_uid = f"{sweep_id}_{label}"

        track_config = copy.deepcopy(base)
        track_config.update(conditions.get(condition, {}))
        track_config.update(overrides)
        track_config["exp_uid"] = exp_uid
        track_path = track_dir / f"{label}.toml"
        with open(track_path, "w") as handle:
            toml.dump(track_config, handle)

        eval_config = copy.deepcopy(eval_stub)
        eval_config["experiment"] = spec["experiment"]
        eval_config["dataset"] = spec["dataset"]
        eval_config["track_result"] = exp_uid
        eval_path = eval_dir / f"{label}.toml"
        with open(eval_path, "w") as handle:
            toml.dump(eval_config, handle)

        records.append(
            {
                "label": label,
                "condition": condition,
                "exp_uid": exp_uid,
                "overrides": overrides,
                "condition_deltas": conditions.get(condition, {}),
                # Full layered config, so collect_sweep.py can compare runs on
                # effective values rather than on which keys happened to be
                # written as run-level overrides.
                "effective": {k: v for k, v in track_config.items() if k != "exp_uid"},
                "track_config": str(track_path.relative_to(REPO)),
                "eval_config": str(eval_path.relative_to(REPO)),
            }
        )
    return records


def bsub(job_name, slots, walltime, queue, log_stem, inner, dep_job=None, dry_run=False):
    cmd = [
        "bsub",
        "-J", job_name,
        "-n", str(slots),
        "-W", walltime,
        "-o", f"{log_stem}.%J.log",
        "-e", f"{log_stem}.%J.err",
    ]
    if queue:
        cmd += ["-q", queue]
    if dep_job is not None:
        cmd += ["-w", f"ended({dep_job})"]
    cmd += [inner]

    if dry_run:
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return None

    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    match = JOB_ID_RE.search(out.stdout)
    if not match:
        raise RuntimeError(f"could not parse job id from bsub output for {job_name!r}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", help="path to the sweep spec TOML")
    parser.add_argument("--dry-run", action="store_true", help="write configs and print bsub commands, do not submit")
    parser.add_argument("--only", nargs="+", default=None, help="subset of run labels to submit")
    parser.add_argument("--allow-protected", action="store_true", help="permit overriding structural keys")
    args = parser.parse_args()

    spec = toml.load(args.spec)
    validate_spec(spec, allow_protected=args.allow_protected)

    lsf = spec.get("lsf", {})
    env = lsf.get("env", "mhat2")
    queue = lsf.get("queue", "local")
    track_slots = lsf.get("track_slots", 2)
    track_walltime = lsf.get("track_walltime", "1:00")
    eval_slots = lsf.get("eval_slots", 1)
    eval_walltime = lsf.get("eval_walltime", "0:30")

    output_base_dir = Path(spec["base"]["output_base_dir"])
    sweep_dir = (
        output_base_dir / "tracking" / spec["experiment"] / spec["dataset"]
        / "sweeps" / spec["sweep_id"]
    )
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    records = build_configs(spec, sweep_dir)
    print(f"Wrote {len(records)} track + eval configs to {sweep_dir}")

    selected = records
    if args.only:
        known = {record["label"] for record in records}
        unknown = [label for label in args.only if label not in known]
        if unknown:
            raise SystemExit(f"unknown run labels: {unknown}")
        selected = [record for record in records if record["label"] in args.only]

    for record in selected:
        label = record["label"]
        print(f"[{label}] {record['condition']} {record['overrides']}")
        track_job = bsub(
            f"{spec['sweep_id']}_trk_{label}",
            track_slots, track_walltime, queue,
            str(log_dir / f"track_{label}"),
            TRACK_INNER.format(threads=track_slots, env=env, cfg=record["track_config"]),
            dry_run=args.dry_run,
        )
        eval_job = bsub(
            f"{spec['sweep_id']}_evl_{label}",
            eval_slots, eval_walltime, queue,
            str(log_dir / f"eval_{label}"),
            EVAL_INNER.format(threads=eval_slots, env=env, cfg=record["eval_config"]),
            dep_job=track_job, dry_run=args.dry_run,
        )
        record["track_job_id"] = track_job or ""
        record["eval_job_id"] = eval_job or ""

    if not args.dry_run:
        manifest_path = sweep_dir / "manifest.toml"
        # Preserve job ids from earlier partial submissions (repeated --only).
        if manifest_path.is_file():
            previous = {
                run["label"]: run for run in toml.load(manifest_path).get("runs", [])
            }
            for record in records:
                old = previous.get(record["label"], {})
                for key in ("track_job_id", "eval_job_id"):
                    if not record.get(key) and old.get(key):
                        record[key] = old[key]
                record.setdefault("track_job_id", "")
                record.setdefault("eval_job_id", "")
        else:
            for record in records:
                record.setdefault("track_job_id", "")
                record.setdefault("eval_job_id", "")

        manifest = {
            "sweep_id": spec["sweep_id"],
            "experiment": spec["experiment"],
            "dataset": spec["dataset"],
            "spec": str(Path(args.spec).resolve()),
            "queue": queue,
            "track_slots": track_slots,
            "track_walltime": track_walltime,
            "eval_slots": eval_slots,
            "eval_walltime": eval_walltime,
            "eval_base_dir": spec["eval"].get("output_base_dir", spec["base"]["output_base_dir"]),
            "runs": records,
        }
        with open(manifest_path, "w") as handle:
            toml.dump(manifest, handle)
        print(f"\nSubmitted {len(selected)} tracking + {len(selected)} eval jobs.")
        print(f"Manifest: {manifest_path}")
    else:
        print(f"\n(dry run) {len(selected)} runs; no jobs submitted, no manifest written.")


if __name__ == "__main__":
    main()
