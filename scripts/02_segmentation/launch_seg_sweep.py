"""Sweep segmentation parameters on LSF and score each run by GT detectability.

The tracking sweeps (``scripts/04_tracking/launch_sweep.py``) start from a fixed
segmentation and vary ILP costs. This does the opposite: it varies the
segmentation and scores the *candidate pool* it produces, without tracking at
all. That is the right loop once the diagnostics say the correct object is
missing from the segmentation -- tracking cannot recover what was never
segmented, and a full tracking run costs hours where this costs minutes.

Each run gets a GPU job for ``create_seg_hypotheses.py`` and a dependent CPU job
for ``scripts/05_evaluation/seg_detectability.py``. A final collect job, gated on
all of them, writes the comparison table.

The sweep lives in ``<output_dir>/segmentation/sweeps/<sweep_id>/``, and so do
its runs: run ``<label>`` is the segmentation run ``sweeps/<sweep_id>/runs/<label>``
(a tracking config reads it with that ``seg_result``). As with tracking sweeps, a
selected label whose run directory exists is refused unless ``--retry``, and
each launch keeps the spec as launched in ``<sweep>/specs/<timestamp>.toml``
(``spec_copy`` in each manifest run).

Spec TOML::

    sweep_id   = "droseg1"

    [lsf]
    env = "mhat-cluster"
    seg_queue = "gpu_a100"      # cellpose needs a GPU; 12 slots is the A100 ratio
    seg_slots = 12
    seg_gpu = "num=1"
    seg_walltime = "3:00"

    [detectability]
    gt_tra = ".../01_GT/TRA"
    baseline = "aniso_none_cp0" # run label the collect step diffs against

    [base]                      # a complete seg config (raw_path, output_dir, ...), minus exp_uid
    ...

    [[runs]]
    label = "aniso5_cp0"
    [runs.overrides]
    "seg_params.cellpose_params.anisotropy" = 5.0

Overrides are dotted paths into ``[base]``. They can only add or replace a key,
never remove one -- so anything whose "off" state is *absent* (cellpose reads
``anisotropy = None`` as isotropic) must be left out of ``[base]`` and added
only by the runs that want it.

Usage::

    conda run -n mhat-cluster python scripts/02_segmentation/launch_seg_sweep.py \\
        scripts/02_segmentation/dro_seg_sweep.toml [--dry-run] [--only LABEL ...]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import toml

REPO = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO / "scripts" / "04_tracking"))
from launch_sweep import (  # noqa: E402
    assign_uids,
    bsub,
    check_no_old_keys,
    check_no_windows_paths,
    read_spec,
    save_spec_copy,
)

RUN_STAGE = "segmentation"

SEG_INNER = (
    "conda run -n {env} --no-capture-output "
    "python -u scripts/02_segmentation/create_seg_hypotheses.py {cfg}"
)

DET_INNER = (
    "export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} "
    "OPENBLAS_NUM_THREADS={threads} && "
    "conda run -n {env} --no-capture-output "
    "python -u scripts/05_evaluation/seg_detectability.py "
    "--seg-dir {seg_dir} --gt-tra {gt_tra} --label {label} "
    "--size-threshold {size_threshold} --min-cost {min_cost} --max-cost {max_cost}"
)

COLLECT_INNER = (
    "conda run -n {env} --no-capture-output "
    "python -u scripts/05_evaluation/collect_seg_sweep.py {manifest}"
)

# Structural keys that decide *what* is being segmented rather than how. An
# override here means the runs are no longer comparable, which is the one thing
# a sweep must guarantee.
PROTECTED = {"raw_path", "output_dir", "exp_uid"}


def set_dotted(config, path, value):
    keys = path.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def validate_spec(spec):
    for key in ("sweep_id", "base", "runs"):
        if key not in spec:
            raise SystemExit(f"spec is missing required key {key!r}")
    check_no_old_keys(spec, "the spec")
    check_no_old_keys(spec["base"], "[base]")
    for key in ("raw_path", "output_dir"):
        if key not in spec["base"]:
            raise SystemExit(f"[base] is missing {key!r}")
    if "exp_uid" in spec["base"]:
        raise SystemExit("[base] must not set exp_uid -- it is generated per run")
    check_no_windows_paths(spec["base"], "[base]")

    det = spec.get("detectability", {})
    if "gt_tra" not in det:
        raise SystemExit("[detectability] is missing gt_tra")
    if not Path(det["gt_tra"]).is_dir():
        raise SystemExit(f"[detectability] gt_tra {det['gt_tra']} is not a directory")

    # The cellpose branch of create_seg_hypotheses reads none of spot_sigma /
    # outline_sigma / threshold / mad_k -- those belong to threshold_labeling --
    # and cellpose ignores flow_threshold when do_3D is set. Sweeping any of
    # them produces identical runs, so refuse rather than burn the GPU.
    seg_params = spec["base"].get("seg_params", {})
    inert = set()
    if seg_params.get("seg_method") == "cellpose":
        inert |= {
            "seg_params.spot_sigma", "seg_params.outline_sigma",
            "seg_params.threshold", "seg_params.mad_k",
        }
        if seg_params.get("cellpose_params", {}).get("do_3D"):
            inert.add("seg_params.cellpose_params.flow_threshold")

    labels = set()
    for run in spec["runs"]:
        if "label" not in run:
            raise SystemExit("every [[runs]] entry needs a label")
        if run["label"] in labels:
            raise SystemExit(f"duplicate run label {run['label']!r}")
        labels.add(run["label"])
        for path in run.get("overrides", {}):
            root = path.split(".")[0]
            if root in PROTECTED:
                raise SystemExit(f"[{run['label']}] may not override {root!r}")
            if path in inert:
                raise SystemExit(
                    f"[{run['label']}] overrides {path!r}, which this segmentation "
                    "method ignores -- the run would be a duplicate"
                )

    baseline = det.get("baseline")
    if baseline and baseline not in labels:
        raise SystemExit(f"[detectability] baseline {baseline!r} is not a run label")


def build_configs(spec, sweep_dir, uids):
    config_dir = sweep_dir / "seg_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for run in spec["runs"]:
        label = run["label"]
        config = copy.deepcopy(spec["base"])
        config["exp_uid"] = uids[label]
        for path, value in run.get("overrides", {}).items():
            set_dotted(config, path, value)

        config_path = config_dir / f"{label}.toml"
        with open(config_path, "w") as handle:
            toml.dump(config, handle)

        seg_dir = Path(spec["base"]["output_dir"]) / RUN_STAGE / config["exp_uid"]
        records.append({
            "label": label,
            "exp_uid": config["exp_uid"],
            "seg_config": str(config_path),
            "seg_dir": str(seg_dir),
            "overrides": run.get("overrides", {}),
        })
    return records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("spec")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument(
        "--retry", action="store_true",
        help="give selected labels whose run directory already exists a new run name "
        "(runs/<label>_r2, ...) instead of refusing",
    )
    args = parser.parse_args()

    raw_spec, spec = read_spec(args.spec)
    validate_spec(spec)

    lsf = spec.get("lsf", {})
    env = lsf.get("env", "mhat-cluster")
    seg_queue = lsf.get("seg_queue", "gpu_a100")
    seg_slots = lsf.get("seg_slots", 12)
    seg_gpu = lsf.get("seg_gpu", "num=1")
    seg_walltime = lsf.get("seg_walltime", "3:00")
    det_queue = lsf.get("det_queue", "local")
    det_slots = lsf.get("det_slots", 4)
    det_walltime = lsf.get("det_walltime", "1:00")

    det = spec["detectability"]
    sweep_dir = Path(spec["base"]["output_dir"]) / RUN_STAGE / "sweeps" / spec["sweep_id"]
    manifest_path = sweep_dir / "manifest.toml"
    previous_runs = (
        {r["label"]: r for r in toml.load(manifest_path).get("runs", [])}
        if manifest_path.is_file() else {}
    )
    labels = [run["label"] for run in spec["runs"]]
    if args.only:
        unknown = [label for label in args.only if label not in labels]
        if unknown:
            raise SystemExit(f"unknown labels: {unknown}")
    selected_labels = set(args.only) if args.only else set(labels)
    uids = assign_uids(
        spec, selected_labels,
        {label: run.get("exp_uid") for label, run in previous_runs.items()},
        args.retry, stage=RUN_STAGE,
    )

    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    records = build_configs(spec, sweep_dir, uids)
    print(f"Wrote {len(records)} seg configs to {sweep_dir / 'seg_configs'}")
    selected = [record for record in records if record["label"] in selected_labels]
    if not args.dry_run:
        spec_copy = save_spec_copy(raw_spec, sweep_dir)
        print(f"Kept the spec as {sweep_dir / spec_copy}")
        for record in selected:
            record["spec_copy"] = spec_copy

    det_jobs = []          # this invocation's det jobs, for the collect gate
    for record in selected:
        label = record["label"]
        print(f"[{label}]")
        seg_job = bsub(
            f"{spec['sweep_id']}_seg_{label}",
            seg_slots, seg_walltime, seg_queue,
            str(log_dir / f"seg_{label}"),
            SEG_INNER.format(env=env, cfg=record["seg_config"]),
            dry_run=args.dry_run,
            gpu=seg_gpu,
        )
        record["seg_job_id"] = seg_job or ""
        det_job = bsub(
            f"{spec['sweep_id']}_det_{label}",
            det_slots, det_walltime, det_queue,
            str(log_dir / f"det_{label}"),
            DET_INNER.format(
                threads=det_slots, env=env,
                seg_dir=record["seg_dir"], gt_tra=det["gt_tra"], label=label,
                size_threshold=det.get("size_threshold", 20),
                min_cost=det.get("min_cost", 0.0),
                max_cost=det.get("max_cost", 1.0),
            ),
            # ended(), not done(): create_seg_hypotheses has the same
            # exit-after-writing quirk the tracking jobs do.
            dep_job=seg_job or None,
            dry_run=args.dry_run,
        )
        record["det_job_id"] = det_job or ""
        if det_job:
            det_jobs.append(det_job)

    if args.dry_run:
        print(f"\n(dry run) {len(selected)} runs; nothing submitted, no manifest.")
        return

    # Preserve job ids and spec copies from earlier partial submissions
    # (repeated --only), unless the label was retried under a new run name since.
    for record in records:
        old = previous_runs.get(record["label"], {})
        for key in ("seg_job_id", "det_job_id", "spec_copy"):
            if not record.get(key):
                same_run = old.get("exp_uid") == record["exp_uid"]
                record[key] = old.get(key, "") if same_run else ""

    manifest = {
        "sweep_id": spec["sweep_id"],
        "output_dir": spec["base"]["output_dir"],
        "stage": RUN_STAGE,
        "spec": str(Path(args.spec).resolve()),
        "sweep_dir": str(sweep_dir),
        "baseline": det.get("baseline", ""),
        "runs": records,
    }
    with open(manifest_path, "w") as handle:
        toml.dump(manifest, handle)

    # Gate the collect on every run's det job, not just this invocation's.
    # A `--only` rerun would otherwise submit a collect that fires as soon as the
    # resubmitted runs land and overwrite results.csv with a table missing every
    # run still on the GPU. Ids carried forward from the manifest above cover the
    # runs this invocation did not touch; already-finished ones satisfy ended()
    # immediately, so including them costs nothing.
    gate = [record["det_job_id"] for record in records if record.get("det_job_id")]
    if gate:
        bsub(
            f"{spec['sweep_id']}_collect", 1, "0:20", det_queue,
            str(log_dir / "collect"),
            COLLECT_INNER.format(env=env, manifest=manifest_path),
            dep_job=" && ".join(f"ended({job})" for job in gate),
            dry_run=False,
            raw_dep=True,
        )

    print(f"\nSubmitted {len(selected)} runs.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
