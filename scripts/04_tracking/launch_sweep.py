"""Generate and submit a coordinate-wise parameter sweep to LSF.

A sweep is described by a single spec TOML (see ``scratch_configs/sweeps/``). The spec
holds a ``sweep_id``, one ``[base]`` tracking config (with ``raw_path``,
``output_dir``, ``seg_result``, ...), one ``[eval]`` stub (GT paths, ``metrics``,
``matcher``), optional ``[conditions.<name>]`` tables, and a list of ``[[runs]]``,
each naming a ``label``, the ``condition`` it belongs to, and a small
``[runs.overrides]`` table. Parameters are layered ``base`` ->
``conditions[condition]`` -> ``overrides``, so a condition states once which costs
it zeroes and each run only states the one or two values it is sweeping.

The sweep lives in ``<output_dir>/tracking/sweeps/<sweep_id>/``, and so do its
runs: run ``<label>`` is the tracking run ``sweeps/<sweep_id>/runs/<label>``. For
each run this script:

  1. writes ``<sweep>/track_configs/<label>.toml`` = the layered config, with
     that ``exp_uid``,
  2. writes ``<sweep>/eval_configs/<label>.toml`` = eval stub plus
     ``output_dir``, ``track_result`` (that run) and ``eval_name = <sweep_id>``,
     which is how ``collect_sweep.py`` finds the eval inside the run,
  3. submits the tracking job, then the eval job gated on ``ended(<track job>)``
     -- ``ended()`` rather than ``done()`` because tracking sometimes exits 120
     during interpreter shutdown *after* writing valid output,
  4. records everything in ``<sweep>/manifest.toml``.

Each launch also keeps the spec exactly as launched, in
``<sweep>/specs/<timestamp>.toml`` (one copy per launch, never overwritten), and
each manifest run names the copy it was launched from (``spec_copy``). So the
spec under ``scratch_configs/`` is only a draft: editing or deleting it later
loses nothing, and nothing needs moving into the output tree.

Everything (configs, logs, manifest) is written under the shared filesystem, not
node-local scratch, so the compute nodes can read it.

``stage = "ssvm"`` in the spec makes it a sweep of SSVM weight fits instead:
each job runs ``fit_weights_ssvm.py`` (``[base]`` then also needs ``gt_tracks`` /
``gt_seg``), runs land in ``<output_dir>/ssvm/sweeps/<sweep_id>/runs/<label>``,
and each eval scores the fit's solution (``track_stage = "ssvm"``). An SSVM
ablation is one such sweep: each condition is a ``[[runs]]`` entry, a feature is
left out of a fit with its ``ablate_*`` flag (``ablate_drift``, ``ablate_area``,
``ablate_intensity``, ``ablate_curvature``, ``ablate_cohesion_adhesion``; that is
the one thing those flags still do), and a different segmentation (``- Merges``,
``- Affinities``) is a ``seg_result`` override::

    sweep_id = "ssvm_ablation"
    stage = "ssvm"

    [lsf]
    track_slots = 8          # the fit job
    track_walltime = "24:00"
    eval_slots = 2
    eval_walltime = "1:00"

    [base]                   # a complete fit config, minus exp_uid
    raw_path = ".../data/Fluo-C3DL-MDA231/01_cells.zarr"
    output_dir = ".../experiments/Fluo-C3DL-MDA231/01_cells"
    seg_result = "fs1cpm6_seg_base_wzfix"
    flow_result = "2026-09-22_14-11-08"
    gt_tracks = ".../Fluo-C3DL-MDA231/01_cells/gt/ctc/correct_tracks.zarr"
    gt_seg = ".../Fluo-C3DL-MDA231/01_cells/gt/ctc/correct_seg.zarr"
    ssvm_reg = 0.1
    # ... the remaining tracking keys

    [eval]
    gt_tracks = ".../gt/ctc/correct_tracks.zarr"
    gt_seg = ".../gt/ctc/correct_seg.zarr"
    ctc_seg_dir = ".../gt/ctc/01_GT/SEG"
    metrics = ["ctc", "f1_aogm", "track_overlap"]
    matcher = "ctc"

    [[runs]]
    label = "ssvm_full"
    [[runs]]
    label = "ssvm_no_drift"
    [runs.overrides]
    ablate_drift = true
    [[runs]]
    label = "ssvm_no_merges"
    [runs.overrides]
    seg_result = "fs1cpm6_seg_nomrg_wzfix"

No run is ever redone in place. Before submitting anything the launcher refuses
if a selected label's run directory already exists (a crashed run leaves one,
since tracking writes its config first). ``--retry`` gives those labels a new run
name, ``runs/<label>_r2`` (``_r3``, ...), recorded in the manifest; the old
directory is left for you to delete. A finished run whose eval failed is
re-evaluated with ``launch_evals.py`` instead.

Usage, from the repo root on the cluster::

    conda run -n mhat-cluster python scripts/04_tracking/launch_sweep.py \\
        scratch_configs/sweeps/mda231_01cells_stageA.toml [--dry-run] [--only LABEL ...] \
        [--dep "ended(<job>) && ended(<job>)"]

``--dep`` makes every tracking job wait on an LSF dependency expression, which is
how a sweep chains onto segmentation jobs submitted just before it.

``--sandbox WRAPPER`` runs every job's command inside a bwrap sandbox on the
compute node via ``scripts/cluster_watcher/job_sandbox.sh``; the sweep watcher
(``scripts/cluster_watcher/sweep_watcher.py``) calls :func:`launch` with the same
option when it submits on behalf of a sandboxed Claude agent.

Read back the results with ``scripts/05_evaluation/collect_sweep.py``.
"""

from __future__ import annotations

import argparse
import copy
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import toml

REPO = Path(__file__).resolve().parents[2]

JOB_ID_RE = re.compile(r"Job <(\d+)>")

# Structural / IO parameters that a sweep must never vary: changing them makes
# runs incomparable (or violates the "do not tune" rule in CLAUDE.md).
PROTECTED_KEYS = {
    "raw_path",
    "output_dir",
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
    "division_weight",
    "graph_cache",
    # SSVM fits: the regularizer, and the features a fit leaves out.
    "ssvm_reg", "ssvm_max_iter", "ssvm_eps", "iogt_threshold",
    "ablate_drift", "ablate_area", "ablate_intensity", "ablate_curvature",
    "ablate_cohesion_adhesion",
}

# Keys of the old results layout. A spec still carrying them was written for it.
OLD_LAYOUT_KEYS = (
    "input_base_dir", "output_base_dir", "raw_base_dir", "experiment", "dataset",
    "graph_cache_dir", "gt_data_dir", "ctc_gt",
)
# Eval keys the launcher fills in for each run.
FILLED_EVAL_KEYS = ("output_dir", "track_result", "track_stage", "eval_name")
RUN_STAGE = "tracking"
SPEC_STAGES = ("tracking", "ssvm")

TRACK_INNER = (
    "export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} "
    "OPENBLAS_NUM_THREADS={threads} && "
    "module load gurobi && conda run -n {env} --no-capture-output "
    "python -u scripts/04_tracking/run_tracking.py {cfg}"
)
FIT_INNER = (
    "export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} "
    "OPENBLAS_NUM_THREADS={threads} && "
    "module load gurobi && conda run -n {env} --no-capture-output "
    "python -u scripts/04_tracking/fit_weights_ssvm.py {cfg}"
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


def check_no_old_keys(table, where):
    old = [key for key in OLD_LAYOUT_KEYS if key in table]
    if old:
        raise SystemExit(
            f"{where} uses keys of the old results layout: {old}. Specs now give "
            "raw_path / output_dir in [base] and GT paths in [eval]; the dataset is "
            "named by output_dir (see this script's docstring)."
        )


def validate_spec(spec, allow_protected=False):
    if "sweep_id" not in spec:
        raise SystemExit("spec is missing required key 'sweep_id'")
    check_no_old_keys(spec, "the spec")
    base = spec.get("base")
    if not base:
        raise SystemExit("spec is missing the [base] tracking config")
    check_no_old_keys(base, "[base]")
    stage = spec.get("stage", RUN_STAGE)
    if stage not in SPEC_STAGES:
        raise SystemExit(f"stage must be one of {SPEC_STAGES}, got {stage!r}")
    required = ("raw_path", "output_dir", "seg_result")
    if stage == "ssvm":
        required += ("gt_tracks", "gt_seg")
    for key in required:
        if key not in base:
            raise SystemExit(f"[base] is missing {key!r}")
    if "exp_uid" in base:
        raise SystemExit("[base] must not set exp_uid -- it is generated per run")
    check_no_windows_paths(base, "[base]")

    eval_stub = spec.get("eval")
    if not eval_stub:
        raise SystemExit("spec is missing the [eval] config stub")
    check_no_old_keys(eval_stub, "[eval]")
    filled = [key for key in FILLED_EVAL_KEYS if key in eval_stub]
    if filled:
        raise SystemExit(f"[eval] must not set {filled}: the launcher fills them per run")
    if "gt_tracks" not in eval_stub:
        raise SystemExit("[eval] is missing gt_tracks")
    check_no_windows_paths(eval_stub, "[eval]")
    # Omitting `matcher` silently falls back to PointMatcher and produces wrong
    # (lower) CTC numbers with no error -- see the phantom-regression note in
    # CLAUDE.md. Require it explicitly.
    # The linajea metric does its own edge-level Hungarian matching and never
    # touches a traccuracy matcher, so a linajea-only eval needs no `matcher`.
    linajea_only = eval_stub.get("metrics") == ["linajea"]
    if "matcher" not in eval_stub and not linajea_only:
        raise SystemExit("[eval] must set `matcher` explicitly (e.g. matcher = \"ctc\")")
    if eval_stub.get("matcher") == "ctc":
        for key in ("match_threshold", "threshold"):
            if key in eval_stub:
                raise SystemExit(f"[eval] must not set {key!r} with the ctc matcher")
    elif eval_stub.get("matcher") == "point" and not (
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


def config_path_for_cmd(path, repo=REPO):
    """Path to hand the inner command, which runs with ``cwd=repo``.

    Sweep artifacts live under ``output_dir``, which is only inside the repo
    when the checkout and the data tree are the same directory. In a worktree
    (repo ``mhat-cluster``, data ``mhat``) they are siblings, so fall back to the
    absolute path rather than failing.
    """
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def run_uid(sweep_id, run_name):
    return f"sweeps/{sweep_id}/runs/{run_name}"


def run_dir_taken(output_dir, uid, stage=RUN_STAGE):
    """Whether a run directory exists and holds anything (cf. Dataset.check_new_run)."""
    run_dir = Path(output_dir) / stage / uid
    return run_dir.is_dir() and any(run_dir.iterdir())


def assign_uids(spec, selected_labels, previous, retry, stage=RUN_STAGE):
    """Each label's run uid: the one the manifest already records, else
    ``sweeps/<sweep_id>/runs/<label>``. A selected label whose run directory is
    taken is refused, or with ``retry`` moved on to the next free ``<label>_rN``.
    """
    output_dir, sweep_id = spec["base"]["output_dir"], spec["sweep_id"]
    uids, taken = {}, []
    for run in spec["runs"]:
        label = run["label"]
        uid = previous.get(label) or run_uid(sweep_id, label)
        if label in selected_labels and run_dir_taken(output_dir, uid, stage):
            if not retry:
                taken.append(uid)
            else:
                n = 2
                while run_dir_taken(output_dir, run_uid(sweep_id, f"{label}_r{n}"), stage):
                    n += 1
                uid = run_uid(sweep_id, f"{label}_r{n}")
                print(f"[{label}] earlier run directory exists; retrying as {uid}")
        uids[label] = uid
    if taken:
        raise SystemExit(
            f"{len(taken)} run directories already exist (crashed or finished runs), "
            f"e.g. {taken[:3]}. Pass --retry to give those labels new run names, or "
            "re-evaluate finished runs with launch_evals.py."
        )
    return uids


def build_configs(spec, sweep_dir, uids, repo=REPO):
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
        exp_uid = uids[label]

        track_config = copy.deepcopy(base)
        track_config.update(conditions.get(condition, {}))
        track_config.update(overrides)
        track_config["exp_uid"] = exp_uid
        track_path = track_dir / f"{label}.toml"
        with open(track_path, "w") as handle:
            toml.dump(track_config, handle)

        eval_config = copy.deepcopy(eval_stub)
        eval_config["output_dir"] = base["output_dir"]
        eval_config["track_result"] = exp_uid
        eval_config["eval_name"] = sweep_id
        if spec.get("stage", RUN_STAGE) != RUN_STAGE:
            eval_config["track_stage"] = spec["stage"]
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
                "track_config": config_path_for_cmd(track_path, repo),
                "eval_config": config_path_for_cmd(eval_path, repo),
            }
        )
    return records


def read_spec(spec_path):
    """The spec's bytes and its parsed contents, from a single read, so the copy
    kept in the sweep is exactly the spec that was launched."""
    raw = Path(spec_path).read_bytes()
    return raw, toml.loads(raw.decode("utf-8"))


def save_spec_copy(raw_spec, sweep_dir):
    """Keep the spec as launched, byte for byte, in ``<sweep_dir>/specs/``.

    Named ``<timestamp>.toml``, one copy per launch, so a sweep launched in rounds
    (``--only``, ``--retry``) keeps every version. Never overwrites. Returns the
    path relative to the sweep, which the manifest records per run as
    ``spec_copy``.
    """
    spec_dir = Path(sweep_dir) / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name, n = f"{stamp}.toml", 2
    while True:
        try:
            with open(spec_dir / name, "xb") as handle:
                handle.write(raw_spec)
            return f"specs/{name}"
        except FileExistsError:
            name, n = f"{stamp}_{n}.toml", n + 1


def bsub(
    job_name, slots, walltime, queue, log_stem, inner,
    dep_job=None, dry_run=False, gpu=None, raw_dep=False,
    repo=REPO, wrap=None,
):
    """Submit one job; return its id (None on a dry run).

    ``gpu`` is passed straight to ``-gpu`` (e.g. ``"num=1"``) for the queues that
    need one. ``raw_dep`` treats ``dep_job`` as a complete LSF dependency
    expression instead of a single job id, which is how a job waits on a whole
    fan-out ("ended(1) && ended(2) && ...").

    ``repo`` is the directory the job starts in (LSF keeps the submission cwd),
    which the inner commands rely on for their relative script paths. ``wrap`` is
    an optional sandbox wrapper script: the inner command is handed to it as one
    quoted argument, so ``bash <wrap> '<inner>'`` is what LSF runs, and the
    wrapper is what confines the job (see ``scripts/cluster_watcher/job_sandbox.sh``).
    """
    if wrap:
        inner = f"bash {shlex.quote(str(wrap))} {shlex.quote(inner)}"
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
    if gpu:
        cmd += ["-gpu", gpu]
    if dep_job is not None:
        cmd += ["-w", dep_job if raw_dep else f"ended({dep_job})"]
    cmd += [inner]

    if dry_run:
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return None

    out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    match = JOB_ID_RE.search(out.stdout)
    if not match:
        raise RuntimeError(f"could not parse job id from bsub output for {job_name!r}")
    return match.group(1)


def sweep_dir_for(spec):
    """Where a spec's configs, logs, manifest and runs land (no side effects)."""
    stage = spec.get("stage", RUN_STAGE)
    return Path(spec["base"]["output_dir"]) / stage / "sweeps" / spec["sweep_id"]


def launch(
    spec_path, *, only=None, allow_protected=False, dep=None, dry_run=False,
    repo=REPO, wrap=None, retry=False,
):
    """Validate a spec, write its configs, submit its jobs, write its manifest.

    This is ``main()`` without the argument parsing, so the sweep watcher can
    call it after applying its own policy checks. Returns a dict with
    ``sweep_id``, ``sweep_dir``, ``manifest`` (``None`` on a dry run) and
    ``records`` (the submitted runs, each carrying ``track_job_id`` /
    ``eval_job_id``). Validation failures raise ``SystemExit`` with the reason,
    exactly as the command line does.
    """
    raw_spec, spec = read_spec(spec_path)
    validate_spec(spec, allow_protected=allow_protected)

    lsf = spec.get("lsf", {})
    env = lsf.get("env", "mhat-cluster")
    queue = lsf.get("queue", "local")
    track_slots = lsf.get("track_slots", 2)
    track_walltime = lsf.get("track_walltime", "1:00")
    eval_slots = lsf.get("eval_slots", 1)
    eval_walltime = lsf.get("eval_walltime", "0:30")

    sweep_dir = sweep_dir_for(spec)
    manifest_path = sweep_dir / "manifest.toml"
    previous_runs = (
        {run["label"]: run for run in toml.load(manifest_path).get("runs", [])}
        if manifest_path.is_file() else {}
    )
    labels = [run["label"] for run in spec["runs"]]
    if only:
        unknown = [label for label in only if label not in labels]
        if unknown:
            raise SystemExit(f"unknown run labels: {unknown}")
    selected_labels = set(only) if only else set(labels)
    stage = spec.get("stage", RUN_STAGE)
    uids = assign_uids(
        spec, selected_labels,
        {label: run.get("exp_uid") for label, run in previous_runs.items()}, retry,
        stage=stage,
    )
    run_inner = FIT_INNER if stage == "ssvm" else TRACK_INNER

    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    records = build_configs(spec, sweep_dir, uids, repo)
    print(f"Wrote {len(records)} track + eval configs to {sweep_dir}")
    selected = [record for record in records if record["label"] in selected_labels]
    if not dry_run:
        spec_copy = save_spec_copy(raw_spec, sweep_dir)
        print(f"Kept the spec as {sweep_dir / spec_copy}")
        for record in selected:
            record["spec_copy"] = spec_copy

    for record in selected:
        label = record["label"]
        print(f"[{label}] {record['condition']} {record['overrides']}")
        track_job = bsub(
            f"{spec['sweep_id']}_trk_{label}",
            track_slots, track_walltime, queue,
            str(log_dir / f"track_{label}"),
            run_inner.format(threads=track_slots, env=env, cfg=record["track_config"]),
            dep_job=dep, raw_dep=True, dry_run=dry_run, repo=repo, wrap=wrap,
        )
        eval_job = bsub(
            f"{spec['sweep_id']}_evl_{label}",
            eval_slots, eval_walltime, queue,
            str(log_dir / f"eval_{label}"),
            EVAL_INNER.format(threads=eval_slots, env=env, cfg=record["eval_config"]),
            dep_job=track_job, dry_run=dry_run, repo=repo, wrap=wrap,
        )
        record["track_job_id"] = track_job or ""
        record["eval_job_id"] = eval_job or ""

    if not dry_run:
        # Preserve job ids and spec copies from earlier partial submissions
        # (repeated --only), unless the label was retried under a new run name since.
        for record in records:
            old = previous_runs.get(record["label"], {})
            if old.get("exp_uid") == record["exp_uid"]:
                for key in ("track_job_id", "eval_job_id", "spec_copy"):
                    if not record.get(key) and old.get(key):
                        record[key] = old[key]
            for key in ("track_job_id", "eval_job_id", "spec_copy"):
                record.setdefault(key, "")

        eval_stub = spec["eval"]
        manifest = {
            "sweep_id": spec["sweep_id"],
            "output_dir": spec["base"]["output_dir"],
            "stage": stage,
            # How collect_sweep.py finds each run's eval: <eval_name>_<metrics>_<time>.
            "eval_name": spec["sweep_id"],
            "metrics": eval_stub.get("metrics", ["basic", "track_overlap"]),
            "gt": {key: eval_stub[key] for key in ("gt_tracks", "gt_seg", "ctc_seg_dir")
                   if key in eval_stub},
            "spec": str(Path(spec_path).resolve()),
            "queue": queue,
            "track_slots": track_slots,
            "track_walltime": track_walltime,
            "eval_slots": eval_slots,
            "eval_walltime": eval_walltime,
            "runs": records,
        }
        with open(manifest_path, "w") as handle:
            toml.dump(manifest, handle)
        print(f"\nSubmitted {len(selected)} tracking + {len(selected)} eval jobs.")
        print(f"Manifest: {manifest_path}")
    else:
        print(f"\n(dry run) {len(selected)} runs; no jobs submitted, no manifest written.")

    return {
        "sweep_id": spec["sweep_id"],
        "sweep_dir": sweep_dir,
        "manifest": None if dry_run else manifest_path,
        "records": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", help="path to the sweep spec TOML")
    parser.add_argument("--dry-run", action="store_true", help="write configs and print bsub commands, do not submit")
    parser.add_argument("--only", nargs="+", default=None, help="subset of run labels to submit")
    parser.add_argument("--allow-protected", action="store_true", help="permit overriding structural keys")
    parser.add_argument(
        "--retry", action="store_true",
        help="give selected labels whose run directory already exists a new run name "
        "(runs/<label>_r2, ...) instead of refusing",
    )
    parser.add_argument(
        "--dep", default=None,
        help="LSF dependency expression every tracking job waits on, e.g. "
        "'ended(123) && ended(456)' to gate a sweep on segmentation jobs",
    )
    parser.add_argument(
        "--sandbox", default=None, metavar="WRAPPER",
        help="run every job inside this bwrap wrapper script on the compute node "
        "(scripts/cluster_watcher/job_sandbox.sh)",
    )
    args = parser.parse_args()

    launch(
        args.spec, only=args.only, allow_protected=args.allow_protected,
        dep=args.dep, dry_run=args.dry_run, wrap=args.sandbox, retry=args.retry,
    )


if __name__ == "__main__":
    main()
