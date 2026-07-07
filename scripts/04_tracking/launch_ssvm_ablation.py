"""Submit the 13 SSVM-fit seg-ablation-parity runs (MDA231 01_cells) on LSF.

For each condition this submits:
  1. a fit job  -> fit_weights_ssvm.py <fit_cfg>   (fits weights + final solve,
     writing pred_tracks.zarr to experiments/tracking/.../<output_name>/)
  2. a dependent eval job -> evaluate_tracks.py <eval_cfg>, gated on ended(fit)
     so it fires even if the fit exits 120 in the shutdown phase (a known LSF
     post-completion exit-code quirk; outputs are already written by then).

Fit configs:  configs/tracking/Fluo-C3DL-MDA231/01_cells/ssvm_fit/<name>.toml
Eval configs: configs/tracking/Fluo-C3DL-MDA231/01_cells/ssvm_fit/eval/<name>_eval.toml
(both generated for the 13 conditions; see the plan / CLAUDE.md SSVM section.)

Usage (run from the repo root on the cluster):
    conda run -n mhat2 python scripts/04_tracking/launch_ssvm_ablation.py [--dry-run] [--only NAME ...]

--dry-run prints the bsub commands without submitting. Adjust the LSF constants
below (queue, cores, walltime, memory) to your allocation before submitting.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = "configs/tracking/Fluo-C3DL-MDA231/01_cells/ssvm_fit"
LOG_DIR = os.path.join(
    REPO, "experiments", "tracking", "Fluo-C3DL-MDA231", "01_cells", "ssvm_fit_logs"
)

# The 13 unique conditions (== output_name == tracking_uid == eval track_result).
CONDITIONS = [
    "ssvm_full",
    "ssvm_no_intensity",
    "ssvm_no_volume",
    "ssvm_no_drift",
    "ssvm_no_cohesion",
    "ssvm_no_all",
    "ssvm_plus_cohesion",
    "ssvm_plus_intensity",
    "ssvm_plus_volume",
    "ssvm_plus_curvature",
    "ssvm_plus_drift",
    "ssvm_no_affinities",
    "ssvm_no_merges",
]

# ---- LSF / bsub knobs (edit to match your allocation) -----------------------
FIT_CORES = "8"
FIT_WALLTIME = "24:00"
EVAL_CORES = "2"
EVAL_WALLTIME = "1:00"
# Optional extra bsub args, e.g. ["-q", "gpu_short"] or memory reservations.
EXTRA_BSUB = []

FIT_INNER = (
    "module load gurobi && conda run -n mhat2 --no-capture-output "
    "python -u scripts/04_tracking/fit_weights_ssvm.py {cfg}"
)
EVAL_INNER = (
    "conda run -n mhat2 --no-capture-output "
    "python -u scripts/05_evaluation/evaluate_tracks.py {cfg}"
)

JOB_ID_RE = re.compile(r"Job <(\d+)>")


def bsub(job_name, cores, walltime, log_path, inner, dep_job=None, dry_run=False):
    cmd = [
        "bsub",
        "-J", job_name,
        "-n", cores,
        "-W", walltime,
        "-o", log_path,
        *EXTRA_BSUB,
    ]
    if dep_job is not None:
        cmd += ["-w", f"ended({dep_job})"]
    cmd += [inner]
    if dry_run:
        print("  " + " ".join(repr(c) if " " in c else c for c in cmd))
        return None
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    m = JOB_ID_RE.search(out.stdout)
    if not m:
        raise RuntimeError(f"could not parse job id from bsub output for {job_name!r}")
    return m.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print bsub commands, do not submit")
    parser.add_argument("--only", nargs="+", default=None, help="subset of condition names to submit")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    conditions = args.only or CONDITIONS
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}\nvalid: {CONDITIONS}")

    for name in conditions:
        fit_cfg = f"{CONFIG_DIR}/{name}.toml"
        eval_cfg = f"{CONFIG_DIR}/eval/{name}_eval.toml"
        for rel in (fit_cfg, eval_cfg):
            if not os.path.isfile(os.path.join(REPO, rel)):
                raise SystemExit(f"missing config: {rel}")

        print(f"[{name}]")
        fit_job = bsub(
            f"ssvm_fit_{name}", FIT_CORES, FIT_WALLTIME,
            os.path.join(LOG_DIR, f"fit_{name}.%J.log"),
            FIT_INNER.format(cfg=fit_cfg), dry_run=args.dry_run,
        )
        bsub(
            f"ssvm_eval_{name}", EVAL_CORES, EVAL_WALLTIME,
            os.path.join(LOG_DIR, f"eval_{name}.%J.log"),
            EVAL_INNER.format(cfg=eval_cfg), dep_job=fit_job, dry_run=args.dry_run,
        )

    print(f"\nSubmitted {len(conditions)} fit + {len(conditions)} eval jobs."
          if not args.dry_run else f"\n(dry run) {len(conditions)} conditions.")


if __name__ == "__main__":
    main()
