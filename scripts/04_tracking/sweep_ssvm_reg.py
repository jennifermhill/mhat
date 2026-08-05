"""Sweep `ssvm_reg` at FULL ground truth to calibrate the regularization strength.

Stage 1 of the GT-amount experiment. Fits the full-GT problem once per value of
`ssvm_reg` and scores each on the train split against the complete train GT. The
winner defines an *effective* regularizer

    effective_reg = ssvm_reg * n_labeled_variables

which stage 2 (`run_gt_amount_experiment.py` with `ssvm_reg_effective_target`) then
holds constant while the amount of GT varies.

Why calibrate here rather than reading the optimum off the GT-amount curve: with
`ssvm_reg` fixed, the effective regularizer is `ssvm_reg * n_labeled`, so it drifts
with GT amount and the curve confounds the two. Picking the regularizer from that
same curve would also be circular. Calibrating once, on a fully annotated reference
dataset, is both cleaner and closer to what a user actually does.

Everything here stays on the train split; the held-out test set is untouched.

The graph, GT and candidate<->GT overlaps are built ONCE and reused across the grid,
so the whole sweep is a single job (~4 min setup + ~1 min per value). Do not fan it
out — parallelizing would pay the 4 min setup per value to save seconds.

Usage:
    # fit the grid
    python scripts/04_tracking/sweep_ssvm_reg.py <gt_amount_config.toml>
    # score them
    python scripts/05_evaluation/run_evals.py --config-dir <printed eval dir>
    # read off the winner and the effective-reg target for stage 2
    python scripts/04_tracking/sweep_ssvm_reg.py <gt_amount_config.toml> --collect
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import toml

from fit_weights_ssvm import configure_logging, fit_and_solve_on_graph, load_gt
from mhat.tracking.gt_annotation import assign_gt_labels, compute_gt_overlaps
from mhat.tracking.gt_subsets import copy_track_graph, run_dir, run_ref
from mhat.tracking.pipeline import build_track_graph, resolve_input_dirs

# Even twelfth-decade grid, 0.1 down to 0.00316 — effective reg ~34.4 to ~1087.7 at
# full GT (n_labeled = 10877). It nests the familiar quarter-decade points (0.0562,
# 0.0316, 0.0178, 0.01) and includes the old default 0.1 (effective 1087.7), so it
# stays comparable to earlier work.
#
# The lower half-decade (indices 13-18) was appended after the first pass: TE rose
# monotonically all the way down the original 0.1-0.01 grid and the winner landed at
# 0.0121, one point off the bottom edge, on a plateau (0.8074 / 0.8093 / 0.8074 across
# the last three points). That satisfies the bracketing gate only by a 0.0019 dip at
# the final point, which is too thin to call a turnover. Values are APPENDED, never
# reordered — token index defines the run directory, so inserting anything above index
# 13 would silently repoint regsweep_r00..r12 at different regularizers.
DEFAULT_REGS = [
    0.1, 0.0825, 0.0681, 0.0562, 0.0464, 0.0383, 0.0316,
    0.0261, 0.0215, 0.0178, 0.0147, 0.0121, 0.01,
    0.00825, 0.00681, 0.00562, 0.00464, 0.00383, 0.00316,
]

SWEEP_KEYS = (
    "sizes", "include_full", "seeds", "arms", "mask_overlap_criterion",
    "gt_subsets_dirname", "run_name_prefix", "ssvm_reg_normalize",
    "ssvm_reg_effective_target", "ssvm_reg_sweep",
    # Stage 1 calibrates on the full GT and never crops, but it shares the sweep
    # config with stage 2, so the arm-C keys have to be stripped here as well.
    "crop_fractions", "crop_axes", "crop_membership", "gt_crops_dirname",
    "crop_target_gt_nodes", "crop_growth_step",
    # Stage 1 and stage 2 group their runs separately, so the stage-2 keys must
    # not leak into a regsweep run's own config.
    "runs_subdir", "test_runs_subdir", "regsweep_subdir",
)


def token(index: int, prefix: str) -> str:
    return f"{prefix}_r{index:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--regs", nargs="+", type=float, default=None)
    parser.add_argument("--prefix", default="regsweep")
    parser.add_argument("--collect", action="store_true", help="read results, print the winner")
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()

    config = toml.load(args.config)
    regs = args.regs or config.get("ssvm_reg_sweep") or DEFAULT_REGS
    prefix = args.prefix

    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    experiment = config["experiment"]
    dataset = config["dataset"]
    train_root = output_base_dir / "tracking" / experiment / dataset
    eval_root = output_base_dir / "evaluation" / experiment / dataset
    # Stage 1 groups its runs separately from stage 2 — they share this config but
    # are different sweeps over the same dataset.
    subdir = config.get("regsweep_subdir", "")
    eval_cfg_dir = (
        Path(__file__).resolve().parents[2]
        / "configs/tracking" / experiment / dataset / (subdir or "gt_amount") / "eval"
    )

    if args.collect:
        rows = []
        for i, reg in enumerate(regs):
            tok = token(i, prefix)
            summary_path = run_dir(train_root, subdir, tok) / "fit_summary.json"
            metrics_path = run_dir(eval_root, subdir, tok) / "track_metrics.json"
            row = {"token": tok, "ssvm_reg": reg, "status": "missing"}
            if summary_path.is_file():
                s = json.loads(summary_path.read_text())
                row["status"] = s.get("status", "ok")
                row["n_labeled"] = s.get("n_labeled_variables")
                row["n_solution_nodes"] = s.get("n_solution_nodes")
            if metrics_path.is_file():
                m = json.loads(metrics_path.read_text())
                row["target_effectiveness"] = m.get("TrackOverlapMetrics", {}).get(
                    "target_effectiveness"
                )
                row["track_purity"] = m.get("TrackOverlapMetrics", {}).get("track_purity")
            rows.append(row)

        def cell(value, width, fmt=""):
            text = "-" if value is None else format(value, fmt)
            return f"{text:>{width}}"

        print(
            f"{'token':<16}{'ssvm_reg':>10}{'n_labeled':>11}{'sol_nodes':>11}"
            f"{'TE':>10}{'purity':>9}  status"
        )
        for r in rows:
            print(
                f"{r['token']:<16}{cell(r['ssvm_reg'], 10, '.4f')}"
                f"{cell(r.get('n_labeled'), 11, 'd' if r.get('n_labeled') is not None else '')}"
                f"{cell(r.get('n_solution_nodes'), 11, 'd' if r.get('n_solution_nodes') is not None else '')}"
                f"{cell(r.get('target_effectiveness'), 10, '.4f' if r.get('target_effectiveness') is not None else '')}"
                f"{cell(r.get('track_purity'), 9, '.4f' if r.get('track_purity') is not None else '')}"
                f"  {r['status']}"
            )

        scored = [r for r in rows if r.get("target_effectiveness") is not None]
        if not scored:
            print("\nNo scored runs yet — run run_evals.py on the eval configs first.")
            return
        best = max(scored, key=lambda r: r["target_effectiveness"])
        target = best["ssvm_reg"] * best["n_labeled"]
        print(f"\nBest: {best['token']}  ssvm_reg={best['ssvm_reg']}  TE={best['target_effectiveness']:.4f}")
        print(f"n_labeled at full GT = {best['n_labeled']}")
        print(f"\n  ssvm_reg_effective_target = {target:.2f}")
        print("\nPut that in the stage-2 config. If the winner sits at an END of the grid,")
        print("extend the grid in that direction before trusting it.")
        if best is scored[0] or best is scored[-1]:
            print("  ^^ WARNING: winner is at a grid edge; the optimum is not bracketed.")
        summary_out = train_root / subdir if subdir else train_root
        summary_out.mkdir(parents=True, exist_ok=True)
        with open(summary_out / f"{prefix}_summary.json", "w") as f:
            json.dump({"rows": rows, "best": best, "effective_target": target}, f, indent=2)
        return

    raw_dir, seg_dir, flow_dirs, gt_data_dir = resolve_input_dirs(config)

    print("Building candidate graph (once)...")
    base_graph, fragments, merge_history, exclusion_sets, scale, axes = build_track_graph(
        config, raw_dir, seg_dir, flow_dirs
    )
    no_merges = len(merge_history) == 0
    print(f"  {len(base_graph.nodes)} nodes, {len(base_graph.edges)} edges")

    print("Loading GT (once)...")
    gt_graph, gt_seg = load_gt(gt_data_dir, scale)
    print("Computing candidate<->GT overlaps (once)...")
    overlaps = compute_gt_overlaps(base_graph, fragments, gt_graph, gt_seg, merge_history)

    eval_cfg_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{len(regs)} regularizer value(s); eval configs -> {eval_cfg_dir}\n")

    for i, reg in enumerate(regs):
        tok = token(i, prefix)
        if args.only and tok not in set(args.only):
            continue
        print(f"=== {tok}  ssvm_reg={reg} ===")

        graph = copy_track_graph(base_graph)
        # Full GT: keep_gt_labels=None -> nothing is unlabeled, nothing is removed.
        stats, masked = assign_gt_labels(
            graph, overlaps, gt_graph, iogt_threshold=config.get("iogt_threshold", 0.5)
        )
        assert not masked, "full-GT annotation should mask nothing"
        n_labeled = sum(1 for d in graph.nodes.values() if d.get("gt_selected") is not None)
        n_labeled += sum(1 for d in graph.edges.values() if d.get("gt_selected") is not None)

        out_dir = run_dir(train_root, subdir, tok)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_config = {k: v for k, v in config.items() if k not in SWEEP_KEYS}
        run_config["ssvm_reg"] = float(reg)
        run_config["output_name"] = tok
        run_config["exp_uid"] = run_ref(subdir, tok)
        with open(out_dir / "config.toml", "w") as f:
            toml.dump(run_config, f)

        configure_logging(out_dir)
        row = {
            "token": tok,
            "ssvm_reg": float(reg),
            "n_labeled_variables": n_labeled,
            "effective_reg": float(reg) * n_labeled,
            **stats,
        }
        try:
            summary = fit_and_solve_on_graph(
                run_config, graph, exclusion_sets, fragments, merge_history,
                scale, axes, out_dir, no_merges=no_merges,
            )
            row.update(summary)
            row["status"] = "empty_solution" if summary.get("empty_solution") else "ok"
        except Exception:
            row["status"] = "fit_failed"
            row["traceback"] = traceback.format_exc()
            print(f"  {tok} FAILED:\n{row['traceback']}")
        with open(out_dir / "fit_summary.json", "w") as f:
            json.dump(row, f, indent=2)

        # Score against the COMPLETE train GT — legitimate here because this stage is
        # the fully-annotated condition. No gt_data_dir override needed.
        eval_cfg = {
            "input_base_dir": str(input_base_dir).replace("\\", "/") + "/",
            "output_base_dir": str(output_base_dir).replace("\\", "/") + "/",
            "experiment": experiment,
            "dataset": dataset,
            "track_result": run_ref(subdir, tok),
            "metrics": ["basic", "track_overlap"],
            "matcher": "point",
            "match_threshold": 10,
        }
        with open(eval_cfg_dir / f"{tok}_eval.toml", "w") as f:
            f.write(f"# ssvm_reg = {reg}; scored against the COMPLETE train GT.\n")
            toml.dump(eval_cfg, f)

    print(f"\nNow run:\n  python scripts/05_evaluation/run_evals.py --config-dir {eval_cfg_dir}")
    print(f"  python scripts/04_tracking/sweep_ssvm_reg.py {args.config} --collect")


if __name__ == "__main__":
    main()
