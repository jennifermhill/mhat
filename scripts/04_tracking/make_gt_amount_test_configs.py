"""Generate held-out-test tracking + eval configs from GT-amount fit results.

For every completed fit under the train dataset, take its learned weights and
drop them into the test-side template (which carries the test dataset's own
seg/flow ids and structural params). Weights are transferred verbatim — nothing
is tuned against test metrics.

A fit that produced an EMPTY solution on the train graph is still carried through to
the test split. "Empty" is a property of the solution those weights imply on a
particular candidate graph, not of the weights themselves, so whether the same weights
are also degenerate on the test graph is a question to answer by solving rather than
to assume. The curve then plots the run's real test score, or marks it empty iff the
TEST solve is empty too. Only a fit that genuinely produced no weights — a crashed fit,
or a missing learned_weights.toml — is skipped.

The dense-crop arm is a *training-annotation* protocol, so nothing is cropped here:
its weights are applied to the whole test volume exactly like every other arm's, and
the crop geometry rides along only as provenance.

The test side is given by the sweep config: ``test_template`` is a tracking
config for the test movie (its own ``raw_path``, ``output_dir``, ``seg_result``,
``flow_result`` and structural parameters), and ``test_gt_tracks`` /
``test_gt_seg`` (``test_ctc_seg_dir``) its GT. Everything is written into the
held-out solves' sweep on the test movie, ``<test output_dir>/tracking/sweeps/
<solves_id>/``: ``track_configs/<token>.toml`` (run ``sweeps/<solves_id>/runs/
<token>``, with ``weights_from`` naming the fit) and ``eval_configs/<token>.toml``
(``eval_name`` = the sweep id). Then::

    python scripts/04_tracking/solve_many_weights.py <solves sweep>/track_configs/*.toml
    python scripts/05_evaluation/run_evals.py --config-dir <solves sweep>/eval_configs
    python scripts/05_evaluation/collect_gt_amount.py <config> --side test

Usage:
    python scripts/04_tracking/make_gt_amount_test_configs.py <gt_amount_config.toml>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import toml

from fit_weights_ssvm import _LEARNED_WEIGHT_TO_TOML
from sweep_ssvm_reg import make_eval_config

from mhat.dataset import Dataset
from mhat.tracking.gt_subsets import (
    check_gt_amount_config,
    iter_sweep_runs,
    sweep_dir,
    sweep_id,
    sweep_run_uid,
    token_regex,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

WEIGHT_KEYS = tuple(_LEARNED_WEIGHT_TO_TOML.values())
PROVENANCE_KEYS = (
    "gt_amount_arm",
    "gt_amount_kind",
    "gt_amount_n_tracks",
    "gt_amount_n_gt_nodes",
    "gt_amount_seed",
    "gt_amount_criterion",
    "gt_amount_crop_fraction",
    "gt_amount_crop_target_gt_nodes",
    "gt_amount_crop_membership",
    "gt_amount_crop_starts",
    "gt_amount_crop_stops",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the GT-amount sweep config (train side)")
    parser.add_argument("--test-template", type=Path, default=None,
                        help="overrides the config's test_template")
    parser.add_argument("--match-threshold", type=int, default=None)
    args = parser.parse_args()

    config = toml.load(args.config)
    check_gt_amount_config(config)
    prefix = config.get("run_name_prefix", "gta")
    token_re = token_regex(prefix)
    if args.match_threshold is not None:
        config = {**config, "eval_match_threshold": args.match_threshold}

    test_template = args.test_template or config.get("test_template")
    if not test_template:
        raise SystemExit("set test_template (a tracking config for the test movie)")
    test_template = Path(test_template)
    if not test_template.is_absolute():
        test_template = REPO_ROOT / test_template
    template = toml.load(test_template)
    test_ds = Dataset.from_config(template)  # refuses old-layout keys
    for key in ("test_gt_tracks", "test_gt_seg"):
        if key not in config:
            raise SystemExit(f"set {key} (the test movie's GT) in the sweep config")

    # Fits are discovered on the train side; the solves they produce live on the
    # test side, as their own sweep.
    fits = sweep_id(config, "fits_id")
    solves = sweep_id(config, "solves_id")
    solves_dir = sweep_dir(test_ds.output_dir, "tracking", solves)
    out_dir, eval_dir = solves_dir / "track_configs", solves_dir / "eval_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    written, skipped, train_empty_tokens = [], [], []
    for fit_dir in iter_sweep_runs(config["output_dir"], "ssvm", fits):
        token = fit_dir.name
        if not token_re.match(token):
            continue
        summary_path = fit_dir / "fit_summary.json"
        train_empty = False
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            status = summary.get("status")
            # `empty_solution` means the fit CONVERGED and wrote weights, but the
            # solution those weights imply on the TRAIN graph was empty. That says
            # nothing certain about the test graph, so solve it and find out.
            if status == "empty_solution":
                train_empty = True
            elif status not in (None, "ok", "not_converged"):
                skipped.append((token, status))
                continue

        learned_path = fit_dir / "learned_weights.toml"
        if not learned_path.is_file():
            skipped.append((token, "no learned_weights.toml"))
            continue

        learned = toml.load(learned_path)
        track_cfg = dict(template)
        for key in WEIGHT_KEYS:
            if key in learned:
                track_cfg[key] = learned[key]
        for key in PROVENANCE_KEYS:
            if key in learned:
                track_cfg[key] = learned[key]
        uid = sweep_run_uid(solves, token)
        track_cfg["exp_uid"] = uid
        # The fit the weights came from (an ssvm run, possibly of another movie).
        track_cfg["weights_from"] = str(fit_dir).replace("\\", "/")
        with open(out_dir / f"{token}.toml", "w") as f:
            f.write(
                f"# Auto-generated by make_gt_amount_test_configs.py from {learned_path}\n"
                "# Weights learned on the TRAIN split are applied to the held-out TEST\n"
                "# split verbatim. Do not tune anything here against test metrics.\n"
            )
            toml.dump(track_cfg, f)

        # Scored against the TEST movie's own GT (and CTC SEG folder): the train
        # values would silently score the wrong annotation, or skip SEG.
        eval_cfg = make_eval_config(
            config, test_ds.output_dir, uid, stage="tracking", eval_name=solves,
            gt_tracks=config["test_gt_tracks"], gt_seg=config["test_gt_seg"],
            ctc_seg_dir=config.get("test_ctc_seg_dir"),
        )
        with open(eval_dir / f"{token}.toml", "w") as f:
            toml.dump(eval_cfg, f)
        written.append(token)
        if train_empty:
            train_empty_tokens.append(token)

    print(f"Wrote {len(written)} tracking configs to {out_dir}")
    print(f"Wrote {len(written)} eval configs to {eval_dir}")
    if train_empty_tokens:
        print(
            f"\n{len(train_empty_tokens)} run(s) were EMPTY on the train graph and are "
            "carried through anyway, to be solved on test:"
        )
        for token in train_empty_tokens:
            print(f"  {token}")
    if skipped:
        print(f"\nSkipped {len(skipped)} run(s) with no usable weights:")
        for token, why in skipped:
            print(f"  {token}: {why}")


if __name__ == "__main__":
    main()
