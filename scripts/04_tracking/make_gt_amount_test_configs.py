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

Usage:
    python scripts/04_tracking/make_gt_amount_test_configs.py \
        configs/tracking/NC281-sparse-label/02_nuclei_denoised_train/gt_amount/gt_amount.toml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import toml

from fit_weights_ssvm import _LEARNED_WEIGHT_TO_TOML
from sweep_ssvm_reg import make_eval_config
from mhat.tracking.gt_subsets import iter_run_dirs, run_ref, token_regex

REPO_ROOT = Path(__file__).resolve().parents[2]
# Fallbacks only. The dataset and template belong to the sweep, so they are read
# from the sweep config (`test_dataset` / `test_template`) when it carries them;
# these keep the original NC281 invocation working unchanged.
DEFAULT_TEST_DATASET = "02_nuclei_denoised_test"
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "configs/tracking/NC281-sparse-label/02_nuclei_denoised_test/ssvm_test_from_train_config.toml"
)

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
    parser.add_argument("--test-template", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--test-dataset", default=None)
    parser.add_argument("--match-threshold", type=int, default=None)
    args = parser.parse_args()

    config = toml.load(args.config)
    prefix = config.get("run_name_prefix", "gta")
    token_re = token_regex(prefix)

    # CLI overrides config overrides the NC281 fallback.
    test_dataset = args.test_dataset or config.get("test_dataset") or DEFAULT_TEST_DATASET
    test_template = args.test_template or config.get("test_template") or DEFAULT_TEMPLATE
    test_template = Path(test_template)
    if not test_template.is_absolute():
        test_template = REPO_ROOT / test_template
    if args.match_threshold is not None:
        config = {**config, "eval_match_threshold": args.match_threshold}

    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    experiment = config["experiment"]
    train_root = input_base_dir / "tracking" / experiment / config["dataset"]

    # Fits are discovered on the train side; the solves they produce live on the
    # test side, which is grouped independently.
    runs_subdir = config.get("runs_subdir", "")
    test_runs_subdir = config.get("test_runs_subdir", "")

    template = toml.load(test_template)
    assert template["dataset"] == test_dataset, (
        f"template targets {template['dataset']!r}, expected {test_dataset!r}"
    )

    # Config layout mirrors the experiments layout.
    out_dir = args.out_dir or (
        REPO_ROOT / "configs/tracking" / experiment / test_dataset
        / (test_runs_subdir or "gt_amount")
    )
    eval_dir = out_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    written, skipped, train_empty_tokens = [], [], []
    for fit_dir in iter_run_dirs(train_root, runs_subdir):
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
        track_cfg["exp_uid"] = run_ref(test_runs_subdir, token)
        with open(out_dir / f"{token}.toml", "w") as f:
            f.write(
                f"# Auto-generated by make_gt_amount_test_configs.py from {learned_path}\n"
                "# Weights learned on the TRAIN split are applied to the held-out TEST\n"
                "# split verbatim. Do not tune anything here against test metrics.\n"
            )
            toml.dump(track_cfg, f)

        # The TEST split has its own CTC GT folder, so `ctc_gt` is overridden here
        # rather than inherited: the sweep config's value names the TRAIN folder,
        # and pointing at it would silently skip SEG on every generated run.
        eval_cfg = make_eval_config(
            {**config, "ctc_gt": config.get("test_ctc_gt", config.get("ctc_gt"))},
            input_base_dir,
            output_base_dir,
            experiment,
            test_dataset,
            run_ref(test_runs_subdir, token),
        )
        with open(eval_dir / f"{token}_eval.toml", "w") as f:
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
