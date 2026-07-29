# Ablation Experiments

Branch-specific instructions for running ablation experiments on the `seg_ablation` branch.

> **Branch note**: The `skip_merges` / `scoring_function = "symmetric"` segmentation overrides only exist on `seg_ablation` and its descendants (added in commit `9271610`). Running these experiments from `main` will fail.

> **Superseded 2026-07-29 — the `ablate_*` flags are gone.** Commit `9ff60e6` removed them; `add_costs` now skips a cost iff its **weight and constant are both 0**. Every `ablate_cohesion_adhesion = true` in this file and in the archived configs is **dead** — a run carrying it alongside `cohesion_weight = 2000` actually solved with cohesion fully active. Zero the weight *and* the constant instead. This bit the MDA231 `- Affinities` condition, which ran with coh/adh on while `- Coh/Adh` and `- Merges` had them off.

> **Path note 2026-07-29**: the figure TOMLs moved to `configs/evaluation/` and the plotting scripts to `scripts/07_plotting/`; `build_config_archive.py` is at `scripts/`. Paths below that say `scripts/05_evaluation/` are stale.

---

## Merge Ablation

The merge ablation experiment compares four conditions to isolate the contributions of cohesion/adhesion costs, affinity-aware fragment scoring, and the merge hyperedges themselves.

### The four conditions

| Condition | Segmentation | `seg_config.toml` overrides | Tracking cohesion/adhesion |
|-----------|--------------|------------------------------|---------------|
| `baseline` | reuse current best seg | (none — default) | active (tuned weights) |
| `- cohesion` | same seg as baseline | (none — default) | `cohesion_weight = cohesion_constant = adhesion_weight = adhesion_constant = 0` |
| `- affinities` | regenerate | `scoring_function = "symmetric"` (under `[waterz_params]`) | all four zeroed |
| `- merges` | regenerate | `skip_merges = true` | all four zeroed (also force-skipped by `no_merges=True`, which `run_tracking.py` infers from the empty merge history) |

### Recipe per condition

All commands run from the repo root in the `mhat-sandbox` conda env (`conda run -n mhat-sandbox --no-capture-output ...`).

**For `- affinities` and `- merges` only — regenerate the segmentation first:**

1. Edit `scripts/02_segmentation/seg_config.toml` with the override from the table above. For `- affinities`, uncomment/set `scoring_function = "symmetric"` under `[waterz_params]`. For `- merges`, set `skip_merges = true`.
2. Run segmentation: `python scripts/02_segmentation/create_seg_hypotheses.py scripts/02_segmentation/seg_config.toml`.
3. Note the resulting `seg_uid` — the timestamped directory created under `experiments/segmentation/<experiment>/<dataset>/`.

**For every condition — run tracking and eval:**

4. Edit `scripts/04_tracking/tracking_config.toml`:
   - Set `seg_result` to the `seg_uid` for this condition (baseline and `- cohesion` share the same seg_uid).
   - Zero `cohesion_weight`, `cohesion_constant`, `adhesion_weight`, `adhesion_constant` for the three non-baseline conditions; leave them at the tuned values for baseline.
5. Run tracking: `python scripts/04_tracking/run_tracking.py scripts/04_tracking/tracking_config.toml`.
6. Read the `exp_uid` from `experiments/tracking/<experiment>/<dataset>/test_run/config.toml`.
7. Update `scripts/05_evaluation/eval_config.toml` — set `track_result` to that `exp_uid`.
8. Run eval: `python scripts/05_evaluation/evaluate_tracks.py scripts/05_evaluation/eval_config.toml`.
9. Confirm `experiments/evaluation/<experiment>/<dataset>/<exp_uid>/track_metrics.json` was written.
10. Record the `seg_uid` and `tracking_uid` in `scripts/05_evaluation/merge_ablation.toml` under the matching dataset and condition.

### Running notes (UIDs)

The single source of truth for which `seg_uid` and `tracking_uid` correspond to each (dataset, condition) is `scripts/05_evaluation/merge_ablation.toml`. The plotting script reads from this file, so keep it up to date as runs complete.

The `seg_overrides` field in the TOML is informational (the canonical record lives in the saved `config.toml` inside each seg result directory). Together, `seg_uid` + `seg_overrides` are enough to identify and regenerate any segmentation used here.

Each condition also carries a `config = "configs/tracking/…"` pointer to a static, git-ignored archive of its tracking config (rebuilt by `scripts/05_evaluation/build_config_archive.py`). The mda231 `baseline` (01_cells) and `- Coh/Adh` (02_cells) runs are **shared with the solver experiments** (one run + one config). See the "Config archive" section in `SOLVER_EXPERIMENTS.md` for the full layout and rebuild step.

### Plotting

Once all four `tracking_uid`s for a dataset are filled in:

```
conda run -n mhat-sandbox --no-capture-output \
  python scripts/05_evaluation/merge_ablation_figure.py \
  scripts/05_evaluation/merge_ablation.toml --dataset {mda231,nc281,nc281_sparse}
```

The output PNG path is read from the TOML's `output_png` field per dataset. Override with `--output PATH` if needed. Missing or unfilled `tracking_uid`s produce a warning and skip the bar, so partial-progress plotting works.

### Recall figure (BasicMetrics) — separate eval + matcher gotcha

`merge_ablation_figure_recall.py` reads **Node Recall / Edge Recall from `BasicMetrics`**, which lives in a **separate** metrics file (`recall_metrics_filename` in the TOML, e.g. `track_metrics_basic.json` for MDA231). This file is **not** produced by the normal CTC eval — `evaluate_tracks.py` always writes `track_metrics.json` only. You have to run a second eval with `metrics = ["basic"]` and then rename/copy the output to the recall filename.

Key gotchas (learned 2026-06-23):

- **`BasicMetrics` rejects the CTC matcher.** It needs a one-to-one matcher; passing `matcher = "ctc"` raises `TypeError: The matched data uses a matcher that does not meet the requirements of the metric`.
- **`PointMatcher` requires an explicit `threshold`** (or `match_threshold`) — it has no default and errors without one.
- **For MDA231 `01_cells` the recall files use `matcher = "point"`, `match_threshold = 10`** (verified: this exactly reproduces the existing `track_metrics_basic.json` files — `iou` at the default 0.6 gives near-zero recall because the 3D segments don't overlap that tightly). Use the same matcher/threshold for any new condition so the recall bars stay comparable.

Procedure to add a recall file for one new run without disturbing its CTC file:

1. Eval with `metrics = ["basic"]`, `matcher = "point"`, `match_threshold = 10` → writes `track_metrics.json` (BasicMetrics).
2. Copy that to `track_metrics_basic.json`.
3. Re-run the CTC eval (`metrics = ["ctc"]`, `matcher = "ctc"`) to restore `track_metrics.json` (and `eval_config.toml`) to the CTC version.

### Regenerating a segmentation for one condition (keep it a one-variable swap)

When regenerating the seg for a `- affinities` / `- merges` condition, change **only** the ablation knob (`scoring_function`, `skip_merges`) and keep every other seg param identical to the **baseline** seg. A stray difference defeats the ablation's one-variable logic.

Concretely: the original MDA231 `01_cells` `- affinities` seg (`2026-04-28_13-43-53`) also carried `outline_sigma = 0.5` vs the baseline's `1.0`. It was regenerated on 2026-06-23 as `2026-06-23_17-04-19` with `outline_sigma = 1.0` (symmetric scoring, cellpose, `merge_thresholds = [1.0]`). The metrics barely moved (TRA 0.8548→0.8553, fp 141→139, LNK unchanged), confirming the conclusion was robust — but the comparison is now clean. The `02_cells` merge segs were already consistent (`outline_sigma = 1.0` throughout); only `01_cells` had the drift. When re-running tracking for a regenerated seg, replicate the existing run's `tracking_config.toml` verbatim and change only `seg_result`.

---

## Other experiments

Addition experiments and solver ablation will get their own sections (or files) when added. Existing one-off `solver_ablation_figure_*.py` and `solver_addition_figure_*.py` scripts in `scripts/05_evaluation/` are out of scope for this doc.
