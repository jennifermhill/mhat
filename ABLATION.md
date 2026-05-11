# Ablation Experiments

Branch-specific instructions for running ablation experiments on the `seg_ablation` branch.

> **Branch note**: The `ablate_*` flags in `scripts/04_tracking/tracking_config.toml` and the `skip_merges` / `scoring_function = "symmetric"` segmentation overrides only exist on `seg_ablation` (added in commit `9271610`). Running these experiments from `main` will fail.

---

## Merge Ablation

The merge ablation experiment compares four conditions to isolate the contributions of cohesion/adhesion costs, affinity-aware fragment scoring, and the merge hyperedges themselves.

### The four conditions

| Condition | Segmentation | `seg_config.toml` overrides | Tracking flag |
|-----------|--------------|------------------------------|---------------|
| `baseline` | reuse current best seg | (none — default) | `ablate_cohesion_adhesion = false` |
| `- cohesion` | same seg as baseline | (none — default) | `ablate_cohesion_adhesion = true` |
| `- affinities` | regenerate | `scoring_function = "symmetric"` (under `[waterz_params]`) | `ablate_cohesion_adhesion = true` |
| `- merges` | regenerate | `skip_merges = true` | `ablate_cohesion_adhesion = true` |

### Recipe per condition

All commands run from the repo root in the `mhat-sandbox` conda env (`conda run -n mhat-sandbox --no-capture-output ...`).

**For `- affinities` and `- merges` only — regenerate the segmentation first:**

1. Edit `scripts/02_segmentation/seg_config.toml` with the override from the table above. For `- affinities`, uncomment/set `scoring_function = "symmetric"` under `[waterz_params]`. For `- merges`, set `skip_merges = true`.
2. Run segmentation: `python scripts/02_segmentation/create_seg_hypotheses.py scripts/02_segmentation/seg_config.toml`.
3. Note the resulting `seg_uid` — the timestamped directory created under `experiments/segmentation/<experiment>/<dataset>/`.

**For every condition — run tracking and eval:**

4. Edit `scripts/04_tracking/tracking_config.toml`:
   - Set `seg_result` to the `seg_uid` for this condition (baseline and `- cohesion` share the same seg_uid).
   - Set `ablate_cohesion_adhesion = true` for the three non-baseline conditions; `false` for baseline.
5. Run tracking: `python scripts/04_tracking/run_tracking.py scripts/04_tracking/tracking_config.toml`.
6. Read the `exp_uid` from `experiments/tracking/<experiment>/<dataset>/test_run/config.toml`.
7. Update `scripts/05_evaluation/eval_config.toml` — set `track_result` to that `exp_uid`.
8. Run eval: `python scripts/05_evaluation/evaluate_tracks.py scripts/05_evaluation/eval_config.toml`.
9. Confirm `experiments/evaluation/<experiment>/<dataset>/<exp_uid>/track_metrics.json` was written.
10. Record the `seg_uid` and `tracking_uid` in `scripts/05_evaluation/merge_ablation.toml` under the matching dataset and condition.

### Running notes (UIDs)

The single source of truth for which `seg_uid` and `tracking_uid` correspond to each (dataset, condition) is `scripts/05_evaluation/merge_ablation.toml`. The plotting script reads from this file, so keep it up to date as runs complete.

The `seg_overrides` field in the TOML is informational (the canonical record lives in the saved `config.toml` inside each seg result directory). Together, `seg_uid` + `seg_overrides` are enough to identify and regenerate any segmentation used here.

### Plotting

Once all four `tracking_uid`s for a dataset are filled in:

```
conda run -n mhat-sandbox --no-capture-output \
  python scripts/05_evaluation/merge_ablation_figure.py \
  scripts/05_evaluation/merge_ablation.toml --dataset {mda231,nc281}
```

The output PNG path is read from the TOML's `output_png` field per dataset. Override with `--output PATH` if needed. Missing or unfilled `tracking_uid`s produce a warning and skip the bar, so partial-progress plotting works.

---

## Other experiments

Addition experiments and solver ablation will get their own sections (or files) when added. Existing one-off `solver_ablation_figure_*.py` and `solver_addition_figure_*.py` scripts in `scripts/05_evaluation/` are out of scope for this doc.
