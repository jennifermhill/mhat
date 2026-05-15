# SSVM Weight Fitting Results on MDA231 / 01_cells

Per-condition TRA / DET / LNK before and after the post-hoc constant offset sweep, all on Fluo-C3DL-MDA231 / 01_cells with seg `2026-04-03_11-09-49`, flow `2026-02-17_15-55-00`. CTC matcher (>50% IoU). All SSVM rows use motile / structsvm `scale-features-and-costs` branch.

## 2026-05-15 update: ilpy fix resolves the underlying convergence bug

After updating ilpy to a version containing a fix for the structsvm QP, **stock motile `Solver.fit_weights()` (no standardization, no tolerant termination, no post-hoc offset) actually converges**. ε decreases monotonically from above:

```
+96.6 → +24.0 → +0.60 → +0.47 → ... → +0.001 → +3e-5 → 0
```

31 iterations, true convergence. Pre-offset inference is non-empty (464 nodes / 263 edges).

| | TRA | DET | LNK | \|ε\| |
|---|-----|-----|-----|-----|
| Hand-tuned | 0.881 | 0.886 | 0.845 | — |
| SSVM (old ilpy) | empty | empty | empty | 5 |
| **SSVM (new ilpy), un-tuned** | **0.797** | **0.841** | 0.478 | **~3e-7** |

This is the **un-tuned result of fitting with SSVM** — no post-hoc offset, no standardization workarounds, no regularizer tuning (used the original default `ssvm_reg = 0.1`). All the per-feature std / tolerant termination / graph normalization / dual-offset workarounds documented below were chasing symptoms of the ilpy bug. With it fixed, they're obsolete for getting *a* working SSVM fit, though the post-hoc offset workflow may still help close the remaining gap to hand-tuned (TRA: 0.797 → 0.881).

Code state as of this update: `fit_weights_ssvm.py` uses stock `solver.fit_weights()`; `MDA231_ssvm_fit.toml` has `ssvm_reg = 0.1`. The `fit_weights_standardized` / `TolerantBundleMethod` helpers are still in the file but unused.

---

The sections below document the pre-fix experiment series, kept for historical context.

## Primary comparison

| Condition | TRA (before) | DET (before) | LNK (before) | TRA (after) | DET (after) | LNK (after) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hand-tuned (canonical)** | 0.881 | 0.886 | 0.845 | — | — | — |
| SSVM unaltered (scale-features-and-costs) | empty | empty | empty | 0.826 | 0.840 | 0.723 |
| SSVM + per-feature feature-matrix standardization | empty | empty | empty | **0.841** | **0.846** | **0.803** |
| SSVM + graph-attribute normalization | 0.772 | 0.834 | 0.318 | 0.832 | 0.845 | 0.739 |

"Empty" = inference returned 0 nodes / 0 edges (offset required to get any selection).

## Diagnostic counts

| Condition | fp_nodes | fn_nodes | ns_nodes | fp_edges | fn_edges | Bundle iter | Final ε |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hand-tuned (canonical) | 101 | 28 | 7 | — | — | — | — |
| SSVM unaltered, after offset | 161 | 40 | — | — | — | 2 | exited early on first ε<0 |
| SSVM + per-feature std, after offset | 156 | 39 | — | — | — | ~25 (plateau) | -0.022 |
| SSVM + graph-norm, before offset | 165 | 33 | 22 | 10 | 219 | 4 | -0.0005 |
| SSVM + graph-norm, after offset | 180 | 37 | 3 | 5 | 83 | 4 | -0.0005 (same as before) |

Notes on diagnostic completeness:
- Per-feature std after-offset eval metric files (under `experiments/evaluation/.../ssvm_off_*/`) were overwritten by the subsequent graph-norm sweep — so fp_edges/fn_edges/ns_nodes for that row are no longer on disk; only fp_nodes/fn_nodes survive in `/tmp/sweep_reg10.out`. If you need those numbers for plotting, re-run the per-feature std fit + sweep.
- SSVM unaltered baseline numbers come from the prior CLAUDE.md "Best result on MDA231" section (pre-current-fixes); only fp=161/fn=40 were recorded there.

## Configuration notes per row

**Hand-tuned (canonical)** — Reproduces via `scripts/04_tracking/MDA231_baseline.toml`. Best config from coordinate-wise hand-tuning: drift_w=57, drift_c=-2000, area_w=1730, area_c=-1500, intensity_w=4, intensity_c=-1000, cohesion_w=2000, adhesion_w=-1000, appear/disappear=200. exp_uid: 2026-04-15_09-42-38 (rerun 2026-05-01_17-12-10).

**SSVM unaltered** — Pre-fix workflow: stock `motile.Solver.fit_weights()`, `ssvm_reg=0.1`, 1D offset sweep on `intensity_constant` only (plateau at -1000). Bundle method exited after 2 iterations on first negative-ε event. From CLAUDE.md history, prior to standardization/tolerant-termination patches.

**SSVM + per-feature feature-matrix standardization** — Current main code path. Standardizes columns of `solver.features.to_ndarray()` (the full feature matrix including 0-padded rows for inactive variable types) by per-column std, then inverse-scales returned weights. `TolerantBundleMethod` keeps iterating past spurious negative-ε. `ssvm_reg=10`. 2D offset sweep on `intensity_constant` and `drift_constant`. Best: intensity_offset=-2000 + drift_offset=-500 (plateaus to identical TRA across multiple offset combos).

**SSVM + graph-attribute normalization** — Experimental (reverted). Divides raw graph attributes (drift_dist, area_diff, intensity_diff, cohesion, adhesion) in place by their active-entry stds, runs stock `motile.Solver.fit_weights()` with no further standardization, then inverse-scales returned weight terms (not constants) to original feature space. `ssvm_reg=10`. Achieves clean convergence (ε ≈ -5e-4 in 4 iterations) but learned weight magnitudes are tiny (drift_w=-8e-6, intensity_w=-3e-6), making the post-hoc offset sweep completely insensitive (all 8 offset combos in the 2D sweep gave identical metrics). Code reverted after the run; results above are from output saved to `experiments/tracking/Fluo-C3DL-MDA231/01_cells/ssvm_fit/` before revert.

## Graph-norm regularizer sweep

Tested three `ssvm_reg` values with graph-attribute normalization to characterize how regularization interacts with weight magnitudes and post-hoc offset effectiveness. All runs used motile's stock `fit_weights` (no `TolerantBundleMethod`), so reg=0.1 and reg=1 exited on the first negative-ε event (2-3 iterations); reg=10 actually converged to small |ε|.

| `ssvm_reg` | Bundle iter | Final ε | Pre-offset (TRA / LNK) | Post-offset best (TRA / LNK) | Sweep behavior |
|------------|:---:|:---:|:---:|:---:|:---|
| 10 | 4 | -5e-4 | 0.772 / 0.318 (non-empty) | 0.832 / 0.739 | flat across all 8 combos |
| 1 | 3 | -0.026 | empty / 0 | 0.833 / 0.781 | varies (best at i=-1000, d=0) |
| 0.1 | 2 | -0.68 | empty / 0 | **0.834** / 0.787 | flat across all 8 combos |

**Observations:**
- Marginal differences post-offset (TRA spans 0.832-0.834 across reg values).
- reg=10 is the only condition where pre-offset inference is *non-empty* (the cleanest SSVM convergence point). All other graph-norm reg values give empty pre-offset, like the per-feature std and unaltered conditions.
- reg=1 is the only reg value where the offset sweep shows variation across configurations (other reg values produce learned weights that are either tiny or saturated, so offsets just shift absolute cost levels without changing selection).
- Best post-offset TRA across all graph-norm conditions (0.834) is still below per-feature std + dual offset (0.841).

The reg=10 row is what's shown in the main comparison table and the `ssvm_results.png` figure, since it has the most informative pre-offset (non-empty) and the cleanest ε convergence.

## Comparison to hand-tuned

| Condition (after offset) | ΔTRA vs hand-tuned | ΔLNK vs hand-tuned |
|---------------------------|:---:|:---:|
| SSVM unaltered | -0.055 | -0.122 |
| SSVM + per-feature std | **-0.040** | **-0.042** |
| SSVM + graph-norm | -0.049 | -0.106 |

## Key observation

The "cleanest convergence" (graph-attribute normalization, |ε|=5e-4) does *not* give the best inference metrics. The "dirtiest convergence" of the standardized approaches (per-feature std, |ε|=0.022) gives the best. This is because:

1. Cleaner convergence comes from more aggressive shrinking of active-entry feature magnitudes
2. Smaller features → smaller learned weights (the regularizer pulls them toward zero)
3. Smaller learned weights → less relative discrimination between cost terms
4. Less relative discrimination → post-hoc offset becomes a uniform absolute shift, losing the per-feature distinguishability that closed the LNK gap

The per-feature std approach hits a sweet spot: enough conditioning to converge to a reasonable fixed point (~25 iterations, ε plateau), but not so aggressive that the learned weight pattern is washed out.

## Source files / reproducibility

- Logfiles with full ε trajectories: `experiments/tracking/Fluo-C3DL-MDA231/01_cells/ssvm_fit/fit_weights_ssvm_*.log`
- Sweep output: `/tmp/sweep_reg10.out` (per-feature std) and `/tmp/sweep_graphnorm.out` (graph norm) — temp files, may not persist
- Fit config: `scripts/04_tracking/MDA231_ssvm_fit.toml` (`ssvm_reg = 10.0`)
- Hand-tuned config: `scripts/04_tracking/MDA231_baseline.toml`
