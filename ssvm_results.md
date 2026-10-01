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

## 2026-07-17: Hamming-cost margin weight sweep

Swept a scalar `weight` on the SSVM Hamming margin (new `weight` param on `structsvm.HammingCosts`, on the `hamming-costs-weight` branch of the in-repo `structsvm` clone; driven from config key `ssvm_hamming_weight` via `fit_weights_hamming_weighted` in `fit_weights_ssvm.py`). Four points on MDA231 / `01_cells`, all with `ssvm_reg = 0.1`, `ssvm_max_iter = 100`, `iogt_threshold = 0.5`, no post-hoc offset, no standardization. CTC matcher.

The margin scales all learned cost magnitudes: `curvature_weight` grows 0.41 → 3.43 from w=1 → w=100, while `appear`/`disappear` constants shrink.

| `ssvm_hamming_weight` | TRA | DET | LNK | fp_nodes | fn_nodes | fn_edges |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | 0.605 | 0.638 | 0.366 | 99 | 118 | 208 |
| **1.0** ⭐ | **0.797** | **0.841** | **0.478** | 149 | 37 | 172 |
| 10.0 | 0.781 | 0.842 | 0.331 | 169 | 36 | 212 |
| 100.0 | 0.767 | 0.842 | 0.213 | 179 | 36 | 241 |
| *Hand-tuned (ref)* | 0.881 | 0.886 | 0.845 | 101 | 28 | — |

**Consistency check:** `weight = 1.0` through the new code path reproduces the stock `fit_weights()` result exactly (TRA 0.7974, DET 0.8409, LNK 0.4783, fp 149, fn 37, fn_edges 172), confirming `weight=1` ≡ default path.

**Conclusion:** TRA peaks at **w=1** and falls off on both sides — the response is non-monotonic with a single interior optimum at the stock setting. Small weight (0.1) collapses **detection** via under-selection (`fn_nodes` 37 → 118, `fp_nodes` drops to 99); large weight (≥10) collapses **linking** (LNK 0.478 → 0.213, `fn_edges` climbs) while DET stays flat (~0.842). No sweep point beats stock SSVM, and all remain short of hand-tuned (gap dominated by LNK). Scaling the Hamming margin is not a productive lever on this dataset.

Reproducibility: fit configs `scripts/04_tracking/MDA231_ssvm_hamming_w{0p1,1,10,100}.toml`; outputs under `experiments/tracking/Fluo-C3DL-MDA231/01_cells/ssvm_hamming_w{0p1,1,10,100}/`; eval configs under `experiments/tracking/Fluo-C3DL-MDA231/01_cells/eval_configs/`; metrics under `experiments/evaluation/Fluo-C3DL-MDA231/01_cells/ssvm_hamming_w*/track_metrics.json`.

---

## 2026-07-17: Does SSVM win on its own objective? (Hamming cost of the two solutions)

**Question.** SSVM fits weights by minimizing a structured loss whose task term is the **Hamming distance** to the GT annotation (`gt_selected` on candidate nodes/edges), whereas hand-tuning was optimized against CTC TRA/DET. Hypothesis: the SSVM solution should have a *lower* Hamming cost than hand-tuned even though its TRA/DET are worse — i.e. the two methods each win their own game and the TRA gap is pure objective mismatch.

**Method.** Rebuilt the MDA231 `01_cells` candidate graph (deterministic; no ILP solve), annotated `gt_selected` via `annotate_gt_on_candidate_graph` (`iogt_threshold=0.5`), then loaded the two already-saved solution graphs and counted, over all candidate variables, where each solution's selection differs from `gt_selected`. No re-solving. Baseline solution = `2026-05-15_10-55-56` (hand-tuned, TRA 0.881); SSVM solution = `ssvm_refit` (default all-features fit reproduced on Y:, TRA ≈ 0.797). Node-ID coverage of both solutions in the rebuilt candidate graph was 100%, and the direct mismatch count matches the affine form `H = Σgt + Σ(1−2·gt)·y` exactly.

Candidate graph: **636 nodes + 1271 edges = 1907** labeled variables; GT-positive = **339 nodes, 298 edges**.

| | selected nodes | selected edges | node FP | node FN | node Hamming | edge FP | edge FN | edge Hamming | **total Hamming** | normalized |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hand-tuned** | 430 | 385 | 122 | 31 | 153 | 128 | 41 | 169 | **322** | 0.169 |
| **SSVM (ssvm_refit)** | 464 | 263 | 187 | 62 | 249 | 128 | 163 | 291 | **540** | 0.283 |
| Δ (ssvm − base) | +34 | −122 | +65 | +31 | +96 | 0 | +122 | +122 | **+218** | +0.114 |

FP = selected but `gt=0` (over-selection); FN = unselected but `gt=1` (under-selection).

**Result: hypothesis falsified.** The hand-tuned solution has the *lower* Hamming cost (322 vs 540) — SSVM loses even on the objective it is trained to minimize. So the TRA/DET gap is **not** a case of "each method wins its own metric": hand-tuning produces a solution that is closer to GT by Hamming *and* by CTC.

**Where SSVM loses.** Almost entirely on **edges / linking**: it recovers only 135/298 GT edges (163 edge FN) vs the baseline's 257/298 (41 FN), while selecting far fewer edges overall (263 vs 385). This is the same under-linking that shows up as SSVM's low LNK (0.478 vs 0.845). It also over-selects nodes slightly (464 vs 430, +65 node FP).

**Why SSVM can lose on its own loss.** `fit_weights` minimizes a *regularized structured max-margin* objective (`½·ssvm_reg·‖w‖² + hinge`), not the inference Hamming of the resulting weights. The learned weights are shrunk by `ssvm_reg=0.1` and balanced against the margin, so their inference argmin can sit well away from GT. Combined with heavy edge class imbalance (973 negative vs 298 positive candidate edges) and positive appear/disappear constants (~0.68) outweighing the weak learned edge-selection incentive, the inference optimum systematically under-links. The takeaway: the lever to close the gap is the **fit objective / edge-selection incentive** (margin balancing, per-class Hamming weighting, or the appear/disappear vs drift trade-off), not a post-hoc constant offset — consistent with the Hamming-margin sweep above, where no scalar reweighting beat the stock setting.

Reproducibility: one-off analysis script (candidate rebuild + `gt_selected` annotation + solution load), not committed. Solution sources as above; the numbers are internally consistent (node TP+FN=339, edge TP+FN=298 for both).

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
- Fit config: `scripts/04_tracking/MDA231_ssvm_fit.toml` (`ssvm_reg = 0.1` — current default after the 2026-05-15 ilpy fix; the pre-fix experiments in this doc used `ssvm_reg = 10.0`)
- Hand-tuned config: `scripts/04_tracking/MDA231_baseline.toml`

---

## 2026-09-30: SSVM refit on the REGENERATED MDA231 segmentation (post waterz fix)

The hand-tuned MDA231 bars had already been regenerated on `fs1cpm6_seg_base_wzfix`
(post waterz affinity-convention fix, cellpose 4.2.1.1) as `regenF01_baseline` /
`regenF02_baseline`; the SSVM bars had not. Both `ssvm_vs_handtuned` figures are now
on the regenerated inputs.

### Protocol

Stage 1 re-calibration on the regenerated **train** split only, then verbatim transfer
to test — no leakage. Spec:
`scratch_configs/tracking/Fluo-C3DL-MDA231/01_cells/gt_amount_regen_stage1.toml`
(a copy of the published stage-2 config with only seg/flow, `regsweep_subdir` and the
grid changed, so `ablate_curvature`, the GT label space and the candidate-graph params
are identical to the published sweep).

13-point twelfth-decade grid, 0.1 → 0.01, chosen to bracket the old plateau
[0.0215, 0.0825] on both sides. All 13 fits converged (ε → 0 from above; the chosen one
in 35 of 100 allowed iterations) and none was empty. `n_labeled` = **1864** on the new
segmentation, down from 2023.

| ssvm_reg | 0.1 | .0825 | .0681 | .0562 | .0464 | **.0383** | .0316 | .0261 | .0215 | .0178 | .0147 | .0121 | .01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train TRA | .9079 | .9061 | .9039 | .9078 | .9083 | **.9099** | .9095 | .8988 | .8993 | .8998 | .9041 | .9034 | .9046 |

Top-3 spread 0.0017 < 0.005, so the **plateau rule** applies rather than the argmax.
The contiguous within-0.005 run is [0.0316, 0.0562], bracketed on both sides
(0.0261 → 0.8988, 0.0681 → 0.9039); its geometric centre is 0.04214 and the nearest
measured point in log space is **0.0383** (marginally closer than 0.0464). That is both
the plateau centre and, here, the argmax — and it is the same `ssvm_reg` the pre-fix
calibration selected. Effective reg = 0.0383 × 1864 = **71.39** (was 77.48).

Chosen fit: `gt_amount/stage1_regsweep_regen/regsweep_r05`. Its weights were applied
verbatim to 02_cells via
`scratch_configs/tracking/Fluo-C3DL-MDA231/02_cells/ssvm_regen_test_config.toml`
(every candidate-graph key asserted equal to `regenF02_baseline`'s, so the two test
bars differ only in the ILP weights) → `gt_amount/stage3_solves_regen/regsweep_r05`.

### Results — regenerated vs superseded

| Split | Bar | TRA | DET | LNK | SEG | run |
|---|---|---|---|---|---|---|
| 01_cells (train) | hand-tuned | **0.9156** | 0.9203 | 0.8812 | 0.6892 | `regenF01_baseline` |
| 01_cells (train) | SSVM | **0.9099** | 0.9118 | 0.8963 | 0.6708 | `gt_amount/stage1_regsweep_regen/regsweep_r05` |
| 02_cells (test) | hand-tuned | **0.9469** | 0.9501 | 0.9236 | 0.7187 | `regenF02_baseline` |
| 02_cells (test) | SSVM | **0.9306** | 0.9337 | 0.9078 | 0.6850 | `gt_amount/stage3_solves_regen/regsweep_r05` |

Superseded (old seg `seg_cp_20260720_fs1_cpm6` / `holdout_fs1_cpm6`): train hand-tuned
0.9061, train SSVM 0.9044; test hand-tuned 0.9369, test SSVM 0.9359.

### Reading

- **On train the gap narrows to 0.0057 TRA** (0.9156 vs 0.9099), the closest SSVM has
  come to hand-tuning on this dataset. SSVM actually wins LNK (0.8963 vs 0.8812) and
  loses on nodes (DET 0.9118 vs 0.9203, fp 136 vs 130, fn 17 vs 12) — the old
  "SSVM's weakness is linking" reading no longer holds here.
- **On test the gap widens to 0.0163 TRA**, where it used to be 0.0010. Both bars moved
  in opposite directions (hand-tuned 0.9369 → 0.9469, SSVM 0.9359 → 0.9306), so the
  near-tie in the published test figure does not survive regeneration. Note the
  asymmetry in how the two were re-optimized: the hand-tuned params got a full
  coordinate retune on the regenerated train data (`rtC_base` / `rtC2_base`,
  `adhesion_weight` → −2500), while SSVM only re-fits and re-calibrates its single
  regularizer. Both are each method's own protocol, so the comparison is fair, but the
  hand-tuned side had more degrees of freedom to exploit the new segmentation.
- SEG is lower for SSVM on both splits (0.6708 / 0.6850 vs 0.6892 / 0.7187), consistent
  with it selecting more nodes than the hand-tuned solve on both: 480 vs 474 on train,
  740 vs 704 on test.

NC281 bars in both figures are untouched and remain on their pre-fix segmentation.
