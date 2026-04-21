# Working Memory: MHAT Tracking Optimization

## Summary

Coordinate-wise search over ILP cost parameters for multi-hypothesis tracking on Fluo-C3DL-MDA231/01_cells. 35 runs across 7 batches (2026-03-19 to 2026-03-20). Primary targets: TRA and DET. Three parameters drove all improvement: drift_constant, drift_weight, and cohesion_weight. All other parameters tested (area, adhesion, appear/disappear, cohesion_constant, adhesion_constant) had negligible effect at the operating point. Optimization has plateaued — further gains likely require changes outside weight/constant tuning (segmentation quality, graph construction, or intensity features).

## Results Comparison Table

| Run | Config summary | TRA | DET | LNK | fp | fn | Notes |
|-----|---------------|-----|-----|-----|----|----|-------|
| Baseline | defaults | 0.741 | 0.755 | 0.637 | 118 | 70 | Starting point |
| B1R2 | drift_c=-2000 | 0.747 | 0.757 | 0.675 | 125 | 71 | First big LNK gain |
| B3R3 | drift_c=-2000, drift_w=50 | 0.748 | 0.757 | 0.684 | 125 | 71 | Best balanced LNK |
| B4R2 | drift_c=-2000, drift_w=50, coh=-5000 | 0.750 | 0.762 | 0.663 | 117 | 68 | First DET-focused best |
| B5R1 | drift_c=-2000, drift_w=50, coh=-7000 | 0.752 | 0.764 | 0.660 | 113 | 67 | |
| B6R1 | drift_c=-2000, drift_w=50, coh=-10000 | 0.755 | 0.767 | 0.663 | 112 | 66 | Previous best |
| FB-I-R4 | +intensity_w=20, intensity_c=-3000 | 0.756 | 0.769 | 0.666 | 112 | 66 | |
| FB-I2-R1 | +area_w=2000, area_c=-1500 | 0.758 | 0.770 | 0.668 | 112 | 66 | |
| **Sanity-R2** | **area_w=3000** | **0.758** | **0.770** | **0.671** | **111** | **66** | **FINAL BEST** |

## Established Principles

Confirmed patterns — require 3+ supporting runs with consistent direction.

- **drift_constant=-2000 is optimal.** Going beyond (-2500, -3000) causes fp_nodes explosion (160+) and DET crash. Confirmed in B3R1, B3R2. Going less (-1000, -1500) gives weaker LNK. Confirmed in baseline, B1R1, B1R2.
- **drift_weight=50 is better than 100.** Lower weight reduces penalty for distant edges, allowing more correct links. drift_w=25 pushes too far (fp=134). Confirmed: B3R3 > B1R2, B3R4 < B3R3, B4R1 too aggressive.
- **cohesion_weight is the primary lever for DET.** Each step from -3000 to -5000 to -7000 to -10000 improved DET monotonically. -15000 and -12000 showed diminishing returns. Confirmed across B1R5, B4R2, B5R1, B6R1, B6R2, B7R4.
- **Reducing curvature weight or constant is harmful.** Both curvature_w=25 (B4R4) and curvature_c=-500 (B1R3) worsened results; curvature_c=-1500 (B2R3) caused fp explosion (157) without intensity cost. With intensity cost active, curv_c=-1500 is safe but doesn't improve (FB-I2-R2).
- **area_weight needs to be ~2000 to have effect.** Earlier tests at w=2.5-10 gave cost std<5 (invisible). At w=2000 (cost std≈910), area became the biggest single improvement. The attribute (mean=0.5, std=0.455) has good relative spread. Confirmed: FB-I2-R1, FB-I2-R5.
- **adhesion_weight and adhesion_constant have no effect.** Cohesion dominates node selection completely. Tested: adh_w=-50 (B5R3), adh_w=-100 (B6R3), adh_c=-200 (B6R5). All identical.
- **cohesion_constant changes have no effect.** Tested: coh_c=-200 (B5R2), coh_c=-50 (B6R4). No improvement.
- **appear/disappear constants have marginal or negative effect** at the current operating point. Tested: 300 (B2R2, B4R3, B5R4), 400 (B1R4). Never improved TRA/DET.
- **There is a DET-vs-LNK trade-off.** Stronger cohesion improves DET but slightly hurts LNK. Best LNK config (drift_c=-2000, drift_w=50, no cohesion) gives LNK=0.684 but DET=0.757. Best DET config (same + coh=-10000) gives DET=0.767 but LNK=0.663.

## Falsified Hypotheses

- "Reducing curvature penalty helps LNK by allowing more edges" — FALSIFIED in B1R3, B2R3, B4R4. Curvature encouragement filters bad trajectories.
- "Increasing curvature weight helps DET at the best operating point" — FALSIFIED in B7R1. No improvement.
- "Stronger appear/disappear costs reduce false positives meaningfully" — FALSIFIED across B1R4, B5R4, B5R5. Minor fp reduction but hurts TRA/DET.
- "Combinations of individually beneficial parameters compound" — FALSIFIED in B2R1, B2R5. Combined configs were often worse than individual changes.
- "drift_constant can be pushed beyond -2000" — FALSIFIED in B3R1, B3R2. Sharp DET/fp degradation past -2000.

## Open Questions

- **Intensity costs provide real signal at high weight/constant.** intensity_w=20, intensity_c=-3000 (cost mean≈-1117, std≈1857) gave new overall best. Costs below ~std=900 had no effect — intensity needs to compete with drift/cohesion scale. w=30 overpenalizes (LNK drops). Confirmed in FB-I-R1 through FB-I-R5.
- **Why does cohesion hurt LNK?** Hypothesis: stronger cohesion preference selects more-merged (larger) fragments, which may shift edge cost distributions unfavorably for linking. Untested.
- **Is the remaining error dominated by segmentation quality?** fp=112, fn=66 may represent a floor given the segmentation hypotheses available.
- **Would different graph construction parameters (max_edge_distance, size_threshold) help?** These are not tunable per CLAUDE.md constraints.

## Current Best Config (Farneback)

```toml
drift_weight = 50.0
drift_constant = -2000.0
area_weight = 3000.0
area_constant = -1500.0
intensity_weight = 20.0
intensity_constant = -3000.0
curvature_weight = 50.0
curvature_constant = -1000.0
cohesion_weight = -10000
cohesion_constant = -100
adhesion_weight = -10
adhesion_constant = -100
appear_constant = 200.0
disappear_constant = 200.0
```

---

## Lucas-Kanade Flow Optimization (2026-03-23)

Flow switched to 3D Lucas-Kanade (flow_result = "2026-03-23_16-49-41", use_lk = true). 15 runs across 3 batches.

### LK Results Comparison Table

| Run | Config summary | TRA | DET | LNK | fp | fn | Notes |
|-----|---------------|-----|-----|-----|----|----|-------|
| LK Baseline | defaults | 0.740 | 0.755 | 0.628 | 116 | 70 | Starting point |
| LK1R1 | Farneback best transferred | 0.755 | 0.767 | 0.663 | 112 | 66 | Perfect transfer |
| LK1R4 | drift_c=-2000, drift_w=50 | 0.748 | 0.757 | 0.684 | 125 | 71 | Best LNK |
| **LK2R3** | **+curv_c=-1500** | **0.755** | **0.768** | **0.663** | **111** | **66** | **LK BEST** |
| LK3R4 | curv_c=-1500, coh=-12000 | 0.755 | 0.768 | 0.663 | 111 | 66 | Matches LK2R3 |

### LK-Specific Findings

- **Farneback-optimized parameters transfer perfectly to LK flow.** LK1R1 matched Farneback best exactly.
- **LK flow is insensitive to drift_constant across [-2000, -3000].** drift_c=-2500 and -3000 give identical results (Farneback had fp explosion at -2500). LK drift distances are much tighter/smaller.
- **LK flow is insensitive to drift_weight across [35, 50].** Both give identical results.
- **curv_c=-1500 is viable with LK (was harmful with Farneback).** Gives marginal DET improvement (0.768 vs 0.767, fp=111 vs 112). This is the only LK-specific improvement found.
- **curv_c=-2000 is too far** — slight regression.
- **coh=-12000 still doesn't help** (same as Farneback finding).
- **appear/disappear=150 doesn't help** (same as Farneback finding).
- **Optimization has plateaued at essentially the same level as Farneback** — TRA: 0.755, DET: 0.768. The marginal curv_c=-1500 improvement is within noise.

### LK Current Best Config

```toml
drift_weight = 50.0
drift_constant = -2000.0
area_weight = 5.0
area_constant = -10.0
intensity_weight = 0.0
intensity_constant = 0.0
curvature_weight = 50.0
curvature_constant = -1500.0
cohesion_weight = -10000
cohesion_constant = -100
adhesion_weight = -10
adhesion_constant = -100
appear_constant = 200.0
disappear_constant = 200.0
```

### LK Open Questions

- **Why is LK flow insensitive to drift parameters?** Likely because LK produces smaller/tighter drift distance values, so the constant term dominates and the weight*distance term is small.
- **Can the tighter LK drift distribution be exploited differently?** Perhaps by using different drift_distance thresholds (currently [0, 3, 6]).
- **Would intensity costs help break the plateau?** Still untested.

---

## Cellpose Segmentation Optimization (2026-03-31)

Segmentation switched to cellpose (seg_result=2026-03-26_12-21-33). Farneback flow. 25 runs across 5 batches. Cellpose dramatically improves baseline over original segmentation. Optimization converged quickly — only cohesion_weight has a strong effect at this operating point.

### Cellpose Results Comparison Table

| Run | Config summary | TRA | DET | LNK | fp | fn | Notes |
|-----|---------------|-----|-----|-----|----|----|-------|
| CP Default | NC281 params, uncalibrated | 0.842 | 0.846 | 0.817 | 167 | 38 | Starting point |
| CP-B1R1 | drift_w=57, drift_c=-2000 only | 0.849 | 0.853 | 0.821 | 159 | 36 | Calibrated drift baseline |
| CP-B1R2 | + pos coh=6000 | 0.836 | 0.840 | 0.809 | 197 | 37 | Positive cohesion hurts |
| CP-B1R3 | + neg coh=-6000 | 0.852 | 0.865 | 0.756 | 98 | 31 | DET best but LNK tanks |
| CP-B1R4 | + area=1730 | 0.854 | 0.857 | 0.832 | 152 | 35 | Best single-param addition |
| CP-B2R1 | area + curvature | 0.853 | 0.855 | 0.841 | 152 | 36 | Curvature helps LNK |
| CP-B2R3 | area + coh=-3000 | 0.879 | 0.882 | 0.850 | 108 | 29 | Breakthrough combo |
| **CP-B3R2** | **area + coh=-4000** | **0.880** | **0.884** | **0.853** | **102** | **29** | **CELLPOSE BEST** |

### Cellpose Established Principles

- **Cellpose segmentation is dramatically better than original.** Baseline TRA 0.842 vs 0.741. Best TRA 0.880 vs 0.758. fp=102 vs 111, fn=29 vs 66.
- **Positive cohesion/adhesion weights hurt.** Tested w=+6000: fp exploded 159→197. Penalizing merged fragments selects too many small fragments.
- **Negative cohesion_weight=-4000 is optimal.** Stable across [-3500, -4500]. At -5000 ns_nodes increase and LNK drops. At -6000 strong degradation. Much milder than old segmentation (-10000).
- **Area cost provides real signal.** area_w=1730 (cost std≈900) improves all metrics uniformly. Insensitive to weight in [1000, 2500] when cohesion is active.
- **drift_weight is insensitive in [40, 57].** Both give identical results.
- **drift_constant=-2000 remains optimal.** -2500 adds fp without benefit.
- **appear/disappear insensitive in [150, 300].** No effect at this operating point.
- **Curvature hurts DET when cohesion is active** (fp 102→122). Without cohesion, curvature helps LNK.
- **Intensity provides no meaningful signal** at calibrated weight (w=1.9).
- **The DET-vs-LNK trade-off is milder with cellpose** — coh=-4000 gives BOTH best DET (0.884) and best LNK (0.853). With old segmentation, best DET and best LNK required different configs.

### Cellpose Current Best Config (pre-fix, seg=2026-03-26_12-21-33)

```toml
drift_weight = 57.0
drift_constant = -2000.0
area_weight = 1730.0
area_constant = -1500.0
intensity_weight = 0.0
intensity_constant = 0.0
curvature_weight = 0.0
curvature_constant = 0.0
cohesion_weight = -4000.0
cohesion_constant = 0.0
adhesion_weight = 0.0
adhesion_constant = 0.0
appear_constant = 200.0
disappear_constant = 200.0
```

### Cellpose Open Questions (pre-fix)

- **Optimization has plateaued.** 4 out of 5 Batch 5 runs gave identical results. The ILP solution is quantized and very stable.
- **Remaining errors: fp=102, fn=29, fn_edges=48.** These likely represent a segmentation floor — the 102 fp_nodes may be small cellpose fragments that don't correspond to GT objects.

---

## Post Cohesion/Adhesion Fix Optimization (2026-04-03)

Cohesion/adhesion calculation fixed (commit 4eae562). New segmentation (seg_result=2026-04-03_11-09-49). With new code: positive cohesion_weight encourages merged fragments, negative adhesion_weight encourages standalone confidence. 16 runs across 3 batches.

### Post-Fix Results Comparison Table

| Run | Config summary | TRA | DET | LNK | fp | fn | ns | Notes |
|-----|---------------|-----|-----|-----|----|----|-----|-------|
| PF-B1R1 | coh=2000, adh=-1200 | 0.872 | 0.883 | 0.797 | 92 | 26 | 15 | High ns, LNK suffers |
| PF-B1R2 | coh=2000, adh=-800 | 0.879 | 0.887 | 0.824 | 93 | 27 | 10 | Strong balance |
| PF-B1R3 | coh=2000, adh=-500 | 0.875 | 0.881 | 0.833 | 99 | 30 | 7 | Best LNK but DET drops |
| PF-B1R4 | coh=3000, adh=-1200 | 0.847 | 0.858 | 0.768 | 68 | 36 | 18 | coh too strong |
| PF-B1R5 | coh=3000, adh=-800 | 0.775 | 0.782 | 0.723 | 54 | 68 | 12 | Much worse |
| **PF-B2R1** | **coh=2000, adh=-1000** | **0.879** | **0.887** | **0.821** | **95** | **26** | **11** | **Best TRA/DET, lowest fn** |
| PF-B2R2 | coh=2000, adh=-900 | 0.879 | 0.887 | 0.824 | 93 | 27 | 10 | Same as adh=-800 |
| PF-B2R3 | coh=2000, adh=-700 | 0.873 | 0.879 | 0.827 | 100 | 30 | 8 | DET drops |
| PF-B2R4 | coh=2000, adh=-600 | 0.875 | 0.881 | 0.833 | 99 | 30 | 7 | Same as adh=-500 |
| PF-B2R5 | coh=2000, adh=-650 | 0.875 | 0.881 | 0.833 | 98 | 30 | 7 | Quantized |
| PF-B3R1 | coh=2000, adh=-750 | 0.873 | 0.880 | 0.815 | 95 | 29 | 10 | Transition zone |
| PF-B3R2 | adh=-800, appear=300 | 0.879 | 0.887 | 0.824 | 93 | 27 | 10 | appear insensitive |
| PF-B3R3 | drift_w=75 | 0.877 | 0.884 | 0.824 | 91 | 28 | 10 | drift_w insensitive |
| PF-B3R4 | area_w=1000 | 0.874 | 0.881 | 0.824 | 97 | 29 | 9 | Worse |
| PF-B3R5 | area_w=2500 | 0.870 | 0.877 | 0.820 | 89 | 31 | 10 | Over-penalizes |
| PF-B3R6 | adh=-800, appear=300 (rerun) | 0.879 | 0.887 | 0.824 | 93 | 27 | 10 | Confirmed |

### Post-Fix Established Principles

- **cohesion_weight=2000 is optimal.** At 3000, fn and ns spike dramatically — too much merged-fragment preference.
- **adhesion_weight has a DET-vs-LNK trade-off.** Stronger adhesion (adh=-800 to -1000) improves DET/TRA but slightly hurts LNK. Weaker adhesion (adh=-500 to -600) improves LNK but drops DET.
- **Two quantized solution regimes:** adh ∈ [-800, -900] gives one solution (TRA=0.879, LNK=0.824); adh ∈ [-500, -650] gives another (TRA=0.875, LNK=0.833). adh=-1000 is a third regime (TRA=0.879, LNK=0.821).
- **drift_weight insensitive in [57, 75].** Consistent with pre-fix finding.
- **area_weight=1730 remains optimal.** 1000 too low, 2500 too high.
- **appear/disappear insensitive in [200, 300].** Consistent with all prior findings.

### Post-Fix Current Best Config (with intensity, 2026-04-15)

Best overall: INT-B1R1 (exp_uid: 2026-04-15_09-42-38)
- TRA=0.881, DET=0.886, LNK=0.845, fp=101, fn=28, ns=7

```toml
# seg_result = "2026-04-03_11-09-49"
drift_weight = 57.0
drift_constant = -2000.0
area_weight = 1730.0
area_constant = -1500.0
intensity_weight = 4.0
intensity_constant = -1000.0
curvature_weight = 0.0
curvature_constant = 0.0
cohesion_weight = 2000.0
cohesion_constant = 0.0
adhesion_weight = -1000.0
adhesion_constant = 0.0
appear_constant = 200.0
disappear_constant = 200.0
```

Prior best (no intensity): PF-B2R1 (exp_uid: 2026-04-03_11-37-48)
- TRA=0.879, DET=0.887, LNK=0.821, fp=95, fn=26, ns=11

### Post-Fix Open Questions

- **Would intensity help now?** Not tested post-fix. intensity_diff std≈481 on old cellpose seg — recalibrate for new seg.
- **Curvature not tested post-fix.** Previously hurt DET when cohesion active.
- **Is the DET-vs-LNK trade-off addressable?** adh=-1000 and adh=-800 give same TRA but different fp/fn/LNK balance. May need edge-cost adjustments to improve LNK without sacrificing DET.

### Negative-Both Cohesion/Adhesion Exploration (2026-04-13 to 2026-04-15)

**Tested but falsified:** "Since both cohesion and adhesion measure confidence (higher=better), both should have negative weights to encourage correct node selection."

30+ runs tested across negative-both configurations. Best achieved: TRA=0.874, DET=0.883, fp=117 (coh=-200, adh=-2000). Never beat positive-coh baseline (TRA=0.879, fp=95).

**Why it fails:** With both negative, the ILP over-selects nodes because average cohesion and adhesion costs are both strongly negative (encouraging). The positive-cohesion approach works because cohesion *actively discourages* fragments (positive cost) while adhesion *encourages* good merges (negative cost) — this push-pull gives better discrimination.

**Confirmed optimal at current best config (seg=2026-04-03_11-09-49):**
- drift_w=57, drift_c=-2000: any deviation worse
- area_w=1730, area_c=-1500: any deviation worse
- appear/disappear=200: insensitive in [200, 1000]
- adhesion_constant: insensitive in [0, 1500]

**Remaining untested:** intensity (need to calibrate for new seg, std≈211), curvature (previously hurt DET).

---

## Flow Unit-Fix Re-optimization (2026-04-15)

**Fix:** Optical flow is now scaled by voxel size at node attachment (`src/mhat/tracking/utils.py:62-68`). Pre-fix, `drift_dist = norm(pos_u + flow_u - pos_v)` mixed world-unit positions with pixel-unit flow — distorted in anisotropic Z.

**MDA231 voxel scale:** Z=6.0, Y=1.242, X=1.242 μm.

**Impact on MDA231:** negligible. drift_dist stats shifted mean 18.60→18.38, std 15.75→16.25. Cost std stayed ~900. Metrics at optimal config unchanged (TRA=0.8808, DET=0.8857, LNK=0.8449, fp=101, fn=28, ns=7).

**Reason:** Cells in MDA231 move predominantly in XY. Z flow magnitudes are small, so scaling them by 6× barely affects drift_dist.

### Drift Re-optimization Results (6 runs)

| Run | drift_w | drift_c | TRA | DET | LNK | fp |
|-----|---------|---------|-----|-----|-----|-----|
| **Baseline** | 57 | -2000 | 0.8808 | 0.8857 | 0.8449 | 101 |
| B1R1 | 50 | -2000 | 0.8796 | 0.8857 | 0.8348 | 101 |
| B1R2 | 65 | -2000 | 0.8808 | 0.8857 | 0.8449 | 101 |
| B1R3 | 40 | -2000 | 0.8796 | 0.8857 | 0.8348 | 101 |
| B1R4 | 57 | -2200 | 0.8803 | 0.8852 | 0.8449 | 103 |
| B1R5 | 57 | -1800 | 0.8808 | 0.8857 | 0.8449 | 101 |

### Post-Fix Confirmed Principles

- drift_w=57, drift_c=-2000 remain optimal (no change from pre-fix).
- Two quantized regimes: drift_w ∈ [57, 65] → optimal; drift_w ∈ [40, 50] → slight LNK drop (0.8449→0.8348, fn_edges 50→52).
- drift_c insensitive in [-2000, -1800]; -2200 adds 2 fp.

### Open Question

- **NC281 likely has more Z motion** — the fix should have a real impact there. Separate re-optimization needed for NC281.
