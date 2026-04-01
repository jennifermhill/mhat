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

### Cellpose Current Best Config

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

### Cellpose Open Questions

- **Optimization has plateaued.** 4 out of 5 Batch 5 runs gave identical results. The ILP solution is quantized and very stable.
- **Remaining errors: fp=102, fn=29, fn_edges=48.** These likely represent a segmentation floor — the 102 fp_nodes may be small cellpose fragments that don't correspond to GT objects.
- **Would intensity help at higher weight?** With old segmentation, intensity_w=20 (cost std≈1857) was needed. For cellpose, intensity_diff has std=481.7 → w≈4 for cost std≈900. Could try w=4-10 range.
- **Adhesion was never tested with cellpose.** Given cohesion is the only strong lever, adhesion may also have signal at calibrated weight (~7000).
