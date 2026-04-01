# Tracking Optimization Full Log

## Baseline (2026-03-19)
Config: drift_w=100, drift_c=-1000, area_w=5, area_c=-10, curv_w=50, curv_c=-1000, coh_w=-3000, coh_c=-100, adh_w=-10, adh_c=-100, appear=200, disappear=200
TRA: 0.741, DET: 0.755, LNK: 0.637, fp_nodes: 118, fn_nodes: 70, fn_edges: 120

---

## Batch 1: Single-parameter sweeps from baseline

### B1R1: drift_constant -1000 -> -1500
Hypothesis: "Stronger edge encouragement will reduce fn_edges and improve LNK"
TRA: 0.743, DET: 0.756, LNK: 0.648, fp: 119, fn: 70, fn_edges: 116
Verdict: Supported — slight LNK improvement (+0.011)

### B1R2: drift_constant -1000 -> -2000
Hypothesis: "Even stronger drift encouragement continues the LNK trend"
TRA: 0.747, DET: 0.757, LNK: 0.675, fp: 125, fn: 71, fn_edges: 107
Verdict: Supported — best LNK so far (+0.038), but fp_nodes increased to 125

### B1R3: curvature_constant -1000 -> -500
Hypothesis: "Relaxing curvature encouragement allows more edges through, improving LNK"
TRA: 0.740, DET: 0.756, LNK: 0.619, fp: 113, fn: 70, fn_edges: 126
Verdict: Falsified — LNK worsened. Curvature encouragement is needed.

### B1R4: appear/disappear 200 -> 400
Hypothesis: "Higher appear/disappear cost reduces spurious short tracks, lowering fp_nodes"
TRA: 0.739, DET: 0.752, LNK: 0.640, fp: 107, fn: 72, fn_edges: 119
Verdict: Partially supported — fp_nodes dropped (107) but DET/TRA also dropped

### B1R5: cohesion_weight -3000 -> -5000
Hypothesis: "Stronger cohesion preference selects better-merged fragments, improving DET"
TRA: 0.744, DET: 0.759, LNK: 0.640, fp: 114, fn: 69, fn_edges: 119
Verdict: Supported — best DET (0.759), lowest fn_nodes (69)

---

## Batch 2: Pairwise combinations of best B1 signals

### B2R1: drift_c=-2000, cohesion_w=-5000
Hypothesis: "Combining best LNK + best DET parameters will improve both"
TRA: 0.745, DET: 0.758, LNK: 0.651, fp: 116, fn: 69, fn_edges: 115
Verdict: Partially supported — DET good but LNK lower than drift alone (0.675 vs 0.651)

### B2R2: drift_c=-2000, appear/disappear=300
Hypothesis: "drift_c=-2000 with mild fp control will maintain LNK while reducing fp"
TRA: 0.746, DET: 0.756, LNK: 0.675, fp: 118, fn: 72, fn_edges: 107
Verdict: Inconclusive — same LNK as drift alone, appear/disappear had no effect

### B2R3: drift_c=-2000, curvature_c=-1500
Hypothesis: "Stronger curvature encouragement + drift will push more correct edges"
TRA: 0.718, DET: 0.726, LNK: 0.660, fp: 157, fn: 80, fn_edges: 112
Verdict: Falsified — fp_nodes exploded (157), DET crashed. Increasing curvature constant is harmful.

### B2R4: drift_c=-1500, cohesion_w=-5000
Hypothesis: "More conservative drift + cohesion gives balanced improvement"
TRA: 0.745, DET: 0.759, LNK: 0.645, fp: 114, fn: 69, fn_edges: 117
Verdict: Partially supported — good DET but weaker LNK than drift_c=-2000

### B2R5: drift_c=-2000, cohesion_w=-5000, appear/disappear=300
Hypothesis: "Triple combination captures all beneficial effects"
TRA: 0.744, DET: 0.757, LNK: 0.651, fp: 111, fn: 70, fn_edges: 115
Verdict: Falsified — no improvement over individual components. Lowest fp (111) but LNK weak.

---

## Batch 3: Push drift further + drift_weight exploration

### B3R1: drift_c=-2500
Hypothesis: "Pushing drift_constant further continues LNK improvement"
TRA: 0.716, DET: 0.723, LNK: 0.663, fp: 162, fn: 81, fn_edges: 111
Verdict: Falsified — fp_nodes exploded (162), DET crashed. drift_c=-2000 is the ceiling.

### B3R2: drift_c=-3000
Hypothesis: "Even stronger drift to confirm the ceiling"
TRA: 0.716, DET: 0.723, LNK: 0.666, fp: 165, fn: 81, fn_edges: 110
Verdict: Falsified — confirms drift_c=-2000 is the limit before fp explosion

### B3R3: drift_c=-2000, drift_w=50
Hypothesis: "Lower drift weight reduces penalty for distant edges, helping LNK"
TRA: 0.748, DET: 0.757, LNK: 0.684, fp: 125, fn: 71, fn_edges: 104
Verdict: Supported — NEW BEST overall. LNK=0.684, fn_edges=104.

### B3R4: drift_c=-2000, drift_w=200
Hypothesis: "Higher drift weight penalizes distant edges more, improving edge quality"
TRA: 0.742, DET: 0.755, LNK: 0.644, fp: 120, fn: 71, fn_edges: 118
Verdict: Falsified — higher weight hurts LNK. Lower weight is better.

### B3R5: drift_c=-2000, area_w=10
Hypothesis: "Stronger area penalty improves edge quality"
TRA: 0.747, DET: 0.757, LNK: 0.675, fp: 125, fn: 71, fn_edges: 107
Verdict: Inconclusive — identical to drift_c=-2000 alone. area_weight has no effect.

---

## Batch 4: Build on new best (drift_c=-2000, drift_w=50)

### B4R1: drift_c=-2000, drift_w=25
Hypothesis: "Even lower drift weight further improves LNK"
TRA: 0.744, DET: 0.752, LNK: 0.690, fp: 134, fn: 73, fn_edges: 102
Verdict: Partially supported — best LNK (0.690) but fp=134, DET dropped

### B4R2: drift_c=-2000, drift_w=50, cohesion_w=-5000
Hypothesis: "Adding cohesion to best LNK config improves DET"
TRA: 0.750, DET: 0.762, LNK: 0.663, fp: 117, fn: 68, fn_edges: 111
Verdict: Supported — NEW BEST TRA/DET. But LNK dropped vs drift alone.

### B4R3: drift_c=-2000, drift_w=50, appear/disappear=300
Hypothesis: "appear/disappear=300 helps at this operating point"
TRA: 0.750, DET: 0.759, LNK: 0.684, fp: 118, fn: 71, fn_edges: 104
Verdict: Inconclusive — same LNK as B3R3, no real improvement

### B4R4: drift_c=-2000, drift_w=50, curvature_w=25
Hypothesis: "Reducing curvature weight allows more edge pairs through"
TRA: 0.720, DET: 0.727, LNK: 0.669, fp: 155, fn: 80, fn_edges: 109
Verdict: Falsified — fp explosion again. Curvature weight must stay at 50+.

### B4R5: drift_c=-2000, drift_w=50, area_c=-50
Hypothesis: "Stronger area constant encouragement helps edges"
TRA: 0.748, DET: 0.757, LNK: 0.684, fp: 125, fn: 71, fn_edges: 104
Verdict: Inconclusive — identical to B3R3. area_constant has no effect.

---

## Batch 5: Focus on TRA/DET, explore node costs at best config

### B5R1: drift_c=-2000, drift_w=50, cohesion_w=-7000
Hypothesis: "Stronger cohesion further improves DET"
TRA: 0.752, DET: 0.764, LNK: 0.660, fp: 113, fn: 67, fn_edges: 112
Verdict: Supported — NEW BEST TRA/DET. Cohesion trend continues.

### B5R2: drift_c=-2000, drift_w=50, cohesion_w=-5000, cohesion_c=-200
Hypothesis: "Stronger cohesion constant helps at coh_w=-5000"
TRA: 0.750, DET: 0.762, LNK: 0.663, fp: 117, fn: 68, fn_edges: 111
Verdict: Falsified — identical to B4R2. Cohesion constant has no effect.

### B5R3: drift_c=-2000, drift_w=50, cohesion_w=-5000, adhesion_w=-50
Hypothesis: "Stronger adhesion helps node selection"
TRA: 0.750, DET: 0.762, LNK: 0.663, fp: 117, fn: 68, fn_edges: 111
Verdict: Falsified — identical to B4R2. Adhesion weight has no effect (cohesion dominates).

### B5R4: drift_c=-2000, drift_w=50, cohesion_w=-5000, appear/disappear=300
Hypothesis: "appear/disappear helps with cohesion"
TRA: 0.749, DET: 0.761, LNK: 0.663, fp: 110, fn: 69, fn_edges: 111
Verdict: Inconclusive — marginal. Slightly worse TRA/DET.

### B5R5: drift_c=-2000, drift_w=50, cohesion_w=-7000, appear/disappear=300
Hypothesis: "Combining best cohesion + appear/disappear"
TRA: 0.748, DET: 0.760, LNK: 0.657, fp: 108, fn: 69, fn_edges: 113
Verdict: Falsified — appear/disappear hurts slightly at this operating point

---

## Batch 6: Push cohesion further + secondary node costs

### B6R1: drift_c=-2000, drift_w=50, cohesion_w=-10000
Hypothesis: "Cohesion=-10000 continues the DET improvement trend"
TRA: 0.755, DET: 0.767, LNK: 0.663, fp: 112, fn: 66, fn_edges: 111
Verdict: Supported — NEW BEST TRA/DET. Still improving.

### B6R2: drift_c=-2000, drift_w=50, cohesion_w=-15000
Hypothesis: "Even stronger cohesion continues the trend"
TRA: 0.752, DET: 0.766, LNK: 0.651, fp: 111, fn: 66, fn_edges: 115
Verdict: Falsified — slight regression. Cohesion=-10000 is near optimal.

### B6R3: drift_c=-2000, drift_w=50, cohesion_w=-7000, adhesion_w=-100
Hypothesis: "Stronger adhesion helps at coh=-7000"
TRA: 0.752, DET: 0.764, LNK: 0.660, fp: 113, fn: 67, fn_edges: 112
Verdict: Falsified — identical to B5R1. Adhesion has no effect.

### B6R4: drift_c=-2000, drift_w=50, cohesion_w=-7000, cohesion_c=-50
Hypothesis: "Reducing cohesion constant helps discrimination"
TRA: 0.751, DET: 0.763, LNK: 0.660, fp: 106, fn: 68, fn_edges: 112
Verdict: Falsified — slightly worse TRA/DET.

### B6R5: drift_c=-2000, drift_w=50, cohesion_w=-7000, adhesion_c=-200
Hypothesis: "Stronger adhesion constant helps"
TRA: 0.752, DET: 0.764, LNK: 0.660, fp: 113, fn: 67, fn_edges: 112
Verdict: Falsified — identical to B5R1. Adhesion constant has no effect.

---

## Batch 7: Edge cost variations at best config (drift_c=-2000, drift_w=50, coh=-10000)

### B7R1: +curvature_w=100
Hypothesis: "Stronger curvature penalty filters bad edge pairs, helping DET indirectly"
TRA: 0.754, DET: 0.767, LNK: 0.654, fp: 112, fn: 66, fn_edges: 114
Verdict: Inconclusive — same DET, slightly worse TRA/LNK

### B7R2: +area_w=2.5
Hypothesis: "Lower area penalty lets more size-varying edges through"
TRA: 0.755, DET: 0.767, LNK: 0.663, fp: 112, fn: 66, fn_edges: 111
Verdict: Inconclusive — identical to B6R1. area_weight has no effect.

### B7R3: +drift_w=75
Hypothesis: "Slightly higher drift weight (between 50 and 100) helps edge discrimination"
TRA: 0.754, DET: 0.767, LNK: 0.660, fp: 112, fn: 66, fn_edges: 112
Verdict: Inconclusive — essentially identical. drift_w=50 is fine.

### B7R4: cohesion_w=-12000
Hypothesis: "Fine-tuning between -10000 and -15000"
TRA: 0.752, DET: 0.766, LNK: 0.651, fp: 112, fn: 66, fn_edges: 115
Verdict: Falsified — slight regression. -10000 is optimal.

### B7R5: +curvature_w=100, area_w=2.5
Hypothesis: "Combining edge cost changes"
TRA: 0.754, DET: 0.767, LNK: 0.654, fp: 112, fn: 66, fn_edges: 114
Verdict: Inconclusive — identical to B7R1. area_weight confirmed inert.

---

## Lucas-Kanade Flow Optimization (2026-03-23)

Flow switched from 2D Farneback to 3D Lucas-Kanade (flow_result = "2026-03-23_16-49-41", use_lk = true).

### LK Baseline: default config with LK flow
Config: drift_w=100, drift_c=-1000, area_w=5, area_c=-10, curv_w=50, curv_c=-1000, coh_w=-3000, coh_c=-100, adh_w=-10, adh_c=-100, appear=200, disappear=200
TRA: 0.740, DET: 0.755, LNK: 0.628, fp_nodes: 116, fn_nodes: 70, fn_edges: 123

---

## Batch LK1: Transfer Farneback-best config + single-param sweeps

### LK1R1: Farneback best (drift_c=-2000, drift_w=50, coh_w=-10000)
Hypothesis: "Previous best config transfers directly to LK flow"
TRA: 0.755, DET: 0.767, LNK: 0.663, fp: 112, fn: 66, fn_edges: 111
Verdict: Supported — identical to Farneback best. Config transfers perfectly.

### LK1R2: drift_c=-2000 only
Hypothesis: "Isolate drift effect with LK flow"
TRA: 0.747, DET: 0.757, LNK: 0.675, fp: 125, fn: 71, fn_edges: 107
Verdict: Supported — matches Farneback B1R2 exactly.

### LK1R3: drift_c=-1500
Hypothesis: "LK flow may have different drift scale, test milder value"
TRA: 0.743, DET: 0.756, LNK: 0.650, fp: 118, fn: 70, fn_edges: 116
Verdict: Supported — weaker than -2000, same pattern as Farneback.

### LK1R4: drift_c=-2000, drift_w=50
Hypothesis: "Drift weight reduction helps with LK flow too"
TRA: 0.748, DET: 0.757, LNK: 0.684, fp: 125, fn: 71, fn_edges: 104
Verdict: Supported — best LNK, matches Farneback B3R3.

### LK1R5: cohesion_w=-5000 only
Hypothesis: "Isolate cohesion effect with LK flow"
TRA: 0.746, DET: 0.762, LNK: 0.634, fp: 112, fn: 68, fn_edges: 121
Verdict: Supported — matches Farneback pattern.

---

## Batch LK2: Push beyond Farneback plateau

### LK2R1: drift_c=-2500, drift_w=50, coh_w=-10000
Hypothesis: "drift_c=-2500 caused fp explosion with Farneback; LK may handle it"
TRA: 0.755, DET: 0.767, LNK: 0.663, fp: 112, fn: 66, fn_edges: 111
Verdict: Supported — identical to drift_c=-2000! LK flow is insensitive to drift_c in [-2000, -2500]. (Farneback gave fp=162 at -2500.)

### LK2R2: drift_w=35, drift_c=-2000, coh_w=-10000
Hypothesis: "Lower drift weight (between 25 and 50) helps with LK"
TRA: 0.755, DET: 0.767, LNK: 0.663, fp: 112, fn: 66, fn_edges: 111
Verdict: Inconclusive — identical to drift_w=50. LK drift distances are small enough that weight doesn't matter much.

### LK2R3: +curv_c=-1500
Hypothesis: "Stronger curvature encouragement; was harmful with Farneback (fp=157), may work with LK"
TRA: 0.755, DET: 0.768, LNK: 0.663, fp: 111, fn: 66, fn_edges: 111
Verdict: Supported — MARGINAL NEW BEST DET (0.768, fp=111). LK flow makes curv_c=-1500 viable.

### LK2R4: coh_w=-12000
Hypothesis: "Stronger cohesion at the LK operating point"
TRA: 0.752, DET: 0.766, LNK: 0.651, fp: 112, fn: 66, fn_edges: 115
Verdict: Falsified — slight regression, same as Farneback. coh=-10000 is optimal.

### LK2R5: appear/disappear=150
Hypothesis: "Lower appear/disappear allows more tracks"
TRA: 0.754, DET: 0.767, LNK: 0.660, fp: 112, fn: 66, fn_edges: 112
Verdict: Inconclusive — negligibly worse.

---

## Batch LK3: Explore curvature direction + extreme drift

### LK3R1: curv_c=-2000
Hypothesis: "Push curvature constant further than -1500"
TRA: 0.752, DET: 0.764, LNK: 0.663, fp: 113, fn: 67, fn_edges: 111
Verdict: Falsified — too far. curv_c=-1500 is the sweet spot.

### LK3R2: curv_c=-1500, curv_w=75
Hypothesis: "Stronger curvature weight + constant together"
TRA: 0.754, DET: 0.767, LNK: 0.660, fp: 112, fn: 66, fn_edges: 112
Verdict: Falsified — curv_w increase hurts slightly.

### LK3R3: curv_c=-1500, drift_c=-2500
Hypothesis: "Combine the two LK-robust changes"
TRA: 0.752, DET: 0.764, LNK: 0.663, fp: 113, fn: 67, fn_edges: 111
Verdict: Falsified — worse together than individually.

### LK3R4: curv_c=-1500, coh_w=-12000
Hypothesis: "curv_c=-1500 + stronger cohesion"
TRA: 0.755, DET: 0.768, LNK: 0.663, fp: 111, fn: 66, fn_edges: 111
Verdict: Inconclusive — matches LK2R3. coh=-12000 vs -10000 makes no difference with curv_c=-1500.

### LK3R5: drift_c=-3000
Hypothesis: "Push drift constant to extreme; was disastrous with Farneback (fp=165)"
TRA: 0.755, DET: 0.767, LNK: 0.663, fp: 112, fn: 66, fn_edges: 111
Verdict: Supported — identical to drift_c=-2000. Confirms LK flow drift distances are much tighter than Farneback, making drift_constant insensitive across [-2000, -3000].

---

## Intensity Cost Exploration — Farneback flow (2026-03-24)

Base config: drift_w=50, drift_c=-2000, coh_w=-10000, curv_c=-1000 (Farneback best). Farneback flow (flow_result = "2026-02-17_15-55-00", use_lk = false).
Graph stats: intensity_diff mean=94.1, std=92.8.

### FB-I-R1: intensity_w=2, intensity_c=-500
Hypothesis: "Gentle intensity cost (cost mean≈-312, std≈186) improves edge selection"
TRA: 0.755, DET: 0.767, LNK: 0.663, fp: 112, fn: 66, fn_edges: 111
Verdict: Falsified — no effect. Cost scale too small.

### FB-I-R2: intensity_w=5, intensity_c=-1000
Hypothesis: "Moderate intensity cost (cost mean≈-529, std≈464) moves the needle"
TRA: 0.755, DET: 0.767, LNK: 0.663, fp: 112, fn: 66, fn_edges: 111
Verdict: Falsified — still no effect. Needs to compete with drift/cohesion scale costs.

### FB-I-R3: intensity_w=10, intensity_c=-1500
Hypothesis: "Larger intensity cost (cost mean≈-559, std≈928) starts to matter"
TRA: 0.755, DET: 0.768, LNK: 0.663, fp: 111, fn: 66, fn_edges: 111
Verdict: Inconclusive — marginal DET improvement (0.768 vs 0.767), 1 fewer fp.

### FB-I-R4: intensity_w=20, intensity_c=-3000
Hypothesis: "Cohesion-scale intensity cost (cost mean≈-1117, std≈1857) provides real signal"
TRA: 0.756, DET: 0.769, LNK: 0.666, fp: 112, fn: 66, fn_edges: 110
Verdict: Supported — NEW OVERALL BEST. All three metrics improved over previous best.

### FB-I-R5: intensity_w=30, intensity_c=-4000
Hypothesis: "Even stronger intensity (cost mean≈-1176, std≈2785) continues trend"
TRA: 0.756, DET: 0.769, LNK: 0.660, fp: 112, fn: 66, fn_edges: 112
Verdict: Falsified — DET held but LNK dropped. Weight=30 overpenalizes high-intensity-diff edges.

---

## Statistics-Informed Exploration — Farneback flow (2026-03-24)

Base config: drift_w=50, drift_c=-2000, coh_w=-10000, intensity_w=20, intensity_c=-3000 (FB-I-R4 best).
Used graph attribute statistics to identify underweighted parameters.

### FB-I2-R1: area_w=2000, area_c=-1500
Hypothesis: "Area_diff has decent spread (std=0.455) but was invisible at w=5 (cost std=2.3). At w=2000 cost std≈910, area becomes discriminating."
TRA: 0.758, DET: 0.770, LNK: 0.668, fp: 112, fn: 66, fn_edges: 108, fp_edges: 3
Verdict: Supported — NEW OVERALL BEST. Overturns earlier finding that area has no effect — it was just severely underweighted.

### FB-I2-R2: curv_c=-1500
Hypothesis: "Intensity cost stabilizes the solution, making curv_c=-1500 viable with Farneback (previously caused fp explosion)"
TRA: 0.756, DET: 0.769, LNK: 0.666, fp: 112, fn: 66, fn_edges: 110
Verdict: Partially supported — no fp explosion (confirmed stabilization), but no improvement. Curvature cost shift from -120 to -620 mean didn't change solution.

### FB-I2-R3: intensity_w=15, intensity_c=-2000
Hypothesis: "Sweet spot between w=10 (no effect, std=928) and w=20 (best, std=1857)"
TRA: 0.755, DET: 0.768, LNK: 0.660, fp: 111, fn: 66, fn_edges: 112
Verdict: Falsified — cost std=1393 not enough. Intensity needs std>~1800 to matter.

### FB-I2-R4: intensity_w=20, intensity_c=-2000
Hypothesis: "Same weight, less blanket encouragement → more discriminating edge selection"
TRA: 0.754, DET: 0.768, LNK: 0.654, fp: 110, fn: 66, fn_edges: 114
Verdict: Falsified — cost mean shifted from -1117 to -117, many edges now have positive (discouraging) costs. The large negative constant is essential.

### FB-I2-R5: area_w=2000, area_c=-1500 + intensity_w=20, intensity_c=-3000
Hypothesis: "Combining area and intensity improvements compounds gains"
TRA: 0.758, DET: 0.770, LNK: 0.668, fp: 112, fn: 66, fn_edges: 108, fp_edges: 3
Verdict: Inconclusive — identical to R1. Area + intensity don't compound.

---

## Area Weight Sanity Check (2026-03-24)

Base config: FB-I2-R1 best + intensity_w=20, intensity_c=-3000. Varying area_w only.

### Sanity-R1: area_w=1000
TRA: 0.758, DET: 0.770, LNK: 0.668, fp: 112, fn: 66, fn_edges: 108, fp_edges: 3
Verdict: Identical to area_w=2000. Plateau from w=1000+.

### Sanity-R2: area_w=3000
TRA: 0.758, DET: 0.770, LNK: 0.671, fp: 111, fn: 66, fn_edges: 107, fp_edges: 3
Verdict: Marginal LNK improvement (+0.003), 1 fewer fn_edge. NEW FINAL BEST.

### Sanity-R3: area_w=4000
TRA: 0.758, DET: 0.770, LNK: 0.671, fp: 111, fn: 66, fn_edges: 107, fp_edges: 3
Verdict: Identical to area_w=3000. Plateau confirmed.

## OPTIMIZATION COMPLETE

Final best config: drift_w=50, drift_c=-2000, area_w=3000, area_c=-1500, intensity_w=20, intensity_c=-3000, curv_w=50, curv_c=-1000, coh_w=-10000, coh_c=-100, adh_w=-10, adh_c=-100, appear=200, disappear=200.
Final metrics: TRA: 0.758, DET: 0.770, LNK: 0.671 (fp: 111, fn: 66, fn_edges: 107).
Improvement from baseline: TRA +0.017, DET +0.015, LNK +0.034.

---

## Cohesion/Adhesion Semantics Optimization (2026-03-26)

Testing three approaches to correcting cohesion/adhesion cost semantics. Cohesion = last merge cost (lower is better). With positive weight: low-cost merges encouraged, high-cost merges penalized. Adhesion = 1 - next_merge_cost (lower is better with positive weight).

### Baseline (old negative cohesion, local data)
Config: drift_w=50, drift_c=-2000, area_w=3000, area_c=-1500, intensity_w=20, intensity_c=-3000, curv_w=50, curv_c=-1000, coh_w=-10000, coh_c=-100, adh_w=-10, adh_c=-100, appear=200, disappear=200
TRA: 0.758, DET: 0.770, LNK: 0.671, fp: 111, fn: 66, fn_edges: 107

Graph stats: cohesion mean=0.084, std=0.184 | adhesion mean=0.536, std=0.130 | drift_dist mean=17.56, std=16.54 | area_diff mean=0.52, std=0.47 | intensity_diff mean=113.5, std=112.7 | curvature mean=29.3, std=23.6

## Option 1: Config-only positive cohesion/adhesion weights

### O1-Cal: coh_w=4900, coh_c=-900, adh_w=6900, adh_c=-4200
Hypothesis: "Both cohesion and adhesion at cost std~900 with corrected positive weights"
TRA: 0.637, DET: 0.643, LNK: 0.594, fp: 254, fn: 102, fn_edges: 133
Verdict: Falsified — massive regression. fp exploded (254 vs 111). Positive cohesion at this scale strongly penalizes merged nodes, selecting too many fragments.

### O1-B1R1: coh_w=4900, coh_c=-900, adh off
Hypothesis: "Isolate cohesion effect — positive weight penalizes high-cost merges"
TRA: 0.620, DET: 0.625, LNK: 0.585, fp: 276, fn: 107, fn_edges: 136
Verdict: Falsified — even worse. Cohesion alone selects almost all fragments over merges.

### O1-B1R2: adh_w=6900, adh_c=-4200, coh off
Hypothesis: "Isolate adhesion effect — positive weight penalizes cheap-next-merge nodes"
TRA: 0.661, DET: 0.668, LNK: 0.609, fp: 222, fn: 96, fn_edges: 128
Verdict: Falsified — adhesion alone also harms performance, though less than cohesion.

### O1-B1R3: coh_w=4900, coh_c=-2500, adh off
Hypothesis: "More negative constant keeps all merges encouraged while still discriminating"
TRA: 0.620, DET: 0.625, LNK: 0.585, fp: 276, fn: 107, fn_edges: 136
Verdict: Falsified — identical to R1. Constant doesn't change relative node ordering.

### O1-B1R4: coh_w=1000, coh_c=-200, adh off
Hypothesis: "Gentle cohesion (cost std ~184) has less harmful effect"
TRA: 0.639, DET: 0.645, LNK: 0.594, fp: 218, fn: 105, fn_edges: 133
Verdict: Falsified — still much worse than baseline. Even gentle positive cohesion hurts on MDA231.

### O1-B1R5: coh_w=1000, coh_c=-200, adh_w=1500, adh_c=-1000
Hypothesis: "Gentle both together might compensate"
TRA: 0.635, DET: 0.641, LNK: 0.594, fp: 243, fn: 104, fn_edges: 133
Verdict: Falsified — adding adhesion to gentle cohesion makes it worse. Combined positive weights consistently harmful.

### Option 1 Summary
All runs with positive cohesion/adhesion weights are significantly worse than the baseline (TRA ~0.62-0.66 vs 0.758). The fundamental issue: with current attribute encoding, fragments have cohesion=0 which gives them the lowest possible cost with positive weight. MDA231 needs merged nodes, but positive cohesion always prefers fragments. The constant cannot fix this since it shifts all costs equally. Option 1 is not viable for MDA231 without code changes to the attribute encoding.

## Option 2: Fragment cohesion = 0.5

Code change: `last_costs.get(node, 0.0)` → `last_costs.get(node, 0.5)` in create_multihypo_graph.py
New stats: cohesion mean=0.479, std=0.095 (was 0.084, 0.184). Fragments now at 0.5, near mid-range merges.

### O2-Cal: coh_w=500, coh_c=-300, adh_w=100, adh_c=-100
Hypothesis: "Calibration run to measure new distributions"
TRA: 0.649, DET: 0.655, LNK: 0.600, fp: 210, fn: 102, fn_edges: 131
Verdict: Calibration only — cost stds too small (47, 13) to have effect.

### O2-B1R1: coh_w=9500, coh_c=-4750, adh off
Hypothesis: "Fragments neutral (cost=0), low-cost merges encouraged, high-cost penalized"
TRA: 0.670, DET: 0.677, LNK: 0.618, fp: 190, fn: 96, fn_edges: 125
Verdict: Falsified — better than O1 but still far from baseline.

### O2-B1R2: coh_w=9500, coh_c=-6000, adh off
Hypothesis: "More negative constant keeps all nodes encouraged"
TRA: 0.666, DET: 0.672, LNK: 0.618, fp: 219, fn: 95, fn_edges: 125
Verdict: Falsified — constant shift doesn't change ILP solution ranking.

### O2-B1R3: coh_w=9500, coh_c=-8000, adh off
Hypothesis: "Very negative constant — all nodes have negative cost"
TRA: 0.666, DET: 0.672, LNK: 0.618, fp: 219, fn: 95, fn_edges: 125
Verdict: Falsified — identical to R2. Confirms constant is irrelevant to ranking.

### O2-B1R4: coh_w=9500, coh_c=-4750, adh_w=6900, adh_c=-3700
Hypothesis: "Adding adhesion at cost std~900 provides complementary discrimination"
TRA: 0.682, DET: 0.690, LNK: 0.624, fp: 189, fn: 91, fn_edges: 123
Verdict: Partially supported — best positive-weight result so far, but still far from baseline.

### O2-B1R5: OLD negative coh_w=-10000 with fragment=0.5 code
Hypothesis: "Does the code change help/hurt in the old negative-weight regime?"
TRA: 0.643, DET: 0.648, LNK: 0.606, fp: 251, fn: 101, fn_edges: 129
Verdict: Falsified — fragment=0.5 HURTS negative cohesion because fragments now get same cost as high-cost merges (-5100), destroying the discrimination that made old config work.

### Option 2 Summary
Best result: O2-B1R4 (TRA=0.682) with both cohesion and adhesion at cost std~900. Still 0.076 below baseline. The code change increased cohesion mean to ~0.5 but reduced std to 0.095 (from 0.184), making discrimination harder. The fragment=0.5 encoding hurts both positive AND negative weight regimes.

## Option 3: Center attributes at 0, constant=0

Code change: `cohesion = last_costs.get(node, 0.5) - 0.5`, `adhesion = (1 - next_costs.get(node, 0.5)) - 0.5`
New stats: cohesion mean=-0.021, std=0.095 | adhesion mean=0.036, std=0.130
Fragments get cohesion=0 (neutral), good merges negative (encouraged), bad merges positive (penalized).

### O3-Cal: coh_w=9500, coh_c=0, adh_w=6900, adh_c=0
Hypothesis: "Both at cost std~900 with centered attributes and zero constants"
TRA: 0.678, DET: 0.686, LNK: 0.618, fp: 183, fn: 93, fn_edges: 125
Verdict: Best first-attempt of all options.

### O3-B1R1: coh_w=9500, coh_c=0, adh off
Hypothesis: "Isolate cohesion"
TRA: 0.670, DET: 0.677, LNK: 0.618, fp: 190, fn: 96, fn_edges: 125
Verdict: Adhesion contributes ~0.008 TRA.

### O3-B1R2: adh_w=6900, adh_c=0, coh off
Hypothesis: "Isolate adhesion"
TRA: 0.663, DET: 0.670, LNK: 0.609, fp: 206, fn: 97, fn_edges: 128
Verdict: Adhesion alone weaker than cohesion alone.

### O3-B1R3: coh_w=19000, adh_w=13800 (2x)
Hypothesis: "Higher weights improve discrimination"
TRA: 0.683, DET: 0.692, LNK: 0.615, fp: 180, fn: 90, fn_edges: 126
Verdict: Supported — monotonic improvement with higher weights.

### O3-B1R4: coh_w=38000, adh_w=27600 (4x)
Hypothesis: "Continue scaling up"
TRA: 0.691, DET: 0.701, LNK: 0.618, fp: 170, fn: 88, fn_edges: 125
Verdict: Supported — still improving. DET crossed 0.700.

### O3-B1R5: coh_w=76000, adh_w=55200 (8x)
Hypothesis: "Continue scaling up"
TRA: 0.692, DET: 0.702, LNK: 0.615, fp: 165, fn: 88, fn_edges: 126
Verdict: Plateau reached — marginal improvement over 4x.

### O3-B2R1: coh_w=76000, adh off
Hypothesis: "Is adhesion helping at high cohesion?"
TRA: 0.682, DET: 0.691, LNK: 0.618, fp: 165, fn: 92, fn_edges: 125
Verdict: Adhesion contributes ~0.01 TRA even at very high cohesion weights.

### O3-B2R2: coh_w=150000, adh_w=55200
Hypothesis: "Push cohesion much higher"
TRA: 0.686, DET: 0.696, LNK: 0.612, fp: 168, fn: 90, fn_edges: 127
Verdict: Falsified — too high. Cohesion dominates other costs.

### O3-B2R3: coh_w=76000, adh_w=55200, curv_c=-2000
Hypothesis: "Stronger curvature encouragement (cost mean → -535)"
TRA: 0.691, DET: 0.701, LNK: 0.618, fp: 167, fn: 88, fn_edges: 125
Verdict: Inconclusive — same as B1R5. Curvature constant doesn't help.

### O3-B2R4: coh_w=76000, adh_w=55200, no area cost
Hypothesis: "Area cost has positive mean (+62), might be hurting"
TRA: 0.687, DET: 0.698, LNK: 0.606, fp: 168, fn: 89, fn_edges: 129
Verdict: Falsified — area helps despite positive mean cost.

### O3-B2R5: coh_w=76000, adh_w=55200, area_c=-2000 (cost mean → -438)
Hypothesis: "Fix area cost to negative mean"
TRA: 0.691, DET: 0.701, LNK: 0.615, fp: 167, fn: 88, fn_edges: 126
Verdict: Inconclusive — same as B1R5. Area constant doesn't matter.

### Option 3 Summary
Best result: O3-B1R5 (TRA=0.692, DET=0.702) with coh_w=76000, adh_w=55200 and constant=0. This is the best of all three options but still 0.066 below the baseline (TRA=0.758). Adhesion provides meaningful signal (~0.01 TRA) for the first time. Higher weights help monotonically up to ~8x but plateau after that.

## Cross-Option Comparison

| Option | Best TRA | Best DET | Best Config | Gap vs Baseline |
|--------|----------|----------|-------------|-----------------|
| Baseline (old coh_w=-10000) | 0.758 | 0.770 | coh_w=-10000, coh_c=-100 | — |
| O1: Config-only positive weights | 0.661 | 0.668 | adh_w=6900, adh_c=-4200 | -0.097 |
| O2: Fragment cohesion=0.5 | 0.682 | 0.690 | coh_w=9500, adh_w=6900 | -0.076 |
| O3: Centered at 0, constant=0 | 0.692 | 0.702 | coh_w=76000, adh_w=55200 | -0.066 |

**Key finding**: All three corrected approaches are significantly worse than the backwards-semantics baseline. The old negative cohesion works because it functions as a strong "select merged nodes" signal — it doesn't actually discriminate by merge quality, it just prefers ANY merge over fragments. On MDA231, this is what works because the cells need large merged representations. Correct semantics (penalize high-cost merges) hurts because even "bad" merges are better than fragments for this dataset.

**Positive finding**: Option 3 (centered) showed for the first time that adhesion provides real signal (~0.01 TRA), suggesting the centered encoding is the most semantically sound. With further work on the merge cost quality or per-dataset tuning, Option 3 could potentially match the baseline.

---

## Cellpose Segmentation Optimization (2026-03-31)

New segmentation: cellpose (seg_result=2026-03-26_12-21-33), Farneback flow (2026-02-17_15-55-00).
Starting from NC281 params. Goal: comprehensive sweep with positive cohesion/adhesion, then full exploration.

### Graph Statistics (Cellpose)
| Attribute | Count | Mean | Std | Target Weight (cost std ~900) |
|-----------|-------|------|-----|-------------------------------|
| cohesion | 633 | 0.059 | 0.147 | ~6100 |
| adhesion | 633 | 0.553 | 0.130 | ~6900 |
| drift_dist | 1269 | 18.602 | 15.753 | ~57 |
| area_diff | 1269 | 0.601 | 0.520 | ~1730 |
| intensity_diff | 1269 | 412.053 | 481.663 | ~1.9 |
| curvature | 2802 | 31.899 | 21.539 | ~42 |

### Uncalibrated Default (NC281 params transferred)
TRA: 0.842, DET: 0.846, LNK: 0.817, fp: 167, fn: 38, fn_edges: 60

---

### Batch 1: Single-parameter exploration from calibrated drift baseline

Base config: drift_w=57, drift_c=-2000, everything else off, appear/disappear=200.

### CP-B1R1: Calibrated drift-only baseline
Hypothesis: "Calibrated drift (cost std ~900) will improve over NC281 uncalibrated defaults"
TRA: 0.849, DET: 0.853, LNK: 0.821, fp: 159, fn: 36, fn_edges: 58
Verdict: Supported — calibration helps across all metrics

### CP-B1R2: + positive cohesion (w=6000, c=-800)
Hypothesis: "Positive cohesion penalizes merged fragments, may reduce fp from over-segmented cellpose"
TRA: 0.836, DET: 0.840, LNK: 0.809, fp: 197, fn: 37, fn_edges: 62
Verdict: Falsified — fp explodes from 159→197. Positive cohesion selects more small fragments.

### CP-B1R3: + negative cohesion (w=-6000, c=0)
Hypothesis: "Negative cohesion encourages merged fragments, will improve DET"
TRA: 0.852, DET: 0.865, LNK: 0.756, fp: 98, fn: 31, ns: 17, fn_edges: 80
Verdict: Supported for DET — but tanks LNK. Same trade-off as old segmentation. ns=17 is high.

### CP-B1R4: + area cost (w=1730, c=-1500)
Hypothesis: "Area similarity encouragement will improve edge selection and LNK"
TRA: 0.854, DET: 0.857, LNK: 0.832, fp: 152, fn: 35, fn_edges: 55
Verdict: Supported — uniform improvement across all metrics. Best single addition.

### CP-B1R5: + curvature (w=42, c=-2000)
Hypothesis: "Trajectory smoothness constraint will improve linking"
TRA: 0.850, DET: 0.854, LNK: 0.823, fp: 158, fn: 36, fn_edges: 58
Verdict: Marginal — slight improvement over drift-only, much less than area.

---

### Batch 2: Build on area as foundation

Base config: drift_w=57, drift_c=-2000, area_w=1730, area_c=-1500, appear/disappear=200.

### CP-B2R1: + curvature (w=42, c=-2000)
Hypothesis: "Curvature + area will compound for better LNK"
TRA: 0.853, DET: 0.855, LNK: 0.841, fp: 152, fn: 36, fn_edges: 52
Verdict: Supported for LNK (0.841 vs 0.832) but DET slightly lower.

### CP-B2R2: + intensity (w=1.9, c=-1500)
Hypothesis: "Intensity similarity will add discriminating signal for edge selection"
TRA: 0.854, DET: 0.857, LNK: 0.832, fp: 149, fn: 35, fn_edges: 55
Verdict: Marginal — same LNK as area-only, slight fp reduction.

### CP-B2R3: + mild negative cohesion (w=-3000, c=0)
Hypothesis: "Mild cohesion will improve DET without tanking LNK like strong cohesion (-6000) did"
TRA: 0.879, DET: 0.882, LNK: 0.850, fp: 108, fn: 29, fn_edges: 49
Verdict: Strongly supported — massive improvement across ALL metrics. Sweet spot found.

### CP-B2R4: area_w=2500 (no cohesion)
Hypothesis: "Higher area weight may improve discrimination"
TRA: 0.855, DET: 0.858, LNK: 0.832, fp: 148, fn: 35, fn_edges: 55
Verdict: Inconclusive — negligible difference from area_w=1730.

### CP-B2R5: area_w=1000 (no cohesion)
Hypothesis: "Check sensitivity to area weight reduction"
TRA: 0.853, DET: 0.856, LNK: 0.827, fp: 159, fn: 35, fn_edges: 56
Verdict: Slightly worse — area_w=1000 is below optimal. 1730 is better.

---

### Batch 3: Explore around B2R3 sweet spot (drift+area+coh=-3000)

Base: drift_w=57, drift_c=-2000, area_w=1730, area_c=-1500, coh_w=-3000, appear=200.

### CP-B3R1: + curvature (w=42, c=-2000)
Hypothesis: "Curvature on top of cohesion+area will improve LNK further"
TRA: 0.875, DET: 0.879, LNK: 0.850, fp: 122, fn: 29, fn_edges: 49
Verdict: Falsified — curvature hurts DET when cohesion is active (fp 108→122). LNK unchanged.

### CP-B3R2: coh=-4000
Hypothesis: "Slightly stronger cohesion may push DET higher without LNK damage"
TRA: 0.880, DET: 0.884, LNK: 0.853, fp: 102, fn: 29, fn_edges: 48
Verdict: Supported — new overall best! All metrics improve.

### CP-B3R3: coh=-2000
Hypothesis: "Check if weaker cohesion is better"
TRA: 0.874, DET: 0.878, LNK: 0.844, fp: 124, fn: 29, fn_edges: 51
Verdict: Falsified — weaker cohesion is worse across all metrics.

### CP-B3R4: coh=-4000 + intensity (w=1.9, c=-1500)
Hypothesis: "Intensity may add signal on top of best config"
TRA: 0.879, DET: 0.884, LNK: 0.848, fp: 103, fn: 29, fn_edges: 49
Verdict: Slightly negative — intensity hurts LNK marginally.

### CP-B3R5: coh=-4000 + area_w=2500
Hypothesis: "Higher area weight with cohesion active"
TRA: 0.880, DET: 0.884, LNK: 0.853, fp: 102, fn: 29, fn_edges: 48
Verdict: Identical to B3R2 — area weight doesn't matter once cohesion is active.

---

### Batch 4: Confirm coh=-4000 optimality, tune drift and appear

Base: drift_w=57, drift_c=-2000, area_w=1730, area_c=-1500, coh_w=-4000, appear=200.

### CP-B4R1: coh=-5000
Hypothesis: "Stronger cohesion may continue improvement trend"
TRA: 0.876, DET: 0.881, LNK: 0.841, fp: 102, fn: 29, ns: 8, fn_edges: 52
Verdict: Falsified — ns increases, LNK drops. coh=-4000 is optimal.

### CP-B4R2: coh=-6000
Hypothesis: "Even stronger cohesion to confirm degradation trend"
TRA: 0.873, DET: 0.879, LNK: 0.826, fp: 100, fn: 29, ns: 10, fn_edges: 57
Verdict: Strongly falsified — LNK=0.826, ns=10. Well past the sweet spot.

### CP-B4R3: drift_c=-2500
Hypothesis: "More negative drift constant may help linking"
TRA: 0.879, DET: 0.883, LNK: 0.850, fp: 107, fn: 29, fn_edges: 49
Verdict: Slightly worse — drift_c=-2000 is better.

### CP-B4R4: drift_w=40
Hypothesis: "Lower drift weight reduces distance penalty"
TRA: 0.880, DET: 0.884, LNK: 0.853, fp: 102, fn: 29, fn_edges: 48
Verdict: Identical — drift_w insensitive in [40, 57] range.

### CP-B4R5: appear/disappear=150
Hypothesis: "Lower appear cost allows more track starts/ends"
TRA: 0.880, DET: 0.884, LNK: 0.853, fp: 103, fn: 29, fn_edges: 48
Verdict: Identical — appear/disappear insensitive at this operating point.

---

### Batch 5: Fine-tune around best (coh, area_c, appear)

Base: drift_w=57, drift_c=-2000, area_w=1730, area_c=-1500, coh_w=-4000, appear=200.

### CP-B5R1: coh=-3500
Hypothesis: "Finer cohesion grid between -3000 and -4000"
TRA: 0.880, DET: 0.884, LNK: 0.853, fp: 102, fn: 29, fn_edges: 48
Verdict: Identical to coh=-4000 — solution is quantized in this range.

### CP-B5R2: coh=-4500
Hypothesis: "Finer cohesion grid between -4000 and -5000"
TRA: 0.880, DET: 0.884, LNK: 0.853, fp: 102, fn: 29, fn_edges: 48
Verdict: Identical — stable across [-3500, -4500].

### CP-B5R3: area_c=-1000
Hypothesis: "Less negative area constant"
TRA: 0.880, DET: 0.884, LNK: 0.853, fp: 102, fn: 29, fn_edges: 48
Verdict: Identical — area_c insensitive in [-1000, -1500].

### CP-B5R4: area_c=-2000
Hypothesis: "More negative area constant encourages more edges"
TRA: 0.879, DET: 0.883, LNK: 0.850, fp: 107, fn: 29, fn_edges: 49
Verdict: Slightly worse — too much edge encouragement.

### CP-B5R5: appear/disappear=300
Hypothesis: "Higher appear cost discourages track fragmentation"
TRA: 0.880, DET: 0.884, LNK: 0.853, fp: 102, fn: 29, fn_edges: 48
Verdict: Identical — appear/disappear has no effect in [150, 300] range.
