# NC281-Fl2mSiH2B Tracking Optimization Full Log

## Baseline (2026-03-25, no curvature)
Config: drift_w=100, drift_c=-2000, area_w=0, area_c=0, intensity_w=0, intensity_c=0, curv_w=0, curv_c=0, coh_w=500, coh_c=-450, adh_w=100, adh_c=-50, appear=50, disappear=50
TE: 0.638, TF: 0.682, Node_Recall: 1.000, Edge_Recall: 0.820, FN_Edges: 83

---

## Batch NC1: Initial parameter exploration

### NC1-R1: cohesion_w=10000, cohesion_c=-9000
Hypothesis: "Scaling cohesion to cost std≈980 gives it real discriminating power"
TE: 0.642, TF: 0.688, Node_Recall: 1.000, Edge_Recall: 0.826, FN_Edges: 80
Verdict: Inconclusive — marginal improvement, cohesion scaling alone insufficient.

### NC1-R2: area_w=2400, area_c=-1500
Hypothesis: "Area at cost std≈920 improves edge selection (worked on MDA231)"
TE: 0.527, TF: 0.574, Node_Recall: 1.000, Edge_Recall: 0.761, FN_Edges: 110
Verdict: Falsified — significantly harmful. Area cost breaks correct links on this dataset.

### NC1-R3: intensity_w=14, intensity_c=-1500
Hypothesis: "Intensity at cost std≈926 improves edge selection (worked on MDA231)"
TE: 0.547, TF: 0.591, Node_Recall: 1.000, Edge_Recall: 0.759, FN_Edges: 111
Verdict: Falsified — significantly harmful. Same as area — additional edge discrimination hurts.

### NC1-R4: drift_w=50
Hypothesis: "Lower drift weight reduces penalty for distant edges, improving linking"
TE: 0.690, TF: 0.739, Node_Recall: 1.000, Edge_Recall: 0.848, FN_Edges: 70
Verdict: Supported — NEW BEST. Big improvement (+0.052 TE). Less edge discrimination helps.

### NC1-R5: drift_w=50, appear/disappear=100
Hypothesis: "Higher appear/disappear costs reduce track fragmentation"
TE: 0.688, TF: 0.736, Node_Recall: 1.000, Edge_Recall: 0.850, FN_Edges: 69
Verdict: Inconclusive — essentially same as R4. Appear/disappear has negligible effect.

---

## Batch NC2: Drift weight exploration

### NC2-R1: drift_w=25
Hypothesis: "Even lower drift weight continues the improvement trend"
TE: 0.703, TF: 0.738, Node_Recall: 1.000, Edge_Recall: 0.846, FN_Edges: 71
Verdict: Supported — NEW BEST TE. Drift cost std=123, nearly flat.

### NC2-R2: drift_w=10
Hypothesis: "Push drift weight to minimal discrimination"
TE: 0.683, TF: 0.715, Node_Recall: 1.000, Edge_Recall: 0.837, FN_Edges: 75
Verdict: Falsified — too far. Some distance signal still needed.

### NC2-R3: drift_w=50, drift_c=-3000
Hypothesis: "Stronger drift constant with moderate weight improves linking"
TE: 0.688, TF: 0.728, Node_Recall: 1.000, Edge_Recall: 0.850, FN_Edges: 69
Verdict: Inconclusive — similar to NC1-R4. Constant increase doesn't help at w=50.

### NC2-R4: drift_w=25, drift_c=-3000
Hypothesis: "Combine low weight with strong constant"
TE: 0.683, TF: 0.715, Node_Recall: 1.000, Edge_Recall: 0.837, FN_Edges: 75
Verdict: Falsified — identical to R2. At low weight, constant doesn't matter.

### NC2-R5: drift_w=25, appear/disappear=25
Hypothesis: "Lower appear/disappear allows more track breaks for better coverage"
TE: 0.703, TF: 0.738, Node_Recall: 1.000, Edge_Recall: 0.846, FN_Edges: 71
Verdict: Inconclusive — identical to R1. Appear/disappear has no effect in 25-100 range.

---

## Batch NC3: Fine-tune drift_w + explore cohesion/appear

### NC3-R1: drift_w=20
Hypothesis: "Fine-tune below w=25"
TE: 0.683, TF: 0.715, Node_Recall: 1.000, Edge_Recall: 0.837, FN_Edges: 75
Verdict: Falsified — same as drift_w=10. Below 25, solution snaps to worse state.

### NC3-R2: drift_w=30
Hypothesis: "Fine-tune above w=25"
TE: 0.703, TF: 0.738, Node_Recall: 1.000, Edge_Recall: 0.848, FN_Edges: 70
Verdict: Supported — identical to drift_w=25. Plateau spans 25-30.

### NC3-R3: drift_w=35
Hypothesis: "Fine-tune upper boundary"
TE: 0.696, TF: 0.734, Node_Recall: 1.000, Edge_Recall: 0.852, FN_Edges: 68
Verdict: Inconclusive — best edge recall (0.852) but lower TE. More edges matched but tracks less continuous.

### NC3-R4: drift_w=25, cohesion_w=-5000, cohesion_c=-100
Hypothesis: "Negative cohesion (prefer merged fragments) like MDA231"
TE: 0.588, TF: 0.651, Node_Recall: 1.000, Edge_Recall: 0.794, FN_Edges: 95
Verdict: Falsified — very harmful. Pred nodes dropped from 8737 to 7128. This dataset needs small fragments.

### NC3-R5: drift_w=25, appear/disappear=200
Hypothesis: "High appear/disappear forces longer tracks, improving TE"
TE: 0.666, TF: 0.700, Node_Recall: 1.000, Edge_Recall: 0.831, FN_Edges: 78
Verdict: Falsified — forcing longer tracks introduces wrong links.

---

## Batch NC4: Retry area/intensity with negative constants + adhesion fix

### NC4-R1: area_w=2400, area_c=-3000 (cost mean≈-1742, all edges negative)
Hypothesis: "Area hurt before because edges had positive costs; with c=-3000 all costs are negative"
TE: 0.479, TF: 0.506, Node_Recall: 1.000, Edge_Recall: 0.714, FN_Edges: 132
Verdict: Falsified — even worse than NC1-R2. Area discrimination is harmful regardless of cost sign.

### NC4-R2: intensity_w=14, intensity_c=-3000 (cost mean≈-1958, all edges negative)
Hypothesis: "Intensity with strongly negative constant keeps all edges encouraged"
TE: 0.445, TF: 0.476, Node_Recall: 1.000, Edge_Recall: 0.688, FN_Edges: 144
Verdict: Falsified — worst result yet. Intensity discrimination harmful at any scale.

### NC4-R3: area_w=1000, area_c=-2000 (gentler, cost std≈383)
Hypothesis: "Lower area cost std still provides signal without breaking links"
TE: 0.490, TF: 0.517, Node_Recall: 1.000, Edge_Recall: 0.738, FN_Edges: 121
Verdict: Falsified — still harmful even at cost std=383.

### NC4-R4: intensity_w=3, intensity_c=-1000 (very gentle, cost std≈199)
Hypothesis: "Minimal intensity discrimination at cost std<200"
TE: 0.503, TF: 0.542, Node_Recall: 1.000, Edge_Recall: 0.746, FN_Edges: 117
Verdict: Falsified — still harmful even at cost std=199. Intensity discrimination is wrong at every scale.

### NC4-R5: adhesion_c=-150 (fix positive cost mean to negative)
Hypothesis: "Adhesion cost mean was positive (+15.5); fixing to negative (-84.5) helps node selection"
TE: 0.703, TF: 0.738, Node_Recall: 1.000, Edge_Recall: 0.846, FN_Edges: 71
Verdict: Falsified — identical to best. Adhesion spread too small (std=17.1) to matter.

---

## Cohesion/Adhesion Semantics Optimization (2026-03-26)

Testing Options 2 and 3 from the MDA231 semantics experiment on NC281. Baseline: TE=0.703, TF=0.738 (drift_w=25, coh_w=500, coh_c=-450, adh_w=100, adh_c=-50).

## Option 2: Fragment cohesion = 0.5

Code: `last_costs.get(node, 0.5)`. New stats: cohesion mean=0.416, std=0.149.

### NC-O2-Cal: coh_w=500, coh_c=-450 (old weights, new code)
TE: 0.668, TF: 0.705, NR: 1.000, ER: 0.829
Verdict: Worse than baseline — code change alone harmful.

### NC-O2-B1R1: coh_w=6040, coh_c=-3020, adh_w=5300, adh_c=-3500
TE: 0.525, TF: 0.562, NR: 1.000, ER: 0.720
Verdict: Falsified — both at cost std~900, much worse.

### NC-O2-B1R2: coh_w=6040, coh_c=-3020, adh off
TE: 0.577, TF: 0.638, NR: 1.000, ER: 0.772
Verdict: Falsified — cohesion alone at std~900 still harmful.

### NC-O2-B1R3: coh_w=48000, coh_c=-24000, adh off (8x)
TE: 0.512, TF: 0.559, NR: 1.000, ER: 0.738
Verdict: Falsified — higher weights worse. Opposite of MDA231 trend.

### Option 2 NC281 Summary
All configs worse than baseline. Higher weights make it worse (opposite of MDA231). Best: coh_w=500 at TE=0.668 (-0.035 vs baseline).

## Option 3: Centered at 0, constant=0

Code: `cohesion = last_costs.get(node, 0.5) - 0.5`, `adhesion = (1 - next_costs.get(node, 0.5)) - 0.5`
New stats: cohesion mean=-0.084, std=0.149 | adhesion mean=0.155, std=0.171

### NC-O3-Cal: coh_w=9200, adh_w=5300 (both at std~900)
TE: 0.531, TF: 0.567, NR: 1.000, ER: 0.727
Verdict: Falsified — both at std~900 very harmful. Adhesion mean is +0.155 (discouraging).

### NC-O3-B1R1: coh_w=9200, adh off
TE: 0.560, TF: 0.616, NR: 1.000, ER: 0.757
Verdict: Falsified — cohesion alone at std~1367 still harmful.

### NC-O3-B1R2: coh_w=36800, adh off (4x)
TE: 0.514, TF: 0.562, NR: 1.000, ER: 0.733
Verdict: Falsified — higher is worse (like O2).

### NC-O3-B1R3: coh_w=2000, adh off (gentle)
TE: 0.672, TF: 0.730, NR: 1.000, ER: 0.839
Verdict: Best O3 result. Gentle discrimination closest to baseline.

### NC-O3-B1R4: coh_w=500, adh off (very gentle)
TE: 0.677, TF: 0.720, NR: 1.000, ER: 0.837

### NC-O3-B1R5: coh_w=1000, adh off
TE: 0.672, TF: 0.722, NR: 1.000, ER: 0.833

### NC-O3-B1R6: coh_w=2000, adh_w=1000
TE: 0.601, TF: 0.665, NR: 1.000, ER: 0.785
Verdict: Falsified — adhesion hurts on NC281. Positive adhesion mean (+0.155) discourages too many nodes.

### Option 3 NC281 Summary
Best: coh_w=500, adh off → TE=0.677. Still 0.026 below baseline. On NC281: lower weights are better (opposite of MDA231), and adhesion is consistently harmful.

## NC281 Cross-Option Comparison

| Option | Best TE | Best Config | Gap vs Baseline |
|--------|---------|-------------|-----------------|
| Baseline (coh_w=500, coh_c=-450) | 0.703 | original | — |
| O2: Fragment cohesion=0.5 | 0.668 | coh_w=500, coh_c=-450 | -0.035 |
| O3: Centered, constant=0 | 0.677 | coh_w=500, adh off | -0.026 |

Key difference from MDA231: NC281 prefers minimal node cost influence (drift dominates). Both corrected approaches degrade performance because they change the relative node ranking, and the existing coh_w=500, coh_c=-450 config already has correct semantics for NC281 (positive weight, negative constant — penalizes high-cost merges, encourages low-cost ones and fragments).

---

## Batch NC5: New cohesion/adhesion semantics (2026-04-01)

Code changed 2026-03-31: cohesion = 1 - last_merge_cost (fragments=1), adhesion = next_merge_cost (top merges=1).
Graph stats now report node costs scaled by num_leaves (moved to solve_with_motile after scale_by_leaves).
New cohesion stats: mean=0.950, std=0.098. With leaves scaling at w=-500: cost std=302 (was 49 without scaling).
Adhesion stats: mean=0.595, std=0.411.
Eval threshold changed to 10.0 (was 15.0).
Base config: drift_w=25, drift_c=-2000, appear/disappear=50, area/intensity/curvature off.

### NC5-Cal: coh_w=-500, coh_c=0, adh off
Hypothesis: "Starting point — negative cohesion rewards fragments (cohesion=1)"
TE: 0.688, TF: 0.730, Node_Recall: 1.000, Edge_Recall: 0.844, FN_Edges: 72
Verdict: Baseline for new semantics.

### NC5-R1: coh_w=0 (cohesion off)
Hypothesis: "Test if cohesion matters at all under new semantics"
TE: 0.679, TF: 0.730, Node_Recall: 1.000, Edge_Recall: 0.848, FN_Edges: 70
Verdict: Slightly worse TE — cohesion has some effect.

### NC5-R2: coh_w=-2000, coh_c=0
Hypothesis: "Stronger negative cohesion improves fragment preference"
TE: 0.688, TF: 0.730, Node_Recall: 1.000, Edge_Recall: 0.846, FN_Edges: 71
Verdict: Inconclusive — identical to Cal. Scaling up doesn't help.

### NC5-R3: coh_w=-9200, coh_c=0
Hypothesis: "Full scale cohesion at cost std~900"
TE: 0.690, TF: 0.734, Node_Recall: 1.000, Edge_Recall: 0.848, FN_Edges: 70
Verdict: Inconclusive — marginal improvement. Cohesion has very weak effect due to low attribute spread.

### NC5-R4: coh_w=-500, adh_w=+2200
Hypothesis: "Positive adhesion penalizes merges (high adhesion=1), preferring fragments"
TE: 0.499, TF: 0.545, Node_Recall: 0.919, Edge_Recall: 0.670, FN_Edges: 152
Verdict: Falsified — disastrous. Positive adhesion kills node selection (41 FN nodes).

### NC5-R5: coh_w=-500, adh_w=-1000
Hypothesis: "Negative adhesion rewards merges — does it complement negative cohesion?"
TE: 0.620, TF: 0.673, Node_Recall: 1.000, Edge_Recall: 0.822, FN_Edges: 82
Verdict: Falsified — harmful. Rewarding merges is wrong for NC281.

### NC5-R6: coh_w=-500, coh_c=50
Hypothesis: "Replicate old cost profile: fragments get -500+50=-450, merges get ~0+50=50"
TE: 0.688, TF: 0.730, Node_Recall: 1.000, Edge_Recall: 0.844, FN_Edges: 72
Verdict: Inconclusive — same as Cal. Small constant has no effect.

### NC5-R7: coh_w=-500, coh_c=450
Hypothesis: "Cost mean near zero — constant offsets weight×mean, maximizing discrimination"
TE: 0.690, TF: 0.734, Node_Recall: 1.000, Edge_Recall: 0.852, FN_Edges: 68
Verdict: Supported — NEW BEST for new semantics. Best TE, TF, ER, and FN edges.

### NC5-R8: coh_w=-500, coh_c=500
Hypothesis: "Positive cost mean actively discourages merged nodes"
TE: 0.681, TF: 0.731, Node_Recall: 1.000, Edge_Recall: 0.852, FN_Edges: 68
Verdict: Falsified — cost mean went positive (+68), slightly worse TE.

### NC5-R9: coh_w=-2000, coh_c=1800
Hypothesis: "Scale up both weight and constant, keeping cost mean near zero but higher cost std"
TE: 0.683, TF: 0.723, Node_Recall: 1.000, Edge_Recall: 0.848, FN_Edges: 70
Verdict: Falsified — higher cost std (490) doesn't help. Leaves scaling already provides adequate discrimination.

### NC5-R10: coh_w=-500, coh_c=400
Hypothesis: "Fine-tune constant below 450"
TE: 0.685, TF: 0.731, Node_Recall: 1.000, Edge_Recall: 0.844, FN_Edges: 72
Verdict: Inconclusive — slightly worse than R7.

---

## Batch NC6: Drift, appear/disappear, curvature with new cohesion (2026-04-01)

Base config: drift_w=25, coh_w=-500, coh_c=450, appear/disappear=50.

### NC6-R1: drift_w=20
Hypothesis: "Recheck drift below optimum with new cohesion — interaction may have shifted"
TE: 0.670, TF: 0.711, Node_Recall: 1.000, Edge_Recall: 0.844, FN_Edges: 72
Verdict: Falsified — confirms drift_w=20 below sweet spot even with new cohesion.

### NC6-R2: drift_w=35
Hypothesis: "Upper plateau edge — more edge discrimination"
TE: 0.679, TF: 0.728, Node_Recall: 1.000, Edge_Recall: 0.859, FN_Edges: 65
Verdict: Inconclusive — best edge recall (0.859) but lower TE. More discrimination helps linking but hurts track continuity.

### NC6-R3: appear/disappear=150
Hypothesis: "Between insensitive range (25-100) and harmful (200)"
TE: 0.657, TF: 0.704, Node_Recall: 1.000, Edge_Recall: 0.842, FN_Edges: 73
Verdict: Falsified — 150 is already harmful. Forces wrong links.

### NC6-R4: appear/disappear=10
Hypothesis: "Very low — allow maximum track fragmentation"
TE: 0.685, TF: 0.727, Node_Recall: 1.000, Edge_Recall: 0.850, FN_Edges: 69
Verdict: Inconclusive — same as baseline range. Confirms insensitivity below 100.

### NC6-R5: curvature_w=5, curvature_c=-1000
Hypothesis: "Re-enable curvature — was beneficial in original baseline (TE=0.672 vs 0.638)"
TE: 0.605, TF: 0.640, Node_Recall: 1.000, Edge_Recall: 0.807, FN_Edges: 89
Verdict: Falsified — very harmful. Curvature cost std only 20.6; constant (-1000) dominates and adds noise to edge pair selection.

---

## Batch NC7: Cost balance strategy — discriminating drift with break-even (2026-04-01)

Strategy: Reduce drift_c magnitude to improve ratio of good-to-bad edge costs, addressing "two bad edges beat one good edge" problem. FN edge analysis showed correct edges (drift~4.8) consistently cheaper than substitutes (drift~10-11) but all-negative costs mean solver maximizes edge count.

Base config: drift_w=25, coh_w=-500, coh_c=450, appear/disappear=50, area/intensity/curvature off.

### NC7-R1: drift_c=-250 (break-even at 10, drift only)
Hypothesis: "Aggressive break-even forces solver to discriminate — edges with drift>10 become positive"
TE: 0.592, TF: 0.658, Node_Recall: 0.984, Edge_Recall: 0.794, FN_Edges: 95
Verdict: Falsified — too aggressive. 8 FN nodes, many correct edges rejected (positive cost).

### NC7-R2: drift_c=-250, area_w=100, area_c=-60 (area safety net, break-even 0.6)
Hypothesis: "Area cost rescues correct high-drift edges with good area_diff scores"
TE: 0.584, TF: 0.644, Node_Recall: 0.988, Edge_Recall: 0.800, FN_Edges: 92
Verdict: Falsified — area recovers some nodes but TE still far below baseline.

### NC7-R3: drift_c=-250, intensity_w=1, intensity_c=-100 (intensity safety net, break-even 100)
Hypothesis: "Intensity cost rescues correct high-drift edges"
TE: 0.594, TF: 0.635, Node_Recall: 0.990, Edge_Recall: 0.787, FN_Edges: 98
Verdict: Falsified — intensity even worse than area as safety net.

### NC7-R4: drift_c=-250, area_w=100, area_c=-60, intensity_w=1, intensity_c=-100 (both)
Hypothesis: "Combined safety nets recover most rejected edges"
TE: 0.597, TF: 0.650, Node_Recall: 0.994, Edge_Recall: 0.794, FN_Edges: 95
Verdict: Falsified — both safety nets recover nodes (NR=0.994) but TE still ~0.1 below baseline.

### NC7-R5: drift_c=-750 (gentle break-even at 30, drift only)
Hypothesis: "Less aggressive constant — all edges stay negative but ratio improves"
TE: 0.692, TF: 0.744, Node_Recall: 1.000, Edge_Recall: 0.857, FN_Edges: 66
Verdict: Supported — NEW BEST. Beats old best on all metrics (TE +0.002, TF +0.010, ER +0.005).

---

## Batch NC8: Fine-tune drift_c around -750 (2026-04-01)

### NC8-R1: drift_c=-500 (break-even at 20)
Hypothesis: "Between -250 (too aggressive) and -750 (best) — find the boundary"
TE: 0.675, TF: 0.719, Node_Recall: 0.998, Edge_Recall: 0.848, FN_Edges: 70
Verdict: Falsified — still below plateau. 1 FN node.

### NC8-R2: drift_c=-600 (break-even at 24)
Hypothesis: "Closer to -750 plateau"
TE: 0.679, TF: 0.732, Node_Recall: 1.000, Edge_Recall: 0.850, FN_Edges: 69
Verdict: Falsified — improving but still below plateau.

### NC8-R3: drift_c=-900 (break-even at 36)
Hypothesis: "Other side of -750 — does plateau extend?"
TE: 0.692, TF: 0.744, Node_Recall: 1.000, Edge_Recall: 0.857, FN_Edges: 66
Verdict: Supported — identical to -750. Plateau extends.

### NC8-R4: drift_c=-1000 (break-even at 40)
Hypothesis: "Further extension — plateau + better ER?"
TE: 0.692, TF: 0.744, Node_Recall: 1.000, Edge_Recall: 0.861, FN_Edges: 64
Verdict: Supported — NEW BEST ER (0.861) and lowest FN edges (64). Same TE/TF as -750/-900.

### NC8-R5: drift_c=-1250 (break-even at 50)
Hypothesis: "Does plateau continue beyond -1000?"
TE: 0.668, TF: 0.723, Node_Recall: 1.000, Edge_Recall: 0.857, FN_Edges: 66
Verdict: Falsified — TE drops. -1250 is past the plateau. Sweet spot is -750 to -1000.

---

## Flow Unit-Fix Re-optimization (2026-04-15)

**Context:** Fixed a unit mismatch in `src/mhat/tracking/utils.py:62-68` — optical flow was saved in pixel units but node centroids were world-unit scaled, so `drift_dist = norm(pos_u + flow_u - pos_v)` was mixing units. Fix: scale flow by voxel size at node attachment.

**NC281 voxel scale:** Z=2.110, Y=0.650, X=0.650 μm. Z/XY ratio ≈ 3.25.

**Post-fix graph stats vs pre-fix:**
- drift_dist mean: 8.462 → **6.451** (-24%)
- drift_dist std: 5.132 → **4.198** (-18%)
- At drift_w=25, cost std: 128 → 105

The fix **significantly** affects NC281 — cells in 03_nuclei have meaningful Z motion, so the previously-under-scaled Z flow now correctly predicts Z displacement, reducing drift_dist residuals.

### UFIX-Baseline: pre-fix best config applied post-fix
exp_uid: 2026-04-15_11-57-47
Config: drift_w=25, drift_c=-1000, coh_w=-500, coh_c=450, appear/disappear=50
TE: 0.6443, TF: 0.6722, Node_Recall: 1.000, Edge_Recall: 0.8308, FN_Edges: 78
Verdict: Regression — pre-fix best was TE=0.692, TF=0.744. Old optimum no longer optimal after drift_dist distribution shift.

### UFIX-B1R1: drift_w=30
exp_uid: 2026-04-15_12-04-23
Hypothesis: "Scale drift_w up ~20% to match pre-fix cost std (~128)."
TE: 0.6551, TF: 0.6883, Node_Recall: 1.000, Edge_Recall: 0.8351, FN_Edges: 76
Verdict: Supported — partial recovery (+0.011 TE). Still below pre-fix best.

### UFIX-B1R2: drift_w=35
exp_uid: 2026-04-15_12-09-31
Hypothesis: "Further drift_w increase continues recovery."
TE: 0.6551, TF: 0.6893, Node_Recall: 1.000, Edge_Recall: 0.8373, FN_Edges: 75
Verdict: Inconclusive — TE identical to B1R1; marginal TF/ER gain. Plateau reached.

### UFIX-B1R3: drift_c=-750
exp_uid: 2026-04-15_12-14-39
Hypothesis: "Proportional drift_c decrease to match smaller drift_dist distribution."
TE: 0.6529, TF: 0.6861, Node_Recall: 1.000, Edge_Recall: 0.8351, FN_Edges: 76
Verdict: Falsified — slightly worse than keeping drift_c=-1000. Constant shift alone insufficient.

### UFIX-B1R4: drift_w=30, drift_c=-750
exp_uid: 2026-04-15_12-19-46
Hypothesis: "Combined weight + constant adjustment."
TE: 0.6551, TF: 0.6893, Node_Recall: 1.000, Edge_Recall: 0.8373, FN_Edges: 75
Verdict: Supported — same plateau as B1R2. drift_c ∈ [-1000, -750] insensitive at drift_w=30.

### UFIX-B1R5: drift_c=-1250
exp_uid: 2026-04-15_12-24-57
Hypothesis: "Stronger encouragement — maybe exploits more-trustworthy drift."
TE: 0.6421, TF: 0.6715, Node_Recall: 1.000, Edge_Recall: 0.8351, FN_Edges: 76
Verdict: Falsified — drops back to baseline level. -1250 is past the plateau (same direction as pre-fix).

### Batch Summary

| Run | drift_w | drift_c | TE | TF | ER | FN |
|-----|---------|---------|-----|-----|-----|-----|
| Pre-fix best (NC8-R4) | 25 | -1000 | 0.692 | 0.744 | 0.861 | 64 |
| UFIX-Baseline | 25 | -1000 | 0.6443 | 0.6722 | 0.8308 | 78 |
| **UFIX-B1R1** | **30** | **-1000** | **0.6551** | 0.6883 | 0.8351 | 76 |
| **UFIX-B1R2** | **35** | **-1000** | **0.6551** | **0.6893** | **0.8373** | 75 |
| UFIX-B1R3 | 25 | -750 | 0.6529 | 0.6861 | 0.8351 | 76 |
| **UFIX-B1R4** | **30** | **-750** | **0.6551** | **0.6893** | **0.8373** | **75** |
| UFIX-B1R5 | 25 | -1250 | 0.6421 | 0.6715 | 0.8351 | 76 |

### Conclusions

- **New drift plateau:** drift_w ∈ [30, 35], drift_c ∈ [-1000, -750]. Previously drift_w=25 was optimal.
- **Partial recovery only:** +0.011 TE from baseline, still -0.037 below pre-fix best (0.655 vs 0.692).
- **drift_w increase mirrors distribution shift:** drift_dist std shrank 18%, and optimal drift_w grew ~20-40%.
- **Full recovery likely needs cohesion re-optimization.** Cohesion was tuned (coh_w=-500, coh_c=450) against the old drift distribution. Now that drift is ~24% smaller in mean, the cost-balance between cohesion (node cost) and drift (edge cost) has shifted — cohesion may be over- or under-weighted relative to the new drift regime.
- **ER improvement is small but consistent:** FN edges 78 → 75 (-3). Drift improvements mostly help linking, not detection.

---

## Flow Unit-Fix Re-optimization Batch 2: Cohesion sweep (2026-04-15)

Base config: drift_w=30, drift_c=-1000 (B1 best), post-fix flow units.

### UFIX-B2R1: coh_c=400
exp_uid: 2026-04-15_12-55-06
Hypothesis: "Shift cohesion cost mean slightly negative — encourage more node selection."
TE: 0.6529, TF: 0.6852, Node_Recall: 1.000, Edge_Recall: 0.8308, FN_Edges: 78
Verdict: Falsified — slight regression in all metrics.

### UFIX-B2R2: coh_c=500
exp_uid: 2026-04-15_13-00-27
Hypothesis: "Shift cost mean slightly positive — discourage more."
TE: 0.6551, TF: 0.6883, Node_Recall: 1.000, Edge_Recall: 0.8373, FN_Edges: 75
Verdict: Supported — ties B1 best TE, matches best ER/FN. Plateau extends.

### UFIX-B2R3: cohesion off (coh_w=0, coh_c=0)
exp_uid: 2026-04-15_13-05-37
Hypothesis: "Cohesion provides little value at this operating point — test if disabling helps."
TE: 0.6529, TF: 0.6871, Node_Recall: 1.000, Edge_Recall: 0.8330, FN_Edges: 77
Verdict: Falsified — marginal regression. Cohesion contributes small but real value.

### UFIX-B2R4: coh_w=-1000, coh_c=900 (scale up 2×)
exp_uid: 2026-04-15_13-10-55
Hypothesis: "Scale up cohesion cost std for more discrimination while keeping cost mean near zero."
TE: 0.6508, TF: 0.6778, Node_Recall: 1.000, Edge_Recall: 0.8330, FN_Edges: 77
Verdict: Falsified — worst of batch. Scaling up actively hurts.

### UFIX-B2R5: coh_w=-250, coh_c=225 (scale down 2×)
exp_uid: 2026-04-15_13-16-12
Hypothesis: "Scale down cohesion — test if even weaker is fine."
TE: 0.6551, TF: 0.6883, Node_Recall: 1.000, Edge_Recall: 0.8351, FN_Edges: 76
Verdict: Supported — ties B1 best. Cohesion magnitude insensitive in [-250, -500].

### Batch 2 Summary

| Run | coh_w | coh_c | TE | TF | ER | FN |
|-----|-------|-------|-----|-----|-----|-----|
| B1 best | -500 | 450 | 0.6551 | 0.6883 | 0.8351 | 76 |
| B2R1 | -500 | 400 | 0.6529 | 0.6852 | 0.8308 | 78 |
| **B2R2** | **-500** | **500** | **0.6551** | **0.6883** | **0.8373** | **75** |
| B2R3 | 0 | 0 | 0.6529 | 0.6871 | 0.8330 | 77 |
| B2R4 | -1000 | 900 | 0.6508 | 0.6778 | 0.8330 | 77 |
| B2R5 | -250 | 225 | 0.6551 | 0.6883 | 0.8351 | 76 |

### Conclusions

- **TE plateau at 0.6551 is robust** — unchanged across 5 cohesion variants spanning off, scale-up, scale-down, and constant shifts.
- **Post-fix best config:** drift_w=30, drift_c=-1000, coh_w=-500, coh_c=500 (B2R2, tied with B1R2/B1R4). TE=0.6551, TF=0.6893, ER=0.8373, FN=75.
- **Gap vs pre-fix (0.692) is not closable via cohesion tuning.** Cohesion has very low attribute spread (std 0.098) — cost std stays <200 regardless of weight, so it can't strongly influence node selection.
- **Hypothesis for residual gap:** the pre-fix "optimum" may have been partially lucky — the buggy drift_dist values happened to reward GT-aligned edge selection in this dataset. Post-fix drift is physically correct but selects a different (still valid) subset of edges. This is an artifact of sparse GT making TE sensitive to which specific edges the solver picks, not a tuning failure.
- **Node Recall stays at 1.000 throughout** — no detection regression at any setting.

### Remaining untested

- adhesion with new drift regime (pre-fix always harmful)
- appear/disappear (pre-fix: insensitive <100, harmful 150+)
- area/intensity (pre-fix always harmful)
- curvature (pre-fix always harmful)

---

## Confidence-Based Z Flow Filtering (2026-04-15)

**Context:** Flow diagnostics showed 3D Farneback/LK XY signal is uncorrelated with 2D Farneback (Pearson 0.04/0.26) and Z signal is small/noisy on thin (4-slice) volumes. Rather than attenuating Z uniformly, use per-pixel confidence to drop Z flow contribution where it's unreliable.

**Implementation:** `src/mhat/tracking/utils.py` nodes_from_segmentation now accepts confidence_3d + z_flow_conf_threshold. For each node region, |confidence| is thresholded; if <z_flow_min_pass_pixels (default 10) pass, the node is flagged z_flow_reliable=False. add_flow_dist_attr then uses XY-only drift_dist (dropping both position Z and flow Z from the norm) for edges where the source is unreliable.

### Threshold Sweep (base: drift_w=30, drift_c=-1000, coh_w=-500, coh_c=500)

| Run | Threshold | % unreliable | TE | TF | ER | FN |
|-----|-----------|--------------|-----|-----|-----|-----|
| Baseline | off | 0% | 0.6551 | 0.6893 | 0.8373 | 75 |
| TH-high | 3e-5 (p75 in-region) | 41.0% | 0.6312 | 0.6623 | 0.8286 | 79 |
| TH-mid-hi | 1e-4 (p90 in-region) | 69.2% | 0.6508 | 0.6835 | 0.8373 | 75 |
| TH-mid | 5e-6 | 21.2% | 0.6551 | 0.7016 | 0.8351 | 76 |
| TH-low | 1e-6 | 11.4% | 0.6573 | 0.7033 | 0.8373 | 75 |
| TH-zero | 0.0 (excl exact zeros) | 3.2% | 0.6594 | 0.6924 | 0.8395 | 74 |
| **TH-optimal** | **1e-7** | **5.3%** | **0.6616** | **0.7057** | **0.8416** | **73** |
| TH-optimal + drift_w=25 | 1e-7 | 5.3% | 0.6508 | 0.6900 | 0.8351 | 76 |

### Conclusions

- **Sweet spot: threshold=1e-7, flagging ~5% of nodes as Z-unreliable.** This is the first post-fix config to improve on the un-filtered baseline.
- **Very aggressive filtering (≥40% unreliable) hurts** — going too far removes signal. The useful Z flow information comes from far more than just the most-confident nodes.
- **Monotonic-then-reversal pattern**: too lenient = baseline, light filtering = best, aggressive = worse than baseline, very aggressive = intermediate.
- **Best post-fix config**: TE=0.6616, TF=0.7057, ER=0.8416, FN=73. Gap vs pre-fix (0.692) closed from -0.037 to -0.030.
- **drift_w still optimal at 30**, not 25 (pre-fix value). Confidence filtering doesn't restore the pre-fix drift regime.
- **TF improvement is most dramatic** (+0.017 over baseline) — filtering particularly helps track continuity/coverage.

### Interpretation

The ~5% of nodes with near-zero confidence across their entire region were contributing pure noise in Z. Removing them prevents that noise from corrupting drift_dist for those edges, without over-filtering the many nodes where Z flow has real signal. This is different from the pre-fix regime where Z flow was uniformly attenuated — here we keep full-physical Z flow where it's trustworthy and drop it entirely where it's not.

---

## Curvature Batch (post-flow-fix neutral cost_mean), 2026-04-28

**Setup**: 5 cluster runs, base = post-flow-fix best (TE=0.6616, TF=0.7057, NodeR=1.000, EdgeR=0.8416). Sweep `curvature_weight` with `curvature_constant = -weight × 6.628` (neutral cost_mean). Curvature stats: count=100732, mean=6.628, std=4.112 (target weight ≈220 for cost_std=900).

### Curv-B1R1: curvature_weight=50, constant=-331.4
exp_uid: 2026-04-28_16-43-58_R1
Hypothesis: "Light curvature penalty discriminates without disrupting edge selection (cost_std ~206)"
TE: 0.6551, TF_mean: 0.6830, Node_Recall: 1.000, Edge_Recall: 0.8395
Verdict: falsified — even mild curvature hurts (TE -0.0065)

### Curv-B1R2: curvature_weight=100, constant=-662.8
exp_uid: 2026-04-28_16-43-58_R2
Hypothesis: "Mid curvature with cost_std≈411 has real influence"
TE: 0.6074, TF_mean: 0.6463, Node_Recall: 0.998, Edge_Recall: 0.8265
Verdict: falsified — large regression (TE -0.054)

### Curv-B1R3: curvature_weight=200, constant=-1325.6
exp_uid: 2026-04-28_16-43-58_R3
Hypothesis: "Target weight (cost_std≈822) produces effective curvature discrimination"
TE: 0.5944, TF_mean: 0.6347, Node_Recall: 0.996, Edge_Recall: 0.7918
Verdict: falsified — TE -0.067, EdgeR -0.05 — curvature is filtering out valid edges

### Curv-B1R4: curvature_weight=400, constant=-2651.2
exp_uid: 2026-04-28_16-43-58_R4
Hypothesis: "Strong curvature filters bad trajectories aggressively"
TE: 0.6009, TF_mean: 0.6367, Node_Recall: 0.996, Edge_Recall: 0.8048
Verdict: falsified — slight non-monotonicity (slightly above R3) but still TE -0.061

### Curv-B1R5: curvature_weight=800, constant=-5302.4
exp_uid: 2026-04-28_16-43-58_R5
Hypothesis: "Aggressive curvature penalty"
TE: 0.5900, TF_mean: 0.6281, Node_Recall: 0.994, Edge_Recall: 0.7766
Verdict: falsified — worst result (TE -0.072, EdgeR -0.065)

### Conclusion

Curvature with neutral cost_mean is monotonically harmful for NC281-Fl2m at the current operating point. The pattern (Edge Recall drops monotonically with curvature weight) suggests curvature is filtering out valid GT edges where cells turn or have non-straight motion. Confirms the pre-fix "curvature harmful" finding at proper cost_std calibration.

Remaining direction: slightly-encouraging cost_mean (`constant = -weight × mean - X` for some X). Untested.

---

## Curvature Addition Sweep — Plan (2026-04-29)

**Setup**: 5 cluster runs to fill the +Curvature bar in `solver_addition_results_nc281.png`. Solver-addition convention: curvature is the *only* active cost, all other weights and constants = 0 (`drift_c=0`, `area_*=0`, `intensity_*=0`, `cohesion_*=0`, `adhesion_*=0`). `appear/disappear=50`, `merges=false`. Curvature stats: count=100732, mean=6.628, std=4.112.

**Status**: pending. **Local solves do not finish** — a single run with curvature-only on NC281 hung past 45 minutes locally without completing the gurobi solve. Submit on cluster.

### Sweep grid

| Run | curv_w | curv_c | cost_mean | cost_std |
|-----|-------:|-------:|----------:|---------:|
| R1 | 25  | -300  | -135 | 103 |
| R2 | 50  | -500  | -169 | 206 |
| R3 | 100 | -800  | -137 | 411 |
| R4 | 200 | -1500 | -176 | 822 |
| R5 | 220 | -2000 | -542 | 905 |

### Methodology note

Distinct from the prior 2026-04-28 Curvature Batch (`Curv-B1R1`–`R5`), which tested curvature on top of the post-flow-fix Full config with neutral cost_mean and showed monotonic harm. This sweep measures curvature's standalone contribution as a single addition over a no-cost baseline, mirroring how `+ Volume`, `+ Intensity`, and `+ Drift` are evaluated on the addition plot.

### Reference: MDA231 +Curvature addition (2026-04-29)

Same methodology run on MDA231 finished in ~3 min/run. Best config: `curv_w=10, curv_c=-500` → TRA=0.853, DET=0.858, LNK=0.815. Comparable to +Volume (0.853) and slightly better than +Drift (0.846). NC281 result pending cluster runs.
