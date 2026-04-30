# NC281-Fl2mSiH2B Tracking Optimization Memory

## Post-Fix Current Best (2026-04-15, flow unit fix + confidence filter)

**TE: 0.6616, TF: 0.7057, Edge Recall: 0.8416, FN Edges: 73**

```toml
drift_weight = 30.0
drift_constant = -1000.0
cohesion_weight = -500.0
cohesion_constant = 500.0
appear_constant = 50.0
disappear_constant = 50.0

# Confidence-based Z flow filtering
z_flow_conf_threshold = 1.0e-7   # ~5% of nodes flagged unreliable
z_flow_min_pass_pixels = 10
```

**Gap vs pre-fix best (0.692 vs 0.6616, -0.030) partially closed via confidence filtering.** Light filtering (~5% unreliable) is the sweet spot.

### Previous Post-Fix Best (no confidence filter, 2026-04-15)

TE=0.6551, TF=0.6893, ER=0.8373, FN=75 (drift_w=30, drift_c=-1000, coh_w=-500, coh_c=500)

## Pre-Fix Best Config (archived for reference)

**TE: 0.692, TF: 0.744, Edge Recall: 0.861, FN Edges: 64** (NC8-R4)
Same as above but with drift_weight = 25.0.

Previous best (drift_c=-2000): TE=0.690, TF=0.734, ER=0.852, FN=68.
Old best (pre-semantics change): TE=0.703, TF=0.738 with coh_w=500, coh_c=-450 (old code).

## Flow Unit Fix Impact (2026-04-15)

Voxel scale: Z=2.11, Y=X=0.65 μm (Z/XY ≈ 3.25).

Pre-fix graph stats: drift_dist mean=8.462, std=5.132
Post-fix graph stats: drift_dist mean=6.451, std=4.198

NC281 cells move meaningfully in Z, so Z-flow rescaling materially affected drift_dist — unlike MDA231 which was a no-op.

### Post-Fix Batch 1 Results (6 runs)

| Run | drift_w | drift_c | TE | TF | ER | FN |
|-----|---------|---------|-----|-----|-----|-----|
| UFIX-Baseline | 25 | -1000 | 0.6443 | 0.6722 | 0.8308 | 78 |
| **UFIX-B1R1** | **30** | **-1000** | **0.6551** | 0.6883 | 0.8351 | 76 |
| UFIX-B1R2 | 35 | -1000 | 0.6551 | 0.6893 | 0.8373 | 75 |
| UFIX-B1R3 | 25 | -750 | 0.6529 | 0.6861 | 0.8351 | 76 |
| UFIX-B1R4 | 30 | -750 | 0.6551 | 0.6893 | 0.8373 | 75 |
| UFIX-B1R5 | 25 | -1250 | 0.6421 | 0.6715 | 0.8351 | 76 |

### Post-Fix Principles

- **New drift plateau: drift_w ∈ [30, 35], drift_c ∈ [-1000, -750].** Previously drift_w=25 was optimal.
- drift_w=25 now too low — drift_dist std shrank 18%, so need higher weight for same discrimination.
- drift_c=-1250 too strong (same direction as pre-fix — plateau boundary shifted little).
- TE plateau at 0.6551 after drift re-opt. Still 0.037 below pre-fix best.

### Post-Fix Batch 2 (cohesion sweep) Results

| Run | coh_w | coh_c | TE | TF | ER | FN |
|-----|-------|-------|-----|-----|-----|-----|
| B1 best | -500 | 450 | 0.6551 | 0.6883 | 0.8351 | 76 |
| B2R1 | -500 | 400 | 0.6529 | 0.6852 | 0.8308 | 78 |
| **B2R2** | **-500** | **500** | **0.6551** | **0.6893** | **0.8373** | **75** |
| B2R3 | 0 | 0 | 0.6529 | 0.6871 | 0.8330 | 77 |
| B2R4 | -1000 | 900 | 0.6508 | 0.6778 | 0.8330 | 77 |
| B2R5 | -250 | 225 | 0.6551 | 0.6883 | 0.8351 | 76 |

### Confirmed Principles (post-fix)

- **TE plateau at 0.6551 is robust** — unchanged across 5 cohesion variants and 3 different drift_w/drift_c combinations.
- **Cohesion magnitude insensitive** in [-250, -500] when cost mean near zero.
- **coh_c=500 marginally better than 450** (FN_Edges 76 → 75).
- **Scaling cohesion up (coh_w=-1000) actively hurts** — dropped TE to 0.6508.
- **Cohesion off is slightly worse** than tuned — contributes small but real value.

### Open Questions

- **Gap vs pre-fix (-0.037 TE) likely structural, not tuning-addressable.** Hypothesis: old optimum was partially lucky — buggy drift_dist inflated Z distances, happening to reward GT-aligned edges in this dataset. Post-fix drift is physically correct but selects a different valid subset. Sparse GT makes TE sensitive to which specific edges are chosen.
- **Remaining untested**: adhesion under new drift regime (always harmful pre-fix), appear/disappear fine-tuning, area/intensity (still untested post-fix). **Curvature now confirmed harmful post-fix with neutral cost_mean** (see Curvature Re-test below).
- Could a different matcher threshold help? threshold=10 was tuned pre-fix.
- **Curvature batch — slightly encouraging variant still untested.** Neutral cost_mean sweep was harmful (below). The remaining direction is `constant = -weight × curvature_mean - X` for some X, which would add a small global edge-encouragement on top of curvature discrimination.

---

## Curvature Re-test, neutral cost_mean (2026-04-28)

Tested curvature on top of post-flow-fix best (TE=0.6616, TF=0.7057, NodeR=1.000, EdgeR=0.8416). Sweep used neutral cost_mean (`curvature_constant = -weight × 6.628`) so curvature only discriminates without globally penalizing edges.

### Results

| Run | curv_w | curv_c | TE | TF | NodeR | EdgeR |
|-----|--------|--------|-----|-----|-------|-------|
| **Baseline** | 0 | 0 | **0.6616** | **0.7057** | **1.000** | **0.8416** |
| Curv-R1 | 50 | -331.4 | 0.6551 | 0.6830 | 1.000 | 0.8395 |
| Curv-R2 | 100 | -662.8 | 0.6074 | 0.6463 | 0.998 | 0.8265 |
| Curv-R3 | 200 | -1325.6 | 0.5944 | 0.6347 | 0.996 | 0.7918 |
| Curv-R4 | 400 | -2651.2 | 0.6009 | 0.6367 | 0.996 | 0.8048 |
| Curv-R5 | 800 | -5302.4 | 0.5900 | 0.6281 | 0.994 | 0.7766 |

### Conclusions

- **Curvature with neutral cost_mean is monotonically harmful.** TE drops -0.0065 at w=50 and bottoms at -0.072 at w=800. EdgeR drops 0.84 → 0.78. TF drops similarly.
- **Edge recall regression suggests curvature is filtering out valid GT edges** (cells turning, dividing, etc.) along with truly curvy bad trajectories.
- **Strengthens prior pre-fix falsified hypothesis**: "Curvature improves linking" — confirmed harmful post-fix at proper cost_std too. Was previously written off due to tiny cost_std (20.6 at w=5); now tested at cost_std up to 3290 (w=800) and still bad.
- **No curvature_weight tested helps NC281-Fl2m at the current operating point.** Remaining curvature variant (slightly encouraging cost_mean) is the only untested form.

## Results Comparison Table

| Run | Key Changes | TE | TF | ER | FN Edges |
|-----|-------------|-----|-----|-----|----------|
| **NC8-R4 (best)** | **drift_c=-1000** | **0.692** | **0.744** | **0.861** | **64** |
| NC8-R3 | drift_c=-900 | 0.692 | 0.744 | 0.857 | 66 |
| NC7-R5 | drift_c=-750 | 0.692 | 0.744 | 0.857 | 66 |
| NC5-R7 | coh_w=-500, coh_c=450 (drift_c=-2000) | 0.690 | 0.734 | 0.852 | 68 |
| NC8-R2 | drift_c=-600 | 0.679 | 0.732 | 0.850 | 69 |
| NC8-R1 | drift_c=-500 | 0.675 | 0.719 | 0.848 | 70 |
| NC8-R5 | drift_c=-1250 | 0.668 | 0.723 | 0.857 | 66 |
| NC7-R1 | drift_c=-250 | 0.592 | 0.658 | 0.794 | 95 |

## Graph Attribute Statistics (with num_leaves scaling)

| Attribute | Count | Mean | Std | Notes |
|-----------|-------|------|-----|-------|
| cohesion | 11941 | 0.950 | 0.098 | Low spread. At w=-500: cost std=302 (leaves amplify) |
| adhesion | 11941 | 0.595 | 0.411 | Better spread but harmful in both directions |
| drift_dist | 34717 | 8.462 | 5.132 | Main lever. At w=25: cost std=128 |
| area_diff | 34717 | 0.523 | 0.383 | Harmful at any scale |
| intensity_diff | 34717 | 74.573 | 66.424 | Harmful at any scale |
| curvature | 100732 | 6.628 | 4.112 | Harmful; constant dominates (cost std=20.6 at w=5) |

## Established Principles

- **drift_c=-1000 optimal** — TE plateau at 0.692 from drift_c=-750 to -1000 (cost mean -538 to -788). Below -750 TE drops; above -1000 TE also drops. drift_c=-1000 gives best ER (0.861).
- **Cost balance strategy works** — reducing drift_c from -2000 to -1000 improved discrimination ratio (gap/|bad_cost|) while keeping all edges negative. This partially addresses "two bad edges > one good edge" problem.
- **drift_w=25-30 optimal** — plateau confirmed with new cohesion. w=20 snaps to worse state, w=35+ degrades TE.
- **Cohesion: negative weight, constant ≈ 450** — cost mean near zero maximizes discrimination. Scaling up weight doesn't help (leaves already amplify).
- **Adhesion harmful in both directions** — positive kills node selection, negative rewards wrong merges. Keep off.
- **Area and intensity harmful as primary costs AND as safety nets** — confirmed across 8+ runs as primary costs, and 3 runs as break-even safety nets with drift_c=-250.
- **Curvature harmful** — cost std=20.6 at w=5, constant dominates. 100k edge pairs adds noise.
- **appear/disappear insensitive below 100, harmful at 150+** — 50 is fine.
- **NC281 benefits from LESS discrimination** — opposite of MDA231.
- **num_leaves scaling matters** — cohesion cost std jumped 49→302 with leaves. Always check scaled stats.

## Falsified Hypotheses

- Scaling cohesion to cost std~900 helps → no effect (NC5-R2, R3, R9)
- Positive adhesion penalizes merges → kills node selection (NC5-R4)
- Negative adhesion complements negative cohesion → harmful (NC5-R5)
- Curvature improves linking with new cohesion → very harmful (NC6-R5)
- drift_w interaction with new cohesion shifts optimum → no, still 25-30 (NC6-R1, R2)
- Aggressive break-even (drift_c=-250) with area/intensity safety nets recovers edges → safety nets help nodes but TE stays ~0.1 below baseline (NC7-R1 through R4)
- drift_c=-1250 extends the plateau → TE drops to 0.668 (NC8-R5)

## Open Questions

- Gap from old baseline (0.703 vs 0.690) — may be inherent to new semantics. Old coh_w=500, coh_c=-450 had a different cost profile that cannot be replicated.
- Could different segmentation or flow results help more than parameter tuning?
- Are there edge cost formulations beyond drift/area/intensity/curvature worth trying?
- **Curvature as a solver-addition (single-cost over no-cost baseline) — complete.** See "Curvature Addition Sweep — Results" below. +Curvature bar value: TE ≈ 0.617 (R3, curv_w=100, curv_c=-800).

## Curvature Addition Sweep — Results (2026-04-30)

**Status**: complete. Submitted 2026-04-29 via `launch_curvature_batch.py` on cluster (`nc281curvadd_2026-04-29_16-49-06`). Standalone curvature only — all other weights/constants = 0; `appear/disappear=50`, `merges=false`. Base config: `scripts/04_tracking/NC281_curvature_addition_baseline.toml`.

For the +Curvature bar on `solver_addition_results_nc281.png`. Distinct from the prior 2026-04-28 Curvature Batch (curvature on top of Full config with neutral cost_mean — harmful).

Curvature stats: count=100732, mean=6.628, std=4.112.

### Results

| Run | curv_w | curv_c | cost_mean | cost_std | NodeR | EdgeR | TE | TF | Notes |
|-----|-------:|-------:|----------:|---------:|------:|------:|-----:|-----:|-------|
| R1 | 25  | -300  | -135 | 103 | — | — | — | — | TIMEOUT 24h |
| R2 | 50  | -500  | -169 | 206 | — | — | — | — | TIMEOUT 24h |
| **R3** | **100** | **-800**  | **-137** | **411** | **0.986** | **0.805** | **0.616** | **0.649** | **canonical +Curvature bar** |
| R4 | 200 | -1500 | -176 | 822 | 0.980 | 0.798 | 0.618 | 0.651 | tied with R3 |
| R5 | 220 | -2000 | -542 | 905 | 0.990 | 0.798 | 0.607 | 0.638 | slightly worse |

### Conclusions

- **+Curvature bar value**: TE ≈ 0.617, TF ≈ 0.650 (R3 or R4, essentially tied). Use R3 (curv_w=100, curv_c=-800) as the canonical config.
- **Low-discrimination configs (R1, R2) timed out at 24h.** Confirms the local-solve-hang note from the plan: when the only active cost has weak discrimination (cost_std < ~400), the ILP becomes intractable. R3 (cost_std=411) is the lower bound that solves in 24h.
- **TE/TF plateau at curv_w=100-200.** R3 ≈ R4; R5 (curv_w=220) slightly worse. The plateau is the meaningful curvature-only result.
- **Track purity is ~0.05** in all completed runs — expected given sparse GT (506 GT nodes vs ~7500 pred nodes).
- **Comparable to MDA231**: equivalent MDA231 sweep best (`curv_w=10, curv_c=-500`) gave TRA=0.853, DET=0.858, LNK=0.815. NC281-Fl2m's standalone curvature TE=0.617 is the parallel data point.
