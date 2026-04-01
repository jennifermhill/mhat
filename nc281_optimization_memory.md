# NC281-Fl2mSiH2B Tracking Optimization Memory

## Current Best Config (cost balance strategy, 2026-04-01)

**TE: 0.692, TF: 0.744, Edge Recall: 0.861, FN Edges: 64**

```toml
drift_weight = 25.0
drift_constant = -1000.0
area_weight = 0.0
area_constant = 0.0
intensity_weight = 0.0
intensity_constant = 0.0
curvature_weight = 0.0
curvature_constant = 0.0
cohesion_weight = -500.0
cohesion_constant = 450.0
adhesion_weight = 0.0
adhesion_constant = 0.0
appear_constant = 50.0
disappear_constant = 50.0
```

Previous best (drift_c=-2000): TE=0.690, TF=0.734, ER=0.852, FN=68.
Old best (pre-semantics change): TE=0.703, TF=0.738 with coh_w=500, coh_c=-450 (old code).

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
