# NC281-sparse-label Tracking Optimization Memory

## Optimization Targets

NC281-sparse-label has **dense GT annotations** (despite the misleading "sparse-label" dataset name — that refers to the labeling sparsity in the raw data, not the GT). Because the GT is dense, **precision and purity are reliable metrics here** and we optimize them alongside recall:

- **Primary**: TE (target_effectiveness), TF (track_fractions mean), Track Purity
- **Diagnostics**: Node Recall, Edge Recall, FP nodes, FN edges
- **Trade-offs to watch**: a config that boosts NodeR by adding many FPs is *not* an improvement here (unlike NC281-Fl2mSiH2B with sparse GT). The Track Purity column is treated as a primary signal, not just a diagnostic.

## Current Best (B1R5, 2026-04-28)

**TE=0.797, TF=0.804, NodeR=0.932, EdgeR=0.884, Purity=0.941** (vs baseline TE=0.769, TF=0.779, NodeR=0.932, EdgeR=0.880 → TE +0.028, TF +0.025)

```toml
# seg_result = "2026-04-01_11-08-44"
# flow_result = "2026-04-01_11-18-35"
drift_weight = 100.0
drift_constant = -2000.0
cohesion_weight = -500
cohesion_constant = 450
adhesion_weight = -100
adhesion_constant = 50
appear_constant = 50.0
disappear_constant = 50.0
max_edge_distance = 35
size_threshold = 20
```

## Baseline (pre-optimization)

TE=0.769, TF=0.779, NodeR=0.932, EdgeR=0.880. Same config as above with `drift_weight = 25.0`.

## Results Comparison Table (Batch 1)

| Run | Change | NodeR | EdgeR | FN_edges | TE | TF | Purity | Notes |
|-----|--------|-------|-------|----------|-----|-----|--------|-------|
| Baseline | — | 0.932 | 0.880 | — | 0.769 | 0.779 | — | Starting point |
| B1R1 | `size_threshold=0` | 0.959 | 0.905 | 737 | 0.794 | 0.804 | 0.874 | Best recall, purity drop |
| B1R2 | `seg_result=2026-04-27_17-00-54` | **0.969** | 0.906 | 725 | 0.786 | 0.797 | 0.756 | Best NodeR, big purity drop |
| B1R3 | `cohesion_constant=0` | 0.933 | 0.877 | 949 | 0.775 | 0.780 | 0.905 | ≈ baseline (no effect) |
| B1R4 | `appear/disappear=0` | 0.934 | 0.877 | 952 | 0.773 | 0.778 | 0.905 | ≈ baseline (same regime as R3) |
| **B1R5** | **`drift_weight=100`** | **0.932** | **0.884** | **894** | **0.797** | **0.804** | **0.941** | **Best TE+TF+purity** |

## Established Principles

- **drift_weight=100 (cost_std≈824) > baseline 25 (cost_std≈206).** Stronger drift discrimination = better edge selection. TE +0.028, TF +0.025, highest purity (0.941). The cleanest single-parameter win.
- **size_threshold=0 recovers small-object FNs** but at the cost of purity (more small fragments become false positives). Big NodeR/EdgeR gain (+0.027/+0.025) offset by purity drop (~0.94 → 0.874).
- **Newer segmentation (`2026-04-27_17-00-54`) gives best Node Recall (0.969) but worst purity (0.756).** Many extra detections from richer seg are false positives; the ILP picks them up.
- **cohesion_constant and appear/disappear are inactive at the current operating point.** B1R3 (cohesion_constant=0) and B1R4 (appear/disappear=0) gave near-identical metrics to each other and to baseline. The baseline values are safely in their "no-effect" regions.

## Falsified Hypotheses

- "cohesion_constant=0 shifts node cost mean ~-490 → strong global node encouragement" — falsified in B1R3, no effect
- "appear/disappear=0 → cheaper track endpoints help recall" — falsified in B1R4, no effect

## Open Questions

- **Combine B1R1 + B1R5** (size_threshold=0 + drift_weight=100): natural next batch — does the recall gain from R1 stack with the purity preservation of R5?
- **Drift weight beyond 100**: untested. cost_std at w=100 is ~824. w=150 → cost_std~1236, w=200 → cost_std~1648. Does TE plateau or grow?
- **Other parameter directions**: area, intensity, adhesion, curvature all untested for sparse-label. Curvature was harmful for both other datasets — likely harmful here too.
- **Why does the newer seg hurt purity so much?** Worth investigating whether the extra detections are specific cell types or artifacts.

## Graph Attribute Statistics (from B1R5, 2026-04-28)

| Attribute | Count | Mean | Std | Weight | Constant | Cost Mean | Cost Std |
|-----------|-------|------|-----|--------|----------|-----------|----------|
| cohesion | 8174 | 0.981 | 0.086 | -500.0 | 450.0 | -32.9 | 86.0 |
| adhesion | 8174 | 0.920 | 0.228 | -100.0 | 50.0 | -44.8 | 28.0 |
| drift_dist | 21398 | 9.304 | 8.243 | 100.0 | -2000.0 | -1069.6 | **824.3** |
| area_diff | 21398 | 0.375 | 0.351 | 0.0 | 0.0 | 0.0 | 0.0 |
| intensity_diff | 21398 | 58.912 | 73.048 | 0.0 | 0.0 | 0.0 | 0.0 |
| curvature | 58090 | 14.666 | 10.272 | 0.0 | 0.0 | 0.0 | 0.0 |

**Notes:**
- Drift cost_std=824 at w=100 is in the "active" regime (≥900 threshold per CLAUDE.md is approximate; 824 clearly works here).
- All other attributes off (weight=0). Curvature std=10.272 → target weight ~88 for cost_std=900, untested.
