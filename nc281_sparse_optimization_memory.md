# NC281-sparse-label Tracking Optimization Memory

## Optimization Targets

NC281-sparse-label has **dense GT annotations** (despite the misleading "sparse-label" dataset name — that refers to the labeling sparsity in the raw data, not the GT). Because the GT is dense, **precision and purity are reliable metrics here** and we optimize them alongside recall:

- **Primary**: TE (target_effectiveness), TF (track_fractions mean), Track Purity
- **Diagnostics**: Node Recall, Edge Recall, FP nodes, FN edges
- **Trade-offs to watch**: a config that boosts NodeR by adding many FPs is *not* an improvement here (unlike NC281-Fl2mSiH2B with sparse GT). The Track Purity column is treated as a primary signal, not just a diagnostic.

## Current Best (B2R1, 2026-04-29)

**TE=0.827, TF=0.838, NodeR=0.959, EdgeR=0.914, Purity=0.923** (vs baseline TE=0.769, TF=0.779 → TE +0.058, TF +0.059; vs B1R5 → TE +0.030, TF +0.034, Purity -0.018)

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
size_threshold = 0
```

## Prior Best (B1R5, 2026-04-28)

TE=0.797, TF=0.804, NodeR=0.932, EdgeR=0.884, Purity=0.941. Same config as B2R1 but with `size_threshold = 20`.

## Baseline (pre-optimization)

TE=0.769, TF=0.779, NodeR=0.932, EdgeR=0.880. Same config as above with `drift_weight = 25.0`, `size_threshold = 20`.

## Results Comparison Table

| Run | Change | NodeR | EdgeR | FN_edges | TE | TF | Purity | Notes |
|-----|--------|-------|-------|----------|-----|-----|--------|-------|
| Baseline | — | 0.932 | 0.880 | — | 0.769 | 0.779 | — | Starting point |
| B1R1 | `size_threshold=0` | 0.959 | 0.905 | 737 | 0.794 | 0.804 | 0.874 | Best recall (B1), purity drop |
| B1R2 | `seg_result=2026-04-27_17-00-54` | 0.969 | 0.906 | 725 | 0.786 | 0.797 | 0.756 | Best NodeR (B1), big purity drop |
| B1R3 | `cohesion_constant=0` | 0.933 | 0.877 | 949 | 0.775 | 0.780 | 0.905 | ≈ baseline (no effect) |
| B1R4 | `appear/disappear=0` | 0.934 | 0.877 | 952 | 0.773 | 0.778 | 0.905 | ≈ baseline (same regime as R3) |
| B1R5 | `drift_weight=100` | 0.932 | 0.884 | 894 | 0.797 | 0.804 | 0.941 | Best TE+TF+purity (B1) |
| **B2R1** | **`size_threshold=0` + `drift_weight=100`** | **0.959** | **0.914** | — | **0.827** | **0.838** | **0.923** | **NEW BEST: stack worked** |
| B2R2 | `drift_weight=150` | — | — | — | — | — | — | TIMEOUT 16h walltime |
| B2R3 | `size_threshold=10` + `drift_weight=100` | 0.951 | 0.905 | — | 0.814 | 0.827 | 0.934 | Dominated by B2R1 |
| B2R4 | + `area_weight=2500, area_constant=-1500` | 0.931 | 0.876 | — | 0.779 | 0.787 | 0.920 | harmful (-0.018 TE vs B1R5) |
| B2R5 | + `intensity_weight=12, intensity_constant=-800` | 0.932 | 0.883 | — | 0.789 | 0.801 | 0.935 | neutral / slightly worse |

## Established Principles

- **size_threshold=0 + drift_weight=100 stacks** — recall gain from threshold combines with edge-selection improvement from drift, net +0.030 TE / +0.034 TF over B1R5 with only -0.018 purity. Confirmed in B2R1.
- **drift_weight=100 (cost_std≈824) > baseline 25 (cost_std≈206).** Stronger drift discrimination = better edge selection. The cleanest single-parameter win at original baseline.
- **size_threshold=0 recovers small-object FNs** but at the cost of purity. Big NodeR/EdgeR gain (+0.027/+0.025 in B1R1) offset by purity drop. With drift_weight=100 (B2R1), the purity penalty is smaller (-0.018 vs B1R5).
- **size_threshold=10 is dominated by 0** — B2R3 had better purity than B2R1 (0.934 vs 0.923) but worse TE/TF (0.814 vs 0.827). The middle-ground threshold is not a useful operating point.
- **Newer segmentation (`2026-04-27_17-00-54`) gives best Node Recall (0.969) but worst purity (0.756).** Many extra detections from richer seg are false positives; the ILP picks them up.
- **cohesion_constant and appear/disappear are inactive at the current operating point.** B1R3, B1R4 confirmed.
- **Area and intensity edge costs are harmful at the current operating point.** B2R4 (area) and B2R5 (intensity) both reduced TE vs B1R5. Mirrors NC281-Fl2m NC1-R2 / NC1-R3 results — additional edge discrimination beyond drift is consistently bad on NC281-style datasets.
- **Drift solve time grows with weight.** B2R2 (drift_weight=150) timed out at 16h walltime. To test drift past 100, need ≥24h walltime.

## Falsified Hypotheses

- "cohesion_constant=0 shifts node cost mean ~-490 → strong global node encouragement" — falsified in B1R3, no effect
- "appear/disappear=0 → cheaper track endpoints help recall" — falsified in B1R4, no effect
- "size_threshold=10 preserves recall without B1R1's purity hit" — partially supported but dominated by B2R1 (size_threshold=0 + drift_weight=100). Not a useful operating point.
- "Area cost (cost_std≈900, std/mean=0.94) improves edge selection" — falsified in B2R4, harmful (-0.018 TE)
- "Intensity cost (cost_std≈900, std/mean=1.24) improves edge selection" — falsified in B2R5, neutral-to-slightly-harmful

## Open Questions

- **Drift weight beyond 100**: B2R2 timed out. Needs ≥24h walltime to test, or accept that drift_weight=100 is the practical operating point.
- **Curvature**: only untested edge attribute on this dataset. Was harmful on both other datasets, but a neutral or slightly-encouraging variant could differ. Lower priority given area/intensity also harmful.
- **Why does the newer seg hurt purity so much?** Still open — worth investigating whether the extra detections are specific cell types or artifacts.
- **Adhesion at higher weight**: untested. Currently at -100 (cost_std=28). Target weight for cost_std=900 is ~3947. Marginal spread (std/mean=0.248) suggests limited upside.

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
