# primary_nk_cells — Optimization Working Memory

Read + update this each batch. Companion to the append-only `nk_cells_optimization_log.md`.

## Objective
Find the baseline tracking-cost config for `primary_nk_cells / 01_cells` by measuring
the relative contribution of every non-curvature ILP cost (weights **and** constants),
then refining the winners. **Sparse GT** → optimize **TE** and **TF** only.

## Current best
- **B4 `size80`** — TE 0.4754, TF 0.4583, NodeR 0.866, EdgeR 0.673, FN_edg 319
  (no_cohesion base + size_threshold=80). Best so far.
- (prev) B1 `no_cohesion` — TE 0.4723, TF 0.4547 (full minus cohesion).
- (prev) B0R0 `full` — TE 0.4713, TF 0.4539 (reproduced 2026-07-09; match_thresh=10 ok).

## Graph-construction params (Phase 4, 2026-07-09)
- **max_edge_distance saturated at 40** — 50/60/80 identical to 40 (real links << 40;
  ILP never picks longer edges). Do NOT increase.
- **size_threshold optimum ≈ 80** — 20→80 helps (0.4723→0.4754); 40 is a no-op; 160
  slightly regresses (over-filters real small objects, NodeR 0.875→0.855).

## Phase 3 base (BEST3)
`full` params minus cohesion: drift 100/−1500, area 1500/−1500, intensity 25/−1000,
adhesion −1500/1500, appear/disappear 50, **cohesion 0/0**, curvature off. TE 0.4723.
Costs to refine (weight AND constant), ranked by Phase-2 contribution:
adhesion > drift > area > intensity > appear/disappear. Skip cohesion (inert).

## Setup (do not change)
- seg `2026-06-25_17-40-15`, flow `2026-06-25_17-53-34`, use_lk=false.
- Eval: metrics=[basic, track_overlap], matcher=point, match_threshold=10.
- Fixed params: max_edge_distance=40, size_threshold=20, drift_distance,
  max_children=3, merges=false, divisions=false, min/max_merge_cost, z_flow_*.
- Curvature excluded (0/0) for the whole study.
- Ablate = zero both weight and constant; appear/disappear ablate = constant 0.

## Priors from other sparse-GT datasets (NC281) — TESTED on nk_cells 2026-07-09, mostly FALSIFIED
- NC281 said drift dominates → here drift helps but is NOT dominant; `drift_only` is the
  WORST config (0.3566). Multi-cost combo is essential; every cost except cohesion helps.
- NC281 said area/intensity harmful → here both are BENEFICIAL (no_area −0.029, no_int
  −0.006; add_area is the strongest standalone add). Prior falsified for nk_cells.
- adhesion is the single most helpful cost (no_adhesion −0.040). Note removing it RAISES
  NodeR (0.875→0.908) but lowers TE → it usefully suppresses spurious nodes.
- cohesion: inert here (attribute std 0.031). add_cohesion == drift_only exactly. Drop it.
- appear/disappear: mild help (no_appeardisappear −0.005). Untested above 50.

## Weight calibration rule (for Phase 3)
From the `--stats-only` attribute table: a cost only influences the ILP when its cost
std is ~400–2000. `target_weight ≈ target_cost_std / attribute_std`. Keep edge-cost
means negative (encouraging) so links get selected. Record the table here once run.

## Attribute-stats table (from --stats-only on `full`, recorded 2026-07-09)
Graph: 38542 nodes, 110883 edges, 319721 curvature edge-pairs.
Z flow: 42.4% of nodes flagged unreliable → XY-only drift_dist.
Columns are at the `full` weights/constants; cost std = current influence on the ILP.
| attribute | mean | std | current w / c | cost mean | cost std | target w for cost_std≈900 |
|---|---|---|---|---|---|---|
| drift_dist     | 10.750 | 7.873  | 100 / -1500   | -425.0 | 787.3  | ≈114 |
| area_diff      | 0.813  | 0.597  | 1500 / -1500  | -280.7 | 896.1  | ≈1508 |
| intensity_diff | 23.454 | 48.880 | 25 / -1000    | -413.6 | 1222.0 | ≈18 |
| cohesion       | 0.994  | 0.031  | -500 / 500    | 11.6   | 68.3   | ≈29000 (near-flat; weak lever) |
| adhesion       | 0.571  | 0.485  | -1500 / 1500  | 1005.8 | 1837.7 | ≈1856 |

Read: at `full` weights the dominant node cost is **adhesion** (cost std 1838, and cost
mean +1006 → discouraging), then **intensity** (1222), **area** (896), **drift** (787).
**cohesion is near-inert** (std 0.031 → cost std only 68) — expect little effect from
ablating/tuning it. adhesion's large positive cost mean is a strong suspect for
suppressing node selection (low-ish NodeR 0.875, EdgeR 0.668 at baseline).

## Open questions
- Does `match_threshold=10` reproduce B0R0 (0.4713/0.4539)? If not, adjust before
  trusting sweep numbers.
- Any EMPTY solutions among ablation conditions? (all-positive-cost configs can select
  nothing — flagged by collect_nk_metrics.py.)

## Next action
Phase 2 batch B1 (12 conditions) on the cluster → collect → rank → design Phase 3.
