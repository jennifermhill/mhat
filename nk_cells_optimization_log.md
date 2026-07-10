# primary_nk_cells / 01_cells — Tracking Cost Optimization Log

Append-only log of the cost-contribution sweep. Sparse ground truth → optimize
**TE (target_effectiveness)** and **TF (track_fractions mean)**; precision / purity /
F1 are unreliable (unannotated objects appear as false positives) and are NOT targets.
Diagnostics only: Node Recall, Edge Recall, FN edges.

Pipeline: seg `2026-06-25_17-40-15`, flow `2026-06-25_17-53-34`. Eval: basic +
track_overlap, point matcher, match_threshold=10. Curvature excluded from the study.
Fixed (never swept): max_edge_distance=40, size_threshold=20, drift_distance,
max_children, merges=false, divisions, min/max_merge_cost, z_flow_*.

Log entry format:
```
### B{batch}R{run}: [param change]
exp_uid (condition name): <name>
Hypothesis: "[testable prediction]"
TE: X, TF: X, Node_Recall: X, Edge_Recall: X, FN_edges: X
Verdict: [supported/falsified/inconclusive] — [explanation]
```

---

## Baseline (initial run, pre-sweep)

### B0R0: initial primary_nk_cells config (full, all costs on)
exp_uid: 2026-06-29_16-24-53  (== condition `full`, to be reproduced)
Config: drift 100/−1500, area 1500/−1500, intensity 25/−1000, cohesion −500/500,
adhesion −1500/1500, appear/disappear 50, curvature off.
TE: 0.4713, TF: 0.4539, Node_Recall: 0.875, Edge_Recall: 0.668, FN_edges: 324
Notes: Edge Recall low (linking problem). area + intensity are "on" here despite prior
sparse-GT campaigns (NC281) finding them harmful — Phase 2 tests whether they help.

---

## Phase 2 — Ablation + Addition (batch B1)

Two reads of each cost's contribution. Ablation drops one cost from `full`; addition
adds one cost onto `drift_only`. Fill TE/TF/recalls from `collect_nk_metrics.py`.

Ran 2026-07-09 on the cluster (all 11 tracking jobs exit 0, no 120-quirk). `full`
reproduced B0R0 exactly → match_threshold=10 validated. Results sorted by TE:

| condition | TE | TF | dTE(vs full) | NodeR | EdgeR | FN_edg |
|---|---|---|---|---|---|---|
| no_cohesion        | 0.4723 | 0.4547 | +0.0010 | 0.875 | 0.669 | 323 |
| **full**           | 0.4713 | 0.4539 |  0.0000 | 0.875 | 0.668 | 324 |
| no_appeardisappear | 0.4662 | 0.4485 | −0.0051 | 0.878 | 0.664 | 328 |
| no_intensity       | 0.4652 | 0.4467 | −0.0061 | 0.841 | 0.651 | 341 |
| no_area            | 0.4426 | 0.4267 | −0.0287 | 0.802 | 0.619 | 372 |
| no_drift           | 0.4355 | 0.4197 | −0.0359 | 0.790 | 0.613 | 378 |
| add_area           | 0.4324 | 0.4157 | −0.0389 | 0.907 | 0.647 | 345 |
| no_adhesion        | 0.4314 | 0.4149 | −0.0400 | 0.908 | 0.662 | 330 |
| add_adhesion       | 0.3904 | 0.3765 | −0.0809 | 0.706 | 0.525 | 464 |
| add_intensity      | 0.3811 | 0.3682 | −0.0902 | 0.899 | 0.629 | 362 |
| add_cohesion       | 0.3566 | 0.3446 | −0.1148 | 0.890 | 0.597 | 393 |
| drift_only         | 0.3566 | 0.3446 | −0.1148 | 0.890 | 0.597 | 393 |

### B1: Ablation entries (drop one cost from full; dTE vs full)
- **no_cohesion** — TE 0.4723 (+0.0010). Verdict: cohesion **inert / marginally harmful** — best-of-batch but within noise of full. Consistent with stats table (cohesion attribute std 0.031 → cost std 68, essentially no ILP influence). → drop from base.
- **no_appeardisappear** — TE 0.4662 (−0.0051). Verdict: appear/disappear **helps slightly**. Keep.
- **no_intensity** — TE 0.4652 (−0.0061). Verdict: intensity **helps** (NodeR 0.875→0.841 when removed). Keep.
- **no_area** — TE 0.4426 (−0.0287). Verdict: area **helps substantially**. Keep.
- **no_drift** — TE 0.4355 (−0.0359). Verdict: drift **helps substantially**. Keep.
- **no_adhesion** — TE 0.4314 (−0.0400). Verdict: adhesion is the **single most helpful cost** (largest drop when removed), even though removing it *raises* NodeR (0.875→0.908) — it suppresses spurious nodes that hurt track coverage. Keep.

### B1: Addition entries (add one cost onto drift_only; Δ vs drift_only 0.3566)
- **add_area** — TE 0.4324 (Δ +0.0758 vs drift_only). Verdict: area is the **strongest standalone add**.
- **add_adhesion** — TE 0.3904 (Δ +0.0338). Verdict: adhesion adds value on its own.
- **add_intensity** — TE 0.3811 (Δ +0.0245). Verdict: intensity adds modest value.
- **add_cohesion** — TE 0.3566 (Δ +0.0000, *identical* to drift_only). Verdict: cohesion contributes **nothing** — confirms inertness from both directions.

### B1 Batch summary — cost contribution ranking
Contribution to TE (ablation drop, larger = more helpful):
**adhesion (0.040) > drift (0.036) > area (0.029) > intensity (0.006) > appear/disappear (0.005) ≫ cohesion (−0.001, inert)**.
Addition ranking agrees on ordering of the strong costs (area > adhesion > intensity ≫ cohesion).

**Key cross-dataset finding:** unlike the NC281 sparse-GT prior (drift dominates; area/intensity harmful), on primary_nk_cells **every cost except cohesion helps**, and `full` (all costs) is near-optimal. `drift_only` (0.3566) is the *worst* config — drift alone is weak here; the multi-cost combination is essential. area and intensity are **beneficial, not harmful**. Prior falsified for this dataset.

**Best config after Phase 2:** `no_cohesion` (TE 0.4723 / TF 0.4547) — `full` with cohesion ablated. This is the Phase 3 base (`BEST3`).

---

## Phase 3 — Coordinate-wise refinement (batches B2+)

_(added after Phase 2; sweep weight AND constant of the costs that helped, ~5 values
each, calibrated from the --stats-only attribute table)_

---

## Phase 4 — Graph-construction sweep: max_edge_distance & size_threshold (batch B4)

Ran 2026-07-09 on the cluster (user request; increasing only). Coordinate-wise from
base = `no_cohesion` (TE 0.4723). Baseline point 40/20 == `no_cohesion`, not repeated.
All 6 tracking jobs exit 0. dTE below is vs `no_cohesion` (0.4723).

| condition | max_edge_dist | size_thresh | TE | TF | dTE | NodeR | EdgeR | FN_edg |
|---|---|---|---|---|---|---|---|---|
| **size80**  | 40 | 80  | **0.4754** | **0.4583** | **+0.0031** | 0.866 | 0.673 | 319 |
| size160     | 40 | 160 | 0.4744 | 0.4574 | +0.0021 | 0.855 | 0.669 | 323 |
| no_cohesion | 40 | 20  | 0.4723 | 0.4547 |  0.0000 | 0.875 | 0.669 | 323 |
| edge50      | 50 | 20  | 0.4723 | 0.4547 |  0.0000 | 0.875 | 0.669 | 323 |
| edge60      | 60 | 20  | 0.4723 | 0.4547 |  0.0000 | 0.875 | 0.669 | 323 |
| edge80      | 80 | 20  | 0.4723 | 0.4547 |  0.0000 | 0.875 | 0.669 | 323 |
| size40      | 40 | 40  | 0.4723 | 0.4547 |  0.0000 | 0.872 | 0.669 | 323 |

### B4 findings
- **max_edge_distance is saturated at 40.** 50/60/80 give byte-identical metrics to the
  40 baseline. drift_dist mean 10.8 (std 7.9) → real links are well within 40; widening
  the radius only adds high-cost candidate edges the ILP never selects. Do not increase.
- **size_threshold helps, optimum ≈ 80.** 20→80 raises TE 0.4723→**0.4754** (TF→0.4583,
  FN_edges 323→319) by filtering small spurious fragments before graph construction.
  20→40 is a no-op; 80→160 slightly regresses (0.4744, NodeR 0.875→0.855) as real small
  objects start getting filtered. Sweet spot is 80.

**New overall best: `size80`** — TE 0.4754 / TF 0.4583 (base = no_cohesion + size_threshold=80).
