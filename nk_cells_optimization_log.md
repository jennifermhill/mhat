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

---

## Phase 5 — Re-evaluation on the CORRECTED ground truth (2026-07-30)

**Everything above this line is scored against a ground truth that was wrong.** The GT was
corrected on 2026-07-30: `correct_tracks.zarr` went from **1011 nodes / 976 edges** to
**1070 / 1029** (+59 nodes, +53 edges). The old version is preserved as
`correct_tracks_old.zarr`. All 22 existing runs (21 MHAT + `ultrack`) were re-evaluated
against the corrected GT with the same eval settings as before — `basic` + `track_overlap`,
point matcher, `match_threshold = 10` — so the Phase 5 numbers are directly comparable to
each other, and the Phase 0–4 numbers are directly comparable to each other, but **the two
sets are not comparable across the line**.

Re-eval submitted with the new `scripts/05_evaluation/launch_evals.py`
(spec `configs/sweeps/nk_cells_reeval_corrgt.toml`, `reeval_id = corrgt0730`); manifest and
`results.csv` under
`experiments/tracking/primary_nk_cells/01_cells/reevals/corrgt0730/`.

Full table, sorted by TE. `old TE` is the same run's Phase 0–4 number for reference only.

| condition | TE | TF | NodeR | EdgeR | FN_edg | old TE |
|---|---|---|---|---|---|---|
| **2026-07-27_14-10-54** | **0.6706** | **0.6696** | 0.9664 | **0.8669** | 137 | — |
| size160             | 0.6638 | 0.6607 | 0.9561 | 0.8601 | 144 | 0.4744 |
| size80              | 0.6608 | 0.6568 | 0.9589 | 0.8620 | 142 | 0.4754 |
| no_cohesion         | 0.6531 | 0.6466 | 0.9551 | 0.8542 | 150 | 0.4723 |
| edge50              | 0.6531 | 0.6466 | 0.9551 | 0.8542 | 150 | 0.4723 |
| edge60              | 0.6531 | 0.6466 | 0.9551 | 0.8542 | 150 | 0.4723 |
| edge80              | 0.6531 | 0.6466 | 0.9551 | 0.8542 | 150 | 0.4723 |
| size40              | 0.6531 | 0.6466 | 0.9551 | 0.8542 | 150 | 0.4723 |
| full                | 0.6521 | 0.6458 | 0.9551 | 0.8523 | 152 | 0.4713 |
| 2026-06-29_16-24-53 | 0.6521 | 0.6458 | 0.9551 | 0.8523 | 152 | 0.4713 |
| no_intensity        | 0.6385 | 0.6252 | 0.9243 | 0.8328 | 172 | 0.4652 |
| no_appeardisappear  | 0.6307 | 0.6281 | 0.9598 | 0.8494 | 155 | 0.4662 |
| 2026-07-27_13-33-39 | 0.6132 | 0.6079 | 0.8729 | 0.7765 | 230 | 0.4088 |
| ultrack             | 0.6064 | 0.6119 | 0.9402 | 0.8183 | 187 | — |
| no_adhesion         | 0.5821 | 0.5854 | 0.9832 | 0.8222 | 183 | 0.4314 |
| add_area            | 0.5695 | 0.5666 | 0.9822 | 0.8144 | 191 | 0.4324 |
| no_area             | 0.5471 | 0.5375 | 0.8607 | 0.7347 | 273 | 0.4426 |
| no_drift            | 0.5452 | 0.5415 | 0.8393 | 0.7269 | 281 | 0.4355 |
| add_intensity       | 0.5073 | 0.5161 | 0.9813 | 0.7775 | 229 | 0.3811 |
| add_cohesion        | 0.4626 | 0.4722 | 0.9692 | 0.7318 | 276 | 0.3566 |
| drift_only          | 0.4626 | 0.4722 | 0.9692 | 0.7318 | 276 | 0.3566 |
| add_adhesion        | 0.4140 | 0.4071 | 0.7449 | 0.5802 | 432 | 0.3904 |

### Phase 5 — re-derived cost contribution

Ablation (drop one cost from `full` = 0.6521; larger drop = more helpful):

| cost | new dTE | old dTE | rank: new → old |
|---|---|---|---|
| drift              | −0.1069 | −0.0359 | 1 → 2 |
| area               | −0.1050 | −0.0287 | 2 → 3 |
| adhesion           | −0.0700 | −0.0400 | 3 → **1** |
| appear/disappear   | −0.0214 | −0.0051 | 4 → 5 |
| intensity          | −0.0136 | −0.0061 | 5 → 4 |
| cohesion           | +0.0010 | +0.0010 | inert → inert |

Addition (add one cost onto `drift_only` = 0.4626):

| cost | new dTE | old dTE |
|---|---|---|
| area      | +0.1069 | +0.0758 |
| intensity | +0.0447 | +0.0245 |
| cohesion  | +0.0000 | +0.0000 |
| adhesion  | **−0.0486** | **+0.0338** |

### What survived the correction and what did not

**Survived:**
- **cohesion is inert.** Ablation +0.0010 (identical to before), addition exactly +0.0000, and
  `add_cohesion` is still byte-identical to `drift_only`. Expected — the attribute has std
  0.031, a property of the candidate graph and nothing to do with the GT. Keep it at 0/0.
- **`max_edge_distance` is saturated at 40.** `edge50`/`edge60`/`edge80` are still byte-identical
  to the 40 baseline. Widening remains pointless; narrowing is still untested.
- **`drift_only` is a weak configuration** and the multi-cost combination is essential.
- **`full` is near-optimal among the cost sets tested** — no single-cost ablation beats it
  except the inert `no_cohesion`.

**Falsified:**
- **"adhesion is the single most helpful cost" is wrong.** It drops to 3rd. **drift and area are
  now the top two and are effectively tied** (−0.1069 vs −0.1050), each roughly 1.5× adhesion.
- **`add_adhesion` flipped sign.** On the old GT adhesion-alone improved on `drift_only`
  (+0.0338); on the corrected GT it is the **worst configuration in the whole study** (0.4140,
  −0.0486 below `drift_only`, NodeR collapsing to 0.745). Adhesion is useful only in combination.
- **The `size_threshold` optimum moved the opposite way to expectation.** The old GT put it at
  80 with 160 regressing; corrected, **160 (0.6638) beats 80 (0.6608) beats 40/20 (0.6531)**.
  Adding 59 GT nodes made *more* aggressive small-fragment filtering better, not less. 160 is
  the top of the tested range, so the optimum is **not bracketed above**.
- **appear/disappear and intensity swapped.** appear/disappear (−0.0214) now matters ~1.6× more
  than intensity (−0.0136); it was the other way around.

### Other Phase 5 readings
- **New overall best: `2026-07-27_14-10-54`** (TE 0.6706 / TF 0.6696 / EdgeR 0.8669), which is
  `size80` with `max_children = 5` instead of 3. `max_children` had never been swept; this one
  accidental comparison is worth +0.0098 TE and is unbracketed in both directions.
- **MHAT beats Ultrack** on the corrected GT: 0.6706 vs 0.6064 TE, 0.8669 vs 0.8183 EdgeR.
- **The newer segmentation is worse.** `2026-07-27_13-33-39` (seg `2026-07-27_11-38-06` +
  flow `2026-07-16_15-56-39`) scores 0.6132 against 0.6608 for the otherwise-identical `size80`
  on seg `2026-06-25_17-40-15` — a 0.048 TE loss, with NodeR 0.873 vs 0.959. Holding seg fixed
  at `2026-06-25_17-40-15` is the right call.
- **GT displacement is smaller than recorded.** The diagnostics on the corrected GT report mean
  raw displacement **6.47** (std 6.35, median 4.46) over all GT edges, not the 10.75 / 7.87 in
  the old attribute table. Any drift range must be re-anchored off a fresh `--stats-only` run.
- **Edge Recall is the bottleneck** at every operating point: 0.867 vs NodeR 0.966 at the best
  run. It is promoted to a co-primary target alongside TE/TF from Phase 6 on.

---

## Phase 6 / Stage 0 — Graph construction on the corrected GT (sweep `nkgraph`, 2026-07-30)

Coordinate-wise from the Phase 5 best (`2026-07-27_14-10-54`: size_threshold 80,
max_edge_distance 40, max_children 5, seg `2026-06-25_17-40-15`). 16 runs, submitted with
`launch_sweep.py --allow-protected` (all three keys are in `PROTECTED_KEYS`).
`anchor` reproduces the base, so it is also the reference row.

### size_threshold (max_children = 5)

| size_thr | TE | TF | EdgeR | NodeR | FN_e | FN_n | cand. edges | runtime |
|---|---|---|---|---|---|---|---|---|
| 0   | 0.6599 | 0.6609 | 0.8630 | **0.9682** | 141 | **34** | 202276 | 3099 s |
| 60  | 0.6531 | 0.6546 | 0.8649 | **0.9682** | 139 | **34** | 161881 | 2792 s |
| 80 (anchor) | 0.6706 | 0.6696 | 0.8669 | 0.9664 | 137 | 36 | 155063 | 2188 s |
| 120 | **0.6744** | **0.6750** | 0.8707 | 0.9654 | 133 | 37 | 144981 | 2617 s |
| 160 | 0.6618 | 0.6633 | 0.8688 | 0.9645 | 135 | 38 | — | — |
| 200 | 0.6638 | 0.6646 | **0.8737** | 0.9645 | **130** | 38 | 131296 | 2110 s |
| 240 | 0.6657 | 0.6593 | 0.8698 | 0.9617 | 134 | 41 | — | — |
| 320 | 0.6638 | 0.6573 | 0.8610 | 0.9523 | 143 | 51 | 117325 | 1963 s |

### max_edge_distance (size_threshold = 80, max_children = 5)

| max_edge_dist | TE | TF | EdgeR | NodeR | FN_e |
|---|---|---|---|---|---|
| 20 | 0.6560 | 0.6513 | 0.8571 | 0.9626 | 147 |
| 25 | 0.6686 | 0.6647 | 0.8649 | 0.9654 | 139 |
| 30 | 0.6686 | 0.6647 | 0.8659 | 0.9664 | 138 |
| **40 (anchor)** | **0.6706** | **0.6696** | 0.8669 | 0.9664 | 137 |
| 50 / 60 / 80 | byte-identical to 40 (Phase 5) | | | | |

### max_children (size_threshold = 80)

| max_children | TE | TF | EdgeR | NodeR | FN_e |
|---|---|---|---|---|---|
| 2 | 0.5841 | 0.5770 | 0.8280 | 0.9421 | 177 |
| 3 | 0.6608 | 0.6568 | 0.8620 | 0.9589 | 142 |
| 4 | 0.6569 | 0.6567 | 0.8649 | 0.9645 | 139 |
| **5 (anchor)** | **0.6706** | **0.6696** | 0.8669 | 0.9664 | 137 |
| 6 | 0.6638 | 0.6646 | 0.8678 | **0.9673** | 136 |
| 8 | 0.6589 | 0.6632 | **0.8717** | **0.9673** | **132** |
| 12 | 0.6638 | 0.6694 | **0.8756** | **0.9710** | **128** |

### Stage 0 findings

- **`max_edge_distance = 40` is optimal and now bracketed on both sides.** Narrowing hurts
  (20 → 0.6560); 25 and 30 are marginally below 40; 50/60/80 are byte-identical to 40. The
  Phase-4 "saturated above 40" claim is confirmed and the untested downward direction is now
  closed. Do not sweep this again.
- **`max_children` is saturated on TE above ~4** — values 4–12 wiggle inside 0.014 with no
  trend (0.6569 / **0.6706** / 0.6638 / 0.6589 / 0.6638); only 2 is clearly bad (0.5841).
  5 retained. Stage 1 was launched without waiting for `children12` (user instruction);
  it landed afterwards and confirms the plateau on TE.
  **But every other metric improves monotonically with max_children**: Edge Recall
  0.8620 → 0.8756, Node Recall 0.9589 → 0.9710, FN edges 142 → 128, FN nodes → 31 (the lowest
  anywhere in Stage 0, below even size_threshold 0's 34). At `max_children = 12`, TF (0.6694)
  also essentially ties the anchor (0.6696). Since Edge Recall is a co-primary, **this axis is
  not actually closed** — TE is flat while the co-primary is still climbing at the top of the
  tested range. Revisit in Stage 2 at the winning cost point, extending to 16/20.
- **`size_threshold` barely matters for TE and the curve is non-monotonic** — the entire
  0–320 range spans 0.021 TE with local maxima at 80 and 120. But **FN nodes rise monotonically
  with filtering** (34 at 0/60 → 51 at 320), so the filter is steadily destroying real
  detections while buying nothing reliable.
- **Two co-primaries disagree across this axis**: TE peaks at 120 (0.6744), Edge Recall at 200
  (0.8737, FN_e 130), Node Recall at 0/60 (0.9682, FN_n 34).
- **`size_threshold` interacts with `max_children`.** At max_children 3 the Phase 5 table had
  160 (0.6638) beating 80 (0.6608); at max_children 5 that reverses (80 → 0.6706 beats
  160 → 0.6618). The Phase 5 conclusion "the size_threshold optimum moved up" was an artefact
  of the lower max_children and is retracted.

### Stage 0 decision — base for Stage 1

**`size_threshold = 0`, `max_edge_distance = 40`, `max_children = 5`.**

`size_threshold = 0` is deliberately **not** the TE-maximising value (120). Design decision by
the user (2026-07-30): prefer the lower threshold that keeps more candidates, on the hypothesis
that **solver cost tuning should suppress incorrect links, rather than pre-filtering them out
of the graph**. The supporting argument: a filtered candidate can never be recovered by any
downstream cost, the detections filtering removes are the mid-track ones TE punishes hardest,
and TE's preference for 120 over 0 (0.6744 vs 0.6599) is inside the axis's own noise band while
the FN-node trend is monotonic and clean.

Recorded fallback: **0 and 60 are identical on node retention** (NodeR 0.9682, 34 FN nodes) —
everything under 60 voxels matches no GT node at all, so 60 discards zero real signal for 40k
fewer candidate edges and ~10% less runtime. If Stage 1 shows the cost terms cannot absorb the
extra spurious mass, switch to 60 at no cost.

### Attribute statistics at the Stage 1 base (size_threshold = 0, max_children = 5)
Graph: 42617 nodes, 202276 edges, 965748 curvature edge-pairs.

| attribute | mean | std | w / c | cost mean | cost std |
|---|---|---|---|---|---|
| adhesion | 0.612 | 0.478 | −1500 / 1500 | **+909.6** | 1772.5 |
| drift_dist | 14.185 | 8.760 | 100 / −1500 | **−81.5** | 876.0 |
| area_diff | 1.005 | 0.644 | 1500 / −1500 | **+7.2** | 965.3 |
| intensity_diff | 27.330 | 55.389 | 25 / −1000 | −316.7 | 1384.7 |
| cohesion | 0.995 | 0.030 | 0 / 0 | 0 | 0 |
| curvature | 20.209 | 11.844 | 0 / 0 | 0 | 0 |

**area's cost mean flips sign when the size filter comes off** (−304.0 at size_threshold=80 →
**+7.2** here), because small fragments have larger relative area changes (area_diff mean
0.797 → 1.005). Area is now net *discouraging* on average — the wrong sign for a cost whose
ablation is worth −0.1050 TE. Together with drift's near-neutral −81.5, that makes
`area_constant` and `drift_constant` the two first-order axes for Stage 1.

---

## Phase 6 / Stage 0b — max_children at the Stage-1 base (sweep `nkmc`, 2026-07-30)

Stage 0 measured `max_children` at `size_threshold = 80` and stopped at 12, where Edge Recall
was still rising. Both gaps closed here: re-swept at **`size_threshold = 0`** (the Stage 1 base;
Stage 0 showed the two keys interact) and extended to 20. `max_children = 5` is not repeated —
it is `nkgraph_size0`.

| max_children | TE | TF | EdgeR | NodeR | FN_e | FN_n |
|---|---|---|---|---|---|---|
| 5 (= nkgraph_size0) | **0.6599** | 0.6609 | 0.8630 | 0.9682 | 141 | 34 |
| 8  | 0.6482 | 0.6545 | 0.8678 | 0.9710 | 136 | 31 |
| 12 | 0.6540 | **0.6620** | 0.8717 | 0.9720 | 132 | 30 |
| **16** | 0.6531 | 0.6608 | **0.8727** | **0.9738** | **131** | **28** |
| 20 | 0.6540 | **0.6620** | 0.8707 | 0.9710 | 133 | 31 |

### Stage 0b findings
- **Edge Recall is now bracketed on this axis**: 0.8630 → 0.8678 → 0.8717 → **0.8727** (16) →
  0.8707 (20). Interior maximum at 16, so the Stage 0 worry that the co-primary was still
  climbing at the top of the range is resolved. Node Recall (0.9738) and FN nodes (**28**, the
  best figure anywhere in this campaign) agree on 16.
- **The co-primaries disagree, at small margins.** TE prefers 5 (0.6599 vs 0.6531, −0.0068);
  Edge Recall prefers 16 (+0.0097, 10 fewer FN edges, 6 fewer FN nodes). TE spans only ~0.012
  across the whole axis and is non-monotonic, while the recall metrics move monotonically to 16
  — the same pattern that decided the `size_threshold` call.
- **The Stage 0 max_children numbers do not transfer across `size_threshold`.** At
  `size_threshold = 80`, max_children 8 gave TE 0.6589 / EdgeR 0.8717; at 0 the same value gives
  0.6482 / 0.8678. Absolute levels shift, though the shape (TE flat, recalls climbing) holds.

### Consequence for Stage 1
Stage 1 (`nkp3A`, 37 runs) was already in flight at `max_children = 5` when this landed and was
**not** relaunched. The planned Stage 2 step — re-check the graph parameters at the winning cost
point — covers it, so this costs a confirmation pass rather than a redo, and TE being flat
across the axis is the condition under which the cost landscape should transfer. Stage 2 must
compare the Stage 1 cost winner at `max_children` 5 vs 16.

---

## Phase 6 / Stage 1 — Cost weights and constants (sweep `nkp3A`, 2026-07-30)

37 coordinate-wise runs from the Stage-0 base (`size_threshold = 0`, `max_edge_distance = 40`,
`max_children = 5`). This is the refinement "Phase 3" declared in 2026-07-09 and never ran —
before this, no cost weight or constant had ever been tuned on this dataset.

### Result: the base was already a local optimum on every axis

| axis | values tried (base*) | TE at each | verdict |
|---|---|---|---|
| drift_weight | 50, 75, **100***, 150, 200, 300 | .6317 .6336 **.6599** .6356 .5889 .5248 | interior max at base |
| area_weight | 750, 1000, **1500***, 2250, 3000 | .6064 .6249 **.6599** .6472 .6220 | interior max at base |
| intensity_weight | 8, 15, **25***, 40 | .6103 .6482 **.6599** .6472 | interior max at base |
| adhesion_weight | −750, −1000, **−1500***, −2250, −3000 | .6190 .6259 **.6599** .5364 .4859 | interior max at base |
| adhesion_constant | 0, 750, **1500***, 2250, 3000 | .6181 .6317 **.6599** .6190 .5520 | interior max at base |
| Σ edge constants | −3250, −3500, **−4000***, −4500, −5000, −5500, −6500 | .6132 .6297 **.6599** .6375 .6346 .6161 .6006 | interior max at base |
| appear/disappear | 0, 25, **50***, 100, 200, 400 | **.6657** .6550 .6599 .6482 .6463 **.6657** | see below |

**Best run: `appdis0` / `appdis400`, TE 0.6657 (+0.0058 over anchor) — treated as NOISE.**
`appear/disappear` = 0 and 400 give *identical* metrics while 25, 100 and 200 all fall *below*
the base. A U-shaped curve with tied endpoints two octaves apart is not a real optimum. No
other run beat the anchor at all.

### `drift_constant`, `area_constant` and `intensity_constant` are ONE axis, not three

Edge cost is a sum of `w·attr + c` terms, so only the **total** of the three constants enters
the ILP. Confirmed exactly — equal sums give byte-identical metrics:

| Σ constants | runs | TE | EdgeR | NodeR |
|---|---|---|---|---|
| −3250 | areac750 | 0.6132 | 0.8095 | 0.9028 |
| −3500 | driftc1000 **=** intc500 | 0.6297 | 0.8406 | 0.9346 |
| **−4000** | **anchor** | **0.6599** | 0.8630 | 0.9682 |
| −4500 | driftc2000 **=** intc1500 | 0.6375 | 0.8669 | 0.9785 |
| −4750 | areac2250 | 0.6433 | 0.8639 | 0.9850 |
| −5000 | driftc2500 | 0.6346 | 0.8610 | 0.9869 |
| −5500 | driftc3000 **=** areac3000 **=** intc2500 | 0.6161 | 0.8533 | 0.9888 |
| −6500 | driftc4000 | 0.6006 | 0.8435 | 0.9925 |

**Sweep-design lesson: never sweep the three edge constants as separate axes.** 11 runs
collapsed to 8 distinct points here. Sweep ONE of them (or their sum) and spend the rest on
weights, which are genuinely independent. The premise in the `nkp3A` spec header — that
`area_constant` and `drift_constant` were two separate first-order levers because area's cost
mean had flipped sign — was wrong for the same reason: only the sum matters, and the sum was
already optimal.

Note the monotone side-effect: making the edge constants more negative drives Node Recall up
to 0.9925 while TE falls — the solver selects almost every GT node but fragments the tracks.
Same TE/recall divergence seen on `size_threshold` and `max_children`.

### Bearing on the `size_threshold = 0` decision — hypothesis NOT supported on TE

The Stage-0 decision (user, 2026-07-30) was to keep `size_threshold = 0` on the hypothesis that
solver cost tuning would suppress bad links rather than pre-filtering them. Best available:

| config | costs | TE | EdgeR | NodeR | FN_n |
|---|---|---|---|---|---|
| size120, mc5 | default | **0.6744** | 0.8707 | 0.9654 | 37 |
| 2026-07-27_14-10-54 (size80, mc5) | default | 0.6706 | 0.8669 | 0.9664 | 36 |
| **size0, mc5, best tuned** | tuned | 0.6657 | 0.8639 | 0.9720 | — |
| size200, mc5 | default | 0.6638 | **0.8737** | 0.9645 | 38 |
| size0, mc16 | default | 0.6531 | 0.8727 | **0.9738** | **28** |

Cost tuning at `size_threshold = 0` tops out at 0.6657, **below** `size_threshold = 120` at
untuned costs (0.6744). The costs did not recover what the missing filter cost.
**Partially vindicated:** node retention did improve as predicted — Node Recall reaches
0.972–0.99 and FN nodes fall to 28 — but TE does not reward it, because extra selected nodes
fragment tracks (the same mechanism that makes Edge Recall and TE diverge).

**Caveat before concluding:** the comparison is not clean. `size120` was never cost-tuned. The
honest test is the Stage-1 cost point run across several `size_threshold` values — that is
Stage 2.

### Stage 1 conclusion
Coordinate-wise cost tuning is **exhausted** on this dataset. Every axis is interior-bracketed
at the base, so there is nothing left to extend, and combinations of interior optima rarely pay.
All the remaining variation lives in graph construction, where the campaign's whole TE spread
(~0.02) actually sits.

---

## Phase 6 / Stage 2 — size_threshold x max_children grid (sweep `nkgrid`, 2026-07-30)

All 12 cells at **base costs** (justified by Stage 1: the base is the interior optimum on every
cost axis). Six cells already existed from `nkgraph`/`nkmc` and were reused, so only six were
run. `max_edge_distance = 40` throughout (bracketed both sides in Stage 0).

| metric | st \ mc | 5 | 12 | 16 |
|---|---|---|---|---|
| **TE** | 0 | 0.6599 | 0.6540 | 0.6531 |
| | 80 | 0.6706 | 0.6638 | 0.6638 |
| | **120** | **0.6744** | 0.6608 | 0.6638 |
| | 160 | 0.6618 | 0.6589 | 0.6618 |
| **TF** | 0 | 0.6609 | 0.6620 | 0.6608 |
| | 80 | 0.6696 | 0.6694 | 0.6694 |
| | **120** | **0.6750** | 0.6658 | 0.6694 |
| | 160 | 0.6633 | 0.6627 | 0.6664 |
| **EdgeR** | 0 | 0.8630 | 0.8717 | 0.8727 |
| | 80 | 0.8669 | 0.8756 | 0.8756 |
| | **120** | 0.8707 | 0.8756 | **0.8776** |
| | 160 | 0.8688 | 0.8727 | 0.8746 |
| **NodeR** | **0** | 0.9682 | 0.9720 | **0.9738** |
| | 80 | 0.9664 | 0.9710 | 0.9710 |
| | 120 | 0.9654 | 0.9692 | 0.9692 |
| | 160 | 0.9645 | 0.9673 | 0.9673 |
| **FN_e** | 0 | 141 | 132 | 131 |
| | 80 | 137 | 128 | 128 |
| | **120** | 133 | 128 | **126** |
| | 160 | 135 | 131 | 129 |
| **FN_n** | **0** | 34 | 30 | **28** |
| | 80 | 36 | 31 | 31 |
| | 120 | 37 | 33 | 33 |
| | 160 | 38 | 35 | 35 |

### Stage 2 findings

- **`size_threshold = 0` is falsified under matched costs.** It is the worst column on TE at
  *every* `max_children` (0.6599 / 0.6540 / 0.6531) and loses on Edge Recall at every
  `max_children` too (0.8630 / 0.8717 / 0.8727 against 120's 0.8707 / 0.8756 / 0.8776). It wins
  only on node retention (FN nodes 28 vs 33). The extra candidates are kept as *detections* but
  convert into neither better track coverage nor better linking. The Stage-0 decision to keep
  everything and let the solver sort it out is therefore not supported — tested properly and
  retired.
- **`size_threshold = 120` wins both co-primaries**: best TE in the whole campaign (0.6744 at
  mc 5) and best Edge Recall in the whole campaign (0.8776 at mc 16), plus the lowest FN edges
  (126). It is the column winner at every `max_children`.
- **The two axes interact, monotonically and in opposite directions.** TE falls with
  `max_children` at every threshold; Edge Recall rises with it at every threshold. This is the
  same TE-vs-recall divergence seen on `size_threshold` and on the edge constants: more
  selected objects/links improve per-link recall but fragment tracks, which TE punishes.
- **Node retention is monotone in `size_threshold` alone** — the FN_n column is ordered
  0 < 80 < 120 < 160 at every `max_children`, independent of it.

### Effect sizes — read before acting on any of this
The whole grid spans **0.021 TE** and **0.015 Edge Recall**. Against 1070 GT nodes / 1029 GT
edges, the (120,5)-vs-(120,16) choice is 7 FN edges and 4 FN nodes. The campaign's total
improvement over the Phase 5 starting point (`2026-07-27_14-10-54`, TE 0.6706 / EdgeR 0.8669)
is **+0.0038 TE and +0.0038 EdgeR** if (120,5) is chosen. The honest summary of Phase 6 is that
**the pre-existing operating point was already very close to optimal**, and the campaign's main
value is the negative results — costs are locally optimal, the three edge constants are one
axis, `max_edge_distance` is closed, and keeping all candidates does not pay.
