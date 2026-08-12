# GT-amount learning curve on Fluo-C3DL-MDA231 — results

Executed 2026-08-06. Second, independent dataset for the experiment first run on
`NC281-sparse-label` (see `gt_amount_experiment_plan.md`). Design, ground rules and arm
definitions are unchanged; the dataset, its scale and its metric family differ.

- Train: `01_cells`, 33 GT tracks / 364 GT nodes. Test: `02_cells` (held out, scored once).
- Segmentation / flow held fixed: `seg_cp_20260720_fs1_cpm6` + `2026-02-17_15-55-00` (train),
  `holdout_fs1_cpm6` + `2026-05-29_10-59-37` (test).
- Metric: CTC `TRA` (primary), with `DET` and `SEG` panels. `matcher = "ctc"`.
- Driver config: `configs/tracking/Fluo-C3DL-MDA231/01_cells/gt_amount/gt_amount_stage2.toml`
- Figures: `experiments/evaluation/Fluo-C3DL-MDA231/02_cells/gt_amount/gt_amount_{curve,panels}.png`
- CSV: `.../02_cells/gt_amount/stage2_test.csv` (151 rows)

---

## Headline result

Mean over seeds, held-out test TRA:

| annotated GT nodes | 45 | 86 | 131 | 174 | 262 | 364 (full) |
|---|---|---|---|---|---|---|
| **Arm A** — unlabeled candidates removed | 0.848 | 0.904 | 0.916 | 0.923 | 0.930 | 0.936 |
| **Arm B** — `gt_selected=None` (motile API) | 0.065 | 0.216 | 0.289 | 0.374 | 0.798 | 0.936 |
| **Arm C** — dense crop | 0.855 | 0.892 | 0.911 | 0.928* | 0.931 | 0.936 |

\* arm C at 174 is a mean over 8 seeds; the other two are empty on the test graph. Every
other cell is a mean over all 10 seeds.

Reference lines: hand-tuned full GT **0.939**; SSVM full GT at the old default
`ssvm_reg = 0.1` **0.934**.

**Arm A and arm C are indistinguishable within error at every budget, and both degrade
gently — 12% of the annotation (45 of 364 object-frames) still buys TRA ≈ 0.85 against a
full-GT 0.936. Arm B collapses.** At 72% of the GT it has already lost 0.14 TRA; at 12% it
is at 0.074, i.e. no usable tracking at all, and 2 of its 10 seeds produce a literally empty
solution.

This is the same ordering NC281 produced (A ≈ C ≫ B), so the finding is not a
single-dataset artifact.

### Why B fails

B marks unannotated candidates `gt_selected = None`, which motile documents as its sparse-GT
API. Those columns still enter `SoftMarginLoss` with `ground_truth = 0`: `HammingCosts` zeroes
their loss coefficient, but the margin terms still read them as "GT never selects this". They
are soft negatives, not ignores. A deletes the same candidates outright, so they cannot vote.
The gap between the two curves is the cost of that distinction, and it grows as the fraction of
unannotated candidates grows — which is exactly what shrinking the GT does.

---

## Cross-dataset comparison

| | NC281 (TE) | | MDA231 (TRA) | |
|---|---|---|---|---|
| budget | arm A | arm C | arm A | arm C |
| full | 0.816 | 0.816 | 0.936 | 0.936 |
| ~70% | 0.818 | 0.797 | 0.930 | 0.931 |
| ~35% | 0.815 | 0.770 | 0.923 | 0.928 |
| ~15% | 0.781 | 0.736 | 0.916 | 0.911 |
| ~7% | **0.257** | 0.675 | 0.904 | 0.892 |
| ~3% | — | — | 0.848 | 0.855 |

Two things replicate and one differs:

1. **A ≈ C at moderate budgets** — replicates.
2. **B collapses well before either** — replicates, and more violently on MDA231 (B is already
   broken at 72% of GT, where on NC281 it survived to ~60%). MDA231 has 5.4x fewer candidate
   variables, so hiding a track exposes proportionally more of the graph to the soft-negative
   effect.
3. **NC281's arm A collapses at its lowest budget (196 nodes -> TE 0.257) while arm C holds at
   0.675. MDA231's arm A does not collapse** — at 45 nodes it is 0.848, matching arm C. So the
   dense-crop protocol's advantage at very low budgets is a real NC281 effect that did not
   transfer. Worth stating rather than generalizing from one dataset.

---

## Empty solutions are decided on the TEST graph

An `empty_solution` fit is one that **converged and wrote weights** — all six across the two
datasets reached ε → 0 in 21–31 iterations, well inside the 100-iteration limit. What is empty
is the ILP solution those weights imply, and that is a property of a particular candidate
graph. The train graph is not the one the curve reports on.

The protocol is therefore: carry empty-on-train fits through to the test split, solve them
there, and plot the real test score — marking a run empty **iff the test solve is also empty**.
Only a fit with no weights at all is dropped. `collect_gt_amount.py` carries `status` (the
outcome on the dataset being scored, which the plot keys on) and `train_status` (what the fit
reported); the two disagree exactly in this case.

Measured, all six such runs:

| run | train | test | test score |
|---|---|---|---|
| MDA231 `gtm_B_n004_s3` | empty | **12 nodes / 11 edges** | TRA 0.021 |
| MDA231 `gtm_B_n004_s4` | empty | **24 nodes / 22 edges** | TRA 0.041 |
| MDA231 `gtm_C_b0174_s0` | empty | empty | — |
| MDA231 `gtm_C_b0174_s9` | empty | empty | — |
| NC281 `gt2_B_n012_s1` | empty | empty | — |
| NC281 `gt2_B_n012_s3` | empty | empty | — |

So two of the six are genuinely scoreable, and both are arm B's. The effect on the figures is
small — MDA231's arm-B bottom rung moves from 0.074 (8 seeds) to 0.065 (10 seeds), and NC281's
figure is unchanged because both of its empties really are empty on test — but the old
behaviour reached the right answer by luck, not by evidence.

### Why the weights go degenerate

Two things fail together, visible by comparing the empties against working neighbours at the
same rung:

| run | drift_c | cohesion_w | appear_c | disappear_c | train solve |
|---|---|---|---|---|---|
| `b0174_s0` | −0.106 | **+0.147** | 0.351 | 0.235 | empty |
| `b0174_s9` | −0.086 | **+0.100** | 0.341 | 0.237 | empty |
| `b0174_s1` | −0.533 | −0.269 | 0.273 | 0.219 | ok |
| `b0174_s2` | −0.394 | −0.071 | 0.316 | 0.264 | ok |
| full GT | −0.785 | −0.267 | 0.527 | 0.419 | ok |

Cohesion's weight comes out **positive**, so with mean cohesion ≈ 0.94 each node costs rather
than pays, and the edge constants are 4–8× weaker than any working run. Appear + disappear
still charge ~0.59 per track. Selecting anything costs more than it saves, so the optimum is
the empty set.

Still unexplained: the plan expected arm-C trouble at the *smallest* crop rung (a possible
`n_nodes_neg == 0` crash). Instead the bottom rung is clean and the two failures are at the
second-*largest* budget, with 147/148 negatives available. Worth a look.

## SEG is flat, by construction

SEG sits at ~0.70 for arms A and C across the entire budget range (full GT 0.705, lowest rung
0.683). That is not insensitivity of the method: SEG is upper-bounded by the held-fixed
segmentation, and the arms move it only by selecting a different merge level. It is also
measured on 13 sparse annotated z-slices. Read the SEG panel as "the GT amount does not change
which fragments get merged", not as a quality plateau. Arm B's SEG tracks its collapse
(0.63 -> 0.07) because a degenerate solution selects the wrong objects entirely.

---

## Stage 1 — regularizer calibration (two passes, 19 points)

Pass 1 (broad, 13 points, third-decade, `10.0 … 0.001`) won at `ssvm_reg = 0.0464`, TRA 0.9038.
Pass 2 (fine, 6 points, twelfth-decade, appended as `r13..r18` inside the pass-1 bracket) won at
0.0681, TRA 0.9044 — an improvement of **0.0006**.

Pass 2's real contribution was not a better winner but showing the pass-1 winner sat on a
**plateau**, not a peak: eight consecutive points from 0.0215 to 0.0825 lie within 0.0028 of each
other, bracketed on both sides (0.0100 -> 0.8973, 0.1000 -> 0.8944). Under the < 0.005 rule
declared before the sweep ran, the argmax was rejected and the geometric centre taken:

    plateau [0.0215, 0.0825], geometric centre 0.04214
    -> nearest measured point ssvm_reg = 0.0383 (TRA 0.9041)
    -> ssvm_reg_effective_target = 77.48   at n_labeled = 2023

All 19 fits converged (max 51 iterations against a limit of 100).

The NC281 transfer was informative but not exact: the plateau contains the "effective
regularizer transfers" prediction (0.0651) and excludes the "`ssvm_reg` transfers" prediction
(0.0121), so the effective-regularizer framing is the better of the two models.

---

## The blocker that had to be fixed first

`gt_subsets.py` and `gt_crops.py` both require `correct_seg.zarr` to be labeled with geff **node
ids**. MDA231's GT, written by `from_ctc_to_geff`, is labeled with CTC **track ids** — 33 labels,
max 51, against 364 node ids — and declares `related_objects.label_prop = "track_id"`. Against
that GT the subset LUT silently drops every node id above 51, and the crop path keeps an
unrelated set of nodes before tripping the driver's
`n_gt_nodes_kept == n_gt_nodes_annotated` assertion.

Fix: `scripts/04_tracking/make_nodeid_gt.py` writes a node-id-labeled *twin* of the same voxels
to `01_cells/gt_amount/gt_nodeid/`, and the driver points `gt_data_dir` at it. Neither
materializer was edited, so NC281's no-oracle verification carries over unchanged. The twin is
verified equivalent through the fit path: `load_gt` on the twin and on the original return an
identical graph, identical renumbered `track_id`s, and a **byte-identical** remapped `gt_seg`.

---

## Verification gates

| gate | result |
|---|---|
| G0 GT twin round-trips | PASS — 364 nodes, 364 labels, byte-identical `gt_seg` |
| G1 `--self-test` | PASS |
| G2 `--dry-run`, all 151 combinations | PASS (the direct test of the blocker fix) |
| G3 stage-1 winner interior + converged | PASS both passes; plateau rule applied |
| G4 graph size recorded | 673 nodes / 1350 edges -> 2023 labeled; 348/364 GT nodes matched at IoGT 0.5. See `gt_amount/G4_graph_stats.md` |
| G5 one non-empty logfile per run | PASS — 151/151 |
| G6 NC281 regression | PASS — `--collect` still 0.0121 / 131.61; `--dry-run` still passes |
| G6b NC281 figure regression | PASS — regenerated figure pixel-identical |
| G7 crop budgets + box collisions | Budgets `[262,174,131,86,45]` match `match_sizes` (±2 nodes). 9 distinct boxes / 10 seeds at the top three rungs, 10/10 at the bottom two. See `gt_crops/effective_replicates.json` |
| G8 arm-A no-oracle verifier | PASS on `n016_s0` |
| G9 crop no-oracle verifier | PASS on `b0131_s0` |

### Caveats to carry into the writeup

- **Effective replicates.** Seeds 2 and 3 clamp to the same crop box at the three largest arm-C
  rungs, so those points have 9 independent boxes, not 10, and their SEM is correspondingly
  slightly optimistic. All five rungs were kept: dropping the top one would leave arm C with
  four points against A/B's five and break the equal-budget comparison the design rests on.
- **Arm C's top rung is 72% of all GT**, so "dense crop" there is close to "annotate everything".
- **Four fits were empty on the train graph; only two are empty on test.** See the section
  below — this is a protocol point, not just a footnote.
- **`02_cells` has three dividing tracks while the graph is built with `divisions = false`**,
  which caps achievable LNK for reasons unrelated to GT amount. LNK is therefore a CSV
  diagnostic here, not a panel.
- **One deliberate dense-GT dependency remains**, exactly as on NC281: stage 1 calibrates
  `ssvm_reg` on the fully annotated train split and stage 2 imposes that scalar on every subset
  fit. Dense-GT knowledge reaches the small-N fits through the hyperparameter, not the labels.
