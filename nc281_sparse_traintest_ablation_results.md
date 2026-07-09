# NC28.1-sparse-label train/test ablation optimization — results

Optimization on `02_nuclei_denoised_train`, final eval on held-out `02_nuclei_denoised_test`
(params copied verbatim from train winners — no tuning on test). Selection metric:
**sel = mean(TE, Track Purity)**, tie-break TF. Point matcher (`match_threshold=10`),
`metrics=["basic","track_overlap"]`. Pipeline held fixed (z_flow filtering on) across all conditions.

Inputs (cluster-generated seg/flow IDs): see `configs/tracking/NC281-sparse-label/02_nuclei_denoised_*`
and the `[nc281_sparse_train]` / `[nc281_sparse_test]` sections of
`scripts/05_evaluation/{merge_ablation,solver_ablation,solver_addition}.toml`.

## Step 0 — full-model baseline tune (train)
Coordinate sweep, drift then cohesion. Winner (`s0_dw100_cw500`): **drift_weight=100, drift_constant=-2000,
cohesion=-500/450, adhesion=-100/50** → TE=0.809, Pur=0.914, TF=0.832, **sel=0.861**.
(drift_weight=300 raised purity but dropped TE; cohesion weight had negligible effect — its cost std ~90.)

## Solver addition (train → test), sel = mean(TE,Pur)
| condition | active weight (train winner) | train sel | test sel | train uid / test uid |
|---|---|---|---|---|
| none | empty (drift_const=-2000, w=0) | 0.326 | 0.316 | add_none_dc2000 / test_none |
| + cohesion | cohesion_w=-10000 | 0.438 | 0.365 | add_coh_w10k / test_plus_coh |
| + volume | area_w=1300 | 0.611 | 0.618 | add_vol_w1300 / test_plus_vol |
| + intensity | intensity_w=3 | 0.639 | 0.629 | add_int_w3 / test_plus_int |
| + curvature | curvature_w=100 | 0.861 | 0.839 | add_curv_w100 / test_plus_curv |
| + drift | drift_w=100 | 0.862 | 0.857 | add_drift_w100 / test_plus_drift |
| full | all costs | 0.861 | 0.858 | s0_dw100_cw500 / test_full |

## Solver ablation (train → test), from full model
Cost removed by zeroing its weight+constant (the `ablate_*` flags are dead code on this branch).
| condition | mechanism (train winner) | train sel | test sel | train uid / test uid |
|---|---|---|---|---|
| baseline | full model | 0.861 | 0.858 | s0_dw100_cw500 / test_full |
| - cohesion | cohesion+adhesion zeroed, drift_w=100 | 0.862 | 0.857 | nocoh2_dw100 / test_nocoh2 |
| - drift | drift zeroed + base_edge_constant=-2000, cohesion active | 0.423 | 0.546 | nodrift2_cw250 / test_nodrift2 |
| - all | empty (= addition none) | 0.326 | 0.316 | add_none_dc2000 / test_none |

## Merge ablation (train → test); tracking re-optimized per segmentation (active costs only)
| condition | seg override / tracking (train winner) | train sel | test sel | train uid / test uid |
|---|---|---|---|---|
| baseline | default seg, full model | 0.861 | 0.858 | s0_dw100_cw500 / test_full |
| - cohesion | default seg, coh+adh zeroed, drift_w=100 | 0.862 | 0.857 | nocoh2_dw100 / test_nocoh2 |
| - affinities | scoring_function=symmetric, coh+adh zeroed, drift_w=200 | 0.862 | 0.846 | noaff2_dw200 / test_noaff2 |
| - merges | skip_merges=true, coh+adh zeroed, drift_w=100 | 0.861 | 0.850 | nomrg2_dw100 / test_nomrg2 |

## Key findings
- **Drift is essential.** Removing it (drift weight+constant zeroed, base-edge floor kept) collapses sel
  from 0.86 baseline to **0.42 (train) / 0.55 (test)** — the largest single-cost effect. Consistently,
  drift-alone (addition) ≈ full model (0.862 vs 0.861): drift carries almost all the signal here.
- **Cohesion/adhesion contribute ~nothing** at the operating point (zeroing them leaves metrics essentially
  unchanged; adding them alone is weak). Their node-cost std is tiny (~90 vs drift ~769).
- **Segmentation (merge ablation) has a modest effect**: symmetric scoring → test 0.846, no-merges → 0.850
  (vs 0.857 baseline). Smaller than removing drift, larger than removing cohesion.
- **Train→test transfer is clean** (Full: 0.861→0.858; test TE actually higher, 0.809→0.829).

## Ablation mechanism (important correction)
The legacy `ablate_*` flags are **dead code** on this branch (`apply_mean_ablation` in `utils.py` is never
called). A cost is ablated **only** by setting its weight AND constant to 0, so `add_costs` skips it. The
first pass mistakenly used `ablate_* = true` with nonzero weights, so `-cohesion`/`-drift`/`-affinities`
silently ran the full/wrong model (baseline = -cohesion = -drift). Fixed by re-sweeping those conditions
with proper zeroing. `- Drift` keeps `base_edge_constant=-2000` so edges remain selectable (isolates loss
of drift *guidance*; keeps `- Drift ≥ - All`).

No code change was needed: a `solve_with_motile.py` edit made in the first pass (to pass `no_merges` into
`add_costs`) was **reverted** — with cohesion properly zeroed, `add_costs` skips the cohesion cost via its
`weight==0` branch and never reads the missing `'cohesion'` node attribute (verified: `nomrg2_*` runs
complete on the reverted code with no `KeyError`).

Figures: `experiments/evaluation/NC281-sparse-label/02_nuclei_denoised_{train,test}/*.png`.
