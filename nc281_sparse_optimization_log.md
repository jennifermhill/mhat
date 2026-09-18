# NC281-sparse-label Tracking Optimization Log

## Setup

**Dataset**: NC281-sparse-label/01_nuclei_denoised. The dataset name is misleading — the **GT is densely annotated** ("sparse-label" refers to the raw-data labeling, not the GT). Because GT is dense, **precision and purity are reliable here** and tracked as primary metrics (unlike NC281-Fl2mSiH2B where sparse GT makes precision/purity unreliable).

**Eval**: matcher=point, metrics=[basic, track_overlap], match_threshold=10
**Pipeline**: cluster-only (mhat2 env, gurobi module). Submitted via `launch_batch.py` / `launch_curvature_batch.py` + `submit_evals.py` with `ended()` deps.

**Per-run log entry includes Purity** as a primary metric (not just a diagnostic) — a config that boosts NodeR by inflating FPs is not an improvement here.

**Baseline (2026-04-28)**: TE=0.769, TF=0.779, NodeR=0.932, EdgeR=0.880 (purity not measured at baseline; tracked from B1R1 onward)

---

## Batch 1 (2026-04-28)

5 parallel cluster runs sweeping different parameters. Used original `launch_batch.py` (no `python -u`); all 5 trackings exited 120 *after* writing valid outputs (recovered via `ended()` dependency switch).

### B1R1: size_threshold=0
exp_uid: 2026-04-28_09-45-06_R1
Hypothesis: "size_threshold=0 (from 20) recovers small-object FNs"
TE: 0.794, TF_mean: 0.804, Node_Recall: 0.959, Edge_Recall: 0.905, Purity: 0.874
Verdict: supported — best recall (NodeR +0.027, EdgeR +0.025) but slight purity drop (~0.94 → 0.874)

### B1R2: seg_result=2026-04-27_17-00-54
exp_uid: 2026-04-28_09-45-06_R2
Hypothesis: "Newer seg with more detections gives ILP more candidates"
TE: 0.786, TF_mean: 0.797, Node_Recall: 0.969, Edge_Recall: 0.906, Purity: 0.756
Verdict: partially supported — best Node Recall but big purity drop (0.756) signals many extra detections are FPs

### B1R3: cohesion_constant=0
exp_uid: 2026-04-28_09-45-06_R3
Hypothesis: "cohesion_constant=0 (from 450) shifts node cost mean ~-490 → strong global node encouragement"
TE: 0.775, TF_mean: 0.780, Node_Recall: 0.933, Edge_Recall: 0.877, Purity: 0.905
Verdict: falsified — essentially baseline metrics; cohesion_constant inactive at this operating point

### B1R4: appear_constant=0, disappear_constant=0
exp_uid: 2026-04-28_09-45-06_R4
Hypothesis: "appear/disappear=0 (from 50) cheaper track endpoints"
TE: 0.773, TF_mean: 0.778, Node_Recall: 0.934, Edge_Recall: 0.877, Purity: 0.905
Verdict: falsified — same regime as R3; appear/disappear inactive

### B1R5: drift_weight=100
exp_uid: 2026-04-28_09-45-06_R5
Hypothesis: "drift_weight=100 (from 25) brings drift cost_std to ~824 → edge discrimination"
TE: 0.797, TF_mean: 0.804, Node_Recall: 0.932, Edge_Recall: 0.884, Purity: 0.941
Verdict: supported — best TE+TF+purity combination overall

### Conclusion

Two real wins (R1, R5). **R5 is the cleanest single-parameter improvement** (+0.028 TE, preserves purity). R1's recall gains are real but cost some purity. R3 and R4 confirmed cohesion_constant and appear/disappear are inactive.

**Next batch direction**: combine B1R1 + B1R5 (size_threshold=0 + drift_weight=100) to test whether recall and purity gains stack.

---

## Batch 2 (2026-04-29)

5 parallel cluster runs. All built on B1R5 base (drift_weight=100). Three completed cleanly; one timed out at 16h walltime; one finished but added a harmful edge cost.

### B2R1: size_threshold=0 + drift_weight=100
exp_uid: 2026-04-29_11-36-23_R1
Hypothesis: "Stack B1R1 + B1R5 — size_threshold=0 recall gain combines with drift_weight=100 purity preservation"
TE: 0.827, TF_mean: 0.838, Node_Recall: 0.959, Edge_Recall: 0.914, Purity: 0.923
Verdict: supported — **NEW BEST overall** (+0.030 TE, +0.034 TF over B1R5; only -0.018 purity)

### B2R2: drift_weight=150
exp_uid: 2026-04-29_11-36-23_R2
Hypothesis: "Push drift past 100 (cost_std≈1236); does TE keep climbing or saturate?"
Result: TIMEOUT at 16h walltime (LSF exit 140). No `pred_tracks.zarr` written.
Verdict: inconclusive — ILP solve time grows with drift_weight; need >16h walltime to test, or accept saturation

### B2R3: size_threshold=10 + drift_weight=100
exp_uid: 2026-04-29_11-36-23_R3
Hypothesis: "Middle-ground threshold preserves recall without B1R1's purity hit"
TE: 0.814, TF_mean: 0.827, Node_Recall: 0.951, Edge_Recall: 0.905, Purity: 0.934
Verdict: supported but dominated by R1 — better purity than R1 (0.934 vs 0.923) but lower TE/TF (0.814 vs 0.827)

### B2R4: drift_weight=100 + area_weight=2500, area_constant=-1500
exp_uid: 2026-04-29_11-36-23_R4
Hypothesis: "First test of area edge cost (untested attr, std/mean=0.94, target cost_std≈900)"
TE: 0.779, TF_mean: 0.787, Node_Recall: 0.931, Edge_Recall: 0.876, Purity: 0.920
Verdict: falsified — adding area cost is harmful (-0.018 TE vs B1R5). Matches NC281-Fl2m NC1-R2 result.

### B2R5: drift_weight=100 + intensity_weight=12, intensity_constant=-800
exp_uid: 2026-04-29_11-36-23_R5
Hypothesis: "First test of intensity edge cost (untested, best spread of any attribute, std/mean=1.24)"
TE: 0.789, TF_mean: 0.801, Node_Recall: 0.932, Edge_Recall: 0.883, Purity: 0.935
Verdict: falsified — neutral-to-slightly-harmful (-0.008 TE vs B1R5). Matches NC281-Fl2m NC1-R3 result.

### Conclusion

**B2R1 is the new best**: stacking confirmed — size_threshold=0 + drift_weight=100 lifts TE 0.797 → 0.827 (+0.030) and TF 0.804 → 0.838 (+0.034) with only a small purity dip. Area and intensity edge costs were both harmful, mirroring NC281-Fl2m results from NC1. Drift past 100 timed out — open question whether saturation or just longer ILP solve.

**Open follow-on**:
- Retry B2R2 (drift_weight=150) with longer walltime (24h+), or accept drift_weight=100 as saturating
- B2R1's seg config + add curvature (only untested edge attribute on this dataset)
- size_threshold below 0 not possible; consider per-fragment filtering by another attribute

## waterz-convention check on 02_nuclei_denoised_train (2026-09-17, cluster, direct on node)

Does the waterz channel-order + one-voxel-shift fix (CLAUDE.md "2D Support > Known hazards" item 1;
MDA231 refit 2026-09-15) change NC281-sparse-label? Arms built from the train seg `2026-07-02_10-32-29`
exactly as for MDA231 (`configs/experiments/waterz_shift_nc281/` in the data tree; control merge
history byte-identical to the reference; shift arm 192/196 merge pairs shared, Spearman 0.71, cost
shift -0.030, final components identical 20/20). Current-best `s0_dw100_cw500` params unchanged.

### WSN-R0-ctrl: control arm `otsu0702_chanorder_xzy`, params = s0_dw100_cw500
exp_uid: wsn_ctrl_R0
Hypothesis: "Cluster stack (mhat-cluster, Gurobi 13.0.3) reproduces the published Windows result"
TE: 0.8093, TF_mean: 0.8319, Node_Recall: 0.9534, Edge_Recall: 0.9035, Purity: 0.9137
Verdict: supported — identical to s0_dw100_cw500 in every metric and count (fp 73, fn 128, fn_edges 249)

### WSN-R0-shift: fixed arm `otsu0702_chanorder_zyx_shift`, params = s0_dw100_cw500
exp_uid: wsn_shift_R0
Hypothesis: "As on MDA231, the fix moves cohesion/adhesion scores and shifts the metrics under untuned weights"
TE: 0.8093, TF_mean: 0.8319, Node_Recall: 0.9530, Edge_Recall: 0.9031, Purity: 0.9133
Verdict: falsified — one node and one edge differ (fn 129, fn_edges 250); TE/TF unchanged, sel 0.8615 -> 0.8613.
Cohesion/adhesion are near-inert here (cost std ~30-50 vs drift ~800), and the fix touches only those
two attributes, so there is nothing for a refit to recover. Not worth a refit unless cohesion/adhesion
are first made active.

### WSN1: paired cohesion/adhesion/drift due-diligence sweep on both waterz arms (2026-09-17, 30 runs, direct on node)
exp_uids: wsn1_<cond>_<ctrl|shift> (sweep dir `configs/experiments/waterz_shift_nc281/sweep1/` in the data tree, results_with_baseline.csv)
Hypothesis: "If the fixed cohesion/adhesion scores carry signal, some stronger node weight will separate the fixed arm from the control"
Grid: cohesion_w −2000/−5000/−10000/+2000, adhesion_w −250/−500/−750/−1000/−2000/−4000 (constants hold the cost mean fixed), drift_w 50/200/300 at c −2000, drift_c −1000/−4000 at w 100.
Best fixed arm: adh_m500_shift TE 0.8109, TF 0.8332, Purity 0.9124, sel 0.8616 (vs control baseline 0.8093 / 0.8319 / 0.9137 / 0.8615); adh −250 = baseline, adh −750 worse on both arms.
Negative cohesion saturates at −2000 (identical solutions through −10000, sel −0.0008); adhesion monotonically harmful past −500; drift bracketed at 100/−2000.
Verdict: falsified — the fixed arm never separates from the control by more than one node/one edge except in conditions that damage both arms. Current best stays s0_dw100_cw500; no refit warranted on this dataset.
