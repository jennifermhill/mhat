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
