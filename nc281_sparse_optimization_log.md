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
