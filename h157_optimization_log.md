# Fluo-C3DH-H157 / 01_cells — Tracking Optimization Log

## Setup
- Segmentation: 2026-04-21_13-35-28 (cellpose, merge_thresholds=[1.0])
- Optical flow: 2026-04-24_14-36-14 (2D Farneback + 3D Lucas-Kanade)
- Mode: skip_merge_hypotheses=true (pre-merged to max level, no multi-hypothesis)
- Truncated to 15 timepoints for optimization speed
- Ground truth: CTC format, truncated to 15 frames (01_GT), full 60 frames saved as 01_GT_full
- Evaluation: CTC metrics, CTC matcher
- Initial: size_threshold=500, max_edge_distance=60

## Graph Statistics (calibration run, size_threshold=500, 86 nodes)
| Attribute | Count | Mean | Std | Weight | Constant | Cost Mean | Cost Std |
|-----------|-------|------|-----|--------|----------|-----------|----------|
| drift_dist | 211 | 19.496 | 18.437 | 57.0 | -2000.0 | -888.7 | 1050.9 |
| area_diff | 211 | 0.572 | 0.611 | 1730.0 | -1500.0 | -509.9 | 1057.1 |
| intensity_diff | 211 | 486.897 | 626.016 | 4.0 | -1000.0 | +947.6 | 2504.1 |
| curvature | 546 | 38.046 | 28.568 | 0.0 | 0.0 | 0.0 | 0.0 |

## Node Size Analysis
- 116 total pre-merged nodes across 15 timepoints
- 4 GT cells per frame, real cells are 100k-495k voxels
- FP fragments: mostly under 20k voxels, many under 1k
- Clear gap: >=20000 gives 60 nodes, >=50000 gives 59 nodes

## Baseline (B0R0)

### B0R0: MDA231 parameters (uncalibrated), size_threshold=500
exp_uid: 2026-04-24_20-02-41
Hypothesis: "MDA231 parameters as starting point"
TRA: 0.860, DET: 0.855, LNK: 0.893, fp: 27, fn: 6, fn_edges: 6
Note: 86 nodes selected, 27 are FP fragments

---

## Batch 1 — Testing ILP cost parameters

Key finding: In skip_merge_hypotheses mode with no exclusion sets, all nodes are selected regardless of cost parameters. FP nodes are linked into continuous tracks, so appear/disappear penalties don't help. Only size_threshold affects results.

### B1R1: intensity_c=-3000 (fix positive cost mean)
exp_uid: 2026-04-24_20-11-30
Hypothesis: "Fixing intensity cost mean from +947 to -1052 will improve linking"
TRA: 0.860, DET: 0.855, LNK: 0.893, fp: 27, fn: 6, fn_edges: 6
Verdict: falsified — identical solution, intensity not discriminating

### B1R2: appear_c=500, disappear_c=500
exp_uid: 2026-04-24_20-16-12
Hypothesis: "Higher appear/disappear will suppress short FP tracks"
TRA: 0.860, DET: 0.855, LNK: 0.893, fp: 27, fn: 6, fn_edges: 6
Verdict: falsified — FP nodes are linked into continuous tracks, not isolated

### B1R3: curvature_w=31, curv_c=-1000
exp_uid: 2026-04-24_20-25-34
Hypothesis: "Curvature will penalize erratic FP tracks"
TRA: 0.860, DET: 0.855, LNK: 0.893, fp: 27, fn: 6, fn_edges: 6
Verdict: falsified — identical solution

### B1R4: appear_c=5000, disappear_c=5000
exp_uid: 2026-04-24_20-33-17
Hypothesis: "Extreme appear/disappear will suppress some FP tracks"
TRA: 0.860, DET: 0.855, LNK: 0.893, fp: 27, fn: 6, fn_edges: 6
Verdict: falsified — confirms FP nodes always linked into tracks

### B1R5: size_threshold=1000
exp_uid: 2026-04-24_20-38-12
Hypothesis: "Higher size filter will remove smallest FP fragments"
TRA: 0.865, DET: 0.862, LNK: 0.893, fp: 23, fn: 6, fn_edges: 6
Verdict: supported — first improvement! 80 nodes, 4 FPs removed

| Run | Changed Params | TRA | DET | LNK | fp | fn | fn_edges | Nodes |
|-----|---------------|------|------|------|----|----|----------|-------|
| B0R0 | baseline (thresh=500) | 0.860 | 0.855 | 0.893 | 27 | 6 | 6 | 86 |
| B1R1 | intensity_c=-3000 | 0.860 | 0.855 | 0.893 | 27 | 6 | 6 | 86 |
| B1R2 | appear/disappear=500 | 0.860 | 0.855 | 0.893 | 27 | 6 | 6 | 86 |
| B1R3 | curvature on | 0.860 | 0.855 | 0.893 | 27 | 6 | 6 | 86 |
| B1R4 | appear/disappear=5000 | 0.860 | 0.855 | 0.893 | 27 | 6 | 6 | 86 |
| B1R5 | size_thresh=1000 | 0.865 | 0.862 | 0.893 | 23 | 6 | 6 | 80 |

---

## Batch 2 — Size threshold optimization

### B2R1: size_threshold=20000
exp_uid: 2026-04-24_20-45-10
Hypothesis: "Threshold at 20k eliminates all small FP fragments (exactly 60 nodes = 4 cells × 15 frames)"
TRA: 0.890, DET: 0.890, LNK: 0.893, fp: 6, fn: 6, fn_edges: 6
Verdict: supported — big jump! 21 FPs eliminated. Remaining 6 FP+6 FN are CTC matcher mismatches.

### B2R2: size_threshold=10000
exp_uid: 2026-04-24_20-50-05
Hypothesis: "10k keeps a few borderline real fragments that 20k removes"
TRA: 0.887, DET: 0.887, LNK: 0.893, fp: 8, fn: 6, fn_edges: 6
Verdict: falsified — extra 4 nodes at 10k-20k range are all FPs, worse than 20k

### B2R3: size_threshold=50000
exp_uid: 2026-04-24_20-55-05
Hypothesis: "50k drops 1 more borderline node (59 nodes); if it's a FP, metrics improve"
TRA: 0.892, DET: 0.892, LNK: 0.893, fp: 5, fn: 6, fn_edges: 6
Verdict: supported — marginal improvement, the dropped node was a FP. Best so far.

### B2R4: size_threshold=20000, drift_c=-3000
exp_uid: 2026-04-24_21-00-03
Hypothesis: "Stronger edge encouragement will fix FN edges"
TRA: 0.890, DET: 0.890, LNK: 0.893, fp: 6, fn: 6, fn_edges: 6
Verdict: falsified — identical to B2R1, solver already selects all feasible edges

### B2R5: size_threshold=20000, max_edge_distance=100
exp_uid: 2026-04-24_21-04-58
Hypothesis: "FN edges are from cells too far apart; wider distance will add needed candidates"
TRA: 0.890, DET: 0.890, LNK: 0.893, fp: 6, fn: 6, fn_edges: 6
Verdict: falsified — 154 candidate edges (vs 140), but same solution. FN edges aren't distance-limited.

| Run | Changed Params | TRA | DET | LNK | fp | fn | fn_edges | Nodes |
|-----|---------------|------|------|------|----|----|----------|-------|
| B2R1 | size_thresh=20000 | 0.890 | 0.890 | 0.893 | 6 | 6 | 6 | 60 |
| B2R2 | size_thresh=10000 | 0.887 | 0.887 | 0.893 | 8 | 6 | 6 | 64 |
| B2R3 | size_thresh=50000 | **0.892** | **0.892** | 0.893 | **5** | 6 | 6 | 59 |
| B2R4 | thresh=20k, drift_c=-3000 | 0.890 | 0.890 | 0.893 | 6 | 6 | 6 | 60 |
| B2R5 | thresh=20k, max_edge=100 | 0.890 | 0.890 | 0.893 | 6 | 6 | 6 | 60 |
