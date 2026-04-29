# Fluo-C3DH-H157 / 01_cells — Optimization Working Memory

## Current Best Config
- **TRA: 0.892, DET: 0.892, LNK: 0.893** (B2R3, size_threshold=50000)
- drift_w=57, drift_c=-2000, area_w=1730, area_c=-1500, intensity_w=4, intensity_c=-1000
- curvature_w=0, curvature_c=0, appear_c=200, disappear_c=200
- size_threshold=50000, max_edge_distance=60, skip_merge_hypotheses=true
- fp=5, fn=6, fn_edges=6, 59 nodes

## Results Comparison Table
| Run | size_thresh | Other changes | TRA | DET | LNK | fp | fn | Nodes |
|-----|------------|---------------|------|------|------|----|----|-------|
| B0R0 | 500 | baseline | 0.860 | 0.855 | 0.893 | 27 | 6 | 86 |
| B1R1-R4 | 500 | various ILP costs | 0.860 | 0.855 | 0.893 | 27 | 6 | 86 |
| B1R5 | 1000 | — | 0.865 | 0.862 | 0.893 | 23 | 6 | 80 |
| B2R1 | 20000 | — | 0.890 | 0.890 | 0.893 | 6 | 6 | 60 |
| B2R2 | 10000 | — | 0.887 | 0.887 | 0.893 | 8 | 6 | 64 |
| **B2R3** | **50000** | — | **0.892** | **0.892** | 0.893 | **5** | 6 | 59 |
| B2R4 | 20000 | drift_c=-3000 | 0.890 | 0.890 | 0.893 | 6 | 6 | 60 |
| B2R5 | 20000 | max_edge=100 | 0.890 | 0.890 | 0.893 | 6 | 6 | 60 |

## Established Principles
1. **ILP costs don't matter in skip_merge_hypotheses mode** — without exclusion sets, every node is selected and every feasible edge is chosen. Tested intensity, appear/disappear (up to 5000), curvature, drift. All identical.
2. **size_threshold is the only effective lever** — directly filters FP fragments before graph construction.
3. **Real cells are 100k-495k voxels; FPs are <20k** — clear size gap in this dataset.
4. **fn=6 and fn_edges=6 are stuck** — these are likely CTC matcher mismatches where predicted segments don't overlap GT segments >50%. Cannot be fixed by ILP tuning.
5. **LNK=0.893 never changes** — linking is at its ceiling given the node selection.

## Falsified Hypotheses
- Intensity cost sign doesn't affect solution
- Appear/disappear constants don't affect solution (FP nodes are linked into continuous tracks)
- Curvature doesn't affect solution
- Stronger drift_c doesn't help
- Wider max_edge_distance doesn't help

## Plateau Assessment
**We have reached a plateau.** The remaining errors (5 FP, 6 FN, 6 FN edges) are segmentation-level issues:
- The 6 FN nodes are segments that don't sufficiently overlap their GT cell (CTC requires >50% IoU)
- The 5 FP nodes are segments that overlap with multiple GT cells or none
- No ILP parameter can fix these — they require better segmentation

## Open Questions
1. Could multi-hypothesis selection (re-enabling merge hypotheses) fix the 6 FN mismatches? Under-merged segments might match GT better as smaller fragments.
2. Would a different segmentation result help? The 2026-04-21_13-36-29 seg was slightly more fragmented.
3. Should we run the full 60 timepoints with size_threshold=50000 to get the real metric?
