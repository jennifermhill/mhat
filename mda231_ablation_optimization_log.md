# MDA231 Merge Ablation Optimization Log

Per-condition optimization of the 3 non-baseline merge ablation conditions on Fluo-C3DL-MDA231 / 01_cells. Starting point for each condition: the MDA231 baseline best config (drift_w=57, drift_c=-2000, area_w=1730, area_c=-1500, intensity_w=4, intensity_c=-1000, coh_w=2000, adh_w=-1000, appear/disappear=200) with `ablate_cohesion_adhesion = true` and the condition-specific `seg_result`.

Baseline metrics (for reference, no ablation): TRA=0.881, DET=0.886, LNK=0.845, fp=101, fn=28.

---

## no_cohesion (seg=2026-04-03_11-09-49)

Current toml uid: `2026-05-11_13-55-31`. TRA=0.8545, DET=0.8582, LNK=0.8268, fp=141, fn=36.

### Batch 1

| Run | Change | exp_uid | TRA | DET | LNK | fp | fn | Verdict |
|-----|--------|---------|-----|-----|-----|----|----|---------|
| B1R1 | appear/disappear=400 | 2026-05-11_15-17-23 | 0.8545 | 0.8582 | 0.8268 | 141 | 36 | reproduces baseline (insensitive) |
| B1R2 | drift_constant=-1000 | 2026-05-11_15-21-59 | 0.8494 | 0.8541 | 0.8147 | 141 | 37 | falsified — weaker drift hurts LNK |
| B1R3 | appear/disappear=800 | 2026-05-11_15-26-29 | 0.8545 | 0.8582 | 0.8268 | 141 | 36 | identical (insensitive) |
| B1R4 | curv_w=40, curv_c=-1275 (neutral) | 2026-05-11_15-31-02 | 0.8539 | 0.8580 | 0.8238 | 142 | 36 | within noise — curvature has negligible effect |
| B1R5 | intensity_w=10, intensity_c=-2000 | 2026-05-11_15-35-28 | 0.8412 | 0.8495 | 0.7805 | 143 | 38 | falsified — stronger intensity hurts |

**Best: no improvement over current.** Keep existing tracking_uid `2026-05-11_13-55-31`.

---

## no_affinities (seg=2026-04-28_13-43-53, symmetric scoring)

Current toml uid: `2026-05-11_14-07-43`. TRA=0.8542, DET=0.8580, LNK=0.8268, fp=142, fn=36.

### Batch 1

| Run | Change | exp_uid | TRA | DET | LNK | fp | fn | Verdict |
|-----|--------|---------|-----|-----|-----|----|----|---------|
| B1R1 | drift_w=30 | 2026-05-11_15-39-53 | 0.8494 | 0.8555 | 0.8046 | 146 | 36 | falsified — lower drift hurts |
| **B1R2** | **drift_w=100** | **2026-05-11_15-44-05** | **0.8548** | **0.8582** | **0.8298** | **141** | **36** | **improvement — best TRA + LNK** |
| B1R3 | drift_w=150 | 2026-05-11_15-48-25 | 0.8506 | 0.8555 | 0.8147 | 141 | 36 | falsified — too far past peak |
| B1R4 | area_w=2500 | 2026-05-11_15-52-26 | 0.8540 | 0.8599 | 0.8107 | 140 | 35 | DET up, LNK down — mixed |
| B1R5 | drift_w=100 + area_w=2500 | 2026-05-11_15-56-46 | 0.8542 | 0.8596 | 0.8147 | 136 | 35 | DET up, lowest fp, but lower LNK than R2 |

**Best by TRA: B1R2** (drift_w=100). Beats current on TRA (+0.0006), DET (+0.0002), LNK (+0.0030), same fp. **Update toml to 2026-05-11_15-44-05.**

---

## no_merges (seg=2026-04-23_12-51-45, skip_merges)

Current toml uid: `2026-05-11_14-06-59`. TRA=0.6292, DET=0.6338, LNK=0.5952, fp=233, fn=108.

### Batch 1

| Run | Change | exp_uid | TRA | DET | LNK | fp | fn | Verdict |
|-----|--------|---------|-----|-----|-----|----|----|---------|
| B1R1 | drift_w=100 | 2026-05-11_16-01-20 | 0.6305 | 0.6357 | 0.5921 | 226 | 108 | TRA/DET up, LNK -, fp -7 — mixed |
| B1R2 | appear/disappear=50 | 2026-05-11_16-05-29 | 0.6292 | 0.6338 | 0.5952 | 233 | 108 | identical to current (insensitive) |
| B1R3 | area_w=2500 | 2026-05-11_16-09-57 | 0.6296 | 0.6357 | 0.5851 | 226 | 108 | TRA/DET up, LNK down |
| B1R4 | drift_c=-3000 | 2026-05-11_16-14-06 | 0.6289 | 0.6335 | 0.5952 | 234 | 108 | falsified — stronger drift_c slightly worse |
| **B1R5** | **drift_w=100 + area_w=2500** | **2026-05-11_16-18-20** | **0.6316** | **0.6365** | **0.5952** | **223** | **108** | **best on TRA+DET, LNK preserved, fp -10** |

**Best: B1R5** (drift_w=100 + area_w=2500). Beats current on TRA (+0.0024), DET (+0.0027), LNK same, fp -10. **Update toml to 2026-05-11_16-18-20.**

### Batch 2 (cellpose seg, 2026-05-12)

After Batch 1, user pointed out an older much-stronger no_merges result (TRA=0.836) on a different seg. Search of evaluation outputs found it: `2026-04-23_12-54-08` on `C:/Users/hillj/Documents/mhat/...`, using seg `2026-04-23_11-58-21`. The difference vs the otsu seg used in Batch 1: `seg_method = "cellpose"` (vs `otsu`). Eval copied to Y:; toml updated to point at the cellpose seg + that tracking uid as the new starting point (TRA=0.8362, DET=0.8412, LNK=0.7996, fp=153, fn=41). Batch 2 sweeps for further improvement on top of cellpose seg.

| Run | Change | exp_uid | TRA | DET | LNK | fp | fn | Verdict |
|-----|--------|---------|-----|-----|-----|----|----|---------|
| **B2R1** | **drift_w=100** | **2026-05-12_10-15-29** | **0.8367** | **0.8418** | **0.7996** | **151** | **41** | **best (tied with B2R5) — small improvement, fp -2** |
| B2R2 | drift_w=100 + area_w=2500 | 2026-05-12_10-20-27 | 0.8360 | 0.8418 | 0.7936 | 151 | 41 | LNK hurt |
| B2R3 | area_w=2500 | 2026-05-12_10-24-54 | 0.8355 | 0.8418 | 0.7895 | 151 | 41 | LNK hurt |
| B2R4 | drift_w=100 + intensity_w=8, intensity_c=-2000 | 2026-05-12_10-29-04 | 0.8352 | 0.8418 | 0.7875 | 151 | 41 | LNK hurt |
| B2R5 | drift_w=75 | 2026-05-12_10-33-20 | 0.8367 | 0.8418 | 0.7996 | 151 | 41 | identical to B2R1 (quantized) |

**Best: B2R1** (drift_w=100). Beats prior cellpose-seg baseline by +0.0005 TRA, +0.0006 DET, same LNK, fp -2. drift_w ∈ [75, 100] is a single quantized regime. **Update toml to 2026-05-12_10-15-29.**

Pattern confirmation: drift_w=100 transfers across no_affinities, no_merges (otsu), and no_merges (cellpose). The merge-ablated conditions consistently prefer drift_w=100 over the baseline 57. Combining drift_w=100 with stronger area or intensity hurts LNK.

---

## Summary of Batch 1 (15 runs)

| Condition | Current TRA/DET/LNK | Best TRA/DET/LNK | Improvement? | New uid |
|-----------|---------------------|------------------|--------------|---------|
| no_cohesion | 0.8545 / 0.8582 / 0.8268 | (no improvement) | — | unchanged |
| no_affinities | 0.8542 / 0.8580 / 0.8268 | 0.8548 / 0.8582 / 0.8298 | yes | 2026-05-11_15-44-05 |
| no_merges | 0.6292 / 0.6338 / 0.5952 | 0.6316 / 0.6365 / 0.5952 | yes | 2026-05-11_16-18-20 |

### Cross-condition patterns

- **appear/disappear is insensitive** in [50, 800] across all three conditions (matches baseline optimization).
- **drift_weight=100 transfers across conditions** — gave the best result on both no_affinities and (combined with area_w=2500) no_merges. Suggests the baseline drift_w=57 is sub-optimal once cohesion/adhesion are ablated.
- **curvature, intensity perturbations don't help** in the ablated regime.
- **The 3-condition fp gap is large**: baseline fp=101, no_cohesion/no_affinities fp~141, no_merges fp~226. Most of the "missing" detection performance comes from the absence of merged-fragment hypotheses (no_merges) or the loss of cohesion-based node scoring (the other two).

---

## Side experiment: Symmetric seg + full costs (2026-05-12)

**Question:** is the symmetric scoring function (used for the `- affinities` condition seg) actually a *better* merge logic than the default affinity scoring? If so, putting cohesion/adhesion back on top of the symmetric seg should beat baseline.

**Setup:** seg = `2026-04-28_13-43-53` (symmetric scoring), `ablate_cohesion_adhesion = false`, baseline ILP params as starting point.

### Graph stats (R1, weight=2000 cohesion / -1000 adhesion)

| Attribute | Mean | Std | Cost Mean | Cost Std |
|-----------|------|-----|-----------|----------|
| cohesion | 0.917 | 0.219 | 2136.5 | 729.3 |
| adhesion | 0.764 | 0.356 | -1023.3 | 817.2 |

vs baseline seg (affinity): cohesion mean=0.984, std=0.259; adhesion mean=0.903, std=0.238. Symmetric seg has **lower-confidence merges on average** and a wider spread on adhesion.

### Batch 1 results

| Run | Change | exp_uid | TRA | DET | LNK | fp | fn | ns | Verdict |
|-----|--------|---------|-----|-----|-----|----|----|-----|---------|
| MDA231 Baseline | affinity seg, full costs | 2026-04-15_09-42-38 | **0.881** | **0.886** | **0.845** | 101 | 28 | 7 | target to beat |
| **R1** | baseline params on symmetric seg | 2026-05-12_10-52-42 | 0.873 | 0.882 | 0.809 | 96 | 27 | 13 | best in batch, still -0.008 TRA vs baseline |
| R2 | cohesion_w=4000 | 2026-05-12_10-57-58 | 0.565 | 0.576 | 0.490 | 20 | 146 | 13 | catastrophic — too much merge penalty |
| R3 | adhesion_w=-2000 | 2026-05-12_11-02-00 | 0.869 | 0.883 | 0.769 | 86 | 24 | 20 | DET marginally up; LNK and ns much worse |
| R4 | drift_w=100 | 2026-05-12_11-07-16 | 0.864 | 0.873 | 0.800 | 93 | 31 | 12 | drift_w=100 doesn't transfer with full costs |
| R5 | cohesion_w=1000 | 2026-05-12_11-12-01 | 0.872 | 0.878 | 0.824 | 113 | 28 | 10 | weaker merge penalty: better LNK, worse DET, more fp |

### Conclusion

**Hypothesis falsified:** the symmetric scoring seg does *not* outperform the affinity scoring seg on MDA231 when both are run with full cohesion/adhesion costs. The TRA gap is 0.008 (mostly LNK -0.036), and none of the parameter variations closed it.

Notable cross-experiment finding: **drift_w=100, which was the consistent winner across all three merge-ablated conditions, hurts in the full-cost regime** (R4 dropped LNK by 0.009 vs R1). The optimal drift weight depends on whether cohesion/adhesion are active.
