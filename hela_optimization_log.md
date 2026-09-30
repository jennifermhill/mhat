# Fluo-N2DL-HeLa / 01_cells — tracking optimization log

Setup (all runs): seg `2026-09-23_11-18-04` (cellpose, `cellprob_threshold = 0`, `merge_thresholds = [1.0]`, fixed waterz convention), 2D flow `2026-09-23_15-37-38` (Farneback), `max_edge_distance = 50`, `max_children = 5`, `size_threshold = 20`, `divisions = true`, cohesion −500/450, adhesion −100/50 (both inert here: cost std 39 / 28). Graph cache `graph_cache/Fluo-N2DL-HeLa/01_cells/1966f34d657ca21d.pkl`. Eval: `metrics = ["ctc"]`, `matcher = "ctc"`. Runs solved directly on the compute node with `mhat-cluster` + Gurobi 13 (2026-09-23); configs under `scratch_configs/tracking/Fluo-N2DL-HeLa/01_cells/hela_*_config.toml`, eval configs beside them under `scratch_configs/evaluation/`.

GT: 265 tracks, 94 divisions, 8639 nodes, 8562 edges, 92 frames, 1.6 µm/px, 10 min/frame.

AOGM weights (CTC): fn_node 10, ns 5, fp_node 1, fn_edge 1.5, fp_edge 1, ws 1.

## B0: Jennifer's runs (2026-09-23_15-40-42, 2026-09-23_15-44-52)

Both give AOGM 5514.5 (TRA 0.9444 / DET 0.9524 / LNK 0.8911, fp 806, fn 323, ns 16, fn_edges 915, ws 20). `hela_B1R0` (dw 350 / dc −2000 / app = dis 50) reproduces this exactly; `hela_B1R1` with the params recorded in run 15-40-42's `config.toml` (dw 100, app 100) does **not** (AOGM 4773) — that `config.toml` was rewritten at 15:44:45, after the run, so both of her runs were effectively the dw 350 config.

Diagnosis (fn_edges / fn_nodes diagnostics on 15-40-42, plus a per-division script):
- 518 of 522 FN edges with both nodes matched were candidates the ILP rejected; in 505 the solver simply ended the track and started a new one. At dw 350 / dc −2000 / app + dis 100, an edge only beats a break when drift_dist < 6 µm. TP edges have drift 1.75 µm mean; the rejected ones 6–40 µm (median 10.9).
- 266 of the 518 sit within 3 edges of a GT division: the edge into the mitotic frame is lost in 52/94 divisions (the plate's centroid jumps 7–20 µm and the flow there is unreliable), the division edges themselves are fully correct in only 14/94 (both daughters linked), and the daughters' first 1–3 edges are lost too (drift 7–16 µm while the daughters are still moving apart).
- The other 252 rejected edges are interphase jumps of 6–40 µm.

## B1: drift break-even (all on the cached graph)

| run | dw | dc | app | dis | TRA | DET | LNK | AOGM | fp | fn | fn_e | ws |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1R0 (= Jennifer) | 350 | −2000 | 50 | 50 | 0.9444 | 0.9524 | 0.8911 | 5514.5 | 806 | 323 | 915 | 20 |
| B1R1 | 100 | −2000 | 100 | 50 | 0.9519 | 0.9535 | 0.9409 | 4773 | 814 | 314 | 462 | 55 |
| B1R2 | 100 | −2000 | 50 | 50 | 0.9531 | 0.9550 | 0.9408 | 4651.5 | 821 | 301 | 463 | 55 |
| B1R3 | 100 | −2000 | 400 | 400 | 0.9524 | 0.9537 | 0.9436 | 4723 | 804 | 314 | 438 | 53 |
| B1R4 | 50 | −2000 | 400 | 400 | 0.9534 | 0.9542 | 0.9478 | 4629 | 809 | 310 | 406 | 44 |

Hypothesis "the breaks are a break-even problem": supported — fn_edges 915 → 406 with fp_edges only 6 → 17. On B1R3 the edge into the mitotic frame is TP in 83/94 divisions; the remaining division loss is 33 divisions where only one daughter is linked (ws_edges).

Why one daughter: on these the daughters are asymmetric about the plate (near daughter 6–12 µm from the mother's flow-predicted position, far one 37–60 µm, midpoint 15–30 µm — verified numerically and by eye, no flow-order or scale bug). The hyperedge is scored on the midpoint, and the flow at the plate follows the near mass, so "near edge + appear(far daughter)" always undercuts the hyperedge unless appear is large or the hyperedge gets its own credit.

## B2 / B3: division trade-off

| run | dw | dc | app | dis | divw | TRA | DET | LNK | AOGM | fp_e | fn_e | ws |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2R1 | 50 | −2000 | 400 | 400 | −300 | 0.9536 | 0.9543 | 0.9495 | 4599.5 | 18 | 395 | 38 |
| B2R2 | 50 | −2000 | 400 | 400 | +300 | 0.9534 | 0.9544 | 0.9465 | 4629 | 9 | 418 | 51 |
| B2R3 | 50 | −2000 | 800 | 800 | 0 | 0.9537 | 0.9542 | 0.9503 | 4597 | 19 | 390 | 34 |
| B2R4 | 30 | −2000 | 400 | 400 | 0 | 0.9537 | 0.9543 | 0.9498 | 4597 | 16 | 394 | 38 |
| B2R5 | 50 | −1000 | 400 | 400 | 0 | 0.9526 | 0.9536 | 0.9458 | 4702 | 18 | 422 | 45 |
| B3R1 | 50 | −2000 | 800 | 400 | 0 | 0.9535 | 0.9541 | 0.9497 | 4614.5 | 21 | 393 | 35 |
| B3R2 | 50 | −2000 | 800 | 800 | −300 | 0.9537 | 0.9542 | 0.9504 | 4597 | 26 | 384 | 35 |
| B3R3 | 50 | −2000 | 800 | 800 | −600 | 0.9538 | 0.9543 | 0.9506 | 4584.5 | 27 | 381 | 36 |
| **B3R4** | **30** | **−2000** | **800** | **800** | **0** | **0.9538** | **0.9543** | **0.9510** | **4581.5** | 25 | 379 | 36 |
| B3R5 | 50 | −2000 | 1200 | 1200 | 0 | 0.9529 | 0.9533 | 0.9506 | 4672 | 24 | 386 | 32 |

fp / fn / ns nodes are 809–812 / 309–310 / 10 on every row except B3R5 (797 / 319). Appear/disappear bracketed at 800 (400 and 1200 both worse); dw 30 vs 50 and divw 0 … −600 are within 1–3 edges of each other (plateau AOGM 4581–4597). **Current best: B3R4** — `drift_weight 30, drift_constant −2000, appear = disappear 800, division_weight 0`, rest as base. Per-division on B3R4: 78/94 divisions fully correct, the edge into the mitotic frame TP in 86/94, only 13 ILP-rejected edges left in the whole movie (drift 8–45 µm). SEG 0.7761 (0.7726 on B0).

## Segmentation arms (B1R3 / B2R3 costs)

| run | seg | cellprob | TRA | DET | LNK | AOGM | fp | fn (seg miss) | fn_e |
|---|---|---|---|---|---|---|---|---|---|
| B2_cpm3 | 2026-09-23_11-20-00 | −3 | 0.9426 | 0.9444 | 0.9304 | 5693.5 | 675 | 407 (371) | 561 |
| B2_cp3 | 2026-09-23_11-15-08 | +3 | 0.8973 | 0.8997 | 0.8810 | 10190 | 602 | 804 | 980 |

cellprob −3 does **not** recover the dim cells (same tracks missed) and newly loses bright cells in dense regions; cellprob 0 stays.

## What is left, and why it is not a cost problem (B3R4, AOGM 4581.5)

- **fn_nodes 309 × 10 = 3090 (67 %)**: 282 are "segmentation misses" — GT tracks 27, 42, 43, 209, 176, 122, 133, 175, 66, … whose raw intensity at the GT centroid (~33 090) is the background level (33 072; TP nodes ~33 500). These cells are essentially invisible; no cellprob setting tested finds them.
- **fp_nodes 812 (18 %)**: 532 are within 16 µm of the image border with no GT node within 25 µm (border cells the GT does not annotate), 216 are bright interior nuclei with no GT annotation (verified by eye: cell-sized, well segmented, no marker). Only 37 are extra pieces of a matched cell and 24 FN+FP pairs. They are chained into long tracks, so no appear/disappear value removes them without removing true cells.
- fn_edges 379 × 1.5: ~330 are edges of the unmatched nodes above.

Possible follow-ups, not done (Jennifer's call): (a) a division-specific drift for hyperedges (e.g. no flow correction, or min over the daughters instead of the midpoint) — the asymmetry above makes the midpoint-vs-flow-predicted measure systematically pessimistic, and it is a `src/mhat` change (mhat_dicty); (b) held-out `02_cells` with the B3R4 params, once.

## CTC BIO measure (2026-09-23, after the AOGM batches)

BIO = mean(CT, TF, BC(i), CCA). All four components are in traccuracy 0.4.3 and already in `metrics_dict` (`complete_tracks`, `track_overlap`, `division`, `cca`), so nothing has to be reimplemented, but two things bite when they run on the CTC matching: `DivisionMetrics` only accepts one-to-one matchings and the CTC matcher is many-to-one here (10 ns nodes), so it must be computed with `override_matcher=True`; and `CellCycleAccuracy._get_lengths` iterates `location_keys` as a sequence, which on MHAT graphs is the composite string `"pos"` (→ `KeyError: 'p'`) — CCA only uses the frame column, so blanking `location_keys` for that call is exact. Both would also break `evaluate_tracks.py` with `metrics = ["division"]` / `["cca"]` as it stands. Script: `scratch_configs/experiments/hela_bio/bio.py`. Mapping: CT = `complete_tracklets` (tracklets between divisions, `error_type="ctc"`), TF = `track_fractions`, BC(i) = `Division F1` at frame buffer i (flat from i = 1 on this data), CCA = `CCA`. GT has 56 complete cell cycles (29–70 frames).

| run | dw | app=dis | divw | CT | TF | BC(1) | div rec / prec (1) | CCA | **BIO(1)** | AOGM |
|---|---|---|---|---|---|---|---|---|---|---|
| Jennifer's | 350 | 50 | 0 | 0.302 | 0.766 | 0.221 | 0.15 / 0.42 | 0.000 | **0.322** | 5514.5 |
| B1R4 | 50 | 400 | 0 | 0.823 | 0.923 | 0.647 | 0.60 / 0.71 | 0.762 | 0.789 | 4629 |
| **B2R3** | 50 | 800 | 0 | 0.808 | 0.923 | 0.714 | 0.76 / 0.68 | 0.778 | **0.805** | 4597 |
| B3R1 | 50 | 800/400 | 0 | 0.808 | 0.923 | 0.710 | 0.76 / 0.67 | 0.778 | 0.802 | 4614.5 |
| B3R4 (AOGM best) | 30 | 800 | 0 | 0.781 | 0.916 | 0.727 | 0.85 / 0.63 | 0.702 | 0.782 | 4581.5 |
| B3R3 | 50 | 800 | −600 | 0.781 | 0.919 | 0.726 | 0.87 / 0.62 | 0.685 | 0.778 | 4584.5 |
| B4R1 | 50 | 800 | +300 | 0.819 | 0.924 | 0.671 | 0.65 / 0.72 | 0.767 | 0.801 | 4619 |
| B4R2 | 50 | 800 | +600 | 0.834 | 0.930 | 0.591 | 0.47 / 0.80 | 0.800 | 0.787 | 4638.5 |
| B4R3 | 50 | 800 | +1000 | 0.838 | 0.930 | 0.507 | 0.37 / 0.80 | 0.875 | 0.786 | 4659 |
| B4R4 | 100 | 800 | 0 | 0.796 | 0.918 | 0.620 | 0.56 / 0.69 | 0.763 | 0.774 | 4708.5 |
| B4R5 | 100 | 800 | +300 | 0.800 | 0.923 | 0.599 | 0.50 / 0.75 | 0.720 | 0.759 | 4705.5 |
| B4R6 | 50 | 1200 | +600 | 0.815 | 0.924 | 0.682 | 0.67 / 0.72 | 0.773 | 0.804 | 4685.5 |

(full component table incl. CT_lineages, TE, BC(0), BC(2): `scratch_configs/experiments/hela_bio/bio_batches1-3.json` and the bio.py output.)

Reading: BIO plateaus at 0.80 (B2R3 / B4R6 / B3R1 / B4R1) and the plateau is a genuine trade-off, not noise. The permissive linking that recovers the true divisions (division recall 0.15 → 0.76–0.87) also manufactures false ones (34–48 FP divisions): on B3R4, 22 have an unannotated object grabbed as a daughter through the hyperedge, 14 are spurious splits among matched cells, 7 are unannotated cells dividing. Each false division creates a 2–10-frame "cell cycle" (pred has 84 complete cycles vs 56 in GT), which is what pulls CCA and BC precision down. A positive `division_weight` removes false divisions but loses true ones faster (B4R2/R3: precision 0.80, recall 0.37–0.47) because with the current features nothing separates them: hyperedge drift, area_diff and intensity_diff overlap (`fpdiv_feat.py`). What *does* separate them is daughter symmetry — min/max daughter area is 0.91 median (p10 0.74) on true divisions vs 0.60 (p10 0.18) on false ones, and daughter separation 50 µm vs 37 µm — but no hyperedge cost uses it. That is a `src/mhat` addition (a daughter-area-ratio attribute on division hyperedges + a cost), so it is Jennifer's call on mhat_dicty.

**Recommended single operating point: B2R3** (dw 50, dc −2000, appear = disappear 800, divw 0): BIO 0.805 (best), AOGM 4597 (within 16 of the AOGM-best B3R4, i.e. ~10 edges). `division_weight` stays 0.

## Segmentation: the SEG gap and the cellprob / flow_threshold sweep (2026-09-23, later)

SEG 0.776 vs published ~0.92 is the cellpose masks themselves (the raw cellprob-0 fragments score 0.779; selection changes nothing). Per annotated object (491 in 28 `01_GT/SEG` frames): 33 unmatched (30 % of the deficit; dim cells and mitotic/anaphase figures annotated as one generous outline), and matched masks systematically smaller than the annotations (area ratio 0.875 at cellprob 0, IoU median 0.85; 1.15 at cellprob −3, IoU no better). Sweep on the network output of the four densest annotated frames (13/52/76/88, 236 objects; cpsam run once per frame, `dynamics.compute_masks` re-run per threshold; script `scratch_configs/experiments/hela_bio/cp_thresh.py`):

| cellprob | flow_thr | SEG (4 frames) | unmatched | IoU med | area ratio |
|---|---|---|---|---|---|
| −3 | 0.4 | 0.796 | 5.5 % | 0.849 | 1.16 |
| −2 | 0.4 | 0.853 | 3.4 % | 0.897 | 1.06 |
| −1.5 | 0.4 | 0.863 | 3.0 % | 0.903 | 1.02 |
| −1 | 0.4 | 0.863 | 2.5 % | 0.902 | 0.98 |
| 0 (Jennifer's) | 0.4 | 0.816 | 4.7 % | 0.875 | 0.89 |
| −1.5 | 1.0 | **0.877** | 0.8 % | 0.901 | 1.01 |
| −2 | 1.0 | 0.873 | 0.8 % | 0.896 | 1.06 |

Jennifer ran the full seg on GPU: **`hela_cpm15_ft10`** (cellprob −1.5, flow_threshold 1.0, config `scratch_configs/segmentation/Fluo-N2DL-HeLa/01_cells/hela_cpm15_ft10_config.toml`). Graph cache `dbaeb9b56c60222f` (size 20). flow_threshold 1.0 keeps ~430 small masks cellpose would reject (FP areas p10 37 / p25 83 px vs TP p1 148 px), which feed false divisions; `size_threshold` 60 removes 241 of them for 4 true nodes (80: 295 / 12; the mitotic plates are all > 148 px).

| run | size | app=dis | divw | TRA | DET | LNK | SEG | AOGM | fp | fn | ws | CT | CCA | BC(1) | rec / prec | **BIO(1)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2R3 (old seg) | 20 | 800 | 0 | 0.9537 | 0.9542 | 0.9503 | 0.7761 | 4597 | 809 | 310 | 34 | 0.808 | 0.778 | 0.714 | 0.76 / 0.68 | 0.805 |
| B5_cpm15 | 20 | 800 | 0 | 0.9635 | 0.9633 | 0.9649 | 0.8497 | 3621 | 1240 | 187 | 47 | 0.834 | 0.636 | 0.66 | 0.83 / 0.54 | 0.767 |
| B5R1 | 20 | 800 | +300 | 0.9634 | 0.9634 | 0.9637 | 0.8497 | 3632 | 1226 | 188 | 46 | 0.849 | 0.700 | 0.66 | 0.71 / 0.61 | 0.789 |
| B5R2 | 20 | 800 | +600 | 0.9632 | 0.9634 | 0.9618 | 0.8497 | 3655.5 | 1225 | 188 | 49 | 0.868 | 0.815 | 0.61 | 0.54 / 0.70 | 0.810 |
| B5R3 | 20 | 1200 | +600 | 0.9627 | 0.9626 | 0.9632 | 0.8464 | 3703 | 1175 | 199 | 44 | 0.845 | 0.737 | 0.66 | 0.71 / 0.63 | 0.799 |
| B5R4 | 20 | 800 | +1000 | 0.9629 | 0.9634 | 0.9597 | 0.8497 | 3679 | 1221 | 188 | 58 | 0.879 | 0.824 | 0.53 | 0.39 / 0.78 | 0.790 |
| **B5S1** | **60** | **800** | **0** | **0.9656** | **0.9656** | **0.9654** | 0.8475 | 3417 | 983 | 193 | 34 | 0.845 | 0.757 | 0.72 | 0.82 / 0.65 | 0.819 |
| B5S2 | 60 | 800 | +300 | 0.9653 | 0.9655 | 0.9638 | 0.8475 | 3445.5 | 981 | 194 | 40 | 0.853 | 0.796 | 0.69 | 0.70 / 0.70 | **0.824** |
| B5S3 | 80 | 800 | 0 | 0.9657 | 0.9658 | 0.9648 | 0.8459 | 3404.5 | 933 | 195 | 33 | 0.834 | 0.741 | 0.72 | 0.82 / 0.67 | 0.816 |

**Current best: B5S1** — seg `hela_cpm15_ft10`, `size_threshold = 60`, dw 50 / dc −2000 / appear = disappear 800 / divw 0: TRA 0.9656 / DET 0.9656 / LNK 0.9654 / SEG 0.8475, BIO(1) 0.819. B5S2 (divw +300) is the BIO best by 0.005 at the cost of ~19 edges of AOGM — a coin flip; B5S3 (size 80) the AOGM best by 12. Remaining FN nodes: 165 seg misses, still the background-level tracks 27 / 42 / 43. The fallback seg config with flow_threshold 0.4 (`hela_cpm15_ft04_config.toml`) is staged but not run: the size threshold already removes the FP objects flow_threshold 1.0 let in, and its lower unmatched rate is worth ~0.02 SEG.

## B6: cellpose flow_threshold sweep at cellprob −1.5 (2026-09-24)

B5S1's seg (`hela_cpm15_ft10`, flow_threshold 1.0) leaves segmentation artifacts, so cellprob −1.5 was re-segmented at flow_threshold 0.4 / 0.6 / 0.8 (then 0.7 / 0.9), Jennifer's GPU runs, configs `scratch_configs/segmentation/Fluo-N2DL-HeLa/01_cells/hela_cpm15_ft0*_config.toml`. Each seg is tracked with the B5S1 params (`hela_B6_ft0*`, size_threshold 60, dw 50 / dc −2000 / app = dis 800 / divw 0); 04/06/08 directly on the compute node, 07/09 through the sweep watcher (`scratch_configs/sweeps/hela_B6_ft0709.toml`). Scripts and logs: `scratch_configs/experiments/hela_ft_sweep/`.

Fragment-level scores (`seg_fp_fn.py`, cellpose fragments vs the TRA markers on all 92 frames with the CTC > 50 % cover rule; SEG on the 28 annotated frames):

| flow_thr | fragments | < 60 px | fn | fp | fp ≥ 60 px | ns | SEG | unmatched |
|---|---|---|---|---|---|---|---|---|
| 0.4 | 9223 | 54 | 253 | 847 | 795 | 10 | 0.835 | 4.5 % |
| 0.6 | 9485 | 206 | 192 | 1049 | 845 | 11 | 0.851 | 2.4 % |
| 0.8 | 9590 | 278 | 177 | 1139 | 865 | 11 | 0.853 | 2.0 % |
| 1.0 | 9944 | 360 | 176 | 1492 | 1136 | 11 | 0.853 | 2.0 % |
| 0.4 @ cellprob 0 (old seg) | 9198 | 134 | 291 | 860 | 738 | 10 | 0.779 | 6.3 % |

Tracking (B5S1 params):

| run | flow_thr | TRA | DET | LNK | SEG | AOGM | fp | fn | ns | fp_e | fn_e | ws |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B6_ft04 | 0.4 | 0.9589 | 0.9594 | 0.9560 | 0.8289 | 4075 | 775 | 268 | 11 | 18 | 344 | 31 |
| B6_ft06 | 0.6 | 0.9657 | 0.9660 | 0.9636 | 0.8469 | 3408 | 820 | 206 | 12 | 15 | 280 | 33 |
| B6_ft07 | 0.7 | 0.9666 | 0.9668 | 0.9648 | 0.8469 | 3319 | 827 | 198 | 12 | 14 | 270 | 33 |
| **B6_ft08** | **0.8** | **0.9669** | **0.9672** | **0.9652** | **0.8475** | **3284.5** | 832 | 194 | 13 | 14 | 267 | 33 |
| B6_ft09 | 0.9 | 0.9666 | 0.9668 | 0.9652 | 0.8475 | 3318.5 | 866 | 194 | 13 | 14 | 267 | 33 |
| B5S1 | 1.0 | 0.9656 | 0.9656 | 0.9654 | 0.8475 | 3417 | 983 | 193 | 12 | 14 | 264 | 34 |

Hypothesis "an interior flow_threshold keeps 1.0's detections with fewer spurious fragments": supported at 0.8 — fn 194 vs 193 and identical SEG, 151 fewer fp nodes, AOGM 3417 → 3284.5 (new best, TRA 0.9669). 0.4 is off the table (75 more fn nodes at 10× the weight of the fp they remove); 0.6 is between. 0.7 / 0.9 (sweep `hela_B6`, LSF jobs 154448065-68) bracket it: 0.7 loses 4 more cells (AOGM 3319), 0.9 keeps the same fn but lets 34 more fp nodes through (3318.5). Optimum bracketed at 0.8; the fn count is flat from 0.8 up and fp rises monotonically with the threshold. **Current best: B6_ft08** — seg `hela_cpm15_ft08` (cellprob −1.5, flow_threshold 0.8) + B5S1 params, TRA 0.9669 / DET 0.9672 / LNK 0.9652 / SEG 0.8475, AOGM 3284.5. BIO not yet computed on it.

BIO on the new best (`bio.py`, same two workarounds; `scratch_configs/experiments/hela_bio/bio_B6_ft08.txt`):

| run | CT | TF | BC(1) | div rec / prec (1) | CCA | **BIO(1)** | AOGM |
|---|---|---|---|---|---|---|---|
| B5S1 (ft 1.0) | 0.845 | 0.939 | 0.733 | 0.82 / 0.66 | 0.757 | 0.819 | 3417 |
| **B6_ft08** | 0.842 | 0.938 | 0.762 | 0.82 / 0.71 | 0.769 | **0.828** | 3284.5 |

BIO improves with the seg change too (0.819 → 0.828): the same 77/94 true divisions are found, but the removed small fragments were feeding false divisions (division precision 0.66 → 0.71, fewer spurious short "cell cycles", so CCA 0.757 → 0.769). CT dips by one tracklet (223 vs 224 of 265). B6_ft08 is now best on AOGM and BIO alike.

## Held-out 02_cells with the B6_ft08 recipe (2026-09-24, run once)

02_cells ingested from `ctc/Fluo-N2DL-HeLa/02` (`scripts/ctc_to_zarr.py`, 92 frames, 700 × 1100, 1.6 µm, 10 min/frame; `02_GT` copied beside it, GT geff built by the eval: 674 raw tracks, 25 131 nodes, 25 168 edges — 3× the 01_cells movie). Seg `hela_cpm15_ft08` (cellprob −1.5, flow_threshold 0.8, Jennifer's GPU run), 2D flow `2026-09-24_14-59-41` (01_cells settings), tracking `hela_holdout_ft08` = the B6_ft08 params verbatim (configs under `scratch_configs/*/Fluo-N2DL-HeLa/02_cells/`). Nothing tuned here.

| run | TRA | DET | LNK | SEG | AOGM | fp | fn | ns | fp_e | fn_e | ws |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01_cells B6_ft08 (train) | 0.9669 | 0.9672 | 0.9652 | 0.8475 | 3284.5 | 832 | 194 | 13 | 14 | 267 | 33 |
| **02_cells hela_holdout_ft08** | **0.9765** | **0.9773** | **0.9707** | **0.8788** | 6870 | 3515 | 162 | 126 | 49 | 622 | 123 |

The recipe transfers: every CTC measure is higher on the held-out movie (published CTC leaderboard numbers for this dataset are in the same range — check before quoting). Error mix differs: fp nodes dominate the raw count (3515, weight 1 — unannotated cells again, the movie is denser) and ns rises to 126 (weight 5), i.e. under-segmentation of touching nuclei is the main 02_cells-specific cost; fn stays low (162 of 25 131).

BIO on the held-out run (`bio.py` with `HELA_DATASET=02_cells`; traccuracy's `DivisionMetrics` shifted-division correction calls the single-match getter and raises on ns pred nodes, so divisions are scored on a one-to-one-filtered copy of the matching — 253 pairs dropped on 02_cells, 26 on 01_cells where the numbers are unchanged; `bio_holdout_ft08.txt`):

| run | CT | TF | BC(1) | div rec / prec (1) | CCA | **BIO(1)** |
|---|---|---|---|---|---|---|
| 01_cells B6_ft08 | 0.842 | 0.938 | 0.762 | 0.82 / 0.71 | 0.769 | 0.828 |
| 02_cells hela_holdout_ft08 | 0.775 | 0.940 | 0.595 | 0.75 / 0.49 | 0.585 | **0.724** |

BIO transfers less well than the CTC measures: division precision halves (0.49 — the denser 02 movie has many more unannotated cells and touching nuclei for the hyperedges to grab as daughters), which also floods CCA with short false cell cycles (0.585). The `division_weight` / daughter-symmetry follow-ups from the B4/B5 notes apply here with more force; not pursued (held-out).
