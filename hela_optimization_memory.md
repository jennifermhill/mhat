# Fluo-N2DL-HeLa / 01_cells — working memory

## Current best (B5S1, 2026-09-23 evening) — new seg + size_threshold 60
TRA 0.9656 / DET 0.9656 / LNK 0.9654 / SEG 0.8475, AOGM 3417, BIO(1) 0.819 (CT 0.845, TF 0.938, BC(1) 0.72, CCA 0.757). B5S2 (divw +300) BIO 0.824 / AOGM 3445.5; B5S3 (size 80) AOGM 3404.5 / BIO 0.816. Old-seg best B2R3: TRA 0.9537 / SEG 0.7761 / BIO 0.805. Jennifer's start: BIO 0.322, TRA 0.9444, SEG 0.7726.

```toml
seg_result = "hela_cpm15_ft10"          # cellpose cellprob -1.5, flow_threshold 1.0 (GPU run by Jennifer)
flow_result = "2026-09-23_15-37-38"
drift_weight = 50.0
drift_constant = -2000.0
appear_constant = 800.0
disappear_constant = 800.0
division_weight = 0.0
cohesion_weight = -500 ; cohesion_constant = 450   # inert (cost std 39)
adhesion_weight = -100 ; adhesion_constant = 50    # inert (cost std 28)
max_edge_distance = 50 ; max_children = 5 ; size_threshold = 60 ; divisions = true
```
Config: `scratch_configs/tracking/Fluo-N2DL-HeLa/01_cells/hela_B5S1_config.toml`; run/eval under `experiments/{tracking,evaluation}/Fluo-N2DL-HeLa/01_cells/hela_B5S1/`.

## Closed
- Drift break-even: dw 350 → 30–50 with app = dis 800; dc −1000 worse than −2000; app/dis 1200 worse. Plateau AOGM 4581–4597 over dw 30/50 × divw 0/−300/−600.
- division_weight: negative (−300/−600) ≈ 0 on AOGM but worse BIO (more false divisions); positive (+300…+1000) raises CT/CCA but loses true divisions faster than it removes false ones → BIO plateau 0.80 over app 800–1200 × divw 0…+600 at dw 50; dw 100 worse everywhere.
- Segmentation: cellprob is a mask-size dial (0 → masks 11 % small, −3 → 16 % large); optimum −1.5 (ratio 1.0), flow_threshold 1.0 keeps ~430 tiny FP masks that size_threshold 60 removes (4 TP lost). SEG 0.776 → 0.848; −3 and +3 full runs were worse.
- Costs are saturated: only 13 ILP-rejected edges remain; 78/94 divisions fully correct.

## Not cost-addressable
- 282 FN nodes = near-invisible GT cells (raw at background level).
- ~750 FP nodes = border cells + bright interior nuclei the GT does not annotate.

## Open
- BC precision (0.68): 34–48 false divisions with the same drift/area/intensity signature as true ones; daughter area ratio would separate them (0.91 vs 0.60 median) — needs a new hyperedge attribute + cost in src/mhat (mhat_dicty).
- traccuracy 0.4.3: DivisionMetrics rejects the CTC (many-to-one) matching; CCA breaks on the `pos` location key — both would fail through evaluate_tracks.py `division` / `cca`.
- Hyperedge drift measure for asymmetric divisions (core change, mhat_dicty).
- 02_cells held-out run with the B5S1 recipe (needs a cellprob −1.5 / flow 1.0 seg of 02_cells first).
