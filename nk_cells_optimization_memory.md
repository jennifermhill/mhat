# primary_nk_cells — Optimization Working Memory

Read + update this each batch. Companion to the append-only `nk_cells_optimization_log.md`.

## ⚠ Ground truth was corrected 2026-07-30
`experiments/tracking/primary_nk_cells/01_cells/correct_tracks.zarr` is now **1070 nodes /
1029 edges** (was 1011 / 976; old version kept as `correct_tracks_old.zarr`). It is stored as
a nested `tracks/` geff group — `evaluate_tracking.geff_group_path()` handles either layout.
All 22 runs were re-scored against it (Phase 5). **Any TE/TF number below 0.5 in this file or
in Phase 0–4 of the log is an old-GT number and is not comparable to a current one.**

## Objective
Optimize `primary_nk_cells / 01_cells` tracking. GT is still **sparse** (1070 annotated nodes
vs ~21k predicted objects), so precision / purity / F1 stay untrustworthy.
**Primary: TE and TF. Co-primary: Edge Recall** (promoted 2026-07-30 — it is the bottleneck,
0.867 against Node Recall 0.966 at the best run). Node Recall and FN edges are diagnostics.

## Current best (as of Phase 6 Stage 2, 2026-07-30)
- **`nkgraph_size120`** — TE **0.6744**, TF 0.6750, NodeR 0.9654, EdgeR 0.8707, FN_edg 133.
  = base params with `size_threshold = 120`, `max_children = 5`. Best TE in the campaign.
  `configs/evaluation/ultrack_vs_mhat_nk_cells.toml` now points here (2026-07-31).
- (2nd) `2026-07-27_14-10-54` = `size80`+`max_children=5` — TE 0.6706, EdgeR 0.8669.
- Best **Edge Recall** in the campaign is `nkgrid_s120m16` (0.8776, FN_edg 126) at TE 0.6638 —
  the TE-vs-recall trade-off. TE was taken as the deciding metric.
- Ultrack baseline: TE 0.6064, EdgeR 0.8183. MHAT is ahead on both.

## Base config (all Phase 6+ sweeps)
seg `2026-06-25_17-40-15`, flow `2026-06-25_17-53-34`, `use_lk=false`,
`drift_distance=[0,3,6]`, drift 100/−1500, area 1500/−1500, intensity 25/−1000,
cohesion **0/0**, adhesion −1500/1500, appear/disappear 50, curvature **0/0**,
`merges=false`, `divisions=false`, `min/max_merge_cost` 0/1,
`size_threshold=80`, `max_edge_distance=40`, `max_children=5`.

**Do not switch segmentation.** The newer seg `2026-07-27_11-38-06` + flow `2026-07-16_15-56-39`
scores 0.6132 vs 0.6608 for the otherwise-identical `size80` — a 0.048 TE loss (NodeR 0.873 vs
0.959). Held fixed by user decision.

## Cost contribution, corrected GT (Phase 5)
Ablation from `full` (0.6521): **drift (−0.1069) ≈ area (−0.1050) > adhesion (−0.0700) >
appear/disappear (−0.0214) > intensity (−0.0136) ≫ cohesion (+0.0010, inert)**.
Addition onto `drift_only` (0.4626): area (+0.1069) > intensity (+0.0447) > cohesion (0.0000)
> **adhesion (−0.0486, hurts alone)**.

### Changed vs the old GT — do not trust the Phase 2 conclusions
- **adhesion is NOT the most helpful cost.** It fell from 1st to 3rd. drift and area now lead
  and are effectively tied.
- **`add_adhesion` flipped sign**: +0.0338 → −0.0486. It is now the worst config in the study
  (TE 0.4140, NodeR 0.745). Adhesion only pays off in combination with the other costs.
- **appear/disappear now outranks intensity** (−0.0214 vs −0.0136); it was the reverse.
- **`size_threshold` optimum moved UP, not down**: 160 > 80 > 40 ≈ 20. The extra 59 GT nodes
  made *more* aggressive small-fragment filtering better. 160 is at the top of the tested
  range → **not bracketed above**, extend to 200/240/320.

### Unchanged by the correction
- **cohesion is inert.** Ablation +0.0010, addition exactly 0.0000, `add_cohesion` byte-identical
  to `drift_only`. Attribute std is 0.031 — a graph property, GT-independent. Keep at 0/0.
- **`max_edge_distance` saturated at 40** — 50/60/80 still byte-identical. Never widen.
  Narrowing (20/25/30) is still untested.
- **`drift_only` is weak**; the multi-cost combination is essential. `full` is near-optimal
  among cost *sets* — only the inert `no_cohesion` beats it.

## Stale numbers — re-measure before using
The attribute-stats table recorded 2026-07-09 was taken on `full` (`size_threshold=20`,
`max_children=3`) and does **not** describe the current base. It also disagrees with the
corrected GT: it lists `drift_dist` mean 10.750 / std 7.873, while the corrected-GT diagnostics
report mean raw GT displacement **6.47** (std 6.35, median 4.46). **Re-run `--stats-only` on the
current base before setting any drift range.**

Two calibration facts worth keeping from that table:
- Rule: a cost influences the ILP only when its cost std is ~400–2000; `target_weight ≈
  target_cost_std / attribute_std`. Keep edge-cost means negative so links get selected.
- **Edge costs follow `|w| × attr_std` exactly; node costs do NOT.** adhesion's tabulated cost
  std (1838) is 2.5× `1500 × 0.485` because node selection is leaves-scaled. Sweep adhesion
  multiplicatively around its current value, not from the std rule.

## Tooling (built 2026-07-30)
- `scripts/05_evaluation/launch_evals.py` — eval-only LSF launcher for already-tracked runs
  (spec TOML with `reeval_id` / `uids` or `all_runs`). Writes a `launch_sweep`-compatible
  manifest. **Overwrites `track_metrics.json` in place** — old numbers survive only in the log.
- `scripts/05_evaluation/collect_sweep.py --metric-set overlap` now reports
  **TE, TF, EdgeR, NodeR, FN_e, Purity** (columns carry their own JSON group). CTC output
  unchanged.
- Sweeping `size_threshold` / `max_edge_distance` / `max_children` needs
  `launch_sweep.py --allow-protected`.

## Plan — Phase 6 complete (2026-07-31)
- ~~**Stage 0** — graph construction (`nkgraph`)~~ **done**. `size_threshold` and
  `max_edge_distance` bracketed; `max_edge_distance = 40` closed (saturated).
- ~~**Stage 0b** — `max_children` at the Stage-1 base (`nkmc`)~~ **done**.
- ~~**Stage 1** — cost weights and constants (`nkp3A`)~~ **done, and exhausted**: the base is
  the interior optimum on every cost axis, so there is nothing left to extend. The three edge
  constants behave as one axis.
- ~~**Stage 2** — `size_threshold` x `max_children` grid at base costs (`nkgrid`)~~ **done**.
  Winner (120, 5). `size_threshold = 0` falsified under matched costs.
- ~~Update `configs/evaluation/ultrack_vs_mhat_nk_cells.toml` to the new best~~ **done
  2026-07-31** (`tracking_uid = "nkgraph_size120"`).
- **Remaining, on Windows only**: regenerate `ultrack_vs_mhat_nk_cells.png` with
  `scripts/07_plotting/trackmate_vs_mhat_figure.py`. The config's `output_png` /
  `eval_base_dir` are `Y:` paths — do not render on the cluster.

### Where the campaign landed
Total gain over the Phase 5 starting point is **+0.0038 TE / +0.0038 EdgeR**; the whole Stage-2
grid spans 0.021 TE. The pre-existing operating point was already near-optimal, and the main
value of Phase 6 is the negative results (costs locally optimal, `max_edge_distance` closed,
keeping all candidates does not pay). Further coordinate-wise tuning is not worth running —
the remaining levers are segmentation and GT quality, not ILP parameters.
