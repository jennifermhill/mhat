# Fluo-N3DL-DRO Tracking Optimization Memory

## Optimization Targets

GT is **sparse** (189 annotated nuclei/frame vs ~4,600 predicted) **and has zero
divisions**, while the data itself does contain divisions.

- **Primary**: TE (target_effectiveness), TF (track_fractions)
- **Secondary**: CTC `SEG` (sparsity-independent, from 20 `man_seg` slices)
- **Diagnostics**: `fn_nodes`, `fn_edges`, `LNK`, division count
- **Do NOT optimize**: `TRA`, `DET`, `track_purity` — all artifacts of the 222k
  correct-but-unannotated objects; AOGM clamps at `min(aogm, 10*n_gt)`.
- **Division target ≈30 total.** Not zero: real divisions must survive for future GT.
  Baseline (`division_weight = 0.0`) produced ~3,955.

## Current Best

**`dro1_g_mc8_nodiv` (batch 1, 2026-08-28) — TE=0.7211, TF=0.7211, fn_nodes=179,
fn_edges=982, SEG=0.7178, LNK=0.8936, 0 divisions** (+0.0860 TE over baseline).

**STILL the best after batches 2-5 (34 runs). ILP cost tuning is EXHAUSTED** — every
cost axis is now flat or worse, and the best result of 34 runs was +0.0004. Do not
open another cost-tuning batch on this dataset; see Falsified Hypotheses.

**Caveat: not a shippable config.** It scores that well by setting `divisions = false`,
which the plan forbids as an endpoint — the data contains real divisions that must
survive for future GT, hence the ~30-division target. Treat it as the *division-free
reference*: it bounds what division tuning is playing for.

**Batch 3 has now priced that caveat.** TE is monotonic in division count all the way
to zero, so the ~30-division requirement is not free: it costs **0.017-0.031 TE**
against the mc=3 division-free reference (bracketed by `dw1000` -> 561 divisions ->
0.6903 and `dw4000` -> 0 divisions -> 0.7044). There is no `division_weight` that
keeps divisions *and* recovers the divisions prize.

Prior: baseline `full_50tp` (2026-08-27), TE=0.6351, reproduced exactly by
`dro1_g_mc3_div`.

```toml
drift_weight = 350.0      # break-even at 5.71 um
drift_constant = -2000.0
area_weight = 0.0 ; intensity_weight = 0.0 ; curvature_weight = 0.0
cohesion_weight = -500 ; cohesion_constant = 450
adhesion_weight = -100 ; adhesion_constant = 50
division_weight = 0.0
appear_constant = 50.0 ; disappear_constant = 50.0
max_children = 3 ; max_edge_distance = 15 ; size_threshold = 20
merges = false ; divisions = true
```

## Results Comparison Table

| Run | Change | TE | TF | fn_nodes | fn_edges | SEG | LNK | divisions | Notes |
|-----|--------|-----|-----|----------|----------|-----|-----|-----------|-------|
| Baseline | — | 0.6351 | 0.6351 | 156 | 912 | 0.7283 | 0.8797 | 3955 | `full_50tp` |
| B1R1 | mc=3, div=true | 0.6351 | 0.6351 | 156 | 912 | 0.7283 | 0.8797 | 3955 | exact baseline reproduction |
| B1R2 | mc=3, div=**false** | 0.7044 | 0.7044 | 190 | 1021 | 0.7178 | 0.8893 | 0 | +0.069 TE from divisions alone |
| B1R3 | mc=5, div=false | 0.7164 | 0.7164 | 183 | 995 | 0.7178 | 0.8922 | 0 | +0.012 vs mc=3 |
| **B1R4** | **mc=8, div=false** | **0.7211** | **0.7211** | 179 | 982 | 0.7178 | 0.8936 | 0 | **best TE; saturating** |
| B1R5 | mc=5, div=true | 0.6358 | 0.6358 | **131** | **849** | 0.7283 | 0.8800 | 5316 | best recall, 2nd-worst TE |
| B2 `ref` | *(none)*, off cache | 0.7211 | 0.7211 | 179 | 982 | 0.7178 | 0.8936 | 0 | = B1R4 exactly (cache val. #3) |
| B2 `ad500` | appear/disap 500 | 0.7226 | 0.7226 | 204 | 983 | 0.7133 | 0.8931 | 0 | +0.0015, real but negligible |
| B2 `dc3500` | drift_const -3500 | 0.6945 | 0.6945 | **137** | 1006 | 0.7218 | 0.8897 | 0 | best fn_n, worst TE of B2 |
| B3 `dw1000` | div_weight 1000 | 0.6903 | 0.6903 | 185 | 1003 | 0.7178 | 0.8877 | **561** | last point with divisions |
| B3 `dw4000` | div_weight 4000 | 0.7044 | 0.7044 | 190 | 1021 | 0.7178 | 0.8893 | **0** | = B1R2 to 16 digits |
| B4 `dc1500` | drift_const -1500 | 0.7091 | 0.7091 | 227 | 1044 | 0.7178 | 0.8868 | 0 | less-negative also worse |
| B4 `dc500` | drift_const -500 | 0.5140 | 0.5140 | 1141 | 2100 | 0.6661 | 0.7732 | 0 | collapse; tracks fail to form |
| B5 `cc470` | cohesion_const 470 | 0.7215 | 0.7215 | 182 | 984 | 0.7178 | 0.8934 | 0 | best of 34 runs: **+0.0004** |
| B5 `aw300` | adh_w -300/const 68 | 0.7215 | 0.7215 | 182 | 984 | 0.7178 | 0.8934 | 0 | **bit-identical to cc470** |
| B5 `dw1000` | drift_weight 1000 | 0.7196 | 0.7196 | 176 | 984 | 0.7178 | 0.8934 | 0 | more edge discrimination hurts |

## Established Principles

- **TE is fragmentation-limited, not recall-limited.** The run with the best node *and*
  edge recall (B1R5: fn_nodes 131, fn_edges 849) scores 0.085 TE *below* the run with the
  worst (B1R4: 179, 982). `target_effectiveness` credits each GT track to its single
  best-matching predicted track, so one spurious division truncates the match however
  complete the linking is.
- **Therefore `fn_edges` is NOT a safe proxy for TE across division regimes.** It points
  the wrong way on the `divisions` axis. It remains valid *within* a fixed regime (it
  tracks TE correctly across the three nodiv runs). Do not steer division decisions by it.
- **Normalizing does not rescue it — the error *rate* runs backwards too.** Against
  **9,261 GT edges** (189 tracks x 49; from `diagnostics_summary.json ->
  fn_edges.n_gt_edges`), every config sits in a 9.2-11.0% band, i.e. ~1 missed link per
  9-11 GT edges, and the ordering is close to inverted from TE: B1R5 has the campaign's
  *lowest* rate (849 = 9.17%) and its second-worst TE, while the best-TE run
  (B5 `cc470`, 984 = 10.63%) is ~1.5 points worse per edge than the baseline it beats by
  +0.086 TE. Incumbent B1R4 = 982 = 10.60%. Per-edge error is a **recall** measure; TE
  is fragmentation-limited. Report it as a diagnostic, never steer on it.
- **Divisions are the dominant lever: +0.069 TE from disabling them** (B1R1 -> B1R2,
  a clean one-variable isolation). 3,955 divisions against a GT containing zero.
- **`max_children` is a weak, saturating lever — plan premise #1 falsified in its strong
  form.** 3->5 gives +0.0120, 5->8 gives +0.0047; 3->8 recovers only 39 of the 332
  never-candidate edges (12%) for 2.66x the candidate edges. Since mc=8 would have caught
  a true successor sitting at rank 4-8 by raw distance, those edges are not
  ranked-but-excluded: the raw-distance KNN never surfaces them at any practical `k`.
- **Raising `max_children` while divisions are on is counterproductive.** B1R1 -> B1R5
  recovers 63 FN edges but yields +0.0007 TE and 1,361 *more* divisions. Extra candidates
  get spent on branching, not linking.
- **Simple candidate edges scale exactly linearly in `max_children`**; with
  `divisions = false` (and `merges = false`) no hyperedges are added at all, so every
  nodiv graph is smaller than the mc=3 baseline. Division hyperedges, not `max_children`,
  drive model size (12.8M of `g_mc5_div`'s 14.94M edges).
- **SEG is invariant to `max_children`** and depends only on the division regime
  (0.7178 nodiv, 0.7283 div) — a useful check that a run changed only edge availability.
- **Only 2.21% of GT edges were ever reachable by cost tuning.** Baseline miss split
  (partitions all 9,261 GT edges): **332 never candidates (3.58%)** — absent from the
  graph; **270 endpoint node unmatched (2.92%)** — upstream segmentation; **205 ILP
  rejected (2.21%)** — the only slice ILP costs can touch. So ~6.5 of a ~10 point error
  rate is decided before the solver runs. *This is the quantitative reason batches 2-5
  were flat*, and the strongest argument for going upstream.
  Caveat: CTC `fn_edges` (912) and the diagnostic partition (537+270=807) differ by 105
  edges for the baseline — CTC also carries `ws_edges` 290 / `ns_nodes` 54, so
  many-to-one matching is penalized differently. Cause **not traced**. Use CTC
  `fn_edges` for rates; the structural split exists only for `full_50tp`, since sweeps
  set `[eval.diagnostics] enabled = false`.
- **The graph cache reproduces solutions exactly** (validated at 3 frames and again at
  50 by B1R1, and by the B2 `ref` and the three B5 anchors — five validations) and cuts
  a ~9.5 h build to a cache load. Caches: 0.44-1.03 GB each.
- **The solver is deterministic here; there is no run-to-run noise floor.** Anchor runs
  reproduce bit-identically across sweeps. So small deltas are *real*, not noise — they
  are simply negligible. Scale: TE averages over 189 GT tracks of 50 frames, so
  **+0.0004 TE is about 4 frame-assignments across the whole dataset.** Prefer "real but
  too small to act on" over "noise" (batch 2's `ad500` verdict used the looser word).
- **The solution sits on a wide plateau in cost space.** B5 `cc470` and `aw300` produced
  the *bit-identical* integer solution from genuinely different configs (verified by
  diffing generated track_configs). Node costs dominate edge costs 15-20x, yet swinging
  net per-leaf pressure through its sign flip moves TE by +0.0008.
- **`cohesion` is a per-leaf size prior, not a discriminator.** Its attribute std/mean
  is 0.009, far under CLAUDE.md's 0.3 line; its huge cost std (15,350) comes entirely
  from `num_leaves` scaling. `adhesion` (std/mean 2.91) carries the per-node signal —
  and B5 showed even that is not a usable lever.
- **`drift_constant = -2000` is a bracketed interior optimum**, closed on both sides by
  batches 2 and 4. `collect_sweep`'s edge-warning is per-sweep and cannot see across
  sweeps, so it emitted a spurious "extend" flag on *each* one-sided batch. Check the
  neighbouring sweep before acting on an edge flag.

## Open Questions / Hypotheses to Test

**All cost-tuning questions are closed (batches 2-5). Everything below is upstream.**

1. **Build the mc=8 `divisions = true` graph and sweep `division_weight` there.**
   *This is the top remaining experiment if the ~30-division requirement stands.*
   Batch 3 ran at mc=3, whose division-free ceiling (0.7044) is structurally below the
   current best (0.7211 at mc=8), so it could never have produced a shippable config.
   The mc=3 -> mc=8 gap is +0.0167 in the nodiv regime; if it carries into the division
   regime, mc=8 with `division_weight` in the 1000-4000 gap should land near
   0.70-0.72 *with* divisions surviving. Cost: ~37M edges, a graph build well over the
   ~9.5 h seen at mc=3 (5.13M) — new cache key, so no reuse. Narrow the weight by
   interpolation: 561 divisions at 1000, 0 at 4000, target ~30.
2. **Segmentation / merge hypotheses are the only untested structural lever.** Batch 1
   located 332 of 537 recoverable FN edges as *never candidates*, and the diagnostics
   attribute them to a merge-hypothesis problem (the GT node matches a fragment at one
   end and a differently-merged object at the other; `area_diff` 0.613 vs 0.048 for TP,
   and flow barely helps them). Levers: `min_merge_cost`, `max_merge_cost`,
   `size_threshold`, or regenerating the segmentation
   (`seg_result = 2026-08-21_12-03-48`). All change the graph cache key. CLAUDE.md
   lists these as do-not-change, but that rule sits in the MDA231 section; batch 1
   already departed from it for `max_children` on this dataset for the same reason.
3. **Flow is not disambiguating the hard cases.** Substitute *incoming* edges have
   lower flow-corrected drift (2.49) than the true edge (5.05). Untested, and now
   relatively more attractive since every cost axis is closed.
4. ~~`max_children` extension to 12~~ — still open in principle but very low value:
   +0.0047 on the 5->8 step, and B5 shows the solve is on a plateau. Not worth a batch.

## Falsified Hypotheses

- **"`max_children` is the single biggest lever" (plan root cause #1)** — it is real but
  small and saturating (+0.017 TE over 3->8), and worth ~1/4 of the divisions axis.
  The plan's contingency applies: lean harder on cost tuning.
- **"Recovering never-candidate FN edges is the path to TE"** — B1R5 recovered the most
  FN edges of any run and still lost 0.085 TE to B1R4. Recovery without fragmentation
  control does not convert into TE.
- **"Drift break-even sits below the FN population" (plan cost-priority #1)** — dead in
  *both* directions. More negative (B2: -2500/-3000/-3500) monotonically worse; less
  negative (B4: -1500/-1000/-500) monotonically worse and collapsing. -2000 is optimal.
- **"Track termination is nearly free, so raising appear/disappear helps" (#2)** — B2
  best was +0.0015; 2000 costs -0.015. Effectively exhausted.
- **"`division_weight` tuning to ~30 divisions recovers most of the +0.069" (#2, was top
  priority)** — B3: TE is monotonic in division count *to zero*. No interior optimum
  exists, so keeping ~30 divisions costs 0.017-0.031 TE rather than recovering the
  prize. `division_weight >= 4000` gives 0 divisions and exactly reproduces `divisions =
  false` at the same `max_children`.
- **"Rebalancing node vs edge costs is the untouched dominant lever" (B5, my own)** —
  correct that node costs dominate 15-20x, wrong that it is a lever. Swinging net
  per-leaf pressure through its sign flip: +0.0008. Raising adhesion discrimination at
  fixed mean: +0.0004. Raising drift discrimination at fixed mean: *worse*. Total
  spread over 12 runs and 3 axes: 0.0019.
