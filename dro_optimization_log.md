# Fluo-N3DL-DRO Tracking Optimization Log

## Setup

**Dataset**: Fluo-N3DL-DRO / 01_nuclei. Seg `2026-08-21_12-03-48`, flow `2026-08-21_12-04-20`
(2D Farneback + 3D Lucas-Kanade), 50 timepoints, shape `(50, 125, 603, 1272)`.

**Eval**: `matcher = "ctc"`, `metrics = ["ctc", "track_overlap"]`, no threshold kwarg.
Diagnostics are **disabled for sweep runs** (the baseline eval spent >45 min in the
`fn_nodes` diagnostic) and re-enabled only on a batch winner.

**Metric semantics — read before interpreting any row.** GT is **sparse**: 189
annotated nuclei per frame against ~4,600 predicted objects, all 189 GT tracks
spanning frames 0-49 unbroken, with **zero divisions**.

| metric | use it? |
|---|---|
| `target_effectiveness` (TE) | **YES — primary target** |
| `track_fractions` (TF) | **YES — primary target** |
| CTC `SEG` | **YES** — from 20 pixel-accurate `man_seg` slices, sparsity-independent |
| `fn_nodes`, `fn_edges` | YES — diagnostics |
| `LNK` | partially — edge-only, less sparsity-sensitive |
| `TRA`, `DET`, `track_purity` | **NO — artifacts.** 222,512 "FP" nodes are correct but unannotated; AOGM clamps |

**Pipeline**: cluster-only, `mhat-cluster` env, `module load gurobi` in every tracking job.
Submitted with `scripts/04_tracking/launch_sweep.py`, collected with
`scripts/05_evaluation/collect_sweep.py --metric-set dro`. Division counts come from
`scripts/05_evaluation/count_divisions.py` (GT has none, so no metric reports them).

**Baseline (`full_50tp`, 2026-08-27)**: TE=0.6351, TF=0.6351, fn_nodes=156/9450,
fn_edges=912/9261, LNK=0.8797, SEG=0.7283, ~3,955 divisions produced.

---

## Phase 0 — tooling (2026-08-27)

Not an optimization batch. Four changes, so that cost-only runs stop paying the
~11.9 h candidate-graph build that dominates a 12 h 41 m run:

- `scripts/04_tracking/run_tracking.py` — candidate-graph pickle cache behind a
  `graph_cache_dir` config key, plus `--save-graph` / `--no-graph-cache`. Keyed on
  the graph-affecting config subset only; every weight/constant is excluded so one
  cache serves every cost sweep. Key and path are logged on every run, hit or miss.
- `scripts/05_evaluation/count_divisions.py` — new; division events, nodes, edges,
  tracks, starts, ends for a tracking output dir.
- `scripts/05_evaluation/collect_sweep.py` — new `dro` metric set; `division_weight`
  added to `TUNABLE_KEYS`.
- `scripts/04_tracking/launch_sweep.py` — `graph_cache_dir` / `division_weight` added
  to `OPTIONAL_KEYS`, and `relative_to(REPO)` replaced by a fallback: sweep artifacts
  live under the *data* tree (`.../jennifer/mhat`), which is a sibling of the worktree
  (`.../jennifer/mhat-cluster`), so the old code raised `ValueError` before submitting
  anything. Not in the plan; required to run it at all.

**Validation gate** (§0.2 of the plan): a 3-timepoint run uncached vs cache-loaded,
compared on node ids, edge list and node positions. Result recorded below.

---

### Phase 0 gate result — PASSED (2026-08-27)

3 timepoints, `dro_cachegate_{uncached,build,hit}.toml`, jobs 153758061-64.

| run | mode | wall time |
|---|---|---|
| `cachegate_uncached` | `--no-graph-cache` | 3129 s |
| `cachegate_build` | cache MISS -> build + save | 3134 s |
| `cachegate_hit` | cache HIT -> load | **127 s** |

`verify_graph_cache.py` compared uncached vs cache-loaded: **13,735 nodes and 9,155
edges in both, identical node ids, identical edge list, identical positions.**

**~25x speedup** on the solve-only path, and the 3-timepoint cache is only 0.03 GB
(213,309 edges after hyperedges), so a 50-timepoint mc=3 cache should land near 0.8 GB
rather than the 3-8 GB the plan budgeted.

`count_divisions.py` was cross-checked against the baseline and reproduces the handoff
numbers exactly: **3,955 divisions, 231,752 nodes, 222,949 edges, 8,803 starts,
12,758 ends** (job 153758068).

---

## Batch 1 (2026-08-27) — graph structure

Sweep `dro1`, spec `scratch_configs/sweeps/dro_01nuclei_batch1.toml`, submitted with
`--allow-protected` (`max_children` and `divisions` are both on `PROTECTED_KEYS`).
Each run builds and caches its own graph, then solves at baseline costs, so every run
is both a data point and a reusable cache for the Phase 2 cost sweeps.

| run | max_children | divisions | track job | eval job |
|---|---|---|---|---|
| `g_mc3_div` | 3 | true | 153758191 | 153758192 |
| `g_mc3_nodiv` | 3 | false | 153758193 | 153758194 |
| `g_mc5_nodiv` | 5 | false | 153758195 | 153758196 |
| `g_mc8_nodiv` | 8 | false | 153758197 | 153758198 |
| `g_mc5_div` | 5 | true | 153758199 | 153758200 |

Hypothesis: "`fn_edges` falls sharply from max_children 3 -> 5 -> 8, because 332 of the
537 both-endpoints-matched FN edges were never candidates and `add_cand_edges` ranks
by raw distance (FN median 6.18 um vs TP 1.00 um)."

`g_mc3_div` doubles as a reproduction check: same parameters as `full_50tp`, so it
should return TE=0.6351.

#### Graph sizes (all five built, 2026-08-28)

| run | mc | div | simple edges | after hyperedges | cache |
|---|---|---|---|---|---|
| `g_mc3_nodiv` | 3 | false | 1,283,064 | 1,283,064 | 0.44 GB |
| `g_mc5_nodiv` | 5 | false | 2,136,127 | 2,136,127 | 0.54 GB |
| `g_mc8_nodiv` | 8 | false | 3,411,620 | 3,411,620 | — |
| `g_mc3_div` | 3 | true | 1,283,064 | **5,129,229** | 0.59 GB |
| `g_mc5_div` | 5 | true | 2,136,127 | **14,938,858** | 1.03 GB |

All five built the identical node set (437,264 nodes, 23,984 = 5.5% z-flow-unreliable),
confirming `max_children` / `divisions` touch only edge construction.

**Simple edges scale exactly linearly in `max_children`** (1,283,064 x 5/3 = 2,138,440
vs 2,136,127 actual; x 8/3 = 3,421,504 vs 3,411,620 actual — the ~0.3% shortfall is
boundary nodes with fewer than `max_children` neighbours inside `max_edge_distance`).

**With `divisions = false` no hyperedges are added at all** (`merges = false` too), so
every nodiv run is *smaller* than the mc=3 baseline: even `g_mc8_nodiv` at 3.41M sits
below the baseline's 5.13M. The plan's concern about mc=8 exhausting memory does not
materialise once divisions are off — which is exactly why it prescribed probing
`max_children` in the division-free regime.

`g_mc5_div` is the only genuinely heavy model at 14.94M edges (2.9x baseline); the
12.8M-edge gap against `g_mc5_nodiv` is entirely division hyperedges, making each
div/nodiv pair a controlled comparison over identical linking candidates.

Peak memory across all five stayed at 11.5 GB against the 240 GB that 16 slots give,
so memory was never the binding constraint. Gurobi (not SCIP) confirmed solving.

#### Results

### B1R2: divisions = false at max_children = 3
exp_uid: dro1_g_mc3_nodiv (track job 153758193, 9h13m, exit DONE)
Hypothesis: "Divisions are free (`division_weight = 0`) and unwanted (GT has none), so
disabling them removes the 3,955 spurious splits and raises TE."

| | baseline `full_50tp` | `g_mc3_nodiv` | delta |
|---|---|---|---|
| **TE** | 0.6351 | **0.7044** | **+0.0692** |
| TF | 0.6351 | 0.7044 | +0.0692 |
| SEG | 0.7283 | 0.7178 | -0.0105 |
| LNK | 0.8797 | 0.8893 | +0.0096 |
| fn_nodes | 156 | 190 | +34 |
| fn_edges | 912 | 1021 | +109 |
| ws_edges | 290 | **0** | -290 |

Verdict: **supported, strongly** — this run differs from the baseline in `divisions`
alone, so the isolation is clean. +0.0692 TE is larger than anything the plan projected
from cost tuning. Mechanism is visible in `ws_edges` 290 -> 0: `target_effectiveness`
scores each GT track by its single best-matching predicted track, so every spurious
division truncates that match. 3,955 divisions against a GT containing zero were
cutting predicted tracks apart wholesale.

**This does NOT mean ship `divisions = false`.** The data does contain divisions even
though this GT does not, and the plan fixes the target at ~30 so real divisions survive
for future GT. What this run establishes is the **prize for division tuning**: ~+0.069 TE
is what `division_weight` is playing for, and this is the division-free reference point
to measure the eventual tuned value against.

Cost of the change is modest and worth watching: +34 FN nodes, +109 FN edges, -0.0105 SEG.


### B1R1: max_children = 3, divisions = true (baseline reproduction)
exp_uid: dro1_g_mc3_div (track 153758191, 9h26m)
Hypothesis: "Same parameters as `full_50tp`, so the refactored builder + cache must
return TE = 0.6351 exactly."
TE: 0.6351, TF: 0.6351, fn_nodes: 156, fn_edges: 912, ws: 290, SEG: 0.7283, LNK: 0.8797,
divisions: 3955, nodes: 231752, edges: 222949, starts: 8803, ends: 12758
Verdict: **supported, exactly** — every metric and every count is bit-identical to
`full_50tp`. The Phase 0 refactor is validated at full 50-frame scale, not just on the
3-frame gate. All other batch-1 comparisons rest on this.

### B1R3: max_children = 5, divisions = false
exp_uid: dro1_g_mc5_nodiv (track 153758195, 9h25m)
TE: 0.7164, fn_nodes: 183, fn_edges: 995, SEG: 0.7178, LNK: 0.8922, divisions: 0
Verdict: supported but weak — +0.0120 TE and only 26 FN edges recovered vs mc=3, for
853k extra candidate edges.

### B1R4: max_children = 8, divisions = false
exp_uid: dro1_g_mc8_nodiv (track 153758197, 9h36m)
TE: 0.7211, fn_nodes: 179, fn_edges: 982, SEG: 0.7178, LNK: 0.8936, divisions: 0
Verdict: **premise #1 falsified in its strong form** — 3->8 recovers only 39 of the 332
never-candidate edges (12%) for 2.66x the candidate edges, and each step returns less
than half the previous one (+0.0120 then +0.0047). `max_children` is not "the single
biggest lever"; divisions are worth 4x more. Since mc=8 would have caught the true
successor if it were merely rank 4-8 by raw distance, those edges are not
ranked-but-excluded -- the raw-distance KNN does not surface them at any practical `k`.

### B1R5: max_children = 5, divisions = true
exp_uid: dro1_g_mc5_div (track 153758199, 10h31m; 14.94M edges, the batch's heavy model)
TE: 0.6358, fn_nodes: 131, fn_edges: 849, SEG: 0.7283, LNK: 0.8800, divisions: 5316
Verdict: **falsified, and the most informative run of the batch** — mc5_div has the
**best recall of all five runs** (fn_nodes 131 and fn_edges 849, better than any nodiv
run) yet the **second-worst TE**. Against mc3_div it recovers 63 FN edges for +0.0007 TE,
while producing 1,361 *more* divisions (3,955 -> 5,316). Extra candidates under
`divisions = true` are spent on more spurious branching, not better linking.

#### Batch 1 summary

| run | mc | div | TE | fn_n | fn_e | SEG | LNK | divisions | tracks |
|---|---|---|---|---|---|---|---|---|---|
| `g_mc8_nodiv` | 8 | false | **0.7211** | 179 | 982 | 0.7178 | 0.8936 | 0 | 10302 |
| `g_mc5_nodiv` | 5 | false | 0.7164 | 183 | 995 | 0.7178 | 0.8922 | 0 | 10547 |
| `g_mc3_nodiv` | 3 | false | 0.7044 | 190 | 1021 | 0.7178 | 0.8893 | 0 | 11281 |
| `g_mc5_div` | 5 | true | 0.6358 | **131** | **849** | 0.7283 | 0.8800 | 5316 | 7231 |
| `g_mc3_div` | 3 | true | 0.6351 | 156 | 912 | 0.7283 | 0.8797 | 3955 | 8803 |

**The headline finding is a dissociation: TE on this dataset is fragmentation-limited,
not recall-limited.** The run with the best node *and* edge recall (`g_mc5_div`) scores
0.085 TE below the run with the worst (`g_mc8_nodiv`). `target_effectiveness` credits
each GT track to its single best-matching predicted track, so a spurious division
truncates that match no matter how complete the underlying linking is.

**Consequence for method: `fn_edges` is a poor proxy for TE here** and must not be used
to steer division-related decisions. The plan lists it as the secondary signal; on the
`divisions` axis it points the wrong way. It stays valid *within* a fixed division
regime (it tracks TE correctly across the three nodiv runs).

Ordering of effects measured so far: **divisions (+0.069) >> max_children 3->8 (+0.017)
> max_children under divisions (+0.0007, i.e. nil)**.

SEG is invariant within each division regime (0.7178 nodiv, 0.7283 div) and independent
of `max_children`, confirming these runs changed only which edges were available, never
which nodes were selected.

---

## Batch 2 (2026-08-28) — cost sweeps on the cached mc=8 graph

Sweep `dro2`, spec `scratch_configs/sweeps/dro_01nuclei_batch2.toml`. `max_children = 8` and
`divisions = false` sit in `[base]`, so all 7 runs hit the one cached graph
(`5bcbe28db3b19a19.pkl`) and no run varies a protected key. Tracking walltime dropped
24:00 -> 4:00 since the ~9.5 h build is skipped.

| run | override | drift break-even |
|---|---|---|
| `ref` | *(none)* — must reproduce TE 0.7211 off the cache | 5.71 um |
| `dc2500` | `drift_constant = -2500` | 7.14 um |
| `dc3000` | `drift_constant = -3000` | 8.57 um |
| `dc3500` | `drift_constant = -3500` | 10.00 um |
| `ad500` | `appear/disappear = 500` | — |
| `ad1000` | `appear/disappear = 1000` | — |
| `ad2000` | `appear/disappear = 2000` | — |

Track jobs 153762723/25/27/29/31/33/35, evals 153762724/26/28/30/32/34/36.

Hypothesis (drift): "ILP-rejected FN edges have median drift 6.00 um but cost turns
positive at 5.71 um, so pushing break-even past the FN population recovers them."
Hypothesis (appear/disappear): "At 50 against edge costs near -1000 ending a track is
nearly free; 145 of 205 ILP-rejected FN edges simply end the predecessor's track."

Note this batch tunes costs in the **division-free** regime. That is deliberate: the
eventual shipped config targets ~30 divisions, which behaves much closer to nodiv than
to the 3,955-division baseline, so nodiv is the better proxy to tune against. Division
tuning gets its own batch on a `divisions = true` cache.

#### Results

| run | override | TE | fn_n | fn_e | SEG | LNK |
|---|---|---|---|---|---|---|
| `ad500` | appear/disappear 500 | **0.7226** | 204 | 983 | 0.7133 | 0.8931 |
| `ad1000` | appear/disappear 1000 | 0.7217 | 239 | 1017 | 0.7173 | 0.8889 |
| `ref` | *(none)* | 0.7211 | 179 | 982 | 0.7178 | 0.8936 |
| `dc2500` | drift_constant -2500 | 0.7131 | 154 | 987 | 0.7178 | 0.8924 |
| `ad2000` | appear/disappear 2000 | 0.7065 | 296 | 1100 | 0.7173 | 0.8792 |
| `dc3000` | drift_constant -3000 | 0.7015 | 140 | 989 | 0.7218 | 0.8921 |
| `dc3500` | drift_constant -3500 | 0.6945 | 137 | 1006 | 0.7218 | 0.8897 |

**`ref` reproduced `g_mc8_nodiv` exactly** (0.7211 / 179 / 982 / 0.7178 / 0.8936) — a
third confirmation of the cache, and the first proving a cache *hit* equals the original
*build* at 50 frames.

Verdict (drift): **falsified, monotonically.** Every more-negative `drift_constant` lost
TE: -2000 -> -3500 costs 0.027. It fails informatively — `fn_nodes` *improves*
monotonically (179 -> 137) as TE drops, i.e. cheaper edges recover more nodes and then
wire them into tracks that match GT worse. Same recall-vs-fragmentation dissociation as
batch 1, now on a cost axis. Plan cost-priority #1 is dead.

Verdict (appear/disappear): **shallow, near-null optimum.** Best point in the batch beats
the untouched reference by **+0.0015** — noise. 2000 is clearly harmful (-0.015), and
`fn_nodes` degrades monotonically with the constant (179 -> 296) as expensive endpoints
suppress track starts. Plan cost-priority #2 is effectively exhausted.

**Net ranking of every lever measured so far: divisions +0.069 >> max_children 3->8
+0.017 >> appear/disappear +0.0015 > drift_constant (negative).** Cost tuning in the
division-free regime is essentially done; the remaining prize is divisions.

#### Side probe: does `max_edge_distance` recover the never-candidate edges? (No.)

Jobs 153765339/40, 3 timepoints, mc=8, `divisions=false`, `--stats-only`:

| max_edge_distance | simple edges |
|---|---|
| 15 | 142,126 |
| 25 | 142,234 |

**+108 edges (+0.076%) for a 67% larger radius.** `add_cand_edges` issues a single
`query(..., k=max_children, distance_upper_bound=max_edge_distance)`, so the radius only
adds an edge where a node has *fewer than* `max_children` neighbours already inside it;
where the k-limit binds, a wider radius changes nothing and never re-ranks. Batch 1's
near-perfect linear scaling in `max_children` (99.5% of slots filled at mc=8) already
implied the radius is slack nearly everywhere, and this measures it directly.

It also cannot help *these* edges in particular: FN edges sit at `Pred node dist`
6.479 +/- 2.912 (median 6.176), so ~99% are already inside 15 um. They are excluded by
**rank, not radius** — with true successors typically ~1.0 um away, a 6.2 um sphere holds
far more than 8 competitors.

---

## Batch 3 (2026-09-08) — division_weight bracket

Sweep `dro3`, on the cached mc=3 `divisions = true` graph (`ec45212f86f9879f.pkl`).
mc=3 rather than 5 because B1R5 showed mc=5-with-divisions costs 3x the model for
+0.0007 TE; mc=8-with-divisions was never built (~37M edges).
`appear/disappear` stay at 50 so this isolates `division_weight` — the batch-2 `ad500`
winner was +0.0015, i.e. noise, so folding it in would only blur the comparison.

`DivisionCost` is weight-only on a 0/1 `is_division` attribute, so `division_weight` acts
as a flat per-division penalty; its natural scale is `appear_constant + disappear_constant`
= 100. Bracketing widely first, per the plan, then narrowing to land at 20-40 divisions.

Runs: `dw0` (anchor, = `g_mc3_div`, 3955 divisions), `dw100`, `dw250`, `dw1000`,
`dw4000`, `dw16000`.

Hypothesis: "TE rises toward the division-free reference (0.7044 at mc=3) as
`division_weight` suppresses spurious divisions, and the ~30-division target sits at a
weight where most of the +0.069 is recovered."

#### Results (collected 2026-09-09)

Division counts from job 154186312 (`logs/dro_opt/count_div3.154186312.log`).

| run | `division_weight` | divisions | TE | fn_n | fn_e | SEG | LNK |
|---|---|---|---|---|---|---|---|
| `dw0` | 0 | 3955 | 0.6351 | 156 | 912 | 0.7283 | 0.8797 |
| `dw100` | 100 | 3554 | 0.6389 | 160 | 914 | 0.7283 | 0.8814 |
| `dw250` | 250 | 2963 | 0.6462 | 163 | 930 | 0.7246 | 0.8829 |
| `dw1000` | 1000 | 561 | 0.6903 | 185 | 1003 | 0.7178 | 0.8877 |
| `dw4000` | 4000 | **0** | **0.7043516** | 190 | 1021 | 0.7178 | 0.8893 |
| `dw16000` | 16000 | **0** | **0.7043516** | 190 | 1021 | 0.7178 | 0.8893 |

Verdict: **first half supported, second half falsified — and the second half is the
one the plan depends on.**

TE does rise monotonically as divisions are suppressed, exactly as predicted, and it
lands on the division-free reference *precisely*: `dw4000` and `dw16000` both give
TE = 0.7043515819026023, identical to full float precision to `dro1_g_mc3_nodiv`
(their graphs differ from it by a single isolated node, 230581 vs 230580, which does
not touch TE). The ceiling was predicted and hit exactly — a clean confirmation that
`division_weight -> inf` on a `divisions = true` graph is equivalent to `divisions =
false` at the same `max_children`.

But **the curve is monotonic all the way to zero — there is no interior optimum.**
Every division the solver keeps costs TE. So there is no `division_weight` at which
"most of the +0.069 is recovered" *and* divisions survive; recovery is complete only
when the division count is zero.

**The ~30-division target band was never sampled.** The curve jumps 561 -> 0 between
`dw1000` and `dw4000`, so ~30 divisions sits somewhere in that gap. Narrowing it is
not worth a batch on its own: monotonicity brackets the answer at **TE between 0.6903
and 0.7044**, i.e. the ~30-division requirement costs **0.017 to 0.031 TE** against
the mc=3 division-free reference, and ~0.017 to ~0.031 *plus* the mc=3 -> mc=8 gap
against the current best.

Note this batch ran at mc=3, so its ceiling (0.7044) is structurally below the
current best (0.7211 at mc=8). That was known at launch and does not affect the
verdict — the mechanism result is `max_children`-independent — but it does mean
**batch 3 could not have produced a new best**, and the shippable-config question
now points at an mc=8 `divisions = true` graph that has never been built.

## Batch 4 (2026-09-08) — drift_constant, less-negative direction

Sweep `dro4`, on the cached mc=8 nodiv graph. `collect_sweep` flagged
`drift_constant = -2000` as a HIGH-edge winner: every more-negative value was tested and
lost, so the optimum may lie *less* negative — a direction the plan never proposed.

Runs: `dc1500` (break-even 4.29 um), `dc1000` (2.86 um), `dc500` (1.43 um).

Hypothesis: "TE improves as the break-even drops toward the TP drift median (0.574 um),
because pricing out marginal edges reduces fragmentation."

#### Results (collected 2026-09-09)

| run | `drift_constant` | break-even | TE | fn_n | fn_e | SEG | LNK |
|---|---|---|---|---|---|---|---|
| *(ref, dro2)* | -2000 | 5.71 um | **0.7211** | 179 | 982 | 0.7178 | 0.8936 |
| `dc1500` | -1500 | 4.29 um | 0.7091 | 227 | 1044 | 0.7178 | 0.8868 |
| `dc1000` | -1000 | 2.86 um | 0.6628 | 405 | 1252 | 0.7060 | 0.8647 |
| `dc500` | -500 | 1.43 um | 0.5140 | 1141 | 2100 | 0.6661 | 0.7732 |

Verdict: **falsified, steeply and monotonically.** Every less-negative value lost TE,
and the loss accelerates: -0.012, -0.058, -0.207. Pricing out marginal edges does not
reduce fragmentation — it starves the solve of links outright, and `fn_nodes` blows up
7x (179 -> 1141) as whole tracks fail to form.

**The decisive result: combined with batch 2, `drift_constant = -2000` is now a
bracketed interior optimum.** Batch 2 tested -2500/-3000/-3500 (monotonically worse);
batch 4 tests -1500/-1000/-500 (monotonically worse). The axis is closed in both
directions and the untouched baseline value is the winner.

This also retires the flag that motivated the batch. `collect_sweep`'s "HIGH edge"
warning on batch 2 was an artifact of batch 2 sampling only one side of the optimum;
the check is per-sweep and cannot see across sweeps. Note that `dro4/results.csv` now
carries the mirror-image artifact — `dc1500` flagged at a LOW edge, "extend downward"
— which is likewise spurious, because the value it would extend toward is -2000, which
is already tested and already the winner. **Do not extend this axis.**

---

## Batch 5 (2026-09-08) — node/edge cost rebalance

Sweep `dro5_costbal`, spec `scratch_configs/sweeps/dro_01nuclei_batch5_costbal.toml`, on the
cached mc=8 nodiv graph (`5bcbe28db3b19a19.pkl`). Launched from a separate session
while the agent session owned batches 3-4; hence the distinct `sweep_id` and spec
name. Track jobs 154088595-154088617 (odd), evals 154088596-154088618 (even).

**Motivation.** Every batch through 4 tunes only edge constants and appear/disappear.
The attribute-statistics table from `dro2_ref` says that is the wrong half of the
model:

| attribute | attr mean | attr std | std/mean | cost std |
|---|---|---|---|---|
| cohesion | 0.993 | 0.009 | 0.009 | 15349.6 |
| adhesion | 0.090 | 0.262 | 2.91 | 20268.4 |
| drift_dist | 5.817 | 2.840 | 0.49 | 993.9 |

Node costs run **15-20x above** the edge costs, so the solve is almost entirely "which
nodes" and linking is a free rider. Two further readings: cohesion's *attribute* has
essentially no spread (std/mean 0.009, far under CLAUDE.md's 0.3 line), so with
`num_leaves` scaling it is a flat per-leaf size prior rather than a discriminator,
while adhesion (std/mean 2.91) carries the real per-node signal. And the two node
terms nearly cancel per leaf: cohesion `-500*0.993 + 450 = -46.5`, adhesion
`-100*0.090 + 50 = +41.0`, **net -5.5** — a difference of two ~45s, so net selection
pressure looked like a very sharp untested lever.

Three conditions, 12 runs, all cache hits (cost params are not in the cache key).
Where an axis would move both the cost mean and the cost spread, the constant is
co-adjusted to hold the mean fixed, so each condition isolates one effect:

| condition | runs | isolates |
|---|---|---|
| `cohesion_bias` | cc430/440/450/460/470 | net per-leaf pressure (`= cohesion_constant - 455.5`), swept -25.5 -> +14.5 **through zero**; weight fixed |
| `adhesion_gain` | aw100/200/300/500, `adhesion_constant` 50/59/68/86 | adhesion *discrimination* at fixed cost mean (+41.0/leaf) |
| `drift_gain` | dw350/700/1000, `drift_constant` -2000/-4036/-5781 | drift *discrimination* at fixed cost mean (+36.0/edge); cost std 994 -> 1988 -> 2840 |

`cc450` / `aw100` / `dw350` are deliberate duplicates of `dro2_ref` — per-condition
anchors so each axis brackets, and a triple cache-reproducibility check.

Hypotheses: (1) TE is sharply non-monotonic in `cohesion_constant` with an interior
optimum near the sign flip; (2) TE improves with adhesion spread at fixed mean;
(3) `drift_gain` moves little, and a null there confirms the node terms are binding.

#### Results (collected 2026-09-09)

| condition | run | TE | fn_n | fn_e | SEG | LNK |
|---|---|---|---|---|---|---|
| `cohesion_bias` | `cc470` | **0.7215204** | 182 | 984 | 0.7178 | 0.8934 |
| `adhesion_gain` | `aw300` | **0.7215204** | 182 | 984 | 0.7178 | 0.8934 |
| `adhesion_gain` | `aw500` | 0.7215 | 184 | 985 | 0.7178 | 0.8933 |
| `cohesion_bias` | `cc460` | 0.7213 | 180 | 982 | 0.7178 | 0.8936 |
| `adhesion_gain` | `aw200` | 0.7213 | 180 | 982 | 0.7178 | 0.8936 |
| `cohesion_bias` | `cc450` *(anchor)* | 0.7210884 | 179 | 982 | 0.7178 | 0.8936 |
| `adhesion_gain` | `aw100` *(anchor)* | 0.7210884 | 179 | 982 | 0.7178 | 0.8936 |
| `drift_gain` | `dw350` *(anchor)* | 0.7210884 | 179 | 982 | 0.7178 | 0.8936 |
| `cohesion_bias` | `cc430` | 0.7207 | 178 | 982 | 0.7178 | 0.8936 |
| `cohesion_bias` | `cc440` | 0.7207 | 178 | 982 | 0.7178 | 0.8936 |
| `drift_gain` | `dw700` | 0.7207 | 177 | 983 | 0.7178 | 0.8935 |
| `drift_gain` | `dw1000` | 0.7196 | 176 | 984 | 0.7178 | 0.8934 |

All three anchors reproduced 0.7210884353741497 exactly — the **fourth** cache
validation. Total spread across all 12 runs and all three axes: **0.0019**.

Verdict (cohesion_bias): **falsified.** Nearly flat and monotonic, +0.0008 across a
40-unit swing that carries net per-leaf pressure from -25.5 to +14.5, i.e. *through
the sign flip*. The near-cancellation arithmetic is correct; the ILP optimum is
simply insensitive to it over this range.

Verdict (adhesion_gain): **effectively falsified.** +0.0004 at 3x the adhesion
spread with the cost mean held fixed. The node term's per-node signal is not a lever.

Verdict (drift_gain): **supported, and it is the informative one.** Raising drift
discrimination at fixed mean makes TE slightly *worse* (0.7211 -> 0.7196 at 2.9x
spread). Edge-cost tuning is confirmed exhausted.

**The solution sits on a wide plateau.** `cc470` and `aw300` produced the
*bit-identical* integer solution (TE equal to 16 digits, fn_n 182, fn_e 984, same SEG
and LNK) from genuinely different configs — verified by diffing the generated
`track_configs/`. Two unrelated perturbations landing on one optimum is direct
evidence that the solve is locked against node-cost tuning, not merely insensitive.

**On calling small deltas "noise" (correcting the batch-2 framing).** The solver is
deterministic, and anchors reproduce bit-identically across four sweeps, so there is
no run-to-run noise floor on this dataset. `cc470`'s +0.0004 and batch 2's `ad500`
+0.0015 are *real* effects. They are simply negligible: TE averages over 189 GT tracks
of 50 frames each, so +0.0004 corresponds to roughly **4 frame-assignments across the
entire dataset**. The right verdict is "real but far too small to act on", not "noise".


---

## Cross-batch: FN edges as an error rate per GT edge (2026-09-09)

Denominator is **9,261 GT edges**, read from the baseline diagnostics
(`diagnostics_summary.json` -> `fn_edges.n_gt_edges`), not assumed. It matches the
189 GT tracks x 49 links implied by the sparse-GT structure.

| config | divs | TE | FN edges | err / GT edge | 1 err per |
|---|---:|---:|---:|---:|---:|
| B1R5 — mc=5, div=**true** | 5316 | 0.6358 | 849 | **9.17%** | 10.9 |
| Baseline / B1R1 — mc=3, div=true | 3955 | 0.6351 | 912 | 9.85% | 10.2 |
| B3 `dw1000` — division_weight 1000 | 561 | 0.6903 | 1003 | 10.83% | 9.2 |
| B1R2 — mc=3, div=false | 0 | 0.7044 | 1021 | 11.02% | 9.1 |
| **B1R4 / B2 `ref` — mc=8, div=false** *(incumbent)* | 0 | 0.7211 | 982 | 10.60% | 9.4 |
| **B5 `cc470` — mc=8, div=false** *(best TE)* | 0 | 0.7215 | 984 | 10.63% | 9.4 |

Every config lands in a narrow **9.2-11.0%** band — roughly **one missed link per 9 to
11 GT edges** — across a TE span of 0.635 to 0.722.

**The rate runs backwards against TE.** Sorted by error rate the ordering is close to
inverted: the lowest rate of the whole campaign is B1R5 at 9.17%, which has the
*second-worst* TE; the best-TE config sits at 10.63%, about 1.5 points worse per edge
than the baseline it beats by +0.086 TE. A per-GT-edge error rate is a **recall**
measure, and TE is not recall-limited but fragmentation-limited — division-heavy runs
recover more individual links and then spend them on spurious branches that truncate
each GT track's best match. **Do not steer on this rate.** It is a diagnostic of
linking completeness only, and it is the same trap as `fn_edges`/`fn_nodes`, now in
normalized form.

#### How much of the error was ever reachable by cost tuning

Splitting the baseline's misses by cause (this partition sums exactly to 9,261):

| cause | edges | % of GT edges | reachable by cost tuning? |
|---|---:|---:|---|
| Never candidates — graph construction | 332 | **3.58%** | no — absent from the graph |
| Endpoint node unmatched — segmentation | 270 | **2.92%** | no — upstream of tracking |
| Candidates the ILP rejected | 205 | **2.21%** | yes |

**Only 2.21 percentage points of a ~10% error rate were ever addressable by tuning ILP
costs.** The remaining 6.50 points are decided before the solver sees the graph. This
is the quantitative reason batches 2-5 came back flat, and the strongest argument for
the segmentation / merge-hypothesis direction.

**Accounting caveat.** The two sources disagree slightly on the baseline: CTC reports
`fn_edges = 912` (9.85%) while the diagnostics partition gives 537 + 270 = 807
(8.71%), a 105-edge gap. The diagnostic classifies purely on "both endpoints matched
*and* edge present", whereas CTC also carries `ws_edges = 290` and `ns_nodes = 54` for
this run, so many-to-one matching is penalized differently. The exact 105 edges have
**not** been traced — treat that as an unverified explanation. **Use the CTC
`fn_edges` column** for the rates above: it is the number in every results table and
is consistent across all sweeps. The structural split is available only for the
baseline, since diagnostics are disabled in the sweep eval configs
(`[eval.diagnostics] enabled = false`).

---

## Campaign status (2026-09-09) — cost tuning closed

Batches 2-5 ran 34 runs across every tunable cost axis. **None beat batch 1's
`dro1_g_mc8_nodiv` (TE 0.7211) by more than +0.0004.**

| axis | batch | verdict |
|---|---|---|
| `divisions` on/off | 1 | **+0.069 — the only real win of the campaign** |
| `max_children` | 1 | weak, saturating (+0.017 over 3->8) |
| `drift_constant` | 2 + 4 | **bracketed interior optimum at -2000**; worse in both directions |
| `appear` / `disappear` | 2 | +0.0015 best; 2000 costs -0.015 |
| `division_weight` | 3 | monotonic to the nodiv ceiling; no interior optimum |
| `cohesion_constant` (node bias) | 5 | +0.0008 through the sign flip |
| `adhesion_weight` (node spread) | 5 | +0.0004 at fixed mean |
| `drift_weight` (edge spread) | 5 | *worse* at fixed mean |

**Recommendation: stop tuning ILP costs on this dataset.** The config sits on a wide
plateau at a bracketed local optimum, and batch 5 established that even the dominant
node term — 15-20x the edge costs — cannot move it. This is consistent with what
batch 1 diagnosed structurally: the 332 never-candidate FN edges are a segmentation /
merge-hypothesis failure that no cost can reach.

Two live directions, both requiring a graph rebuild (new cache key):

1. **mc=8 with `divisions = true`, sweeping `division_weight` in the 1000-4000 gap.**
   The top experiment *if the ~30-division requirement stands* — batch 3 only ever ran
   at mc=3, whose ceiling (0.7044) is structurally below the current best, so no batch
   so far has tested a shippable division-preserving config at the winning
   `max_children`. ~37M edges.
2. **Upstream segmentation / merge hypotheses** — `min_merge_cost`, `max_merge_cost`,
   `size_threshold`, or a new `seg_result`. This is where the 332 never-candidate FN
   edges actually live.

Batch 5 was run from a separate session while LSF job 153758047 (the agent session
that owned batches 0-4) was idle; that job was killed 2026-09-09 after its results
were collected and merged here. Its raw terminal log is retained at
`logs/dro_opt/agent_153758047.out`.

---

## 03_nuclei_div — segmentation series under the linajea metric (2026-09-11 to 2026-09-16)

### Setup

**Dataset**: Fluo-N3DL-DRO / `03_nuclei_div`, the t 261-310 crop of the Amat 120828
movie (rebased to frames 0-49, full volume, no spatial crop), chosen because the CTC
`01`/`02` windows contain zero divisions while this crop has 120. Data
`data/Fluo-N3DL-DRO/03_nuclei_div.zarr`; flow `2026-09-10_16-36-58` (shared by every
run below).

**GT**: linajea figshare GT converted by `scripts/05_evaluation/linajea_to_geff.py`,
split into `gt_side_1` (64 divisions, tuning side) and `gt_side_2` (56 divisions,
held out) under `experiments/tracking/Fluo-N3DL-DRO/03_nuclei_div/`. **Tune on side 1
only.** The runs below did not tune on either side.

**Eval**: `metrics = ["linajea"]`, `linajea_gt = ["gt_side_1", "gt_side_2"]`,
`linajea_match_threshold = 6.09` (um; = 15 linajea world units), `linajea_sparse =
true`, diagnostics disabled. No traccuracy matcher is involved.

**Metric semantics.** `src/mhat/evaluation/linajea_metrics.py`, validated to 0.5%
against Malin-Mayor et al. 2023. Edge-level Hungarian matching; every count is
divided by `gt_edges` (5751 side 1, 5488 side 2); **`sum = FN + IS + FP-D + FN-D`,
lower is better**; FP edges are excluded, which is what makes the metric usable on
sparse GT. A predicted division is judged only when anchored to annotated GT, so
`rec_divisions` (total predicted divisions) is a diagnostic, not a penalty. Division
recall = `tp_divisions / gt_divisions`.

**Baselines on this crop** (`linajea_baselines_t261-310.json`, sum / division recall):

| method | side 1 | side 2 |
|---|---|---|
| linajea | 0.0570, 16/64 | 0.0833, 10/56 |
| TGMM | 0.2420, 6/64 | 0.3198, 2/56 |

The crop is harder than the full movie (linajea 0.046 / 0.054 there).

**Tracking config for every run** = `dro1_g_mc5_div` from 01_nuclei, unchanged:
`drift 350/-2000`, `cohesion -500/450`, `adhesion -100/50`, `appear = disappear =
50`, `division_weight = 0`, `max_children = 5`, `divisions = true`, `merges = false`,
`max_edge_distance = 15`, `size_threshold = 20`, all other costs 0. So the series is a
**one-variable comparison on the segmentation**; nothing about the ILP has been
tuned for divisions yet.

Not submitted through `launch_sweep.py` — hand-written configs in
`scripts/_scratch/dro_03div_full/` (cp3/cp4) and equivalent per-run configs for the
first two; logs under `logs/dro_03div/`, `logs/dro_03div_aniso5fs1/`,
`logs/dro_03div_full/`. `logs/dro_03div_aniso5fs1/` also holds a set of diagnostic
jobs on the `aniso5_fs1` result (candidate recall, detectability, nucleus size,
off-axis; 2026-09-14) whose conclusions are **not** written up here.

### Runs

All cellpose 3D, `merge_thresholds = [1.0]`, `smoothing = [1,1,1]`, 3D affinities.
Seg jobs on `gpu_a100`, 12 slots, one GPU, ~8.3-8.5 h and ~113 GB each.

| exp_uid | seg (`seg_result`) | cellpose deltas | seg job | trk job (eval job) | graph cache | build+solve | peak mem |
|---|---|---|---|---|---|---|---|
| `dro03div_mc5_div` | `2026-09-10_16-35-38` | isotropic, `cellprob 0` | not traced (`2026-09-10_16-10-39` sibling dir is an earlier attempt) | 154271867 | `4443c39e549434e4` (1.11 GB) | 4.0 h / 16 slots | 57 GB |
| `dro03div_aniso5fs1_mc5_div` | `aniso5_fs1` | `anisotropy 5`, `flow3D_smooth 1`, `cellprob 0` | 154271931 | 154273254 (154273255) | `1980a812b9cc15dd` (0.97 GB) | 3.0 h / 8 slots | 45 GB |
| `dro03div_cp3_mc5_div` | `aniso5fs1_cp3` | + `cellprob -3` | 154286573 | 154286574 (154286575) | `c5e96237cd6fdf75` (1.84 GB) | 21.3 h / 12 slots | 94 GB |
| `dro03div_cp4_mc5_div` | `aniso5fs1_cp4` | + `cellprob -4` | 154286576 | 154286577 (154286578) | `de2afc7416e3f85e` (2.20 GB) | 28.4 h / 12 slots | 116 GB |

The Gurobi solve itself is 15-55 min in every case (925 / 1014 / 3204 / 2597 s, one
node, root LP); the rest is the candidate-graph build, which grows with object count.
`seg_result` is in the cache key, so every run was a cache MISS.

#### Results (linajea, both sides; `sum` is the number to compare)

| run | side | sum | FN edges | IS | FP-D | FN-D | div recall | matched / GT edges | pred divisions |
|---|---|---|---|---|---|---|---|---|---|
| `dro03div_mc5_div` | 1 | 0.7460 | 0.7181 (4130) | 39 | 57 | 64 | 0/64 | 1621 / 5751 | 14240 |
| | 2 | 0.8161 | 0.7815 (4289) | 66 | 69 | 55 | 1/56 | 1199 / 5488 | |
| `dro03div_aniso5fs1_mc5_div` | 1 | 0.6743 | 0.6575 (3781) | 21 | 13 | 63 | 1/64 | 1970 / 5751 | 8039 |
| | 2 | 0.7001 | 0.6835 (3751) | 13 | 24 | 54 | 2/56 | 1737 / 5488 | |
| `dro03div_cp3_mc5_div` | 1 | 0.3669 | 0.3398 (1954) | 51 | 53 | 52 | 12/64 | 3797 / 5751 | 11545 |
| | 2 | 0.3427 | 0.3087 (1694) | 63 | 72 | 52 | 4/56 | 3794 / 5488 | |
| **`dro03div_cp4_mc5_div`** | 1 | **0.2841** | 0.2500 (1438) | 70 | 78 | 48 | **16/64** | 4313 / 5751 | 13069 |
| | 2 | **0.2684** | 0.2278 (1250) | 84 | 91 | 48 | 8/56 | 4238 / 5488 | |
| *linajea* | 1 | *0.0570* | *0.0261 (150)* | *117* | *13* | *48* | *16/64* | *5601 / 5751* | *1028* |
| | 2 | *0.0833* | *0.0547 (300)* | *101* | *10* | *46* | *10/56* | *5188 / 5488* | *777* |
| *TGMM* | 1 | *0.2420* | *0.2198 (1264)* | *40* | *30* | *58* | *6/64* | *4487 / 5751* | *3096* |
| | 2 | *0.3198* | *0.3016 (1655)* | *25* | *21* | *54* | *2/56* | *3833 / 5488* | |

FN-D breakdown (side 1 / side 2) as `no_connection + one_unconnected_child`:
`mc5_div` 56+7 / 53+2; `aniso5fs1` 58+5 / 49+5; `cp3` 33+19 / 28+24; `cp4` 25+23 / 19+29.
`unconnected_parent` is 0-1 everywhere.

### S1: isotropic cellpose, cellprob 0 (`dro03div_mc5_div`, 2026-09-11)
exp_uid: `dro03div_mc5_div`
Hypothesis: "The 01_nuclei best config transfers; the crop's error is dominated by
division handling, since `division_weight = 0` was never tested against real divisions."
sum: 0.7460 / 0.8161, FN edges 71.8% / 78.2%, IS 39 / 66, div recall 0/64 / 1/56
Verdict: **falsified — the error is upstream of tracking.** Seven of ten GT edges are
missed outright and only 28% / 22% of GT edges are matched at all; division handling
is invisible behind that. 14,240 predicted divisions against 120 in GT, none of them
on annotated lineages. Worse than TGMM by 3x on the sum.

### S2: anisotropy 5, flow3D_smooth 1 (`dro03div_aniso5fs1_mc5_div`, 2026-09-11)
exp_uid: `dro03div_aniso5fs1_mc5_div`
Hypothesis: "Telling cellpose the true z anisotropy (5x) and smoothing the 3D flows
recovers objects the isotropic model splits or drops, and FN edges fall with them."
sum: 0.6743 / 0.7001, FN edges 65.7% / 68.3%, IS 21 / 13, div recall 1/64 / 2/56
Verdict: **supported but small.** -0.07 / -0.12 on the sum, all of it FN edges; IS and
FP-D both roughly halve (cleaner objects, fewer spurious branches: predicted divisions
14,240 -> 8,039). Still two thirds of GT edges missed, so detection, not linking, is
the bottleneck. The diagnostic jobs on this result (2026-09-14,
`logs/dro_03div_aniso5fs1/`) motivated the cellprob step below.

### S3: cellprob_threshold -3 (`dro03div_cp3_mc5_div`, 2026-09-15)
exp_uid: `dro03div_cp3_mc5_div`
Hypothesis: "The missed GT nuclei are dim and fall below cellpose's default mask
threshold; lowering `cellprob_threshold` recovers them and FN edges drop sharply."
sum: 0.3669 / 0.3427, FN edges 34.0% / 30.9%, IS 51 / 63, div recall 12/64 / 4/56
Verdict: **supported — the largest single step in the DRO campaign.** Sum halves on
both sides; matched GT edges go 1970 -> 3797 (side 1). Division recall appears for the
first time (12/64) *without any division cost*, and the FN-D failure mode shifts: at
cp 0 nearly every missed division had no connected child at all, now 19 of 52 have one
child linked and the other not. Cost: IS and FP-D climb back (more objects, more
branching), but both stay ~0.01 normalized.

### S4: cellprob_threshold -4 (`dro03div_cp4_mc5_div`, 2026-09-16) — **current best**
exp_uid: `dro03div_cp4_mc5_div`
Hypothesis: "Still monotonic: one more unit recovers more dim nuclei than it adds
false objects."
sum: 0.2841 / 0.2684, FN edges 25.0% / 22.8%, IS 70 / 84, div recall 16/64 / 8/56
Verdict: **supported, and the optimum is not bracketed.** -0.08 / -0.07 on the sum;
FN edges are still 88% of the remaining error. Side-1 division recall (16/64) now
equals linajea's on this crop, side 2 (8/56) is two short, all with
`division_weight = 0`. Now at TGMM level on side 1 and better than TGMM on side 2,
still 4-5x linajea. IS (70/84) is under linajea's own (117/101), so identity switches
are not yet the cost of going lower; FP-D (78/91) is where MHAT is worst relative to
linajea (13/10) and is the first thing to watch as the threshold drops further.

### Series summary

- **`cellprob_threshold` is the lever.** 0 -> -3 -> -4 gives sum 0.674 -> 0.367 ->
  0.284 (side 1) and 0.700 -> 0.343 -> 0.268 (side 2), monotonic, driven by FN edges.
  The last step was still the size of the TGMM-linajea gap, so **-5 and -6 must be run
  before anything else**; MDA231's best sits at -6 for the same reason.
- **Anisotropy / flow smoothing alone was worth 0.07-0.12**; the threshold did the rest.
- **The failure mode has moved from detection toward linking.** At cp 0, ~90% of
  missed divisions had no child connected; at cp -4 it is about half, and IS / FP-D
  are rising. Once the threshold is bracketed, `division_weight` (never tuned against
  real divisions — see the 01_nuclei caveat) and the drift / appear terms become live
  again on this dataset, on side 1 only.
- **Both sides move together** (side 2 is now slightly *better* than side 1), so the
  series is not overfitting the tuning side.
- Nothing here changes the 01_nuclei conclusions: this crop has divisions, the CTC
  window does not, and TE was never computed for these runs.

### Next: sweep `dro03div`, cellprob -5 / -6 (written 2026-09-18, not yet submitted)

Spec `scratch_configs/sweeps/dro_03div_cp56.toml`; seg configs and the submit script in
`scratch_configs/experiments/dro_03div_cp56/` (`submit.sh` submits the two GPU seg jobs, then
`launch_sweep.py --dep "ended(seg5) && ended(seg6)"`). Runs `cp5_mc5_div` /
`cp6_mc5_div` -> exp_uids `dro03div_cp5_mc5_div` / `dro03div_cp6_mc5_div`, segs
`aniso5fs1_cp5` / `aniso5fs1_cp6`, tracking and eval identical to S4. Tracking sized
at 16 slots / 48 h (cp4 took 28 h and 116 GB on 12).

Hypothesis: "sum keeps falling to at least -5; the curve flattens or turns by -6 as
FP-D and IS overtake the FN-edge gain, bracketing the optimum at -4 / -5 / -6."

`launch_sweep.py` was changed for this sweep: a linajea-only `[eval]` no longer
requires `matcher`, and `--dep` gates every tracking job on an LSF dependency.
