# GT-amount experiment — cluster run plan

**Question.** How much ground truth does SSVM weight fitting actually need, and does it
matter *how* the annotation budget is spent? Raw data, segmentation and optical flow are
held fixed; only the amount and the shape of the annotation vary. Regularization is
calibrated once, up front, so it cannot confound the curve.

Two protocols are compared at equal annotated-GT-node count:

- **sparser GT** over the whole field of view (arms A and B, parameterized by track count)
- **a smaller densely annotated region** (arm C, parameterized by crop volume fraction)

**Dataset.** `NC281-sparse-label`
- train `02_nuclei_denoised_train` — **167** GT tracks (funtracks-renumbered; the raw
  geff carries 164 `track_id`s, see *Label spaces* below), seg `2026-07-02_10-32-29`,
  flow `2026-07-02_10-43-07`
- test `02_nuclei_denoised_test` — 179 GT tracks, seg `2026-07-08_14-40-36`,
  flow `2026-07-08_14-46-03`

**Environment.** `conda run -n mhat2`, `module load gurobi`. Branch `ssvm-fit`.

**Reference points on test.** hand-tuned full GT `test_full` TE = 0.8290; SSVM full GT
at the old default `ssvm_reg = 0.1` `ssvm_test` TE = 0.7209.

---

## Ground rules

1. **Nothing is selected on the test set.** Test is scored once, for weights already
   chosen. No hyperparameter, threshold or arm is picked using a test number.
2. **No oracle conditions.** Every arm may use only what a real annotator would
   possess. The one exception is stage 1, which *is* the fully-annotated condition,
   so scoring it against the complete train GT is legitimate.

   **This was verified on real data, not merely audited** (2026-08, gates G7/G8,
   since retired): deriving an arm-A fit graph from the subset directory alone —
   where the unannotated tracks are absent from disk — reproduced the dense-GT path
   exactly (node set, edge set, every `gt_selected` label, exclusion sets, all
   counts); and for arm C, no surviving candidate overlapped an unannotated GT
   track while negatives still survived. The verifier scripts were deleted once
   both passed end to end. **If `assign_gt_labels`, `gt_subsets.py`, `gt_crops.py`
   or the driver's arm handling is ever edited again, that guarantee is no longer
   under test** — restore the verifiers and re-run them rather than assuming it
   still holds. They exist in exactly one commit, `36968df` ("for git posterity,
   will remove once verified"), and were deleted in `5da9400`:

   ```bash
   git show 36968df:scripts/04_tracking/verify_arm_a_no_oracle.py > scripts/04_tracking/verify_arm_a_no_oracle.py
   git show 36968df:scripts/04_tracking/verify_crop_no_oracle.py  > scripts/04_tracking/verify_crop_no_oracle.py
   python scripts/04_tracking/verify_arm_a_no_oracle.py <config> --subset n050_s0
   python scripts/04_tracking/verify_crop_no_oracle.py  <config> --crop b0821_s1
   ```

   Both need stage 2a to have run (the subsets and crops must exist on disk).

   **One dense-GT dependency remains, deliberately.** Stage 1 calibrates `ssvm_reg`
   on the fully annotated dataset and stage 2 imposes that single scalar on every
   subset fit. So dense-GT knowledge does reach the small-N fits — through the
   hyperparameter, not through the labels. This is the protocol as designed
   (calibrate once on a fully annotated reference, reuse thereafter), but say so
   explicitly in the writeup rather than claiming the dense GT was fully removed.
   It matters most at the low end, where earlier data showed small-N fits prefer a
   *lighter* regularizer than the full-GT optimum. Closing it would mean calibrating
   per subset against that subset's own GT, which changes what the curve measures —
   from "fixed calibrated operating point" to "best achievable per annotation budget".
3. **Report TE.** Use `target_effectiveness` — GT-track coverage, so it depends only
   on tracks that exist. Against a subset GT, `track_purity`, precision and F1 are
   meaningless (unannotated objects count as false positives). Purity ≈ 1.0 on a
   near-empty solution is an artifact, not a result.

   **Every reported number comes from the TEST split.** Train-side TE could only
   honestly be scored against each run's own annotation, which is not comparable
   *between* protocols — crops truncate tracks and TE rewards short tracks — so it
   was dropped entirely. See stage 3.

4. **Compare arms on annotated GT nodes, never on track count.** A crop truncates the
   tracks it touches, so "64 tracks" in arm C is far less annotation work than 64 tracks
   in arm A. The shared budget is `n_gt_nodes_annotated` (objects × frames drawn), which
   `collect_gt_amount.py` records for every arm. Set `x_field = "n_gt_nodes_annotated"`
   in the curve config.

5. **Crops are sized by annotation budget, not by volume.** `crop_target_gt_nodes =
   "match_sizes"` grows each crop until it holds about as many annotated GT nodes as the
   corresponding track subset cost, so arm C's points land on top of arm A/B's instead of
   scattered between them. Sizing by a fixed *volume* fraction instead does not work on
   this data: the GT is spatially clumpy, so at 50% of the volume seed 0 caught 643 GT
   nodes and seed 1 caught 1780 — a 3× spread landing directly on the x-axis. Budget
   sizing collapses that to ±10 nodes (824 / 822 / 821 / 818 / 818 for a target of 821)
   and lets the crop *volume* absorb the density difference instead (27% to 55% of the
   field for that same budget).

6. **Crop placement is still drawn from the seed alone** — only the *size* adapts, never
   the position. That is what keeps budget sizing from becoming a "lucky crop" oracle:
   the search counts only the GT inside the box it is considering, which is the
   annotator's own completed work, never the GT outside. "Annotate outward from here
   until you have drawn N objects, then stop" is an ordinary way to spend a fixed budget.

   *Caveat at the largest budget:* a box needing ~50–70% of the field can only be placed
   within a narrow margin before clamping, so two seeds can clamp to the same corner. At
   `b1634` seeds 2 and 3 produce an identical box, giving 4 effective replicates rather
   than 5. This is geometry, not a bug — but do not read the top-budget error bar as five
   independent draws.

---

## Stage 0 — cluster paths

The committed configs use `Y:/jennifer/mhat/...`. Rewrite the three path keys to
`/groups/sgro/sgrolab/jennifer/mhat/...` in every config used below:

```
raw_base_dir    = "/groups/sgro/sgrolab/jennifer/mhat/data/"
input_base_dir  = "/groups/sgro/sgrolab/jennifer/mhat/experiments/"
output_base_dir = "/groups/sgro/sgrolab/jennifer/mhat/experiments/"
```

Leave everything else alone. The driver config is
`configs/tracking/NC281-sparse-label/02_nuclei_denoised_train/gt_amount/gt_amount_stage2.toml`.

## Stage 0b — where everything lives

This sweep is ~80 runs across two datasets. Left flat they buried the ordinary runs
sharing those dataset directories, so each stage groups its runs in a subdirectory,
mirrored identically under `experiments/tracking/`, `experiments/evaluation/` and
`configs/tracking/`:

| stage | subdirectory | train | test |
|---|---|---|---|
| 1 — regularizer calibration | `gt_amount/stage1_regsweep/` | 19 fits | — |
| 2 — GT-amount fits | `gt_amount/stage2_fits/` | 61 fits | — |
| 3 — test solves | `gt_amount/stage2_solves/` | — | 59 solves |

Materialized GT sits alongside them at `.../02_nuclei_denoised_train/gt_amount/{gt_subsets,gt_crops}/`,
and the stage-2 CSVs and figures at `experiments/evaluation/.../<dataset>/gt_amount/`.

Set by four keys in the driver config — `regsweep_subdir`, `runs_subdir`,
`test_runs_subdir`, and the existing `gt_subsets_dirname` / `gt_crops_dirname`.
Leaving them unset restores the old flat layout, which is what every other sweep in
the repo still uses.

Run **tokens stay flat** (`gt2_A_n050_s0`): they are the CSV `run` column, the
`run_summary_<token>.json` stems and what `parse_run_token` matches on a single
directory name. Only the *path* to a run and the `exp_uid` / `track_result` that
addresses it carry the subdirectory, e.g.
`track_result = "gt_amount/stage2_fits/gt2_A_n050_s0"`. Every consumer resolves
those as `<dataset root> / <value>`, so nothing else had to change.

---

## Stage 1 — calibrate the regularizer at full GT

**Single job. Do not fan out** — the graph, GT and overlap matrices are built once and
shared across the grid (~4 min setup, ~1 min per value). Splitting it would pay the
setup once per grid point to save seconds.

```bash
python scripts/04_tracking/sweep_ssvm_reg.py <gt_amount_stage2.toml>
python scripts/05_evaluation/run_evals.py \
    --config-dir configs/tracking/NC281-sparse-label/02_nuclei_denoised_train/gt_amount/stage1_regsweep/eval
python scripts/04_tracking/sweep_ssvm_reg.py <gt_amount_stage2.toml> --collect
```

Stage 1 and stage 2 share one config: `SWEEP_KEYS` in `sweep_ssvm_reg.py` strips every
stage-2-only key (arms, sizes, seeds, crops, `ssvm_reg_effective_target`, the subdir
keys), so the regsweep sees only the tracking parameters and fits on the full GT.

Grid: even twelfth-decade, `0.1 … 0.00316` (19 values), spanning effective reg ≈ 34 →
1088 at full GT (`n_labeled = 10877`). Resolution is spent inside the range the optimum
was expected to fall in rather than on decades outside it. The grid nests the familiar
quarter-decade points (`0.0562`, `0.0316`, `0.0178`, `0.01`) and includes the old
default `0.1` (effective 1087.7), so it stays comparable to earlier work. All fits use
the complete train GT and are scored against it on train.

The grid was run in two passes: `0.1 … 0.01` (13 values) first, then a six-value
half-decade extension `0.00825 … 0.00316` **appended** to `DEFAULT_REGS` in
`sweep_ssvm_reg.py` after the first pass came back still climbing at the bottom.
Extensions must always be appended, never inserted or reordered — the run directory is
`regsweep_r{list index}`, so inserting a value would silently repoint completed runs at
different regularizers.

`--collect` prints the table and the line you need:

```
ssvm_reg_effective_target = <value>
```

where `effective_reg = ssvm_reg × n_labeled_variables`, and `n_labeled = 10877` at
full GT (2896 nodes + 7981 edges).

**Gate:** if the winner sits at either end of the grid the optimum is not bracketed —
`--collect` warns. Extend the grid in that direction and rerun before continuing. Treat
a near-edge win on a flat tail as an edge hit too: `--collect` only warns on the literal
endpoint, and an argmax one point in, sitting on a plateau whose far side is unexplored,
is not bracketed in any useful sense.

**Result (2026-08-04): `ssvm_reg = 0.0121` → `ssvm_reg_effective_target = 131.61`.**
All 19 fits `status: ok`, `n_labeled = 10877` throughout. The curve has two regimes with
a clean knee. From `0.1` down to ≈ `0.015` TE climbs steeply and monotonically
(0.7512 → 0.8093) — that whole upper decade was over-regularized. Below ≈ `0.015` it
flattens: `r10`–`r16` span TE 0.8047–0.8093, a 0.0046 band that is within run-to-run
noise. At the very bottom (`r17`, `r18`) TE and purity fall together (purity 0.9206 /
0.9190 vs a flat ≈ 0.928 everywhere above) and `n_solution_nodes` drops to 2642 —
under-regularization starting to cost coverage *and* precision.

So 0.0121 is the top of a plateau, not a sharp peak, but the plateau is genuinely
bounded on both sides and the gate passes on substance. Effective reg anywhere in
≈ 50–160 would perform about equally at full GT; if a stage-2 subset run looks
pathological at 131.61, that is a plateau to move within, not evidence the calibration
was wrong.

*Note on the prior expectation:* earlier work suggested a winner around `ssvm_reg`
≈ 0.02–0.05 (effective ≈ 261). The actual winner is below that. This is expected rather
than alarming — the 2026-08-03 masking fix changed the labeled-variable count, so the
old effective-261 target does not carry over.

---

## Stage 2 — GT-amount sweep at the calibrated regularizer

Copy `gt_amount.toml` → `gt_amount_stage2.toml` and set:

```toml
run_name_prefix            = "gt2"          # any unused prefix
ssvm_reg_effective_target  = <from stage 1>
sizes                      = [100, 50, 25, 12]
include_full               = true           # appends the true full size (167)
seeds                      = [0, 1, 2, 3, 4]
arms                       = ["A", "B", "C"]
crop_target_gt_nodes       = "match_sizes"  # one budget per entry in `sizes`
crop_growth_step           = 0.05
crop_axes                  = ["y", "x"]     # full 20 frames and all 25 z-slices
crop_membership            = "contained"
```

`crop_axes = ["y", "x"]` is a deliberate choice: shrinking only the field of view keeps
track lengths intact, so a crop reduces the *number* of annotated cells without also
shortening them. Adding `"t"` would confound spatial and temporal budget in one number.

`"match_sizes"` resolves to the mean annotated-GT-node count of the track subsets at each
size — **[1634, 821, 416, 196]** on this dataset with `sizes = [100, 50, 25, 12]` and 5
seeds. Those numbers go into the run tokens (`gt2_C_b0821_s3`), so they must not drift:
the first resolve pins them in `gt_crops/crop_budgets.json`, and a later run that
resolves anything different — the easy way being a fan-out that narrowed `--seeds`,
since the budgets are a mean over seeds — exits with an explanation rather than quietly
starting a second token namespace. **Do not pass `--seeds` in the 2c fan-out.**

An explicit list works too, and skips all of that: `crop_target_gt_nodes = [1634, 821,
416, 196]`. Leaving `crop_target_gt_nodes` unset falls back to fixed-volume crops via
`crop_fractions` (tokens `f500`, `f250`, …), which is kept for comparison but is not the
recommended setup — see ground rule 5.

Do **not** set `ssvm_reg_normalize` as well; the explicit target takes precedence but
leaving both set is confusing.

### 2a. Materialize the GT subsets — once, before fanning out

```bash
python scripts/04_tracking/run_gt_amount_experiment.py <gt_amount_stage2.toml> --materialize-only
```

Writes 21 dirs under
`experiments/tracking/NC281-sparse-label/02_nuclei_denoised_train/gt_amount/gt_subsets/n<N>_s<seed>/`
plus 20 under `.../gt_amount/gt_crops/b<NNNN>_s<seed>/` (each: `correct_tracks.zarr`,
`correct_seg.zarr`, `subset.json`), and `gt_crops/crop_budgets.json`. **Concurrent jobs
would race on these**, so this must complete first. Existing dirs are skipped — for
crops, only after checking the box on disk matches the one this config asks for, so
changing `crop_axes` or `crop_growth_step` cannot silently reuse a stale crop (it errors
and tells you to pass `--force-gt`).

The crop-growth search is cheap and needs no candidate graph: `build_gt_voxel_index`
pulls the GT's own nonzero voxels out once (~10⁶ of the volume's 131M), and each "how
many GT nodes are in this box?" query is a boolean mask over that. A coarse
`crop_growth_step` scan brackets the target, then a bisection lands within a few nodes of
it — hundreds of queries per seed, all told a few seconds.

Crop GT objects are written **whole**, never clipped at the box edge: an annotator who
annotates a cell annotates all of it, and clipping would corrupt every area and IoGT the
fit and the evaluation compute. A few `correct_seg.zarr` labels have no node in
`correct_tracks.zarr` (3 on this dataset); they are dropped and counted in `subset.json`,
matching `remap_seg_to_track_ids`, which already renders them background.

### 2b. Verify before spending the fleet

```bash
python scripts/04_tracking/run_gt_amount_experiment.py <gt_amount_stage2.toml> --self-test
python scripts/04_tracking/run_gt_amount_experiment.py <gt_amount_stage2.toml> --dry-run
```

`--self-test` needs no data (checks the graph surgery against a real solver, plus crop
nesting, seed sensitivity, bbox union and containment). `--dry-run` runs all 61
combinations' annotation + masking with assertions, no fitting. Both must pass.
Assertions cover: full GT masks nothing; subsets nest; positives scale with N; **node
removal destroys no GT-positive edge**; arm B's unlabeled count matches; and for arm C,
that the mask is exactly the out-of-box set, that **no surviving candidate overlaps a GT
track the crop did not annotate**, and that negatives survive at all.

### 2c. Fan out the fits — 61 jobs

One job per token: `gt2_full_n167_s0`, `gt2_{A,B}_n{100,050,025,012}_s{0..4}`, and
`gt2_C_b{1634,0821,0416,0196}_s{0..4}` (budget tokens come from stage 2a — read them off
`gt_crops/crop_budgets.json` rather than assuming):

```bash
python scripts/04_tracking/run_gt_amount_experiment.py <gt_amount_stage2.toml> --only <token>
```

Each job rebuilds the graph (~4 min) then fits (~1 min) — wasteful serially,
irrelevant in parallel. Per-job summaries are written as `run_summary_<token>.json`
(they no longer collide).

**Expected non-failures:** arm B at N=12 (and sometimes N=25) produces an empty
solution. That is a result, not a job error. Those runs write `fit_summary.json` with
`status: empty_solution` and no prediction; downstream scripts skip and report them.

---

## Stage 3 — score (test only)

The test GT is dense, full-length and identical for every arm. Every reported number
comes from here.

```bash
python scripts/04_tracking/make_gt_amount_test_configs.py <gt_amount_stage2.toml>
python scripts/04_tracking/solve_many_weights.py \
    --config-dir configs/tracking/NC281-sparse-label/02_nuclei_denoised_test/gt_amount/stage2_solves --skip-existing
python scripts/05_evaluation/run_evals.py \
    --config-dir configs/tracking/NC281-sparse-label/02_nuclei_denoised_test/gt_amount/stage2_solves/eval --skip-existing
```

`solve_many_weights.py` is **one job**: it builds the test candidate graph once and
loops (~2 s per solve). Fanning it out would pay 41 graph builds to save ~80 s.

### Why there is no train-side scoring

There used to be a stage 3a that scored each fit against its own annotated subset
(`make_gt_amount_train_evals.py`, deleted 2026-08-05). It was honest — no oracle,
each run scored against exactly what its own protocol produced — but it could not
support the comparison the experiment exists to make, because the protocols produce
GT of different *shape* and TE is sensitive to that:

| GT used for scoring | nodes/track |
|---|---|
| full train GT | 16.4 |
| `n100_s0` / `n050_s0` / `n025_s0` (arms A, B) | 16.2 / 16.0 / 16.0 |
| `b1634_s0` / `b0821_s0` / `b0821_s1` (arm C) | 12.3 / 11.4 / 12.5 |

The sparse arms sample by track id and keep tracks whole; a crop truncates every
track that walks out of the box. A short track is easier to cover end to end, so arm
C's train TE ran optimistic relative to A/B's — by a margin that varied per crop
(`b0416_s0` landed at 17.3 nodes/track, its tracks having stayed inside), so not even
a constant offset that could be subtracted out. That left it usable only for
within-arm sanity checks, which the test split already covers.

**Do not reintroduce it by pointing an eval at the full train GT instead.** Scoring a
reduced-GT fit against the complete train GT is an oracle — that was the original
sin this whole protocol was rebuilt to remove.

---

## Stage 4 — collect and plot

```bash
B=/groups/sgro/sgrolab/jennifer/mhat/experiments/evaluation/NC281-sparse-label

python scripts/05_evaluation/collect_gt_amount.py <gt_amount_stage2.toml> \
    --output $B/02_nuclei_denoised_test/gt_amount/stage2_test.csv

python scripts/07_plotting/gt_amount_curve.py <curve_config.toml>
python scripts/07_plotting/gt_amount_curve.py <curve_config.toml> \
    --metrics target_effectiveness edge_recall node_recall --output panels.png
```

`collect_gt_amount.py` still has an `--eval-dataset` flag, but with the train-side
evals gone the train dataset has no `track_metrics.json` to read, so test is the only
useful target.

Curve config: `configs/evaluation/gt_amount_curve.toml` produces the figure
(`gt_amount_curve.png`, plus `gt_amount_panels.png` from the `--metrics` form).
To make a variant, copy it, point `csv_path` at the new CSV, list three arms
(`A`, `B`, `C`), and set

```toml
x_field = "n_gt_nodes_annotated"
xticks  = []          # crop budgets are measured, so fixed track ticks no longer fit
```

The multi-condition `arm|label` form is only needed when comparing sweeps via
`merge_gt_amount_csvs.py`.

x = annotated GT nodes, y = TE, one line per arm, mean over 5 seeds ± SEM, seed points
scattered, empty-solution runs drawn as open markers at 0 and **excluded from the mean**.
Leaving `x_field` at its `n_tracks` default reproduces the old two-arm figure, but arm C
must not be plotted on that axis — a crop-truncated track is not the same unit of work as
a full-length one.

---

## What the three arms mean

All three are realizable protocols.

| arm | what is unlabeled, and why | what happens to it | what it represents |
|---|---|---|---|
| **A** | touches **none of the annotated tracks** — decidable from the annotations alone | deleted from the fit graph with its edges; exclusion sets pruned | restrict fitting to the annotated neighbourhoods — a true ignore |
| **B** | same rule as A | `gt_selected = None`, kept in the graph | motile's documented sparse-GT API |
| **C** | not **fully inside the annotated box** — decidable from the box alone, without looking at the GT at all | deleted from the fit graph, as in A | annotate one region exhaustively instead of everything thinly, until the budget runs out |

**A vs B.** They are not equivalent, and that is the first finding to reproduce.
`structsvm.HammingCosts` zeroes the loss coefficient for masked variables, but
`SoftMarginLoss` still builds `_d = features @ ground_truth` and `a = <f, y'>` from the
**unmasked** vector, which is 0 in masked slots (`soft_margin_loss.py:70,96`). A masked
variable therefore still enters the margin and gradient as "GT says do not select" — a
soft negative with no counterbalancing penalty — so arm B drifts toward selecting
nothing as the GT shrinks. Arm A avoids it because the variable does not exist.

**A/B vs C.** The difference is *which* negatives exist, not how many. The sparse arms
keep every competing merge hypothesis around an annotated cell — those all touch an
annotated track — and their negative fraction actually *rises* as the GT shrinks
(173/923 at N=50 vs 348/2896 at full GT), because the survivors get enriched in
multi-cell merged hypotheses. What they can never supply is a candidate overlapping no
annotated object **at all**: a detection that is simply wrong. Under sparse annotation
that is indistinguishable from a cell the annotator skipped, so it is masked. Measured:
74 such candidates exist in this graph, and arms A and B label **0** of them at every
budget, N=100 and N=12 alike. A crop supervises them normally (36 of the 74 at
`f250_s1`). The driver prints this as the `neg_fp` column and records it as
`n_neg_no_gt_overlap`.

So arm C's negative sample is selected by geometry rather than by proximity to an
annotated cell — unbiased with respect to the GT, and containing spurious detections in
the proportion the solver meets at inference.

**Why C's membership rule is what it is.** A GT object is annotated iff it *intersects*
the box; a candidate is fitted iff it is *fully contained* in the box. The asymmetry is
deliberate and makes the pairing exact: any GT object overlapping a used candidate shares
a voxel with a region lying entirely inside the box, so it intersects the box, so it is
annotated. No candidate can be labeled `0` because of a cell the annotator never
annotated — the failure mode that made the pre-2026-08-03 sparse rule an oracle. This
was confirmed on real data (`b0821_s1`) rather than left as an argument, by the
since-retired gate G8; see ground rule 2 on when to resurrect it.

**Boundary transitions are charged normally.** A track walking out of the box loses its
successor candidate, so the fit sees a selected node with no outgoing edge and pays
`disappear_constant` — the crop is treated as a standalone dataset, which is what
cropping the raw data and re-segmenting would have given. Expect learned appear/disappear
constants to drift downward as the crop shrinks and read them with that in mind; it is a
property of the protocol, not a bug. (`add_appear_ignore_attr` / `add_disappear` still
exempt t=0 and t=T−1; with an XY-only crop the time range is untouched, so those stay
correct.)

---

## Verification gates

| gate | check |
|---|---|
| G1 | `--self-test` passes (graph surgery + a real solver build) |
| G2 | `--dry-run` passes all assertions across all 41 combinations |
| G3 | stage-1 winner is **not** at a grid edge |
| G4 | full-GT run at `ssvm_reg = 0.1` reproduces `ssvm_fit/learned_weights.toml` exactly (all 14 weights, e.g. `drift_weight = 0.24322485820272768`). Optional but cheap, and it catches any regression in graph construction |
| G5 | the candidate graph is 2896 nodes / 7981 edges → 16669 ILP variables |
| G6 | each run dir holds exactly one non-empty `fit_weights_ssvm_*.log` |

**Retired (2026-08-05):** G7 (`verify_arm_a_no_oracle.py`) and G8
(`verify_crop_no_oracle.py`) both passed end to end, and the scripts were deleted.
They are the only checks that the arms stay oracle-free, so restore and re-run them
before trusting any future edit to `assign_gt_labels`, `gt_subsets.py`,
`gt_crops.py`, `crop_membership`, or the driver's arm handling — restore commands in
ground rule 2.

---

## Known gotchas

- **Label spaces.** Three incompatible numberings: geff node id (what
  `correct_seg.zarr` is labelled with), raw geff `track_id`, and the
  funtracks-renumbered `track_id` (what `gt_seg` carries and what sampling uses).
  `import_from_geff` renumbering **changes the count** — 164 raw → 167 funtracks. Never
  hard-code the full size; `include_full = true` derives it. Cross-check only by node id.
- **`ssvm_reg` is not the regularizer you think.** `HammingCosts.set_scaling_factor`
  divides the data term by the number of labeled variables while `BundleMethod` adds
  `½·ssvm_reg·‖w‖²` outside that scaling, so the effective regularizer is
  `ssvm_reg × n_labeled`. This is the whole reason for stage 1.
- **`run_tracking.py` writes to `<exp_uid>/`**, not the old hardcoded `test_run/`.
  Configs with no `exp_uid` get a timestamp directory.
- **Empty solutions** break `geff.read`/traccuracy, so no prediction is written. This is
  handled everywhere; treat as data.
- Do not evaluate a subset fit against the complete train GT, and do not re-derive the
  regularizer from the GT-amount curve. Both were tried earlier and are superseded —
  the first is an oracle, the second is circular.

---

## Prior results (superseded — do not mix)

> **Deleted from disk 2026-08-05.** Three sweeps from an earlier protocol used to sit
> beside the stage-2 runs — `gta_*` (fixed `ssvm_reg = 0.1`), `gtar_*` (effective reg
> normalized to the full-GT value) and `gtae_*` (effective reg pinned at 261.42,
> derived from the GT-amount curve rather than a calibration sweep) — 474 run
> directories and 357 configs in all, plus their merged CSVs and figures. All three
> predate the masking fix below, nothing in stage 2 reads them, and re-deriving the
> regularizer from the GT-amount curve is circular. **The numbers below and in the
> `project_gt_amount_experiment` memory are now the only record of them**; the runs
> themselves are gone and are not reproducible without reverting the masking rule.

They were parameterized as: `gta_*` fixed `ssvm_reg = 0.1`; `gtar_*` effective reg
normalized to the full-GT value; `gtae_*` effective reg pinned at 261.42.

All three predate a masking fix (2026-08-03). They used
`masked = hits_removed & ~hits_kept`, which read the hidden tracks' extent and left the
74 candidates overlapping no GT at all labeled `0` — supervision a sparse annotator
could not supply. The rule is now `masked = ~hits_kept`. Effect per subset: +74 masked,
−74 negatives, positives unchanged, full GT unaffected. Labeled-variable counts drop
too (arm A at N=50: 2619 → 2423), so **the old 261.42 target does not carry over**.

Useful only as a rough expectation: at effective reg ≈ 261, test TE was ~0.80 and flat
from 50 to 167 tracks, with a knee below 50; arm B collapsed below 100 regardless of
regularization.

---

## Script inventory

| script | role |
|---|---|
| `scripts/04_tracking/sweep_ssvm_reg.py` | stage 1 calibration + `--collect` |
| `scripts/04_tracking/run_gt_amount_experiment.py` | stage 2 driver (`--self-test`, `--dry-run`, `--materialize-only`, `--only`) |
| `scripts/04_tracking/make_gt_amount_test_configs.py` | learned weights → test configs |
| `scripts/04_tracking/solve_many_weights.py` | batched test solves on one shared graph |
| `scripts/05_evaluation/run_evals.py` | batched evaluation |
| `scripts/05_evaluation/collect_gt_amount.py` | results → CSV |
| `scripts/05_evaluation/merge_gt_amount_csvs.py` | combine sweeps for one figure |
| `scripts/07_plotting/gt_amount_curve.py` | learning curve, one or many metrics |
| `src/mhat/tracking/gt_subsets.py` | track sampling, graph surgery, masked-GT materialization, run-token parsing |
| `src/mhat/tracking/gt_crops.py` | crop-box sampling, candidate bboxes, cropped-GT materialization |
| `src/mhat/tracking/gt_annotation.py` | `compute_gt_overlaps` (cacheable) + `assign_gt_labels` |
