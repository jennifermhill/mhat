# Solver Experiments (Addition & Ablation)

Branch-specific instructions for the solver **addition** and **ablation**
experiments on the `seg_ablation` branch. These isolate the contribution of each
ILP cost term by setting its weight and constant in
`scripts/04_tracking/tracking_config.toml`.

> **Relationship to merge ablation**: the *merge* ablation (see `ABLATION.md`)
> changes the **segmentation** to remove cohesion/adhesion, affinity scoring, or
> merge hyperedges. The *solver* experiments here change only the **ILP cost
> terms** and hold the segmentation fixed at the dataset's baseline `seg_uid`.

> **Branch note**: the `base_edge_constant` knob (used for the `- All` / `None`
> conditions) lives on `seg_ablation`. Running these configs from `main` may behave
> differently.

> **Implementation note (2026-06-25)**: ablation is now done purely through cost
> weights/constants — a cost is removed by setting **both its weight and constant
> to 0** (`add_costs` skips any cost where both are 0). The earlier `ablate_*`
> flags have been **removed**: they were over-engineered and caused a consistency
> bug where a flagged-off cost still instantiated zero-weight ILP structure
> (notably the curvature edge-pair cost), shifting the solver's tie-breaking so
> that `None` and `- All` — which should be identical — differed by ~0.002. Under
> the weight/constant rule a zeroed cost adds nothing at all, and `None` == `- All`
> exactly. Both use the same `base_edge_constant`.

---

## What the two experiments are

Both hold segmentation fixed and vary which ILP cost terms are active via each
cost's **weight and constant** in `scripts/04_tracking/tracking_config.toml`.

**Ablation rule:** a cost is removed by setting **both its weight and constant to
0** — `add_costs` (in `solve_with_motile.py`) skips any cost whose weight and
constant are both 0, so it adds no ILP variables/constraints and cannot affect the
solution. Appear/disappear costs are always on.

**`- All` / `None`:** with every feature cost zeroed there is no edge incentive, so
the ILP would select nothing. Both conditions instead use `base_edge_constant` — a
constant-only per-edge selection cost (weight 0, added whenever nonzero) — tuned
negative to offset appear/disappear and give a non-empty solution. `None`
(addition) and `- All` (ablation) are the **same point** and use the **same**
`base_edge_constant`.

- **Ablation** — start from the **full tuned model** and zero exactly one cost per
  condition, plus an "− All" condition (all feature costs zeroed, `base_edge_constant`
  only). Measures the marginal *loss* from removing each term.
- **Addition** — start from the **empty model** ("None", `base_edge_constant` only)
  and activate exactly one cost per condition (tuning its weight/constant), up to
  "Full" (the full tuned model). Measures the marginal *gain* from adding each term.

### Condition sets (curated per dataset)

Only the terms that matter for a dataset are swept. Current sets:

| Dataset | Ablation conditions | Addition conditions |
|---|---|---|
| `mda231` (01_cells) | Baseline, − Intensity, − Volume, − Drift, − Coh/Adh, − All | None, + Coh/Adh, + Intensity, + Volume, + Curvature, + Drift, Full |
| `mda231_02cells` | (same as 01_cells) | (same as 01_cells) |
| `nc281` (03_nuclei) | Baseline, − Cohesion, − Drift, − All | None, + Cohesion, + Volume, + Intensity, + Curvature, + Drift, Full |
| `nc281_sparse` (01_nuclei_denoised) | Baseline, − Cohesion, − Drift, − All | None, + Cohesion, + Volume, + Intensity, + Curvature, + Drift, Full |

Add or remove conditions by editing `condition_order` and the `[<dataset>.conditions.*]`
tables in the relevant TOML — the plotting script renders whatever is listed.

---

## Mapping conditions → parameters

**Ablation** (start from the full tuned model; set the named cost's weight AND
constant to 0):

| Condition | Zeroed cost(s) |
|---|---|
| Baseline | (none — full model) |
| − Intensity | intensity |
| − Volume | area |
| − Drift | drift |
| − Coh/Adh (− Cohesion) | cohesion + adhesion |
| − All | all feature costs (use `base_edge_constant`) |

**Addition** (start from all feature costs zeroed; activate one):

| Condition | Active cost (all others zeroed) |
|---|---|
| None | (none — `base_edge_constant` only) |
| + Coh/Adh | cohesion + adhesion |
| + Volume | area |
| + Intensity | intensity |
| + Curvature | curvature |
| + Drift | drift |
| Full | all (full tuned model) |

Each non-baseline condition is **mini-optimized per dataset** — a coordinate-wise
search over the active cost set (for `- All` / `None`, over `base_edge_constant`),
maximizing TRA then DET (see the optimization strategy in CLAUDE.md). The
Baseline/Full conditions are the dataset's full tuned best. The optimization is
done on 01_cells and the resulting params are transferred to 02_cells; nc281 and
nc281_sparse are optimized separately.

---

## Recipe per condition

All commands run from the repo root. **On `Y:` use the `mhat2` env**
(`conda run -n mhat2 --no-capture-output ...`); the `mhat-sandbox` env is for the
`C:` checkout.

1. Edit `scripts/04_tracking/tracking_config.toml`:
   - Set `seg_result` to the dataset's **baseline `seg_uid`** (held fixed across
     every solver condition — see `merge_ablation.toml` for each dataset's
     baseline seg).
   - Zero the weight AND constant of each ablated cost per the mapping table
     above. For "− All" / "None", zero all feature costs and set a negative
     `base_edge_constant`.
   - For addition "+ X" conditions, set that term's weight/constant to the tuned
     values for this bar; keep all other costs zeroed.
2. Run tracking: `python scripts/04_tracking/run_tracking.py scripts/04_tracking/tracking_config.toml`.
3. Read the `exp_uid` from `experiments/tracking/<experiment>/<dataset>/test_run/config.toml`.
4. Update `scripts/05_evaluation/eval_config.toml` — set `track_result` to that
   `exp_uid`. Confirm the correct `matcher` per dataset (MDA231 → `"ctc"`;
   NC281 → `"point"`, `match_threshold = 10`).
5. Run eval: `python scripts/05_evaluation/evaluate_tracks.py scripts/05_evaluation/eval_config.toml`.
6. Confirm `experiments/evaluation/<experiment>/<dataset>/<exp_uid>/<metrics_file>.json`
   was written.
7. Record the `tracking_uid` in the matching TOML
   (`scripts/05_evaluation/solver_ablation.toml` or `solver_addition.toml`),
   under the dataset and condition.

---

## Running notes (UIDs)

The single source of truth for which `tracking_uid` corresponds to each
(dataset, condition) is the per-experiment TOML:

- `scripts/05_evaluation/solver_ablation.toml`
- `scripts/05_evaluation/solver_addition.toml`

We store `tracking_uid` and a `config` pointer (to the archived static config —
see Config archive below) per condition; the canonical ILP parameters live in
that run's saved `tracking_config.toml`. The `params` / `note` fields are
informational.

### Back-fill status (as of setup)

`Y:` is the same storage as the cluster's `/groups/sgro/sgrolab/...`, so the
sparse-label solver runs (done on the cluster) are reachable here.

- **`nc281_sparse` — fully back-filled and verified.** Both experiments use the
  named solver batch `2026-05-01_10-02-14_*` (`_ablat_no_*`, `_add_*`) for every
  non-Full bar, and `2026-04-29_11-36-23_R1` for Baseline/Full. Loaded values
  reproduce the old `solver_*_figure_nc281-sparse.py` numbers exactly. Note: this
  dataset is **point-matched** — `metrics_filename = "track_metrics.json"` for
  the batch runs, with a per-condition override to `track_metrics_point.json` for
  the Full/Baseline run (same matcher, different filename).
- **`mda231` (01_cells) — fully back-filled and verified.** The original solver
  figure runs lived on the `C:` sandbox (seg `2026-04-03_11-09-49`). All 11
  non-baseline eval dirs (tiny — just `track_metrics.json` + `tracking_config.toml`)
  were copied to `Y:`; Baseline/Full = the documented best `2026-04-15_09-42-38`
  (already on `Y:`). Loaded values reproduce the old
  `solver_*_figure_mda231.py` numbers exactly (all 13 conditions). The runs were
  disambiguated by reading each saved config's `ablate_*` flags — important for
  ablation, where several full-cost runs shared the − Intensity metric values.
  **Caveat on addition:** these runs isolate the active term by **zeroing the
  other costs** (the exact tuned weights are in each run's saved
  `tracking_config.toml`, summarized in the `note` fields) — consistent with the
  current weight/constant gating.
- **`nc281` (03_nuclei) — fully back-filled and verified.** The original solver
  figure runs lived on the `C:` sandbox (seg `2026-02-25_11-56-23`); all conditions
  were copied to `Y:` except `+ Curvature`, which was already on `Y:` as
  `2026-04-29_16-49-06_R3` (the cluster curvature batch `nc281curvadd_...`). Loaded
  values reproduce the old `solver_*_figure_nc281.py` numbers exactly (all 11
  conditions). **Matcher note:** these solver runs store **point-matched** TE/TF in
  `track_metrics.json`, not `track_metrics_iou.json` (which the *merge* figure
  used), so the nc281 sections set `metrics_filename = "track_metrics.json"`.
  Addition used the weight-zeroing method (see the mda231 caveat above).
- **`mda231_02cells` — run fresh (2026-06-23), all conditions except − All.**
  Ran on the 02_cells baseline seg `2026-05-29_10-47-41` + flow
  `2026-05-29_10-59-37`, each condition using ILP params identical to the
  equivalent 01_cells solver condition (no re-optimization; addition conditions
  replicate the weight-zeroing method). Baseline/Full reuse the merge baseline
  run `2026-06-01_14-41-03`; − Coh/Adh was regenerated (`2026-06-23_18-26-30`)
  so its `z_flow` matches the 01_cells condition (the reused merge no_cohesion
  had `z_flow` absent). **− All** now uses the `base_edge_constant` mechanism
  (all feature costs zeroed + a tuned negative `base_edge_constant`), set from the
  01_cells mini-optimization.

> **Superseded (2026-06-25):** the per-condition values above were the original
> back-fill. **mda231 (01_cells and 02_cells) has since been re-optimized** under
> the new weight/constant gating (see Implementation note) — 01_cells via a
> coordinate-wise mini-optimization, 02_cells by transferring the 01_cells params
> — and the TOMLs hold the re-optimized `tracking_uid`s. The merge `baseline`
> (01_cells) and `- Coh/Adh` (02_cells) runs were **consolidated** with their
> solver counterparts (one shared run + config). **nc281 / nc281_sparse still hold
> the original back-fill** and will be re-optimized separately; until then their
> merge and solver `baseline`/`no_cohesion` runs differ and are kept apart.

---

## Config archive

Every run referenced by `merge_ablation.toml`, `solver_ablation.toml`, and
`solver_addition.toml` has a static, human-readable copy of its config in a
**git-ignored `configs/`** tree, so any condition can be re-run on future code
even if the original run dir is pruned. Each TOML condition also carries a
`config = "configs/tracking/…"` pointer (the `tracking_uid` still names the real
run with its data under `experiments/`).

Layout mirrors `experiments/`:

```
configs/
├── tracking/<experiment>/<dataset>/<condition>_config.toml
├── segmentation/<experiment>/<dataset>/<role>_config.toml      # role: baseline|no_affinities|no_merges
└── opticalflow/<experiment>/<dataset>/opticalflow_{2d,3d}/flow_config.toml
```

- Tracking configs are named by **condition** (e.g. `no_intensity_config.toml`).
  Conditions that resolve to the **same run** share one file (e.g. merge
  `baseline`, solver `baseline`, and addition `full` all point to
  `baseline_config.toml`). Where merge and solver genuinely differ (the deferred
  nc281 / nc281_sparse `baseline` / `no_cohesion`), the file is suffixed
  `_merge` / `_solver`.
- Each tracking config records `seg_result` / `flow_result`, which resolve to the
  archived seg / flow configs by role.

**Rebuild** after changing the TOMLs (e.g. new re-optimized UIDs):

```
conda run -n mhat2 --no-capture-output \
  python scripts/05_evaluation/build_config_archive.py
```

`build_config_archive.py` parses the three TOMLs, dedups by `tracking_uid`,
applies the naming rule above, and copies each run's saved config verbatim
(asserting byte-identical). The per-condition `config` pointers in the TOMLs use
deterministic paths, so a rebuild keeps them valid without re-inserting; pass
`--print-map` to dump the condition→config mapping.

> **Caveat:** archived configs are verbatim copies — some (the C:-sandbox
> back-fills) carry `C:` base-dir paths, and all keep their `exp_uid`. To re-run,
> point the base dirs at the current checkout and drop `exp_uid` (a fresh one is
> generated per run).

---

## Plotting

One generic script drives both experiments (the TOML schema is identical):

```
conda run -n mhat2 --no-capture-output \
  python scripts/05_evaluation/solver_figure.py \
  scripts/05_evaluation/solver_ablation.toml \
  --dataset {mda231,mda231_02cells,nc281,nc281_sparse}
```

Swap in `solver_addition.toml` for the addition plots. Per dataset the script
reads `condition_order`, each condition's `label`/`color`/`tracking_uid`, the
`metrics` specs, `metrics_filename`, `reference` (dashed line at `first`=Baseline
or `last`=Full), and `xtick_rotation`. Output path comes from the TOML's
`output_png`; override with `--output PATH`. Conditions whose `tracking_uid` is
still `<fill in>` produce a warning and are skipped, so partial-progress plotting
works.

Colors follow the Wong colorblind-friendly palette, consistent per concept
across all figures: baseline/full `#0072B2`, intensity `#56B4E9`, volume
`#E69F00`, drift `#009E73`, coh/adh `#CC79A7`, curvature `#D55E00`, all/none
`#999999`.

> The old hardcoded `solver_ablation_figure_*.py` / `solver_addition_figure_*.py`
> scripts are superseded by the TOML-driven `solver_figure.py` and can be removed
> once their numbers are reproduced through the new pipeline.
