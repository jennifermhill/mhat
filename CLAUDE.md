# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MHAT (Multi-Hypothesis Affinity Tracking) is a Python framework for multi-hypothesis segmentation and tracking in microscopy data. It uses zarr-based data storage, networkx graphs for track representation, and integer linear programming (via motile/gurobi) for solving tracking problems.

## Build & Install

```bash
# Install in development mode
pip install -e ".[dev]"

# Conda environments available for platform-specific setup
conda env create -f environment_linux.yml   # Linux
conda env create -f environment_windows.yml # Windows
```

## Linting & Formatting

Pre-commit hooks run ruff (linter + formatter). Line length is 88, target Python 3.8+.

```bash
pre-commit run --all-files    # Run all hooks
ruff check --fix .            # Lint with auto-fix
ruff format .                 # Format
```

## Architecture

### Pipeline Stages (scripts/)

The workflow follows numbered stages, each with a TOML config file:

1. **01_training** — Train MTLSD segmentation model (gunpowder pipeline)
2. **02_segmentation** — Generate segmentation hypotheses (cellpose, affinities, agglomeration)
3. **03_opticalflow** — Compute optical flow between frames (Farneback/Lucas-Kanade)
4. **04_tracking** — Build multi-hypothesis graph and solve ILP (motile solver)
5. **05_evaluation** — Evaluate tracking/segmentation metrics (traccuracy, IoU)
6. **06_visualization** — View results in napari/neuroglancer

### Core Modules (src/mhat/)

- **data.py** — `DataZarr`, `RawDataZarr`, `SegmentationZarr` classes wrapping zarr storage. Data is organized as `fov=N/channel=name` groups within zarr containers.
- **segmentation/** — Fragment generation, affinity computation, agglomeration, cellpose integration, threshold-based methods
- **opticalflow/** — 2D Farneback and 3D Lucas-Kanade optical flow with contrast enhancement (AHE)
- **tracking/** — `create_multihypo_graph.py` builds candidate graphs with costs; `solve_with_motile.py` solves via ILP; `tracks_io.py` handles CSV/graph conversion
- **evaluation/** — Tracking metrics via traccuracy; segmentation metrics via IoU
- **visualization/** — Napari viewer utilities and widget setup

### Local Dependencies

Some dependencies are installed from local checkouts in the repo root:

- `./funtracks` — funtracks
- `./motile_tracker` — motile-tracker

### Key Dependencies

- **motile** — ILP-based network flow solver for tracking
- **gurobi/ilpy/pyscipopt** — ILP optimizer backends
- **zarr** — Chunked array storage format
- **gunpowder** — Training data pipeline
- **traccuracy** — Tracking evaluation metrics
- **cellpose** — Deep learning cell segmentation

## Tracking Optimization Pipeline

Iterative workflow: modify ILP cost parameters, run tracking, run evaluation, read metrics, repeat.

All pipeline commands must be run in the `mhat-sandbox` conda environment using `conda run -n mhat-sandbox --no-capture-output`.

### Pipeline Steps

1. **Modify ILP cost parameters** in `scripts/04_tracking/tracking_config.toml`
   - Tunable parameters (weights and constants only):
     - `drift_weight`, `drift_constant`
     - `area_weight`, `area_constant`
     - `intensity_weight`, `intensity_constant`
     - `curvature_weight`, `curvature_constant`
     - `cohesion_weight`, `cohesion_constant`
     - `adhesion_weight`, `adhesion_constant`
     - `appear_constant`, `disappear_constant`
   - Do NOT change: `drift_distance`, `max_children`, `merges`, `divisions`, `min_merge_cost`, `max_merge_cost`, `max_edge_distance`, `size_threshold`, or any path/IO parameters

2. **Run tracking**: `conda run -n mhat-sandbox --no-capture-output python scripts/04_tracking/run_tracking.py scripts/04_tracking/tracking_config.toml`
   - Tracking results overwrite in place (no new directories per run)
   - Get the `exp_uid` from the saved `config.toml` inside the tracking result directory (e.g., `experiments/tracking/<experiment>/<dataset>/test_run/config.toml`, field `exp_uid`)

3. **Update eval config**: Edit `scripts/05_evaluation/eval_config.toml` and set `track_result` to the `exp_uid` from the tracking config

4. **Run evaluation**: `conda run -n mhat-sandbox --no-capture-output python scripts/05_evaluation/evaluate_tracks.py scripts/05_evaluation/eval_config.toml`

5. **Read metrics**: Read `experiments/evaluation/<experiment>/<dataset>/<exp_uid>/track_metrics.json`
   - Metrics: `TRA`, `DET`, `LNK` (all under `CTCMetrics`)
   - **Primary goal**: maximize TRA and DET first — correct detections should also improve tracking
   - LNK is secondary; improvements will follow from better detection
   - Use fp_nodes and fn_nodes as diagnostic signals to understand what's going wrong, not as direct optimization targets

6. **Analyze and iterate**: Based on metrics, decide which parameters to adjust next and repeat from step 1

### Baseline Metrics (run 2026-03-10_11-10-29)

- TRA: 0.759, DET: 0.768, LNK: 0.694

### Key Files

- `scripts/04_tracking/tracking_config.toml` — tracking parameters (modify this)
- `scripts/04_tracking/run_tracking.py` — tracking script
- `scripts/05_evaluation/eval_config.toml` — evaluation config (update `track_result` each run)
- `scripts/05_evaluation/evaluate_tracks.py` — evaluation script

### Optimization Log Files

- `tracking_optimization_log.md` — **Full log** (append-only). Every run with hypothesis, config, metrics, and verdict. Read this for detailed history.
- `tracking_optimization_memory.md` — **Working memory** (read + update each batch). Contains: results comparison table, established principles, falsified hypotheses, open questions, current best config. Read at start of each optimization session, update at end of each batch.

### Optimization Strategy

**Batch size**: Run 5 iterations per batch, then report all results and propose next steps.

**Parameter semantics**: The ILP **minimizes total cost**. For each selected node/edge: `cost = weight × attribute + constant`. Positive values discourage selection; negative values encourage it.

| Parameter | Type | Attribute | What it measures |
|-----------|------|-----------|-----------------|
| drift_weight/constant | Edge | drift_dist | Spatial distance between linked detections |
| area_weight/constant | Edge | area_diff | Normalized area change between frames |
| curvature_weight/constant | EdgePair | centroid positions | Trajectory smoothness (deviation from straight-line) |
| cohesion_weight/constant | Node | cohesion | Merge quality (0=fragment, higher=confident merge) |
| adhesion_weight/constant | Node | adhesion | Standalone confidence (1-next_merge_cost) |
| appear_constant | Node | — | Cost per new track start |
| disappear_constant | Node | — | Cost per track end |

**Using graph attribute statistics to set parameter ranges**:

`run_tracking.py` prints a statistics table before solving that shows mean, std, weight, constant, cost mean, and cost std for every attribute. Use this to set weights in the correct regime:

- **A parameter only influences the ILP if its cost std is ~900+.** Below that, it's invisible relative to dominant costs (cohesion, drift). This was confirmed across 50+ runs on Fluo-C3DL-MDA231.
- **To estimate a target weight**: `target_weight = target_cost_std / attribute_std`. For example, if `area_diff` has std=0.455 and you want cost std≈900, set `area_weight ≈ 900 / 0.455 ≈ 2000`.
- **Constants must keep cost mean negative** (encouraging) for edge/node selection to work. If `weight × attribute_mean + constant > 0` on average, many edges get positive (discouraging) costs and linking suffers.
- **First step for any new dataset**: Run tracking once, read the statistics table, and set all active weights so their cost stds are in the 400-2000 range before starting optimization.
- **Attributes with low relative spread** (std/mean < 0.3) will have limited discriminating power regardless of weight. Check whether an attribute is worth tuning before spending runs on it.

**Approach — coordinate-wise search with informed adjustments**:
1. Change 1-2 parameters at a time to isolate effects
2. Use metric changes to diagnose which parameter to adjust next:
   - **High fp_nodes** (false positive detections) → may be selecting too-small fragments. Try increasing cohesion_weight (more negative = prefer merged fragments) or adjusting other node costs. But this isn't the only cause — also consider appear/disappear constants and edge costs.
   - **High fn_nodes** (false negative detections) → may be selecting too-merged fragments. Try decreasing cohesion_weight magnitude or increasing adhesion. But also consider other causes.
   - **Low DET** (detection) → driven by fp_nodes + fn_nodes. Adjust node costs: cohesion, adhesion, appear/disappear constants
   - **Low LNK** (linking) → adjust edge costs: drift, area, curvature weights/constants
   - **Low TRA** (overall) → usually follows from DET and LNK improvements
3. When adjusting a parameter, try ~2× or 0.5× the current value as a starting perturbation
4. If a change improves one metric but worsens another, try a smaller step size
5. Track the best-so-far parameter set and always compare against it

**Parameter interaction notes**:
- drift_constant and curvature_constant are both large negative values (-1000) that strongly encourage edge selection — changing one may require adjusting the other
- cohesion_weight (-3000) dominates node selection — small changes to adhesion_weight (-10) may have little effect until cohesion is also adjusted
- appear/disappear constants control track fragmentation — increase to get fewer, longer tracks; decrease to allow more track breaks

**Current baseline** (from latest CTCMatcher run):
- TRA: 0.741, DET: 0.755, LNK: 0.637, fp_nodes: 118, fn_nodes: 70

**Best configs found so far** (from Batch 1-4 optimization, 2026-03-19):
- Best TRA/DET: drift_c=-2000, drift_w=50, cohesion_w=-5000 → TRA: 0.750, DET: 0.762, LNK: 0.663, fp: 117, fn: 68
- Best LNK: drift_c=-2000, drift_w=50 → TRA: 0.748, DET: 0.757, LNK: 0.684, fp: 125, fn: 71

**Metrics to track**: TRA, DET (primary optimization targets); LNK (secondary, expected to improve with better detections); fp_nodes, fn_nodes (diagnostic signals only — use to guide parameter adjustments, not as optimization targets)

**Iteration workflow**:
1. Read `tracking_optimization_memory.md` for context (established principles, current best, open questions)
2. Formulate hypothesis for each run — what you expect and why
3. Run tracking + evaluation
4. Analyze: was the hypothesis supported, falsified, or inconclusive?
5. After each batch: append runs to `tracking_optimization_log.md`, update `tracking_optimization_memory.md` (comparison table, principles, hypotheses)

**Log entry format** (append to `tracking_optimization_log.md`):
```
### B{batch}R{run}: [param change description]
exp_uid: <from experiments/tracking/<experiment>/<dataset>/test_run/config.toml>
Hypothesis: "[specific testable prediction]"
TRA: X, DET: X, LNK: X, fp: X, fn: X, fn_edges: X
Verdict: [supported/falsified/inconclusive] — [one line explanation]
```

**Reporting format** after each batch:
```
| Run | Changed Params | TRA | DET | LNK | fp_nodes | fn_nodes | Notes |
```

## NC281-Fl2mSiH2B Tracking Optimization Pipeline

Iterative optimization for the NC281-Fl2mSiH2B/03_nuclei dataset. Same pipeline steps as above (modify config → run tracking → run evaluation → read metrics → iterate), but with **different evaluation metrics and optimization targets** due to sparse ground truth.

### Evaluation Setup

- **Metrics**: `basic` (BasicMetrics) + `track_overlap` (TrackOverlapMetrics)
- **Matcher**: `point` (PointMatcher) — Hungarian matching on node positions with distance threshold
- **Eval config**: `scripts/05_evaluation/eval_config.toml` with `experiment = "NC281-Fl2mSiH2B"`, `dataset = "03_nuclei"`, `metrics = ["basic", "track_overlap"]`, `matcher = "point"`

### Metrics Reference

**BasicMetrics** output fields:
- `Total GT Nodes`, `Total Pred Nodes`, `True Positive Nodes`, `False Positive Nodes`, `False Negative Nodes`
- `Node Recall`, `Node Precision`, `Node F1`
- `Total GT Edges`, `Total Pred Edges`, `True Positive Edges`, `False Positive Edges`, `False Negative Edges`
- `Edge Recall`, `Edge Precision`, `Edge F1`

**TrackOverlapMetrics** output fields:
- `target_effectiveness` — weighted average: for each GT track, find the predicted track with maximum overlap, divide overlap by GT track length, average weighted by track length. Measures how well GT tracks are covered.
- `track_purity` — analogous with GT/pred swapped. Measures how "pure" each predicted track is.
- `track_fractions` — per-GT-track overlap fractions (unweighted)

### Optimization Targets

**Ground truth is sparse** — not all objects are annotated. This means:
- **Precision, F1, and Track Purity are unreliable** — false positives cannot be properly counted since unannotated objects appear as FP
- **Do NOT optimize**: Node F1, Edge F1, Node Precision, Edge Precision, Track Purity

**Primary targets** (maximize these):
- `target_effectiveness` — fraction of GT tracks covered by best-matching predicted tracks
- `track_fractions` — per-track overlap scores (report mean)

**Diagnostic heuristics** (use to guide parameter adjustments):
- `Node Recall` — fraction of GT nodes matched to a prediction. Higher = better detection coverage
- `Edge Recall` — fraction of GT edges matched. Higher = better linking of GT-annotated objects

### Baseline Metrics (2026-03-25)

- Target Effectiveness: 0.672, Track Fractions (mean): 0.719
- Node Recall: 1.000, Edge Recall: 0.835, FN Edges: 76

### Optimization Log Files

- `nc281_optimization_log.md` — **Full log** (append-only). Same format as Fluo-C3DL-MDA231 logs.
- `nc281_optimization_memory.md` — **Working memory** (read + update each batch).

### Optimization Strategy

Same general approach as Fluo-C3DL-MDA231 (coordinate-wise search, 5 runs per batch, hypothesis-driven), with these differences:

**First step**: Run tracking once and read the `report_graph_statistics()` output to understand attribute distributions for this dataset. Use statistics to set initial weights in the correct regime (cost std ~900+ for each parameter to have effect).

**Diagnostic guidance for sparse GT**:
- **Low target_effectiveness** → GT tracks are being fragmented or missed. Consider:
  - Lowering appear/disappear constants (allow fewer track breaks)
  - Strengthening edge encouragement (more negative drift/curvature constants)
  - Adjusting cohesion to select better node hypotheses
- **Low Node Recall** → GT nodes not being detected. Adjust node selection costs (cohesion, adhesion, appear/disappear)
- **Low Edge Recall** → GT edges not being linked. Adjust edge costs (drift, area, intensity, curvature)
- **track_fractions has high variance** → some GT tracks well-covered, others not. Look at which tracks are failing (long vs short, specific timepoints)

**Log entry format** (append to `nc281_optimization_log.md`):
```
### B{batch}R{run}: [param change description]
exp_uid: <from experiments/tracking/<experiment>/<dataset>/test_run/config.toml>
Hypothesis: "[specific testable prediction]"
TE: X, TF_mean: X, Node_Recall: X, Edge_Recall: X
Verdict: [supported/falsified/inconclusive] — [one line explanation]
```

**Reporting format** after each batch:
```
| Run | Changed Params | TE | TF_mean | Node_Recall | Edge_Recall | Notes |
```

## NC281-Sparse-Label Tracking Optimization Pipeline

Iterative optimization for the NC281-sparse-label/01_nuclei_denoised dataset. Same pipeline shape as the other two datasets, but **runs on the Janelia cluster** (LSF, mhat2 env, gurobi module) with parallel batches rather than sequential local runs.

### Cluster Pipeline

Tracking and evaluation are submitted as parallel LSF jobs:

- `scripts/04_tracking/launch_batch.py` — orchestrator that generates unique `exp_uid`s and submits parallel tracking jobs. Run-spec list is hardcoded inside `main()` and edited per batch.
- `scripts/04_tracking/launch_curvature_batch.py` — variant for curvature-weight sweeps (takes paired `--curvature-weights` and `--curvature-constants`).
- `scripts/04_tracking/submit_evals.py` — submits dependent eval jobs with LSF `ended()` deps so they fire even if tracking exits with code 120 (a known post-completion exit-code anomaly; outputs are still valid).

Inner command for tracking jobs:
```bash
module load gurobi && conda run -n mhat2 --no-capture-output python -u <run_tracking.py> <config>
```

Per-batch artifacts (configs + manifest) land in `experiments/tracking/NC281-sparse-label/01_nuclei_denoised/batches/<batch_id>/`. LSF logs land in the sibling `logs/` directory; eval logs in `experiments/evaluation/.../logs/`.

### Evaluation Setup

- **Metrics**: `basic` (BasicMetrics) + `track_overlap` (TrackOverlapMetrics)
- **Matcher**: `point` (PointMatcher) — Hungarian matching on node positions with distance threshold
- **match_threshold**: 10

### Metrics Reference

Same fields as NC281-Fl2mSiH2B (BasicMetrics + TrackOverlapMetrics).

### Optimization Targets

The NC281-sparse-label dataset name is misleading — the **GT is densely annotated** ("sparse-label" refers to the raw-data labeling sparsity, not GT sparsity). Because GT is dense, **precision and purity are reliable metrics here** (unlike NC281-Fl2mSiH2B where sparse GT makes them unreliable). Optimize all three of recall, coverage, and purity.

**Primary targets**:
- `target_effectiveness` (TE)
- `track_fractions` mean (TF)
- `track_purity` — promoted to primary here; a config that inflates NodeR by adding FPs is *not* an improvement

**Diagnostic signals**:
- `Node Recall`, `Edge Recall` — detection / linking coverage of GT
- `False Positive Nodes`, `False Negative Edges` — counterbalancing signals to recall

### Baseline Metrics (2026-04-28)

- TE: 0.769, TF: 0.779, Node Recall: 0.932, Edge Recall: 0.880

### Optimization Log Files

- `nc281_sparse_optimization_log.md` — full log (append-only)
- `nc281_sparse_optimization_memory.md` — working memory (read + update each batch)

### Optimization Strategy

Same general approach as the other two datasets (coordinate-wise search, 5 runs per batch, hypothesis-driven), with these dataset-specific notes:

**Cluster ergonomics**:
- Tracking runs occasionally exit 120 *after* writing valid outputs (Python shutdown phase quirk). `submit_evals.py` uses `ended()` dependencies so evals fire regardless. If output dirs have complete `pred_tracks.zarr/{nodes,edges,.zattrs}` and `pred_seg.zarr`, the run succeeded despite the exit code.
- Use `python -u` in the bsub inner command (already in the launchers) for unbuffered stdout — best practice on long runs.
- Walltime: 24h tracking, 1h eval is sufficient for current operating point. R2 (newer seg) and R3 (cohesion variants) had previous runs at 14h+ — consider queue default for those.

**Diagnostic guidance**:
- **Low TE/TF** → adjust drift / cohesion. drift_weight=100 (from 25) was the cleanest single-param win at baseline.
- **Low Node Recall** → consider `size_threshold=0` (recovers small-object FNs) or richer seg, but watch purity.
- **Low Edge Recall** → drift weight, edge costs.
- **Inactive parameters at current operating point**: cohesion_constant, appear_constant, disappear_constant. Confirmed in B1R3, B1R4.

**Log entry format** (append to `nc281_sparse_optimization_log.md`):
```
### B{batch}R{run}: [param change description]
exp_uid: <from manifest.toml or config.toml>
Hypothesis: "[specific testable prediction]"
TE: X, TF_mean: X, Node_Recall: X, Edge_Recall: X, Purity: X
Verdict: [supported/falsified/inconclusive] — [one line explanation]
```

**Reporting format** after each batch:
```
| Run | Changed Params | NodeR | EdgeR | TE | TF | Purity | Notes |
```

## Fluo-C3DH-H157 Tracking Optimization Pipeline

Iterative optimization for Fluo-C3DH-H157/01_cells. Same pipeline shape as the other datasets, but with a key structural difference: **runs in `skip_merge_hypotheses=true` mode**, which disables multi-hypothesis selection and exclusion sets. As a consequence, **ILP cost tuning has essentially no effect on this dataset** — the only lever that moves metrics is `size_threshold`.

### Run Mode and Why ILP Costs Don't Matter

- `skip_merge_hypotheses = true` means the pipeline pre-merges fragments to the maximum level and feeds a single set of node hypotheses to the ILP (no alternatives, no exclusion sets).
- Without exclusion sets, the ILP has no node-selection trade-off — every node is selected, and every feasible edge is chosen.
- This makes drift/area/intensity/curvature/cohesion/adhesion/appear/disappear costs all inert: B1R1-B1R4 confirmed identical metrics across intensity sign flips, appear/disappear up to 5000, curvature on/off, and stronger drift_c.
- The only effective lever is `size_threshold` — it filters FP fragments **before** graph construction, so it changes the candidate node set itself rather than the ILP's selection from a fixed set.

### Eval Setup

- **Metrics**: `ctc` (CTCMetrics)
- **Matcher**: `ctc` (CTCMatcher) — requires >50% IoU between predicted and GT segments to count as a match
- **No `match_threshold`** for CTC matcher (it takes no kwargs)

### Optimization Targets

- **Primary**: TRA, DET (same priority as MDA231 — correct detections drive tracking)
- **Secondary**: LNK (will follow from better detection)
- **Diagnostics**: `fp_nodes`, `fn_nodes`, `fn_edges` — useful for diagnosing CTC matcher mismatches

### Baseline Metrics (B0R0, 2026-04-24)

- TRA: 0.860, DET: 0.855, LNK: 0.893 (size_threshold=500, MDA231 ILP params transferred uncalibrated)

### Optimization Log Files

- `h157_optimization_log.md` — full log (append-only)
- `h157_optimization_memory.md` — working memory (read + update each batch)

### Optimization Strategy

**This dataset is unusual** — coordinate-wise ILP cost tuning won't move metrics. Strategy is fundamentally different:

1. **Sweep size_threshold first** — it's the only lever. Real cells in H157 are 100k-495k voxels; FP fragments are <20k. There's a clear size gap.
2. **Skip ILP cost tuning entirely** in skip_merge_hypotheses mode. Only revisit if `skip_merge_hypotheses=false` is re-enabled (which would restore exclusion sets and make ILP costs effective again).
3. **Diagnose with CTC matcher in mind** — fn_nodes/fn_edges may be CTC IoU mismatches (predicted segment doesn't overlap its GT cell ≥50%) rather than tracking errors. These cannot be fixed by tuning.

### Optimization Setup Notes

- Timeline truncated to 15 timepoints for optimization speed (full sequence is 60 frames).
- Segmentation: `2026-04-21_13-35-28` (cellpose, merge_thresholds=[1.0])
- Optical flow: `2026-04-24_14-36-14` (2D Farneback + 3D Lucas-Kanade)
- GT format: CTC (`01_GT` for the truncated 15-frame eval; `01_GT_full` saved for the full sequence).

### Current Best (2026-04-24)

- **B2R3 (size_threshold=50000)**: TRA=0.892, DET=0.892, LNK=0.893, fp=5, fn=6, fn_edges=6, 59 nodes selected.
- **Plateau reached** — remaining errors are segmentation-level CTC matcher mismatches (predicted segments don't overlap GT segments ≥50%), not fixable via ILP tuning.

### Open Questions

1. Could re-enabling merge hypotheses (`skip_merge_hypotheses=false`) recover the 6 FN nodes via under-merged fragments? Would also restore ILP cost tuning as a real lever.
2. Would a different (more fragmented) segmentation help bridge the IoU gap on the 6 mismatched nodes?
3. Need to run full 60 timepoints with `size_threshold=50000` to confirm metrics generalize beyond the 15-frame optimization window.

### Log Entry Format

```
### B{batch}R{run}: [param change description]
exp_uid: <from manifest.toml or config.toml>
Hypothesis: "[specific testable prediction]"
TRA: X, DET: X, LNK: X, fp: X, fn: X, fn_edges: X
Verdict: [supported/falsified/inconclusive] — [one line explanation]
```

### Reporting Format

```
| Run | Changed Params | TRA | DET | LNK | fp | fn | fn_edges | Nodes |
```

## SSVM Weight Fitting (`ssvm-fit` branch only)

Experimental: use motile's `Solver.fit_weights()` (structsvm bundle method) to learn ILP cost weights from CTC ground truth, as an alternative to coordinate-wise hand-tuning. Un-tuned result on MDA231 (no post-hoc adjustments): TRA=0.797 vs hand-tuned 0.881. SSVM does not replace hand-tuning, but it gives a usable starting point with no manual coordinate search.

### Files

- `src/mhat/tracking/gt_annotation.py` — annotates `gt_selected ∈ {0, 1}` on candidate nodes/edges via Hungarian matching (IoGT > 0.5 filter, IoU as ranking tiebreaker).
- `src/mhat/tracking/pipeline.py` — `build_track_graph()` shared between `run_tracking.py` and the SSVM scripts.
- `src/mhat/tracking/leaves_scaled_costs.py` — `LeavesScaledNodeSelection` (bakes `num_leaves` into features so SSVM and inference see the same costs).
- `src/mhat/tracking/solve_with_motile.py` — `add_costs(solver, config)` shared cost-adding helper, gated by `ablate_*` flags (not weight==0).
- `scripts/04_tracking/fit_weights_ssvm.py` — runs the fit using stock `motile.Solver.fit_weights()`, writes `learned_weights.toml`, runs a final solve. Also contains unused `fit_weights_standardized()` and `TolerantBundleMethod` helpers left over from pre-ilpy-fix workarounds (see "History" below); safe to delete if no longer wanted.
- `scripts/04_tracking/inspect_gt_annotation.py` — napari overlay for visual sanity check of the GT→candidate matching.
- `scripts/04_tracking/MDA231_ssvm_fit.toml` — fit config; new keys vs `MDA231_baseline.toml`: `ssvm_reg`, `ssvm_max_iter`, `ssvm_eps`, `iogt_threshold`.

### Workflow

1. **Inspect first** (interactive, optional but recommended once per dataset):
   ```
   conda run -n mhat-sandbox --no-capture-output python scripts/04_tracking/inspect_gt_annotation.py scripts/04_tracking/MDA231_ssvm_fit.toml
   ```
   Verify `matched_cand` layer aligns with `gt_seg` in napari.

2. **Fit**:
   ```
   conda run -n mhat-sandbox --no-capture-output python scripts/04_tracking/fit_weights_ssvm.py scripts/04_tracking/MDA231_ssvm_fit.toml
   ```
   Outputs `learned_weights.toml` and `pred_tracks.zarr` under `experiments/tracking/<experiment>/<dataset>/ssvm_fit/`. Logging goes to a timestamped logfile in the same directory (UTF-8 encoding so structsvm's `ε` character writes correctly on Windows cp1252). Per-iteration ε convergence is in the logfile; gurobi solver output prints to stdout directly (C-level write — not captured by Python logging without fd-level redirection).

That's the whole workflow. No post-hoc offset sweep, no standardization step — stock `solver.fit_weights()` converges (ε → 0 from above) and the final solve produces a non-empty inference solution directly. To evaluate, point `scripts/05_evaluation/eval_config.toml`'s `track_result` at `ssvm_fit` and run `evaluate_tracks.py`.

### Key parameters

- `ssvm_reg` (default 0.1) — regularization strength on `½λ‖w‖²`. Acts as a 1/k scale on learned weight magnitudes; does not change their direction.
- `ssvm_max_iter` (default 100) — bundle method on MDA231 converges in ~30 iterations.
- `iogt_threshold` (default 0.5) — minimum intersection-over-GT for a candidate to be a valid match (CTC detection criterion). Use IoGT (not IoU) because GT segs on this dataset are often smaller than the actual cell extent; IoU underestimates match quality for over-merged candidates that fully contain the GT.

### Un-tuned result on MDA231

```
ssvm_reg = 0.1 (default)
no post-hoc offset, no standardization
→ TRA=0.797, DET=0.841, LNK=0.478, fp=149, fn=37, fn_edges=172
```

Hand-tuned best for comparison: `TRA=0.881, DET=0.886, LNK=0.845` (config: `scripts/04_tracking/MDA231_baseline.toml`). DET is comparable (0.841 vs 0.886); the main gap is LNK (0.478 vs 0.845). The LNK gap likely reflects the SSVM/CTC objective mismatch — Hamming-distance margin (per-variable mismatch) doesn't perfectly track CTC linking error (trajectory-level edge errors weighted by track continuity). Closing this gap would require either modifying the SSVM loss to better proxy CTC, or running a post-hoc adjustment on top of the SSVM weights.

### History

The workflow above is the current, simplified one. Previously, structsvm's bundle method appeared not to converge on this problem — ε plateaued at large negative values (-5 at default reg) and inference produced empty solutions. A series of workarounds was developed to manage this: per-feature feature-matrix standardization, a `TolerantBundleMethod` subclass that ignored spurious negative ε events, and a 2D post-hoc offset sweep on `intensity_constant` + `drift_constant`. Best result with that pipeline: TRA=0.841 (see `ssvm_results.md` for the full comparison table).

**2026-05-15: an ilpy update fixed the underlying bug**. With the new ilpy, ε converges monotonically from above to ≈0 over ~30 iterations, inference is non-empty directly, and all the workarounds become unnecessary. The previous results are preserved in `ssvm_results.md` for historical reference.

Full comparison table, including pre-fix experiment runs, lives in `ssvm_results.md` at the repo root.
