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
- `./geff` — geff

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
Hypothesis: "[specific testable prediction]"
TE: X, TF_mean: X, Node_Recall: X, Edge_Recall: X
Verdict: [supported/falsified/inconclusive] — [one line explanation]
```

**Reporting format** after each batch:
```
| Run | Changed Params | TE | TF_mean | Node_Recall | Edge_Recall | Notes |
```
