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
- `./motile_toolbox` — motile-toolbox

### Key Dependencies

- **motile** — ILP-based network flow solver for tracking
- **gurobi/ilpy/pyscipopt** — ILP optimizer backends
- **zarr** — Chunked array storage format
- **gunpowder** — Training data pipeline
- **traccuracy** — Tracking evaluation metrics
- **cellpose** — Deep learning cell segmentation
