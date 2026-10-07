# mhat: Multi-Hypothesis Affinity Tracking

[![ci](https://github.com/jennifermhill/mhat/actions/workflows/ci.yaml/badge.svg)](https://github.com/jennifermhill/mhat/actions/workflows/ci.yaml)

MHAT does multi-hypothesis segmentation and tracking for microscopy data. Fragment
hypotheses are generated per frame, assembled into a candidate graph, and resolved into
tracks by an integer linear program.

## Installation

One environment covers the whole pipeline, on Linux, Windows and macOS. Everything comes
from public conda-forge and PyPI.

```bash
git clone https://github.com/jennifermhill/mhat
cd mhat
conda env create -f environment.yml
conda activate mhat
pip install -e ".[dev]"                 # test tooling
pytest                                  # 24 passed, 2 skipped (3D flow tests need .[flow3d])
```

For a byte-reproducible install, use the committed lockfile instead. `uv.lock` is a
single universal lockfile — it resolves every platform at once, so the same file serves
Linux, Windows and macOS:

```bash
uv sync --locked --extra dev            # exactly the versions CI tests
uv run pytest
```

Regenerate it with `uv lock` whenever `pyproject.toml` changes; CI fails if the two
drift apart.

The core install is **headless and CPU-only** — no torch, no CUDA, no GPU. It covers
stages 02_segmentation (threshold backends and waterz affinity agglomeration) through
05_evaluation, plus 07_plotting.

Requires **Python 3.11 or newer**. That floor comes from `motile-tracker`; everything
else in the stack accepts 3.10.

### Solver

The ILP is solved with **SCIP by default**, which ships inside the `pyscipopt` wheel —
no separate install and no license. Gurobi is optional:

```bash
pip install -e ".[gurobi]"              # requires your own Gurobi license
```

### Optional extras

None of these are needed for stages 02–05 or 07_plotting.

| Extra | What it enables | Notes |
|---|---|---|
| `.[napari]` | The per-stage `visualize_results.py` viewers | Already included by `environment.yml` |
| `.[flow3d]` | 3D Farneback optical flow | Pulls `torch` |
| `.[segmentation]` | The cellpose backend (`seg_method = "cellpose"`) | Pulls `torch` |
| `.[all]` | `napari` + `flow3d` + `segmentation` | Not literally everything — omits `gurobi` |
| `.[gurobi]` | Gurobi instead of SCIP | Needs your own license |
| `.[dev]` | Tests, linting, lockfile tooling | |

Extras are additive and can be combined: `pip install -e ".[napari,flow3d]"` is the same
as installing each in turn. Note pip has no "uninstall an extra" operation — to back one
out, rebuild the environment.

All extras are safe to add to an `environment.yml` environment, including together via
`.[all]`. That is not automatic — it is why `environment.yml` keeps the conda layer down
to just the interpreter and lets pip install everything else, napari included.

<details>
<summary><b>Why the conda layer is minimal (measured, not folklore)</b></summary>

Three environment shapes were built and probed on 2026-08-24:

| Shape | 3D Farneback | cellpose |
|---|---|---|
| conda-forge napari + pip torch | aborts | aborts |
| conda-forge napari + conda-forge torch | aborts | aborts |
| **all-pip (what `environment.yml` builds)** | **works** | **works** |

The abort is `OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already
initialized`, and it happens at *import*, not at install — so the install looks fine and
the failure only shows up when you actually run 3D flow or cellpose.

conda-forge's napari pulls in conda numpy (linked against MKL, which loads Intel's
`libiomp5md`) and conda numba (linked against LLVM's `libomp`). pip's torch wheel bundles
a third copy. Two OpenMP families in one process abort. pip's numpy ships a
self-contained `scipy-openblas` instead of MKL, so an all-pip environment only ever loads
one runtime.

Installing torch from conda-forge first does **not** help: the `flow3d`/`segmentation`
extras declare `torch`, so pip installs its own wheel over the top of conda's.

For CUDA builds follow [pytorch.org](https://pytorch.org) rather than pinning a `+cuXXX`
wheel here.
</details>

**`opticalflow3d`** is a PyTorch fork
([jennifermhill/opticalflow3d](https://github.com/jennifermhill/opticalflow3d)), not the
cupy/CUDA package of the same name on PyPI. It is GPLv3 while mhat is BSD-3, so it stays
a separate distribution. Note the import name is capitalised: `from opticalflow3D.helpers ...`.

**`waterz`** (affinity agglomeration, stage 02) is a core dependency. Since 0.10.0 PyPI
ships prebuilt wheels for Linux, Windows and macOS, and the two scoring functions MHAT
uses — the default mean affinity and the size-ratio `scoring_function = "symmetric"` —
are compiled into the wheel, so no C++ toolchain or Boost is needed. Any other
`scoring_function` (for example `"random"`) is compiled on first use through `witty`
and does need a C++ compiler plus Boost headers.

## Pipeline

Each numbered stage under `scripts/` takes a TOML config (see the `*_config_example.toml`
next to each script) and writes into an output directory you choose.

| Stage | Does |
|---|---|
| `02_segmentation` | Generate fragment hypotheses and a merge history |
| `03_opticalflow` | Farneback optical flow between frames |
| `04_tracking` | Build the multi-hypothesis graph and solve the ILP |
| `05_evaluation` | Tracking and segmentation metrics via traccuracy |
| `07_plotting` | Figures from evaluation results |

### Data layout

You give three kinds of path, with no required structure relative to each other:

- **`raw_path`** — the raw movie, a zarr array `(t, c, *spatial)` with an `axes`
  attribute (`scripts/ctc_to_zarr.py` converts Cell Tracking Challenge sequences).
- **`output_dir`** — one directory per raw movie, holding everything computed from it.
- **ground truth** — for evaluation only: `gt_tracks` (a geff), optionally `gt_seg` and
  `ctc_seg_dir`. The pipeline never writes to these.

Inside the output directory the layout is fixed, so each stage finds the others' runs by
uid (a timestamp unless you set `exp_uid`):

```
<output_dir>/
    raw.toml                       the raw movie this directory belongs to
    segmentation/<uid>/            data.zarr  merge_history.csv  config.toml
    opticalflow/{2d,3d}/<uid>/     flow.zarr  config.toml
    tracking/<uid>/                pred_tracks.zarr  pred_seg.zarr  config.toml
        evaluation/<label>/            track_metrics.json  eval_config.toml
```

- Every run saves the config that made it. A tracking run's config names the
  segmentation and flow runs it read (`seg_result`, `flow_result`).
- `raw.toml` ties the directory to one raw movie: segmentation, optical flow and
  tracking refuse to run with a different `raw_path`, so results from two movies can
  never be mixed. If the movie moves, or you work on another machine, update its `path`.
- Evaluations live inside the tracking run they score, one directory each, named
  `<ground truth>_<metrics>_<timestamp>`, so evaluating against another ground truth or
  metric set never overwrites an earlier result. No run is ever redone in place: a
  fixed `exp_uid` must be new, so to redo a run use a new uid, or delete the old run
  directory first.
- Cell Tracking Challenge ground truth is converted once with
  `scripts/05_evaluation/convert_ctc_gt.py`.
- The `visualize_results.py` viewers take a run directory and find everything else from
  it; `--raw` / `--gt-tracks` cover paths that differ on the viewing machine.

### Bringing your own segmentation

Any integer label image can be the fragments for stage 02: set `seg_method = "file"` and
`fragments_path` to a zarr array `(t, *spatial)` of the raw movie's shape (per-frame
numbering is fine). Stage 02 then computes the affinities, the merge history and the
segmentation hypotheses from it as usual. Optical flow is always computed by stage 03.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

`tests/` is a fast in-memory smoke suite. `test_tracking_pipeline.py` builds a synthetic
tracking problem and solves it for real — one solve exercises all four of the custom
motile cost/variable subclasses in `src/mhat/tracking/`, which subclass extension points
that are not public motile API. It is the contract behind the `motile>=0.3,<0.4` pin, so
run it before widening any version range.

Only `src/mhat` is packaged. Keep stray folders out of `src/` — setuptools auto-discovers
every package there, so a directory dropped in would be built into the wheel.
