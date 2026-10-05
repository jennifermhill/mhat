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

Each numbered stage under `scripts/` takes a TOML config and writes into `experiments/`.

| Stage | Does |
|---|---|
| `02_segmentation` | Generate fragment hypotheses and a merge history |
| `03_opticalflow` | Farneback optical flow between frames |
| `04_tracking` | Build the multi-hypothesis graph and solve the ILP |
| `05_evaluation` | Tracking and segmentation metrics via traccuracy |
| `07_plotting` | Figures from evaluation results |

### The 02 → 03 boundary is data, not code

Stage 02 produces exactly two artifacts that anything downstream reads:

- `data.zarr/fragments` — uint32 labels, shape `(T, Z, Y, X)`, with an `axes` attribute
- `merge_history.csv` — columns `a, b, c, cost, timepoint`, sorted by ascending cost

Any segmentation that emits those two can be tracked, so stages 03–05 and 07 can also be
run on fragments and merge histories produced elsewhere.

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
