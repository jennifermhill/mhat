# mhat: Multi-Hypothesis Affinity Tracking

[![tests](https://github.com/funkelab/mhat/actions/workflows/tests.yaml/badge.svg)](https://github.com/funkelab/mhat/actions/workflows/tests.yaml)
[![black](https://github.com/funkelab/mhat/actions/workflows/black.yaml/badge.svg)](https://github.com/funkelab/mhat/actions/workflows/black.yaml)
[![mypy](https://github.com/funkelab/mhat/actions/workflows/mypy.yaml/badge.svg)](https://github.com/funkelab/mhat/actions/workflows/mypy.yaml)
[![codecov](https://codecov.io/gh/funkelab/mhat/branch/main/graph/badge.svg)](https://codecov.io/gh/funkelab/mhat)

## Optical flow environment (macOS + Linux)

`environment.yml` builds a cross-platform environment for the optical flow stage
(`scripts/03_opticalflow`). It works on both Apple Silicon macOS and Linux. Run it **from the
repo root** so the `.` pip entry resolves to this checkout:

```bash
conda env create -f environment.yml   # installs the conda base, opticalflow3d, and mhat itself
```

Notes:
- `torch` is installed from **conda-forge** (not pip) on purpose: pip's torch bundles its own
  `libomp`, which collides with conda's `llvm-openmp` and aborts with `OMP Error #15`. Using
  conda's torch keeps a single OpenMP runtime.
- `opticalflow3d` is a custom PyTorch fork (not the stock cupy/CUDA build on PyPI), pinned in
  `environment.yml` as a `git+https` URL
  (`git+https://github.com/jennifermhill/opticalflow3d`). This makes the environment portable to
  any machine with network access — no shared-filesystem path needed. Note the import name is
  capitalized: `from opticalflow3D.helpers... import ...`.
- The `mhat` package itself is installed from this checkout via the `.` entry in the pip section,
  which pulls in mhat's full dependency set (motile/ilpy, gunpowder, traccuracy, neuroglancer, ...).
  The deep-segmentation backends (`waterz`, `cellpose`) are not declared mhat deps and are not
  included here.
- Only `src/mhat` is packaged. Keep stray folders out of `src/` — setuptools auto-discovers every
  package there, so a directory dropped into `src/` would get built into the wheel.
