"""Read figure specs: which run and which eval each bar shows.

A figure spec (``configs/evaluation/*.toml``) names, for each dataset table, the
``output_dir`` of the movie, and for each condition (bar, method, ...) the
``run`` it shows -- a path relative to ``output_dir`` such as
``tracking/regenF01_baseline``, ``tracking/sweeps/coh01c/runs/a1400_ec500`` or
``ssvm/sweeps/stage2_fits/runs/gta_A_n050_s0`` -- and the exact ``eval``
directory name inside it, timestamp included. A second eval a figure reads
(e.g. basic metrics for a recall panel) is another key of the condition, such as
``recall_eval``. Naming the exact eval, rather than looking one up, means a later
re-eval can never change a figure.

``PLACEHOLDER`` as a ``run`` or eval skips that bar while a figure is in
progress. Any other named result that is missing raises: a zero bar in a figure
is too easy to miss. Old specs (``experiment`` / ``dataset_dir`` /
``eval_base_dir`` / ``tracking_uid``) are refused with a pointer to
``scripts/convert_figure_specs.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

PLACEHOLDER = "<fill in>"
OLD_SPEC_KEYS = (
    "experiment", "dataset_dir", "eval_base_dir", "metrics_filename",
    "recall_metrics_filename", "tracking_uid", "seg_uid",
)


def check_spec_table(table: dict, where: str = "spec") -> None:
    """Refuse a table written for the old results layout."""
    old = [key for key in OLD_SPEC_KEYS if key in table]
    if old:
        raise ValueError(
            f"{where} uses keys of the old results layout: {old}. Convert the spec "
            "with scripts/convert_figure_specs.py (output_dir, and run + eval per "
            "condition)."
        )


def walk_json_path(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def metrics_path(table: dict, cond: dict, key: str = "eval") -> Path | None:
    """The ``track_metrics.json`` of the eval a condition names under ``key``.

    ``table`` is the dataset table holding ``output_dir``; ``cond`` the condition
    holding ``run`` and the eval name. Returns None for a placeholder.

    Raises:
        ValueError: If either table uses the old spec keys.
        KeyError: If the condition names no run or eval.
        FileNotFoundError: If the named eval does not exist.
    """
    check_spec_table(table)
    check_spec_table(cond, "condition")
    if "run" not in cond or key not in cond:
        raise KeyError(f"condition {cond.get('label', cond)!r} needs 'run' and {key!r}")
    run, eval_name = cond["run"], cond[key]
    if PLACEHOLDER in (run, eval_name):
        return None
    path = Path(table["output_dir"]) / run / "evaluation" / eval_name / "track_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: the spec names an eval that does not exist "
            "(a typo, or a moved run)"
        )
    return path


def load_metrics(table: dict, cond: dict, key: str = "eval") -> dict | None:
    """The condition's ``track_metrics.json`` as a dict, or None for a placeholder."""
    path = metrics_path(table, cond, key)
    if path is None:
        return None
    with open(path) as handle:
        return json.load(handle)


def is_placeholder(cond: dict, key: str = "eval") -> bool:
    return PLACEHOLDER in (cond.get("run"), cond.get(key))


def figure_path(table: dict, default_name: str, override=None, key: str = "output_png") -> Path:
    """Where a figure goes: ``--output``, else the table's ``key``, else
    ``<output_dir>/figures/<default_name>``."""
    if override:
        return Path(override)
    if table.get(key):
        return Path(table[key])
    return Path(table["output_dir"]) / "figures" / default_name
