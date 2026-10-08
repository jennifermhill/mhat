"""The layout of one output directory, and the raw array it belongs to.

A user gives three kinds of path, with no required structure relative to each
other: the raw movie (``raw_path``), one output directory per raw movie
(``output_dir``), and ground-truth files (given to evaluation only). Inside the
output directory the layout is fixed, so each stage finds the others' runs by
uid::

    <output_dir>/
        raw.toml                       the raw movie this directory belongs to
        segmentation/<uid>/            data.zarr  merge_history.csv  config.toml
        opticalflow/{2d,3d}/<uid>/     flow.zarr  config.toml
        tracking/<uid>/                pred_tracks.zarr  pred_seg.zarr  config.toml
            evaluation/<label>/            track_metrics.json  eval_config.toml
        ssvm/<uid>/                    weight fits, laid out like tracking runs

A run is a directory holding a ``config.toml``. Each tracking run's config
records the uids of the segmentation and flow runs it read (``seg_result``,
``flow_result``); evaluations live inside the tracking (or ssvm) run they score.

A uid is a directory name or a relative path with forward slashes, so runs can
be grouped inside a stage, e.g. ``tracking/sweeps/<sweep_id>/runs/<label>``.
Listing a stage's runs shows only the ungrouped ones.

``Dataset`` locates things and guards the raw binding and existing runs. It
never deletes or overwrites data.
"""

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import toml
import zarr

from mhat.utils import get_axes_metadata

RAW_RECORD = "raw.toml"
FLOW_KINDS = ("2d", "3d")
RUN_STAGES = ("segmentation", "opticalflow", "tracking", "ssvm")
# Stages whose runs are tracking solutions: evaluations nest inside them.
SOLUTION_STAGES = ("tracking", "ssvm")
Stage = Literal["segmentation", "opticalflow", "tracking", "ssvm"]

# IO keys from the old layout (results keyed by experiment/dataset under base
# directories). A stage config still carrying them was written for that layout.
OLD_IO_KEYS = (
    "input_base_dir",
    "output_base_dir",
    "raw_base_dir",
    "experiment",
    "dataset",
    "ctc_gt",
)


def require_dir(path: Path, what: str) -> Path:
    """Return ``path`` if it is a directory, else raise naming what was expected."""
    if not path.is_dir():
        raise FileNotFoundError(f"{what} {path} is missing")
    return path


def absolute_path(path: str | Path) -> Path:
    """An absolute, normalized path that keeps the drive letter.

    ``Path.resolve()`` would turn a mapped Windows drive (``Y:\\...``) into its
    UNC path (``\\\\server\\share\\...``), which is unreadable in ``raw.toml``
    and compares unequal to the drive-letter form.
    """
    return Path(os.path.abspath(path))


def same_path(a: str | Path, b: str | Path) -> bool:
    """Whether two paths name the same location (case-insensitive on Windows)."""
    return os.path.normcase(absolute_path(a)) == os.path.normcase(absolute_path(b))


def check_uid(uid: str) -> str:
    """Return ``uid`` if it names a run inside its stage directory, else raise.

    A uid is one directory name or a relative path of them, separated by
    forward slashes so the configs that record it read the same on every OS.
    ``..``, ``.``, empty parts and absolute paths are refused: they would place
    the run outside its stage directory, possibly in another output directory.
    """
    parts = uid.split("/") if isinstance(uid, str) else None
    if (
        not parts
        or "\\" in uid
        or ":" in uid
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(
            f"Invalid run uid {uid!r}: use a directory name, or a relative path "
            f"of them separated by '/' (e.g. 'sweeps/s1/runs/a'), without '..'"
        )
    return uid


@dataclass(frozen=True)
class Dataset:
    """One output directory and the raw movie it belongs to.

    Args:
        output_dir: Directory holding this movie's segmentation, optical flow,
            tracking and evaluation results.
        raw_path: The raw movie, a zarr array ``(t, c, *spatial)``. Only
            needed by the stages that read it (segmentation, optical flow,
            tracking).
    """

    output_dir: Path
    raw_path: Path | None = None

    @classmethod
    def from_config(cls, config: dict) -> "Dataset":
        """Build a Dataset from a stage config's ``output_dir`` and ``raw_path``.

        Raises:
            ValueError: If the config uses the old layout's IO keys.
        """
        old_keys = [key for key in OLD_IO_KEYS if key in config]
        if old_keys:
            raise ValueError(
                f"Config uses IO keys from the old results layout: {old_keys}. "
                f"Stage configs now take raw_path and output_dir (and gt_tracks "
                f"etc. for evaluation); see the Data layout section of README.md."
            )
        if "output_dir" not in config:
            raise ValueError("Config needs an output_dir")
        raw_path = config.get("raw_path")
        return cls(
            output_dir=Path(config["output_dir"]),
            raw_path=Path(raw_path) if raw_path is not None else None,
        )

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        stage: Stage | tuple[Stage, ...],
    ) -> "Dataset":
        """The Dataset a run belongs to, found from where the run sits.

        The output directory is the nearest folder above the run holding a
        ``raw.toml`` (every stage that writes runs records one), so grouped runs
        such as ``tracking/sweeps/<id>/runs/<label>`` are found too. Works on
        any machine, whatever paths the run's saved config holds; ``raw_path``
        is the recorded one.

        Args:
            run_dir: The run's directory, or its ``config.toml``.
            stage: The stage that made the run, or a tuple of acceptable stages
                (e.g. ``SOLUTION_STAGES`` for a viewer of tracking solutions).

        Raises:
            ValueError: If ``run_dir`` is not inside an output directory, or is
                not a run of ``stage`` there.
        """
        run_dir = absolute_path(run_dir)
        if run_dir.name == "config.toml":
            run_dir = run_dir.parent
        stages = (stage,) if isinstance(stage, str) else tuple(stage)

        output_dir = next(
            (d for d in run_dir.parents if (d / RAW_RECORD).is_file()), None
        )
        if output_dir is None:
            raise ValueError(
                f"{run_dir} is not inside an output directory (no {RAW_RECORD} "
                f"in any folder above it)"
            )
        parts = run_dir.relative_to(output_dir).parts
        found = parts[0]
        n_prefix = 2 if found == "opticalflow" else 1
        if (
            found not in stages
            or len(parts) <= n_prefix
            or (found == "opticalflow" and parts[1] not in FLOW_KINDS)
        ):
            layouts = " or ".join(
                f"<output_dir>/{s}/{'<2d|3d>/' if s == 'opticalflow' else ''}<uid>"
                for s in stages
            )
            raise ValueError(
                f"{run_dir} is not a {' or '.join(stages)} run directory in "
                f"{output_dir} ({layouts})"
            )
        dataset = cls(output_dir=output_dir)
        return cls(output_dir=output_dir, raw_path=dataset.recorded_raw_path())

    @property
    def name(self) -> str:
        """Label for this movie in viewers and tables: the output directory's name."""
        return absolute_path(self.output_dir).name

    # --- raw ---

    def recorded_raw_path(self) -> Path | None:
        """The raw path recorded in ``raw.toml``, or None if there is none yet."""
        record = self.output_dir / RAW_RECORD
        if not record.is_file():
            return None
        return Path(toml.load(record)["path"])

    def bind_raw(self) -> None:
        """Tie the output directory to this raw movie, or check that it is tied.

        The first stage to run records ``raw_path`` in ``raw.toml``. Every later
        call compares against it, so runs made from different movies can never
        share an output directory -- tracking would otherwise read another
        movie's segmentation without complaint whenever the shapes match.

        Raises:
            ValueError: If ``raw_path`` differs from the recorded one.
        """
        raw_path = absolute_path(self.require_raw_path())
        recorded = self.recorded_raw_path()
        if recorded is None:
            raw = self.raw()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / RAW_RECORD, "w") as f:
                toml.dump(
                    {
                        "path": str(raw_path),
                        "shape": list(raw.shape),
                        "axes": get_axes_metadata(raw),
                    },
                    f,
                )
        elif not same_path(recorded, raw_path):
            raise ValueError(
                f"{self.output_dir} holds results for the raw movie {recorded}, "
                f"but this run reads {raw_path}. Use a separate output_dir for "
                f"each movie. If the movie itself moved, or you are on another "
                f"machine, update 'path' in {self.output_dir / RAW_RECORD}."
            )

    def require_raw_path(self) -> Path:
        """``raw_path``, checked to be set and to exist."""
        if self.raw_path is None:
            raise ValueError("No raw_path given (the raw movie this stage reads)")
        return require_dir(self.raw_path, "Raw data")

    def raw(self) -> zarr.Array:
        """The raw movie as a read-only zarr array, ``(t, c, *spatial)``."""
        return zarr.open(self.require_raw_path(), mode="r")

    # --- run directories ---

    def seg_dir(self, uid: str) -> Path:
        return self.output_dir / "segmentation" / check_uid(uid)

    def flow_dir(self, uid: str, kind: Literal["2d", "3d"]) -> Path:
        if kind not in FLOW_KINDS:
            raise ValueError(f"kind must be one of {FLOW_KINDS}, got {kind!r}")
        return self.output_dir / "opticalflow" / kind / check_uid(uid)

    def track_dir(self, uid: str) -> Path:
        return self.output_dir / "tracking" / check_uid(uid)

    def solution_dir(
        self, uid: str, stage: Literal["tracking", "ssvm"] = "tracking"
    ) -> Path:
        """A tracking solution's run directory: a tracking run or an ssvm fit."""
        if stage not in SOLUTION_STAGES:
            raise ValueError(f"stage must be one of {SOLUTION_STAGES}, got {stage!r}")
        return self.output_dir / stage / check_uid(uid)

    def flow_dirs(
        self, flow_uid: str | None, seg_uid: str
    ) -> dict[str, Path | None]:
        """The flow directories a tracking run uses, as ``{"2d": ..., "3d": ...}``.

        Which flow is required depends on the data's rank, read from the
        segmentation's ``fragments`` array (metadata only):

        - 3D data needs the 3D flow, since nothing else estimates axial
          motion. A 2D flow beside it is optional and, when present, supplies
          the better-resolved in-plane components.
        - 2D data needs the 2D flow; there is no 3D flow to fall back on.

        Raises:
            FileNotFoundError: If the required flow directory is missing.
        """
        if flow_uid is None:
            return {"2d": None, "3d": None}

        fragments_path = self.seg_dir(seg_uid) / "data.zarr"
        ndim = zarr.open(fragments_path, mode="r")["fragments"].ndim - 1
        flow_2d = self.flow_dir(flow_uid, "2d")
        flow_3d = self.flow_dir(flow_uid, "3d")
        if ndim == 3:
            require_dir(flow_3d, "3D optical flow data directory")
            return {"2d": flow_2d if flow_2d.is_dir() else None, "3d": flow_3d}
        require_dir(
            flow_2d,
            "2D optical flow data directory (2D data has no 3D flow to fall back on)",
        )
        return {"2d": flow_2d, "3d": None}

    def run_dir(
        self,
        stage: Stage,
        uid: str,
        kind: Literal["2d", "3d"] | None = None,
    ) -> Path:
        """One run's directory; ``kind`` is required for optical flow only."""
        if stage == "opticalflow":
            return self.flow_dir(uid, kind)
        if stage not in RUN_STAGES:
            raise ValueError(f"stage must be one of {RUN_STAGES}, got {stage!r}")
        return self.output_dir / stage / check_uid(uid)

    def check_new_run(
        self,
        stage: Stage,
        uid: str,
        kind: Literal["2d", "3d"] | None = None,
    ) -> Path:
        """A run's directory, checked not to hold an earlier run.

        Everything inside a run directory -- downstream runs' inputs,
        evaluations, plots or movies made from it -- is attributed to that
        run's result, so a run is never redone in place, and a grouped uid
        never places a run inside another run. Redoing one means a new uid, or
        deleting the old run directory first.

        Raises:
            FileExistsError: If the run directory exists and is not empty, or
                lies inside another run.
        """
        run_dir = self.run_dir(stage, uid, kind)
        if run_dir.is_dir() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"{stage} run {run_dir} already exists. Use a new exp_uid, or "
                f"delete that directory first (including anything saved in it) "
                f"to redo the run under the same uid."
            )
        n_groups = len(uid.split("/")) - 1
        for group in list(run_dir.parents)[:n_groups]:
            if (group / "config.toml").is_file():
                raise FileExistsError(
                    f"{stage} run {run_dir} would lie inside the run {group}; "
                    f"group runs in folders that are not runs themselves."
                )
        return run_dir

    def prepare_run_dir(
        self,
        stage: Stage,
        uid: str,
        kind: Literal["2d", "3d"] | None = None,
    ) -> Path:
        """Create a new run's directory (see ``check_new_run``)."""
        run_dir = self.check_new_run(stage, uid, kind)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def new_eval_dir(
        self,
        track_uid: str,
        gt_label: str,
        metrics: list[str],
        stage: Literal["tracking", "ssvm"] = "tracking",
    ) -> Path:
        """Create a new evaluation directory inside a tracking (or ssvm) run.

        Named ``<gt_label>_<metrics>_<timestamp>``, with the metric names sorted
        and joined by ``+``, so evaluations against different ground truths or
        metric sets never overwrite each other and sort by time.

        Raises:
            FileExistsError: If that name is taken (same label, same second).
        """
        label = f"{gt_label}_{'+'.join(sorted(metrics))}_{self.new_uid()}"
        eval_dir = self.solution_dir(track_uid, stage) / "evaluation" / label
        eval_dir.mkdir(parents=True, exist_ok=False)
        return eval_dir

    @staticmethod
    def new_uid() -> str:
        """A fresh run uid: the current time, ``%Y-%m-%d_%H-%M-%S``."""
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- discovery and provenance ---

    def runs(
        self,
        stage: Stage,
        kind: Literal["2d", "3d"] | None = None,
    ) -> list[str]:
        """Uids of a stage's ungrouped runs, oldest first (for timestamp uids).

        A run is a subdirectory holding a ``config.toml``; anything else in the
        stage directory (groups such as ``sweeps/``, caches) is left out.
        """
        if stage not in RUN_STAGES:
            raise ValueError(f"stage must be one of {RUN_STAGES}, got {stage!r}")
        if stage == "opticalflow":
            if kind not in FLOW_KINDS:
                raise ValueError(f"opticalflow needs a kind in {FLOW_KINDS}")
            parent = self.output_dir / "opticalflow" / kind
        else:
            parent = self.output_dir / stage
        return _config_subdirs(parent)

    def evals(
        self, track_uid: str, stage: Literal["tracking", "ssvm"] = "tracking"
    ) -> list[str]:
        """Labels of a run's evaluations, oldest first within a label."""
        parent = self.solution_dir(track_uid, stage) / "evaluation"
        if not parent.is_dir():
            return []
        return sorted(entry.name for entry in os.scandir(parent) if entry.is_dir())

    def inputs_of(
        self, track_uid: str, stage: Literal["tracking", "ssvm"] = "tracking"
    ) -> tuple[str, str | None]:
        """The ``(seg_uid, flow_uid)`` a tracking (or ssvm) run read;
        ``flow_uid`` is ``None`` for a run without optical flow."""
        config = toml.load(self.solution_dir(track_uid, stage) / "config.toml")
        return config["seg_result"], config.get("flow_result")


def _config_subdirs(parent: Path) -> list[str]:
    if not parent.is_dir():
        return []
    return sorted(
        entry.name
        for entry in os.scandir(parent)
        if entry.is_dir() and (parent / entry.name / "config.toml").is_file()
    )
