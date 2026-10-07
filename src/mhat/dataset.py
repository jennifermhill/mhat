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

A run is a directory holding a ``config.toml``. Each tracking run's config
records the uids of the segmentation and flow runs it read (``seg_result``,
``flow_result``); evaluations live inside the tracking run they score.

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
RUN_STAGES = ("segmentation", "opticalflow", "tracking")

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
        stage: Literal["segmentation", "opticalflow", "tracking"],
    ) -> "Dataset":
        """The Dataset a run belongs to, found from where the run sits.

        Works on any machine, whatever paths the run's saved config holds.
        ``raw_path`` comes from ``raw.toml`` and is ``None`` if there is none.

        Args:
            run_dir: The run's directory, or its ``config.toml``.
            stage: The stage that made the run.

        Raises:
            ValueError: If ``run_dir`` is not a ``stage`` run in this layout.
        """
        run_dir = absolute_path(run_dir)
        if run_dir.name == "config.toml":
            run_dir = run_dir.parent
        if stage == "opticalflow":
            stage_dir = run_dir.parent.parent
            in_layout = run_dir.parent.name in FLOW_KINDS
        else:
            stage_dir = run_dir.parent
            in_layout = True
        if not in_layout or stage_dir.name != stage:
            kind = "<2d|3d>/" if stage == "opticalflow" else ""
            raise ValueError(
                f"{run_dir} is not a {stage} run directory "
                f"(<output_dir>/{stage}/{kind}<uid>)"
            )
        dataset = cls(output_dir=stage_dir.parent)
        recorded = dataset.recorded_raw_path()
        return cls(output_dir=dataset.output_dir, raw_path=recorded)

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
        return self.output_dir / "segmentation" / uid

    def flow_dir(self, uid: str, kind: Literal["2d", "3d"]) -> Path:
        if kind not in FLOW_KINDS:
            raise ValueError(f"kind must be one of {FLOW_KINDS}, got {kind!r}")
        return self.output_dir / "opticalflow" / kind / uid

    def track_dir(self, uid: str) -> Path:
        return self.output_dir / "tracking" / uid

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
        stage: Literal["segmentation", "opticalflow", "tracking"],
        uid: str,
        kind: Literal["2d", "3d"] | None = None,
    ) -> Path:
        """One run's directory; ``kind`` is required for optical flow only."""
        if stage == "opticalflow":
            return self.flow_dir(uid, kind)
        if stage not in RUN_STAGES:
            raise ValueError(f"stage must be one of {RUN_STAGES}, got {stage!r}")
        return self.output_dir / stage / uid

    def check_new_run(
        self,
        stage: Literal["segmentation", "opticalflow", "tracking"],
        uid: str,
        kind: Literal["2d", "3d"] | None = None,
    ) -> Path:
        """A run's directory, checked not to hold an earlier run.

        Everything inside a run directory -- downstream runs' inputs,
        evaluations, plots or movies made from it -- is attributed to that
        run's result, so a run is never redone in place. Redoing one means a
        new uid, or deleting the old run directory first.

        Raises:
            FileExistsError: If the run directory exists and is not empty.
        """
        run_dir = self.run_dir(stage, uid, kind)
        if run_dir.is_dir() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"{stage} run {run_dir} already exists. Use a new exp_uid, or "
                f"delete that directory first (including anything saved in it) "
                f"to redo the run under the same uid."
            )
        return run_dir

    def prepare_run_dir(
        self,
        stage: Literal["segmentation", "opticalflow", "tracking"],
        uid: str,
        kind: Literal["2d", "3d"] | None = None,
    ) -> Path:
        """Create a new run's directory (see ``check_new_run``)."""
        run_dir = self.check_new_run(stage, uid, kind)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def new_eval_dir(
        self, track_uid: str, gt_label: str, metrics: list[str]
    ) -> Path:
        """Create a new evaluation directory inside a tracking run.

        Named ``<gt_label>_<metrics>_<timestamp>``, with the metric names sorted
        and joined by ``+``, so evaluations against different ground truths or
        metric sets never overwrite each other and sort by time.

        Raises:
            FileExistsError: If that name is taken (same label, same second).
        """
        label = f"{gt_label}_{'+'.join(sorted(metrics))}_{self.new_uid()}"
        eval_dir = self.track_dir(track_uid) / "evaluation" / label
        eval_dir.mkdir(parents=True, exist_ok=False)
        return eval_dir

    @staticmethod
    def new_uid() -> str:
        """A fresh run uid: the current time, ``%Y-%m-%d_%H-%M-%S``."""
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- discovery and provenance ---

    def runs(
        self,
        stage: Literal["segmentation", "opticalflow", "tracking"],
        kind: Literal["2d", "3d"] | None = None,
    ) -> list[str]:
        """Uids of a stage's runs, oldest first (for timestamp uids).

        A run is a subdirectory holding a ``config.toml``; anything else in the
        stage directory (sweeps, caches) is left out.
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

    def evals(self, track_uid: str) -> list[str]:
        """Labels of a tracking run's evaluations, oldest first within a label."""
        parent = self.track_dir(track_uid) / "evaluation"
        if not parent.is_dir():
            return []
        return sorted(entry.name for entry in os.scandir(parent) if entry.is_dir())

    def inputs_of(self, track_uid: str) -> tuple[str, str | None]:
        """The ``(seg_uid, flow_uid)`` a tracking run read; ``flow_uid`` is
        ``None`` for a run without optical flow."""
        config = toml.load(self.track_dir(track_uid) / "config.toml")
        return config["seg_result"], config.get("flow_result")


def _config_subdirs(parent: Path) -> list[str]:
    if not parent.is_dir():
        return []
    return sorted(
        entry.name
        for entry in os.scandir(parent)
        if entry.is_dir() and (parent / entry.name / "config.toml").is_file()
    )
