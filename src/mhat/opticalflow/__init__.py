from .farneback import compute_farneback_flow_2d, compute_farneback_flow_3d
from .utils import enhance_contrast_AHE, rename_flow_uid, frame_average
from .visualization import generate_flow_frames

__version__ = "0.1.0"

# name -> submodule it lives in. Resolved on demand by __getattr__ below.
_LAZY_ATTRS = {
    "compute_lucaskanade_flow_3d": ".lucaskanade",
}

__all__ = [
    "compute_farneback_flow_2d",
    "compute_farneback_flow_3d",
    "compute_lucaskanade_flow_3d",
    "enhance_contrast_AHE",
    "rename_flow_uid",
    "frame_average",
    "generate_flow_frames",
]


def __getattr__(name):
    """PEP 562 lazy attribute access for the torch-dependent submodules."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    try:
        module = import_module(module_name, __name__)
    except ImportError as exc:
        raise ImportError(
            f"{name} requires torch, which is not installed. Install it with:\n"
            '    conda install -c conda-forge pytorch\n'
            "(prefer conda-forge over pip so there is a single OpenMP runtime)."
        ) from exc

    attr = getattr(module, name)
    globals()[name] = attr  # cache so __getattr__ only runs once per name
    return attr


def __dir__():
    return sorted(__all__)
