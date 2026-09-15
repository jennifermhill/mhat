import numpy as np

from cellpose import models

def segment_with_cellpose(image: np.ndarray, gpu: bool = False, **kwargs) -> np.ndarray:
    """
    Segment cells in one frame with Cellpose.

    A 3D frame is segmented volumetrically (do_3D). A 2D frame is segmented as
    an ordinary single image: `z_axis`, `do_3D` and `channel_axis` are all
    dropped, because on a (y, x) array they would name axes that are not there
    -- cellpose would read y as z and x as the channel axis.

    Parameters:
    - image: numpy array to segment, (z, y, x) or (y, x).
    - gpu: Boolean indicating whether to use GPU for segmentation.
    - **kwargs: Additional keyword arguments passed to model.eval(). These
      override the defaults below, including the rank-specific ones.

    Returns:
    - masks: numpy array of the same shape as the input image, where each cell
      is labeled with a unique integer.
    """

    # Initialize Cellpose model
    model = models.CellposeModel(gpu=gpu)

    # Defaults — overridden by anything in kwargs
    eval_params = {
        "batch_size": 32,
        "flow_threshold": 0.4,
        "cellprob_threshold": -3.0,
    }
    if image.ndim == 3:
        eval_params.update({
            "z_axis": 0,
            "channel_axis": 1,
            "do_3D": True,
        })
    else:
        assert image.ndim == 2, (
            f"expected a 2D or 3D frame, got shape {image.shape}"
        )
    eval_params.update(kwargs)

    masks, _, _ = model.eval(image, **eval_params)
    masks = masks.astype(np.uint32)

    return masks