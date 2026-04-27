import numpy as np

from cellpose import models

def segment_with_cellpose(image: np.ndarray, gpu: bool = False, **kwargs) -> np.ndarray:
    """
    Segment cells in a 3D image using Cellpose. Computes masks in 2D slices and stitches in 3D based on mask overlap.

    Parameters:
    - image: 3D numpy array representing the image to segment.
    - gpu: Boolean indicating whether to use GPU for segmentation.
    - **kwargs: Additional keyword arguments passed to model.eval().

    Returns:
    - masks: 3D numpy array of the same shape as input image, where each cell is labeled with a unique integer.
    """

    # Initialize Cellpose model
    model = models.CellposeModel(gpu=gpu)

    # Defaults — overridden by anything in kwargs
    eval_params = {
        "z_axis": 0,
        "channel_axis": 1,
        "batch_size": 32,
        "flow_threshold": 0.4,
        "cellprob_threshold": -3.0,
        "do_3D": True,
    }
    eval_params.update(kwargs)

    masks, _, _ = model.eval(image, **eval_params)
    masks = masks.astype(np.uint32)

    return masks