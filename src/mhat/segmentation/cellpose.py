import numpy as np

from cellpose import models

def segment_with_cellpose(image: np.ndarray, gpu: bool = False, diameter: int = 30) -> np.ndarray:
    """
    Segment cells in a 3D image using Cellpose. Computes masks in 2D slices and stitches in 3D based on mask overlap.

    Parameters:
    - image: 3D numpy array representing the image to segment.
    - model_type: Type of Cellpose model to use ('cyto' or 'nuclei').
    - diameter: Estimated diameter of the cells.

    Returns:
    - masks: 3D numpy array of the same shape as input image, where each cell is labeled with a unique integer.
    """

    # Initialize Cellpose model
    model = models.CellposeModel(gpu=gpu)

    # Run Cellpose segmentation
    masks, _, _ = model.eval(image, diameter=diameter, z_axis=0,
                             channel_axis=1, batch_size=32,
                             do_3D=False, stitch_threshold=0.2)
    masks = masks.astype(np.uint32)

    return masks