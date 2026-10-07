import numpy as np


def offset_labels(
    labels: np.ndarray, max_id: int, dtype: np.dtype
) -> tuple[np.ndarray, int]:
    """Shift one frame's labels above every id used by earlier frames.

    Fragment ids must be unique across the whole movie: tracking builds one
    graph per frame and joins them with ``nx.compose``, which silently merges
    nodes that share an id. Background (0) stays 0.

    Args:
        labels: Label image for one frame, numbered however the segmenter
            numbers it (typically 1..n per frame).
        max_id: Largest id used by any earlier frame, 0 before the first.
        dtype: dtype of the array the labels will be written into.

    Returns:
        The shifted labels as ``dtype``, and the new running maximum. An empty
        frame leaves the maximum unchanged, so the next frame still starts
        above every earlier one.

    Raises:
        OverflowError: If the ids no longer fit in ``dtype``.
    """
    dtype = np.dtype(dtype)
    shifted = labels.astype(np.uint64)
    shifted[shifted != 0] += np.uint64(max_id)
    max_id = max(max_id, int(shifted.max(initial=0)))
    if max_id > np.iinfo(dtype).max:
        raise OverflowError(f"Fragment ids reach {max_id}, more than {dtype} holds")
    return shifted.astype(dtype), max_id
