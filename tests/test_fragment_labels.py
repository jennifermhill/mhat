"""Fragment ids stay unique across frames, including after an empty frame.

Regression test: the running maximum used to be reset to 0 by an all-background
frame, so the next frame's ids restarted at 1 and collided with frame 0's. The
tracker then silently merged those nodes (nx.compose joins nodes by id).
"""

import numpy as np

from mhat.segmentation.labels import offset_labels


def test_ids_stay_unique_after_an_empty_frame():
    frames = [
        np.array([[1, 2], [0, 0]]),
        np.zeros((2, 2), dtype=int),  # empty frame
        np.array([[1, 0], [0, 2]]),
    ]

    max_id = 0
    out = []
    for labels in frames:
        shifted, max_id = offset_labels(labels, max_id, dtype=np.uint32)
        out.append(shifted)

    assert out[0].tolist() == [[1, 2], [0, 0]]
    assert not out[1].any()
    assert out[2].tolist() == [[3, 0], [0, 4]]
    assert max_id == 4
