"""The drawn scale bar: a wrong length or a bar that moves when the text is
left out would go unnoticed in a figure.
"""

import numpy as np
import pytest
from mhat.visualization.scale_bar import ScaleBar, nice_length


def bar_pixels(frame: np.ndarray) -> tuple[slice, slice]:
    """Rows and columns of the white pixels, which must form one rectangle."""
    rows, cols = np.nonzero(frame[..., 0] == 255)
    assert len(rows) == (np.ptp(rows) + 1) * (np.ptp(cols) + 1)
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        # 100 x 200 frame, margin 10: a 40 px bar, 4 px thick.
        ("bottom_right", (slice(86, 90), slice(150, 190))),
        ("bottom_left", (slice(86, 90), slice(10, 50))),
        # top: margin + 1.5 x cap height (8 px) above the bar
        ("top_left", (slice(22, 26), slice(10, 50))),
    ],
)
def test_bar_length_and_place(position, expected):
    bar = ScaleBar(
        length_px=40.0,
        text="",
        thickness=4,
        cap_height=8,
        margin=10,
        position=position,
    )
    frame = bar.draw(np.zeros((100, 200, 4), np.uint8))
    assert bar_pixels(frame) == expected
    assert (frame[expected][..., 3] == 255).all()


def test_text_does_not_move_bar():
    bar = ScaleBar(length_px=40.0, text="50 µm", thickness=4, cap_height=8, margin=10)
    blank = np.zeros((100, 200, 3), np.uint8)
    with_text = bar.draw(blank)
    without = bar.draw(blank, text=False)
    rows, cols = bar_pixels(without)
    assert (with_text[rows, cols] == 255).all()
    # The text is drawn, above the bar only.
    assert with_text[: rows.start].any()
    assert not with_text[rows.stop :].any()


def test_nice_length():
    assert [nice_length(x) for x in (59.2, 50.0, 0.3, 1000.0, 1999.0)] == [
        50.0,
        50.0,
        0.2,
        1000.0,
        1000.0,
    ]
