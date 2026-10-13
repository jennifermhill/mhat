"""Frame timestamps: the units chosen, and exact carries between them.

A wrong unit or a botched carry gives a plausible label on every frame of a
video, so nothing downstream would notice.
"""

from mhat.visualization.timestamps import time_formatter


def test_units_follow_interval_and_movie_length():
    # Huh7: 900 s per frame, 30 frames, ending at 29 * 15 min = 7 h 15 min.
    huh7 = time_formatter({"unit": "second", "scale": 900.0}, n_frames=30)
    assert huh7(0) == "0hr:00min"
    assert huh7(29) == "7hr:15min"
    # 30 s per frame over 30 minutes: minutes and seconds.
    half_min = time_formatter({"unit": "second", "scale": 30.0}, n_frames=61)
    assert half_min(3) == "1min:30s"
    assert half_min(60) == "30min:00s"
    # Short movies keep the interval's unit alone.
    assert time_formatter({"unit": "second", "scale": 1.0}, 12)(11) == "11s"
    assert time_formatter({"unit": "minute", "scale": 5.0}, 12)(11) == "55min"


def test_fractional_interval_carries_exactly():
    # 600 * 0.1 s is 60.000000000000007 in floating point.
    label = time_formatter({"unit": "second", "scale": 0.1}, n_frames=1000)
    assert label(599) == "0min:59.9s"
    assert label(600) == "1min:00.0s"
