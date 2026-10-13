"""Timestamps for movie frames, in units that fit the movie.

The finest unit shown is the largest one the frame interval is a whole number
of (900 s per frame counts in minutes, 30 s in seconds); the coarsest is the
largest one the movie's last frame reaches. So a 15-minute interval over 30
frames ends at ``7hr:15min``, a 30-second interval over 60 frames at
``29min:30s``, and a movie shorter than a minute counts in plain seconds.
"""

from collections.abc import Callable

# Seconds per unit, by the names a time axis may carry.
SECONDS_PER = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}
# Displayed units, coarsest first.
LADDER = (("hr", 3600), ("min", 60), ("s", 1))
MAX_DECIMALS = 3


def _is_whole(value: float) -> bool:
    return abs(value - round(value)) < 1e-9 * max(1.0, abs(value))


def _seconds_decimals(dt: float) -> int:
    """Decimal places needed to show multiples of ``dt`` seconds exactly."""
    for decimals in range(MAX_DECIMALS + 1):
        if _is_whole(dt * 10**decimals):
            return decimals
    return MAX_DECIMALS


def time_formatter(
    time_axis: dict, n_frames: int, unit: str | None = None
) -> Callable[[int], str]:
    """A function from frame index to its timestamp label.

    Args:
        time_axis: The movie's time axis, with ``scale`` (time per frame) and
            ``unit``.
        n_frames: Frames in the whole movie. The format depends only on this,
            not on which frames are rendered, so every frame of one movie gets
            the same format.
        unit: Show the time in this one unit (s, min or h) instead.

    Returns:
        ``label(t)``. Frames are labeled by index when the axis unit is not a
        time unit this module knows.
    """
    axis_unit = time_axis.get("unit")
    per_unit = SECONDS_PER.get(str(axis_unit).lower()) if axis_unit else None
    if per_unit is None:
        if unit is not None:
            raise ValueError(
                f"time unit {unit!r} given, but the time axis unit is "
                f"{axis_unit!r}, which can't be converted"
            )
        return lambda t: f"frame {t}"
    dt = float(time_axis.get("scale", 1.0)) * per_unit  # seconds per frame

    if unit is not None:
        if unit.lower() not in SECONDS_PER:
            raise ValueError(f"Unknown time unit {unit!r}; use s, min or h")
        size = SECONDS_PER[unit.lower()]
        label = next(name for name, s in LADDER if s == size)
        return lambda t: f"{t * dt / size:g}{label}"

    # Finest unit: the largest the interval is a whole multiple of.
    finest = next(
        (
            i
            for i, (_, size) in enumerate(LADDER)
            if dt >= size and _is_whole(dt / size)
        ),
        len(LADDER) - 1,
    )
    decimals = _seconds_decimals(dt) if LADDER[finest][1] == 1 else 0
    # Coarsest unit: the largest the last frame reaches.
    final = (max(n_frames, 1) - 1) * dt
    coarsest = next((i for i, (_, size) in enumerate(LADDER) if final >= size), finest)
    coarsest = min(coarsest, finest)
    units = LADDER[coarsest : finest + 1]

    # Count in integer ticks of the finest displayed step, so carries are
    # exact (no "0min:60s" from 59.9999...).
    tick = LADDER[finest][1] / 10**decimals

    def label(t: int) -> str:
        remaining = round(t * dt / tick)
        parts = []
        for i, (name, size) in enumerate(units):
            per = round(size / tick)
            last = i == len(units) - 1
            if not last:
                value, remaining = divmod(remaining, per)
                text = str(value)
            else:  # the finest unit takes what is left, fractions included
                text = f"{remaining / per:.{decimals}f}"
            if i > 0:  # fixed width below the leading unit, e.g. 7hr:05min
                text = text.zfill(2 + (decimals + 1 if last and decimals else 0))
            parts.append(text + name)
        return ":".join(parts)

    return label
