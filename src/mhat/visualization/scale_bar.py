"""A scale bar drawn onto a rendered frame.

napari's own scale bar can't be made thicker, so the renders draw theirs onto
each captured frame instead: one horizontal line, without end ticks, with its
length written above it. Nothing here knows about napari; the caller converts
the view into frame pixels.
"""

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import pint
from matplotlib import font_manager
from PIL import Image, ImageColor, ImageDraw, ImageFont

POSITIONS = ("bottom_right", "bottom_left", "top_right", "top_left")
# Space between the text's baseline and the bar, as a fraction of the cap
# height; enough to keep a descender (the µ of µm) off the bar.
TEXT_GAP = 0.5


def nice_length(max_length: float) -> float:
    """The largest 1, 2 or 5 times a power of ten that is at most
    ``max_length``.
    """
    if not max_length > 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    power = 10.0 ** math.floor(math.log10(max_length))
    # The tolerance keeps float noise in log10 from dropping a whole step.
    return next(m * power for m in (5, 2, 1) if m * power <= max_length * (1 + 1e-9))


def unit_symbol(unit: str | None) -> str:
    """The abbreviated unit (``micrometer`` -> ``µm``), as given if pint
    doesn't know it, or "" for no unit.
    """
    if not unit:
        return ""
    registry = pint.get_application_registry()
    try:
        parsed = registry.Unit(unit)
    except (pint.UndefinedUnitError, ValueError):
        return str(unit)
    if parsed == registry.micron:  # pint's symbol for it is a bare µ
        parsed = registry.micrometer
    return f"{parsed:~P}"


@cache
def _font_file(family: str) -> str:
    # Falls back to matplotlib's default font, with a warning, if missing.
    return font_manager.findfont(font_manager.FontProperties(family=family))


@cache
def load_font(family: str, cap_height: float) -> ImageFont.FreeTypeFont:
    """``family`` at the size whose digits are ``cap_height`` pixels tall."""
    path = _font_file(family)
    left, top, right, bottom = ImageFont.truetype(path, 1000).getbbox("0123456789")
    return ImageFont.truetype(path, max(1, round(cap_height * 1000 / (bottom - top))))


@dataclass(frozen=True)
class ScaleBar:
    """A scale bar in frame pixels.

    Attributes:
        length_px: Length of the bar.
        text: What is written above it, e.g. ``"50 µm"``.
        thickness: Height of the bar.
        cap_height: Height of the text's digits.
        margin: Distance from the frame's edges to the bar (bottom) or to the
            top of the text (top).
        color: A color name, ``#rrggbb``, or an RGB(A) sequence of 0-255.
        position: One of ``POSITIONS``.
        font: A font family that matplotlib can find.
    """

    length_px: float
    text: str
    thickness: float
    cap_height: float
    margin: float
    color: str | tuple = "white"
    position: str = "bottom_right"
    font: str = "Arial"

    def __post_init__(self):
        if self.position not in POSITIONS:
            raise ValueError(
                f"scale bar position must be one of {POSITIONS}, "
                f"got {self.position!r}"
            )

    def bar_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` of the bar in a frame of this size, ends
        exclusive. The same whether or not the text is drawn, so frames with
        and without it line up.
        """
        thickness = max(1, round(self.thickness))
        if self.position.startswith("bottom"):
            y1 = round(height - self.margin)
            y0 = y1 - thickness
        else:  # the text sits between the top margin and the bar
            y0 = round(self.margin + (1 + TEXT_GAP) * self.cap_height)
            y1 = y0 + thickness
        if self.position.endswith("right"):
            x1 = round(width - self.margin)
            x0 = x1 - round(self.length_px)
        else:
            x0 = round(self.margin)
            x1 = x0 + round(self.length_px)
        return x0, y0, x1, y1

    def draw(self, frame: np.ndarray, text: bool = True) -> np.ndarray:
        """A copy of ``frame`` (RGB or RGBA uint8) with the bar drawn on, and
        the text above it if ``text``.
        """
        image = Image.fromarray(frame)
        height, width = frame.shape[:2]
        color = (
            ImageColor.getrgb(self.color)
            if isinstance(self.color, str)
            else tuple(int(c) for c in self.color)
        )
        if image.mode == "RGBA" and len(color) == 3:
            color = (*color, 255)
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = self.bar_box(height, width)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color)  # PIL: inclusive
        if text and self.text:
            font = load_font(self.font, self.cap_height)
            # Centered over the bar, but kept inside the margins when the text
            # is wider than the bar.
            half = draw.textlength(self.text, font=font) / 2
            center = (x0 + x1) / 2
            center = min(max(center, self.margin + half), width - self.margin - half)
            baseline = y0 - TEXT_GAP * self.cap_height
            draw.text((center, baseline), self.text, fill=color, font=font, anchor="ms")
        return np.asarray(image)
