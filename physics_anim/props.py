"""Scenery and objects: football, cliff, ground, plus axis-scaling helpers."""

from __future__ import annotations

import math

import numpy as np
from manim import WHITE, Ellipse, Line, Polygon, VGroup

FOOTBALL_BROWN = "#8B4513"
ROCK = "#6B5B4B"


def football(length: float = 0.45) -> VGroup:
    """A football pointing along +x (rotate it to match the velocity)."""
    body = Ellipse(width=length, height=0.58 * length, color=FOOTBALL_BROWN,
                   fill_color=FOOTBALL_BROWN, fill_opacity=1, stroke_width=2)
    seam = Line(0.28 * length * np.array([-1, 0, 0]), 0.28 * length * np.array([1, 0, 0]),
                color=WHITE, stroke_width=2)
    laces = VGroup(*[
        Line([x, -0.07 * length, 0], [x, 0.07 * length, 0], color=WHITE, stroke_width=2)
        for x in np.linspace(-0.18 * length, 0.18 * length, 4)
    ])
    return VGroup(body, seam, laces)


def cliff(axes, x_left: float, y_top: float, jaggedness: float = 0.04, seed: int = 3) -> Polygon:
    """Rock face from x_left..0 (axis coords) up to y_top, with a rough right-hand face."""
    rng = np.random.default_rng(seed)
    width = -x_left
    face = [(width * jaggedness * rng.uniform(-1, 1), y)
            for y in np.linspace(y_top * 0.9, y_top * 0.1, 6)]
    pts = [(x_left, 0), (x_left, y_top), (0, y_top), *face, (0, 0)]
    return Polygon(*[axes.c2p(x, y) for x, y in pts], color=ROCK, fill_color=ROCK,
                   fill_opacity=1, stroke_width=2)


def nice_step(span: float, target_ticks: int = 6) -> float:
    """A 1/2/5 x 10^n tick spacing giving roughly ``target_ticks`` ticks over ``span``."""
    raw = span / target_ticks
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag
