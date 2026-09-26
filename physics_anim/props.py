"""Scenery and objects: grenade, cliff, ground, plus axis-scaling helpers."""

from __future__ import annotations

import math

import numpy as np
from manim import GRAY_B, GREEN_D, LEFT, ORIGIN, PI, RIGHT, UP, UR, Circle, Ellipse, Line, Polygon, Rectangle, VGroup

GRENADE_GREEN = GREEN_D
GROOVE = "#2E3F14"
METAL = GRAY_B
ROCK = "#6B5B4B"


def grenade(pos=ORIGIN, angle: float = 0.0, size: float = 0.45, opacity: float = 1.0) -> VGroup:
    """A pineapple grenade centred on its body at ``pos``, fuze pointing ``angle`` radians from +x."""
    body = Ellipse(width=0.8, height=1.0, fill_color=GRENADE_GREEN, fill_opacity=opacity,
                   stroke_color=GROOVE, stroke_width=3, stroke_opacity=opacity)
    grooves = VGroup(
        *[Line([-0.4 * math.sqrt(1 - (y / 0.5) ** 2), y, 0], [0.4 * math.sqrt(1 - (y / 0.5) ** 2), y, 0])
          for y in (-0.25, 0.0, 0.25)],
        Ellipse(width=0.4, height=1.0),
        Line([0, -0.5, 0], [0, 0.5, 0]),
    ).set_stroke(GROOVE, width=3, opacity=opacity)
    neck = Rectangle(width=0.3, height=0.18, fill_color=METAL, fill_opacity=opacity,
                     stroke_color=METAL, stroke_opacity=opacity).next_to(body, UP, buff=-0.03)
    lever = Line(neck.get_corner(UR), body.get_right() + 0.06 * RIGHT + 0.15 * UP,
                 color=METAL, stroke_width=4, stroke_opacity=opacity)
    ring = Circle(radius=0.11, stroke_color=METAL, stroke_width=3,
                  stroke_opacity=opacity).next_to(neck, LEFT, buff=0.02)
    # Built upright with the body at the origin, so rotating about the origin keeps pos on the body.
    return (VGroup(body, grooves, neck, lever, ring)
            .scale(size, about_point=ORIGIN)
            .rotate(angle - PI / 2, about_point=ORIGIN)
            .shift(pos))


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
