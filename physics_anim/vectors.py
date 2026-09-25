"""Reusable Manim helpers for drawing and combining physics vectors.

Vectors are given in physical units (e.g. m/s) and converted to screen units
with ``scale`` (screen units per physical unit), so arrows stay proportional
to each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manim import (
    RIGHT,
    WHITE,
    YELLOW,
    Angle,
    Arrow,
    Create,
    FadeIn,
    GrowArrow,
    Line,
    MathTex,
    VGroup,
    Write,
    config,
)

MIN_ARROW_LENGTH = 1e-3


@dataclass
class VectorSpec:
    vx: float
    vy: float
    color: str = WHITE
    label: str | None = None  # LaTeX, e.g. r"v_{0x} = 8.0\,\text{m/s}"

    @property
    def vec(self) -> np.ndarray:
        return np.array([self.vx, self.vy, 0.0])


def velocity_arrow(start, vx: float, vy: float, scale: float, color=WHITE, stroke_width: float = 6) -> Arrow | VGroup:
    """Arrow from ``start`` representing (vx, vy). Returns an empty VGroup for ~zero vectors."""
    start = np.asarray(start, dtype=float)
    delta = np.array([vx, vy, 0.0]) * scale
    if np.linalg.norm(delta) < MIN_ARROW_LENGTH:
        return VGroup()
    return Arrow(
        start,
        start + delta,
        buff=0,
        color=color,
        stroke_width=stroke_width,
        max_tip_length_to_length_ratio=0.3,
        max_stroke_width_to_length_ratio=12,
    )


def keep_in_frame(mob, margin: float = 0.15):
    """Shift ``mob`` just enough to be fully on screen."""
    half_w, half_h = config.frame_width / 2 - margin, config.frame_height / 2 - margin
    dx = max(0, -half_w - mob.get_left()[0]) - max(0, mob.get_right()[0] - half_w)
    dy = max(0, -half_h - mob.get_bottom()[1]) - max(0, mob.get_top()[1] - half_h)
    return mob.shift([dx, dy, 0])


def place_label(label, arrow, away_from=None, buff: float = 0.12):
    """Put ``label`` beside the middle of ``arrow``, on the side facing away from ``away_from``."""
    start, end = arrow.get_start(), arrow.get_end()
    d = end - start
    perp = np.array([d[1], -d[0], 0.0])
    norm = np.linalg.norm(perp)
    if norm < MIN_ARROW_LENGTH:
        return keep_in_frame(label.next_to(start, RIGHT, buff=buff))
    perp /= norm
    mid = (start + end) / 2
    if away_from is not None and np.dot(perp, mid - np.asarray(away_from)) < 0:
        perp = -perp
    return keep_in_frame(label.next_to(mid, perp, buff=buff))


def tex_label(tex: str, color=WHITE, font_size: float = 30) -> MathTex:
    """MathTex with a translucent dark backing so it stays readable over scenery."""
    label = MathTex(tex, color=color, font_size=font_size)
    label.add_background_rectangle(opacity=0.65, buff=0.06)
    return label


def make_vector(start, spec: VectorSpec, scale: float, away_from=None, font_size: float = 30) -> VGroup:
    """VGroup(arrow, label?) for ``spec`` with its tail at ``start``."""
    arrow = velocity_arrow(start, spec.vx, spec.vy, scale, color=spec.color)
    group = VGroup(arrow)
    if spec.label and isinstance(arrow, Arrow):
        label = tex_label(spec.label, spec.color, font_size)
        place_label(label, arrow, away_from)
        group.add(label)
    return group


def angle_marker(origin, vx: float, vy: float, scale: float, label: str | None = None,
                 radius: float = 0.6, color=WHITE, font_size: float = 28) -> VGroup:
    """Arc (with optional label) from the +x direction to the vector (vx, vy) at ``origin``."""
    origin = np.asarray(origin, dtype=float)
    tip = origin + np.array([vx, vy, 0.0]) * scale
    radius = min(radius, 0.6 * np.linalg.norm(tip - origin))
    arc = Angle(Line(origin, origin + RIGHT), Line(origin, tip), radius=radius,
                other_angle=vy < 0, color=color)
    group = VGroup(arc)
    if label:
        half = math.atan2(vy, vx) / 2
        text = tex_label(label, color, font_size)
        # Just outside the arc, on the bisector, pushed out far enough to clear the arc.
        direction = np.array([math.cos(half), math.sin(half), 0.0])
        text.next_to(origin + radius * direction, direction, buff=0.1)
        group.add(keep_in_frame(text))
    return group


def animate_tip_to_tail(scene, origin, components: list[VectorSpec], scale: float,
                        resultant_color=YELLOW, resultant_label: str | None = None,
                        angle_label: str | None = None, end_at=None, combine: bool = True,
                        grow_order: list[int] | None = None, after_each=None) -> VGroup:
    """Animate vector addition: grow each component from ``origin``, slide them
    tip-to-tail, then draw the resultant from ``origin`` to the final tip.

    If ``end_at`` is given, ``origin`` is ignored and the construction is placed
    so the resultant *ends* at ``end_at`` (handy for "velocity on impact").
    With ``combine=False`` it stops after growing the labelled components.

    Components grow one at a time, in ``grow_order`` (default: list order);
    ``after_each(i, component)`` is called right after component ``i`` appears.

    Returns VGroup(components..., resultant[, angle]) so the caller can fade it;
    each component is VGroup(arrow, label).
    """
    total = sum((c.vec for c in components), np.zeros(3))
    if end_at is not None:
        origin = np.asarray(end_at, dtype=float) - total * scale
    origin = np.asarray(origin, dtype=float)

    # Corners of the finished tip-to-tail polygon; labels go on the outside.
    corners = [origin]
    for c in components:
        corners.append(corners[-1] + c.vec * scale)
    centroid = np.mean(corners, axis=0)

    # Label each component on the side it will end up on after sliding, so labels
    # don't flip sides mid-animation (or start out on top of the y-axis numbers).
    vecs = [make_vector(origin, c, scale, away_from=centroid - (tail - origin))
            for c, tail in zip(components, corners)]
    for i in grow_order or range(len(vecs)):
        v = vecs[i]
        if isinstance(v[0], Arrow):
            scene.play(GrowArrow(v[0]), *[FadeIn(label) for label in v[1:]])
        if after_each:
            after_each(i, v)
    scene.wait(0.5)
    if not combine:
        return VGroup(*vecs)

    # Slide components so each tail sits on the previous tip.
    moves = []
    for v, tail in zip(vecs[1:], corners[1:-1]):
        if not isinstance(v[0], Arrow):
            continue
        shift = tail - v[0].get_start()
        moves.append(v[0].animate.shift(shift))
        if len(v) > 1:
            target = v[1].copy()
            place_label(target, v[0].copy().shift(shift), away_from=centroid)
            moves.append(v[1].animate.move_to(target))
    if moves:
        scene.play(*moves, run_time=1.5)
        scene.wait(0.3)

    resultant = make_vector(origin, VectorSpec(total[0], total[1], resultant_color, resultant_label),
                            scale, away_from=centroid)
    anims = [GrowArrow(resultant[0])] if isinstance(resultant[0], Arrow) else []
    if len(resultant) > 1:
        anims.append(Write(resultant[1]))
    if anims:
        scene.play(*anims, run_time=1.2)

    out = VGroup(*vecs, resultant)
    if angle_label and isinstance(resultant[0], Arrow):
        marker = angle_marker(origin, total[0], total[1], scale, angle_label)
        scene.play(Create(marker[0]), *[Write(m) for m in marker[1:]])
        out.add(marker)
    return out
