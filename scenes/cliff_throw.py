"""A football thrown off a cliff, driven by y(t) = 1/2 g t^2 + v0y t + y0.

Sequence: write the general equations, then pull each known value out of the
picture one at a time (cliff height, launch velocity components, gravity from
how v_y changes), collect them under the equations, plug them in, and fly.

Render with custom numbers via the runner:
    python render.py --g -9.81 --v0y 12 --v0x 8 --y0 40

or with the defaults straight through Manim:
    manim -pql scenes/cliff_throw.py CliffThrow
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    PURPLE_A,
    RED,
    RIGHT,
    UL,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Axes,
    Brace,
    Create,
    DashedLine,
    DashedVMobject,
    DecimalNumber,
    DoubleArrow,
    FadeIn,
    FadeOut,
    Flash,
    GrowArrow,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    Line,
    MathTex,
    ReplacementTransform,
    Scene,
    SurroundingRectangle,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
    config,
    linear,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_anim.kinematics import Projectile  # noqa: E402
from physics_anim.props import cliff, football, nice_step  # noqa: E402
from physics_anim.vectors import (  # noqa: E402
    VectorSpec,
    animate_tip_to_tail,
    keep_in_frame,
    make_vector,
    velocity_arrow,
)

# Two layouts, picked by whichever draws the trajectory bigger:
#   wide: axes span the frame, equations + readout in a band across the top
#   tall: axes on the left, equations + readout in a column on the right
WIDE_AXES = (11.8, 4.5)  # max (width, height) in screen units
TALL_AXES = (8.3, 6.3)
DIAGRAM_ARROW = 2.4  # screen length of the resultant in the launch/impact diagrams
EQ_SCALE = 0.72  # equations' size once parked in the corner
EDGE = 0.35
X_COLOR, Y_COLOR, V_COLOR, G_COLOR, H_COLOR = RED, BLUE, YELLOW, GREEN, PURPLE_A

# Pieces of the equations, so individual symbols can be swapped for numbers.
Y_PARTS = ["y(t)", "=", r"\tfrac{1}{2}", "g", "t^2", "+", "v_{0y}", "t", "+", "y_0"]
X_PARTS = ["x(t)", "=", "v_{0x}", "t"]
# given -> (which equation, index of its symbol in that equation's parts)
SLOTS = {"y0": ("y", 9), "v0y": ("y", 6), "g": ("y", 3), "v0x": ("x", 2)}


def num(v: float) -> str:
    """Compact number for LaTeX: 12.0 -> '12', -9.810 -> '-9.81'."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def coef(v: float) -> str:
    """Number to substitute into an equation, parenthesised if negative."""
    return rf"\left({num(v)}\right)" if v < 0 else num(v)


def glyphs(mob):
    """``mob`` without its background rectangle, if it has one (for clean morphs)."""
    bg = getattr(mob, "background_rectangle", None)
    return VGroup(*[m for m in mob.submobjects if m is not bg]) if bg else mob


class CliffThrow(Scene):
    def __init__(self, projectile: Projectile | None = None, show_launch_vectors: bool = True,
                 show_impact_vectors: bool = False, **kwargs):
        self.p = projectile or Projectile()
        self.show_launch_vectors = show_launch_vectors
        self.show_impact_vectors = show_impact_vectors
        super().__init__(**kwargs)

    def construct(self):
        self.axes = self.make_axes()
        self.show_equations()
        self.build_world()
        self.measure_height()
        self.launch_vectors()
        self.derive_gravity()
        self.plug_in()
        self.fly()
        self.impact()
        self.wait(2)

    @property
    def velocity_scale(self) -> float:
        """Screen units per m/s for vectors drawn on the ball in flight."""
        p = self.p
        return min(DIAGRAM_ARROW / max(p.speed(0), 1e-6), 3.0 / max(p.max_speed, 1e-6))

    # --- equations and the list of given values ----------------------- #
    def show_equations(self):
        p = self.p
        self.eq = {"y": MathTex(*Y_PARTS), "x": MathTex(*X_PARTS)}
        eqs = VGroup(self.eq["y"], self.eq["x"]).arrange(DOWN, aligned_edge=LEFT, buff=0.35).scale(1.2)
        self.play(Write(self.eq["y"]))
        self.play(Write(self.eq["x"]))
        self.wait(0.5)

        # Park them top-right, leaving room for the wider substituted version.
        final = MathTex(*[coef(p.g) if s == "g" else coef(p.v0y) if s == "v_{0y}" else s
                          for s in Y_PARTS]).scale(EQ_SCALE)
        width = max(final.width, eqs.width / 1.2 * EQ_SCALE)
        left = config.frame_width / 2 - EDGE - width
        self.play(eqs.animate.scale(EQ_SCALE / 1.2)
                  .move_to([left, config.frame_height / 2 - EDGE, 0], aligned_edge=UL))
        self.equations = eqs
        self.givens = self.make_givens()

    def make_givens(self) -> dict[str, MathTex]:
        """Empty slots (not yet shown) under the equations for each known value."""
        p = self.p
        rows = {
            "y0": MathTex("y_0", "=", num(p.y0), r"\,\text{m}", color=H_COLOR, font_size=32),
            "v0y": MathTex("v_{0y}", "=", num(p.v0y), r"\,\text{m/s}", color=Y_COLOR, font_size=32),
            "v0x": MathTex("v_{0x}", "=", num(p.v0x), r"\,\text{m/s}", color=X_COLOR, font_size=32),
            "g": MathTex("g", "=", num(p.g), r"\,\text{m/s}^2", color=G_COLOR, font_size=32),
        }
        col = VGroup(*rows.values()).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        for row in col:  # line up the '=' signs
            row.shift(RIGHT * (col[0][1].get_x() - row[1].get_x()))
        col.next_to(self.equations, DOWN, buff=0.45, aligned_edge=LEFT)
        self.givens_col = col
        return rows

    def send_to_givens(self, key: str, source=None):
        """Morph ``source`` into the given's slot, then flash its symbol in the equation."""
        target = self.givens[key]
        if source is None:
            self.play(FadeIn(target, shift=RIGHT * 0.2))
        else:
            self.play(ReplacementTransform(glyphs(source).copy(), target), run_time=1.2)
        which, idx = SLOTS[key]
        self.play(Indicate(self.eq[which][idx], color=target.get_color()), Indicate(target[0]))

    # --- the world ------------------------------------------------------ #
    def make_axes(self) -> Axes:
        p = self.p
        xs = nice_step(p.x_land * 1.25)
        self.x_min = -xs * max(1, math.ceil(0.2 * p.x_land / xs))
        x_max = xs * math.ceil(p.x_land * 1.08 / xs)
        ys = nice_step(p.y_max * 1.15)
        y_max = ys * math.ceil(p.y_max * 1.12 / ys)
        x_span = x_max - self.x_min

        # Same screen scale on both axes so arrows point along the real path.
        units = {name: min(w / x_span, h / y_max) for name, (w, h) in
                 (("wide", WIDE_AXES), ("tall", TALL_AXES))}
        self.layout = max(units, key=units.get)
        unit = units[self.layout]

        axes = Axes(
            x_range=[self.x_min, x_max, xs], y_range=[0, y_max, ys],
            x_length=unit * x_span, y_length=unit * y_max,
            tips=False, axis_config={"color": GREY_B},
        )
        axes.to_edge(DOWN, buff=0.6).to_edge(LEFT, buff=0.7)
        axes.add_coordinates(np.arange(xs, x_max + xs / 2, xs), np.arange(ys, y_max + ys / 2, ys))
        return axes

    def build_world(self):
        p, axes = self.p, self.axes
        labels = axes.get_axis_labels(MathTex(r"x\ (\text{m})", font_size=30),
                                      MathTex(r"y\ (\text{m})", font_size=30))
        self.play(Create(axes), FadeIn(labels))

        if p.y0 > 0:
            # Behind the axes, so the y-axis ticks (at x = 0, the cliff edge) stay visible.
            rock = cliff(axes, self.x_min, p.y0).set_z_index(-1)
            self.play(FadeIn(rock, shift=UP * 0.3))
        self.ball = football().move_to(axes.c2p(0, p.y0))
        self.play(FadeIn(self.ball, scale=0.5))

    # --- pulling the givens out of the picture ------------------------- #
    def measure_height(self):
        p, axes = self.p, self.axes
        label_tex = rf"y_0 = {num(p.y0)}\,\text{{m}}"
        if p.y0 <= 0:
            label = MathTex(label_tex, color=H_COLOR, font_size=32).next_to(self.ball, UP)
            self.play(Write(label))
            self.send_to_givens("y0", label)
            self.play(FadeOut(label))
            return

        offset = RIGHT * 0.35  # just right of the cliff edge, clear of the y-axis numbers
        bar = DoubleArrow(axes.c2p(0, 0) + offset, axes.c2p(0, p.y0) + offset, buff=0,
                          color=H_COLOR, stroke_width=4, tip_length=0.2)
        label = MathTex(label_tex, color=H_COLOR, font_size=32).next_to(bar, RIGHT, buff=0.15)
        self.play(GrowFromCenter(bar))
        self.play(Write(label))
        self.wait(0.5)
        self.send_to_givens("y0", label)
        self.play(FadeOut(bar), FadeOut(label))

    def launch_vectors(self):
        """Grow v0y then v0x at the ball, sending each to the givens; optionally combine into v0."""
        p = self.p
        keys = {0: "v0x", 1: "v0y"}

        def send(i, component):
            self.wait(0.3)
            self.send_to_givens(keys[i], component[1] if len(component) > 1 else None)

        diagram = animate_tip_to_tail(
            self, self.axes.c2p(0, p.y0),
            [VectorSpec(p.v0x, 0, X_COLOR, rf"v_{{0x}} = {num(p.v0x)}"),
             VectorSpec(0, p.v0y, Y_COLOR, rf"v_{{0y}} = {num(p.v0y)}")],
            DIAGRAM_ARROW / max(p.speed(0), 1e-6),
            resultant_color=V_COLOR,
            resultant_label=rf"v_0 = {num(p.speed(0))}\,\text{{m/s}}",
            angle_label=rf"\theta = {p.angle_deg(0):.1f}^\circ",
            combine=self.show_launch_vectors,
            grow_order=[1, 0],
            after_each=send,
        )
        self.wait(1.5 if self.show_launch_vectors else 0.3)
        self.play(FadeOut(diagram))

    def derive_gravity(self):
        """Strobe the ball's v_y at equal time steps, line the arrows up on a v_y-t
        chart, and read g off as the (constant) change in v_y per step."""
        p, axes = self.p, self.axes

        # Equal time steps: the smallest 'nice' step giving at most 6 samples.
        dt = next(d for d in (0.25, 0.5, 1, 2, 5, 10, 20, 50, 100) if int(p.t_land / d + 1e-9) <= 5)
        ts = [k * dt for k in range(int(p.t_land / dt + 1e-9) + 1)]
        if len(ts) < 3:
            dt = p.t_land / 2
            ts = [0, dt, 2 * dt]
        vys = [p.vy(t) for t in ts]

        # Vectors on the (dashed, not yet flown) path. Shrink them all if a downward
        # one would poke far through the ground, ignoring samples that are almost
        # on the ground already (those just get trimmed to the bottom of the screen).
        scale = self.velocity_scale
        ground = axes.c2p(0, 0)[1]
        screen_bottom = -config.frame_height / 2 + 0.15
        for t, vy in zip(ts, vys):
            room = axes.c2p(*p.position(t))[1] - (ground - 0.3)
            if vy < 0 and room > 1.0:
                scale = min(scale, room / -vy)

        preview = DashedVMobject(axes.plot_parametric_curve(
            lambda s: np.array(p.position(s)), t_range=[0, p.t_land, p.t_land / 80],
            color=GREY_B, stroke_width=2), num_dashes=45)
        self.play(Create(preview), run_time=1.2)

        ghosts, path_arrows = VGroup(), []
        for t, vy in zip(ts, vys):
            pos = axes.c2p(*p.position(t))
            shown_vy = max(vy, -(pos[1] - screen_bottom) / scale)
            arrow = velocity_arrow(pos, 0, shown_vy, scale, Y_COLOR, stroke_width=5)
            anims = [GrowArrow(arrow)] if isinstance(arrow, Arrow) else []
            if t > 0:
                ghost = football().rotate(math.atan2(vy, p.vx(t))).move_to(pos).set_opacity(0.5)
                ghosts.add(ghost)
                anims.append(FadeIn(ghost))
            path_arrows.append(arrow)
            self.play(*anims, run_time=0.6)
        self.wait(0.5)

        # v_y vs t chart.
        vmin, vmax = min(0, *vys), max(0, *vys)
        pad = 0.1 * (vmax - vmin)
        chart = Axes(
            x_range=[0, ts[-1] + 0.7 * dt, dt], y_range=[vmin - pad, vmax + pad, nice_step(vmax - vmin)],
            x_length=3.2 if self.layout == "tall" else 2.6, y_length=2.2 if self.layout == "tall" else 1.7,
            tips=False, axis_config={"color": GREY_B, "include_ticks": False},
        )
        chart_labels = chart.get_axis_labels(MathTex("t", font_size=26), MathTex("v_y", font_size=26))
        chart_group = VGroup(chart, chart_labels)
        if self.layout == "tall":
            chart_group.next_to(self.givens_col, DOWN, buff=0.5, aligned_edge=LEFT)
        else:
            chart_group.to_corner(UL, buff=0.4)
        self.play(Create(chart), FadeIn(chart_labels))

        bars, moves = VGroup(), []
        for t, vy, src in zip(ts, vys, path_arrows):
            if isinstance(src, Arrow):
                bar = Arrow(chart.c2p(t, 0), chart.c2p(t, vy), buff=0, color=Y_COLOR, stroke_width=5,
                            max_tip_length_to_length_ratio=0.25)
                bars.add(bar)
                moves.append(ReplacementTransform(src.copy(), bar))
        self.play(LaggedStart(*moves, lag_ratio=0.3), run_time=2)

        # Every step down is the same size: that step per unit time is g.
        steps = VGroup()
        for k in range(len(ts) - 1):
            corner = chart.c2p(ts[k + 1], vys[k])
            steps.add(VGroup(
                DashedLine(chart.c2p(ts[k], vys[k]), corner, color=WHITE, stroke_width=2),
                Line(corner, chart.c2p(ts[k + 1], vys[k + 1]), color=G_COLOR, stroke_width=6),
            ))
        self.play(LaggedStart(*[Create(s) for s in steps], lag_ratio=0.4), run_time=1.5)
        # Label the last step: nothing sits to the right of it, and Δt goes on a
        # brace under the chart where no bars reach.
        run, rise = steps[-1]
        floor = vmin - pad
        dt_brace = Brace(Line(chart.c2p(ts[-2], floor), chart.c2p(ts[-1], floor)), DOWN, buff=0.05)
        dt_label = VGroup(dt_brace, MathTex(rf"\Delta t = {num(dt)}\,\text{{s}}", font_size=26)
                          .next_to(dt_brace, DOWN, buff=0.05))
        dv_label = MathTex(rf"\Delta v_y = {num(p.g * dt)}\,\text{{m/s}}", font_size=26,
                           color=G_COLOR).next_to(rise, RIGHT, buff=0.12)
        keep_in_frame(dv_label)
        self.play(FadeIn(dt_label), FadeIn(dv_label))
        self.play(LaggedStart(*[Indicate(s[1], color=G_COLOR) for s in steps], lag_ratio=0.2))

        formula = MathTex(
            r"g = \frac{\Delta v_y}{\Delta t}", "=",
            rf"\frac{{{num(p.g * dt)}\,\text{{m/s}}}}{{{num(dt)}\,\text{{s}}}}", "=",
            rf"{num(p.g)}\,\text{{m/s}}^2",
            font_size=30 if self.layout == "tall" else 26, color=G_COLOR,
        )
        if self.layout == "tall":
            formula.next_to(VGroup(chart_group, dt_label), DOWN, buff=0.3, aligned_edge=LEFT)
        else:
            formula.next_to(chart_group, RIGHT, buff=0.5)
        keep_in_frame(formula)
        self.play(Write(formula))
        self.wait(0.5)
        self.send_to_givens("g", formula)
        self.wait(0.3)

        self.play(FadeOut(VGroup(preview, ghosts, *[a for a in path_arrows if isinstance(a, Arrow)],
                                 chart_group, bars, steps, dt_label, dv_label, formula)))

    def plug_in(self):
        """Fly each given's number into the equation in place of its symbol."""
        parts = {"y": list(Y_PARTS), "x": list(X_PARTS)}
        p = self.p
        values = {"y0": p.y0, "v0y": p.v0y, "g": p.g, "v0x": p.v0x}
        for key in ("y0", "v0y", "g", "v0x"):
            which, idx = SLOTS[key]
            given = self.givens[key]
            old = self.eq[which]
            parts[which][idx] = coef(values[key])
            new = MathTex(*parts[which]).scale(EQ_SCALE).move_to(old, aligned_edge=LEFT)
            new[idx].set_color(given.get_color())

            self.play(Indicate(given), Indicate(old[idx], color=given.get_color()), run_time=0.8)
            self.play(
                *[ReplacementTransform(old[i], new[i]) for i in range(len(old)) if i != idx],
                FadeOut(old[idx]),
                ReplacementTransform(given[2].copy(), new[idx]),
                run_time=1.2,
            )
            self.remove(*new)
            self.add(new)
            self.eq[which] = new
        self.equations = VGroup(self.eq["y"], self.eq["x"])
        self.play(Indicate(self.equations, scale_factor=1.05, color=YELLOW))
        self.wait(0.5)

    # --- the throw ------------------------------------------------------ #
    def fly(self):
        p, axes = self.p, self.axes
        # One scale for the whole flight so you can see |v| change.
        scale = self.velocity_scale
        t = ValueTracker(0)
        self.t = t

        def ball_point():
            return axes.c2p(*p.position(t.get_value()))

        ball = always_redraw(lambda: football()
                             .rotate(math.atan2(p.vy(t.get_value()), p.vx(t.get_value())))
                             .move_to(ball_point()))
        path = always_redraw(lambda: axes.plot_parametric_curve(
            lambda s: np.array(p.position(s)),
            t_range=[0, max(t.get_value(), 1e-3), max(t.get_value(), 1e-3) / 60],
            color=YELLOW, stroke_width=3, stroke_opacity=0.7,
        ))

        def live_vector(vx_fn, vy_fn, color, label):
            def build():
                tt = t.get_value()
                vx, vy = vx_fn(tt), vy_fn(tt)
                start = ball_point()
                away = start + np.array([p.vx(tt), p.vy(tt), 0]) * scale / 2
                return make_vector(start, VectorSpec(vx, vy, color, label), scale, away_from=away, font_size=26)
            return always_redraw(build)

        vectors = VGroup(
            live_vector(p.vx, lambda tt: 0, X_COLOR, "v_x"),
            live_vector(lambda tt: 0, p.vy, Y_COLOR, "v_y"),
            live_vector(p.vx, p.vy, V_COLOR, r"\vec v"),
        )

        self.readout = self.make_readout(t)
        self.remove(self.ball)
        self.add(path, ball)
        self.play(FadeIn(vectors), FadeIn(self.readout))

        # Play the flight in (roughly) real time, pausing at the top if there is one.
        speed = max(1.0, p.t_land / 8)  # never longer than ~8 s on screen
        stops = [s for s in (p.t_apex, p.t_land) if s is not None]
        for stop in stops:
            self.play(t.animate.set_value(stop), run_time=max(0.5, (stop - t.get_value()) / speed),
                      rate_func=linear)
            if stop == p.t_apex:
                note = MathTex(r"v_y = 0", color=Y_COLOR, font_size=30).next_to(ball, UP, buff=0.3)
                self.play(FadeIn(note, shift=DOWN * 0.1), run_time=0.4)
                self.wait(0.6)
                self.play(FadeOut(note), run_time=0.3)

        ball.clear_updaters()
        path.clear_updaters()
        self.play(FadeOut(vectors), run_time=0.4)
        self.ball = ball

    def make_readout(self, t: ValueTracker) -> VGroup:
        p = self.p
        rows = VGroup()
        for sym, fn, unit in [
            ("t", lambda tt: tt, r"\text{s}"),
            ("x", p.x, r"\text{m}"),
            ("y", p.y, r"\text{m}"),
            ("v_x", p.vx, r"\text{m/s}"),
            ("v_y", p.vy, r"\text{m/s}"),
        ]:
            name = MathTex(f"{sym} =", font_size=30)
            value = DecimalNumber(fn(0), num_decimal_places=2, font_size=30, include_sign=False)
            unit_tex = MathTex(unit, font_size=30)
            row = VGroup(name, value, unit_tex).arrange(RIGHT, buff=0.15)
            value.add_updater(lambda m, fn=fn: m.set_value(fn(t.get_value())))
            unit_tex.add_updater(lambda m, v=value: m.next_to(v, RIGHT, buff=0.15))
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        # Line up the '=' signs.
        for row in rows:
            row.shift(RIGHT * (rows[0][0].get_right()[0] - row[0].get_right()[0]))
        if self.layout == "tall":
            return rows.next_to(self.givens_col, DOWN, buff=0.5, aligned_edge=LEFT)
        return rows.next_to(self.equations, LEFT, buff=0.8, aligned_edge=UP)

    def impact(self):
        p = self.p
        t_f = p.t_land
        hit = self.axes.c2p(p.x_land, 0)
        self.play(Flash(hit, color=WHITE, flash_radius=0.4, line_length=0.25))
        for mob in self.readout.get_family():
            mob.clear_updaters()
        self.play(Create(SurroundingRectangle(self.readout, color=YELLOW, buff=0.15)))
        if not self.show_impact_vectors:
            return

        animate_tip_to_tail(
            self, None,
            [VectorSpec(p.vx(t_f), 0, X_COLOR, rf"v_{{fx}} = {num(p.vx(t_f))}"),
             VectorSpec(0, p.vy(t_f), Y_COLOR, rf"v_{{fy}} = {num(p.vy(t_f))}")],
            DIAGRAM_ARROW / p.speed(t_f),
            resultant_color=V_COLOR,
            resultant_label=rf"v_f = {num(p.speed(t_f))}\,\text{{m/s}}",
            angle_label=rf"\theta_f = {p.angle_deg(t_f):.1f}^\circ",
            end_at=hit,
        )
