"""Pure kinematics (no Manim) for constant-acceleration projectile motion.

Sign convention follows the equation as written:

    y(t) = 1/2 * g * t^2 + v0y * t + y0
    x(t) = v0x * t + x0

so ``g`` is the signed vertical acceleration: use g = -9.81 for Earth gravity
pointing down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Projectile:
    g: float = -9.81
    v0y: float = 12.0
    y0: float = 40.0
    v0x: float = 8.0
    x0: float = 0.0

    def __post_init__(self):
        if self.y0 < 0:
            raise ValueError(f"y0 must be >= 0 (the ground is y = 0), got {self.y0}")
        # Touch t_land so bad parameter combinations fail immediately.
        self.t_land

    # --- state at time t -------------------------------------------------
    def x(self, t: float) -> float:
        return self.x0 + self.v0x * t

    def y(self, t: float) -> float:
        return 0.5 * self.g * t**2 + self.v0y * t + self.y0

    def vx(self, t: float) -> float:
        return self.v0x

    def vy(self, t: float) -> float:
        return self.v0y + self.g * t

    def position(self, t: float) -> tuple[float, float]:
        return self.x(t), self.y(t)

    def velocity(self, t: float) -> tuple[float, float]:
        return self.vx(t), self.vy(t)

    def speed(self, t: float) -> float:
        return math.hypot(*self.velocity(t))

    def angle_deg(self, t: float) -> float:
        """Direction of the velocity, degrees above the +x axis (negative = below)."""
        return math.degrees(math.atan2(self.vy(t), self.vx(t)))

    # --- derived quantities ----------------------------------------------
    @property
    def t_land(self) -> float:
        """First time t > 0 at which y(t) = 0."""
        a, b, c = 0.5 * self.g, self.v0y, self.y0
        if abs(a) < 1e-12:
            roots = [-c / b] if abs(b) > 1e-12 else []
        else:
            disc = b * b - 4 * a * c
            roots = [] if disc < 0 else [(-b + s * math.sqrt(disc)) / (2 * a) for s in (1, -1)]
        positive = [r for r in roots if r > 1e-9]
        if not positive:
            hint = " (g > 0 means gravity points up; did you mean a negative g?)" if self.g > 0 else ""
            raise ValueError(f"The projectile never reaches y = 0 with {self}{hint}")
        return min(positive)

    @property
    def x_land(self) -> float:
        return self.x(self.t_land)

    @property
    def t_apex(self) -> float | None:
        """Time of maximum height, or None if the peak isn't during the flight."""
        if self.g == 0:
            return None
        t = -self.v0y / self.g
        return t if 0 < t < self.t_land else None

    @property
    def y_max(self) -> float:
        t = self.t_apex
        return self.y0 if t is None else max(self.y0, self.y(t))

    @property
    def max_speed(self) -> float:
        # |v| is convex in t, so the max over [0, t_land] is at an endpoint.
        return max(self.speed(0), self.speed(self.t_land))
