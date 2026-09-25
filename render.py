"""Render the cliff-throw animation with your own numbers.

Uses the sign convention of y(t) = 1/2 g t^2 + v0y t + y0, so gravity is g = -9.81.

    python render.py --g -9.81 --v0y 12 --v0x 8 --y0 40
    python render.py --v0y 20 --y0 15 -q h --preview
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

QUALITY = {"l": "low_quality", "m": "medium_quality", "h": "high_quality", "k": "fourk_quality"}


def ensure_latex_on_path():
    """MiKTeX installs per-user and new shells pick it up, but make sure this one does too."""
    if shutil.which("latex"):
        return
    miktex = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"
    if (miktex / "latex.exe").exists():
        os.environ["PATH"] = f"{miktex}{os.pathsep}{os.environ['PATH']}"
    else:
        sys.exit("LaTeX not found. Install MiKTeX (winget install MiKTeX.MiKTeX) and retry.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--g", type=float, default=-9.81, help="vertical acceleration, m/s^2 (default -9.81)")
    parser.add_argument("--v0y", type=float, default=12.0, help="initial vertical velocity, m/s (default 12)")
    parser.add_argument("--v0x", type=float, default=8.0, help="initial horizontal velocity, m/s (default 8)")
    parser.add_argument("--y0", type=float, default=40.0, help="cliff height, m (default 40)")
    parser.add_argument("--launch-vectors", action=argparse.BooleanOptionalAction, default=True,
                        help="combine the launch components v0x + v0y into v0 and its angle (default on)")
    parser.add_argument("--impact-vectors", action=argparse.BooleanOptionalAction, default=False,
                        help="show the vfx + vfy -> vf breakdown at impact (default off)")
    parser.add_argument("-q", "--quality", choices=QUALITY, default="m", help="l=480p m=720p h=1080p k=4K")
    parser.add_argument("-p", "--preview", action="store_true", help="open the video when done")
    parser.add_argument("-o", "--output", help="output file name (default built from the parameters)")
    args = parser.parse_args()

    from physics_anim.kinematics import Projectile

    try:
        projectile = Projectile(g=args.g, v0y=args.v0y, v0x=args.v0x, y0=args.y0)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    print(f"Flight time {projectile.t_land:.2f} s, lands at x = {projectile.x_land:.2f} m, "
          f"impact speed {projectile.speed(projectile.t_land):.2f} m/s")

    ensure_latex_on_path()
    from manim import tempconfig

    from scenes.cliff_throw import CliffThrow

    name = args.output or f"CliffThrow_g{args.g:g}_v0y{args.v0y:g}_v0x{args.v0x:g}_y0{args.y0:g}"
    with tempconfig({
        "quality": QUALITY[args.quality],
        "preview": args.preview,
        "output_file": name,
        "media_dir": str(ROOT / "media"),
    }):
        CliffThrow(projectile=projectile, show_launch_vectors=args.launch_vectors,
                   show_impact_vectors=args.impact_vectors).render()


if __name__ == "__main__":
    main()
