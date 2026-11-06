# manim_pipeline

Physics animations with [Manim Community](https://www.manim.community/) (v0.21).

## Setup (already done on this machine)

- `.venv/` uses Python 3.12 (Anaconda). Python 3.14 has no prebuilt wheels for Manim's OpenGL dependencies yet.
- LaTeX comes from MiKTeX (`winget install MiKTeX.MiKTeX`). It installs missing LaTeX packages on first use.
- ffmpeg is on PATH.

To recreate the venv run `python.exe -m venv .venv` then `.venv\Scripts\python -m pip install -r requirements.txt`.

## Cliff throw

A grenade thrown off a cliff, using

    y(t) = 1/2 g t^2 + v0y t + y0        x(t) = v0x t

`g` is signed, following the equation: use **g = -9.81** for gravity pointing down.

```powershell
.venv\Scripts\python render.py --g -9.81 --v0y 12 --v0x 8 --y0 40          # 720p
.venv\Scripts\python render.py --v0y 20 --y0 15 -q h -p                     # 1080p, open when done
.venv\Scripts\python render.py --v0y -5 --v0x 6 --y0 30 -o thrown_down      # custom file name
```

Vector breakdowns: `--launch-vectors` (combine `v0x` and `v0y` into `v0` and θ) is on by default. `--no-launch-vectors` still shows the two component arrows but doesn't combine them. `--impact-vectors` is off by default (turn it on to add the final-velocity breakdown).

`-q`: `l` 480p, `m` 720p (default), `h` 1080p, `k` 4K. Output goes to `media/videos/<quality>/`.

The animation does this:
1. Writes the general equations and parks them top-right.
2. Draws the axes and cliff (the ground is the x-axis). Both axes use the same screen scale, so vectors point along the real path.
3. Pulls each known value out of the picture, one at a time, into a list under the equations. The matching symbol in the equation flashes as each one arrives:
   - `y0`: a measuring arrow up the cliff.
   - `v0y`, then `v0x`: arrows grown at the ball. With `--launch-vectors`, they then slide tip to tail into `v0` and its angle θ.
   - `g`: ghost grenades at equal time steps along the upcoming path, each with its `v_y` arrow. The arrows collect into a `v_y` vs `t` chart where every step is the same `Δv_y`, which gives `g = Δv_y / Δt`.
4. Plugs each value into the equation in place of its symbol.
5. Flies the ball with live `v_x`, `v_y` and `v` vectors, a trail and a readout, pausing at the top (where `v_y = 0`).
6. At impact, flashes and boxes the final readout. With `--impact-vectors`, it also builds the same component diagram for the final velocity `v_f`.

## Reusable pieces (`physics_anim/`)

- `kinematics.Projectile`: pure physics. Gives `x(t)`, `y(t)`, `vx(t)`, `vy(t)`, `t_land`, `x_land`, `t_apex`, `y_max`, `speed(t)` and `angle_deg(t)`.
- `vectors.animate_tip_to_tail(scene, origin, [VectorSpec, ...], scale, ...)`: animates the addition of any number of vectors, not only x/y components. Also useful for forces and displacements. Pass `end_at=` to place the diagram so the resultant ends at a point.
- `vectors.make_vector`, `velocity_arrow`, `angle_marker`, `tex_label`: building blocks for arrows with labels.
- `props.grenade`, `props.cliff`, `props.nice_step`: scenery and tick spacing.

Note: rendering several scenes in parallel can make MiKTeX fail with a `dvisvgm` error. Render one at a time.
