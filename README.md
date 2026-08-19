# matterlib: Theory of matter helper code

Python code to assist in THeory of Matter course

Includes

- `symbolic` module for easily declaring and working with equations
- `anim2d` module for ipycanvas animations
- `anim3d` module for k3d animations

The precomputed 3D player is thread-free and uses `ipywidgets.Play`. Stateful
3D, chunked 3D, and 2D animations use background threads. Z is the vertical
axis in 3D. Camera framing is stabilized by an invisible bounds object.

## 3D Animation Example 

```python
import numpy as np
import k3d
from matterlib import anim3d

y0 = 10.0
dt = 0.01
nsteps = 300
t = np.arange(nsteps) * dt
z = y0 - 0.5 * 9.81 * t**2

plot = k3d.plot(
    height=400,
    background_color=0x1E1E1E,
    grid_color=0xD2D2D2,
    label_color=0xF0F0F0,
    grid_visible=True,
    axes=["x", "y", "z"],
)

ball = k3d.points(positions=[[0, 0, z[0]]], point_size=0.9, color=0xFF5500)
plot += ball
plot.display()


def render(i: int) -> None:
    ball.positions = [[0, 0, z[i]]]


player = anim3d.make_player(
    plot=plot,
    frames={"z": z, "t": t},
    render_fn=render,
    dt=dt,
    camera_bounds=(-5, -5, 0, 5, 5, y0),
    title="Free fall",
    loop=True,
)

player.ui  # display the controls beneath the plot
```

Dark-theme helper:
```python
from matterlib import anim3d
plot = anim3d.make_dark_plot(height=400)
```

## Stateful animation (wall-clock dt + target FPS)

Use `anim3d.K3DAnimator` when you want the animation loop to advance with real time and
manage internal state (no precomputed frames). Exceptions are shown in the UI.
Prefer passing the plot to your animator's `__init__` and constructing it yourself.

```python
import k3d
from matterlib import anim3d

g = 9.81

class FreeFall(anim3d.K3DAnimator):
    def __init__(self, plot: k3d.plot, y0: float, v0: float) -> None:
        self.y = y0
        self.v = v0
        self._plot = plot
        # Create + attach once during __init__ (plot provided by animator_factory).
        self.ball = k3d.points(positions=[[0, 0, self.y]], point_size=0.9, color=0xFF5500)
        plot += self.ball

    def on_start(self, plot: k3d.plot) -> None:
        # Keep this cheap; objects are already attached in __init__.
        return

    def on_update(self, dt: float) -> None:
        self.v -= g * dt
        self.y += self.v * dt
        self.ball.positions = [[0, 0, self.y]]

    def on_reset(self) -> None:
        self.y, self.v = y0, v0
        self.ball.positions = [[0, 0, self.y]]


y0, v0 = 10.0, 0.0
plot = anim3d.make_dark_plot(height=600)
plot.display()

player = anim3d.make_stateful_player(
    plot=plot,
    animator=FreeFall(plot, y0=y0, v0=v0),
    target_fps=60,
    camera_bounds=(-5, -5, 0, 5, 5, y0 + 1),
    title="Free fall (stateful)",
)

player.ui
```

## 2D canvas example

For 2D widgets, use `anim2d.Canvas2DAnimator` + `anim2d.Canvas2DPlayer`. A
`anim2d.RepeatedTimer` drives frames at `target_fps`, and parameter widgets are
declared via `anim2d.ParamSpec`.

### Instructions

Inherit from `anim2d.Canvas2DAnimator` and implement
 - on_start
 - on_update
 - on_draw
 - on_reset

It is helpful to call `with ipycanvas.hold_canvas` context manager during the draw call.

The player handles:
- Play/Pause/Step/Reset controls.
- Recreating the `anim2d.RepeatedTimer` if `target_fps` changes.
- Parameter widgets that trigger `reset`/`redraw`/`restart_timer` based on `anim2d.ParamSpec.on_change`.
- Drawing the first frame immediately (disable via `auto_draw_initial=False` if you prefer a blank canvas until Play).
- Canvas sizing: pass `width`/`height` or supply your own `Canvas` instance via `canvas=`.

```python
from ipycanvas import hold_canvas
from matterlib import anim2d
import ipywidgets as widgets
from IPython.display import display

class EulerOrbitAnimator(anim2d.Canvas2DAnimator):
    # Expose key simulation controls as widgets in the player UI.
    PARAMS = {
        "method": anim2d.ParamSpec(
            kind="dropdown",
            default="euler",
            options=[
                ("Euler", "euler"),
                ("Euler-Cromer", "euler_cromer"),
                ("Velocity Verlet", "verlet"),
                ("RK4", "rk4"),
            ],
            description="method",
            on_change="reset",
        ),
        "dt_scale": anim2d.ParamSpec(
            kind="int_slider",
            default=200,
            min=1,
            max=600,
            step=1,
            description="substeps",
            on_change="reset",
        ),
    }

    def __init__(self, x0, y0, vx0, vy0, dt_scale: float = 1.0, method: str = "euler") -> None:
        self.initial = {'x0': x0, 'y0': y0, 'vx0': vx0, 'vy0': vy0}
        self.dt_scale = float(dt_scale)
        self.method = str(method).lower().strip()
        self.t = 0.0
        self.x = float(x0)
        self.y = float(y0)
        self.vx = float(vx0)
        self.vy = float(vy0)
        self.trail: list[tuple[float, float]] = [(self.x, self.y)]

    def on_reset(self):
        self.t = 0.0
        self.x = float(self.initial['x0'])
        self.y = float(self.initial['y0'])
        self.vx = float(self.initial['vx0'])
        self.vy = float(self.initial['vy0'])
        self.trail = [(self.x, self.y)]
    
    def on_start(self, canvas):
        self.canvas = canvas

        # --- world (AU) -> pixel transform ---
        # Keep the origin (sun) centered and scale so r≈1 AU is visible.
        self.world_half_width = 1.6  # AU visible from center to edge
        self.cx = canvas.width / 2
        self.cy = canvas.height / 2
        self.scale = min(canvas.width, canvas.height) / (2 * self.world_half_width)

    def _to_canvas(self, x: float, y: float) -> tuple[float, float]:
        # +x right, +y up in world coords
        return (self.cx + x * self.scale, self.cy - y * self.scale)

    def on_update(self, dt: float):
        # Treat dt_scale as "substeps per update": take n smaller Euler steps
        # whose total time equals the provided dt.
        n = max(1, int(round(self.dt_scale)))
        dt_step = dt / n

        def _accel(x: float, y: float) -> tuple[float, float]:
            r = (x * x + y * y) ** 0.5
            return (-GM * x / r**3, -GM * y / r**3)

        def _f(state: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            x, y, vx, vy = state
            ax, ay = _accel(x, y)
            return (vx, vy, ax, ay)

        for _ in range(n):
            ax, ay = _accel(self.x, self.y)

            if self.method == "verlet":
                # Velocity Verlet
                x_new = self.x + self.vx * dt_step + 0.5 * ax * dt_step**2
                y_new = self.y + self.vy * dt_step + 0.5 * ay * dt_step**2

                ax_new, ay_new = _accel(x_new, y_new)
                vx_new = self.vx + 0.5 * (ax + ax_new) * dt_step
                vy_new = self.vy + 0.5 * (ay + ay_new) * dt_step

                self.x, self.y, self.vx, self.vy = x_new, y_new, vx_new, vy_new

            elif self.method == "euler_cromer":
                # Euler-Cromer: update v, then x using updated v
                vx_new = self.vx + ax * dt_step
                vy_new = self.vy + ay * dt_step
                x_new = self.x + vx_new * dt_step
                y_new = self.y + vy_new * dt_step
                self.x, self.y, self.vx, self.vy = x_new, y_new, vx_new, vy_new

            elif self.method == "rk4":
                # RK4 on state = (x, y, vx, vy)
                s0 = (self.x, self.y, self.vx, self.vy)
                k1 = _f(s0)
                s1 = (
                    s0[0] + 0.5 * dt_step * k1[0],
                    s0[1] + 0.5 * dt_step * k1[1],
                    s0[2] + 0.5 * dt_step * k1[2],
                    s0[3] + 0.5 * dt_step * k1[3],
                )
                k2 = _f(s1)
                s2 = (
                    s0[0] + 0.5 * dt_step * k2[0],
                    s0[1] + 0.5 * dt_step * k2[1],
                    s0[2] + 0.5 * dt_step * k2[2],
                    s0[3] + 0.5 * dt_step * k2[3],
                )
                k3 = _f(s2)
                s3 = (
                    s0[0] + dt_step * k3[0],
                    s0[1] + dt_step * k3[1],
                    s0[2] + dt_step * k3[2],
                    s0[3] + dt_step * k3[3],
                )
                k4 = _f(s3)

                self.x = s0[0] + (dt_step / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                self.y = s0[1] + (dt_step / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                self.vx = s0[2] + (dt_step / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
                self.vy = s0[3] + (dt_step / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

            else:
                # Forward Euler (world units)
                self.x = self.x + self.vx * dt_step
                self.y = self.y + self.vy * dt_step
                self.vx = self.vx + ax * dt_step
                self.vy = self.vy + ay * dt_step

        # Record ONE point per rendered frame (otherwise you add hundreds of
        # nearly-identical points and can visually "catch" partial-looking states).
        self.trail.append((self.x, self.y))

    def on_draw(self):
        c = self.canvas
        with hold_canvas(c):
            c.fill_style = "#0b0e14"
            c.fill_rect(0, 0, c.width, c.height)

            # Trail (convert world -> pixels)
            if len(self.trail) >= 2:
                c.stroke_style = "rgba(255,255,255,0.35)"
                c.line_width = 2
                c.begin_path()
                x0c, y0c = self._to_canvas(*self.trail[0])
                c.move_to(x0c, y0c)
                stride = 2 if len(self.trail) > 2000 else 1
                for (xw, yw) in self.trail[1::stride]:
                    xc, yc = self._to_canvas(xw, yw)
                    c.line_to(xc, yc)
                c.stroke()

            # Particle (convert world -> pixels)
            px, py = self._to_canvas(self.x, self.y)
            c.fill_style = "#00ffcc"
            c.fill_circle(px, py, 6)
            
# Circular-orbit initial conditions at r=1 AU require vy0 = sqrt(GM/r) = 2π.
# Use lots of substeps because forward Euler is unstable for orbits at large dt.
anim = EulerOrbitAnimator(x0, y0, vx0, vy0, dt_scale=200, method="verlet")
player = anim2d.Canvas2DPlayer(animator=anim, target_fps=30, dt=1 / 60, title="Orbit (circular IC)")
player.canvas.layout = widgets.Layout(
    width="640px",
    height="420px",
    flex="0 0 auto",
)
display(
    widgets.HBox(
        [
            widgets.VBox([player.ui, widgets.HBox([player.canvas])])
        ],
        layout=widgets.Layout(align_items="flex-start", gap="12px"),
    )
)
```