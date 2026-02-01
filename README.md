# matterlib k3d animation helper

Minimal, thread-free animation controls for k3d in JupyterLab 4.5+ using
`ipywidgets.Play`. Z is the vertical axis. Camera framing is stabilized by an
invisible bounds object.

## Quick start

```python
import numpy as np
import k3d
from matterlib import make_player

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


player = make_player(
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
from matterlib import make_dark_plot
plot = make_dark_plot(height=400)
```

## Stateful animation (wall-clock dt + target FPS)

Use `K3DAnimator` when you want the animation loop to advance with real time and
manage internal state (no precomputed frames). Exceptions are shown in the UI.
Prefer passing the plot to your animator's `__init__` and constructing it yourself.

```python
import k3d
from matterlib import K3DAnimator, make_stateful_player, make_dark_plot

g = 9.81

class FreeFall(K3DAnimator):
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
plot = make_dark_plot(height=600)
plot.display()

player = make_stateful_player(
    plot=plot,
    animator=FreeFall(plot, y0=y0, v0=v0),
    target_fps=60,
    camera_bounds=(-5, -5, 0, 5, 5, y0 + 1),
    title="Free fall (stateful)",
)

player.ui
```

## 2D canvas animation (ipycanvas + RepeatedTimer)

For 2D widgets, use `Canvas2DAnimator` + `Canvas2DPlayer`. The player follows the
`example_2d.py` pattern: a `RepeatedTimer` drives frames at `target_fps`, and parameter
widgets are declared via `ParamSpec`.

```python
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from matterlib import Canvas2DAnimator, Canvas2DPlayer, ParamSpec


class Pendulum2D(Canvas2DAnimator):
    PARAMS = {
        "L": ParamSpec("float_slider", default=1.2, min=0.2, max=3.0, step=0.01, description="L (m)", on_change="reset"),
        "g": ParamSpec("float_slider", default=9.81, min=0.1, max=30.0, step=0.01, description="g (m/s²)", on_change="reset"),
        "damp": ParamSpec("float_slider", default=0.02, min=0.0, max=2.0, step=0.01, description="damping", on_change="reset"),
    }

    def __init__(self) -> None:
        self.theta0 = 0.9
        self.omega0 = 0.0
        self.theta = self.theta0
        self.omega = self.omega0
        self.px_per_m = 140.0
        self.pivot = (320, 80)

    def on_start(self, canvas):
        self.canvas = canvas

    def on_reset(self):
        self.theta = self.theta0
        self.omega = self.omega0
        self._draw()

    def on_frame(self, dt: float):
        a = -(self.g / self.L) * np.sin(self.theta) - self.damp * self.omega
        self.omega += a * dt
        self.theta += self.omega * dt
        self._draw()

    def _draw(self):
        x = self.pivot[0] + (self.L * self.px_per_m) * np.sin(self.theta)
        y = self.pivot[1] + (self.L * self.px_per_m) * np.cos(self.theta)
        c = self.canvas
        c.fill_style = "#0b0e14"
        c.fill_rect(0, 0, c.width, c.height)
        c.stroke_style = "rgba(255,255,255,0.9)"
        c.line_width = 3
        c.stroke_line(self.pivot[0], self.pivot[1], x, y)
        c.fill_style = "#00ffcc"
        c.fill_circle(x, y, 16)


player = Canvas2DPlayer(animator=Pendulum2D(), target_fps=60, dt=1 / 240, title="Pendulum 2D")
display(widgets.HBox([player.ui, player.canvas]))
```

The player handles:
- Play/Pause/Step/Reset controls.
- Recreating the `RepeatedTimer` if `target_fps` changes.
- Parameter widgets that trigger `reset`/`redraw`/`restart_timer` based on `ParamSpec.on_change`.
- Drawing the first frame immediately (disable via `auto_draw_initial=False` if you prefer a blank canvas until Play).
- Canvas sizing: pass `width`/`height` or supply your own `Canvas` instance via `canvas=`.