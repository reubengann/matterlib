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