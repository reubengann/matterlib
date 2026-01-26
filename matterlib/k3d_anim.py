from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, cast
from abc import ABC, abstractmethod
import html
import threading
import time
import traceback

import ipywidgets as widgets
import k3d
import numpy as np

# k3d publishes runtime types but lacks accurate type stubs; fall back to Any so
# static checking does not mis-classify the factory functions/modules.
K3DPlot = Any
K3DLine = Any


Bounds = Tuple[float, float, float, float, float, float]


@dataclass
class Player:
    """Lightweight handle returned by make_player."""

    ui: widgets.Widget
    play: widgets.Play
    slider: widgets.IntSlider
    reset_button: widgets.Button
    title_widget: widgets.HTML
    extent_obj: Any
    plot: K3DPlot

    def reset(self) -> None:
        """Return to the first frame and pause."""
        self.play.playing = False
        self.play.value = self.play.min

    def close(self) -> None:
        """Detach callbacks and remove extent geometry from the plot if present."""
        self.play.unobserve_all()
        try:
            widgets.jslink.unlink(self.play, self.slider)
        except Exception:
            pass
        if self.extent_obj is not None:
            try:
                self.plot -= self.extent_obj
            except Exception:
                pass


class K3DAnimator(ABC):
    """Stateful animation interface driven by wall-clock dt."""

    @abstractmethod
    def on_start(self, plot: K3DPlot) -> None:
        """Create k3d objects and initialize state.

        Should be idempotent: safe to call after a reset. If you add objects to
        the plot here, remove/recreate them in `on_reset` or guard against
        double-adding. If you accept `plot` in __init__ (recommended via
        animator construction), you can attach objects there; keep on_start cheap.
        """

    @abstractmethod
    def on_update(self, dt: float) -> None:
        """Advance simulation by wall-clock dt seconds and mutate k3d objects."""

    def on_reset(self) -> None:
        """Optional: restore state and refresh plot objects; default is a no-op."""
        return


@dataclass
class StatefulPlayer:
    """Handle returned by make_stateful_player (Play/Reset, status + error UI)."""

    ui: widgets.Widget
    play_toggle: widgets.ToggleButton
    reset_button: widgets.Button
    status_widget: widgets.HTML
    error_banner: widgets.HTML
    traceback_accordion: widgets.Accordion
    extent_obj: Any
    plot: K3DPlot
    _stop_event: threading.Event
    _thread: Optional[threading.Thread]
    _lock: threading.Lock
    _started: bool
    _reset_callback: Optional[Callable[[], None]] = None
    _stop_callback: Optional[Callable[[], None]] = None

    def reset(self) -> None:
        """Stop playback and reset state."""
        if self._reset_callback is not None:
            self._reset_callback()

    def close(self) -> None:
        """Stop thread and remove extent geometry."""
        try:
            self.play_toggle.unobserve_all()
        except Exception:
            pass
        try:
            self.reset_button.on_click(lambda _: None)
        except Exception:
            pass
        if self._stop_callback is not None:
            self._stop_callback()
        else:
            self._stop_event.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        if self.extent_obj is not None:
            try:
                self.plot -= self.extent_obj
            except Exception:
                pass


def _validate_frames(frames: Dict[str, Sequence[Any]]) -> int:
    if not frames:
        raise ValueError("frames must be a non-empty dict of equally long sequences")
    lengths = {k: len(v) for k, v in frames.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        msg = ", ".join(f"{k}: {lengths[k]}" for k in sorted(lengths))
        raise ValueError(f"All frame sequences must share the same length; got {msg}")
    n = unique_lengths.pop()
    if n == 0:
        raise ValueError("frames must contain at least one timestep")
    return n


def _make_extent(camera_bounds: Bounds) -> K3DLine:
    xmin, ymin, zmin, xmax, ymax, zmax = camera_bounds
    if not (xmin < xmax and ymin < ymax and zmin < zmax):
        raise ValueError("camera_bounds must satisfy xmin<xmax, ymin<ymax, zmin<zmax")

    corners = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    vertices: list[tuple[float, float, float]] = []
    indices: list[tuple[int, int]] = []
    for a, b in edges:
        start_idx = len(vertices)
        vertices.append(corners[a])
        vertices.append(corners[b])
        indices.append((start_idx, start_idx + 1))

    # Zero-opacity still contributes to bounds but is visually invisible.
    return k3d.line(
        vertices=vertices,
        indices=indices,
        color=0x000000,
        width=0.0001,
        opacity=0.0,
        shader="simple",
    )


def make_dark_plot(**kwargs: Any) -> K3DPlot:
    """Convenience: create a k3d plot with dark-theme defaults.

    Users can override any default by passing the same keyword in kwargs.
    """

    defaults = dict(
        height=600,
        background_color=0x1E1E1E,
        grid_color=0xD2D2D2,
        label_color=0xF0F0F0,
        grid_visible=True,
        axes=["x", "y", "z"],
    )
    defaults.update(kwargs)
    return cast(Callable[..., K3DPlot], k3d.plot)(**defaults)


def make_player(
    *,
    plot: K3DPlot,
    frames: Dict[str, Sequence[Any]],
    render_fn: Callable[[int], None],
    dt: float,
    camera_bounds: Bounds,
    title: str = "Animation",
    loop: bool = True,
) -> Player:
    """Create a play/pause/reset controller for k3d animations (no threads).

    Parameters
    ----------
    plot:
        Existing k3d.plot instance.
    frames:
        Dict of equally long sequences (e.g., {"z": z, "t": t}). Used only to
        define frame count; data is already precomputed.
    render_fn:
        Function called with frame index i; should mutate k3d objects
        (e.g., `ball.positions = [...]`).
    dt:
        Timestep in seconds between frames; controls Play interval (ms).
    camera_bounds:
        (xmin, ymin, zmin, xmax, ymax, zmax) used to add invisible extent
        geometry so camera auto-fit stays stable.
    title:
        Label displayed above the controls.
    loop:
        If True, wrap to frame 0 after the last frame while continuing playback.
    """

    nframes = _validate_frames(frames)
    interval_ms = max(1, int(round(dt * 1000)))

    play = widgets.Play(
        value=0,
        min=0,
        max=nframes - 1,
        step=1,
        interval=interval_ms,
        description="Play",
    )
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=nframes - 1,
        step=1,
        description="Frame",
        readout=True,
        continuous_update=False,
        style={"description_width": "60px"},
    )
    widgets.jslink((play, "value"), (slider, "value"))

    reset_button = widgets.Button(description="Reset", icon="refresh")
    title_widget = widgets.HTML(f"<b>{title}</b>")

    extent_obj = _make_extent(camera_bounds)
    plot += extent_obj

    def do_reset(stop_playback: bool = True) -> None:
        if stop_playback:
            play.playing = False
        play.value = play.min

    def on_value(change: dict) -> None:
        if change.get("name") != "value":
            return
        i = int(change["new"])
        render_fn(i)
        if loop and i >= play.max:
            playing_state = bool(getattr(play, "playing", False))
            play.unobserve(on_value, names="value")
            try:
                play.value = play.min
                render_fn(play.min)
                if playing_state:
                    play.playing = True
            finally:
                play.observe(on_value, names="value")

    play.observe(on_value, names="value")
    reset_button.on_click(lambda _: do_reset(stop_playback=True))

    controls = widgets.HBox([play, slider, reset_button])
    ui = widgets.VBox([title_widget, controls])

    return Player(
        ui=ui,
        play=play,
        slider=slider,
        reset_button=reset_button,
        title_widget=title_widget,
        extent_obj=extent_obj,
        plot=plot,
    )


def _make_error_widgets() -> tuple[widgets.HTML, widgets.Accordion, widgets.Output]:
    error_banner = widgets.HTML("")
    trace_output = widgets.Output(
        layout={
            "max_height": "160px",
            "overflow_y": "auto",
            "border": "1px solid #ddd",
            "padding": "6px",
        }
    )
    accordion = widgets.Accordion(children=[trace_output])
    accordion.set_title(0, "Traceback")
    accordion.selected_index = None
    return error_banner, accordion, trace_output


def make_stateful_player(
    *,
    plot: K3DPlot,
    animator: K3DAnimator,
    target_fps: float,
    camera_bounds: Bounds,
    title: str = "Animation",
) -> StatefulPlayer:
    """Create a play/reset controller for stateful animations using wall-clock dt.

    The animator is started once (via `on_start`) and advanced on a background
    thread that calls `on_update(dt)` with wall-clock dt seconds. Playback is
    throttled to `target_fps`; if `target_fps<=0`, it defaults to 60. Exceptions
    are captured and rendered in the widget UI.

    Reset behavior: the player calls `animator.on_reset()` while paused. Keep
    `on_reset` responsible for restoring both state and visuals (no automatic
    `on_start` on reset). `on_start` should be idempotent and typically runs
    only once.
    """

    interval = 1.0 / target_fps if target_fps and target_fps > 0 else 1.0 / 60.0

    play_toggle = widgets.ToggleButton(
        value=False,
        description="Play",
        icon="play",
        tooltip="Play / pause animation",
    )
    reset_button = widgets.Button(description="Reset", icon="refresh")
    status_widget = widgets.HTML("<span>Paused</span>")
    error_banner, traceback_accordion, trace_output = _make_error_widgets()
    title_widget = widgets.HTML(f"<b>{title}</b>")

    extent_obj = _make_extent(camera_bounds)
    plot += extent_obj

    stop_event = threading.Event()
    lock = threading.Lock()
    thread: Optional[threading.Thread] = None
    started = False

    def set_play_ui(playing: bool) -> None:
        if playing:
            play_toggle.description = "Pause"
            play_toggle.icon = "pause"
        else:
            play_toggle.description = "Play"
            play_toggle.icon = "play"

    def set_status(msg: str) -> None:
        status_widget.value = f"<span>{html.escape(msg)}</span>"

    def clear_error() -> None:
        error_banner.value = ""
        with trace_output:
            trace_output.clear_output()
        traceback_accordion.selected_index = None

    def handle_exception(exc: BaseException) -> None:
        stop_event.set()
        banner = (
            '<div style="color:#fff;background:#b00020;padding:6px 8px;'
            'border-radius:4px;font-weight:bold;">'
            f"Exception: {html.escape(str(exc))}"
            "</div>"
        )
        error_banner.value = banner
        tb_str = "".join(traceback.format_exception(exc))
        with trace_output:
            trace_output.clear_output()
            print(tb_str)
        traceback_accordion.selected_index = 0
        if play_toggle.value:
            play_toggle.value = False
        set_play_ui(False)
        set_status("Error")

    def stop_thread(join: bool = True) -> None:
        nonlocal thread
        stop_event.set()
        if join and thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        thread = None

    def run_loop() -> None:
        nonlocal thread
        last = time.perf_counter()
        period = max(interval, 0.001)
        try:
            while not stop_event.is_set():
                now = time.perf_counter()
                dt = now - last
                last = now
                try:
                    animator.on_update(dt)
                except Exception as exc:  # pragma: no cover - UI side
                    handle_exception(exc)
                    return
                # throttle to target FPS
                deadline = last + period
                sleep_for = deadline - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            with lock:
                thread = None

    def start_playback() -> None:
        nonlocal started, thread
        with lock:
            clear_error()
            if not started:
                try:
                    animator.on_start(plot)
                except Exception as exc:  # pragma: no cover - UI side
                    handle_exception(exc)
                    return
                started = True
            stop_event.clear()
            if thread is None or not thread.is_alive():
                thread = threading.Thread(target=run_loop, daemon=True)
                thread.start()
            set_play_ui(True)
            set_status("Playing")

    def stop_playback() -> None:
        with lock:
            stop_thread(join=True)
            set_play_ui(False)
            set_status("Paused")

    def do_reset(_) -> None:
        nonlocal started
        if play_toggle.value:
            play_toggle.value = False
        stop_playback()
        clear_error()
        try:
            animator.on_reset()
            started = True
        except Exception as exc:  # pragma: no cover - UI side
            handle_exception(exc)
            return
        set_play_ui(False)
        set_status("Reset")

    def on_play_change(change: dict) -> None:
        if change.get("name") != "value":
            return
        if change.get("new"):
            start_playback()
        else:
            stop_playback()

    play_toggle.observe(on_play_change, names="value")
    reset_button.on_click(do_reset)

    controls = widgets.HBox([play_toggle, reset_button, status_widget])
    ui = widgets.VBox([title_widget, controls, error_banner, traceback_accordion])

    player = StatefulPlayer(
        ui=ui,
        play_toggle=play_toggle,
        reset_button=reset_button,
        status_widget=status_widget,
        error_banner=error_banner,
        traceback_accordion=traceback_accordion,
        extent_obj=extent_obj,
        plot=plot,
        _stop_event=stop_event,
        _thread=thread,
        _lock=lock,
        _started=started,
    )

    # expose callbacks for reset/close
    player._reset_callback = lambda: do_reset(None)
    player._stop_callback = stop_playback

    return player


def axis_aligned_plane(
    *,
    axis: str,  # 'x', 'y', or 'z'  (the normal axis)
    value: float,  # coordinate along that axis where the plane sits
    umin: float,
    umax: float,  # bounds along first in-plane axis
    vmin: float,
    vmax: float,  # bounds along second in-plane axis
    color: int = 0x444444,
    opacity: float = 0.6,
    wireframe: bool = False,
):
    """
    Create an axis-aligned rectangular plane as a k3d.mesh.

    axis:
      'x' => plane is YZ at x=value
      'y' => plane is XZ at y=value
      'z' => plane is XY at z=value

    u/v bounds correspond to the plane's two in-plane axes:
      axis='x' => u=y, v=z
      axis='y' => u=x, v=z
      axis='z' => u=x, v=y
    """
    axis = axis.lower().strip()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be one of: 'x', 'y', 'z'")

    # Four corners in a consistent winding order
    if axis == "x":  # YZ plane at x=value
        corners = [
            (value, umin, vmin),
            (value, umax, vmin),
            (value, umax, vmax),
            (value, umin, vmax),
        ]
    elif axis == "y":  # XZ plane at y=value
        corners = [
            (umin, value, vmin),
            (umax, value, vmin),
            (umax, value, vmax),
            (umin, value, vmax),
        ]
    else:  # axis == "z": XY plane at z=value
        corners = [
            (umin, vmin, value),
            (umax, vmin, value),
            (umax, vmax, value),
            (umin, vmax, value),
        ]

    vertices = np.array(corners, dtype=np.float32)

    # Two triangles: (0,1,2) and (0,2,3)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    return k3d.mesh(
        vertices=vertices,
        indices=indices,
        color=color,
        opacity=opacity,
        wireframe=wireframe,
    )
