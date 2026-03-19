from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
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


@dataclass(frozen=True)
class FrameBinding:
    """Bind a sequence in `frames` to a k3d widget trait.

    Parameters
    ----------
    key:
        Key into the `frames` dict (must exist, and its sequence length defines nframes).
    obj:
        A k3d widget/object instance (e.g., a `k3d.mesh`, `k3d.points`, `k3d.line`).
    trait:
        Name of the trait to set each frame (e.g., "positions", "vertices", "model_matrix").
    transform:
        Optional function applied to the per-frame value before assignment.
        Useful for casting to np.float32, reshaping, etc.
    """

    key: str
    obj: Any
    trait: str
    transform: Optional[Callable[[Any], Any]] = None


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
    _set_frames_callback: Optional[
        Callable[[Dict[str, Sequence[Any]], int | None], None]
    ] = None

    def reset(self) -> None:
        """Return to the first frame and pause."""
        self.play.playing = False
        self.play.value = self.play.min

    def set_frames(
        self,
        frames: Dict[str, Sequence[Any]],
        *,
        reset: bool = False,
        start: int | None = None,
    ) -> None:
        """Replace the frame dict used by the player.

        This enables interactive UIs that recompute frames (e.g., after slider changes)
        while keeping the same plot/widgets.

        Notes
        -----
        - If the new frame count differs, `play.max` and `slider.max` are updated.
        - If the current frame index exceeds the new max, it is clamped.
        """

        if self._set_frames_callback is None:
            raise RuntimeError(
                "This Player does not support set_frames() (internal callback missing)."
            )
        self._set_frames_callback(frames, start)
        if reset:
            self.reset()

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


class ChunkedAnimator(ABC):
    """Stateful simulator interface for chunked playback.

    A "chunked" player advances the simulation in batches (e.g. 2 seconds worth
    of dt steps), then plays those frames back using ipywidgets.Play (no tight
    realtime loop). This reduces comm pressure while preserving stateful
    evolution.
    """

    @abstractmethod
    def on_start(self, plot: K3DPlot) -> None:
        """Create k3d objects and initialize state."""

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance simulation state by dt seconds."""

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """Return render values keyed by FrameBinding.key."""

    def on_reset(self) -> None:
        """Optional: restore state; default is a no-op."""
        return


@dataclass
class ChunkedPlayer:
    """Handle returned by make_chunked_player."""

    ui: widgets.Widget
    player: Player
    status_widget: widgets.HTML
    _stop_event: threading.Event
    _thread: Optional[threading.Thread]
    _lock: threading.Lock
    _reseed_callback: Optional[Callable[[], None]] = None
    _rechunk_callback: Optional[Callable[[bool], None]] = None

    def reset(self) -> None:
        self.player.reset()

    def close(self) -> None:
        try:
            self._stop_event.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        self.player.close()

    def reseed(self) -> None:
        """Pause, clear buffers, reset animator, and regenerate the current chunk."""
        if self._reseed_callback is None:
            raise RuntimeError(
                "This ChunkedPlayer does not support reseed() (internal callback missing)."
            )
        self._reseed_callback()

    def rechunk(self, *, reset_state: bool = False) -> None:
        """Pause, clear buffers, and regenerate the current chunk.

        Parameters
        ----------
        reset_state:
            If True, calls animator.on_reset() before regenerating.
            If False, keeps the current simulation state and just rebuilds the
            buffered frames from "now".
        """
        if self._rechunk_callback is None:
            raise RuntimeError(
                "This ChunkedPlayer does not support rechunk() (internal callback missing)."
            )
        self._rechunk_callback(bool(reset_state))


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


def set_camera_overhead(
    plot: K3DPlot,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    height: float = 10.0,
    up: Sequence[float] = (0.0, 1.0, 0.0),
) -> None:
    """Place the camera above looking down with +x right and +y up.

    K3D expects `plot.camera` as 9 floats: camera position (x, y, z), look-at
    target (x, y, z), and up vector (x, y, z) [1]_. This helper sets:

    * position = (center.x, center.y, center.z + height)
    * target   = center
    * up       = provided `up` (defaults to +Y)

    Parameters
    ----------
    plot:
        The k3d plot whose camera will be set.
    center:
        Point the camera looks at.
    height:
        Distance above `center` along +Z to place the camera.
    up:
        Up vector; keep it orthogonal to the view direction for best results.

    References
    ----------
    .. [1] https://k3d-jupyter.org/reference/factory.plot.html
    """

    if len(center) != 3 or len(up) != 3:
        raise ValueError("center and up must be length-3 sequences")

    cx, cy, cz = map(float, center)
    ux, uy, uz = map(float, up)
    pos = (cx, cy, cz + float(height))
    look = (cx, cy, cz)

    # Avoid a degenerate up vector that is parallel to the view direction.
    view = np.array(look) - np.array(pos)
    up_vec = np.array(up)
    if np.allclose(np.cross(view, up_vec), 0.0):
        raise ValueError("up vector must not be parallel to the view direction")

    plot.camera = [
        pos[0],
        pos[1],
        pos[2],
        look[0],
        look[1],
        look[2],
        ux,
        uy,
        uz,
    ]


def set_camera_side(
    plot: K3DPlot,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    distance: float = 10.0,
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> None:
    """Place the camera on -Y looking toward +Y with +Z up and +X right.

    Sets `plot.camera` as [pos, target, up]:
      * position = (center.x, center.y - distance, center.z)
      * target   = center
      * up       = provided `up` (defaults to +Z)

    This yields a side view with x to the right and z up.
    """

    if len(center) != 3 or len(up) != 3:
        raise ValueError("center and up must be length-3 sequences")

    cx, cy, cz = map(float, center)
    ux, uy, uz = map(float, up)
    pos = (cx, cy - float(distance), cz)
    look = (cx, cy, cz)

    view = np.array(look) - np.array(pos)
    up_vec = np.array(up)
    if np.allclose(np.cross(view, up_vec), 0.0):
        raise ValueError("up vector must not be parallel to the view direction")

    plot.camera = [
        pos[0],
        pos[1],
        pos[2],
        look[0],
        look[1],
        look[2],
        ux,
        uy,
        uz,
    ]


def translate_camera_to_bounds_center(plot: K3DPlot, *, camera_bounds: Bounds) -> bool:
    """Translate camera (no rotation) so its target is the bounds center.

    This keeps the same view direction and distance, but shifts both camera
    position and target by the same delta.

    Returns True if it applied the translation, False if it aborted due to an
    invalid/degenerate camera or if the center is behind the camera (i.e. the
    camera is "facing away").
    """

    cam = list(getattr(plot, "camera", []) or [])
    if len(cam) != 9:
        return False

    pos = np.asarray(cam[0:3], dtype=float)
    target = np.asarray(cam[3:6], dtype=float)
    up = np.asarray(cam[6:9], dtype=float)
    if not (
        np.all(np.isfinite(pos))
        and np.all(np.isfinite(target))
        and np.all(np.isfinite(up))
    ):
        return False

    view = target - pos
    view_norm = float(np.linalg.norm(view))
    if not np.isfinite(view_norm) or view_norm <= 1.0e-12:
        return False

    xmin, ymin, zmin, xmax, ymax, zmax = map(float, camera_bounds)
    center = np.array(
        [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0], dtype=float
    )

    # Abort if the center is behind the camera (facing away).
    if float(np.dot(center - pos, view)) <= 0.0:
        return False

    delta = center - target
    pos2 = pos + delta
    target2 = target + delta
    plot.camera = [
        float(pos2[0]),
        float(pos2[1]),
        float(pos2[2]),
        float(target2[0]),
        float(target2[1]),
        float(target2[2]),
        float(up[0]),
        float(up[1]),
        float(up[2]),
    ]
    return True


def make_player(
    *,
    plot: K3DPlot,
    frames: Dict[str, Sequence[Any]],
    render_fn: Optional[Callable[[int], None]] = None,
    dt: float,
    camera_bounds: Bounds,
    title: str = "Animation",
    loop: bool = True,
    bindings: Optional[Sequence[FrameBinding]] = None,
    batch_sync: bool = True,
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
        If omitted, you must provide `bindings` so `make_player` can apply the
        per-frame values automatically.
    dt:
        Timestep in seconds between frames; controls Play interval (ms).
    camera_bounds:
        (xmin, ymin, zmin, xmax, ymax, zmax) used to add invisible extent
        geometry so camera auto-fit stays stable.
    title:
        Label displayed above the controls.
    loop:
        If True, wrap to frame 0 after the last frame while continuing playback.
    bindings:
        Optional list of FrameBinding objects. Each binding maps a key in `frames`
        to a k3d trait assignment each frame. This supports "precomputed frames"
        use cases without writing a custom `render_fn`.
    batch_sync:
        If True, wrap each frame render in `hold_sync()` for the plot and its
        objects to reduce comm traffic (often a major performance win).
    """

    nframes = _validate_frames(frames)
    if render_fn is None:
        if not bindings:
            raise ValueError(
                "Either render_fn must be provided, or bindings must be a non-empty list."
            )
        for b in bindings:
            if b.key not in frames:
                raise ValueError(f"FrameBinding key not found in frames: {b.key!r}")

    interval_ms = max(1, int(round(dt * 1000)))
    frames_ref: dict[str, Dict[str, Sequence[Any]]] = {"frames": frames}

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
    recenter_button = widgets.Button(
        description="Recenter",
        icon="crosshairs",
        tooltip="Translate camera to bounds center",
    )
    title_widget = widgets.HTML(f"<b>{title}</b>")
    camera_msg = widgets.HTML("")

    extent_obj = _make_extent(camera_bounds)
    plot += extent_obj

    def _render_bound(i: int) -> None:
        if not bindings:
            return
        local_frames = frames_ref["frames"]
        for b in bindings:
            value = local_frames[b.key][i]
            if b.transform is not None:
                value = b.transform(value)
            setattr(b.obj, b.trait, value)

    def _render(i: int) -> None:
        if batch_sync:
            with ExitStack() as stack:
                if hasattr(plot, "hold_sync"):
                    stack.enter_context(plot.hold_sync())
                for obj in getattr(plot, "objects", []) or []:
                    if hasattr(obj, "hold_sync"):
                        stack.enter_context(obj.hold_sync())
                if render_fn is not None:
                    render_fn(i)
                else:
                    _render_bound(i)
        else:
            if render_fn is not None:
                render_fn(i)
            else:
                _render_bound(i)

    def do_reset(stop_playback: bool = True) -> None:
        if stop_playback:
            play.playing = False
        play.value = play.min

    def on_value(change: dict) -> None:
        if change.get("name") != "value":
            return
        i = int(change["new"])
        _render(i)
        if loop and i >= play.max:
            playing_state = bool(getattr(play, "playing", False))
            play.unobserve(on_value, names="value")
            try:
                play.value = play.min
                _render(play.min)
                if playing_state:
                    play.playing = True
            finally:
                play.observe(on_value, names="value")

    play.observe(on_value, names="value")
    reset_button.on_click(lambda _: do_reset(stop_playback=True))
    recenter_button.on_click(
        lambda _: (
            camera_msg.__setattr__(
                "value",
                (
                    ""
                    if translate_camera_to_bounds_center(
                        plot, camera_bounds=camera_bounds
                    )
                    else "<span style='color:#ffa657'>Recenter aborted (camera invalid or facing away).</span>"
                ),
            )
        )
    )

    controls = widgets.HBox([play, slider, reset_button, recenter_button])
    ui = widgets.VBox([title_widget, controls, camera_msg])

    player = Player(
        ui=ui,
        play=play,
        slider=slider,
        reset_button=reset_button,
        title_widget=title_widget,
        extent_obj=extent_obj,
        plot=plot,
        _set_frames_callback=None,
    )

    def _set_frames(
        new_frames: Dict[str, Sequence[Any]], *, start: int | None = None
    ) -> None:
        new_n = _validate_frames(new_frames)
        frames_ref["frames"] = new_frames

        # Update widget ranges if length changed.
        new_max = new_n - 1
        if play.max != new_max:
            play.max = new_max
            slider.max = new_max

        # Choose target frame index.
        if start is None:
            target = int(play.value)
        else:
            target = int(start)
        if target < play.min:
            target = int(play.min)
        if target > play.max:
            target = int(play.max)

        # Setting play.value triggers `on_value` which will call _render using the
        # updated frames_ref. If the value is unchanged, force a render.
        if int(play.value) != target:
            play.value = target
        else:
            _render(target)

    # Expose a frames mutator for interactive UIs.
    player._set_frames_callback = lambda f, start=None: _set_frames(f, start=start)
    return player


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
    animator: K3DAnimator | Iterable[K3DAnimator],
    target_fps: float,
    camera_bounds: Bounds,
    title: str = "Animation",
) -> StatefulPlayer:
    """Create a play/reset controller for stateful animations using wall-clock dt.

    The animator is started once (via `on_start`) and advanced on a background
    thread that calls `on_update(dt)` with wall-clock dt seconds. Playback is
    throttled to `target_fps`; if `target_fps<=0`, it defaults to 60. Exceptions
    are captured and rendered in the widget UI.

    The `animator` argument may be a single `K3DAnimator` or an iterable of
    animators. When multiple are provided, their `on_start`, `on_update`, and
    `on_reset` methods are invoked sequentially each step.

    Reset behavior: the player calls `on_reset()` on each animator while paused.
    Keep `on_reset` responsible for restoring both state and visuals (no
    automatic `on_start` on reset). `on_start` should be idempotent and
    typically runs only once.
    """

    # Normalize animator(s) into a list.
    if isinstance(animator, Iterable) and not isinstance(animator, K3DAnimator):
        animators = list(animator)
    else:
        animators = [cast(K3DAnimator, animator)]

    if not animators:
        raise ValueError("animator iterable must contain at least one animator")

    interval = 1.0 / target_fps if target_fps and target_fps > 0 else 1.0 / 60.0

    play_toggle = widgets.ToggleButton(
        value=False,
        description="Play",
        icon="play",
        tooltip="Play / pause animation",
    )
    reset_button = widgets.Button(description="Reset", icon="refresh")
    recenter_button = widgets.Button(
        description="Recenter",
        icon="crosshairs",
        tooltip="Translate camera to bounds center",
    )
    status_widget = widgets.HTML("<span>Paused</span>")
    camera_msg = widgets.HTML("")
    error_banner, traceback_accordion, trace_output = _make_error_widgets()
    title_widget = widgets.HTML(f"<b>{title}</b>")

    extent_obj = _make_extent(camera_bounds)
    plot += extent_obj

    # NOTE on performance / responsiveness:
    # - We keep at most ONE worker thread alive. Pausing does NOT join the thread;
    #   it simply clears `play_event`. This prevents "pause lag" where the UI blocks
    #   until a slow `on_update` finishes.
    # - We also guard `on_update`/`on_reset`/`on_start` with `update_lock` to prevent
    #   concurrent calls into the animator (which can lead to state corruption and
    #   multiple traitlet update streams piling up).
    # - We throttle based on *elapsed update time* so a slow frame does not cause a
    #   tight loop that floods the Jupyter comm channel.
    shutdown_event = threading.Event()
    play_event = threading.Event()
    lock = threading.Lock()
    update_lock = threading.Lock()
    thread: Optional[threading.Thread] = None
    started = False
    last_time = time.perf_counter()

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
        # Stop playback and terminate the worker thread; repeated failing updates
        # can otherwise flood comms and make the notebook unresponsive.
        play_event.clear()
        shutdown_event.set()
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
        shutdown_event.set()
        play_event.clear()
        if join and thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        thread = None

    def run_loop() -> None:
        nonlocal thread, last_time
        period = max(interval, 0.001)
        try:
            while not shutdown_event.is_set():
                # Block while paused (keeps CPU + comm traffic low).
                if not play_event.wait(timeout=0.1):
                    continue

                now = time.perf_counter()
                with lock:
                    dt = now - last_time
                    last_time = now
                # Protect animator from concurrent reset/start.
                t0 = time.perf_counter()
                try:
                    # Batch trait updates to avoid spamming the comm channel:
                    # one "sync" per widget per frame instead of one per assignment.
                    with ExitStack() as stack:
                        if hasattr(plot, "hold_sync"):
                            stack.enter_context(plot.hold_sync())
                        for obj in getattr(plot, "objects", []) or []:
                            if hasattr(obj, "hold_sync"):
                                stack.enter_context(obj.hold_sync())

                        with update_lock:
                            for anim in animators:
                                anim.on_update(dt)
                except Exception as exc:  # pragma: no cover - UI side
                    handle_exception(exc)
                    return

                # Throttle to target FPS. If the update itself took longer than
                # the desired period, we do NOT "catch up" by running multiple
                # updates back-to-back (that floods comms and makes pause laggy).
                elapsed = time.perf_counter() - t0
                sleep_for = period - elapsed
                if sleep_for > 0:
                    # Allow a reasonably quick response to shutdown.
                    shutdown_event.wait(timeout=sleep_for)
        finally:
            with lock:
                thread = None

    def start_playback() -> None:
        nonlocal started, thread, shutdown_event, last_time
        with lock:
            clear_error()
            if shutdown_event.is_set():
                # If we previously crashed or were explicitly stopped, start a new worker.
                shutdown_event = threading.Event()
            if thread is None or not thread.is_alive():
                thread = threading.Thread(target=run_loop, daemon=True)
                thread.start()

            # Ensure we never call on_start concurrently with on_update.
            if not started:
                try:
                    with update_lock:
                        for anim in animators:
                            anim.on_start(plot)
                except Exception as exc:  # pragma: no cover - UI side
                    handle_exception(exc)
                    return
                started = True

            last_time = time.perf_counter()
            play_event.set()
            set_play_ui(True)
            set_status("Playing")

    def stop_playback() -> None:
        with lock:
            # Do not join here: joining blocks the UI until `on_update` returns.
            play_event.clear()
        set_play_ui(False)
        set_status("Paused")

    def do_reset(_) -> None:
        nonlocal started
        if play_toggle.value:
            play_toggle.value = False
        stop_playback()
        clear_error()
        try:
            with update_lock:
                for anim in animators:
                    anim.on_reset()
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
    recenter_button.on_click(
        lambda _: (
            camera_msg.__setattr__(
                "value",
                (
                    ""
                    if translate_camera_to_bounds_center(
                        plot, camera_bounds=camera_bounds
                    )
                    else "<span style='color:#ffa657'>Recenter aborted (camera invalid or facing away).</span>"
                ),
            )
        )
    )

    controls = widgets.HBox([play_toggle, reset_button, recenter_button, status_widget])
    ui = widgets.VBox(
        [title_widget, controls, camera_msg, error_banner, traceback_accordion]
    )

    player = StatefulPlayer(
        ui=ui,
        play_toggle=play_toggle,
        reset_button=reset_button,
        status_widget=status_widget,
        error_banner=error_banner,
        traceback_accordion=traceback_accordion,
        extent_obj=extent_obj,
        plot=plot,
        _stop_event=shutdown_event,
        _thread=thread,
        _lock=lock,
        _started=started,
    )

    # expose callbacks for reset/close
    player._reset_callback = lambda: do_reset(None)
    # For close(), stop the worker thread and remove extent geometry.
    player._stop_callback = lambda: stop_thread(join=True)

    return player


def make_chunked_player(
    *,
    plot: K3DPlot,
    animator: ChunkedAnimator,
    bindings: Sequence[FrameBinding],
    dt: float,
    camera_bounds: Bounds,
    title: str = "Animation",
    target_fps: float = 60.0,
    chunk_seconds: float = 2.0,
    buffer_chunks: int = 2,
    loop: bool = True,
    batch_sync: bool = True,
    show_frame_slider: bool = False,
) -> ChunkedPlayer:
    """Chunked/ring-buffer style player for stateful simulations.

    The simulation is advanced in chunks on a background thread. Each chunk is
    played back via `make_player`, which keeps the UI responsive and avoids a
    tight realtime loop that can flood Jupyter comms.

    Notes
    -----
    - Playback runs at `target_fps` (widget frame rate), independent of `dt`
      (simulation timestep).
    - Chunk generation uses `animator.step(dt)` and `animator.snapshot()`.
    - Vector/mesh trait updates are still per-displayed-frame; chunking removes
      per-frame physics, not per-frame rendering.
    """

    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")

    target_fps = float(target_fps)
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        target_fps = 60.0

    chunk_seconds = float(chunk_seconds)
    if not np.isfinite(chunk_seconds) or chunk_seconds <= 0.0:
        raise ValueError("chunk_seconds must be finite and > 0")

    buffer_chunks = int(buffer_chunks)
    if buffer_chunks < 1:
        raise ValueError("buffer_chunks must be >= 1")

    keys = [b.key for b in bindings]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "FrameBinding.key values must be unique for make_chunked_player"
        )

    chunk_nframes = max(2, int(round(chunk_seconds / dt)))
    play_dt = 1.0 / target_fps

    lock = threading.Lock()
    stop_event = threading.Event()
    status = widgets.HTML("")

    animator.on_start(plot)

    def _stack(vals: list[Any]) -> Any:
        v0 = vals[0]
        if isinstance(v0, np.ndarray):
            return np.stack(vals, axis=0)
        return vals

    def build_chunk(nframes: int) -> Dict[str, Any]:
        series: Dict[str, list[Any]] = {k: [] for k in keys}
        for _ in range(int(nframes)):
            snap = animator.snapshot()
            for k in keys:
                series[k].append(snap[k])
            animator.step(dt)
        return {k: _stack(v) for k, v in series.items()}

    with lock:
        current_frames = build_chunk(chunk_nframes)

    # Queue of prepared chunks (current chunk lives in the player).
    next_frames: Dict[str, Any] | None = None
    queue: list[Dict[str, Any]] = []

    def worker() -> None:
        nonlocal next_frames
        try:
            while not stop_event.is_set():
                with lock:
                    backlog = (1 if next_frames is not None else 0) + len(queue)
                if backlog >= max(0, buffer_chunks - 1):
                    stop_event.wait(timeout=0.05)
                    continue

                with lock:
                    status.value = "<span>Buffering…</span>"
                chunk = build_chunk(chunk_nframes)
                with lock:
                    if next_frames is None:
                        next_frames = chunk
                    else:
                        queue.append(chunk)
                    status.value = ""
        except Exception as exc:  # pragma: no cover
            with lock:
                status.value = (
                    "<span style='color:#ffa657'>Chunk worker error:</span> "
                    + html.escape(str(exc))
                )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    player = make_player(
        plot=plot,
        frames=current_frames,
        bindings=bindings,
        dt=play_dt,
        camera_bounds=camera_bounds,
        title=title,
        loop=False,
        batch_sync=batch_sync,
    )
    if not show_frame_slider:
        try:
            player.slider.layout.display = "none"
        except Exception:
            pass

    def rollover(change: dict) -> None:
        nonlocal current_frames, next_frames
        if change.get("name") != "value":
            return
        i = int(change.get("new", 0))
        if i < player.play.max:
            return

        with lock:
            nf = next_frames
            if nf is None:
                status.value = "<span style='color:#ffa657'>Buffer underrun…</span>"
                return
            current_frames = nf
            next_frames = queue.pop(0) if queue else None

        # ipywidgets.Play will often auto-stop (`playing=False`) when it hits max.
        # For looping chunk playback, explicitly re-enable playing after rollover.
        player.play.unobserve(rollover, names="value")
        try:
            player.set_frames(current_frames, reset=False, start=int(player.play.min))
            if loop:
                player.play.playing = True
            else:
                player.play.playing = False
        finally:
            player.play.observe(rollover, names="value")

    player.play.observe(rollover, names="value")

    def reseed() -> None:
        rechunk(reset_state=True)

    def rechunk(*, reset_state: bool) -> None:
        nonlocal current_frames, next_frames, queue, thread
        # Pause playback while we rebuild.
        player.play.playing = False
        with lock:
            stop_event.set()
        if thread.is_alive():
            thread.join(timeout=1.0)

        with lock:
            status.value = "<span>Rebuilding…</span>"
            if reset_state:
                animator.on_reset()
            current_frames = build_chunk(chunk_nframes)
            next_frames = None
            queue.clear()
            stop_event.clear()
            status.value = ""

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        player.set_frames(current_frames, reset=True)

    ui = widgets.VBox([player.ui, status])
    cp = ChunkedPlayer(
        ui=ui,
        player=player,
        status_widget=status,
        _stop_event=stop_event,
        _thread=thread,
        _lock=lock,
        _reseed_callback=None,
        _rechunk_callback=None,
    )
    cp._reseed_callback = reseed
    cp._rechunk_callback = lambda rs: rechunk(reset_state=bool(rs))
    return cp


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
    texture: bytes | None = None,
    texture_rotation_deg: int = 0,
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

    texture_rotation_deg:
      Clockwise rotation of the texture coordinates in degrees. Must be a
      multiple of 90 (0, 90, 180, 270). Useful when the provided image is
      oriented differently from the desired mapping.
    """
    axis = axis.lower().strip()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be one of: 'x', 'y', 'z'")

    rotation = int(texture_rotation_deg) % 360
    if rotation % 90 != 0:
        raise ValueError("texture_rotation_deg must be a multiple of 90")

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

    # Cover the quad with the full texture in vertex winding order.
    base_uvs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if rotation:
        # Rotate UVs around the texture center (0.5, 0.5).
        theta = np.deg2rad(rotation)
        c, s = float(np.cos(theta)), float(np.sin(theta))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        centered = base_uvs - 0.5
        uvs = centered @ rot.T + 0.5
    else:
        uvs = base_uvs

    # Two triangles: (0,1,2) and (0,2,3)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    return k3d.mesh(
        vertices=vertices,
        indices=indices,
        color=color,
        opacity=opacity,
        wireframe=wireframe,
        texture=texture,
        texture_file_format="jpg",
        uvs=uvs,
    )
