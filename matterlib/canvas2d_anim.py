from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union, cast
from abc import ABC
import html
import threading
import time
import traceback

import ipywidgets as widgets
from ipycanvas import Canvas


# ----------------------------
# Timer primitive (from example_2d.py)
# ----------------------------
class RepeatedTimer:
    """
    Call fn() roughly every interval_s seconds in a background thread.
    Designed for notebook demos; stop() is important.
    """

    def __init__(self, interval_s: float, fn: Callable[[], None]):
        self.interval = float(interval_s)
        self.fn = fn
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            next_t += self.interval
            try:
                self.fn()
            except Exception:
                # If something goes wrong, stop so we don't spam errors.
                self._stop.set()
                raise
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.25)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ----------------------------
# Animator + parameter binding
# ----------------------------
class Canvas2DAnimator(ABC):
    """
    Stateful 2D animation interface driven by wall-clock dt.

    Subclasses can either override `on_frame(dt)` directly, or override
    `on_update(dt)` and `on_draw()` and rely on the default `on_frame`
    implementation (which calls update then draw).
    """

    # Optional parameter specs: {"param_name": ParamSpec}
    PARAMS: Dict[str, "ParamSpec"] = {}

    def on_start(self, canvas: Canvas) -> None:
        """Create canvas objects and initialize state. Idempotent."""
        return

    def on_frame(self, dt: float) -> None:
        """Advance simulation by dt seconds and draw. Override or rely on update/draw."""
        self.on_update(dt)
        self.on_draw()

    def on_update(self, dt: float) -> None:  # pragma: no cover - default no-op
        return

    def on_draw(self) -> None:  # pragma: no cover - default no-op
        return

    def on_reset(self) -> None:
        """Restore state and visuals; default is a no-op."""
        return


OnChangeBehavior = Union[None, str]


@dataclass
class ParamSpec:
    """
    Declarative parameter binding to an ipywidgets control.

    kind: one of {"float_slider", "int_slider", "checkbox", "dropdown", "toggle"}
    on_change: "reset" (default), "redraw", "none", or "restart_timer"
    """

    kind: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[Sequence[Any]] = None  # for dropdown
    description: Optional[str] = None
    tooltip: Optional[str] = None
    readout: bool = True
    readout_format: Optional[str] = None  # for FloatSlider
    continuous_update: bool = False
    on_change: str = "reset"  # "reset" | "redraw" | "none" | "restart_timer"


@dataclass
class BoundParam:
    """Runtime pairing of a ParamSpec and its widget."""

    name: str
    spec: ParamSpec
    widget: widgets.Widget


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


def _widget_from_spec(name: str, spec: ParamSpec) -> widgets.Widget:
    desc = spec.description or name
    if spec.kind == "float_slider":
        return widgets.FloatSlider(
            value=float(spec.default),
            min=float(spec.min if spec.min is not None else 0.0),
            max=float(spec.max if spec.max is not None else 1.0),
            step=float(spec.step if spec.step is not None else 0.01),
            description=desc,
            tooltip=spec.tooltip or "",
            readout=spec.readout,
            readout_format=spec.readout_format or ".3f",
            continuous_update=spec.continuous_update,
            style={"description_width": "80px"},
        )
    if spec.kind == "int_slider":
        return widgets.IntSlider(
            value=int(spec.default),
            min=int(spec.min if spec.min is not None else 0),
            max=int(spec.max if spec.max is not None else 10),
            step=int(spec.step if spec.step is not None else 1),
            description=desc,
            tooltip=spec.tooltip or "",
            readout=spec.readout,
            continuous_update=spec.continuous_update,
            style={"description_width": "80px"},
        )
    if spec.kind == "checkbox":
        return widgets.Checkbox(
            value=bool(spec.default),
            description=desc,
            tooltip=spec.tooltip or "",
        )
    if spec.kind == "dropdown":
        opts = list(spec.options or [])
        if not opts:
            raise ValueError(f"Param '{name}' dropdown requires non-empty options")
        return widgets.Dropdown(
            options=opts,
            value=spec.default if spec.default in opts else opts[0],
            description=desc,
            tooltip=spec.tooltip or "",
            style={"description_width": "80px"},
        )
    if spec.kind == "toggle":
        return widgets.ToggleButton(
            value=bool(spec.default),
            description=desc,
            tooltip=spec.tooltip or "",
            icon="check" if spec.default else "",
        )
    raise ValueError(f"Unsupported ParamSpec kind: {spec.kind}")


# ----------------------------
# Canvas2DPlayer
# ----------------------------
class Canvas2DPlayer:
    """
    Timer-driven 2D animation controller for ipycanvas.

    Uses RepeatedTimer (same pattern as example_2d.py) to invoke animator(s)
    at a fixed interval derived from target_fps. Exposes play/pause/step/reset
    controls and renders errors in-widget.

    Canvas sizing:
    - Pass an existing Canvas via `canvas=` to control size explicitly.
    - Or pass `width`/`height` (used when canvas is not provided).
    """

    def __init__(
        self,
        *,
        animator: Canvas2DAnimator | Iterable[Canvas2DAnimator],
        width: int = 640,
        height: int = 420,
        target_fps: float = 60.0,
        dt: float | Callable[[], float] = 1.0 / 240.0,
        title: str = "2D Animation",
        canvas: Optional[Canvas] = None,
        auto_draw_initial: bool = True,
    ) -> None:
        # Normalize animators to a list.
        if isinstance(animator, Iterable) and not isinstance(animator, Canvas2DAnimator):
            self.animators: List[Canvas2DAnimator] = list(animator)
        else:
            self.animators = [cast(Canvas2DAnimator, animator)]
        if not self.animators:
            raise ValueError("animator iterable must contain at least one animator")

        self.canvas = canvas or Canvas(width=width, height=height)
        self._target_fps = float(target_fps) if target_fps and target_fps > 0 else 60.0
        self._dt_provider = dt

        self._timer: Optional[RepeatedTimer] = None
        self._started = False
        self._lock = threading.Lock()

        # UI elements
        self.play_toggle = widgets.ToggleButton(
            value=False,
            description="Play",
            icon="play",
            tooltip="Play / pause animation",
        )
        self.step_button = widgets.Button(description="Step", icon="forward")
        self.reset_button = widgets.Button(description="Reset", icon="refresh")
        self.status_widget = widgets.HTML("<span>Paused</span>")
        self.error_banner, self.traceback_accordion, self._trace_output = _make_error_widgets()
        self.title_widget = widgets.HTML(f"<b>{title}</b>")

        self._param_bindings: List[BoundParam] = []
        self._param_container: Optional[widgets.Widget] = None
        self._build_param_bindings()

        # Wire controls
        self.play_toggle.observe(self._on_play_change, names="value")
        self.step_button.on_click(self._on_step)
        self.reset_button.on_click(self._on_reset_click)

        controls = widgets.HBox([self.play_toggle, self.step_button, self.reset_button, self.status_widget])
        pieces = [self.title_widget, controls, self.error_banner, self.traceback_accordion]
        if self._param_container is not None:
            pieces.insert(1, self._param_container)
        self.ui = widgets.VBox([w for w in pieces if w is not None])

        # Optionally draw the first frame immediately so the canvas is not blank.
        if auto_draw_initial:
            self._ensure_started()

    # --- public control methods ---
    def start(self) -> None:
        """Start playback (creates timer if needed)."""
        self._clear_error()
        self._ensure_started()
        self._start_timer()
        self._set_play_ui(True)
        self._set_status("Playing")
        self.play_toggle.value = True

    def pause(self) -> None:
        """Pause playback."""
        self._stop_timer()
        self._set_play_ui(False)
        self._set_status("Paused")
        self.play_toggle.value = False

    def step(self) -> None:
        """Advance one frame even if paused."""
        self._tick_once()

    def reset(self) -> None:
        """Pause, reset state, redraw."""
        self.pause()
        self._clear_error()
        try:
            for anim in self.animators:
                anim.on_reset()
        except Exception as exc:
            self._handle_exception(exc)
            return
        self._redraw_once()
        self._set_status("Reset")

    def close(self) -> None:
        """Stop timer and detach callbacks."""
        self._stop_timer()
        try:
            self.play_toggle.unobserve(self._on_play_change, names="value")
        except Exception:
            pass
        try:
            self.step_button.on_click(lambda _: None)
        except Exception:
            pass
        try:
            self.reset_button.on_click(lambda _: None)
        except Exception:
            pass

    # --- internal helpers ---
    def _ensure_started(self) -> None:
        if self._started:
            return
        try:
            for anim in self.animators:
                anim.on_start(self.canvas)
        except Exception as exc:
            self._handle_exception(exc)
            return
        self._started = True
        self._redraw_once()

    def _resolve_dt(self) -> float:
        provider = self._dt_provider
        try:
            val = provider() if callable(provider) else provider
        except Exception:
            val = provider  # type: ignore
        try:
            dt = float(val)
        except Exception:
            dt = 1.0 / max(self._target_fps, 1.0)
        if dt <= 0:
            dt = 1.0 / max(self._target_fps, 1.0)
        return dt

    def _tick_once(self) -> None:
        dt = self._resolve_dt()
        try:
            with self._lock:
                for anim in self.animators:
                    anim.on_frame(dt)
        except Exception as exc:
            self._handle_exception(exc)
            return

    def _redraw_once(self) -> None:
        try:
            with self._lock:
                for anim in self.animators:
                    # Prefer on_draw if overridden; else fall back to on_frame(0).
                    if anim.on_draw.__func__ is not Canvas2DAnimator.on_draw:  # type: ignore[attr-defined]
                        anim.on_draw()
                    else:
                        anim.on_frame(0.0)
        except Exception as exc:
            self._handle_exception(exc)
            return

    def _timer_callback(self) -> None:
        self._tick_once()

    def _start_timer(self) -> None:
        self._stop_timer()
        interval = 1.0 / max(self._target_fps, 1.0)
        self._timer = RepeatedTimer(interval, self._timer_callback)
        self._timer.start()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    # --- UI wiring ---
    def _on_play_change(self, change: dict) -> None:
        if change.get("name") != "value":
            return
        if change.get("new"):
            self.start()
        else:
            self.pause()

    def _on_step(self, _) -> None:
        self.step()

    def _on_reset_click(self, _) -> None:
        self.reset()

    def _set_play_ui(self, playing: bool) -> None:
        if playing:
            self.play_toggle.description = "Pause"
            self.play_toggle.icon = "pause"
        else:
            self.play_toggle.description = "Play"
            self.play_toggle.icon = "play"

    def _set_status(self, msg: str) -> None:
        self.status_widget.value = f"<span>{html.escape(msg)}</span>"

    def _clear_error(self) -> None:
        self.error_banner.value = ""
        with self._trace_output:
            self._trace_output.clear_output()
        self.traceback_accordion.selected_index = None

    def _handle_exception(self, exc: BaseException) -> None:
        self._stop_timer()
        banner = (
            '<div style="color:#fff;background:#b00020;padding:6px 8px;'
            'border-radius:4px;font-weight:bold;">'
            f"Exception: {html.escape(str(exc))}"
            "</div>"
        )
        self.error_banner.value = banner
        tb_str = "".join(traceback.format_exception(exc))
        with self._trace_output:
            self._trace_output.clear_output()
            print(tb_str)
        self.traceback_accordion.selected_index = 0
        self.play_toggle.value = False
        self._set_play_ui(False)
        self._set_status("Error")

    # --- parameter binding ---
    def _build_param_bindings(self) -> None:
        specs: Dict[str, ParamSpec] = {}
        binding_widgets: List[widgets.Widget] = []
        for anim in self.animators:
            params = getattr(anim, "PARAMS", {}) or {}
            for name, spec in params.items():
                if name in specs:
                    raise ValueError(f"Duplicate param '{name}' across animators")
                specs[name] = spec
                widget = _widget_from_spec(name, spec)
                # Initialize animator attribute to default if not set.
                if not hasattr(anim, name):
                    setattr(anim, name, spec.default)

                def _on_change(change: dict, *, _anim=anim, _name=name, _spec=spec, _widget=widget) -> None:
                    if change.get("name") != "value":
                        return
                    setattr(_anim, _name, change["new"])
                    if _spec.on_change == "reset":
                        self.reset()
                    elif _spec.on_change == "redraw":
                        self._redraw_once()
                    elif _spec.on_change == "restart_timer":
                        if self.play_toggle.value:
                            self._start_timer()
                    elif _spec.on_change == "none":
                        pass

                widget.observe(_on_change, names="value")
                binding_widgets.append(widget)
                self._param_bindings.append(BoundParam(name=name, spec=spec, widget=widget))

        if binding_widgets:
            # Lay out in a reasonable column; chunk rows of 2.
            rows = []
            for i in range(0, len(binding_widgets), 2):
                rows.append(widgets.HBox(binding_widgets[i : i + 2]))
            self._param_container = widgets.VBox(rows)
        else:
            self._param_container = None

