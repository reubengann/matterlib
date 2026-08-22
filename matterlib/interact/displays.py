from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Callable, Optional, cast

from ipycanvas import Canvas, hold_canvas
import ipywidgets as widgets

from .explorer import RenderContext


def _state_value(state: object, name: str) -> float:
    value_method = getattr(state, "value", None)
    if callable(value_method):
        get_value = cast(Callable[[str], Any], value_method)
        return float(get_value(name))
    return float(getattr(state, name))


class CanvasDisplay(ABC):
    """Base class for a display that owns a single ipycanvas widget."""

    def __init__(self, width: int, height: int):
        self.canvas = Canvas(width=width, height=height)

    @property
    def widget(self) -> widgets.Widget:
        return self.canvas

    @abstractmethod
    def render(self, context: RenderContext) -> None:
        ...

    @staticmethod
    def _clear(canvas: Canvas, background: str) -> None:
        canvas.fill_style = background
        canvas.fill_rect(0, 0, canvas.width, canvas.height)


class PistonDisplay(CanvasDisplay):
    """A schematic piston whose position and gas color follow V and T."""

    def __init__(
        self,
        width: int = 700,
        height: int = 250,
        volume_name: str = "V",
        temperature_name: str = "T",
    ):
        super().__init__(width, height)
        self.volume_name = volume_name
        self.temperature_name = temperature_name

    def render(self, context: RenderContext) -> None:
        canvas = self.canvas
        theme = context.theme
        state = context.state
        volume = _state_value(state, self.volume_name)
        temperature = _state_value(state, self.temperature_name)
        volume_spec = context.quantities[self.volume_name]

        fraction = (volume - volume_spec.min_val) / (
            volume_spec.max_val - volume_spec.min_val
        )
        fraction = max(0.0, min(1.0, fraction))
        x0, y0 = 60, 70
        cylinder_width, cylinder_height = 520, 100
        piston_x = x0 + 40 + fraction * (cylinder_width - 80)

        temperature_fraction = max(
            0.0, min(1.0, (temperature - 250.0) / 200.0)
        )
        blue = int(150 + 100 * (1 - temperature_fraction))
        red = int(80 + 150 * temperature_fraction)
        green = int(80 + 150 * temperature_fraction)

        with hold_canvas():
            self._clear(canvas, str(theme["bg"]))
            canvas.stroke_style = str(theme["stroke"])
            canvas.line_width = 2
            canvas.stroke_rect(x0, y0, cylinder_width, cylinder_height)

            canvas.fill_style = f"rgb({red},{green},{blue})"
            canvas.fill_rect(x0, y0, piston_x - x0, cylinder_height)

            canvas.fill_style = str(theme["piston"])
            canvas.fill_rect(
                piston_x - 6, y0 - 10, 12, cylinder_height + 20
            )
            canvas.fill_style = str(theme["rod"])
            canvas.fill_rect(
                piston_x + 6,
                y0 + cylinder_height / 2 - 3,
                80,
                6,
            )

            canvas.fill_style = str(theme["text"])
            canvas.font = "16px sans-serif"
            canvas.fill_text(
                f"P = {_state_value(state, 'P'):,.0f} Pa", 60, 35
            )
            canvas.fill_text(f"V = {volume:.4f} m³", 220, 35)
            canvas.fill_text(f"T = {temperature:.1f} K", 380, 35)
            canvas.fill_text(
                f"Constraint: {context.selection}", 60, 210
            )


class XYChartDisplay(CanvasDisplay):
    """A stable-axis path chart for two named state quantities."""

    def __init__(
        self,
        x: str = "V",
        y: str = "P",
        width: int = 350,
        height: int = 250,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ):
        super().__init__(width, height)
        self.x = x
        self.y = y
        self.x_label = x_label or x
        self.y_label = y_label or y

    def render(self, context: RenderContext) -> None:
        canvas = self.canvas
        theme = context.theme
        x_spec = context.quantities[self.x]
        y_spec = context.quantities[self.y]
        x_min, x_max = float(x_spec.min_val), float(x_spec.max_val)
        y_min, y_max = float(y_spec.min_val), float(y_spec.max_val)
        left, top = 55, 20
        width, height = canvas.width - 95, canvas.height - 65

        def screen_x(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * width

        def screen_y(value: float) -> float:
            return top + height - (value - y_min) / (y_max - y_min) * height

        history = context.history
        with hold_canvas():
            self._clear(canvas, str(theme["bg"]))
            canvas.stroke_style = str(theme["axis"])
            canvas.line_width = 1
            canvas.stroke_rect(left, top, width, height)
            canvas.fill_style = str(theme["text"])
            canvas.font = "14px sans-serif"
            canvas.fill_text(self.y_label, 12, 25)
            canvas.fill_text(
                self.x_label, left + width + 10, top + height + 5
            )

            colors = theme.get("process_colors", {})
            for previous, current in zip(history, history[1:]):
                previous_state = getattr(previous, "state", previous)
                current_state = getattr(current, "state", current)
                constraint = getattr(current, "constraint", context.selection)
                canvas.stroke_style = str(
                    colors.get(constraint, theme["path"])
                )
                canvas.begin_path()
                canvas.move_to(
                    screen_x(_state_value(previous_state, self.x)),
                    screen_y(_state_value(previous_state, self.y)),
                )
                canvas.line_to(
                    screen_x(_state_value(current_state, self.x)),
                    screen_y(_state_value(current_state, self.y)),
                )
                canvas.stroke()

            state = context.state
            point_x = screen_x(_state_value(state, self.x))
            point_y = screen_y(_state_value(state, self.y))
            if math.isfinite(point_x) and math.isfinite(point_y):
                canvas.fill_style = str(theme["point"])
                canvas.begin_path()
                canvas.arc(point_x, point_y, 4, 0, 2 * math.pi)
                canvas.fill()


class SimpleChartDisplay(XYChartDisplay):
    """Backward-friendly name for the configurable XY chart."""

