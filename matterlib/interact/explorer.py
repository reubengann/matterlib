from __future__ import annotations

from dataclasses import dataclass
import html
import traceback
from typing import Any, Mapping, Protocol, Sequence, Tuple

import ipywidgets as widgets

from .controls import Gauge, SliderQuantity


class Choice(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def label(self) -> str:
        ...


@dataclass(frozen=True)
class RenderContext:
    """Immutable data passed to every display during a refresh."""

    state: object
    history: Tuple[object, ...]
    selection: str
    quantities: Mapping[str, SliderQuantity]
    theme: Mapping[str, Any]


class DemoDisplay(Protocol):
    @property
    def widget(self) -> widgets.Widget:
        ...

    def render(self, context: RenderContext) -> None:
        ...


class DemoController(Protocol):
    selection_label: str

    @property
    def constraints(self) -> Sequence[Choice]:
        ...

    @property
    def selection(self) -> str:
        ...

    @property
    def values(self) -> Mapping[str, float]:
        ...

    @property
    def valid_drivers(self) -> frozenset[str]:
        ...

    def set_driver(self, name: str, value: float) -> None:
        ...

    def set_constraint(self, name: str) -> None:
        ...

    def reset(self) -> None:
        ...

    def clear_history(self) -> None:
        ...

    def readout(self, name: str) -> object:
        ...

    def render_context(
        self, quantities: Mapping[str, SliderQuantity]
    ) -> RenderContext:
        ...


class ExplorerPanel:
    """Widget orchestration for a stateful, non-animated interactive demo."""

    def __init__(
        self,
        quantities: Sequence[SliderQuantity],
        gauges: Sequence[Gauge],
        displays: Sequence[DemoDisplay],
        controller: DemoController,
    ):
        if not quantities:
            raise ValueError("at least one slider quantity is required")

        self.quantities = {quantity.var_name: quantity for quantity in quantities}
        if len(self.quantities) != len(quantities):
            raise ValueError("slider quantity names must be unique")
        self.gauges = tuple(gauges)
        self.displays = tuple(displays)
        self.controller = controller
        self._suspend_events = False

        self.sliders = {
            name: quantity.make_widget()
            for name, quantity in self.quantities.items()
        }
        for name, slider in self.sliders.items():
            slider.observe(
                lambda change, driver=name: self._on_slider(driver, change),
                names="value",
            )

        options = [
            (constraint.label, constraint.name)
            for constraint in self.controller.constraints
        ]
        self.constraint_widget = widgets.Dropdown(
            options=options,
            value=self.controller.selection,
            description=self.controller.selection_label,
            style={"description_width": "80px"},
        )
        self.constraint_widget.observe(self._on_constraint, names="value")

        self.reset_button = widgets.Button(description="Reset")
        self.clear_button = widgets.Button(description="Clear chart")
        self.reset_button.on_click(self._on_reset)
        self.clear_button.on_click(self._on_clear)

        self.gauge_widgets = {
            gauge.name: widgets.HTML() for gauge in self.gauges
        }
        self.error_banner = widgets.HTML()
        self.trace_output = widgets.Output(
            layout={
                "max_height": "160px",
                "overflow": "auto",
                "border": "1px solid #ddd",
                "padding": "6px",
            }
        )
        self.trace_accordion = widgets.Accordion(children=[self.trace_output])
        self.trace_accordion.set_title(0, "Traceback")
        self.trace_accordion.selected_index = None

        slider_box = widgets.VBox(list(self.sliders.values()))
        controls = widgets.HBox(
            [
                slider_box,
                widgets.VBox(
                    [
                        self.constraint_widget,
                        widgets.HBox([self.clear_button, self.reset_button]),
                    ]
                ),
            ]
        )
        readouts = widgets.HBox(list(self.gauge_widgets.values()))
        display_box = widgets.HBox(
            [display.widget for display in self.displays],
            layout=widgets.Layout(flex_flow="row wrap"),
        )
        self._widget = widgets.VBox(
            [
                controls,
                readouts,
                display_box,
                self.error_banner,
                self.trace_accordion,
            ]
        )
        self.refresh()

    def ui(self) -> widgets.Widget:
        return self._widget

    def _on_slider(self, driver: str, change: Mapping[str, object]) -> None:
        if self._suspend_events or change.get("name") != "value":
            return
        try:
            new_value = change.get("new")
            if isinstance(new_value, bool) or not isinstance(
                new_value, (int, float)
            ):
                raise TypeError("slider value must be numeric")
            self.controller.set_driver(driver, float(new_value))
            self.refresh()
        except Exception as exc:
            self._show_error(exc)
            self._sync_controls()

    def _on_constraint(self, change: Mapping[str, object]) -> None:
        if self._suspend_events or change.get("name") != "value":
            return
        try:
            self.controller.set_constraint(str(change["new"]))
            self.refresh()
        except Exception as exc:
            self._show_error(exc)
            self._sync_controls()

    def _on_reset(self, _button: widgets.Button) -> None:
        try:
            self.controller.reset()
            self.refresh()
        except Exception as exc:
            self._show_error(exc)

    def _on_clear(self, _button: widgets.Button) -> None:
        try:
            self.controller.clear_history()
            self.refresh()
        except Exception as exc:
            self._show_error(exc)

    def refresh(self) -> None:
        self._clear_error()
        self._sync_controls()
        self._update_readouts()
        context = self.controller.render_context(self.quantities)
        for display in self.displays:
            display.render(context)

    def _sync_controls(self) -> None:
        self._suspend_events = True
        try:
            values = self.controller.values
            valid_drivers = self.controller.valid_drivers
            for name, slider in self.sliders.items():
                spec = self.quantities[name]
                display_value = spec.snap(float(values[name]))
                slider.min = min(float(spec.min_val), display_value)
                slider.max = max(float(spec.max_val), display_value)
                slider.value = display_value
                slider.disabled = name not in valid_drivers
            self.constraint_widget.value = self.controller.selection
        finally:
            self._suspend_events = False

    def _update_readouts(self) -> None:
        for gauge in self.gauges:
            text = gauge.format_value(self.controller.readout(gauge.name))
            self.gauge_widgets[gauge.name].value = html.escape(text)

    def _clear_error(self) -> None:
        self.error_banner.value = ""
        self.trace_accordion.selected_index = None
        self.trace_output.clear_output()

    def _show_error(self, exc: Exception) -> None:
        self.error_banner.value = (
            "<b>Interactive demo error:</b> " + html.escape(str(exc))
        )
        with self.trace_output:
            self.trace_output.clear_output()
            traceback.print_exc()
        self.trace_accordion.selected_index = 0
