from __future__ import annotations

from typing import Any, Optional, Sequence, cast

import ipywidgets as widgets

from ..controls import Gauge, SliderQuantity
from ..explorer import DemoDisplay, ExplorerPanel
from .model import (
    DEFAULT_CONSTRAINTS,
    Constraint,
    EquationOfState,
    ThermoSession,
    ThermoState,
)


class ThermoDemo:
    """Declarative thermodynamic explorer composed from controls and displays."""

    def __init__(
        self,
        slider_quantities: Sequence[SliderQuantity],
        constraints: Sequence[Constraint],
        gauges: Sequence[Gauge],
        charts: Sequence[DemoDisplay],
        equation_of_state: EquationOfState,
        initial_state: Optional[ThermoState] = None,
        initial_constraint: Optional[str] = None,
    ):
        quantity_defaults = {
            quantity.var_name: float(quantity.default)
            for quantity in slider_quantities
        }
        missing = {"P", "V", "T"} - quantity_defaults.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"thermo demos require sliders for {names}")

        if initial_state is None:
            volume = quantity_defaults["V"]
            temperature = quantity_defaults["T"]
            if not hasattr(equation_of_state, "n") or not hasattr(
                equation_of_state, "R"
            ):
                raise ValueError(
                    "initial_state is required for non-ideal-gas models"
                )
            ideal_gas = cast(Any, equation_of_state)
            pressure = (
                ideal_gas.n
                * ideal_gas.R
                * temperature
                / volume
            )
            initial_state = ThermoState(
                P=pressure, V=volume, T=temperature
            )

        selected_constraints = tuple(constraints) or DEFAULT_CONSTRAINTS
        self.session = ThermoSession(
            equation_of_state,
            initial_state,
            selected_constraints,
            initial_constraint=initial_constraint,
        )
        self.panel = ExplorerPanel(
            slider_quantities,
            gauges,
            charts,
            self.session,
        )

    def ui(self) -> widgets.Widget:
        return self.panel.ui()

