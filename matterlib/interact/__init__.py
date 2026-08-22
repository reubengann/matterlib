"""Reusable, thread-free interactive demos for Jupyter."""

from .controls import Gauge, SliderQuantity
from .displays import (
    CanvasDisplay,
    PistonDisplay,
    SimpleChartDisplay,
    XYChartDisplay,
)
from .explorer import DemoDisplay, ExplorerPanel, RenderContext
from .thermo import (
    ADIABATIC,
    DEFAULT_CONSTRAINTS,
    ISOBARIC,
    ISOTHERMAL,
    Constraint,
    EquationOfState,
    IdealGasEquationOfState,
    IdealGasState,
    PathPoint,
    ProcessLeg,
    ThermoDemo,
    ThermoSession,
    ThermoState,
    TransitionResult,
)

__all__ = [
    "ADIABATIC",
    "DEFAULT_CONSTRAINTS",
    "ISOBARIC",
    "ISOTHERMAL",
    "CanvasDisplay",
    "Constraint",
    "DemoDisplay",
    "EquationOfState",
    "ExplorerPanel",
    "Gauge",
    "IdealGasEquationOfState",
    "IdealGasState",
    "PathPoint",
    "PistonDisplay",
    "ProcessLeg",
    "RenderContext",
    "SimpleChartDisplay",
    "SliderQuantity",
    "ThermoDemo",
    "ThermoSession",
    "ThermoState",
    "TransitionResult",
    "XYChartDisplay",
]
