from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from ..controls import SliderQuantity
from ..explorer import RenderContext


@dataclass(frozen=True)
class ThermoState:
    """A thermodynamic state snapshot.

    ``phase``, ``quality``, and ``extras`` leave room for future multiphase
    models without changing the interactive-demo machinery.
    """

    P: float
    V: float
    T: float
    phase: str = "gas"
    quality: Optional[float] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def value(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        try:
            return self.extras[name]
        except KeyError as exc:
            raise KeyError(f"unknown thermodynamic quantity {name!r}") from exc


@dataclass(frozen=True, init=False)
class Constraint:
    """A selectable process constraint and its valid independent variables."""

    name: str
    label: str
    valid_drivers: frozenset[str]

    def __init__(
        self,
        name: str,
        label: Optional[str] = None,
        valid_drivers: Optional[frozenset[str]] = None,
    ) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("constraint name must not be empty")
        object.__setattr__(self, "name", normalized)
        object.__setattr__(self, "label", label or normalized.title())
        known_drivers = {
            "isothermal": frozenset({"P", "V"}),
            "isobaric": frozenset({"V", "T"}),
            "adiabatic": frozenset({"P", "V", "T"}),
        }
        object.__setattr__(
            self,
            "valid_drivers",
            (
                valid_drivers
                if valid_drivers is not None
                else known_drivers.get(
                    normalized, frozenset({"P", "V", "T"})
                )
            ),
        )


ISOTHERMAL = Constraint(
    "isothermal", "Isothermal", frozenset({"P", "V"})
)
ISOBARIC = Constraint("isobaric", "Isobaric", frozenset({"V", "T"}))
ADIABATIC = Constraint(
    "adiabatic", "Adiabatic", frozenset({"P", "V", "T"})
)
DEFAULT_CONSTRAINTS = (ISOTHERMAL, ISOBARIC, ADIABATIC)


@dataclass(frozen=True)
class TransitionResult:
    state: ThermoState
    constraint: str
    driver: str


class EquationOfState(Protocol):
    """Structural interface for ideal-gas and future phase-aware models."""

    def solve(
        self,
        anchor: ThermoState,
        constraint: Constraint,
        driver: str,
        value: float,
    ) -> TransitionResult:
        ...

    def internal_energy_change(
        self, start: ThermoState, end: ThermoState
    ) -> float:
        ...

    def process_work(self, leg: "ProcessLeg") -> float:
        ...


@dataclass(frozen=True)
class ProcessLeg:
    start: ThermoState
    end: ThermoState
    constraint: str
    driver: str


@dataclass(frozen=True)
class PathPoint:
    state: ThermoState
    constraint: str


class IdealGasEquationOfState:
    """Numerical ideal-gas process solver used by interactive demos."""

    def __init__(
        self, n: float = 1.0, R: float = 8.314, gamma: float = 1.4
    ):
        if n <= 0 or R <= 0:
            raise ValueError("n and R must be positive")
        if gamma <= 1:
            raise ValueError("gamma must be greater than one")
        self.n = float(n)
        self.R = float(R)
        self.gamma = float(gamma)

    def solve(
        self,
        anchor: ThermoState,
        constraint: Constraint,
        driver: str,
        value: float,
    ) -> TransitionResult:
        if driver not in constraint.valid_drivers:
            raise ValueError(
                f"{driver} cannot drive an {constraint.name} process"
            )
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{driver} must be a positive finite value")

        nR = self.n * self.R
        if constraint.name == "isothermal":
            T = anchor.T
            if driver == "P":
                P, V = value, nR * T / value
            elif driver == "V":
                V, P = value, nR * T / value
            else:  # guarded by valid_drivers
                raise ValueError(driver)
        elif constraint.name == "isobaric":
            P = anchor.P
            if driver == "V":
                V, T = value, P * value / nR
            elif driver == "T":
                T, V = value, nR * value / P
            else:
                raise ValueError(driver)
        elif constraint.name == "adiabatic":
            gamma = self.gamma
            pv_gamma = anchor.P * anchor.V**gamma
            tv_gamma = anchor.T * anchor.V ** (gamma - 1.0)
            if driver == "P":
                P = value
                V = (pv_gamma / P) ** (1.0 / gamma)
                T = tv_gamma / V ** (gamma - 1.0)
            elif driver == "V":
                V = value
                P = pv_gamma / V**gamma
                T = tv_gamma / V ** (gamma - 1.0)
            elif driver == "T":
                T = value
                V = (tv_gamma / T) ** (1.0 / (gamma - 1.0))
                P = pv_gamma / V**gamma
            else:
                raise ValueError(driver)
        else:
            raise ValueError(f"unsupported constraint {constraint.name!r}")

        state = ThermoState(P=P, V=V, T=T)
        self._validate_state(state)
        return TransitionResult(state, constraint.name, driver)

    def internal_energy_change(
        self, start: ThermoState, end: ThermoState
    ) -> float:
        cv = self.R / (self.gamma - 1.0)
        return self.n * cv * (end.T - start.T)

    def process_work(self, leg: ProcessLeg) -> float:
        """Return exact work done by the gas along an ideal-gas process leg."""
        delta_volume = leg.end.V - leg.start.V
        if delta_volume == 0:
            return 0.0
        if leg.constraint == "isothermal":
            return (
                self.n
                * self.R
                * leg.start.T
                * math.log(leg.end.V / leg.start.V)
            )
        if leg.constraint == "isobaric":
            return leg.start.P * delta_volume
        if leg.constraint == "adiabatic":
            return (
                leg.end.P * leg.end.V - leg.start.P * leg.start.V
            ) / (1.0 - self.gamma)
        raise ValueError(f"unsupported constraint {leg.constraint!r}")

    @staticmethod
    def _validate_state(state: ThermoState) -> None:
        values = (state.P, state.V, state.T)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("the process produced a nonphysical state")


# Compatibility with the API sketched in piston2.ipynb.
IdealGasState = IdealGasEquationOfState


DEFAULT_THEME: Mapping[str, Any] = {
    "bg": "#1e1e1e",
    "stroke": "#cccccc",
    "text": "#dddddd",
    "piston": "#888888",
    "rod": "#aaaaaa",
    "axis": "#888888",
    "path": "#4da6ff",
    "point": "#ff5555",
    "process_colors": {
        "isothermal": "#4da6ff",
        "isobaric": "#34c759",
        "adiabatic": "#ff9f0a",
    },
}


class ThermoSession:
    """Owns process anchors, history, derived energy, and the current state."""

    selection_label = "Constraint"

    def __init__(
        self,
        equation_of_state: EquationOfState,
        initial_state: ThermoState,
        constraints: Sequence[Constraint] = DEFAULT_CONSTRAINTS,
        initial_constraint: Optional[str] = None,
        samples_per_leg: int = 1,
        theme: Mapping[str, Any] = DEFAULT_THEME,
    ):
        if not constraints:
            raise ValueError("at least one constraint is required")
        self.equation_of_state = equation_of_state
        self._constraints = tuple(constraints)
        self._constraint_map = {
            constraint.name: constraint for constraint in self._constraints
        }
        if len(self._constraint_map) != len(self._constraints):
            raise ValueError("constraint names must be unique")
        self._initial_state = initial_state
        self._initial_constraint = (
            initial_constraint or self._constraints[0].name
        ).lower()
        if self._initial_constraint not in self._constraint_map:
            raise ValueError("initial_constraint is not in constraints")
        if samples_per_leg < 1:
            raise ValueError("samples_per_leg must be at least one")
        self.samples_per_leg = int(samples_per_leg)
        self.theme = theme
        self.reset()

    @property
    def constraints(self) -> Sequence[Constraint]:
        return self._constraints

    @property
    def selection(self) -> str:
        return self._constraint.name

    @property
    def values(self) -> Mapping[str, float]:
        values = {"P": self.state.P, "V": self.state.V, "T": self.state.T}
        values.update(
            {
                name: float(value)
                for name, value in self.state.extras.items()
                if isinstance(value, (int, float))
            }
        )
        return values

    @property
    def valid_drivers(self) -> frozenset[str]:
        return self._constraint.valid_drivers

    @property
    def legs(self) -> Tuple[ProcessLeg, ...]:
        return tuple(self._trace_legs)

    def set_driver(self, name: str, value: float) -> None:
        previous = self.state
        result = self.equation_of_state.solve(
            self.anchor, self._constraint, name, value
        )
        self.state = result.state
        if self.state != previous:
            self._trace_legs.append(
                ProcessLeg(
                    previous,
                    self.state,
                    self.selection,
                    name,
                )
            )

    def set_constraint(self, name: str) -> None:
        normalized = name.lower()
        try:
            new_constraint = self._constraint_map[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown constraint {name!r}") from exc
        if new_constraint.name == self._constraint.name:
            return
        self.anchor = self.state
        self._constraint = new_constraint

    def reset(self) -> None:
        self.state = self._initial_state
        self.anchor = self._initial_state
        self.baseline = self._initial_state
        self._constraint = self._constraint_map[self._initial_constraint]
        self._trace_legs: list[ProcessLeg] = []

    def clear_history(self) -> None:
        self._trace_legs = []
        self.anchor = self.state
        self.baseline = self.state

    def readout(self, name: str) -> object:
        if name == "work":
            return self.net_work
        if name == "heat":
            return self.net_heat
        return self.state.value(name)

    @property
    def net_work(self) -> float:
        return sum(
            self.equation_of_state.process_work(leg) for leg in self.legs
        )

    @property
    def net_heat(self) -> float:
        delta_u = self.equation_of_state.internal_energy_change(
            self.baseline, self.state
        )
        return delta_u + self.net_work

    def sample_path(self) -> Tuple[PathPoint, ...]:
        legs = self.legs
        if not legs:
            return (PathPoint(self.state, self.selection),)

        points: list[PathPoint] = []
        for leg in legs:
            constraint = self._constraint_map[leg.constraint]
            for index in range(self.samples_per_leg + 1):
                if points and index == 0:
                    continue
                fraction = index / self.samples_per_leg
                start_value = float(leg.start.value(leg.driver))
                end_value = float(leg.end.value(leg.driver))
                driver_value = start_value + fraction * (
                    end_value - start_value
                )
                state = self.equation_of_state.solve(
                    leg.start, constraint, leg.driver, driver_value
                ).state
                points.append(PathPoint(state, leg.constraint))
        return tuple(points)

    def render_context(
        self, quantities: Mapping[str, SliderQuantity]
    ) -> RenderContext:
        return RenderContext(
            state=self.state,
            history=self.sample_path(),
            selection=self.selection,
            quantities=quantities,
            theme=self.theme,
        )
