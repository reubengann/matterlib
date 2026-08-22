from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ipywidgets as widgets


@dataclass(frozen=True)
class SliderQuantity:
    """Declarative description of an editable numeric quantity."""

    var_name: str
    default: float
    min_val: float
    max_val: float
    step: float
    label: Optional[str] = None
    unit: str = ""
    readout_format: str = ".4g"
    continuous_update: bool = False

    def __post_init__(self) -> None:
        if not self.var_name:
            raise ValueError("var_name must not be empty")
        if self.min_val >= self.max_val:
            raise ValueError("min_val must be less than max_val")
        if self.step <= 0:
            raise ValueError("step must be positive")
        if not self.min_val <= self.default <= self.max_val:
            raise ValueError("default must lie within the slider bounds")

    @property
    def description(self) -> str:
        return self.label or self.var_name

    def snap(self, value: float) -> float:
        """Return the nearest value representable by this slider's step grid."""
        step_count = round((float(value) - self.min_val) / self.step)
        return float(self.min_val + step_count * self.step)

    def make_widget(self) -> widgets.FloatSlider:
        return widgets.FloatSlider(
            value=float(self.default),
            min=float(self.min_val),
            max=float(self.max_val),
            step=float(self.step),
            description=self.description,
            readout=True,
            readout_format=self.readout_format,
            continuous_update=self.continuous_update,
            style={"description_width": "80px"},
        )


@dataclass(frozen=True)
class Gauge:
    """A formatted read-only value supplied by a demo controller."""

    name: str
    label: Optional[str] = None
    unit: str = ""
    format_spec: str = ",.3g"

    @property
    def description(self) -> str:
        return self.label or self.name

    def format_value(self, value: object) -> str:
        if isinstance(value, (int, float)):
            rendered = format(value, self.format_spec)
        else:
            rendered = str(value)
        suffix = f" {self.unit}" if self.unit else ""
        return f"{self.description}: {rendered}{suffix}"
