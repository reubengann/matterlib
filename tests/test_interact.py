import ipywidgets as widgets

from matterlib.interact import (
    Constraint,
    DEFAULT_CONSTRAINTS,
    ExplorerPanel,
    Gauge,
    IdealGasEquationOfState,
    PistonDisplay,
    SimpleChartDisplay,
    SliderQuantity,
    ThermoDemo,
    ThermoSession,
    ThermoState,
)


class RecordingDisplay:
    def __init__(self) -> None:
        self._widget = widgets.HTML()
        self.contexts = []
        self.fail = False

    @property
    def widget(self) -> widgets.Widget:
        return self._widget

    def render(self, context) -> None:
        if self.fail:
            raise RuntimeError("display failed")
        self.contexts.append(context)


def make_panel():
    eos = IdealGasEquationOfState()
    volume = eos.n * eos.R * 300.0 / 101_325.0
    state = ThermoState(P=101_325.0, V=volume, T=300.0)
    session = ThermoSession(eos, state, DEFAULT_CONSTRAINTS)
    quantities = [
        SliderQuantity("P", state.P, 50_000, 200_000, 100, unit="Pa"),
        SliderQuantity("V", state.V, 0.01, 0.05, 0.0001, unit="m³"),
        SliderQuantity("T", state.T, 200, 500, 1, unit="K"),
    ]
    display = RecordingDisplay()
    panel = ExplorerPanel(
        quantities,
        [Gauge("P", unit="Pa"), Gauge("work", unit="J")],
        [display],
        session,
    )
    return panel, session, display


def test_slider_event_solves_and_renders_once_without_observer_recursion():
    panel, session, display = make_panel()
    initial_render_count = len(display.contexts)

    panel.sliders["V"].value = 0.03

    assert session.state.V == 0.03
    assert len(display.contexts) == initial_render_count + 1
    assert panel.sliders["P"].value == panel.quantities["P"].snap(
        session.state.P
    )
    assert panel.sliders["T"].disabled


def test_derived_slider_thumbs_stay_on_grid_without_driving_feedback():
    panel, session, display = make_panel()
    initial_render_count = len(display.contexts)

    panel.sliders["P"].value = 90_000

    assert session.state.P == 90_000
    assert panel.sliders["V"].value == panel.quantities["V"].snap(
        session.state.V
    )
    assert panel.sliders["V"].value != session.state.V
    assert len(session.legs) == 1
    assert len(display.contexts) == initial_render_count + 1


def test_constraint_change_reanchors_and_updates_valid_drivers():
    panel, session, _display = make_panel()
    panel.sliders["V"].value = 0.03
    state_before_switch = session.state

    panel.constraint_widget.value = "isobaric"

    assert session.state == state_before_switch
    assert panel.sliders["P"].disabled
    assert not panel.sliders["V"].disabled
    assert not panel.sliders["T"].disabled


def test_clear_and_reset_buttons_apply_atomically():
    panel, session, display = make_panel()
    panel.sliders["V"].value = 0.03
    panel.sliders["V"].value = 0.025
    changed = session.state
    assert len(session.legs) == 2
    assert len(display.contexts[-1].history) == 3

    panel.clear_button.click()
    assert session.state == changed
    assert session.net_work == 0
    assert session.legs == ()
    assert len(display.contexts[-1].history) == 1

    render_count = len(display.contexts)
    panel.reset_button.click()
    assert session.state == session.anchor
    assert session.selection == "isothermal"
    assert len(display.contexts) == render_count + 1


def test_readouts_are_formatted_and_render_errors_are_visible():
    panel, _session, display = make_panel()
    assert "P:" in panel.gauge_widgets["P"].value
    assert "Pa" in panel.gauge_widgets["P"].value

    display.fail = True
    panel.sliders["V"].value = 0.03

    assert "display failed" in panel.error_banner.value


def test_declarative_thermo_demo_constructs_real_canvas_displays():
    eos = IdealGasEquationOfState()
    volume = eos.n * eos.R * 300.0 / 101_325.0
    quantities = [
        SliderQuantity("P", 101_325, 50_000, 200_000, 100),
        SliderQuantity("V", volume, 0.01, 0.05, 0.0001),
        SliderQuantity("T", 300, 200, 500, 1),
    ]
    piston = PistonDisplay()
    chart = SimpleChartDisplay("V", "P")

    demo = ThermoDemo(
        quantities,
        [
            Constraint("Isothermal"),
            Constraint("Isobaric"),
            Constraint("Adiabatic"),
        ],
        [Gauge("work", unit="J"), Gauge("heat", unit="J")],
        [piston, chart],
        eos,
    )

    assert isinstance(demo.ui(), widgets.Widget)
    assert demo.panel.sliders["T"].disabled
    demo.panel.sliders["V"].value = 0.03
    assert demo.session.state.V == 0.03
