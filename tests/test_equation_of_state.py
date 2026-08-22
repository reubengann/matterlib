import math

import pytest

from matterlib.interact import (
    ADIABATIC,
    ISOBARIC,
    ISOTHERMAL,
    Constraint,
    IdealGasEquationOfState,
    ThermoSession,
    ThermoState,
)


@pytest.fixture
def eos() -> IdealGasEquationOfState:
    return IdealGasEquationOfState()


@pytest.fixture
def initial(eos: IdealGasEquationOfState) -> ThermoState:
    volume = eos.n * eos.R * 300.0 / 101_325.0
    return ThermoState(P=101_325.0, V=volume, T=300.0)


@pytest.mark.parametrize(
    ("constraint", "driver", "value"),
    [
        (ISOTHERMAL, "P", 90_000.0),
        (ISOTHERMAL, "V", 0.03),
        (ISOBARIC, "V", 0.03),
        (ISOBARIC, "T", 340.0),
        (ADIABATIC, "P", 90_000.0),
        (ADIABATIC, "V", 0.03),
        (ADIABATIC, "T", 320.0),
    ],
)
def test_valid_constraint_driver_pairs_preserve_invariants(
    eos: IdealGasEquationOfState,
    initial: ThermoState,
    constraint: Constraint,
    driver: str,
    value: float,
) -> None:
    result = eos.solve(initial, constraint, driver, value)
    state = result.state

    assert state.P * state.V == pytest.approx(eos.n * eos.R * state.T)
    if constraint.name == "isothermal":
        assert state.T == pytest.approx(initial.T)
    elif constraint.name == "isobaric":
        assert state.P == pytest.approx(initial.P)
    else:
        assert state.P * state.V**eos.gamma == pytest.approx(
            initial.P * initial.V**eos.gamma
        )


def test_known_constraint_names_infer_valid_drivers() -> None:
    assert Constraint("Isothermal").valid_drivers == frozenset({"P", "V"})
    assert Constraint("Isobaric").valid_drivers == frozenset({"V", "T"})
    assert Constraint("Adiabatic").valid_drivers == frozenset(
        {"P", "V", "T"}
    )


def test_degenerate_driver_is_rejected(
    eos: IdealGasEquationOfState, initial: ThermoState
) -> None:
    with pytest.raises(ValueError, match="cannot drive"):
        eos.solve(initial, ISOTHERMAL, "T", 310.0)
    with pytest.raises(ValueError, match="cannot drive"):
        eos.solve(initial, ISOBARIC, "P", 100_000.0)


def test_constraint_switch_reanchors_without_state_jump(
    eos: IdealGasEquationOfState, initial: ThermoState
) -> None:
    session = ThermoSession(eos, initial)
    session.set_driver("V", 0.03)
    before = session.state

    session.set_constraint("adiabatic")

    assert session.state is before
    assert session.anchor is before
    assert len(session.legs) == 1


def test_work_and_heat_do_not_depend_on_slider_event_count(
    eos: IdealGasEquationOfState, initial: ThermoState
) -> None:
    direct = ThermoSession(eos, initial)
    direct.set_driver("V", 0.035)

    noisy = ThermoSession(eos, initial)
    for volume in (0.026, 0.031, 0.028, 0.035):
        noisy.set_driver("V", volume)

    assert noisy.state == direct.state
    assert noisy.net_work == pytest.approx(direct.net_work)
    assert noisy.net_heat == pytest.approx(direct.net_heat)
    assert len(noisy.legs) == 4
    assert [point.state.V for point in noisy.sample_path()] == pytest.approx(
        [initial.V, 0.026, 0.031, 0.028, 0.035]
    )


def test_process_leg_sampling_uses_the_last_moved_driver(
    eos: IdealGasEquationOfState, initial: ThermoState
) -> None:
    session = ThermoSession(eos, initial)
    session.set_constraint("adiabatic")
    session.set_driver("T", 330.0)

    assert session.legs[0].driver == "T"
    assert session.sample_path()[-1].state == session.state


def test_clear_and_reset_have_distinct_scopes(
    eos: IdealGasEquationOfState, initial: ThermoState
) -> None:
    session = ThermoSession(eos, initial)
    session.set_driver("V", 0.03)
    current = session.state

    session.clear_history()
    assert session.state == current
    assert session.anchor == current
    assert session.net_work == 0
    assert session.net_heat == 0

    session.set_constraint("adiabatic")
    session.set_driver("V", 0.025)
    assert not math.isclose(session.net_work, 0.0)

    session.reset()
    assert session.state == initial
    assert session.selection == "isothermal"
    assert session.net_work == 0
