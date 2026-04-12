import numpy as np
import pytest
from sympy.physics.units import convert_to
from sympy.physics.units import Quantity

from matterlib import spp


def test_constrained_partial_latex():
    P, v, T = spp.symbols("P v T")
    expr = spp.partial(P, v, hold=T)
    assert spp.latex(expr) == r"\left(\frac{\partial P}{\partial v}\right)_{T}"


def test_diff_implicit_returns_unsolved_equation():
    P, v, b, R, T = spp.symbols("P v b R T")
    eq = spp.Eq(P * (v - b), R * T)

    differentiated = eq.diff_implicit(wrt=T, dependent=v, hold=P)
    assert spp.simplify(differentiated.lhs - (P * spp.partial(v, T, hold=P))) == 0
    assert spp.simplify(differentiated.rhs - R) == 0


def test_partial_for_solves_constrained_derivative():
    P, v, b, R, T = spp.symbols("P v b R T")
    eq = spp.Eq(P * (v - b), R * T)

    result = eq.partial_for(v, T, hold=P)

    assert result.lhs == spp.partial(v, T, hold=P)
    assert spp.simplify(result.rhs - (R / P)) == 0


def test_partial_for_matches_residual_formula():
    P, R, T, a, b, v = spp.symbols("P R T a b v")
    eq = spp.Eq((P + a / v**2) * (v - b), R * T)

    result = eq.partial_for(P, v, hold=T)
    residual = eq.lhs - eq.rhs
    expected = -spp.diff(residual, v) / spp.diff(residual, P)

    assert result.lhs == spp.partial(P, v, hold=T)
    assert spp.simplify(result.rhs - expected) == 0


def test_regular_diff_is_unchanged():
    P, v = spp.symbols("P v")
    assert spp.diff(P * v, v) == P


def test_with_state_infers_dependent_for_diff():
    P, v, b, R, T = spp.symbols("P v b R T")
    eq = spp.Eq(P * (v - b), R * T).with_state(P, v, T)

    differentiated = eq.diff_implicit(wrt=T, hold=P)
    assert spp.simplify(differentiated.lhs - (P * spp.partial(v, T, hold=P))) == 0
    assert spp.simplify(differentiated.rhs - R) == 0


def test_with_state_infers_dependent_for_partial_for():
    P, v, b, R, T = spp.symbols("P v b R T")
    eq = spp.Eq(P * (v - b), R * T).with_state(P, v, T)

    result = eq.partial_for(wrt=T, hold=P)

    assert result.lhs == spp.partial(v, T, hold=P)
    assert spp.simplify(result.rhs - (R / P)) == 0


def test_with_state_repr_and_latex_delegate_to_equation():
    P, v, b, R, T = spp.symbols("P v b R T")
    state_eq = spp.Eq(P * (v - b), R * T).with_state(P, v, T)
    raw_eq = spp.Eq(P * (v - b), R * T)

    assert repr(state_eq) == repr(raw_eq)
    assert state_eq._repr_latex_() == raw_eq._repr_latex_()


def test_with_state_solve_for_passthrough():
    P, v, b, R, T = spp.symbols("P v b R T")
    state_eq = spp.Eq(P * (v - b), R * T).with_state(P, v, T)

    solved = state_eq.solve_for(v)
    expected = spp.Eq(v, R * T / P + b)

    assert spp.simplify(solved.lhs - expected.lhs) == 0
    assert spp.simplify(solved.rhs - expected.rhs) == 0


def test_with_state_preserves_state_on_equation_methods():
    P, v, b, R, T = spp.symbols("P v b R T")
    state_eq = spp.Eq(P * (v - b), R * T).with_state(P, v, T)

    solved = state_eq.solve_for(v)
    rediffed = solved.diff_implicit(wrt=T, hold=P)

    assert isinstance(solved, type(state_eq))
    assert solved.state_variables == state_eq.state_variables
    assert isinstance(rediffed, type(state_eq))
    assert spp.simplify(rediffed.lhs - spp.partial(v, T, hold=P)) == 0
    assert spp.simplify(rediffed.rhs - (R / P)) == 0


def test_diff_implicit_preserves_nonzero_rhs_terms():
    P, v, a, b, R, T = spp.symbols("P v a b R T")
    eq = spp.Eq(P * (v - b) * spp.exp(a / (v * R * T)), R * T).with_state(P, v, T)

    differentiated = eq.diff_implicit(T, hold=v)

    assert differentiated.rhs == R


def test_with_state_allows_chained_implicit_derivatives():
    P, R, T, a, b, v = spp.symbols("P R T a b v")
    eq = spp.Eq((P + a / v**2) * (v - b), R * T).with_state(v, T, P)

    second = eq.solve_for(P).diff_implicit(wrt=v, hold=T).diff_implicit(wrt=v, hold=T)

    assert isinstance(second, type(eq))


def test_solve_system_accepts_state_equations():
    x, y = spp.symbols("x y")
    eq1 = spp.Eq(x + y, 3).with_state(x, y)
    eq2 = spp.Eq(x - y, 1).with_state(x, y)

    solutions = spp.solve_system([eq1, eq2], [x, y])
    assert solutions == {x: 2, y: 1}


def test_R_kmol_matches_eval_constant():
    joule, kilogram, molar_gas_constant, kelvin, kilomole = spp.units(
        "joule kilogram molar_gas_constant kelvin kilomole"
    )
    expected = spp.eval_constant(molar_gas_constant, joule / (kelvin * kilomole))
    assert spp.simplify(spp.R_kmol() - expected) == 0


def test_state_equation_supports_mixed_equation_multiplication():
    x, y = spp.symbols("x y")
    state_eq = spp.Eq(x, 2).with_state(x, y)
    eq = spp.Eq(y, 3)

    product = state_eq * eq

    assert isinstance(product, type(state_eq))
    assert product.state_variables == state_eq.state_variables
    assert product.eq == spp.Eq(x * y, 6).eq


def test_reverse_division_for_equation():
    x, y = spp.symbols("x y")
    eq = spp.Eq(x, y)

    result = 1 / eq
    assert result.eq == spp.Eq(1 / x, 1 / y).eq


def test_reverse_multiplication_for_equation():
    x, y = spp.symbols("x y")
    eq = spp.Eq(x, y)

    result = 3 * eq
    assert result.eq == spp.Eq(3 * x, 3 * y).eq


def test_reverse_addition_for_equation():
    x, y = spp.symbols("x y")
    eq = spp.Eq(x, y)

    result = 3 + eq
    assert result.eq == spp.Eq(3 + x, 3 + y).eq


def test_reverse_subtraction_for_equation():
    x, y = spp.symbols("x y")
    eq = spp.Eq(x, y)

    result = 3 - eq
    assert result.eq == spp.Eq(3 - x, 3 - y).eq


def test_equation_evalf_shortcut():
    x = spp.symbols("x")
    eq = spp.Eq(x, spp.pi / 2)

    result = eq.evalf(6)

    assert result.lhs == x
    assert result.rhs == spp.N(spp.pi / 2, 6)


def test_subs_accepts_equation_directly():
    P, V, R, T = spp.symbols("P V R T")
    ideal_gas = spp.Eq(P * V, R * T)
    work_expr = spp.Eq(P + 1, 0)

    result = work_expr.subs(ideal_gas.solve_for(P))
    expected = spp.Eq(R * T / V + 1, 0)

    assert spp.simplify(result.lhs - expected.lhs) == 0
    assert result.rhs == expected.rhs


def test_subs_replaces_inside_constrained_partials_by_default():
    P, v, T = spp.symbols("P v T")
    eq = spp.Eq(spp.partial(v, T, hold=P) + v, 0)

    replaced = eq.subs({P: 2, v: 3, T: 4})
    expected = spp.Eq(spp.partial(3, 4, hold=2) + 3, 0)

    assert replaced.eq == expected.eq


def test_subs_can_preserve_constrained_partials():
    P, v, T = spp.symbols("P v T")
    eq = spp.Eq(spp.partial(v, T, hold=P) + v, 0)

    replaced = eq.subs({P: 2, v: 3, T: 4}, preserve_partials=True)
    expected = spp.Eq(spp.partial(v, T, hold=P) + 3, 0)

    assert replaced.eq == expected.eq


def test_subs_accepts_variadic_replacements():
    x, y, z = spp.symbols("x y z")
    eq = spp.Eq(x + y + z, 0)

    result = eq.subs(spp.Eq(x, 1), {y: 2}, spp.Eq(z, 3))

    assert result.eq == spp.Eq(6, 0).eq


def test_subs_rejects_non_mapping_non_equation():
    x, y = spp.symbols("x y")
    eq = spp.Eq(x + y, 0)

    with pytest.raises(TypeError, match="dict/Equation"):
        eq.subs((x, 1))


def test_combine_logs_equation_method():
    x, y = spp.symbols("x y")
    eq = spp.Eq(spp.log(y) - spp.log(x), 0)

    combined = eq.combine_logs()
    expected = spp.Eq(spp.log(y / x), 0)

    assert combined.eq == expected.eq


def test_combine_logs_namespace_helper():
    x, y = spp.symbols("x y")
    eq = spp.Eq(spp.log(y) - spp.log(x), 0)

    combined = spp.combine_logs(eq)
    expected = spp.Eq(spp.log(y / x), 0)

    assert combined.eq == expected.eq


def test_combine_logs_preserves_prefactor_outside_log():
    dS = spp.symbols("dS")
    joule, kelvin = spp.units("joule kelvin")
    eq = spp.Eq(
        dS,
        spp.log(1.20467337540508 ** (4184 * joule / kelvin)),
    )

    combined = spp.combine_logs(eq)

    assert combined.lhs == dS
    assert combined.rhs.has(spp.log)
    assert not (
        combined.rhs.func == spp.log
        and combined.rhs.args
        and isinstance(combined.rhs.args[0], spp.Pow)
    )
    expected = (4184 * joule / kelvin) * spp.log(1.20467337540508, evaluate=False)
    assert spp.simplify(combined.rhs - expected) == 0


def test_combine_logs_factors_symbolic_log_coefficients():
    dS, m, T_1, T_2, a, b = spp.symbols("dS m T_1 T_2 a b")
    eq = spp.Eq(dS, m * (-T_1 * b + T_2 * b - a * spp.log(T_1) + a * spp.log(T_2)))

    combined = spp.combine_logs(eq)

    expected = spp.Eq(dS, m * (-T_1 * b + T_2 * b + a * spp.log(T_2 / T_1)))
    assert spp.simplify(combined.rhs - expected.rhs) == 0
    assert str(combined.rhs).count("**a") == 0


def test_convert_to_handles_factored_unit_sum_terms():
    W = spp.symbols("W")
    atmosphere, meter, joule = spp.units("atmosphere meter joule")
    rhs = (
        64
        * 2 ** spp.Rational(1, 3)
        * atmosphere
        * meter**5
        * (-0.259108664993958 / meter**2 + 3 * 2 ** spp.Rational(2, 3) / (8 * meter**2))
    )
    eq = spp.Eq(W, rhs)

    converted = eq.convert_to(joule)

    assert converted.rhs.has(joule)
    assert not converted.rhs.has(meter)
    expected = eq.evalf().convert_to(joule).rhs
    assert float(spp.N(converted.rhs / joule)) == pytest.approx(float(spp.N(expected / joule)))


def test_lambdify_units_ideal_gas_vectorized_pressure():
    P, R, T, V, n = spp.symbols("P R T V n")
    kelvin, kilomole, meter, pascal = spp.units("kelvin kilomole meter pascal")
    pressure_expr = spp.Eq(P * V, R * T * n).solve_for(P).rhs.subs(
        {R: spp.R_kmol(), T: 300 * kelvin, n: 0.25 * kilomole}
    )

    pressure_fn = spp.lambdify_units(
        V,
        pressure_expr,
        inputs={V: meter**3},
        output=pascal,
        modules="numpy",
    )

    volumes = np.array([10.0, 8.0, 5.0])
    pressures = pressure_fn(volumes)

    expected = np.array(
        [
            float(
                spp.N(
                    convert_to(pressure_expr.subs({V: volume * meter**3}), pascal) / pascal
                )
            )
            for volume in volumes
        ]
    )
    np.testing.assert_allclose(pressures, expected)
    assert pressure_fn(10.0) == pytest.approx(expected[0])


def test_lambdify_units_respects_argument_order():
    x, y = spp.symbols("x y")
    meter = spp.units("meter")

    fn = spp.lambdify_units(
        (x, y),
        x + 2 * y,
        inputs={x: meter, y: meter},
        output=meter,
        modules="numpy",
    )

    assert fn(3.0, 4.0) == 11.0


def test_lambdify_units_requires_units_for_all_arguments():
    x, y = spp.symbols("x y")
    meter = spp.units("meter")

    with pytest.raises(ValueError, match="Missing unit declarations"):
        spp.lambdify_units((x, y), x + y, inputs={x: meter}, output=meter)


def test_lambdify_units_handles_nested_units_inside_sqrt():
    power, t, m, a, b, T, Tprime, T_1 = spp.symbols("P t m a b T Tprime T_1")
    eq1 = spp.Eq(power * t, spp.Integral((1 / m) * (a + b * Tprime), (Tprime, T_1, T)))
    eq2 = eq1.simplify().solve_for_all(T)[1]
    joule, kilogram, kelvin, watt, second = spp.units("joule kilogram kelvin watt second")
    eq3 = eq2.subs(
        {
            a: 750 * joule / (kilogram * kelvin),
            b: 0.5 * joule / (kilogram * kelvin**2),
            T_1: 300 * kelvin,
            power: 1 * watt,
            m: 0.01 * kilogram,
        }
    )

    fn = spp.lambdify_units(
        t,
        eq3.rhs,
        inputs={t: second},
        output=kelvin,
        modules="numpy",
    )

    ts = np.linspace(0, 5, 6)
    temperatures = fn(ts)
    converted_unitless = spp.simplify(convert_to(eq3.rhs.subs({t: t * second}), kelvin) / kelvin)
    for _ in range(8):
        quantities = converted_unitless.atoms(Quantity)
        if not quantities:
            break
        converted_unitless = spp.simplify(
            converted_unitless.xreplace(
                {quantity: spp.sympify(quantity.scale_factor) for quantity in quantities}
            )
        )
    expected = np.array(
        [float(spp.N(converted_unitless.subs({t: time_s}))) for time_s in ts]
    )

    np.testing.assert_allclose(temperatures, expected)
