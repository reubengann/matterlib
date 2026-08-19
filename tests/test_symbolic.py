import numpy as np
import pytest
from sympy.physics.units import convert_to
from sympy.physics.units import Quantity

from matterlib import spp
from matterlib import symbolic as symbolic_mod


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

    result = eq.solve_for_partial(v, T, hold=P)

    assert result.lhs == spp.partial(v, T, hold=P)
    assert spp.simplify(result.rhs - (R / P)) == 0


def test_partial_for_matches_residual_formula():
    P, R, T, a, b, v = spp.symbols("P R T a b v")
    eq = spp.Eq((P + a / v**2) * (v - b), R * T)

    result = eq.solve_for_partial(P, v, hold=T)
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


def test_nsolve_accepts_equation():
    x = spp.symbols("x")
    eq = spp.Eq(x**2, 4)

    solution = spp.nsolve(eq, x, 1)

    assert float(solution) == pytest.approx(2.0)


def test_nsolve_accepts_equation_system():
    x, y = spp.symbols("x y")
    eq1 = spp.Eq(x + y, 3).with_state(x, y)
    eq2 = spp.Eq(x - y, 1)

    solution = spp.nsolve([eq1, eq2], [x, y], [1, 1])

    assert float(solution[0]) == pytest.approx(2.0)
    assert float(solution[1]) == pytest.approx(1.0)


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


def test_subs_preserves_constrained_partials_by_default():
    P, v, T = spp.symbols("P v T")
    eq = spp.Eq(spp.partial(v, T, hold=P) + v, 0)

    replaced = eq.subs({P: 2, v: 3, T: 4})
    expected = spp.Eq(spp.partial(v, T, hold=P) + 3, 0)

    assert replaced.eq == expected.eq


def test_subs_can_replace_inside_constrained_partials_when_requested():
    P, v, T = spp.symbols("P v T")
    eq = spp.Eq(spp.partial(v, T, hold=P) + v, 0)

    replaced = eq.subs({P: 2, v: 3, T: 4}, preserve_partials=False)
    expected = spp.Eq(spp.partial(3, 4, hold=2) + 3, 0)

    assert replaced.eq == expected.eq


def test_subs_can_preserve_constrained_partials():
    P, v, T = spp.symbols("P v T")
    eq = spp.Eq(spp.partial(v, T, hold=P) + v, 0)

    replaced = eq.subs({P: 2, v: 3, T: 4}, preserve_partials=True)
    expected = spp.Eq(spp.partial(v, T, hold=P) + 3, 0)

    assert replaced.eq == expected.eq


def test_subs_preserves_partial_argument_when_replacing_solved_variable():
    P, R, T, a, b, beta, c_P, h, v = spp.symbols("P R T a b beta c_P h v")
    van_der_waals_eqn_of_state = spp.Eq((P + a / v**2) * (-b + v), R * T).with_state(
        P, v, T
    )
    eq = spp.Eq(spp.partial(T, P, hold=h), v * (T * beta - 1) / c_P)

    replaced = eq.subs(van_der_waals_eqn_of_state.solve_for(P))

    assert replaced.lhs == spp.partial(T, P, hold=h)
    assert replaced.rhs == v * (T * beta - 1) / c_P


def test_subs_preserve_partials_allows_whole_partial_replacement():
    P, R, T, a, b, beta, v = spp.symbols("P R T a b beta v")
    partial = spp.partial(v, T, hold=P)
    eq1 = spp.Eq(partial, R * v**3 / (P * v**3 + 2 * a * b - a * v))
    expansivity_definition = spp.Eq(beta, partial / v)

    replaced = expansivity_definition.subs(eq1, preserve_partials=True)

    expected = spp.Eq(beta, R * v**2 / (P * v**3 + 2 * a * b - a * v))
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


def test_subs_rhs_accepts_variadic_replacements():
    x, y, z = spp.symbols("x y z")
    eq = spp.Eq(x, y + z)

    result = eq.subs_rhs(spp.Eq(y, 2), {z: 3})

    assert result.eq == spp.Eq(x, 5).eq


def test_subs_rhs_preserves_constrained_partials_by_default():
    P, v, T = spp.symbols("P v T")
    eq = spp.Eq(0, spp.partial(v, T, hold=P) + v)

    replaced = eq.subs_rhs({P: 2, v: 3, T: 4})
    expected = spp.Eq(0, spp.partial(v, T, hold=P) + 3)

    assert replaced.eq == expected.eq


def test_subs_rhs_can_replace_inside_constrained_partials_when_requested():
    P, v, T = spp.symbols("P v T")
    eq = spp.Eq(0, spp.partial(v, T, hold=P) + v)

    replaced = eq.subs_rhs({P: 2, v: 3, T: 4}, preserve_partials=False)
    expected = spp.Eq(0, spp.partial(3, 4, hold=2) + 3)

    assert replaced.eq == expected.eq


def test_subs_rhs_can_preserve_constrained_partials():
    P, v, T = spp.symbols("P v T")
    eq = spp.Eq(0, spp.partial(v, T, hold=P) + v)

    replaced = eq.subs_rhs({P: 2, v: 3, T: 4}, preserve_partials=True)
    expected = spp.Eq(0, spp.partial(v, T, hold=P) + 3)

    assert replaced.eq == expected.eq


def test_subs_rhs_preserve_partials_allows_whole_partial_replacement():
    P, R, T, a, b, beta, v = spp.symbols("P R T a b beta v")
    partial = spp.partial(v, T, hold=P)
    eq1 = spp.Eq(partial, R * v**3 / (P * v**3 + 2 * a * b - a * v))
    expansivity_definition = spp.Eq(beta, partial / v)

    replaced = expansivity_definition.subs_rhs(eq1, preserve_partials=True)

    expected = spp.Eq(beta, R * v**2 / (P * v**3 + 2 * a * b - a * v))
    assert replaced.eq == expected.eq


def test_subs_rhs_rejects_non_mapping_non_equation():
    x, y = spp.symbols("x y")
    eq = spp.Eq(x, y)

    with pytest.raises(TypeError, match="dict/Equation"):
        eq.subs_rhs((y, 1))


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
    assert float(spp.N(converted.rhs / joule)) == pytest.approx(
        float(spp.N(expected / joule))
    )


def test_convert_to_prefers_target_unit_basis_for_pressure_expressions():
    kelvin, meter, newton, atmosphere, joule = spp.units(
        "kelvin meter newton atmosphere joule"
    )
    beta_val = 5.35e-2 / kelvin
    temp_val = 6 * kelvin
    kappa_val = 9.42e-8 / (newton / meter**2)
    press_val = 19.7 * atmosphere
    P, T, beta, kappa, u, v = spp.symbols("P T beta kappa u v")

    converted = (
        spp.Eq(spp.partial(u, v, hold=T), -P + T * beta / kappa)
        .subs(
            {P: press_val, T: temp_val, beta: beta_val, kappa: kappa_val},
            preserve_partials=True,
        )
        .convert_to(joule / meter**3)
    )

    assert converted.rhs.has(joule)
    assert converted.rhs.has(meter)
    assert not converted.rhs.has(newton)
    expected_value = 1411540.81210191
    assert float(spp.N(converted.rhs / (joule / meter**3))) == pytest.approx(
        expected_value
    )


def test_normalize_units_decimal_evaluates_nested_numeric_radicals():
    R, T, a, b, v = spp.symbols("R T a b v")
    joule, meter, kilomole, kelvin = spp.units("joule meter kilomole kelvin")
    eq = spp.Eq(v, b / (-spp.sqrt(2) * spp.sqrt(R * T * b / a) / 2 + 1)).subs(
        {
            a: 3.44e3 * joule * meter**3 / kilomole**2,
            b: 0.0234 * meter**3 / kilomole,
            R: spp.R_kmol(),
            T: 20 * kelvin,
        }
    )

    normalized = eq.normalize_units(decimal=True)

    assert not normalized.rhs.has(spp.sqrt)
    assert float(spp.N(normalized.rhs / (meter**3 / kilomole))) == pytest.approx(
        0.0943732622042842
    )


def test_normalize_units_sigfigs_uses_decimal_significant_figures_with_units():
    p = spp.symbols("p")
    atmosphere = spp.units("atmosphere")

    normalized = spp.Eq(p, 2 * atmosphere / 3).normalize_units(sigfigs=3)

    assert normalized.rhs == 0.667 * atmosphere


def test_scientific_notation_expression_helper():
    expr = spp.sympify("2746595.14017649*x + 0.0001234")
    converted = spp.scientific_notation(expr, sigfigs=5)

    assert ("10**6" in str(converted)) or ("e+6" in str(converted))
    assert ("10**(-4)" in str(converted)) or ("/10**4" in str(converted))


def test_scientific_notation_equation_helper():
    x = spp.symbols("x")
    eq = spp.Eq(x, 2746595.14017649)

    converted = spp.scientific_notation(eq, sigfigs=4)

    assert ("10**6" in str(converted.rhs)) or ("e+6" in str(converted.rhs))


def test_equation_latex_scientific_fluent_output():
    x = spp.symbols("x")
    eq = spp.Eq(x, 2746595.14017649)
    captured: list[tuple[object, bool, int]] = []

    def fake_display_latex(expr, scientific: bool = False, sigfigs: int = 6):
        captured.append((expr, scientific, sigfigs))

    original_display_latex = symbolic_mod.display_latex
    symbolic_mod.display_latex = fake_display_latex
    try:
        returned = eq.latex_scientific(sigfigs=4)
    finally:
        symbolic_mod.display_latex = original_display_latex

    assert returned is eq
    assert len(captured) == 1
    _, scientific, sigfigs = captured[0]
    assert scientific is True
    assert sigfigs == 4


def test_spp_latex_scientific_helper():
    x = spp.symbols("x")
    eq = spp.Eq(x, 0.0179256)
    captured: list[tuple[object, bool, int]] = []

    def fake_display_latex(expr, scientific: bool = False, sigfigs: int = 6):
        captured.append((expr, scientific, sigfigs))

    original_display_latex = symbolic_mod.display_latex
    symbolic_mod.display_latex = fake_display_latex
    try:
        returned = spp.latex_scientific(eq, sigfigs=6)
    finally:
        symbolic_mod.display_latex = original_display_latex

    assert returned is eq
    assert len(captured) == 1
    _, scientific, sigfigs = captured[0]
    assert scientific is True
    assert sigfigs == 6


def test_spp_latex_scientific_places_units_at_end():
    x = spp.symbols("x")
    meter, kilomole = spp.units("meter kilomole")
    raw_eq = spp.Eq(x, 0.0179256 * meter**3 / kilomole).unwrap()

    latex = spp.latex_scientific(raw_eq, sigfigs=6)

    assert "1.79256 \\times 10^{-2} \\frac{\\text{m}^{3}}{\\text{kmol}}" in latex


def test_spp_latex_scientific_spaces_adjacent_units():
    x = spp.symbols("x")
    joule, kilomole = spp.units("joule kilomole")
    raw_eq = spp.Eq(x, 3 * joule * kilomole).unwrap()

    latex = spp.latex_scientific(raw_eq, sigfigs=6)

    assert "\\text{J} \\, \\text{kmol}" in latex


def test_equation_display_fluent_output():
    x = spp.symbols("x")
    eq = spp.Eq(x, 1)
    captured: list[object] = []

    def fake_display_expr(expr):
        captured.append(expr)

    original_display_expr = symbolic_mod.display_expr
    symbolic_mod.display_expr = fake_display_expr
    try:
        returned = eq._display()
    finally:
        symbolic_mod.display_expr = original_display_expr

    assert returned is eq
    assert captured == [eq.eq]


def test_spp_display_helper_returns_equation():
    x = spp.symbols("x")
    eq = spp.Eq(x, 1)
    captured: list[object] = []

    def fake_display_expr(expr):
        captured.append(expr)

    original_display_expr = symbolic_mod.display_expr
    symbolic_mod.display_expr = fake_display_expr
    try:
        returned = spp.display(eq)
    finally:
        symbolic_mod.display_expr = original_display_expr

    assert returned is eq
    assert captured == [eq.eq]


def test_lambdify_units_ideal_gas_vectorized_pressure():
    P, R, T, V, n = spp.symbols("P R T V n")
    kelvin, kilomole, meter, pascal = spp.units("kelvin kilomole meter pascal")
    pressure_expr = (
        spp.Eq(P * V, R * T * n)
        .solve_for(P)
        .rhs.subs({R: spp.R_kmol(), T: 300 * kelvin, n: 0.25 * kilomole})
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
                    convert_to(pressure_expr.subs({V: volume * meter**3}), pascal)
                    / pascal
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
    joule, kilogram, kelvin, watt, second = spp.units(
        "joule kilogram kelvin watt second"
    )
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
    converted_unitless = spp.simplify(
        convert_to(eq3.rhs.subs({t: t * second}), kelvin) / kelvin
    )
    for _ in range(8):
        quantities = converted_unitless.atoms(Quantity)
        if not quantities:
            break
        converted_unitless = spp.simplify(
            converted_unitless.xreplace(
                {
                    quantity: spp.sympify(quantity.scale_factor)
                    for quantity in quantities
                }
            )
        )
    expected = np.array(
        [float(spp.N(converted_unitless.subs({t: time_s}))) for time_s in ts]
    )

    np.testing.assert_allclose(temperatures, expected)
