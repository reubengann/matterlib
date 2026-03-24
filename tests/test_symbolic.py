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
