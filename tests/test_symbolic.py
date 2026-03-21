from matterlib import spp


def test_constrained_partial_latex():
    P, v, T = spp.symbols("P v T")
    expr = spp.partial(P, v, hold=T)
    assert spp.latex(expr) == r"\left(\frac{\partial P}{\partial v}\right)_{T}"


def test_diff_implicit_returns_unsolved_equation():
    P, v, b, R, T = spp.symbols("P v b R T")
    eq = spp.Eq(P * (v - b), R * T)

    differentiated = eq.diff_implicit(wrt=T, dependent=v, hold=P)
    expected = P * spp.partial(v, T, hold=P) - R

    assert spp.simplify(differentiated.lhs - expected) == 0
    assert differentiated.rhs == 0


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
    expected = P * spp.partial(v, T, hold=P) - R

    assert spp.simplify(differentiated.lhs - expected) == 0
    assert differentiated.rhs == 0


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
    assert spp.simplify(rediffed.lhs - (spp.partial(v, T, hold=P) - R / P)) == 0
    assert rediffed.rhs == 0
