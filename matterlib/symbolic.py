import sympy as sp
from dataclasses import dataclass
from typing import Any, Callable, cast
from types import ModuleType
from sympy.core.relational import Equality
from sympy.physics.units import Quantity, convert_to, mol
import sympy.physics.units as units


def _normalize_hold_variables(hold) -> tuple[sp.Expr, ...]:
    if hold is None:
        return tuple()
    if isinstance(hold, (tuple, list, set, sp.Tuple)):
        return tuple(cast(sp.Expr, sp.sympify(x)) for x in hold)
    return (cast(sp.Expr, sp.sympify(hold)),)


def _normalize_lambdify_args(args: Any) -> tuple[sp.Expr, ...]:
    if isinstance(args, (tuple, list, sp.Tuple)):
        normalized = tuple(cast(sp.Expr, sp.sympify(arg)) for arg in args)
    else:
        normalized = (cast(sp.Expr, sp.sympify(args)),)
    if not normalized:
        raise ValueError("No lambdify arguments provided.")
    return normalized


def _replace_quantities_with_scale_factors(
    expr: sp.Expr, max_passes: int = 8
) -> sp.Expr:
    """
    Replace unit quantities with scale factors until no quantities remain.

    SymPy's unit conversion can leave Quantity atoms inside nested expressions
    (for example under sqrt). Those atoms are not NumPy-callable, so we strip
    them by repeatedly substituting each Quantity with its scale_factor.
    """
    reduced = cast(sp.Expr, sp.sympify(expr))
    for _ in range(max_passes):
        quantities = reduced.atoms(Quantity)
        if not quantities:
            break
        substitutions = {
            quantity: cast(sp.Expr, sp.sympify(quantity.scale_factor))
            for quantity in quantities
        }
        next_reduced = cast(sp.Expr, reduced.xreplace(substitutions))
        if next_reduced == reduced:
            break
        reduced = cast(sp.Expr, sp.simplify(next_reduced))
    return reduced


class ConstrainedPartial(sp.Expr):
    """
    Symbolic thermodynamic constrained partial derivative.

    Represents objects like (∂P/∂v)_T.
    """

    is_commutative = True

    def __new__(cls, dependent, wrt, hold=None):
        dependent_sym = cast(sp.Expr, sp.sympify(dependent))
        wrt_sym = cast(sp.Expr, sp.sympify(wrt))
        hold_tuple = sp.Tuple(*_normalize_hold_variables(hold))
        return cast(
            "ConstrainedPartial",
            sp.Expr.__new__(cls, dependent_sym, wrt_sym, hold_tuple),
        )

    @property
    def dependent(self) -> sp.Expr:
        return cast(sp.Expr, self.args[0])

    @property
    def wrt(self) -> sp.Expr:
        return cast(sp.Expr, self.args[1])

    @property
    def hold(self) -> tuple[sp.Expr, ...]:
        hold_tuple = cast(sp.Tuple, self.args[2])
        return tuple(cast(sp.Expr, item) for item in hold_tuple)

    def _latex(self, printer) -> str:
        dependent_latex = printer._print(self.dependent)
        wrt_latex = printer._print(self.wrt)
        base = (
            rf"\left(\frac{{\partial {dependent_latex}}}"
            rf"{{\partial {wrt_latex}}}\right)"
        )
        if not self.hold:
            return base
        hold_latex = ", ".join(printer._print(symbol) for symbol in self.hold)
        return rf"{base}_{{{hold_latex}}}"

    def _sympystr(self, printer) -> str:
        hold_str = ", ".join(printer.doprint(symbol) for symbol in self.hold)
        if hold_str:
            return (
                f"partial({printer.doprint(self.dependent)}, "
                f"{printer.doprint(self.wrt)}, hold=({hold_str}))"
            )
        return (
            f"partial({printer.doprint(self.dependent)}, {printer.doprint(self.wrt)})"
        )


def constrained_partial(dependent, wrt, hold=None) -> ConstrainedPartial:
    return ConstrainedPartial(dependent, wrt, hold=hold)


def _round_sig(x: sp.Expr, sig: int) -> sp.Expr:
    """Round a SymPy number to `sig` significant figures."""
    if x == 0:
        return x
    return sp.Float(x, sig)


def solve_all(expr, var):
    sol = sp.solve(expr, var)
    if not sol:
        raise ValueError(f"No solution found for {var}")
    return sol


def solve_for(expr, var):
    sol = solve_all(expr, var)
    if len(sol) != 1:
        raise ValueError(f"Expected one solution for {var}, got {sol}")
    return sol[0]


def normalize_units(
    eq: Equality, decimal: bool = True, sigfigs: int | None = None
) -> Equality:
    rhs = sp.cancel(eq.rhs)
    coeff = sp.Integer(1)
    unit_factors: list[sp.Expr] = []
    for factor in sp.Mul.make_args(rhs):
        if factor.is_number:
            coeff *= factor
        else:
            unit_factors.append(factor)
    units = sp.Mul(*unit_factors) if unit_factors else sp.Integer(1)

    if decimal and coeff.is_number and not coeff.is_Integer:
        coeff = coeff.evalf()

    if sigfigs is not None:
        coeff = _round_sig(coeff, sigfigs)

    return cast(Equality, sp.Eq(eq.lhs, coeff * units))


def subs_rhs(eq: Equality, replacements: dict[Any, Any]) -> Equality:
    return cast(Equality, sp.Eq(eq.lhs, eq.rhs.subs(replacements)))


def convert_eq(eq: Equality, target_units: sp.Expr) -> Equality:
    """Convert the RHS of an equation to target_units."""
    return cast(Equality, sp.Eq(eq.lhs, convert_to(eq.rhs, target_units)))


def eval_constant(
    constant: sp.Expr,
    target_units: sp.Expr,
    decimal: bool = True,
    sigfigs: int | None = None,
) -> sp.Expr:
    """
    Evaluate a symbolic physical constant in explicit target units.

    Example:
        eval_constant(molar_gas_constant, joule/(kelvin*kilomole))
    """
    value = convert_to(constant, target_units)
    if decimal:
        value = cast(sp.Expr, value).evalf()
    if sigfigs is not None:
        value = _round_sig(cast(sp.Expr, value), sigfigs)
    return cast(sp.Expr, value)


@dataclass(frozen=True)
class Equation:
    eq: Equality

    def _new(self, eq: Equality) -> "Equation":
        return Equation(eq)

    @property
    def lhs(self) -> sp.Expr:
        return cast(sp.Expr, self.eq.lhs)

    @property
    def rhs(self) -> sp.Expr:
        return cast(sp.Expr, self.eq.rhs)

    def unwrap(self) -> Equality:
        return self.eq

    def solve_for(self, var, root=None):
        sol = sp.solve(self.eq, var)
        if len(sol) != 1 and root is None:
            raise ValueError(f"Expected one solution for {var}, got {sol}")
        root = root or 0
        return self._new(cast(Equality, sp.Eq(var, sol[root])))

    def solve_for_all(self, var):
        sol = sp.solve(self.eq, var)
        if not sol:
            raise ValueError(f"No solution found for {var}")
        return [self._new(cast(Equality, sp.Eq(var, s))) for s in sol]

    def subs(self, *args: Any, **kwargs: Any) -> "Equation":
        if len(args) == 1 and not kwargs:
            replacement = args[0]
            if isinstance(replacement, Equation):
                args = (replacement.as_dict(),)
            elif isinstance(replacement, Equality):
                args = ({replacement.lhs: replacement.rhs},)
        return self._new(cast(Equality, self.eq.subs(*args, **kwargs)))

    def replace(self, replacements: dict[Any, Any]) -> "Equation":
        return self._new(
            cast(
                Equality,
                sp.Eq(self.lhs.subs(replacements), self.rhs.subs(replacements)),
            )
        )

    def subs_rhs(self, replacements: dict[Any, Any]) -> "Equation":
        return self._new(subs_rhs(self.eq, replacements))

    def map(self, f: Callable[[sp.Expr], sp.Expr]) -> "Equation":
        return self._new(cast(Equality, sp.Eq(f(self.lhs), f(self.rhs))))

    def simplify(self) -> "Equation":
        return self.map(sp.simplify)

    def evalf(self, *args: Any, **kwargs: Any) -> "Equation":
        return self.map(lambda s: cast(sp.Expr, s.evalf(*args, **kwargs)))

    def reverse_sides(self) -> "Equation":
        return self._new(cast(Equality, sp.Eq(self.rhs, self.lhs)))

    def expand(self) -> "Equation":
        return self.map(sp.expand)

    def factor(self) -> "Equation":
        return self.map(sp.factor)

    def cancel(self) -> "Equation":
        return self.map(sp.cancel)

    def collect(self, *args: Any, **kwargs: Any) -> "Equation":
        return self.map(lambda s: sp.collect(s, *args, **kwargs))

    def convert_to(self, target_units: sp.Expr) -> "Equation":
        return self._new(convert_eq(self.eq, target_units))

    def as_dict(self):
        return {self.lhs: self.rhs}

    def diff_rhs(self, var, simplify_result: bool = True) -> "Equation":
        rhs = sp.diff(self.rhs, var)
        if simplify_result:
            rhs = sp.simplify(rhs)
        return self._new(cast(Equality, sp.Eq(sp.Derivative(self.lhs, var), rhs)))

    def diff_implicit(
        self,
        wrt,
        dependent,
        hold=None,
        simplify_result: bool = True,
    ) -> "Equation":
        wrt_expr = cast(sp.Expr, sp.sympify(wrt))
        dependent_expr = cast(sp.Expr, sp.sympify(dependent))
        hold_variables = _normalize_hold_variables(hold)
        if wrt_expr in hold_variables:
            raise ValueError(
                f"Cannot differentiate with respect to held variable {wrt_expr}"
            )

        constrained = constrained_partial(dependent_expr, wrt_expr, hold=hold_variables)
        lhs_differentiated = (
            sp.diff(self.lhs, wrt_expr)
            + sp.diff(self.lhs, dependent_expr) * constrained
        )
        rhs_differentiated = (
            sp.diff(self.rhs, wrt_expr)
            + sp.diff(self.rhs, dependent_expr) * constrained
        )
        if simplify_result:
            lhs_differentiated = sp.simplify(lhs_differentiated)
            rhs_differentiated = sp.simplify(rhs_differentiated)
        return self._new(cast(Equality, sp.Eq(lhs_differentiated, rhs_differentiated)))

    def partial_for(
        self,
        dependent,
        wrt,
        hold=None,
        simplify_result: bool = True,
        root=None,
    ) -> "Equation":
        dependent_expr = cast(sp.Expr, sp.sympify(dependent))
        wrt_expr = cast(sp.Expr, sp.sympify(wrt))
        hold_variables = _normalize_hold_variables(hold)
        target = constrained_partial(dependent_expr, wrt_expr, hold=hold_variables)
        differentiated = self.diff_implicit(
            wrt=wrt_expr,
            dependent=dependent_expr,
            hold=hold_variables,
            simplify_result=simplify_result,
        )
        sol = sp.solve(differentiated.eq, target)
        if len(sol) != 1 and root is None:
            raise ValueError(f"Expected one solution for {target}, got {sol}")
        root = root or 0
        rhs = sol[root]
        if simplify_result:
            rhs = sp.simplify(rhs)
        return self._new(cast(Equality, sp.Eq(target, rhs)))

    def normalize_units(
        self, decimal: bool = True, sigfigs: int | None = None
    ) -> "Equation":
        return self._new(normalize_units(self.eq, decimal=decimal, sigfigs=sigfigs))

    def apply(
        self, expr: sp.Expr, op: Callable[[sp.Expr, sp.Expr], sp.Expr]
    ) -> "Equation":
        return self.map(lambda s: op(s, expr))

    def __add__(self, other) -> "Equation":
        if isinstance(other, Equation):
            return self._new(
                cast(Equality, sp.Eq(self.lhs + other.lhs, self.rhs + other.rhs))
            )
        return self.map(lambda s: s + other)

    def __sub__(self, other) -> "Equation":
        if isinstance(other, Equation):
            return self._new(
                cast(Equality, sp.Eq(self.lhs - other.lhs, self.rhs - other.rhs))
            )
        return self.map(lambda s: s - other)

    def __mul__(self, other) -> "Equation":
        if isinstance(other, Equation):
            return self._new(
                cast(Equality, sp.Eq(self.lhs * other.lhs, self.rhs * other.rhs))
            )
        return self.map(lambda s: s * other)

    def __truediv__(self, other) -> "Equation":
        if isinstance(other, Equation):
            return self._new(
                cast(Equality, sp.Eq(self.lhs / other.lhs, self.rhs / other.rhs))
            )
        return self.map(lambda s: s / other)

    def __rtruediv__(self, other) -> "Equation":
        return self.map(lambda s: other / s)

    def __pow__(self, other: sp.Expr) -> "Equation":
        return self.map(lambda s: s**other)

    def __neg__(self) -> "Equation":
        return self.map(lambda s: -s)

    def __repr__(self) -> str:
        return repr(self.eq)

    def _repr_latex_(self) -> str:
        return self.eq._repr_latex_()

    def with_state(self, *state_variables) -> "StateEquation":
        return StateEquation(self.eq, state_variables)


@dataclass(frozen=True)
class StateEquation(Equation):
    state_variables: tuple[Any, ...]

    def _new(self, eq: Equality) -> "StateEquation":
        return StateEquation(eq, self.state_variables)

    def _normalized_state_variables(self) -> tuple[sp.Expr, ...]:
        if not self.state_variables:
            raise ValueError("with_state requires at least one state variable")
        return tuple(cast(sp.Expr, sp.sympify(var)) for var in self.state_variables)

    def _infer_dependent(self, wrt, hold) -> sp.Expr:
        wrt_expr = cast(sp.Expr, sp.sympify(wrt))
        hold_variables = _normalize_hold_variables(hold)
        state_variables = self._normalized_state_variables()

        if wrt_expr not in state_variables:
            raise ValueError(
                f"differentiation variable {wrt_expr} is not in declared state variables "
                f"{state_variables}"
            )
        for held in hold_variables:
            if held not in state_variables:
                raise ValueError(
                    f"held variable {held} is not in declared state variables {state_variables}"
                )

        remaining = [
            var
            for var in state_variables
            if var != wrt_expr and var not in hold_variables
        ]
        if len(remaining) != 1:
            raise ValueError(
                "Could not infer dependent variable uniquely; "
                "provide dependent explicitly."
            )
        return remaining[0]

    def diff_implicit(
        self,
        wrt,
        hold=None,
        dependent=None,
        simplify_result: bool = True,
    ) -> "StateEquation":
        dependent_expr = (
            cast(sp.Expr, sp.sympify(dependent))
            if dependent is not None
            else self._infer_dependent(wrt=wrt, hold=hold)
        )
        return cast(
            "StateEquation",
            super().diff_implicit(
                wrt=wrt,
                dependent=dependent_expr,
                hold=hold,
                simplify_result=simplify_result,
            ),
        )

    def partial_for(
        self,
        wrt,
        hold=None,
        dependent=None,
        simplify_result: bool = True,
        root=None,
    ) -> "StateEquation":
        dependent_expr = (
            cast(sp.Expr, sp.sympify(dependent))
            if dependent is not None
            else self._infer_dependent(wrt=wrt, hold=hold)
        )
        return cast(
            "StateEquation",
            super().partial_for(
                dependent=dependent_expr,
                wrt=wrt,
                hold=hold,
                simplify_result=simplify_result,
                root=root,
            ),
        )

    def __repr__(self) -> str:
        return super().__repr__()

    def _repr_latex_(self) -> str:
        return super()._repr_latex_()


class _UnitsNamespace:
    """
    Hybrid units accessor:
    - callable like symbols: spp.units("torr cm kelvin")
    - attribute passthrough: spp.units.torr
    """

    def __init__(self, module: ModuleType):
        self._module = module
        kmol = Quantity("kilomole", abbrev="kmol")
        kmol.set_global_relative_scale_factor(1000, mol)
        self._aliases: dict[str, Any] = {
            "kmol": kmol,
            "kilomole": kmol,
        }

    def __call__(self, names: str):
        tokens = [token for token in names.split() if token]
        if not tokens:
            raise ValueError("No unit names provided.")
        resolved = [self._resolve_unit(token) for token in tokens]
        if len(resolved) == 1:
            return resolved[0]
        return tuple(resolved)

    def _resolve_unit(self, name: str):
        if name in self._aliases:
            return self._aliases[name]
        try:
            return getattr(self._module, name)
        except AttributeError as err:
            raise AttributeError(
                f"Unknown unit '{name}'. "
                f"Check sympy.physics.units for available names."
            ) from err

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


class _SympyPhys:
    """
    Thin namespace wrapper:
    - spp.Eq(...) returns Equation
    - all other names fall through to sympy
    """

    def __init__(self):
        self.units = _UnitsNamespace(units)
        self.Derivative = sp.Derivative
        self.lambdify = sp.lambdify
        self.Function = sp.Function
        self.diff = sp.diff
        self.ConstrainedPartial = ConstrainedPartial
        self.exp = sp.exp
        self.log = sp.log
        self.sin = sp.sin
        self.cos = sp.cos
        self.tan = sp.tan
        self.asin = sp.asin
        self.acos = sp.acos
        self.atan = sp.atan
        self.Integral = sp.Integral

    def Eq(self, lhs: sp.Expr, rhs: sp.Expr, **kwargs: Any) -> Equation:
        return Equation(cast(Equality, sp.Eq(lhs, rhs, **kwargs)))

    def solve_for(self, expr: Equality | sp.Expr, var: sp.Symbol) -> sp.Expr:
        return solve_for(expr, var)

    def partial(self, dependent, wrt, hold=None) -> ConstrainedPartial:
        return constrained_partial(dependent, wrt, hold=hold)

    def lambdify_units(
        self,
        args: Any,
        expr: sp.Expr,
        *,
        inputs: dict[Any, sp.Expr],
        output: sp.Expr,
        modules: Any = "numpy",
    ):
        arg_tuple = _normalize_lambdify_args(args)
        missing = [arg for arg in arg_tuple if arg not in inputs]
        if missing:
            missing_names = ", ".join(str(arg) for arg in missing)
            raise ValueError(f"Missing unit declarations for arguments: {missing_names}")

        substitutions = {arg: arg * inputs[arg] for arg in arg_tuple}
        replacements = cast(dict[Any, Any], substitutions)
        expr_with_units = cast(sp.Expr, sp.sympify(expr).subs(replacements))
        converted = cast(sp.Expr, convert_to(expr_with_units, output))
        unitless_expr = cast(sp.Expr, sp.simplify(converted / output))
        unitless_expr = _replace_quantities_with_scale_factors(unitless_expr)
        return sp.lambdify(args, unitless_expr, modules=modules)

    def normalize_units(
        self, eq: Equality | Equation, decimal: bool = True, sigfigs: int | None = None
    ) -> Equation:
        raw = eq.eq if isinstance(eq, Equation) else eq
        return Equation(normalize_units(raw, decimal=decimal, sigfigs=sigfigs))

    def convert_to(self, eq: Equality | Equation, target_units: sp.Expr) -> Equation:
        raw = eq.eq if isinstance(eq, Equation) else eq
        return Equation(convert_eq(raw, target_units))

    def subs_rhs(
        self, eq: Equality | Equation, replacements: dict[Any, Any]
    ) -> Equation:
        raw = eq.eq if isinstance(eq, Equation) else eq
        return Equation(subs_rhs(raw, replacements))

    # This is a hack to get things like the molar gas constant to stop being symbolic.
    def eval_constant(
        self,
        constant: sp.Expr,
        target_units: sp.Expr,
        decimal: bool = True,
        sigfigs: int | None = None,
    ) -> sp.Expr:
        return eval_constant(constant, target_units, decimal=decimal, sigfigs=sigfigs)

    def R_kmol(self, decimal: bool = True, sigfigs: int | None = None) -> sp.Expr:
        """
        Gas constant evaluated in J/(K*kmol).

        This is a convenience wrapper around eval_constant for notebook ergonomics.
        """
        joule, kelvin, kilomole, molar_gas_constant = self.units(
            "joule kelvin kilomole molar_gas_constant"
        )
        target_units = joule / (kelvin * kilomole)
        return self.eval_constant(
            molar_gas_constant,
            target_units,
            decimal=decimal,
            sigfigs=sigfigs,
        )

    def solve_system(self, eqs, vars):
        raw = []
        for e in eqs:
            if isinstance(e, StateEquation):
                raw.append(e.eq)
            elif isinstance(e, Equation):
                raw.append(e.eq)
            else:
                raw.append(e)
        return sp.solve(raw, vars)

    def __getattr__(self, name: str) -> Any:
        return getattr(sp, name)


sympy_phys = _SympyPhys()
