import sympy as sp
from dataclasses import dataclass
from typing import Any, Callable, cast
from types import ModuleType
from sympy.core.relational import Equality
from sympy.physics.units import Quantity, convert_to, mol
import sympy.physics.units as units


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
        value = value.evalf()
    if sigfigs is not None:
        value = _round_sig(cast(sp.Expr, value), sigfigs)
    return cast(sp.Expr, value)


@dataclass(frozen=True)
class Equation:
    eq: Equality

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
        return Equation(sp.Eq(var, sol[root]))

    def solve_for_all(self, var):
        sol = sp.solve(self.eq, var)
        if not sol:
            raise ValueError(f"No solution found for {var}")
        return [Equation(sp.Eq(var, s)) for s in sol]

    def subs(self, *args: Any, **kwargs: Any) -> "Equation":
        return Equation(cast(Equality, self.eq.subs(*args, **kwargs)))

    def replace(self, replacements: dict[Any, Any]) -> "Equation":
        return Equation(
            cast(
                Equality,
                sp.Eq(self.lhs.subs(replacements), self.rhs.subs(replacements)),
            )
        )

    def subs_rhs(self, replacements: dict[Any, Any]) -> "Equation":
        return Equation(subs_rhs(self.eq, replacements))

    def map(self, f: Callable[[sp.Expr], sp.Expr]) -> "Equation":
        return Equation(cast(Equality, sp.Eq(f(self.lhs), f(self.rhs))))

    def simplify(self) -> "Equation":
        return self.map(sp.simplify)

    def expand(self) -> "Equation":
        return self.map(sp.expand)

    def factor(self) -> "Equation":
        return self.map(sp.factor)

    def cancel(self) -> "Equation":
        return self.map(sp.cancel)

    def collect(self, *args: Any, **kwargs: Any) -> "Equation":
        return self.map(lambda s: sp.collect(s, *args, **kwargs))

    def convert_to(self, target_units: sp.Expr) -> "Equation":
        return Equation(convert_eq(self.eq, target_units))

    def as_dict(self):
        return {self.lhs: self.rhs}

    def diff_rhs(self, var, simplify_result: bool = True) -> "Equation":
        rhs = sp.diff(self.rhs, var)
        if simplify_result:
            rhs = sp.simplify(rhs)
        return Equation(cast(Equality, sp.Eq(sp.Derivative(self.lhs, var), rhs)))

    def normalize_units(
        self, decimal: bool = True, sigfigs: int | None = None
    ) -> "Equation":
        return Equation(normalize_units(self.eq, decimal=decimal, sigfigs=sigfigs))

    def apply(
        self, expr: sp.Expr, op: Callable[[sp.Expr, sp.Expr], sp.Expr]
    ) -> "Equation":
        return self.map(lambda s: op(s, expr))

    def __add__(self, other) -> "Equation":
        if isinstance(other, Equation):
            return Equation(
                cast(Equality, sp.Eq(self.lhs + other.lhs, self.rhs + other.rhs))
            )
        return self.map(lambda s: s + other)

    def __sub__(self, other) -> "Equation":
        if isinstance(other, Equation):
            return Equation(
                cast(Equality, sp.Eq(self.lhs - other.lhs, self.rhs - other.rhs))
            )
        return self.map(lambda s: s - other)

    def __mul__(self, other) -> "Equation":
        if isinstance(other, Equation):
            return Equation(
                cast(Equality, sp.Eq(self.lhs * other.lhs, self.rhs * other.rhs))
            )
        return self.map(lambda s: s * other)

    def __truediv__(self, other) -> "Equation":
        if isinstance(other, Equation):
            return Equation(
                cast(Equality, sp.Eq(self.lhs / other.lhs, self.rhs / other.rhs))
            )
        return self.map(lambda s: s / other)

    def __pow__(self, other: sp.Expr) -> "Equation":
        return self.map(lambda s: s**other)

    def __neg__(self) -> "Equation":
        return self.map(lambda s: -s)

    def __repr__(self) -> str:
        return repr(self.eq)

    def _repr_latex_(self) -> str:
        return self.eq._repr_latex_()


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

    def Eq(self, lhs: sp.Expr, rhs: sp.Expr, **kwargs: Any) -> Equation:
        return Equation(cast(Equality, sp.Eq(lhs, rhs, **kwargs)))

    def solve_for(self, expr: Equality | sp.Expr, var: sp.Symbol) -> sp.Expr:
        return solve_for(expr, var)

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

    def solve_system(self, eqs, vars):
        raw = [e.eq if isinstance(e, Equation) else e for e in eqs]
        return sp.solve(raw, vars)

    def __getattr__(self, name: str) -> Any:
        return getattr(sp, name)


sympy_phys = _SympyPhys()
