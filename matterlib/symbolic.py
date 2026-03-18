import sympy as sp
from dataclasses import dataclass
from typing import Any, Callable, cast
from types import ModuleType
from sympy.core.relational import Equality
from sympy.physics.units import convert_to
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
    coeff, units = rhs.as_coeff_Mul()

    if decimal and coeff.is_Rational and coeff.q != 1:
        coeff = coeff.evalf()

    if sigfigs is not None:
        coeff = _round_sig(coeff, sigfigs)

    return cast(Equality, sp.Eq(eq.lhs, coeff * units))


def substitute_rhs(eq: Equality, symbol: sp.Symbol, replacement: sp.Expr) -> Equality:
    return cast(Equality, sp.Eq(eq.lhs, eq.rhs.subs(symbol, replacement)))


def convert_eq(eq: Equality, target_units: sp.Expr) -> Equality:
    """Convert the RHS of an equation to target_units."""
    return cast(Equality, sp.Eq(eq.lhs, convert_to(eq.rhs, target_units)))


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
        return Equation(sp.Eq(sp.Derivative(self.lhs, var), rhs))

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

    def __call__(self, names: str):
        tokens = [token for token in names.split() if token]
        if not tokens:
            raise ValueError("No unit names provided.")
        resolved = [self._resolve_unit(token) for token in tokens]
        if len(resolved) == 1:
            return resolved[0]
        return tuple(resolved)

    def _resolve_unit(self, name: str):
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

    def solve_system(self, eqs, vars):
        raw = [e.eq if isinstance(e, Equation) else e for e in eqs]
        return sp.solve(raw, vars)

    def __getattr__(self, name: str) -> Any:
        return getattr(sp, name)


sympy_phys = _SympyPhys()
