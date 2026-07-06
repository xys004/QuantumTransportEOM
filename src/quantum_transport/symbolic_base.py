"""Shared base classes for symbolic operator wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import sympy as sp


def eom_system_latex(operators: Sequence[sp.Expr], eom_matrix: sp.Matrix, residuals: Sequence[sp.Expr]) -> str:
    """LaTeX for a projected EOM system [O_i, H] = sum_j M_ij O_j + R_i."""
    ops = sp.Matrix([[op] for op in operators])
    body = rf"\left[\hat{{O}}, H\right] = {sp.latex(eom_matrix)} {sp.latex(ops)}"
    if any(sp.simplify(residual) != 0 for residual in residuals):
        res = sp.Matrix([[residual] for residual in residuals])
        body += rf" + {sp.latex(res)}"
    return f"${body}$"


@dataclass(frozen=True)
class SymbolicOperator:
    """Common ergonomic base for symbolic operator wrappers."""

    expr: sp.Expr
    statistics: str = field(init=False, default="generic")

    def latex(self) -> str:
        return sp.latex(self.expr)

    def _repr_latex_(self) -> str:
        """Render nicely in Jupyter notebooks."""
        return f"${sp.latex(self.expr)}$"

    def doit(self) -> sp.Expr:
        return self.expr

    def _unwrap(self, value):
        return value.expr if isinstance(value, SymbolicOperator) else value

    def __add__(self, other):
        return type(self)(self.expr + self._unwrap(other))

    def __radd__(self, other):
        return type(self)(self._unwrap(other) + self.expr)

    def __sub__(self, other):
        return type(self)(self.expr - self._unwrap(other))

    def __rsub__(self, other):
        return type(self)(self._unwrap(other) - self.expr)

    def __mul__(self, other):
        return type(self)(self.expr * self._unwrap(other))

    def __rmul__(self, other):
        return type(self)(self._unwrap(other) * self.expr)

    def __truediv__(self, other):
        return type(self)(self.expr / self._unwrap(other))

    def __neg__(self):
        return type(self)(-self.expr)
