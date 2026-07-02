"""Shared base classes for symbolic operator wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp


@dataclass(frozen=True)
class SymbolicOperator:
    """Common ergonomic base for symbolic operator wrappers."""

    expr: sp.Expr
    statistics: str = field(init=False, default="generic")

    def latex(self) -> str:
        return sp.latex(self.expr)

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
