"""Stationary Keldysh helpers for non-equilibrium Green functions and currents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import sympy as sp

from .devices import LeadSelfEnergy, MatrixTransportView, spin_axis_projector_numeric
from .greens import fermi_dirac


ArrayFn = Callable[[float], np.ndarray]


def _as_matrix(value: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.shape != (dim, dim):
        raise ValueError(f"Expected matrix shape {(dim, dim)}, got {arr.shape}.")
    return arr


def _interpolate_complex_matrix(omega: float, omega_grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    omega_grid = np.asarray(omega_grid, dtype=float)
    values = np.asarray(values, dtype=np.complex128)
    dim = values.shape[1]
    out = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            re = np.interp(omega, omega_grid, np.real(values[:, i, j]))
            im = np.interp(omega, omega_grid, np.imag(values[:, i, j]))
            out[i, j] = re + 1j * im
    return out


@dataclass(frozen=True)
class KeldyshSelfEnergy:
    dim: int
    sigma_retarded_fn: ArrayFn
    sigma_lesser_fn: ArrayFn
    sigma_greater_fn: ArrayFn
    name: str = "self_energy"

    @classmethod
    def from_lead(cls, lead: LeadSelfEnergy) -> "KeldyshSelfEnergy":
        return cls(
            dim=lead.dim,
            sigma_retarded_fn=lead.sigma_retarded,
            sigma_lesser_fn=lead.sigma_lesser,
            sigma_greater_fn=lead.sigma_greater,
            name=lead.name,
        )

    @classmethod
    def from_functions(
        cls,
        *,
        dim: int,
        sigma_retarded_fn: ArrayFn,
        sigma_lesser_fn: ArrayFn,
        sigma_greater_fn: ArrayFn,
        name: str = "self_energy",
    ) -> "KeldyshSelfEnergy":
        return cls(
            dim=dim,
            sigma_retarded_fn=sigma_retarded_fn,
            sigma_lesser_fn=sigma_lesser_fn,
            sigma_greater_fn=sigma_greater_fn,
            name=name,
        )

    @classmethod
    def equilibrium_from_retarded(
        cls,
        *,
        dim: int,
        sigma_retarded_fn: ArrayFn,
        mu: float = 0.0,
        temperature: float = 0.0,
        name: str = "self_energy",
    ) -> "KeldyshSelfEnergy":
        def sigma_lesser_fn(omega: float) -> np.ndarray:
            sigma_r = _as_matrix(sigma_retarded_fn(omega), dim)
            gamma = 1j * (sigma_r - sigma_r.conj().T)
            f = float(fermi_dirac(np.array([omega]), mu=mu, temperature=temperature)[0])
            return 1j * f * gamma

        def sigma_greater_fn(omega: float) -> np.ndarray:
            sigma_r = _as_matrix(sigma_retarded_fn(omega), dim)
            gamma = 1j * (sigma_r - sigma_r.conj().T)
            f = float(fermi_dirac(np.array([omega]), mu=mu, temperature=temperature)[0])
            return 1j * (f - 1.0) * gamma

        return cls(
            dim=dim,
            sigma_retarded_fn=lambda omega: _as_matrix(sigma_retarded_fn(omega), dim),
            sigma_lesser_fn=sigma_lesser_fn,
            sigma_greater_fn=sigma_greater_fn,
            name=name,
        )

    @classmethod
    def sampled(
        cls,
        omega_grid: np.ndarray,
        sigma_retarded_values: np.ndarray,
        *,
        sigma_lesser_values: np.ndarray | None = None,
        sigma_greater_values: np.ndarray | None = None,
        mu: float | None = None,
        temperature: float = 0.0,
        name: str = "self_energy",
    ) -> "KeldyshSelfEnergy":
        omega_grid = np.asarray(omega_grid, dtype=float)
        sigma_retarded_values = np.asarray(sigma_retarded_values, dtype=np.complex128)
        if sigma_retarded_values.ndim != 3 or sigma_retarded_values.shape[0] != omega_grid.size:
            raise ValueError("sigma_retarded_values must have shape (n_omega, dim, dim).")
        dim = sigma_retarded_values.shape[1]
        if sigma_retarded_values.shape[2] != dim:
            raise ValueError("sigma_retarded_values must contain square matrices.")

        def sigma_retarded_fn(omega: float) -> np.ndarray:
            return _interpolate_complex_matrix(omega, omega_grid, sigma_retarded_values)

        lesser_fn = None
        if sigma_lesser_values is not None:
            sigma_lesser_values = np.asarray(sigma_lesser_values, dtype=np.complex128)
            lesser_fn = lambda omega: _interpolate_complex_matrix(omega, omega_grid, sigma_lesser_values)

        greater_fn = None
        if sigma_greater_values is not None:
            sigma_greater_values = np.asarray(sigma_greater_values, dtype=np.complex128)
            greater_fn = lambda omega: _interpolate_complex_matrix(omega, omega_grid, sigma_greater_values)

        if lesser_fn is None or greater_fn is None:
            if mu is None:
                raise ValueError("Provide sigma_lesser_values and sigma_greater_values, or provide mu for equilibrium reconstruction.")
            return cls.equilibrium_from_retarded(
                dim=dim,
                sigma_retarded_fn=sigma_retarded_fn,
                mu=mu,
                temperature=temperature,
                name=name,
            )

        return cls(
            dim=dim,
            sigma_retarded_fn=sigma_retarded_fn,
            sigma_lesser_fn=lesser_fn,
            sigma_greater_fn=greater_fn,
            name=name,
        )

    @classmethod
    def combine(cls, *self_energies: "KeldyshSelfEnergy", name: str = "combined") -> "KeldyshSelfEnergy":
        if not self_energies:
            raise ValueError("At least one self-energy is required.")
        dim = self_energies[0].dim
        if any(sigma.dim != dim for sigma in self_energies):
            raise ValueError("All self-energies must have the same dimension.")

        def sigma_retarded_fn(omega: float) -> np.ndarray:
            return sum((sigma.sigma_retarded(omega) for sigma in self_energies), np.zeros((dim, dim), dtype=np.complex128))

        def sigma_lesser_fn(omega: float) -> np.ndarray:
            return sum((sigma.sigma_lesser(omega) for sigma in self_energies), np.zeros((dim, dim), dtype=np.complex128))

        def sigma_greater_fn(omega: float) -> np.ndarray:
            return sum((sigma.sigma_greater(omega) for sigma in self_energies), np.zeros((dim, dim), dtype=np.complex128))

        return cls(dim=dim, sigma_retarded_fn=sigma_retarded_fn, sigma_lesser_fn=sigma_lesser_fn, sigma_greater_fn=sigma_greater_fn, name=name)

    def sigma_retarded(self, omega: float) -> np.ndarray:
        return _as_matrix(self.sigma_retarded_fn(float(omega)), self.dim)

    def sigma_advanced(self, omega: float) -> np.ndarray:
        return self.sigma_retarded(omega).conj().T

    def sigma_lesser(self, omega: float) -> np.ndarray:
        return _as_matrix(self.sigma_lesser_fn(float(omega)), self.dim)

    def sigma_greater(self, omega: float) -> np.ndarray:
        return _as_matrix(self.sigma_greater_fn(float(omega)), self.dim)

    def sigma_keldysh(self, omega: float) -> np.ndarray:
        return sigma_keldysh(self.sigma_lesser(omega), self.sigma_greater(omega))

    def gamma(self, omega: float) -> np.ndarray:
        sigma_r = self.sigma_retarded(omega)
        sigma_a = sigma_r.conj().T
        return 1j * (sigma_r - sigma_a)


@dataclass(frozen=True)
class KeldyshTransportView:
    transport: MatrixTransportView
    extra_self_energies: tuple[KeldyshSelfEnergy, ...] = ()

    @property
    def dim(self) -> int:
        return self.transport.dim

    def with_self_energy(self, *self_energies: KeldyshSelfEnergy) -> "KeldyshTransportView":
        for sigma in self_energies:
            if sigma.dim != self.dim:
                raise ValueError(f"Expected self-energy dimension {self.dim}, got {sigma.dim}.")
        return KeldyshTransportView(self.transport, self.extra_self_energies + tuple(self_energies))

    def sigma_left(self) -> KeldyshSelfEnergy:
        return KeldyshSelfEnergy.from_lead(self.transport.left_lead)

    def sigma_right(self) -> KeldyshSelfEnergy:
        return KeldyshSelfEnergy.from_lead(self.transport.right_lead)

    def sigma_interactions(self) -> KeldyshSelfEnergy | None:
        if not self.extra_self_energies:
            return None
        return KeldyshSelfEnergy.combine(*self.extra_self_energies, name="interactions")

    def _sigma_from_extras(self, omega: float, component: str) -> np.ndarray:
        if not self.extra_self_energies:
            return np.zeros((self.dim, self.dim), dtype=np.complex128)
        if component == "retarded":
            return sum((sigma.sigma_retarded(omega) for sigma in self.extra_self_energies), np.zeros((self.dim, self.dim), dtype=np.complex128))
        if component == "lesser":
            return sum((sigma.sigma_lesser(omega) for sigma in self.extra_self_energies), np.zeros((self.dim, self.dim), dtype=np.complex128))
        if component == "greater":
            return sum((sigma.sigma_greater(omega) for sigma in self.extra_self_energies), np.zeros((self.dim, self.dim), dtype=np.complex128))
        raise ValueError("component must be 'retarded', 'lesser', or 'greater'.")

    def sigma_retarded(self, omega: float, lead: str | None = None) -> np.ndarray:
        if lead == "left":
            return self.transport.left_lead.sigma_retarded(omega)
        if lead == "right":
            return self.transport.right_lead.sigma_retarded(omega)
        if lead == "interaction":
            return self._sigma_from_extras(omega, "retarded")
        if lead is None or lead == "total":
            return self.transport.sigma_retarded_total(omega) + self._sigma_from_extras(omega, "retarded")
        raise ValueError("lead must be 'left', 'right', 'interaction', or None/'total'.")

    def sigma_advanced(self, omega: float, lead: str | None = None) -> np.ndarray:
        return self.sigma_retarded(omega, lead=lead).conj().T

    def sigma_lesser(self, omega: float, lead: str | None = None) -> np.ndarray:
        if lead == "left":
            return self.transport.left_lead.sigma_lesser(omega)
        if lead == "right":
            return self.transport.right_lead.sigma_lesser(omega)
        if lead == "interaction":
            return self._sigma_from_extras(omega, "lesser")
        if lead is None or lead == "total":
            return self.transport.sigma_lesser_total(omega) + self._sigma_from_extras(omega, "lesser")
        raise ValueError("lead must be 'left', 'right', 'interaction', or None/'total'.")

    def sigma_greater(self, omega: float, lead: str | None = None) -> np.ndarray:
        if lead == "left":
            return self.transport.left_lead.sigma_greater(omega)
        if lead == "right":
            return self.transport.right_lead.sigma_greater(omega)
        if lead == "interaction":
            return self._sigma_from_extras(omega, "greater")
        if lead is None or lead == "total":
            return self.transport.sigma_greater_total(omega) + self._sigma_from_extras(omega, "greater")
        raise ValueError("lead must be 'left', 'right', 'interaction', or None/'total'.")

    def sigma_keldysh(self, omega: float, lead: str | None = None) -> np.ndarray:
        return sigma_keldysh(self.sigma_lesser(omega, lead=lead), self.sigma_greater(omega, lead=lead))

    def retarded(self, omega: float, eta: float = 0.0) -> np.ndarray:
        identity = np.eye(self.dim, dtype=np.complex128)
        h = np.asarray(self.transport.hamiltonian, dtype=np.complex128)
        sigma_r = self.sigma_retarded(omega, lead="total")
        return np.linalg.inv((omega + 1j * eta) * identity - h - sigma_r)

    def advanced(self, omega: float, eta: float = 0.0) -> np.ndarray:
        return self.retarded(omega, eta=eta).conj().T

    def lesser(self, omega: float, eta: float = 0.0) -> np.ndarray:
        return lesser_from_retarded_sigma(
            self.retarded(omega, eta=eta),
            self.sigma_lesser(omega, lead="total"),
            self.advanced(omega, eta=eta),
        )

    def greater(self, omega: float, eta: float = 0.0) -> np.ndarray:
        return greater_from_retarded_sigma(
            self.retarded(omega, eta=eta),
            self.sigma_greater(omega, lead="total"),
            self.advanced(omega, eta=eta),
        )

    def keldysh(self, omega: float, eta: float = 0.0) -> np.ndarray:
        return keldysh_from_retarded_sigma(
            self.retarded(omega, eta=eta),
            self.sigma_keldysh(omega, lead="total"),
            self.advanced(omega, eta=eta),
        )

    def meir_wingreen_current_density(self, omega: float, lead: str = "left", charge: float = 1.0, eta: float = 0.0) -> float:
        return meir_wingreen_current_density(
            self.sigma_lesser(omega, lead=lead),
            self.sigma_greater(omega, lead=lead),
            self.lesser(omega, eta=eta),
            self.greater(omega, eta=eta),
            charge=charge,
        )

    def meir_wingreen_current(self, omega_grid: np.ndarray, lead: str = "left", charge: float = 1.0, eta: float = 0.0) -> float:
        omega_grid = np.asarray(omega_grid, dtype=float)
        values = np.array(
            [self.meir_wingreen_current_density(float(omega), lead=lead, charge=charge, eta=eta) for omega in omega_grid],
            dtype=float,
        )
        return float(np.trapezoid(values, omega_grid))

    def meir_wingreen_spin_current_density(
        self,
        omega: float,
        lead: str = "left",
        *,
        axis: str = "z",
        component: str | int | float | None = None,
        charge: float = 1.0,
        eta: float = 0.0,
    ) -> float:
        sigma_l = self.sigma_lesser(omega, lead=lead)
        sigma_g = self.sigma_greater(omega, lead=lead)
        if component is not None:
            projector = spin_axis_projector_numeric(self.transport.basis_labels, axis=axis, component=component)
            sigma_l = projector @ sigma_l @ projector
            sigma_g = projector @ sigma_g @ projector
            return meir_wingreen_current_density(sigma_l, sigma_g, self.lesser(omega, eta=eta), self.greater(omega, eta=eta), charge=charge)
        plus = self.meir_wingreen_spin_current_density(omega, lead=lead, axis=axis, component="+", charge=charge, eta=eta)
        minus = self.meir_wingreen_spin_current_density(omega, lead=lead, axis=axis, component="-", charge=charge, eta=eta)
        return float(plus - minus)

    def meir_wingreen_spin_current(self, omega_grid: np.ndarray, lead: str = "left", *, axis: str = "z", component: str | int | float | None = None, charge: float = 1.0, eta: float = 0.0) -> float:
        omega_grid = np.asarray(omega_grid, dtype=float)
        values = np.array(
            [
                self.meir_wingreen_spin_current_density(float(omega), lead=lead, axis=axis, component=component, charge=charge, eta=eta)
                for omega in omega_grid
            ],
            dtype=float,
        )
        return float(np.trapezoid(values, omega_grid))

    def meir_wingreen_spin_current_vector(self, omega_grid: np.ndarray, lead: str = "left", charge: float = 1.0, eta: float = 0.0) -> dict[str, float]:
        return {
            axis: self.meir_wingreen_spin_current(omega_grid, lead=lead, axis=axis, charge=charge, eta=eta)
            for axis in ("x", "y", "z")
        }


# Numeric component relations

def sigma_keldysh(sigma_lesser_value: np.ndarray, sigma_greater_value: np.ndarray) -> np.ndarray:
    return np.asarray(sigma_lesser_value, dtype=np.complex128) + np.asarray(sigma_greater_value, dtype=np.complex128)



def green_keldysh_from_lesser_greater(g_lesser_value: np.ndarray, g_greater_value: np.ndarray) -> np.ndarray:
    return np.asarray(g_lesser_value, dtype=np.complex128) + np.asarray(g_greater_value, dtype=np.complex128)



def lesser_from_retarded_sigma(g_ret: np.ndarray, sigma_lesser_value: np.ndarray, g_adv: np.ndarray) -> np.ndarray:
    return np.asarray(g_ret, dtype=np.complex128) @ np.asarray(sigma_lesser_value, dtype=np.complex128) @ np.asarray(g_adv, dtype=np.complex128)



def greater_from_retarded_sigma(g_ret: np.ndarray, sigma_greater_value: np.ndarray, g_adv: np.ndarray) -> np.ndarray:
    return np.asarray(g_ret, dtype=np.complex128) @ np.asarray(sigma_greater_value, dtype=np.complex128) @ np.asarray(g_adv, dtype=np.complex128)



def keldysh_from_retarded_sigma(g_ret: np.ndarray, sigma_keldysh_value: np.ndarray, g_adv: np.ndarray) -> np.ndarray:
    return np.asarray(g_ret, dtype=np.complex128) @ np.asarray(sigma_keldysh_value, dtype=np.complex128) @ np.asarray(g_adv, dtype=np.complex128)



def meir_wingreen_current_density(
    sigma_lesser_value: np.ndarray,
    sigma_greater_value: np.ndarray,
    g_lesser_value: np.ndarray,
    g_greater_value: np.ndarray,
    *,
    charge: float = 1.0,
) -> float:
    integrand = (charge / (2.0 * np.pi)) * np.trace(
        np.asarray(sigma_lesser_value, dtype=np.complex128) @ np.asarray(g_greater_value, dtype=np.complex128)
        - np.asarray(sigma_greater_value, dtype=np.complex128) @ np.asarray(g_lesser_value, dtype=np.complex128)
    )
    return float(np.real(integrand))



def meir_wingreen_current(
    omega_grid: np.ndarray,
    sigma_lesser_values: np.ndarray,
    sigma_greater_values: np.ndarray,
    g_lesser_values: np.ndarray,
    g_greater_values: np.ndarray,
    *,
    charge: float = 1.0,
) -> float:
    omega_grid = np.asarray(omega_grid, dtype=float)
    values = np.array(
        [
            meir_wingreen_current_density(
                sigma_lesser_values[i],
                sigma_greater_values[i],
                g_lesser_values[i],
                g_greater_values[i],
                charge=charge,
            )
            for i in range(omega_grid.size)
        ],
        dtype=float,
    )
    return float(np.trapezoid(values, omega_grid))


# Symbolic helpers

def sigma_keldysh_symbolic(sigma_lesser_expr: Any, sigma_greater_expr: Any) -> Any:
    return sp.simplify(sigma_lesser_expr + sigma_greater_expr)



def green_keldysh_symbolic(g_lesser_expr: Any, g_greater_expr: Any) -> Any:
    return sp.simplify(g_lesser_expr + g_greater_expr)



def lesser_from_retarded_sigma_symbolic(g_ret_expr: Any, sigma_lesser_expr: Any, g_adv_expr: Any) -> Any:
    return sp.simplify(g_ret_expr * sigma_lesser_expr * g_adv_expr)



def greater_from_retarded_sigma_symbolic(g_ret_expr: Any, sigma_greater_expr: Any, g_adv_expr: Any) -> Any:
    return sp.simplify(g_ret_expr * sigma_greater_expr * g_adv_expr)



def keldysh_from_retarded_sigma_symbolic(g_ret_expr: Any, sigma_keldysh_expr: Any, g_adv_expr: Any) -> Any:
    return sp.simplify(g_ret_expr * sigma_keldysh_expr * g_adv_expr)
