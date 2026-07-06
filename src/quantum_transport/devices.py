"""Matrix-device models and explicit leads/self-energies for transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .greens import fermi_dirac
from .hamiltonians import build_rashba_hubbard_ring_real_space
from .numerics import (
    batched_current_spectral_density,
    batched_keldysh_component,
    batched_retarded_green,
    batched_transmission,
    blocked_over_grid,
    default_block_target,
    gamma_from_sigma_stack,
    get_backend,
    precision_dtype,
    sigma_stack,
    to_numpy,
)


ArrayLike = np.ndarray

SIGMA_X_NUMERIC = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y_NUMERIC = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z_NUMERIC = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
IDENTITY_2_NUMERIC = np.eye(2, dtype=np.complex128)



def _as_matrix(value: Any, dim: int | None = None) -> np.ndarray:
    if np.isscalar(value):
        if dim is None:
            raise ValueError("dim is required when promoting a scalar to a matrix.")
        return np.eye(dim, dtype=np.complex128) * complex(value)
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Expected a square matrix.")
    if dim is not None and arr.shape != (dim, dim):
        raise ValueError(f"Expected matrix shape {(dim, dim)}, got {arr.shape}.")
    return arr



def _spin_mask(label: Any, spin: str) -> bool:
    spin_key = str(spin).lower()
    if isinstance(label, (tuple, list)) and label:
        return str(label[-1]).lower() == spin_key
    text = str(label).lower()
    if text == spin_key:
        return True
    return text.endswith(f"_{spin_key}") or text.endswith(f"({spin_key})") or text.endswith(spin_key)



def _normalize_axis(axis: str) -> str:
    key = str(axis).lower()
    aliases = {"sx": "x", "sy": "y", "sz": "z"}
    key = aliases.get(key, key)
    if key not in {"x", "y", "z"}:
        raise ValueError("axis must be one of 'x', 'y', 'z', 'sx', 'sy', 'sz'.")
    return key



def _axis_matrix(axis: str) -> np.ndarray:
    key = _normalize_axis(axis)
    if key == "x":
        return SIGMA_X_NUMERIC
    if key == "y":
        return SIGMA_Y_NUMERIC
    return SIGMA_Z_NUMERIC



def _normalize_component(component: str | int | float) -> int:
    if isinstance(component, (int, float, np.integer, np.floating)):
        return 1 if float(component) >= 0 else -1
    key = str(component).lower()
    if key in {"+", "plus", "up", "positive", "p"}:
        return 1
    if key in {"-", "minus", "down", "negative", "m"}:
        return -1
    raise ValueError("component must be '+', '-', 'up', 'down', or a signed number.")



def _split_spin_label(label: Any) -> tuple[Any, str] | None:
    if isinstance(label, (tuple, list)) and len(label) >= 2:
        orbital = tuple(label[:-1]) if len(label) > 2 else label[0]
        spin = str(label[-1]).lower()
        if spin in {"up", "down"}:
            return orbital, spin
        return None

    text = str(label)
    lowered = text.lower()
    if lowered in {"up", "down"}:
        return "__single_spin_block__", lowered
    if lowered.endswith("_up"):
        return text[:-3], "up"
    if lowered.endswith("_down"):
        return text[:-5], "down"
    if lowered.endswith("(up)"):
        return text[:-4], "up"
    if lowered.endswith("(down)"):
        return text[:-6], "down"
    return None



def _spin_blocks(basis_labels: Sequence[Any]) -> dict[Any, dict[str, int]]:
    blocks: dict[Any, dict[str, int]] = {}
    for index, label in enumerate(basis_labels):
        parsed = _split_spin_label(label)
        if parsed is None:
            continue
        orbital, spin = parsed
        blocks.setdefault(orbital, {})[spin] = index
    return blocks



def spin_projector_numeric(basis_labels: Sequence[Any], spin: str) -> np.ndarray:
    mask = np.array([1.0 if _spin_mask(label, spin) else 0.0 for label in basis_labels], dtype=np.complex128)
    return np.diag(mask)



def spin_axis_projector_numeric(basis_labels: Sequence[Any], axis: str = "z", component: str | int | float = "+") -> np.ndarray:
    sign = _normalize_component(component)
    local_projector = 0.5 * (IDENTITY_2_NUMERIC + sign * _axis_matrix(axis))
    dim = len(basis_labels)
    projector = np.zeros((dim, dim), dtype=np.complex128)
    for block in _spin_blocks(basis_labels).values():
        if "up" not in block or "down" not in block:
            continue
        indices = [block["up"], block["down"]]
        projector[np.ix_(indices, indices)] += local_projector
    return projector



def spin_axis_operator_numeric(basis_labels: Sequence[Any], axis: str = "z") -> np.ndarray:
    local_operator = 0.5 * _axis_matrix(axis)
    dim = len(basis_labels)
    operator = np.zeros((dim, dim), dtype=np.complex128)
    for block in _spin_blocks(basis_labels).values():
        if "up" not in block or "down" not in block:
            continue
        indices = [block["up"], block["down"]]
        operator[np.ix_(indices, indices)] += local_operator
    return operator


def spin_rotation_matrix_numeric(theta: float = 0.0, phi: float = 0.0) -> np.ndarray:
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)
    return np.array(
        [
            [np.exp(-0.5j * phi) * c, -np.exp(-0.5j * phi) * s],
            [np.exp(0.5j * phi) * s, np.exp(0.5j * phi) * c],
        ],
        dtype=np.complex128,
    )



def _ordered_spin_blocks(basis_labels: Sequence[Any]) -> list[tuple[Any, dict[str, int]]]:
    blocks = _spin_blocks(basis_labels)
    return sorted(blocks.items(), key=lambda item: min(item[1].values()))



def _expand_orbital_data(basis_labels: Sequence[Any], values: Any, *, name: str) -> list[Any]:
    blocks = _ordered_spin_blocks(basis_labels)
    n_blocks = len(blocks)
    if np.isscalar(values) or isinstance(values, np.ndarray) and np.asarray(values).shape == (2, 2):
        return [values] * n_blocks
    if isinstance(values, dict):
        expanded = []
        for orbital, _block in blocks:
            if orbital not in values:
                raise ValueError(f"Missing {name} entry for orbital {orbital!r}.")
            expanded.append(values[orbital])
        return expanded
    seq = list(values)
    if len(seq) != n_blocks:
        raise ValueError(f"Expected {n_blocks} {name} entries, got {len(seq)}.")
    return seq



def rotated_local_spin_matrix_numeric(
    basis_labels: Sequence[Any],
    local_spin_matrix: Any,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
) -> np.ndarray:
    dim = len(basis_labels)
    gamma = np.zeros((dim, dim), dtype=np.complex128)
    rotation = spin_rotation_matrix_numeric(theta=theta, phi=phi)
    local_blocks = _expand_orbital_data(basis_labels, local_spin_matrix, name="local spin matrix")

    for (_orbital, block), local in zip(_ordered_spin_blocks(basis_labels), local_blocks):
        if "up" not in block or "down" not in block:
            continue
        local_matrix = _as_matrix(local, dim=2)
        rotated = rotation @ local_matrix @ rotation.conj().T
        indices = [block["up"], block["down"]]
        gamma[np.ix_(indices, indices)] += rotated
    return gamma



def _surface_green_semi_infinite_chain(omega: float, onsite: float, hopping: float, eta: float = 1e-12) -> complex:
    z = complex(omega, eta) - onsite
    root = np.lib.scimath.sqrt(z * z - 4.0 * hopping * hopping)
    g = (z - root) / (2.0 * hopping * hopping)
    if np.imag(g) > 0:
        g = (z + root) / (2.0 * hopping * hopping)
    return g



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
class LeadSelfEnergy:
    dim: int
    sigma_retarded_fn: Callable[[float], np.ndarray]
    mu: float = 0.0
    temperature: float = 0.0
    name: str = "lead"
    sigma_lesser_fn: Callable[[float], np.ndarray] | None = None
    sigma_greater_fn: Callable[[float], np.ndarray] | None = None
    # True when sigma_retarded does not depend on omega (wide-band limit);
    # frequency sweeps then evaluate it once and broadcast.
    omega_independent: bool = False

    @classmethod
    def wide_band(cls, gamma: Any, *, mu: float = 0.0, temperature: float = 0.0, name: str = "lead") -> "LeadSelfEnergy":
        gamma_matrix = _as_matrix(gamma) if not np.isscalar(gamma) else None
        dim = gamma_matrix.shape[0] if gamma_matrix is not None else 1

        def sigma_retarded_fn(_omega: float) -> np.ndarray:
            local_gamma = gamma_matrix if gamma_matrix is not None else _as_matrix(gamma, dim=dim)
            return -0.5j * local_gamma

        return cls(
            dim=dim,
            sigma_retarded_fn=sigma_retarded_fn,
            mu=mu,
            temperature=temperature,
            name=name,
            omega_independent=True,
        )

    @classmethod
    def polarized_wide_band(
        cls,
        basis_labels: Sequence[Any],
        gamma: Any,
        *,
        polarization: float = 0.0,
        axis: str = "z",
        mu: float = 0.0,
        temperature: float = 0.0,
        name: str = "lead",
    ) -> "LeadSelfEnergy":
        if abs(polarization) > 1.0:
            raise ValueError("polarization must satisfy |polarization| <= 1.")
        dim = len(basis_labels)
        gamma_matrix = _as_matrix(gamma, dim=dim) if np.isscalar(gamma) else _as_matrix(gamma)
        p_plus = spin_axis_projector_numeric(basis_labels, axis=axis, component="+")
        p_minus = spin_axis_projector_numeric(basis_labels, axis=axis, component="-")
        gamma_plus = p_plus @ gamma_matrix @ p_plus
        gamma_minus = p_minus @ gamma_matrix @ p_minus
        gamma_polarized = (1.0 + polarization) * gamma_plus + (1.0 - polarization) * gamma_minus
        return cls.wide_band(gamma_polarized, mu=mu, temperature=temperature, name=name)

    @classmethod
    def ferromagnetic_wide_band(
        cls,
        basis_labels: Sequence[Any],
        gamma_majority: Any,
        gamma_minority: Any,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        mu: float = 0.0,
        temperature: float = 0.0,
        name: str = "lead",
    ) -> "LeadSelfEnergy":
        majorities = _expand_orbital_data(basis_labels, gamma_majority, name="majority coupling")
        minorities = _expand_orbital_data(basis_labels, gamma_minority, name="minority coupling")
        local_blocks = [np.diag([complex(gmaj), complex(gmin)]) for gmaj, gmin in zip(majorities, minorities)]
        gamma_matrix = rotated_local_spin_matrix_numeric(basis_labels, local_blocks, theta=theta, phi=phi)
        return cls.wide_band(gamma_matrix, mu=mu, temperature=temperature, name=name)

    @classmethod
    def rotated_spin_mixing_wide_band(
        cls,
        basis_labels: Sequence[Any],
        local_spin_matrix: Any,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        mu: float = 0.0,
        temperature: float = 0.0,
        name: str = "lead",
    ) -> "LeadSelfEnergy":
        gamma_matrix = rotated_local_spin_matrix_numeric(basis_labels, local_spin_matrix, theta=theta, phi=phi)
        return cls.wide_band(gamma_matrix, mu=mu, temperature=temperature, name=name)

    @classmethod
    def semi_infinite_chain(
        cls,
        coupling_matrix: Any,
        *,
        onsite: float = 0.0,
        hopping: float = 1.0,
        mu: float = 0.0,
        temperature: float = 0.0,
        eta: float = 1e-12,
        name: str = "lead",
    ) -> "LeadSelfEnergy":
        coupling = _as_matrix(coupling_matrix) if not np.isscalar(coupling_matrix) else None
        dim = coupling.shape[0] if coupling is not None else 1

        def sigma_retarded_fn(omega: float) -> np.ndarray:
            local_coupling = coupling if coupling is not None else _as_matrix(coupling_matrix, dim=dim)
            g_surface = _surface_green_semi_infinite_chain(omega, onsite=onsite, hopping=hopping, eta=eta)
            return local_coupling.conj().T @ (g_surface * np.eye(dim, dtype=np.complex128)) @ local_coupling

        return cls(dim=dim, sigma_retarded_fn=sigma_retarded_fn, mu=mu, temperature=temperature, name=name)

    @classmethod
    def sampled(
        cls,
        omega_grid: np.ndarray,
        sigma_retarded_values: np.ndarray,
        *,
        mu: float = 0.0,
        temperature: float = 0.0,
        name: str = "lead",
        sigma_lesser_values: np.ndarray | None = None,
        sigma_greater_values: np.ndarray | None = None,
    ) -> "LeadSelfEnergy":
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

        return cls(
            dim=dim,
            sigma_retarded_fn=sigma_retarded_fn,
            mu=mu,
            temperature=temperature,
            name=name,
            sigma_lesser_fn=lesser_fn,
            sigma_greater_fn=greater_fn,
        )

    @classmethod
    def from_retarded(
        cls,
        sigma_retarded_fn: Callable[[float], np.ndarray],
        *,
        dim: int,
        mu: float = 0.0,
        temperature: float = 0.0,
        name: str = "lead",
        sigma_lesser_fn: Callable[[float], np.ndarray] | None = None,
        sigma_greater_fn: Callable[[float], np.ndarray] | None = None,
    ) -> "LeadSelfEnergy":
        return cls(
            dim=dim,
            sigma_retarded_fn=sigma_retarded_fn,
            mu=mu,
            temperature=temperature,
            name=name,
            sigma_lesser_fn=sigma_lesser_fn,
            sigma_greater_fn=sigma_greater_fn,
        )

    def sigma_retarded(self, omega: float) -> np.ndarray:
        return _as_matrix(self.sigma_retarded_fn(float(omega)), dim=self.dim)

    def sigma_advanced(self, omega: float) -> np.ndarray:
        return self.sigma_retarded(omega).conj().T

    def gamma(self, omega: float) -> np.ndarray:
        sigma_r = self.sigma_retarded(omega)
        sigma_a = sigma_r.conj().T
        return 1j * (sigma_r - sigma_a)

    def sigma_lesser(self, omega: float) -> np.ndarray:
        if self.sigma_lesser_fn is not None:
            return _as_matrix(self.sigma_lesser_fn(float(omega)), dim=self.dim)
        f = float(fermi_dirac(np.array([omega]), mu=self.mu, temperature=self.temperature)[0])
        return 1j * f * self.gamma(omega)

    def sigma_greater(self, omega: float) -> np.ndarray:
        if self.sigma_greater_fn is not None:
            return _as_matrix(self.sigma_greater_fn(float(omega)), dim=self.dim)
        f = float(fermi_dirac(np.array([omega]), mu=self.mu, temperature=self.temperature)[0])
        return 1j * (f - 1.0) * self.gamma(omega)


def _lead_component_stack(
    lead: LeadSelfEnergy,
    grid: np.ndarray,
    component: str,
    *,
    workers: int | None = None,
) -> np.ndarray:
    """Self-energy component over a grid: (n, d, d) stack, or (d, d) when omega-independent."""
    if component == "retarded" and lead.omega_independent:
        return lead.sigma_retarded(0.0)
    if component == "lesser" and lead.omega_independent and lead.sigma_lesser_fn is None:
        occupation = fermi_dirac(grid, mu=lead.mu, temperature=lead.temperature)
        return 1j * occupation[:, None, None] * lead.gamma(0.0)
    if component == "greater" and lead.omega_independent and lead.sigma_greater_fn is None:
        occupation = fermi_dirac(grid, mu=lead.mu, temperature=lead.temperature)
        return 1j * (occupation - 1.0)[:, None, None] * lead.gamma(0.0)
    return sigma_stack(getattr(lead, f"sigma_{component}"), grid, workers=workers)


@dataclass
class MatrixTransportView:
    hamiltonian: np.ndarray
    basis_labels: list[Any]
    left_lead: LeadSelfEnergy
    right_lead: LeadSelfEnergy

    def __post_init__(self) -> None:
        dim = self.hamiltonian.shape[0]
        if self.hamiltonian.shape != (dim, dim):
            raise ValueError("Hamiltonian must be square.")
        if len(self.basis_labels) != dim:
            raise ValueError("basis_labels length must match Hamiltonian dimension.")
        if self.left_lead.dim != dim or self.right_lead.dim != dim:
            raise ValueError("Lead dimensions must match Hamiltonian dimension.")

    @property
    def dim(self) -> int:
        return self.hamiltonian.shape[0]

    def _lead(self, lead: str) -> LeadSelfEnergy:
        if lead == "left":
            return self.left_lead
        if lead == "right":
            return self.right_lead
        raise ValueError("lead must be 'left' or 'right'.")

    def _spin_projector(self, spin: str) -> np.ndarray:
        return spin_projector_numeric(self.basis_labels, spin)

    def _spin_axis_projector(self, axis: str, component: str | int | float) -> np.ndarray:
        return spin_axis_projector_numeric(self.basis_labels, axis=axis, component=component)

    def _project_spin_block(self, matrix: np.ndarray, spin: str) -> np.ndarray:
        projector = self._spin_projector(spin)
        return projector @ matrix @ projector

    def _project_spin_axis_block(self, matrix: np.ndarray, axis: str, component: str | int | float) -> np.ndarray:
        projector = self._spin_axis_projector(axis, component)
        return projector @ matrix @ projector

    def _sigma_total_retarded(self, omega: float) -> np.ndarray:
        return self.left_lead.sigma_retarded(omega) + self.right_lead.sigma_retarded(omega)

    def sigma_retarded_total(self, omega: float) -> np.ndarray:
        return self._sigma_total_retarded(omega)

    def sigma_advanced_total(self, omega: float) -> np.ndarray:
        return self.sigma_retarded_total(omega).conj().T

    def sigma_lesser_total(self, omega: float) -> np.ndarray:
        return self.left_lead.sigma_lesser(omega) + self.right_lead.sigma_lesser(omega)

    def sigma_greater_total(self, omega: float) -> np.ndarray:
        return self.left_lead.sigma_greater(omega) + self.right_lead.sigma_greater(omega)

    def sigma_keldysh_total(self, omega: float) -> np.ndarray:
        return self.sigma_lesser_total(omega) + self.sigma_greater_total(omega)

    def retarded(self, omega: float, eta: float = 0.0) -> np.ndarray:
        identity = np.eye(self.dim, dtype=np.complex128)
        sigma_r = self._sigma_total_retarded(float(omega))
        return np.linalg.inv((omega + 1j * eta) * identity - self.hamiltonian - sigma_r)

    def advanced(self, omega: float, eta: float = 0.0) -> np.ndarray:
        return self.retarded(omega, eta=eta).conj().T

    def lesser(self, omega: float, eta: float = 0.0) -> np.ndarray:
        g_r = self.retarded(omega, eta=eta)
        g_a = g_r.conj().T
        sigma_less = self.left_lead.sigma_lesser(omega) + self.right_lead.sigma_lesser(omega)
        return g_r @ sigma_less @ g_a

    def greater(self, omega: float, eta: float = 0.0) -> np.ndarray:
        g_r = self.retarded(omega, eta=eta)
        g_a = g_r.conj().T
        sigma_greater = self.sigma_greater_total(omega)
        return g_r @ sigma_greater @ g_a

    def keldysh_green(self, omega: float, eta: float = 0.0) -> np.ndarray:
        return self.lesser(omega, eta=eta) + self.greater(omega, eta=eta)

    def _lead_sigma_stacks(
        self,
        omega_grid: np.ndarray,
        *,
        components: Sequence[str] = ("retarded",),
        workers: int | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Evaluate (left, right) lead self-energy stacks on the host.

        Entries are ``(n, dim, dim)`` stacks, or plain ``(dim, dim)`` matrices
        for omega-independent components (wide-band leads) — the batched
        kernels broadcast those without copying, which keeps the setup cost
        O(d^2) instead of O(n d^2).
        """
        grid = np.asarray(omega_grid, dtype=float)
        stacks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for component in components:
            stacks[component] = (
                _lead_component_stack(self.left_lead, grid, component, workers=workers),
                _lead_component_stack(self.right_lead, grid, component, workers=workers),
            )
        return stacks

    def retarded_values(
        self,
        omega_grid: np.ndarray,
        eta: float = 0.0,
        *,
        backend: Any = None,
        workers: int | None = None,
        precision: str | None = None,
    ) -> np.ndarray:
        """
        Retarded Green function on a frequency grid via blocked batched inversions, shape (n, dim, dim).

        ``precision="single"`` runs the inversions in complex64 — roughly 1e-6
        relative accuracy, and the fast path on consumer GPUs whose float64
        throughput is capped.
        """
        xp = get_backend(backend)
        dtype = precision_dtype(precision)

        def compute(subgrid: np.ndarray) -> np.ndarray:
            sig_l, sig_r = self._lead_sigma_stacks(subgrid)["retarded"]
            g_r = batched_retarded_green(self.hamiltonian, xp.asarray(sig_l + sig_r), subgrid, eta=eta, xp=xp, dtype=dtype)
            return to_numpy(g_r)

        return blocked_over_grid(compute, omega_grid, self.dim, workers=workers, target_bytes=default_block_target(xp))

    def lesser_values(
        self,
        omega_grid: np.ndarray,
        eta: float = 0.0,
        *,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        xp = get_backend(backend)

        def compute(subgrid: np.ndarray) -> np.ndarray:
            stacks = self._lead_sigma_stacks(subgrid, components=("retarded", "lesser"))
            sig_l, sig_r = stacks["retarded"]
            g_r = batched_retarded_green(self.hamiltonian, xp.asarray(sig_l + sig_r), subgrid, eta=eta, xp=xp)
            less_l, less_r = stacks["lesser"]
            return to_numpy(batched_keldysh_component(g_r, xp.asarray(less_l + less_r), xp=xp))

        return blocked_over_grid(compute, omega_grid, self.dim, workers=workers, target_bytes=default_block_target(xp))

    def greater_values(
        self,
        omega_grid: np.ndarray,
        eta: float = 0.0,
        *,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        xp = get_backend(backend)

        def compute(subgrid: np.ndarray) -> np.ndarray:
            stacks = self._lead_sigma_stacks(subgrid, components=("retarded", "greater"))
            sig_l, sig_r = stacks["retarded"]
            g_r = batched_retarded_green(self.hamiltonian, xp.asarray(sig_l + sig_r), subgrid, eta=eta, xp=xp)
            great_l, great_r = stacks["greater"]
            return to_numpy(batched_keldysh_component(g_r, xp.asarray(great_l + great_r), xp=xp))

        return blocked_over_grid(compute, omega_grid, self.dim, workers=workers, target_bytes=default_block_target(xp))

    def current_spectral_density(self, omega: float, lead: str = "left", charge: float = 1.0, eta: float = 0.0) -> float:
        selected = self._lead(lead)
        sigma_less = selected.sigma_lesser(omega)
        sigma_greater = selected.sigma_greater(omega)
        g_less = self.lesser(omega, eta=eta)
        g_greater = self.greater(omega, eta=eta)
        integrand = (charge / (2.0 * np.pi)) * np.trace(sigma_less @ g_greater - sigma_greater @ g_less)
        return float(np.real(integrand))

    def current_spectral_density_values(
        self,
        omega_grid: np.ndarray,
        lead: str = "left",
        charge: float = 1.0,
        eta: float = 0.0,
        *,
        backend: Any = None,
        workers: int | None = None,
        spin_projector: np.ndarray | None = None,
    ) -> np.ndarray:
        xp = get_backend(backend)
        lead_index = 0 if self._lead(lead) is self.left_lead else 1

        def compute(subgrid: np.ndarray) -> np.ndarray:
            stacks = self._lead_sigma_stacks(subgrid, components=("retarded", "lesser", "greater"))
            sig_l, sig_r = stacks["retarded"]
            g_r = batched_retarded_green(self.hamiltonian, xp.asarray(sig_l + sig_r), subgrid, eta=eta, xp=xp)
            less_l, less_r = stacks["lesser"]
            great_l, great_r = stacks["greater"]
            g_lesser = batched_keldysh_component(g_r, xp.asarray(less_l + less_r), xp=xp)
            g_greater = batched_keldysh_component(g_r, xp.asarray(great_l + great_r), xp=xp)
            sigma_lesser_lead = (less_l, less_r)[lead_index]
            sigma_greater_lead = (great_l, great_r)[lead_index]
            if spin_projector is not None:
                projected_lesser = spin_projector @ sigma_lesser_lead @ spin_projector
                projected_greater = spin_projector @ sigma_greater_lead @ spin_projector
            else:
                projected_lesser = sigma_lesser_lead
                projected_greater = sigma_greater_lead
            density = batched_current_spectral_density(
                g_lesser,
                g_greater,
                xp.asarray(projected_lesser),
                xp.asarray(projected_greater),
                charge=charge,
                xp=xp,
            )
            return to_numpy(density).astype(float)

        return blocked_over_grid(compute, omega_grid, self.dim, workers=workers, target_bytes=default_block_target(xp))

    def current_from_keldysh(self, omega_grid: np.ndarray, lead: str = "left", charge: float = 1.0, eta: float = 0.0) -> float:
        omega_grid = np.asarray(omega_grid, dtype=float)
        values = self.current_spectral_density_values(omega_grid, lead=lead, charge=charge, eta=eta)
        return float(np.trapezoid(values, omega_grid))

    def meir_wingreen_current(self, omega_grid: np.ndarray, lead: str = "left", charge: float = 1.0, eta: float = 0.0) -> float:
        return self.current_from_keldysh(omega_grid, lead=lead, charge=charge, eta=eta)

    def spin_resolved_current_spectral_density(
        self,
        omega: float,
        spin: str,
        *,
        lead: str = "left",
        charge: float = 1.0,
        eta: float = 0.0,
        axis: str = "z",
    ) -> float:
        selected = self._lead(lead)
        sigma_less = self._project_spin_axis_block(selected.sigma_lesser(omega), axis, spin)
        sigma_greater = self._project_spin_axis_block(selected.sigma_greater(omega), axis, spin)
        g_less = self.lesser(omega, eta=eta)
        g_greater = self.greater(omega, eta=eta)
        integrand = (charge / (2.0 * np.pi)) * np.trace(sigma_less @ g_greater - sigma_greater @ g_less)
        return float(np.real(integrand))

    def spin_resolved_current_spectral_density_values(
        self,
        omega_grid: np.ndarray,
        spin: str,
        *,
        lead: str = "left",
        charge: float = 1.0,
        eta: float = 0.0,
        axis: str = "z",
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        projector = self._spin_axis_projector(axis, spin)
        return self.current_spectral_density_values(
            omega_grid,
            lead=lead,
            charge=charge,
            eta=eta,
            backend=backend,
            workers=workers,
            spin_projector=projector,
        )

    def spin_resolved_current_from_keldysh(
        self,
        omega_grid: np.ndarray,
        spin: str,
        *,
        lead: str = "left",
        charge: float = 1.0,
        eta: float = 0.0,
        axis: str = "z",
    ) -> float:
        omega_grid = np.asarray(omega_grid, dtype=float)
        values = self.spin_resolved_current_spectral_density_values(omega_grid, spin, lead=lead, charge=charge, eta=eta, axis=axis)
        return float(np.trapezoid(values, omega_grid))

    def spin_current_spectral_density(self, omega: float, lead: str = "left", charge: float = 1.0, eta: float = 0.0, axis: str = "z") -> float:
        return self.spin_resolved_current_spectral_density(omega, "+", lead=lead, charge=charge, eta=eta, axis=axis) - self.spin_resolved_current_spectral_density(omega, "-", lead=lead, charge=charge, eta=eta, axis=axis)

    def spin_current_spectral_density_values(
        self,
        omega_grid: np.ndarray,
        lead: str = "left",
        charge: float = 1.0,
        eta: float = 0.0,
        axis: str = "z",
        *,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        plus = self.spin_resolved_current_spectral_density_values(omega_grid, "+", lead=lead, charge=charge, eta=eta, axis=axis, backend=backend, workers=workers)
        minus = self.spin_resolved_current_spectral_density_values(omega_grid, "-", lead=lead, charge=charge, eta=eta, axis=axis, backend=backend, workers=workers)
        return plus - minus

    def spin_current_from_keldysh(self, omega_grid: np.ndarray, lead: str = "left", charge: float = 1.0, eta: float = 0.0, axis: str = "z") -> float:
        omega_grid = np.asarray(omega_grid, dtype=float)
        values = self.spin_current_spectral_density_values(omega_grid, lead=lead, charge=charge, eta=eta, axis=axis)
        return float(np.trapezoid(values, omega_grid))

    def spin_current_vector_from_keldysh(self, omega_grid: np.ndarray, lead: str = "left", charge: float = 1.0, eta: float = 0.0) -> dict[str, float]:
        return {
            axis: self.spin_current_from_keldysh(omega_grid, lead=lead, charge=charge, eta=eta, axis=axis)
            for axis in ("x", "y", "z")
        }

    def spectral_function(self, omega: float, eta: float = 0.0) -> np.ndarray:
        g_r = self.retarded(omega, eta=eta)
        g_a = g_r.conj().T
        return 1j * (g_r - g_a)

    def spectral_density(self, omega: float, eta: float = 0.0) -> np.ndarray:
        return self.spectral_function(omega, eta=eta) / (2.0 * np.pi)

    def transmission(self, omega: float, eta: float = 0.0) -> float:
        g_r = self.retarded(omega, eta=eta)
        g_a = g_r.conj().T
        gamma_l = self.left_lead.gamma(omega)
        gamma_r = self.right_lead.gamma(omega)
        return float(np.real(np.trace(gamma_l @ g_r @ gamma_r @ g_a)))

    def _transmission_values_batched(
        self,
        omega_grid: np.ndarray,
        eta: float,
        *,
        backend: Any,
        workers: int | None,
        precision: str | None = None,
        left_projector: np.ndarray | None = None,
        right_projector: np.ndarray | None = None,
    ) -> np.ndarray:
        xp = get_backend(backend)
        dtype = precision_dtype(precision)

        def compute(subgrid: np.ndarray) -> np.ndarray:
            sig_l, sig_r = self._lead_sigma_stacks(subgrid)["retarded"]
            g_r = batched_retarded_green(self.hamiltonian, xp.asarray(sig_l + sig_r), subgrid, eta=eta, xp=xp, dtype=dtype)
            gamma_l = gamma_from_sigma_stack(sig_l).astype(dtype, copy=False)
            gamma_r = gamma_from_sigma_stack(sig_r).astype(dtype, copy=False)
            if left_projector is not None:
                gamma_l = (left_projector @ gamma_l @ left_projector).astype(dtype, copy=False)
            if right_projector is not None:
                gamma_r = (right_projector @ gamma_r @ right_projector).astype(dtype, copy=False)
            values = batched_transmission(g_r, xp.asarray(gamma_l), xp.asarray(gamma_r), xp=xp)
            return to_numpy(values).astype(float)

        return blocked_over_grid(compute, omega_grid, self.dim, workers=workers, target_bytes=default_block_target(xp))

    def transmission_values(
        self,
        omega_grid: np.ndarray,
        eta: float = 0.0,
        *,
        backend: Any = None,
        workers: int | None = None,
        precision: str | None = None,
    ) -> np.ndarray:
        """
        Landauer transmission on a frequency grid (batched).

        ``workers=N`` maps memory-capped frequency blocks onto N CPU threads;
        ``backend="cupy"`` runs on a CUDA GPU; ``precision="single"`` uses
        complex64 (~1e-6 accuracy, the fast path on consumer GPUs).
        """
        return self._transmission_values_batched(omega_grid, eta, backend=backend, workers=workers, precision=precision)

    def spin_transmission(self, omega: float, left_spin: str, right_spin: str, eta: float = 0.0) -> float:
        g_r = self.retarded(omega, eta=eta)
        g_a = g_r.conj().T
        gamma_l = self._project_spin_block(self.left_lead.gamma(omega), left_spin)
        gamma_r = self._project_spin_block(self.right_lead.gamma(omega), right_spin)
        return float(np.real(np.trace(gamma_l @ g_r @ gamma_r @ g_a)))

    def spin_transmission_values(
        self,
        omega_grid: np.ndarray,
        left_spin: str,
        right_spin: str,
        eta: float = 0.0,
        *,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        return self._transmission_values_batched(
            omega_grid,
            eta,
            backend=backend,
            workers=workers,
            left_projector=self._spin_projector(left_spin),
            right_projector=self._spin_projector(right_spin),
        )

    def spin_resolved_transmission(self, omega: float, spin: str, eta: float = 0.0, axis: str = "z") -> float:
        g_r = self.retarded(omega, eta=eta)
        g_a = g_r.conj().T
        gamma_l = self.left_lead.gamma(omega)
        gamma_r = self._project_spin_axis_block(self.right_lead.gamma(omega), axis, spin)
        return float(np.real(np.trace(gamma_l @ g_r @ gamma_r @ g_a)))

    def spin_resolved_transmission_values(
        self,
        omega_grid: np.ndarray,
        spin: str,
        eta: float = 0.0,
        axis: str = "z",
        *,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        return self._transmission_values_batched(
            omega_grid,
            eta,
            backend=backend,
            workers=workers,
            right_projector=self._spin_axis_projector(axis, spin),
        )

    def spin_polarization(self, omega: float, eta: float = 0.0, axis: str = "z") -> float:
        plus = self.spin_resolved_transmission(omega, "+", eta=eta, axis=axis)
        minus = self.spin_resolved_transmission(omega, "-", eta=eta, axis=axis)
        total = plus + minus
        if abs(total) < 1e-15:
            return 0.0
        return float((plus - minus) / total)

    def spin_polarization_values(
        self,
        omega_grid: np.ndarray,
        eta: float = 0.0,
        axis: str = "z",
        *,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        plus = self.spin_resolved_transmission_values(omega_grid, "+", eta=eta, axis=axis, backend=backend, workers=workers)
        minus = self.spin_resolved_transmission_values(omega_grid, "-", eta=eta, axis=axis, backend=backend, workers=workers)
        total = plus + minus
        polarization = np.zeros_like(total)
        mask = np.abs(total) >= 1e-15
        polarization[mask] = (plus[mask] - minus[mask]) / total[mask]
        return polarization

    def conductance(self, mu: float = 0.0, temperature: float = 0.0, charge: float = 1.0, eta: float = 0.0, omega_grid: np.ndarray | None = None) -> float:
        prefactor = charge**2 / (2.0 * np.pi)
        if temperature <= 0.0:
            return prefactor * self.transmission(mu, eta=eta)
        if omega_grid is None:
            raise ValueError("omega_grid is required for finite-temperature conductance.")
        omega_grid = np.asarray(omega_grid, dtype=float)
        f = fermi_dirac(omega_grid, mu=mu, temperature=temperature)
        kernel = np.real(f * (1.0 - f) / temperature)
        transmission_vals = self.transmission_values(omega_grid, eta=eta)
        return float(prefactor * np.trapezoid(kernel * transmission_vals, omega_grid))

    def spin_conductance(self, mu: float = 0.0, temperature: float = 0.0, charge: float = 1.0, eta: float = 0.0, omega_grid: np.ndarray | None = None, axis: str = "z") -> float:
        prefactor = charge**2 / (2.0 * np.pi)
        if temperature <= 0.0:
            polarized = self.spin_resolved_transmission(mu, "+", eta=eta, axis=axis) - self.spin_resolved_transmission(mu, "-", eta=eta, axis=axis)
            return prefactor * polarized
        if omega_grid is None:
            raise ValueError("omega_grid is required for finite-temperature spin conductance.")
        omega_grid = np.asarray(omega_grid, dtype=float)
        f = fermi_dirac(omega_grid, mu=mu, temperature=temperature)
        kernel = np.real(f * (1.0 - f) / temperature)
        polarized = self.spin_resolved_transmission_values(omega_grid, "+", eta=eta, axis=axis) - self.spin_resolved_transmission_values(omega_grid, "-", eta=eta, axis=axis)
        return float(prefactor * np.trapezoid(kernel * polarized, omega_grid))

    def landauer_current(self, omega_grid: np.ndarray, mu_left: float, mu_right: float, temperature: float = 0.0, charge: float = 1.0, eta: float = 0.0) -> float:
        omega_grid = np.asarray(omega_grid, dtype=float)
        window = fermi_dirac(omega_grid, mu=mu_left, temperature=temperature) - fermi_dirac(omega_grid, mu=mu_right, temperature=temperature)
        transmission_vals = self.transmission_values(omega_grid, eta=eta)
        integrand = (charge / (2.0 * np.pi)) * window * transmission_vals
        return float(np.trapezoid(integrand, omega_grid))

    def spin_resolved_landauer_current(
        self,
        omega_grid: np.ndarray,
        spin: str,
        *,
        mu_left: float,
        mu_right: float,
        temperature: float = 0.0,
        charge: float = 1.0,
        eta: float = 0.0,
        axis: str = "z",
    ) -> float:
        omega_grid = np.asarray(omega_grid, dtype=float)
        window = fermi_dirac(omega_grid, mu=mu_left, temperature=temperature) - fermi_dirac(omega_grid, mu=mu_right, temperature=temperature)
        transmission_vals = self.spin_resolved_transmission_values(omega_grid, spin, eta=eta, axis=axis)
        integrand = (charge / (2.0 * np.pi)) * window * transmission_vals
        return float(np.trapezoid(integrand, omega_grid))

    def spin_landauer_current(
        self,
        omega_grid: np.ndarray,
        *,
        mu_left: float,
        mu_right: float,
        temperature: float = 0.0,
        charge: float = 1.0,
        eta: float = 0.0,
        axis: str = "z",
    ) -> float:
        plus = self.spin_resolved_landauer_current(omega_grid, "+", mu_left=mu_left, mu_right=mu_right, temperature=temperature, charge=charge, eta=eta, axis=axis)
        minus = self.spin_resolved_landauer_current(omega_grid, "-", mu_left=mu_left, mu_right=mu_right, temperature=temperature, charge=charge, eta=eta, axis=axis)
        return float(plus - minus)

    def current_spin_polarization(
        self,
        omega_grid: np.ndarray,
        *,
        mu_left: float,
        mu_right: float,
        temperature: float = 0.0,
        charge: float = 1.0,
        eta: float = 0.0,
        axis: str = "z",
    ) -> float:
        charge_current = self.landauer_current(omega_grid, mu_left=mu_left, mu_right=mu_right, temperature=temperature, charge=charge, eta=eta)
        if abs(charge_current) < 1e-15:
            return 0.0
        spin_current = self.spin_landauer_current(omega_grid, mu_left=mu_left, mu_right=mu_right, temperature=temperature, charge=charge, eta=eta, axis=axis)
        return float(spin_current / charge_current)

    def spin_landauer_current_vector(
        self,
        omega_grid: np.ndarray,
        *,
        mu_left: float,
        mu_right: float,
        temperature: float = 0.0,
        charge: float = 1.0,
        eta: float = 0.0,
    ) -> dict[str, float]:
        return {
            axis: self.spin_landauer_current(
                omega_grid,
                mu_left=mu_left,
                mu_right=mu_right,
                temperature=temperature,
                charge=charge,
                eta=eta,
                axis=axis,
            )
            for axis in ("x", "y", "z")
        }

    def spin_conductance_vector(
        self,
        mu: float = 0.0,
        temperature: float = 0.0,
        charge: float = 1.0,
        eta: float = 0.0,
        omega_grid: np.ndarray | None = None,
    ) -> dict[str, float]:
        return {
            axis: self.spin_conductance(mu=mu, temperature=temperature, charge=charge, eta=eta, omega_grid=omega_grid, axis=axis)
            for axis in ("x", "y", "z")
        }

    def current_spin_polarization_vector(
        self,
        omega_grid: np.ndarray,
        *,
        mu_left: float,
        mu_right: float,
        temperature: float = 0.0,
        charge: float = 1.0,
        eta: float = 0.0,
    ) -> dict[str, float]:
        return {
            axis: self.current_spin_polarization(
                omega_grid,
                mu_left=mu_left,
                mu_right=mu_right,
                temperature=temperature,
                charge=charge,
                eta=eta,
                axis=axis,
            )
            for axis in ("x", "y", "z")
        }

    def keldysh_view(self):
        from .keldysh import KeldyshTransportView

        return KeldyshTransportView(self)


@dataclass
class MatrixDevice:
    hamiltonian: np.ndarray
    basis_labels: list[Any]
    name: str = "matrix_device"

    def __post_init__(self) -> None:
        self.hamiltonian = np.asarray(self.hamiltonian, dtype=np.complex128)
        if self.hamiltonian.ndim != 2 or self.hamiltonian.shape[0] != self.hamiltonian.shape[1]:
            raise ValueError("hamiltonian must be a square matrix.")
        if len(self.basis_labels) != self.hamiltonian.shape[0]:
            raise ValueError("basis_labels length must match Hamiltonian dimension.")

    @property
    def dim(self) -> int:
        return self.hamiltonian.shape[0]

    def greens(self, left_lead: LeadSelfEnergy, right_lead: LeadSelfEnergy) -> MatrixTransportView:
        return MatrixTransportView(self.hamiltonian, list(self.basis_labels), left_lead, right_lead)

    def transport(self, left_lead: LeadSelfEnergy, right_lead: LeadSelfEnergy) -> MatrixTransportView:
        return self.greens(left_lead, right_lead)


class SpinfulSingleSite(MatrixDevice):
    def __init__(self, *, eps_up: float = 0.0, eps_down: float = 0.0, spin_flip: complex = 0.0):
        h = np.array([[eps_up, spin_flip], [np.conjugate(spin_flip), eps_down]], dtype=np.complex128)
        super().__init__(hamiltonian=h, basis_labels=["up", "down"], name="spinful_single_site")


class SpinfulDimer(MatrixDevice):
    def __init__(
        self,
        *,
        eps_left_up: float = 0.0,
        eps_left_down: float = 0.0,
        eps_right_up: float = 0.0,
        eps_right_down: float = 0.0,
        hopping: complex = 1.0,
        spin_orbit: complex = 0.0,
        onsite_spin_flip_left: complex = 0.0,
        onsite_spin_flip_right: complex = 0.0,
    ):
        h = np.zeros((4, 4), dtype=np.complex128)
        h[0, 0] = eps_left_up
        h[1, 1] = eps_left_down
        h[2, 2] = eps_right_up
        h[3, 3] = eps_right_down
        h[0, 1] = onsite_spin_flip_left
        h[1, 0] = np.conjugate(onsite_spin_flip_left)
        h[2, 3] = onsite_spin_flip_right
        h[3, 2] = np.conjugate(onsite_spin_flip_right)
        hop_block = np.array([[hopping, spin_orbit], [-np.conjugate(spin_orbit), hopping]], dtype=np.complex128)
        h[2:4, 0:2] = hop_block
        h[0:2, 2:4] = hop_block.conj().T
        labels = ["left_up", "left_down", "right_up", "right_down"]
        super().__init__(hamiltonian=h, basis_labels=labels, name="spinful_dimer")


class RashbaRingDevice(MatrixDevice):
    def __init__(
        self,
        *,
        n_sites: int,
        gamma: float = 1.0,
        lambda_r: float = 0.0,
        phi_over_phi0: float = 0.0,
        onsite_up: np.ndarray | None = None,
        onsite_down: np.ndarray | None = None,
        u_hubbard: float = 0.0,
        mean_n_up: np.ndarray | None = None,
        mean_n_down: np.ndarray | None = None,
    ):
        h = build_rashba_hubbard_ring_real_space(
            n_sites=n_sites,
            gamma=gamma,
            lambda_r=lambda_r,
            phi_over_phi0=phi_over_phi0,
            onsite_up=onsite_up,
            onsite_down=onsite_down,
            u_hubbard=u_hubbard,
            mean_n_up=mean_n_up,
            mean_n_down=mean_n_down,
        )
        labels = [f"site{site}_{spin}" for site in range(n_sites) for spin in ("up", "down")]
        super().__init__(hamiltonian=h, basis_labels=labels, name="rashba_ring")
