"""Closed Aharonov-Bohm ring helpers with persistent-current observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .greens import advanced_green, retarded_green
from .hamiltonians import build_rashba_hubbard_ring_real_space, real_to_k_space, split_k_blocks
from .mean_field import CollinearHartreeResult, collinear_hartree_self_consistent
from .transport import current_density_omega, drude_weight as finite_difference_drude_weight, persistent_current as persistent_current_from_hamiltonian, persistent_spin_current as persistent_spin_current_from_hamiltonian, persistent_spin_resolved_current as persistent_spin_resolved_current_from_hamiltonian, spin_current_density_omega, spin_resolved_current_density_omega


def _chemical_potential_from_hf(result: CollinearHartreeResult) -> float:
    occupied = np.where(result.occupations > 1e-12)[0]
    if occupied.size == 0:
        return float(result.eigenvalues[0] - 1.0)
    last = int(occupied[-1])
    if last >= result.eigenvalues.size - 1:
        return float(result.eigenvalues[-1] + 1.0)
    if result.occupations[last] < 1.0 - 1e-12:
        return float(result.eigenvalues[last])
    return float(0.5 * (result.eigenvalues[last] + result.eigenvalues[last + 1]))


@dataclass
class AharonovBohmRing:
    n_sites: int
    gamma: float = 1.0
    lambda_r: float = 0.0
    onsite_up: np.ndarray | None = None
    onsite_down: np.ndarray | None = None
    u_hubbard: float = 0.0
    mean_n_up: np.ndarray | None = None
    mean_n_down: np.ndarray | None = None
    name: str = "aharonov_bohm_ring"

    def hamiltonian(self, phi_over_phi0: float = 0.0) -> np.ndarray:
        return build_rashba_hubbard_ring_real_space(
            n_sites=self.n_sites,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            phi_over_phi0=phi_over_phi0,
            onsite_up=self.onsite_up,
            onsite_down=self.onsite_down,
            u_hubbard=self.u_hubbard,
            mean_n_up=self.mean_n_up,
            mean_n_down=self.mean_n_down,
        )

    def spectrum(self, phi_over_phi0: float = 0.0) -> np.ndarray:
        return np.linalg.eigvalsh(self.hamiltonian(phi_over_phi0=phi_over_phi0))

    def eigensystem(self, phi_over_phi0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(self.hamiltonian(phi_over_phi0=phi_over_phi0))

    def k_space(self, phi_over_phi0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        return real_to_k_space(self.hamiltonian(phi_over_phi0=phi_over_phi0), n_sites=self.n_sites, spin_dim=2)

    def k_blocks(self, phi_over_phi0: float = 0.0, *, require_block_diagonal: bool = True, atol: float = 1e-10) -> list[np.ndarray]:
        h_k, _ = self.k_space(phi_over_phi0=phi_over_phi0)
        return split_k_blocks(h_k, n_sites=self.n_sites, spin_dim=2, require_block_diagonal=require_block_diagonal, atol=atol)

    def current_density_spectrum(
        self,
        omega_grid: np.ndarray,
        *,
        phi_over_phi0: float,
        mu: float = 0.0,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
    ) -> np.ndarray:
        omega_grid = np.asarray(omega_grid, dtype=float)
        h = self.hamiltonian(phi_over_phi0=phi_over_phi0)
        g_ret = retarded_green(h, omega_grid, eta=eta)
        g_adv = advanced_green(h, omega_grid, eta=eta)
        return np.array(
            [
                current_density_omega(
                    g_ret=g_ret[index],
                    g_adv=g_adv[index],
                    omega=float(omega),
                    n_sites=self.n_sites,
                    gamma=self.gamma,
                    lambda_r=self.lambda_r,
                    phi_over_phi0=phi_over_phi0,
                    mu=mu,
                    temperature=temperature,
                    charge=charge,
                )
                for index, omega in enumerate(omega_grid)
            ],
            dtype=float,
        )

    def persistent_current(
        self,
        omega_grid: np.ndarray,
        *,
        phi_over_phi0: float,
        mu: float = 0.0,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
    ) -> float:
        return persistent_current_from_hamiltonian(
            self.hamiltonian(phi_over_phi0=phi_over_phi0),
            omega_grid=omega_grid,
            n_sites=self.n_sites,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            phi_over_phi0=phi_over_phi0,
            mu=mu,
            temperature=temperature,
            eta=eta,
            charge=charge,
        )

    def persistent_current_vs_flux(
        self,
        flux_values: np.ndarray,
        omega_grid: np.ndarray,
        *,
        mu: float = 0.0,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
    ) -> np.ndarray:
        flux_values = np.asarray(flux_values, dtype=float)
        return np.array(
            [
                self.persistent_current(
                    omega_grid,
                    phi_over_phi0=float(phi),
                    mu=mu,
                    temperature=temperature,
                    eta=eta,
                    charge=charge,
                )
                for phi in flux_values
            ],
            dtype=float,
        )

    def persistent_spin_resolved_current(
        self,
        omega_grid: np.ndarray,
        *,
        phi_over_phi0: float,
        axis: str = "z",
        component: str | int | float = "+",
        mu: float = 0.0,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
    ) -> float:
        return persistent_spin_resolved_current_from_hamiltonian(
            self.hamiltonian(phi_over_phi0=phi_over_phi0),
            omega_grid=omega_grid,
            n_sites=self.n_sites,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            phi_over_phi0=phi_over_phi0,
            axis=axis,
            component=component,
            mu=mu,
            temperature=temperature,
            eta=eta,
            charge=charge,
        )

    def persistent_spin_current(
        self,
        omega_grid: np.ndarray,
        *,
        phi_over_phi0: float,
        axis: str = "z",
        mu: float = 0.0,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
    ) -> float:
        return persistent_spin_current_from_hamiltonian(
            self.hamiltonian(phi_over_phi0=phi_over_phi0),
            omega_grid=omega_grid,
            n_sites=self.n_sites,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            phi_over_phi0=phi_over_phi0,
            axis=axis,
            mu=mu,
            temperature=temperature,
            eta=eta,
            charge=charge,
        )

    def persistent_spin_current_vs_flux(
        self,
        flux_values: np.ndarray,
        omega_grid: np.ndarray,
        *,
        axis: str = "z",
        mu: float = 0.0,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
    ) -> np.ndarray:
        flux_values = np.asarray(flux_values, dtype=float)
        return np.array(
            [
                self.persistent_spin_current(
                    omega_grid,
                    phi_over_phi0=float(phi),
                    axis=axis,
                    mu=mu,
                    temperature=temperature,
                    eta=eta,
                    charge=charge,
                )
                for phi in flux_values
            ],
            dtype=float,
        )


    def drude_weight(
        self,
        omega_grid: np.ndarray,
        *,
        phi_over_phi0: float,
        delta_phi: float,
        mu: float = 0.0,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
    ) -> float:
        current_plus = self.persistent_current(
            omega_grid,
            phi_over_phi0=phi_over_phi0 + delta_phi,
            mu=mu,
            temperature=temperature,
            eta=eta,
            charge=charge,
        )
        current_minus = self.persistent_current(
            omega_grid,
            phi_over_phi0=phi_over_phi0 - delta_phi,
            mu=mu,
            temperature=temperature,
            eta=eta,
            charge=charge,
        )
        return finite_difference_drude_weight(current_plus, current_minus, delta_phi)


    def hartree_fock(
        self,
        *,
        n_electrons: float,
        phi_over_phi0: float = 0.0,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> CollinearHartreeResult:
        return collinear_hartree_self_consistent(
            n_sites=self.n_sites,
            n_electrons=n_electrons,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            phi_over_phi0=phi_over_phi0,
            u_hubbard=self.u_hubbard,
            onsite_up=self.onsite_up,
            onsite_down=self.onsite_down,
            mixing=mixing,
            tol=tol,
            max_iter=max_iter,
        )

    def collinear_hartree(
        self,
        *,
        n_electrons: float,
        phi_over_phi0: float = 0.0,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> CollinearHartreeResult:
        return self.hartree_fock(
            n_electrons=n_electrons,
            phi_over_phi0=phi_over_phi0,
            mixing=mixing,
            tol=tol,
            max_iter=max_iter,
        )

    def hf_hamiltonian(
        self,
        *,
        n_electrons: float,
        phi_over_phi0: float = 0.0,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> np.ndarray:
        return self.hartree_fock(
            n_electrons=n_electrons,
            phi_over_phi0=phi_over_phi0,
            mixing=mixing,
            tol=tol,
            max_iter=max_iter,
        ).hamiltonian

    def persistent_current_hf(
        self,
        omega_grid: np.ndarray,
        *,
        n_electrons: float,
        phi_over_phi0: float,
        mu: float | None = None,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> float:
        result = self.hartree_fock(
            n_electrons=n_electrons,
            phi_over_phi0=phi_over_phi0,
            mixing=mixing,
            tol=tol,
            max_iter=max_iter,
        )
        effective_mu = _chemical_potential_from_hf(result) if mu is None else mu
        return persistent_current_from_hamiltonian(
            result.hamiltonian,
            omega_grid=omega_grid,
            n_sites=self.n_sites,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            phi_over_phi0=phi_over_phi0,
            mu=effective_mu,
            temperature=temperature,
            eta=eta,
            charge=charge,
        )

    def persistent_current_vs_flux_hf(
        self,
        flux_values: np.ndarray,
        omega_grid: np.ndarray,
        *,
        n_electrons: float,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> np.ndarray:
        flux_values = np.asarray(flux_values, dtype=float)
        return np.array(
            [
                self.persistent_current_hf(
                    omega_grid,
                    n_electrons=n_electrons,
                    phi_over_phi0=float(phi),
                    temperature=temperature,
                    eta=eta,
                    charge=charge,
                    mixing=mixing,
                    tol=tol,
                    max_iter=max_iter,
                )
                for phi in flux_values
            ],
            dtype=float,
        )

    def persistent_spin_resolved_current_hf(
        self,
        omega_grid: np.ndarray,
        *,
        n_electrons: float,
        phi_over_phi0: float,
        axis: str = "z",
        component: str | int | float = "+",
        mu: float | None = None,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> float:
        result = self.hartree_fock(
            n_electrons=n_electrons,
            phi_over_phi0=phi_over_phi0,
            mixing=mixing,
            tol=tol,
            max_iter=max_iter,
        )
        effective_mu = _chemical_potential_from_hf(result) if mu is None else mu
        return persistent_spin_resolved_current_from_hamiltonian(
            result.hamiltonian,
            omega_grid=omega_grid,
            n_sites=self.n_sites,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            phi_over_phi0=phi_over_phi0,
            axis=axis,
            component=component,
            mu=effective_mu,
            temperature=temperature,
            eta=eta,
            charge=charge,
        )

    def persistent_spin_current_hf(
        self,
        omega_grid: np.ndarray,
        *,
        n_electrons: float,
        phi_over_phi0: float,
        axis: str = "z",
        mu: float | None = None,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> float:
        result = self.hartree_fock(
            n_electrons=n_electrons,
            phi_over_phi0=phi_over_phi0,
            mixing=mixing,
            tol=tol,
            max_iter=max_iter,
        )
        effective_mu = _chemical_potential_from_hf(result) if mu is None else mu
        return persistent_spin_current_from_hamiltonian(
            result.hamiltonian,
            omega_grid=omega_grid,
            n_sites=self.n_sites,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            phi_over_phi0=phi_over_phi0,
            axis=axis,
            mu=effective_mu,
            temperature=temperature,
            eta=eta,
            charge=charge,
        )

    def persistent_spin_current_vs_flux_hf(
        self,
        flux_values: np.ndarray,
        omega_grid: np.ndarray,
        *,
        n_electrons: float,
        axis: str = "z",
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> np.ndarray:
        flux_values = np.asarray(flux_values, dtype=float)
        return np.array(
            [
                self.persistent_spin_current_hf(
                    omega_grid,
                    n_electrons=n_electrons,
                    phi_over_phi0=float(phi),
                    axis=axis,
                    temperature=temperature,
                    eta=eta,
                    charge=charge,
                    mixing=mixing,
                    tol=tol,
                    max_iter=max_iter,
                )
                for phi in flux_values
            ],
            dtype=float,
        )


    def drude_weight_hf(
        self,
        omega_grid: np.ndarray,
        *,
        n_electrons: float,
        phi_over_phi0: float,
        delta_phi: float,
        temperature: float = 0.0,
        eta: float = 1e-3,
        charge: float = 1.0,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> float:
        current_plus = self.persistent_current_hf(
            omega_grid,
            n_electrons=n_electrons,
            phi_over_phi0=phi_over_phi0 + delta_phi,
            temperature=temperature,
            eta=eta,
            charge=charge,
            mixing=mixing,
            tol=tol,
            max_iter=max_iter,
        )
        current_minus = self.persistent_current_hf(
            omega_grid,
            n_electrons=n_electrons,
            phi_over_phi0=phi_over_phi0 - delta_phi,
            temperature=temperature,
            eta=eta,
            charge=charge,
            mixing=mixing,
            tol=tol,
            max_iter=max_iter,
        )
        return finite_difference_drude_weight(current_plus, current_minus, delta_phi)
