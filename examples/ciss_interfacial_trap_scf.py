from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantum_transport import LeadSelfEnergy, MatrixDevice

from ciss_seed_transport import IDENTITY_2, SIGMA_X, SIGMA_Y, SIGMA_Z


@dataclass(frozen=True)
class TrapModelParameters:
    n_sites: int = 6
    hopping: float = 1.0
    lambda_soc: float = 0.14
    pitch_tilt: float = 0.25
    trap_chain_coupling: float = 0.45
    eps_trap: float = 0.05
    u_trap: float = 0.0
    reference_occupation: float = 1.0
    gamma_right: float = 0.5
    gamma_majority: float = 0.75
    gamma_minority: float = 0.18
    mu_left: float = 0.25
    mu_right: float = -0.15
    temperature: float = 0.0


@dataclass(frozen=True)
class ScfResult:
    chirality: int
    magnetization: str
    u_trap: float
    current: float
    spin_current_z: float
    n_trap: float
    sx_trap: float
    sy_trap: float
    sz_trap: float
    eps_trap_eff: float
    iterations: int
    converged: bool


def orbital_from_label(label: str) -> str:
    if label.endswith("_up"):
        return label[:-3]
    if label.endswith("_down"):
        return label[:-5]
    raise ValueError(f"Expected spinful label ending in _up or _down, got {label!r}.")


def spinful_labels(n_sites: int) -> list[str]:
    labels = ["trap_up", "trap_down"]
    labels.extend(f"site{site}_{spin}" for site in range(n_sites) for spin in ("up", "down"))
    return labels


def unique_orbitals(labels: list[str]) -> list[str]:
    orbitals: list[str] = []
    for label in labels:
        orbital = orbital_from_label(label)
        if orbital not in orbitals:
            orbitals.append(orbital)
    return orbitals


def orbital_coupling_dict(labels: list[str], target: str, value: float) -> dict[str, float]:
    return {orbital: (value if orbital == target else 0.0) for orbital in unique_orbitals(labels)}


def normal_contact_gamma(labels: list[str], target: str, gamma: float) -> np.ndarray:
    out = np.zeros((len(labels), len(labels)), dtype=np.complex128)
    for index, label in enumerate(labels):
        if orbital_from_label(label) == target:
            out[index, index] = gamma
    return out


def site_block(site: int) -> slice:
    start = 2 + 2 * site
    return slice(start, start + 2)


def build_interfacial_trap_device(
    params: TrapModelParameters,
    *,
    chirality: int,
    n_trap: float,
) -> MatrixDevice:
    labels = spinful_labels(params.n_sites)
    dim = len(labels)
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    eps_eff = params.eps_trap + params.u_trap * (n_trap - params.reference_occupation)
    trap = slice(0, 2)
    hamiltonian[trap, trap] = eps_eff * IDENTITY_2

    first_site = site_block(0)
    trap_hop = params.trap_chain_coupling * IDENTITY_2
    hamiltonian[first_site, trap] = trap_hop
    hamiltonian[trap, first_site] = trap_hop.conj().T

    for site in range(params.n_sites - 1):
        phi = chirality * 2.0 * np.pi * site / 3.0
        axis = np.cos(phi) * SIGMA_X + np.sin(phi) * SIGMA_Y + params.pitch_tilt * SIGMA_Z
        hop_block = params.hopping * IDENTITY_2 + 1.0j * params.lambda_soc * axis
        left = site_block(site)
        right = site_block(site + 1)
        hamiltonian[right, left] = hop_block
        hamiltonian[left, right] = hop_block.conj().T

    return MatrixDevice(hamiltonian, labels, name=f"trap_chiral_chain_chi_{chirality:+d}")


def lead_pair(
    device: MatrixDevice,
    params: TrapModelParameters,
    *,
    magnetization_theta: float,
    magnetization_phi: float,
) -> tuple[LeadSelfEnergy, LeadSelfEnergy]:
    gamma_majority = orbital_coupling_dict(device.basis_labels, "trap", params.gamma_majority)
    gamma_minority = orbital_coupling_dict(device.basis_labels, "trap", params.gamma_minority)
    left = LeadSelfEnergy.ferromagnetic_wide_band(
        device.basis_labels,
        gamma_majority=gamma_majority,
        gamma_minority=gamma_minority,
        theta=magnetization_theta,
        phi=magnetization_phi,
        mu=params.mu_left,
        temperature=params.temperature,
        name="L_fm",
    )
    right = LeadSelfEnergy.wide_band(
        normal_contact_gamma(device.basis_labels, f"site{params.n_sites - 1}", params.gamma_right),
        mu=params.mu_right,
        temperature=params.temperature,
        name="R_normal",
    )
    return left, right


def local_density_matrix(view, omega_grid: np.ndarray, block: slice) -> np.ndarray:
    values = np.array([view.lesser(float(omega))[block, block] for omega in omega_grid], dtype=np.complex128)
    rho = -1.0j * np.trapezoid(values, omega_grid, axis=0) / (2.0 * np.pi)
    return 0.5 * (rho + rho.conj().T)


def spin_moments(rho: np.ndarray) -> tuple[float, float, float]:
    sx = 0.5 * np.trace(SIGMA_X @ rho)
    sy = 0.5 * np.trace(SIGMA_Y @ rho)
    sz = 0.5 * np.trace(SIGMA_Z @ rho)
    return float(np.real(sx)), float(np.real(sy)), float(np.real(sz))


def solve_trap_scf(
    params: TrapModelParameters,
    *,
    chirality: int,
    magnetization: str,
    magnetization_theta: float,
    magnetization_phi: float,
    omega_grid: np.ndarray,
    mix: float = 0.25,
    tolerance: float = 1e-5,
    max_iterations: int = 80,
) -> ScfResult:
    n_trap = params.reference_occupation
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        device = build_interfacial_trap_device(params, chirality=chirality, n_trap=n_trap)
        left, right = lead_pair(
            device,
            params,
            magnetization_theta=magnetization_theta,
            magnetization_phi=magnetization_phi,
        )
        view = device.transport(left, right).keldysh_view()
        rho_trap = local_density_matrix(view, omega_grid, slice(0, 2))
        raw_n = float(np.real(np.trace(rho_trap)))
        if abs(params.u_trap) < 1e-15:
            n_trap = raw_n
            converged = True
            break
        mixed_n = (1.0 - mix) * n_trap + mix * raw_n
        if abs(mixed_n - n_trap) < tolerance:
            n_trap = mixed_n
            converged = True
            break
        n_trap = mixed_n

    device = build_interfacial_trap_device(params, chirality=chirality, n_trap=n_trap)
    left, right = lead_pair(
        device,
        params,
        magnetization_theta=magnetization_theta,
        magnetization_phi=magnetization_phi,
    )
    view = device.transport(left, right).keldysh_view()
    rho_trap = local_density_matrix(view, omega_grid, slice(0, 2))
    sx, sy, sz = spin_moments(rho_trap)
    current = view.meir_wingreen_current(omega_grid, lead="left")
    spin_current_z = view.meir_wingreen_spin_current(omega_grid, lead="left", axis="z")
    eps_eff = params.eps_trap + params.u_trap * (n_trap - params.reference_occupation)

    return ScfResult(
        chirality=chirality,
        magnetization=magnetization,
        u_trap=params.u_trap,
        current=current,
        spin_current_z=spin_current_z,
        n_trap=float(np.real(np.trace(rho_trap))),
        sx_trap=sx,
        sy_trap=sy,
        sz_trap=sz,
        eps_trap_eff=eps_eff,
        iterations=iterations,
        converged=converged,
    )


def asymmetry(current_plus: float, current_minus: float) -> float:
    denom = abs(current_plus) + abs(current_minus)
    if denom < 1e-15:
        return 0.0
    return float((current_plus - current_minus) / denom)


def run_scan() -> None:
    omega_grid = np.linspace(-4.0, 4.0, 401)
    u_values = (0.0, 2.5, 4.5)
    magnetizations = {
        "+z": (0.0, 0.0),
        "-z": (np.pi, 0.0),
        "+x": (0.5 * np.pi, 0.0),
        "-x": (0.5 * np.pi, np.pi),
    }

    print("CISS interfacial trap SCF experiment using QuantumTransportEOM")
    print("Toy model: FM lead -> Hartree trap -> chiral SOC chain -> normal lead")
    print("Occupation closure: n_trap = -i int dE Tr[G^<_trap(E)] / 2pi")
    print()
    print("chi  Utrap  M    I_charge        I_spin_z        n_trap    sx_trap    sy_trap    sz_trap    eps_eff   it conv")

    rows: dict[tuple[int, float, str], ScfResult] = {}
    for chirality in (+1, -1):
        for u_trap in u_values:
            params = TrapModelParameters(u_trap=u_trap)
            for label, (theta, phi) in magnetizations.items():
                result = solve_trap_scf(
                    params,
                    chirality=chirality,
                    magnetization=label,
                    magnetization_theta=theta,
                    magnetization_phi=phi,
                    omega_grid=omega_grid,
                )
                rows[(chirality, u_trap, label)] = result
                print(
                    f"{chirality:+d}  {u_trap:5.2f}  {label:>2s}  "
                    f"{result.current:+.8e}  {result.spin_current_z:+.8e}  "
                    f"{result.n_trap:8.5f}  {result.sx_trap:+.3e}  {result.sy_trap:+.3e}  "
                    f"{result.sz_trap:+.3e}  {result.eps_trap_eff:+.4f}  "
                    f"{result.iterations:2d}  {str(result.converged):>5s}"
                )

    print()
    print("Magnetization-reversal current asymmetry A = (I_M - I_-M) / (|I_M| + |I_-M|)")
    print("chi  Utrap   axis     A_current")
    for chirality in (+1, -1):
        for u_trap in u_values:
            iz_plus = rows[(chirality, u_trap, "+z")].current
            iz_minus = rows[(chirality, u_trap, "-z")].current
            ix_plus = rows[(chirality, u_trap, "+x")].current
            ix_minus = rows[(chirality, u_trap, "-x")].current
            print(f"{chirality:+d}  {u_trap:5.2f}    z    {asymmetry(iz_plus, iz_minus):+.8e}")
            print(f"{chirality:+d}  {u_trap:5.2f}    x    {asymmetry(ix_plus, ix_minus):+.8e}")


if __name__ == "__main__":
    run_scan()
