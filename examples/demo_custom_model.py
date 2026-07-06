"""Custom Hamiltonians end-to-end: symbolic EOM, mean-field closure, and open transport.

Build any second-quantized Hamiltonian with the operator constructors
``f``/``fd`` (fermions) and ``b``/``bd`` (bosons); ``CustomModel`` detects the
statistics and mode content automatically.
"""

import numpy as np
import sympy as sp

from quantum_transport import CustomModel, n, f, fd, b, bd


def interacting_dot() -> None:
    print("=== Custom Anderson dot: eps (n_up + n_down) + U n_up n_down ===")
    eps, u = sp.symbols("epsilon U", real=True)
    model = CustomModel(eps * (n("up") + n("down")) + u * n("up") * n("down"), name="my_dot")

    print("H =", model.latex_hamiltonian())

    # Untruncated EOM: the interaction generates cubic strings (open hierarchy).
    result = model.model.eom()
    print("closed without truncation?", result.is_closed)

    # One expansion step reaches the atomic-limit closure (4x4 hierarchy).
    expanded = model.model.eom(auto_expand_steps=1)
    print("closed on expanded basis?", expanded.is_closed, "| dim:", expanded.eom_matrix.shape)

    # Hartree mean field closes on the seed basis with <n> parameters.
    hartree = model.model.eom(truncation="hartree")
    print("Hartree matrix:", hartree.eom_matrix)

    omega, eta = sp.symbols("omega eta", positive=True)
    green = model.gf("c_up").retarded(omega=omega, eta=eta, method="hartree")
    print("G^r_up (Hartree):", sp.simplify(green))
    print()


def quadratic_chain_transport() -> None:
    print("=== Quadratic 3-site chain, opened into a two-terminal device ===")
    t = 1.0
    hamiltonian = sum(
        t * (fd(i) * f(i + 1) + fd(i + 1) * f(i)) for i in range(2)
    ) + sp.Float(0.0) * n(0)
    model = CustomModel(hamiltonian, name="chain3")

    matrix, modes = model.single_particle_matrix()
    print("modes:", [str(mode) for mode in modes])
    print("h =", matrix)

    # Site-resolved wide-band contacts: left lead on site 0, right lead on
    # site 2 (a scalar would couple every site; full matrices and
    # LeadSelfEnergy objects are also accepted).
    view = model.open({"0": 0.5}, {"2": 0.5})
    grid = np.linspace(-3.0, 3.0, 401)
    transmission = view.transmission_values(grid, eta=1e-6)
    print(f"T(omega=0) = {transmission[200]:.4f}   max T = {transmission.max():.4f} (single channel: T <= 1)")
    print()


def mixed_fermion_boson() -> None:
    print("=== Mixed model: level coupled to a phonon (Holstein-like) ===")
    eps, omega0, g = sp.symbols("epsilon omega_0 g", real=True)
    model = CustomModel(eps * n(0) + omega0 * bd(0) * b(0) + g * n(0) * (b(0) + bd(0)))
    print("statistics:", model.model.statistics)
    print("operators:", sorted(model.operators))
    result = model.model.eom()
    print("closed?", result.is_closed, "(phonon coupling opens the hierarchy, as expected)")


if __name__ == "__main__":
    interacting_dot()
    quadratic_chain_transport()
    mixed_fermion_boson()
