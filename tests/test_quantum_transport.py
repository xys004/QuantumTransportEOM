import numpy as np
import pytest
import sympy as sp

from quantum_transport import (
    advanced_green,
    anticommutator,
    build_rashba_hubbard_ring_real_space,
    commutator,
    greater_green_equilibrium,
    hartree_fock_self_consistent,
    lesser_green_equilibrium,
    bose_einstein,
    real_to_k_space,
    retarded_green,
    split_k_blocks,
)


def test_commutator_anticommutator_basic():
    a, b = sp.symbols("a b", commutative=False)
    assert sp.expand(commutator(a, b, simplify=False) - (a * b - b * a)) == 0
    assert sp.expand(anticommutator(a, b, simplify=False) - (a * b + b * a)) == 0


def test_ring_hamiltonian_is_hermitian():
    h = build_rashba_hubbard_ring_real_space(
        n_sites=6,
        gamma=1.0,
        lambda_r=0.3,
        phi_over_phi0=0.2,
        u_hubbard=0.8,
        mean_n_up=np.full(6, 0.5),
        mean_n_down=np.full(6, 0.5),
    )
    assert np.allclose(h, h.conj().T, atol=1e-12)


def test_real_to_k_space_preserves_spectrum():
    h = build_rashba_hubbard_ring_real_space(n_sites=4, gamma=1.0, lambda_r=0.2, phi_over_phi0=0.1)
    h_k, _ = real_to_k_space(h, n_sites=4, spin_dim=2)
    e1 = np.sort(np.linalg.eigvalsh(h))
    e2 = np.sort(np.linalg.eigvalsh(h_k))
    assert np.allclose(e1, e2, atol=1e-10)

    with pytest.raises(ValueError, match="not block diagonal"):
        split_k_blocks(h_k, n_sites=4, spin_dim=2)

    blocks = split_k_blocks(h_k, n_sites=4, spin_dim=2, require_block_diagonal=False)
    assert len(blocks) == 4
    assert blocks[0].shape == (2, 2)


def test_spin_independent_ring_has_valid_k_blocks():
    h = build_rashba_hubbard_ring_real_space(n_sites=4, gamma=1.0, lambda_r=0.0, phi_over_phi0=0.1)
    h_k, _ = real_to_k_space(h, n_sites=4, spin_dim=2)
    blocks = split_k_blocks(h_k, n_sites=4, spin_dim=2)
    e_full = np.sort(np.linalg.eigvalsh(h_k))
    e_blocks = np.sort(np.concatenate([np.linalg.eigvalsh(block) for block in blocks]))
    assert np.allclose(e_full, e_blocks, atol=1e-10)


def test_green_equilibrium_relations_shapes():
    h = build_rashba_hubbard_ring_real_space(n_sites=4, gamma=1.0, lambda_r=0.0, phi_over_phi0=0.0)
    omega = np.array([-1.0, 0.0, 1.0])
    gr = retarded_green(h, omega=omega, eta=1e-3)
    ga = advanced_green(h, omega=omega, eta=1e-3)
    gl = lesser_green_equilibrium(gr, ga, omega=omega, mu=0.0, temperature=0.0)
    gg = greater_green_equilibrium(gr, ga, omega=omega, mu=0.0, temperature=0.0)
    assert gr.shape == (3, 8, 8)
    assert ga.shape == (3, 8, 8)
    assert gl.shape == (3, 8, 8)
    assert gg.shape == (3, 8, 8)


def test_bose_einstein_pole_returns_infinity_not_nan():
    value = bose_einstein(np.array([0.0]), mu=0.0, temperature=1.0)
    assert np.isposinf(value[0])


def test_hf_runs_and_returns_densities():
    res = hartree_fock_self_consistent(
        n_sites=4,
        n_electrons=4,
        gamma=1.0,
        lambda_r=0.2,
        phi_over_phi0=0.2,
        u_hubbard=0.4,
        max_iter=100,
        tol=1e-7,
    )
    assert res.hamiltonian.shape == (8, 8)
    assert res.n_up.shape == (4,)
    assert res.n_down.shape == (4,)
    assert np.all(res.n_up >= -1e-10)
    assert np.all(res.n_down >= -1e-10)
