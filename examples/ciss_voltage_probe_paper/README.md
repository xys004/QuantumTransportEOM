# CISS voltage-probe paper — reproduction package

Scripts and data behind every figure of

> N. Bolívar, *Dephasing is not enough: energy relaxation as the minimal ingredient
> for a chirality-induced magnetocurrent* (2026).

The model is a chiral multichannel ladder with an explicit `rho ⊗ tau ⊗ sigma`
(strand ⊗ orbital-channel ⊗ physical-spin) local Hilbert space, driven between a
ferromagnetic and a normal contact, with conserving Büttiker probes on the interior
rungs. Three boundary conditions are compared: fully coherent, elastic dephasing
probes (`I_p(E) = 0` at every energy), and an inelastic voltage probe
(`∫dE I_p(E) = 0`, single self-consistent `mu_p`).

## Contents

| File | Purpose |
|------|---------|
| `generate_data_and_figures.py` | Baseline scans: probe-coupling scan, detuning map, mechanism/symmetry controls |
| `extended_scans.py` | Bias, spin-orbit, polarization, length, temperature scans; fine detuning map |
| `validation_checks.py` | Independent checks: grid convergence, pointwise reciprocity of the coherent and elastic-probe cases, linear-response limit |
| `make_figures.py` | Journal-quality figures from the CSV data |
| `data/` | All CSV outputs used in the paper |

Model builders live in `../ciss_rho_tau_sigma_ladder.py`,
`../ciss_rho_tau_sigma_voltage_probe.py`, and `../ciss_ladder_elastic_probe.py`;
the transport engine is the `quantum_transport` package of this repository.

## Reproduce

```bash
pip install -e ../..          # install quantum_transport
python generate_data_and_figures.py
python extended_scans.py
python validation_checks.py
python make_figures.py
```

Headline result: with identical Hamiltonian and contacts, the coherent and
elastically dephased devices give a magnetocurrent asymmetry `A_M = 0` as an exact
symmetry (pointwise in energy, < 1e-15 numerically), while the inelastic voltage
probe activates `A_M = -3.34e-3` at the reference point, reversing with chirality
and vanishing for `lambda = 0` or `p_FM = 0`, with scaling law
`A_M ≈ -0.9 χ (λ/t)² p_FM (eV/t)`.
