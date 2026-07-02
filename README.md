# Quantum Transport Package

Beta release: `0.1.0b1`.

Standalone Python package for:

- Symbolic commutators and anticommutators.
- Fermionic second-quantization operator algebra built on SymPy.
- Equation-of-motion (EOM) closure checks.
- Real-space Rashba-Hubbard ring Hamiltonians.
- Real-space to reciprocal-space transforms.
- Retarded/advanced/lesser/greater Green functions.
- Self-consistent collinear-Hartree loop.
- Persistent-current and Drude-response helpers.

## Install

```bash
python -m pip install -e ".[test]"
```

Run from this folder.

## Quick Start

```bash
python examples/demo.py
python examples/demo_secondquant.py
python examples/keldysh_guide_01_contour.py
python examples/keldysh_guide_02_langreth.py
python examples/keldysh_guide_03_quantum_dot.py
python examples/keldysh_guide_04_stationary_solution.py
python examples/keldysh_guide_05_wide_band_two_terminal.py
python examples/keldysh_guide_run_all.py
python -m pytest
```

PowerShell automation:

```powershell
.\run_all.ps1
.\run_all.ps1 -InstallPytest -BuildManual
.\run_all.ps1 -Python "C:\path\to\.venv\Scripts\python.exe"
```

## Notes

- This package is intentionally separated from the warp-bubble code.
- It now has two complementary layers: symbolic second quantization and numerical transport workflows.
- The symbolic layer is the starting point for more complete EOM and future Keldysh extensions.
- The beta targets NumPy 2.x because numerical integration uses `numpy.trapezoid`.

## Documentation

- LaTeX manual: `docs/user_manual.tex`
- PDF manual: `docs/user_manual.pdf` (build with `.\run_all.ps1 -BuildManual`)
