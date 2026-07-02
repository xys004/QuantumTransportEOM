from __future__ import annotations

import runpy
from pathlib import Path


GUIDE_EXAMPLES = [
    "keldysh_guide_01_contour.py",
    "keldysh_guide_02_langreth.py",
    "keldysh_guide_03_quantum_dot.py",
    "keldysh_guide_04_stationary_solution.py",
    "keldysh_guide_05_wide_band_two_terminal.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for filename in GUIDE_EXAMPLES:
        print("\n" + "=" * 78)
        print(filename)
        print("=" * 78)
        runpy.run_path(str(root / filename), run_name="__main__")


if __name__ == "__main__":
    main()
