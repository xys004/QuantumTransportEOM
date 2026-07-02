from __future__ import annotations

import sympy as sp

from quantum_transport import keldysh_system


def main() -> None:
    omega = sp.Symbol("omega", real=True)
    k = keldysh_system(omega)
    A = k.green("A")
    B = k.green("B")
    W = k.green("W")
    X = k.green("X")
    Y = k.green("Y")

    product = k.langreth(
        {"r": A.retarded(omega), "a": A.advanced(omega), "<": A.lesser(omega), ">": A.greater(omega)},
        {"r": B.retarded(omega), "a": B.advanced(omega), "<": B.lesser(omega), ">": B.greater(omega)},
    )
    triple = k.langreth3(
        {"r": W.retarded(omega), "a": W.advanced(omega), "<": W.lesser(omega), ">": W.greater(omega)},
        {"r": X.retarded(omega), "a": X.advanced(omega), "<": X.lesser(omega), ">": X.greater(omega)},
        {"r": Y.retarded(omega), "a": Y.advanced(omega), "<": Y.lesser(omega), ">": Y.greater(omega)},
    )

    print("Langreth rules for C = A * B")
    print("C^r =")
    sp.pprint(product["r"].doit())
    print("\nC^< =")
    sp.pprint(product["<"].doit())
    print("\nC^> =")
    sp.pprint(product[">"].doit())

    print("\nLangreth rules for Z = W * X * Y")
    print("Z^< =")
    sp.pprint(triple["<"].doit())
    print("\nLaTeX Z^<:")
    print(triple["<"].latex())


if __name__ == "__main__":
    main()
