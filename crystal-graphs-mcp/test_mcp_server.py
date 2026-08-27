"""
Tests for the local (non-API) half of mcp_server.py, plus the README's
reference numbers.

    python test_mcp_server.py

No API key and no network needed — every structure is built from a spacegroup
and lattice parameter offline. The Materials Project tools are not covered
here; they need a live key and are unverified. See the mcp_server.py header.

The reference-number checks exist because the README's fcc Cu value was wrong
for the project's actual cutoff, and nobody noticed until it was checked
mechanically. Reference tables rot. Test them like code.
"""

from __future__ import annotations

import sys

import numpy as np
from pymatgen.core import Lattice, Structure

from crystal_graph import CUTOFF, structure_to_graph

FAILURES: list[str] = []


def check(label: str, got, want, tol=None) -> None:
    if tol is None:
        ok = got == want
        detail = f"got {got}, want {want}"
    else:
        ok = abs(got - want) <= tol
        detail = f"got {got:.3f}, want {want:.3f} +/- {tol}"
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        FAILURES.append(label)


def structures() -> dict[str, Structure]:
    return {
        "NaCl": Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]]),
        "fcc Cu": Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(3.615), ["Cu"], [[0, 0, 0]]),
        "bcc Fe": Structure.from_spacegroup(
            "Im-3m", Lattice.cubic(2.866), ["Fe"], [[0, 0, 0]]),
        "diamond Si": Structure.from_spacegroup(
            "Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]]),
    }


def test_nacl_shells() -> None:
    print("\nNaCl coordination shells (README section 01)")
    nacl = structures()["NaCl"]
    _, _, _, dist = nacl.get_neighbor_list(4.2)

    first = dist.min()
    n_first = np.sum(np.isclose(dist, first, atol=1e-3)) / len(nacl)
    second = np.sort(np.unique(np.round(dist, 3)))[1]
    n_second = np.sum(np.isclose(dist, second, atol=1e-3)) / len(nacl)

    check("1st shell distance = a/2", float(first), 5.64 / 2, tol=1e-3)
    check("1st shell count", int(n_first), 6)
    check("2nd shell distance = a/sqrt2", float(second), 5.64 / np.sqrt(2),
          tol=1e-3)
    check("2nd shell count", int(n_second), 12)


def test_self_image_edges() -> None:
    print("\nSelf-image edges (README section 02)")
    nacl = structures()["NaCl"]
    src, dst, _, _ = nacl.get_neighbor_list(6.5)
    check("NaCl self-image edges at 6.5 A", int(np.sum(src == dst)), 48)


def test_edges_per_atom() -> None:
    print(f"\nEdges per atom at the project cutoff ({CUTOFF} A)")
    expected = {"NaCl": 26, "fcc Cu": 54, "bcc Fe": 58, "diamond Si": 28}
    for name, s in structures().items():
        g = structure_to_graph(s, target=0.0, material_id=name, cutoff=CUTOFF)
        assert g is not None, f"{name} produced no graph"
        check(name, round(g.edge_index.shape[1] / g.n_atoms), expected[name])


def test_cutoff_discontinuity() -> None:
    """The thing that made the README wrong. Guard it explicitly."""
    print("\nCutoff sensitivity — fcc Cu across the a*sqrt2 = 5.112 A shell")
    cu = structures()["fcc Cu"]
    for cutoff, want in ((5.0, 42), (5.2, 54)):
        g = structure_to_graph(cu, target=0.0, cutoff=cutoff)
        check(f"Cu at {cutoff} A", round(g.edge_index.shape[1] / g.n_atoms),
              want)


def test_graph_invariants() -> None:
    print("\nCrystalGraph structural invariants")
    for name, s in structures().items():
        g = structure_to_graph(s, target=-1.23, material_id=name)
        assert g is not None
        e = g.edge_index.shape[1]
        ok = (g.edge_index.shape[0] == 2
              and g.edge_dist.shape[0] == e
              and g.edge_offset.shape == (e, 3)
              and g.z.shape[0] == g.n_atoms
              and float(g.edge_dist.max()) <= CUTOFF + 1e-6
              and g.target == -1.23)
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: shapes, cutoff bound, target")
        if not ok:
            FAILURES.append(f"invariants/{name}")


def test_isolated_atom_returns_none() -> None:
    print("\nDegenerate input handling")
    lone = Structure(Lattice.cubic(30.0), ["He"], [[0, 0, 0]])
    got = structure_to_graph(lone, target=0.0, cutoff=5.2)
    print(f"  [{'ok' if got is None else 'FAIL'}] isolated atom in huge cell "
          f"-> None (got {type(got).__name__})")
    if got is not None:
        FAILURES.append("isolated atom should return None")


def main() -> None:
    for fn in (test_nacl_shells, test_self_image_edges, test_edges_per_atom,
               test_cutoff_discontinuity, test_graph_invariants,
               test_isolated_atom_returns_none):
        fn()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
