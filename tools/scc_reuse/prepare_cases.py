#!/usr/bin/env python3
"""Prepare diagnostic case.spec inputs for the issue #343 Phase-1 tooling.

Writes the corpus-style flat spec layout consumed by scc_reuse_capture:

    nat
    atomic numbers (nat)
    positions x y z in bohr (nat*3)
    molecular_charge
    unpaired_electrons
    temperature_kelvin
    mixer_memory
    mixer_damping
    maximum_iterations
    n_point_charges
    (x y z q gamma)*npc

The five conformance corpus specs in data/conformance/scc-traces/specs are
used verbatim. Additional molecules are built with ASE 3.29 builders and
converted from Angstrom to bohr; they are diagnostic inputs with no golden,
so their exact geometry is informational only. tmacl is converted from the
committed data/conformance/inputs/tmacl.xyz (Angstrom), with the policy that
issue #217 documents as nonconverging at 300 K.
"""

import argparse
from pathlib import Path

import numpy as np

BOHR = 1.8897261254578281
KELVIN_TO_HARTREE = 3.166808578545117e-6


def write_spec(
    path,
    atomic_numbers,
    positions_bohr,
    charge=0.0,
    unpaired=0,
    temperature=300.0,
    mixer_memory=8,
    mixer_damping=0.4,
    maximum_iterations=60,
    pc=None,
):
    lines = [str(len(atomic_numbers))]
    lines.append(" ".join(str(z) for z in atomic_numbers))
    for xyz in positions_bohr:
        lines.append(f"{xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g}")
    lines.append(f"{charge:.17g}")
    lines.append(str(unpaired))
    lines.append(f"{temperature:.17g}")
    lines.append(str(mixer_memory))
    lines.append(f"{mixer_damping:.17g}")
    lines.append(str(maximum_iterations))
    pc = pc or []
    lines.append(str(len(pc)))
    for row in pc:
        lines.append(
            f"{row[0]:.17g} {row[1]:.17g} {row[2]:.17g} {row[3]:.17g} {row[4]:.17g}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {path}: {len(atomic_numbers)} atoms")


def ase_molecule(name, builder):
    from ase.build import molecule

    atoms = molecule(name)
    return [int(a.number) for a in atoms], atoms.get_positions() * BOHR


def trans_planar_alkane(n_carbon):
    """Build a trans-planar (zigzag) alkane C_nH_{2n+2} with regular sp3
    geometry: C-C 1.53 A, C-H 1.09 A, tetrahedral angles. This is a diagnostic
    input only (no golden), so the unrelaxed geometry is acceptable.
    """
    from ase import Atoms

    cc = 1.53 * BOHR
    ch = 1.09 * BOHR
    alpha = np.deg2rad(109.47122063)
    pos = []
    symbols = []
    # Build the carbon zigzag in the xy plane.
    cpos = [np.array([0.0, 0.0, 0.0])]
    direction = np.array([1.0, 0.0, 0.0])
    up = True
    for i in range(1, n_carbon):
        if i % 2 == 1:
            step = (
                np.array([np.cos(np.pi - alpha), np.sin(alpha), 0.0])
                * cc
                * (1 if up else -1)
            )
            up = not up
        else:
            step = np.array([1.0, 0.0, 0.0]) * cc
        cpos.append(cpos[-1] + step)
    cpos = np.asarray(cpos)
    # Recenter and reorient along x for stable H placement.
    cpos -= cpos.mean(axis=0)
    # Place hydrogens along the bisector directions (one per carbon for
    # chain ends, adjusted per geometry).
    pos = [p.copy() for p in cpos]
    symbols = ["C"] * n_carbon
    for i in range(n_carbon):
        bonds = [
            j
            for j in range(n_carbon)
            if j != i and np.linalg.norm(cpos[i] - cpos[j]) < 2.2 * BOHR
        ]
        # Hydrogen directions: for a TC planar chain, H's lie out of plane and
        # along the in-plane bisectors.
        neighbors = [cpos[j] - cpos[i] for j in bonds]
        if len(neighbors) == 1:
            u = neighbors[0] / np.linalg.norm(neighbors[0])
            # Two in-plane H and one out-of-plane H.
            v = np.array([-u[1], u[0], 0.0])
            for s in (1.0, -1.0):
                pos.append(
                    cpos[i] + ch * (np.cos(alpha) * (-u) + np.sin(alpha) * v * s)
                )
                symbols.append("H")
            pos.append(cpos[i] + ch * np.array([0.0, 0.0, 1.0]))
            symbols.append("H")
        elif len(neighbors) == 2:
            u1 = neighbors[0] / np.linalg.norm(neighbors[0])
            u2 = neighbors[1] / np.linalg.norm(neighbors[1])
            bis = (u1 + u2) / np.linalg.norm(u1 + u2)
            u = np.array([-bis[1], bis[0], 0.0])
            for s in (1.0, -1.0):
                pos.append(cpos[i] + ch * (np.cos(alpha) * bis + np.sin(alpha) * u * s))
                symbols.append("H")
        else:  # chain-end cap path already handled; fallback for safety
            pos.append(cpos[i] + ch * np.array([0.0, 0.0, 1.0]))
            symbols.append("H")
    atoms = Atoms(symbols=symbols, positions=pos)
    return [int(a.number) for a in atoms], atoms.get_positions()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "cases"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Medium/large neutral diagnostic systems built with ASE g2 builders or
    # the deterministic trans-planar alkane generator above.
    for label, key in (("benzene", "C6H6"), ("pyridine", "C5H5N")):
        numbers, pos = ase_molecule(key, key)
        write_spec(out / f"{label}.spec", numbers, pos)
    numbers, pos = trans_planar_alkane(12)
    write_spec(out / "dodecane.spec", numbers, pos)

    # tmacl (Me4N+ / Cl- separated ion pair) from the committed XYZ fixture,
    # 300 K policy per issue #217 (known charge sloshing at this temperature).
    xyz = Path(__file__).resolve().parents[2] / "data/conformance/inputs/tmacl.xyz"
    lines = xyz.read_text().splitlines()
    nat = int(lines[0])
    numbers, pos = [], []
    for line in lines[2 : 2 + nat]:
        toks = line.split()
        numbers.append(
            int(
                {"C": 6, "H": 1, "N": 7, "O": 8, "F": 9, "Cl": 17, "S": 16, "Si": 14}[
                    toks[0]
                ]
            )
        )
        pos.append([float(toks[1]), float(toks[2]), float(toks[3])])
    write_spec(
        out / "tmacl.spec",
        numbers,
        np.asarray(pos) * BOHR,
        charge=0.0,
        unpaired=0,
        temperature=300.0,
        mixer_memory=8,
        mixer_damping=0.4,
        maximum_iterations=100,
    )

    # Two displaced geometries of dodecane for a warm-start trajectory: the
    # second is a small deterministic non-rigid perturbation (each atom moved
    # by up to 0.05 bohr in a random direction), representing a tiny MD or
    # optimization step that actually changes internal coordinates.
    numbers, pos = trans_planar_alkane(12)
    rng = np.random.default_rng(343)
    pos2 = pos + rng.normal(0.0, 0.015, size=pos.shape)
    write_spec(out / "dodecane_traj1.spec", numbers, pos, maximum_iterations=60)
    write_spec(out / "dodecane_traj2.spec", numbers, pos2, maximum_iterations=60)


if __name__ == "__main__":
    main()
