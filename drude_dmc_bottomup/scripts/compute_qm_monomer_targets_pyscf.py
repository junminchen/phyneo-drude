from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from openmm import unit
from openmm.app import PDBFile
from pyscf import dft, gto

from common import DEFAULT_DMC_PDB, DEFAULT_MONOMER_TEMPLATE, write_json


BOHR3_TO_NM3 = 0.0529177210903**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute QM DMC monomer dipole/polarizability targets with PySCF.")
    parser.add_argument("--pdb", default=str(DEFAULT_DMC_PDB))
    parser.add_argument("--method", default="PBE0")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--field-au", type=float, default=5.0e-4)
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--output", default=str(DEFAULT_MONOMER_TEMPLATE))
    return parser.parse_args()


def build_molecule(pdb_path: Path, basis: str) -> gto.Mole:
    pdb = PDBFile(str(pdb_path))
    atoms = []
    for atom, pos in zip(pdb.topology.atoms(), pdb.positions):
        x, y, z = pos.value_in_unit(unit.angstrom)
        atoms.append(f"{atom.element.symbol} {x:.10f} {y:.10f} {z:.10f}")
    return gto.M(atom="; ".join(atoms), basis=basis, charge=0, spin=0, unit="Angstrom", verbose=0)


def make_rks(mol: gto.Mole, method: str, grid_level: int, conv_tol: float):
    mf = dft.RKS(mol)
    mf.xc = method
    mf.grids.level = grid_level
    mf.conv_tol = conv_tol
    return mf


def main() -> None:
    args = parse_args()
    mol = build_molecule(Path(args.pdb), args.basis)
    mf = make_rks(mol, args.method, args.grid_level, args.conv_tol)
    mf.kernel()
    dipole_debye = np.asarray(mf.dip_moment(unit="Debye", verbose=0), dtype=float)

    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    hcore0 = mf.get_hcore()
    dm0 = mf.make_rdm1()

    def dipole_under_field(field_vec_au: np.ndarray) -> np.ndarray:
        mff = make_rks(mol, args.method, args.grid_level, args.conv_tol)
        field_term = np.einsum("x,xij->ij", field_vec_au, ao_dip)
        mff.get_hcore = lambda *unused, h=hcore0, v=field_term: h + v
        mff.kernel(dm0=dm0)
        return np.asarray(mff.dip_moment(unit="AU", verbose=0), dtype=float)

    polar_au = np.zeros((3, 3), dtype=float)
    for axis in range(3):
        field = np.zeros(3, dtype=float)
        field[axis] = args.field_au
        mu_plus = dipole_under_field(field)
        mu_minus = dipole_under_field(-field)
        polar_au[:, axis] = (mu_plus - mu_minus) / (2.0 * args.field_au)

    polar_nm3 = polar_au * BOHR3_TO_NM3
    payload = {
        "description": "QM monomer dipole/polarizability target generated with PySCF.",
        "source": {
            "program": "PySCF",
            "version": __import__("pyscf").__version__,
            "method": args.method,
            "basis": args.basis,
            "field_au": args.field_au,
            "grid_level": args.grid_level,
            "conv_tol": args.conv_tol,
            "pdb": str(Path(args.pdb).resolve()),
        },
        "units": {
            "dipole_debye": "debye",
            "polarizability_tensor_nm3": "nm^3",
        },
        "dipole_debye": dipole_debye.tolist(),
        "polarizability_tensor_nm3": polar_nm3.tolist(),
        "polarizability_tensor_angstrom3": (polar_nm3 * 1000.0).tolist(),
        "isotropic_polarizability_nm3": float(np.trace(polar_nm3) / 3.0),
        "isotropic_polarizability_angstrom3": float(np.trace(polar_nm3) / 3.0 * 1000.0),
        "notes": [
            "Finite-field polarizability tensor from central differences of the SCF dipole.",
            "This target can be used together with DMC-DMC dimer lr_es+lr_pol in joint bottom-up fitting.",
        ],
    }
    write_json(Path(args.output), payload)
    print(f"Dipole (Debye): {dipole_debye.tolist()}")
    print(f"Isotropic alpha (A^3): {payload['isotropic_polarizability_angstrom3']:.6f}")
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
