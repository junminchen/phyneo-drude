from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit

from common import (
    DEFAULT_DMC_PDB,
    DEFAULT_DRUDE_MODEL,
    OUTPUT,
    evaluate_monomer_response,
    load_drude_model,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DMC monomer dipole and finite-field Drude response.")
    parser.add_argument("--pdb", default=str(DEFAULT_DMC_PDB))
    parser.add_argument("--param-file", default=str(DEFAULT_DRUDE_MODEL))
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--field-strength", type=float, default=1.0, help="Internal OpenMM field strength in kJ/mol/(e*nm).")
    parser.add_argument("--output-dir", default=str(OUTPUT / "monomer_response"))
    parser.add_argument("--prefix", default="dmc_monomer_drude")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb = app.PDBFile(args.pdb)
    model = load_drude_model(Path(args.param_file))
    real_positions_nm = pdb.positions.value_in_unit(unit.nanometer)
    response = evaluate_monomer_response(
        pdb.topology,
        np.asarray(real_positions_nm, dtype=float),
        model,
        args.platform,
        args.precision,
        field_strength_internal=args.field_strength,
    )
    polarizability = response["polarizability_tensor_nm3"]
    zero_dipole_debye = response["dipole_debye"]
    zero_dipole_enm = response["dipole_e_nm"]

    write_json(
        output_dir / f"{args.prefix}_summary.json",
        {
            "pdb": str(Path(args.pdb).resolve()),
            "param_file": str(Path(args.param_file).resolve()),
            "platform": args.platform,
            "field_strength_internal": args.field_strength,
            "dipole_e_nm": zero_dipole_enm.tolist(),
            "dipole_debye": zero_dipole_debye.tolist(),
            "polarizability_tensor_nm3": polarizability.tolist(),
            "polarizability_tensor_angstrom3": (polarizability * 1000.0).tolist(),
            "isotropic_polarizability_nm3": float(np.trace(polarizability) / 3.0),
            "isotropic_polarizability_angstrom3": float(np.trace(polarizability) / 3.0 * 1000.0),
        },
    )
    print(f"Dipole (Debye): {zero_dipole_debye.tolist()}")
    print(f"Isotropic alpha (A^3): {float(np.trace(polarizability) / 3.0 * 1000.0):.6f}")


if __name__ == "__main__":
    main()
