from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_DRUDE_MODEL,
    DEFAULT_DMC_JSON,
    DEFAULT_DMC_PDB,
    DEFAULT_MONOMER_TEMPLATE,
    DEFAULT_RAW_DIMER_PICKLE,
    INPUTS,
    extract_dmc_dimer_targets,
    generate_initial_drude_model,
    save_target_bundle,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract bottom-up DMC dimer targets and initial Drude guesses.")
    parser.add_argument("--raw-pickle", default=str(DEFAULT_RAW_DIMER_PICKLE))
    parser.add_argument("--config-key", default="conf_001_DMC_DMC")
    parser.add_argument("--batch-key", default="000")
    parser.add_argument("--target-npz", default=str(INPUTS / "targets" / "dmc_dimer_batch000_targets.npz"))
    parser.add_argument("--target-csv", default=str(INPUTS / "targets" / "dmc_dimer_batch000_targets.csv"))
    parser.add_argument("--model-json", default=str(DEFAULT_DRUDE_MODEL))
    parser.add_argument("--dmc-json", default=str(DEFAULT_DMC_JSON))
    parser.add_argument("--dmc-pdb", default=str(DEFAULT_DMC_PDB))
    parser.add_argument("--monomer-template", default=str(DEFAULT_MONOMER_TEMPLATE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = extract_dmc_dimer_targets(Path(args.raw_pickle), args.config_key, args.batch_key)
    save_target_bundle(targets, Path(args.target_npz), Path(args.target_csv))
    generate_initial_drude_model(Path(args.dmc_json), Path(args.dmc_pdb), Path(args.model_json))
    write_json(
        Path(args.monomer_template),
        {
            "description": "Fill this with QM monomer dipole/polarizability targets when available. This file is not used as a fitting target by default.",
            "units": {
                "dipole_debye": "debye",
                "polarizability_tensor_nm3": "nm^3",
            },
            "dipole_debye": None,
            "polarizability_tensor_nm3": None,
            "notes": [
                "Bottom-up workflow defaults to SAPT DMC-DMC dimer lr_es+lr_pol targets.",
                "Use this file only when you have explicit QM monomer response targets.",
            ],
        },
    )
    print(f"Saved dimer target bundle to {Path(args.target_npz).resolve()}")
    print(f"Saved initial Drude model to {Path(args.model_json).resolve()}")


if __name__ == "__main__":
    main()
