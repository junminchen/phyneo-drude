from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import openmm.app as app

from common import (
    DEFAULT_DMC_DIMER_PDB,
    DEFAULT_DIMER_TARGETS,
    DEFAULT_DRUDE_MODEL,
    DEFAULT_DMC_PDB,
    OUTPUT,
    create_drude_espol_system,
    create_scf_context,
    load_drude_model,
    make_positions_with_drudes,
    relax_drude_positions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DMC-DMC Drude espol scan against bottom-up targets.")
    parser.add_argument("--param-file", default=str(DEFAULT_DRUDE_MODEL))
    parser.add_argument("--target-npz", default=str(DEFAULT_DIMER_TARGETS))
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--target-mode", choices=["lr_espol", "total_espol"], default="lr_espol")
    parser.add_argument("--output-dir", default=str(OUTPUT / "dimer_scan"))
    parser.add_argument("--prefix", default="dmc_dimer_drude")
    parser.add_argument("--monomer-pdb", default=str(DEFAULT_DMC_PDB))
    return parser.parse_args()


def interaction_curve(
    monomer_topology,
    dimer_topology,
    dimer_positions_a_ang: np.ndarray,
    dimer_positions_b_ang: np.ndarray,
    shifts_ang: np.ndarray,
    model: dict,
    platform: str,
    precision: str,
) -> np.ndarray:
    monomer_system, mono_meta = create_drude_espol_system(monomer_topology, model)
    dimer_system, dimer_meta = create_drude_espol_system(dimer_topology, model)

    mono_integrator_a, mono_context_a = create_scf_context(monomer_system, platform, precision)
    mono_integrator_b, mono_context_b = create_scf_context(monomer_system, platform, precision)
    dimer_integrator, dimer_context = create_scf_context(dimer_system, platform, precision)
    try:
        values = np.full(len(shifts_ang), np.nan, dtype=float)
        order = np.argsort(shifts_ang)[::-1]
        for frame_index in order:
            pos_a_ang = dimer_positions_a_ang[frame_index]
            pos_b_ang = dimer_positions_b_ang[frame_index]
            pos_a_nm = pos_a_ang * 0.1
            pos_b_nm = pos_b_ang * 0.1
            pos_ab_nm = np.vstack([pos_a_nm, pos_b_nm])
            energy_ab, _ = relax_drude_positions(
                dimer_context,
                dimer_integrator,
                make_positions_with_drudes(pos_ab_nm, dimer_meta["drude_system_indices"]),
            )
            energy_a, _ = relax_drude_positions(
                mono_context_a,
                mono_integrator_a,
                make_positions_with_drudes(pos_a_nm, mono_meta["drude_system_indices"]),
            )
            energy_b, _ = relax_drude_positions(
                mono_context_b,
                mono_integrator_b,
                make_positions_with_drudes(pos_b_nm, mono_meta["drude_system_indices"]),
            )
            values[frame_index] = energy_ab - energy_a - energy_b
        return values
    finally:
        del dimer_context, dimer_integrator
        del mono_context_a, mono_integrator_a
        del mono_context_b, mono_integrator_b


def main() -> None:
    global args
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = dict(np.load(args.target_npz))
    model = load_drude_model(Path(args.param_file))
    monomer_pdb = app.PDBFile(args.monomer_pdb)
    dimer_pdb = app.PDBFile(str(DEFAULT_DMC_DIMER_PDB))

    predicted = interaction_curve(
        monomer_pdb.topology,
        dimer_pdb.topology,
        targets["posA_angstrom"],
        targets["posB_angstrom"],
        targets["shift_angstrom"],
        model,
        args.platform,
        args.precision,
    )

    if args.target_mode == "lr_espol":
        target = targets["target_lr_espol_kj_mol"]
    else:
        target = targets["target_total_espol_kj_mol"]
    error = predicted - target
    finite_mask = np.isfinite(predicted)
    rmse = float(np.sqrt(np.mean(error[finite_mask] ** 2))) if np.any(finite_mask) else float("nan")
    mae = float(np.mean(np.abs(error[finite_mask]))) if np.any(finite_mask) else float("nan")
    max_abs = float(np.max(np.abs(error[finite_mask]))) if np.any(finite_mask) else float("nan")

    csv_path = output_dir / f"{args.prefix}_{args.target_mode}.csv"
    with open(csv_path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shift_angstrom", "target_kj_mol", "predicted_kj_mol", "error_kj_mol"])
        for shift, tgt, pred, err in zip(targets["shift_angstrom"], target, predicted, error):
            writer.writerow([f"{shift:.8f}", f"{tgt:.8f}", f"{pred:.8f}", f"{err:.8f}"])

    write_json(
        output_dir / f"{args.prefix}_{args.target_mode}_summary.json",
        {
            "param_file": str(Path(args.param_file).resolve()),
            "target_file": str(Path(args.target_npz).resolve()),
            "target_mode": args.target_mode,
            "platform": args.platform,
            "rmse_kj_mol": rmse,
            "mae_kj_mol": mae,
            "max_abs_kj_mol": max_abs,
            "num_points": int(len(predicted)),
            "num_finite_points": int(np.count_nonzero(finite_mask)),
        },
    )
    print(f"{args.target_mode} RMSE: {rmse:.6f} kJ/mol")
    print(f"{args.target_mode} MAE:  {mae:.6f} kJ/mol")


if __name__ == "__main__":
    main()
