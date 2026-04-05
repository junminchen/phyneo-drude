from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit
from scipy.optimize import minimize

from common import (
    DEFAULT_DMC_DIMER_PDB,
    DEFAULT_DMC_PDB,
    DEFAULT_DIMER_TARGETS,
    DEFAULT_DRUDE_MODEL,
    DEFAULT_MONOMER_TEMPLATE,
    OUTPUT,
    create_drude_espol_system,
    create_scf_context,
    evaluate_monomer_response,
    load_drude_model,
    make_positions_with_drudes,
    relax_drude_positions,
    scale_drude_model,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a minimal Drude espol model to DMC-DMC bottom-up targets.")
    parser.add_argument("--param-file", default=str(DEFAULT_DRUDE_MODEL))
    parser.add_argument("--target-npz", default=str(DEFAULT_DIMER_TARGETS))
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--target-mode", choices=["lr_espol", "total_espol"], default="lr_espol")
    parser.add_argument("--monomer-target-file", default=str(DEFAULT_MONOMER_TEMPLATE))
    parser.add_argument("--monomer-field-strength", type=float, default=1.0)
    parser.add_argument("--dimer-weight", type=float, default=1.0)
    parser.add_argument("--monomer-dipole-weight", type=float, default=0.5)
    parser.add_argument("--monomer-polar-weight", type=float, default=0.5)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--output-dir", default=str(OUTPUT / "fit"))
    parser.add_argument("--prefix", default="dmc_drude_fit")
    return parser.parse_args()


def evaluate_curve(topology, targets: dict[str, np.ndarray], model: dict, platform: str, precision: str) -> np.ndarray:
    monomer_system, mono_meta = create_drude_espol_system(topology, model)
    dimer_topology = app.PDBFile(str(DEFAULT_DMC_DIMER_PDB)).topology
    dimer_system, dimer_meta = create_drude_espol_system(dimer_topology, model)
    mono_integrator_a, mono_context_a = create_scf_context(monomer_system, platform, precision)
    mono_integrator_b, mono_context_b = create_scf_context(monomer_system, platform, precision)
    dimer_integrator, dimer_context = create_scf_context(dimer_system, platform, precision)
    try:
        curve = np.full(len(targets["shift_angstrom"]), np.nan, dtype=float)
        order = np.argsort(targets["shift_angstrom"])[::-1]
        for frame_index in order:
            pos_a_ang = targets["posA_angstrom"][frame_index]
            pos_b_ang = targets["posB_angstrom"][frame_index]
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
            curve[frame_index] = energy_ab - energy_a - energy_b
        return curve
    finally:
        del dimer_context, dimer_integrator
        del mono_context_a, mono_integrator_a
        del mono_context_b, mono_integrator_b


def load_monomer_targets(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    dipole = data.get("dipole_debye")
    polar = data.get("polarizability_tensor_nm3")
    if dipole is None or polar is None:
        return None
    return {
        "raw": data,
        "dipole_debye": np.asarray(dipole, dtype=float),
        "polarizability_tensor_nm3": np.asarray(polar, dtype=float),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = load_drude_model(Path(args.param_file))
    targets = dict(np.load(args.target_npz))
    topology = app.PDBFile(str(DEFAULT_DMC_PDB)).topology
    monomer_pdb = app.PDBFile(str(DEFAULT_DMC_PDB))
    monomer_positions_nm = np.asarray(monomer_pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    target = targets["target_lr_espol_kj_mol"] if args.target_mode == "lr_espol" else targets["target_total_espol_kj_mol"]
    monomer_targets = load_monomer_targets(Path(args.monomer_target_file))
    best_components: dict[str, float] = {}

    def objective(x: np.ndarray) -> float:
        nonlocal best_components
        scaled = scale_drude_model(
            base_model,
            alpha_scale=float(x[0]),
            drude_charge_scale=float(x[1]),
            thole_scale=float(x[2]),
        )
        curve = evaluate_curve(topology, targets, scaled, args.platform, args.precision)
        finite_mask = np.isfinite(curve)
        if not np.any(finite_mask):
            return 1.0e8
        error = curve[finite_mask] - target[finite_mask]
        dimer_scale = max(float(np.sqrt(np.mean(target[finite_mask] ** 2))), 1.0)
        dimer_term = float(np.sqrt(np.mean(error**2)) / dimer_scale)
        dipole_term = 0.0
        polar_term = 0.0
        if monomer_targets is not None:
            response = evaluate_monomer_response(
                topology,
                monomer_positions_nm,
                scaled,
                args.platform,
                args.precision,
                field_strength_internal=args.monomer_field_strength,
            )
            pred_mu = response["dipole_debye"]
            pred_alpha = response["polarizability_tensor_nm3"]
            mu_target = monomer_targets["dipole_debye"]
            alpha_target = monomer_targets["polarizability_tensor_nm3"]
            dipole_term = float(np.linalg.norm(pred_mu - mu_target) / max(np.linalg.norm(mu_target), 1.0e-8))
            polar_term = float(np.linalg.norm(pred_alpha - alpha_target) / max(np.linalg.norm(alpha_target), 1.0e-8))
        regularization = 0.1 * float(np.sum((x - np.array([1.0, 1.0, 1.0])) ** 2))
        value = float(
            args.dimer_weight * dimer_term
            + args.monomer_dipole_weight * dipole_term
            + args.monomer_polar_weight * polar_term
            + regularization
        )
        best_components = {
            "dimer_term": dimer_term,
            "dipole_term": dipole_term,
            "polar_term": polar_term,
            "regularization": regularization,
        }
        return value

    result = minimize(
        objective,
        x0=np.array([1.0, 1.0, 1.0], dtype=float),
        method="L-BFGS-B",
        bounds=[(0.5, 1.5), (0.5, 1.5), (0.5, 2.0)],
        options={"maxiter": args.maxiter},
    )
    fitted = scale_drude_model(
        base_model,
        alpha_scale=float(result.x[0]),
        drude_charge_scale=float(result.x[1]),
        thole_scale=float(result.x[2]),
    )
    final_curve = evaluate_curve(topology, targets, fitted, args.platform, args.precision)
    finite_mask = np.isfinite(final_curve)
    final_dimer_rmse = float(np.sqrt(np.mean((final_curve[finite_mask] - target[finite_mask]) ** 2))) if np.any(finite_mask) else float("nan")
    monomer_summary = None
    if monomer_targets is not None:
        response = evaluate_monomer_response(
            topology,
            monomer_positions_nm,
            fitted,
            args.platform,
            args.precision,
            field_strength_internal=args.monomer_field_strength,
        )
        monomer_summary = {
            "target_file": str(Path(args.monomer_target_file).resolve()),
            "predicted_dipole_debye": response["dipole_debye"].tolist(),
            "target_dipole_debye": monomer_targets["dipole_debye"].tolist(),
            "predicted_polarizability_tensor_nm3": response["polarizability_tensor_nm3"].tolist(),
            "target_polarizability_tensor_nm3": monomer_targets["polarizability_tensor_nm3"].tolist(),
        }
    fitted_path = output_dir / f"{args.prefix}_{args.target_mode}_model.json"
    write_json(fitted_path, fitted)
    write_json(
        output_dir / f"{args.prefix}_{args.target_mode}_summary.json",
        {
            "param_file": str(Path(args.param_file).resolve()),
            "target_file": str(Path(args.target_npz).resolve()),
            "target_mode": args.target_mode,
            "platform": args.platform,
            "success": bool(result.success),
            "message": result.message,
            "fun": float(result.fun),
            "x": result.x.tolist(),
            "weights": {
                "dimer": args.dimer_weight,
                "monomer_dipole": args.monomer_dipole_weight,
                "monomer_polar": args.monomer_polar_weight,
            },
            "objective_components": best_components,
            "final_dimer_rmse_kj_mol": final_dimer_rmse,
            "num_finite_dimer_points": int(np.count_nonzero(finite_mask)),
            "monomer": monomer_summary,
            "fitted_model": str(fitted_path.resolve()),
        },
    )
    print(f"Fit success: {result.success}")
    print(f"alpha_scale={result.x[0]:.6f} drude_charge_scale={result.x[1]:.6f} thole_scale={result.x[2]:.6f}")


if __name__ == "__main__":
    main()
