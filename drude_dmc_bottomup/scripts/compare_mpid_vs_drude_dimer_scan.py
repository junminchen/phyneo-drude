from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_phyneo_drude")

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
INPUTS = ROOT / "inputs"
OUTPUT = ROOT / "output"

DEFAULT_DRUDE_MODEL = ROOT / "output" / "fit_joint" / "dmc_drude_joint_lr_espol_model.json"
DEFAULT_TARGETS = INPUTS / "targets" / "dmc_dimer_batch000_targets.npz"
DEFAULT_MONOMER_PDB = INPUTS / "structures" / "DMC.pdb"
DEFAULT_DIMER_PDB = INPUTS / "structures" / "dimer_001_DMC_DMC.pdb"
DEFAULT_MPID_XML = INPUTS / "mpid" / "phyneo_ecl.xml"


def resolve_plugin_root() -> Path | None:
    candidates = []
    env_value = os.environ.get("OPENMM_PHYNEO_PLUGIN_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser().resolve())
    candidates.append((ROOT.parent / "openmm-phyneo-plugin").resolve())
    candidates.append((ROOT.parent / "openmm-phyneo-plugin-amoeba").resolve())
    for candidate in candidates:
        if (candidate / "python" / "phyneoforceplugin.py").exists() or (candidate / "python" / "dmff_sr_custom_forces.py").exists():
            return candidate
    return None


def add_local_plugin_paths(plugin_root: Path | None) -> None:
    if plugin_root is None:
        return
    repo_str = str(plugin_root)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    python_dir = plugin_root / "python"
    if python_dir.exists():
        sys.path.insert(0, str(python_dir))
    for build_dir in sorted(plugin_root.glob("build*/")):
        candidate = build_dir / "python"
        if (candidate / "phyneoforceplugin.py").exists():
            sys.path.insert(0, str(candidate))


PLUGIN_ROOT = resolve_plugin_root()
add_local_plugin_paths(PLUGIN_ROOT)

import phyneoforceplugin  # noqa: E402

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import load_drude_model, write_json  # noqa: E402
from run_dmc_drude_dimer_scan import interaction_curve as drude_interaction_curve  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DMC-DMC long-range espol scans from MPID and Drude models.")
    parser.add_argument("--drude-param-file", default=str(DEFAULT_DRUDE_MODEL))
    parser.add_argument("--target-npz", default=str(DEFAULT_TARGETS))
    parser.add_argument("--mpid-xml", default=str(DEFAULT_MPID_XML))
    parser.add_argument("--monomer-pdb", default=str(DEFAULT_MONOMER_PDB))
    parser.add_argument("--dimer-pdb", default=str(DEFAULT_DIMER_PDB))
    parser.add_argument("--drude-platform", default="Reference")
    parser.add_argument("--mpid-platform", default="Reference")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--output-dir", default=str(OUTPUT / "mpid_vs_drude_dimer_scan"))
    parser.add_argument("--prefix", default="dmc_dimer_lr_espol_mpid_vs_drude")
    return parser.parse_args()


def _strip_to_admp_pme(system: mm.System) -> None:
    keep = []
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if phyneoforceplugin.ADMPPmeForce.isinstance(force):
            keep.append(force_index)
    if not keep:
        raise RuntimeError("ADMPPmeForce not found in system.")
    for force_index in reversed(range(system.getNumForces())):
        if force_index not in keep:
            system.removeForce(force_index)


def _build_mpid_context(topology: app.Topology, xml_path: Path, platform_name: str):
    forcefield = app.ForceField(str(xml_path))
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=False,
        polarization="extrapolated",
    )
    _strip_to_admp_pme(system)
    integrator = mm.VerletIntegrator(1.0e-6)
    platform = mm.Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(topology, system, integrator, platform)
    return integrator, simulation.context


def _interaction_curve_mpid(
    monomer_topology: app.Topology,
    dimer_topology: app.Topology,
    dimer_positions_a_ang: np.ndarray,
    dimer_positions_b_ang: np.ndarray,
    xml_path: Path,
    platform_name: str,
) -> np.ndarray:
    mono_integrator_a, mono_context_a = _build_mpid_context(monomer_topology, xml_path, platform_name)
    mono_integrator_b, mono_context_b = _build_mpid_context(monomer_topology, xml_path, platform_name)
    dimer_integrator, dimer_context = _build_mpid_context(dimer_topology, xml_path, platform_name)
    try:
        values = np.full(len(dimer_positions_a_ang), np.nan, dtype=float)
        for frame_index, (pos_a_ang, pos_b_ang) in enumerate(zip(dimer_positions_a_ang, dimer_positions_b_ang)):
            pos_a_nm = np.asarray(pos_a_ang, dtype=np.float64) * 0.1
            pos_b_nm = np.asarray(pos_b_ang, dtype=np.float64) * 0.1
            pos_ab_nm = np.vstack([pos_a_nm, pos_b_nm])
            mono_context_a.setPositions(pos_a_nm)
            mono_context_b.setPositions(pos_b_nm)
            dimer_context.setPositions(pos_ab_nm)
            energy_a = mono_context_a.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            energy_b = mono_context_b.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            energy_ab = dimer_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            values[frame_index] = energy_ab - energy_a - energy_b
        return values
    finally:
        del dimer_context, dimer_integrator
        del mono_context_a, mono_integrator_a
        del mono_context_b, mono_integrator_b


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = predicted - target
    finite_mask = np.isfinite(predicted)
    if not np.any(finite_mask):
        return {"rmse_kj_mol": float("nan"), "mae_kj_mol": float("nan"), "max_abs_kj_mol": float("nan")}
    finite_error = error[finite_mask]
    return {
        "rmse_kj_mol": float(np.sqrt(np.mean(np.square(finite_error)))),
        "mae_kj_mol": float(np.mean(np.abs(finite_error))),
        "max_abs_kj_mol": float(np.max(np.abs(finite_error))),
    }


def _write_csv(path: Path, shifts: np.ndarray, target: np.ndarray, drude: np.ndarray, mpid: np.ndarray) -> None:
    with open(path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shift_angstrom",
                "target_lr_espol_kj_mol",
                "drude_kj_mol",
                "mpid_kj_mol",
                "drude_error_kj_mol",
                "mpid_error_kj_mol",
            ]
        )
        for row in zip(shifts, target, drude, mpid, drude - target, mpid - target):
            writer.writerow([f"{float(value):.8f}" for value in row])


def _plot_png(path: Path, shifts: np.ndarray, target: np.ndarray, drude: np.ndarray, mpid: np.ndarray) -> None:
    order = np.argsort(shifts)
    x = shifts[order]
    target_ord = target[order]
    drude_ord = drude[order]
    mpid_ord = mpid[order]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax0.plot(x, target_ord, marker="o", linewidth=1.8, label="SAPT target (lr es+pol)")
    ax0.plot(x, drude_ord, marker="s", linewidth=1.8, label="Drude")
    ax0.plot(x, mpid_ord, marker="^", linewidth=1.8, label="MPID ADMPPme")
    ax0.set_ylabel("Interaction Energy (kJ/mol)")
    ax0.set_title("DMC-DMC Dimer Long-Range es+pol Scan")
    ax0.legend()

    ax1.plot(x, drude_ord - target_ord, marker="s", linewidth=1.6, label="Drude - target")
    ax1.plot(x, mpid_ord - target_ord, marker="^", linewidth=1.6, label="MPID - target")
    ax1.axhline(0.0, color="black", linewidth=1.0)
    ax1.set_xlabel("Scan Coordinate (angstrom)")
    ax1.set_ylabel("Error (kJ/mol)")
    ax1.legend()

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_bundle = dict(np.load(args.target_npz))
    target = np.asarray(target_bundle["target_lr_espol_kj_mol"], dtype=float)
    shifts = np.asarray(target_bundle["shift_angstrom"], dtype=float)
    pos_a = np.asarray(target_bundle["posA_angstrom"], dtype=float)
    pos_b = np.asarray(target_bundle["posB_angstrom"], dtype=float)

    model = load_drude_model(Path(args.drude_param_file))
    monomer_pdb = app.PDBFile(args.monomer_pdb)
    dimer_pdb = app.PDBFile(args.dimer_pdb)

    drude = drude_interaction_curve(
        monomer_pdb.topology,
        dimer_pdb.topology,
        pos_a,
        pos_b,
        shifts,
        model,
        args.drude_platform,
        args.precision,
    )
    mpid = _interaction_curve_mpid(
        monomer_pdb.topology,
        dimer_pdb.topology,
        pos_a,
        pos_b,
        Path(args.mpid_xml),
        args.mpid_platform,
    )

    csv_path = output_dir / f"{args.prefix}.csv"
    png_path = output_dir / f"{args.prefix}.png"
    json_path = output_dir / f"{args.prefix}_summary.json"

    _write_csv(csv_path, shifts, target, drude, mpid)
    _plot_png(png_path, shifts, target, drude, mpid)

    summary = {
        "drude_param_file": str(Path(args.drude_param_file).resolve()),
        "target_file": str(Path(args.target_npz).resolve()),
        "mpid_xml": str(Path(args.mpid_xml).resolve()),
        "drude_platform": args.drude_platform,
        "mpid_platform": args.mpid_platform,
        "drude_vs_target": _metrics(drude, target),
        "mpid_vs_target": _metrics(mpid, target),
        "drude_vs_mpid": _metrics(drude, mpid),
        "num_points": int(len(shifts)),
    }
    write_json(json_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
