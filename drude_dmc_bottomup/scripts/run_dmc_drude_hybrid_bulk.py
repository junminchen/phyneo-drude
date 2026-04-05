from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

ROOT = Path(__file__).resolve().parents[1]


def resolve_plugin_root() -> Path:
    candidates = []
    env_value = os.environ.get("OPENMM_PHYNEO_PLUGIN_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser().resolve())
    candidates.append((ROOT.parent / "openmm-phyneo-plugin").resolve())
    candidates.append((ROOT.parent / "openmm-phyneo-plugin-amoeba").resolve())
    for candidate in candidates:
        if (candidate / "python" / "dmff_sr_custom_forces.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate openmm-phyneo-plugin. Set OPENMM_PHYNEO_PLUGIN_ROOT to a checkout "
        "that contains python/dmff_sr_custom_forces.py and openmmtool.py."
    )


PLUGIN_ROOT = resolve_plugin_root()
PLUGIN_PYTHON = PLUGIN_ROOT / "python"


def add_local_plugin_paths() -> None:
    repo_str = str(PLUGIN_ROOT)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    candidates = [PLUGIN_PYTHON]
    candidates.extend(
        build_dir / "python"
        for build_dir in sorted(PLUGIN_ROOT.glob("build*/"))
        if (build_dir / "python" / "phyneoforceplugin.py").exists()
    )
    for candidate in reversed(candidates):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


add_local_plugin_paths()

from common import (  # noqa: E402
    DEFAULT_DMC_BULK_PDB,
    DEFAULT_DRUDE_MODEL,
    OUTPUT,
    create_drude_espol_system,
    density_from_state,
    elapsed_ns_per_day,
    load_drude_model,
    make_positions_with_drudes,
    read_last_csv_row,
    resolve_platform,
    shortest_path_pairs,
    temperature_from_state,
    write_json,
)
from dmff_sr_custom_forces import add_dmff_short_range_forces_from_xml, add_undamped_dispersion_force  # noqa: E402
from openmmtool import IntraGromacsForceBuilder  # noqa: E402


DEFAULT_XML = PLUGIN_ROOT / "examples" / "dmc_bulk_compare" / "inputs" / "phyneo_ecl.xml"
DEFAULT_INTRA_PARAMS = PLUGIN_ROOT / "examples" / "dmc_bulk_compare" / "inputs" / "params_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DMC bulk MD with Drude espol + plugin Slater short-range + intra + dispersion.")
    parser.add_argument("--pdb", default=str(DEFAULT_DMC_BULK_PDB))
    parser.add_argument("--param-file", default=str(DEFAULT_DRUDE_MODEL))
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--intra-params-dir", default=str(DEFAULT_INTRA_PARAMS))
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-atm", type=float, default=1.0)
    parser.add_argument("--dt-fs", type=float, default=0.25)
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--max-drude-distance-nm", type=float, default=0.02)
    parser.add_argument("--integrator", choices=["nose-hoover", "langevin"], default="nose-hoover")
    parser.add_argument("--nvt-steps", type=int, default=1000)
    parser.add_argument("--npt-steps", type=int, default=4000)
    parser.add_argument("--barostat-frequency", type=int, default=25)
    parser.add_argument("--minimize-max-iter", type=int, default=200)
    parser.add_argument("--report-interval", type=int, default=500)
    parser.add_argument("--traj-interval", type=int, default=1000)
    parser.add_argument("--short-range-s12", type=float, default=0.169)
    parser.add_argument("--output-dir", default=str(OUTPUT / "bulk_hybrid"))
    parser.add_argument("--prefix", default="dmc_bulk_drude_hybrid")
    return parser.parse_args()


def _pad_custom_nonbonded_force(force: mm.CustomNonbondedForce, num_total_particles: int) -> None:
    missing = num_total_particles - force.getNumParticles()
    if missing < 0:
        raise ValueError("CustomNonbondedForce has more particles than the system.")
    if missing == 0:
        return
    zeros = [0.0] * force.getNumPerParticleParameters()
    for _ in range(missing):
        force.addParticle(zeros)


def _pad_new_custom_nonbonded_forces(system: mm.System, start_force_index: int) -> None:
    num_total_particles = system.getNumParticles()
    for force_index in range(start_force_index, system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, mm.CustomNonbondedForce):
            _pad_custom_nonbonded_force(force, num_total_particles)


def _collect_exclusions(force: mm.CustomNonbondedForce) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for index in range(force.getNumExclusions()):
        atom1, atom2 = force.getExclusionParticles(index)
        pairs.add(tuple(sorted((int(atom1), int(atom2)))))
    return pairs


def _find_nonbonded_force(system: mm.System) -> mm.NonbondedForce:
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            return force
    raise RuntimeError("NonbondedForce not found in Drude hybrid system.")


def _align_nonbonded_exclusions(system: mm.System, topology: app.Topology, drude_system_indices: dict[int, int]) -> None:
    nonbonded = _find_nonbonded_force(system)
    real_bonds = [(bond[0].index, bond[1].index) for bond in topology.bonds()]
    adjacency = [set() for _ in range(len(list(topology.atoms())))]
    for atom1, atom2 in real_bonds:
        adjacency[atom1].add(atom2)
        adjacency[atom2].add(atom1)

    target_pairs = set()
    shells = shortest_path_pairs(len(adjacency), real_bonds, max_distance=5)
    for shell in range(1, 6):
        target_pairs.update(shells[shell])
    for parent_index, drude_index in drude_system_indices.items():
        target_pairs.add(tuple(sorted((parent_index, drude_index))))
        for neighbor in adjacency[parent_index]:
            target_pairs.add(tuple(sorted((neighbor, drude_index))))

    for atom1, atom2 in sorted(target_pairs):
        nonbonded.addException(
            atom1,
            atom2,
            0.0 * unit.elementary_charge**2,
            0.1 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
            replace=True,
        )

    for force in system.getForces():
        if not isinstance(force, mm.CustomNonbondedForce):
            continue
        existing = _collect_exclusions(force)
        for atom1, atom2 in sorted(target_pairs - existing):
            force.addExclusion(atom1, atom2)


def add_intra_bonded_forces(system: mm.System, topology: app.Topology, params_dir: Path, box_vectors) -> None:
    builder = IntraGromacsForceBuilder(params_dir)
    builder.add_to_system(
        system,
        topology,
        box_vectors=box_vectors,
        start_group=system.getNumForces(),
    )


def force_inventory(system: mm.System) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        records.append(
            {
                "index": index,
                "name": force.getName() or type(force).__name__,
                "type": type(force).__name__,
                "group": force.getForceGroup(),
            }
        )
    return records


def build_integrator(args: argparse.Namespace):
    if args.integrator == "langevin":
        integrator = mm.DrudeLangevinIntegrator(
            args.temperature_k * unit.kelvin,
            1 / unit.picosecond,
            1 * unit.kelvin,
            20 / unit.picosecond,
            args.dt_fs * unit.femtoseconds,
        )
    else:
        integrator = mm.DrudeNoseHooverIntegrator(
            args.temperature_k * unit.kelvin,
            1 / unit.picosecond,
            1 * unit.kelvin,
            20 / unit.picosecond,
            args.dt_fs * unit.femtoseconds,
        )
    integrator.setMaxDrudeDistance(args.max_drude_distance_nm * unit.nanometer)
    return integrator


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb = app.PDBFile(args.pdb)
    num_real_atoms = sum(1 for _ in pdb.topology.atoms())
    model = load_drude_model(Path(args.param_file))
    xml_path = Path(args.xml)
    intra_params_dir = Path(args.intra_params_dir)

    system, metadata = create_drude_espol_system(
        pdb.topology,
        model,
        nonbonded_method="PME",
        cutoff_nm=args.cutoff_nm,
        include_bonded=False,
    )
    vectors = pdb.topology.getPeriodicBoxVectors()
    if vectors is None:
        raise ValueError("Periodic bulk box vectors are required.")
    system.setDefaultPeriodicBoxVectors(*vectors)

    start_force = system.getNumForces()
    add_dmff_short_range_forces_from_xml(
        system,
        pdb.topology,
        str(xml_path),
        start_group=start_force,
        s12=args.short_range_s12,
    )
    _pad_new_custom_nonbonded_forces(system, start_force)

    start_force = system.getNumForces()
    add_undamped_dispersion_force(
        system,
        pdb.topology,
        str(xml_path),
        cutoff_nm=args.cutoff_nm,
        force_group=start_force,
    )
    _pad_new_custom_nonbonded_forces(system, start_force)

    add_intra_bonded_forces(system, pdb.topology, intra_params_dir, vectors)
    _align_nonbonded_exclusions(system, pdb.topology, metadata["drude_system_indices"])
    system.addForce(mm.CMMotionRemover())
    system.addForce(
        mm.MonteCarloBarostat(
            args.pressure_atm * unit.atmospheres,
            args.temperature_k * unit.kelvin,
            args.barostat_frequency,
        )
    )

    for index in range(system.getNumForces()):
        system.getForce(index).setForceGroup(min(index, 31))

    integrator = build_integrator(args)
    platform, properties = resolve_platform(args.platform, args.precision)
    if platform is None:
        simulation = app.Simulation(pdb.topology, system, integrator)
    else:
        simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)

    simulation.context.setPeriodicBoxVectors(*vectors)
    real_positions_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    simulation.context.setPositions(make_positions_with_drudes(real_positions_nm, metadata["drude_system_indices"]))

    initial_state = simulation.context.getState(getEnergy=True)
    initial_energy = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    mm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=args.minimize_max_iter)
    simulation.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin)
    if args.nvt_steps > 0:
        simulation.step(args.nvt_steps)

    state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    nvt_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    npt_integrator = build_integrator(args)
    if platform is None:
        simulation = app.Simulation(pdb.topology, system, npt_integrator)
    else:
        simulation = app.Simulation(pdb.topology, system, npt_integrator, platform, properties)
    simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    simulation.context.setPositions(state.getPositions())
    simulation.context.setVelocities(state.getVelocities())

    csv_path = output_dir / f"{args.prefix}_state.csv"
    traj_path = output_dir / f"{args.prefix}.dcd"
    simulation.reporters.append(
        app.StateDataReporter(
            str(csv_path),
            args.report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            separator=",",
        )
    )
    wrote_traj = False
    if system.getNumParticles() == num_real_atoms and args.traj_interval > 0:
        simulation.reporters.append(app.DCDReporter(str(traj_path), args.traj_interval))
        wrote_traj = True

    wall_start = time.time()
    try:
        if args.npt_steps > 0:
            simulation.step(args.npt_steps)
        status = "completed"
        error_message = None
    except Exception as exc:
        status = "error"
        error_message = f"{exc.__class__.__name__}: {exc}"

    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    last_row = read_last_csv_row(csv_path)

    write_json(
        output_dir / f"{args.prefix}_summary.json",
        {
            "param_file": str(Path(args.param_file).resolve()),
            "xml": str(xml_path.resolve()),
            "pdb": str(Path(args.pdb).resolve()),
            "intra_params_dir": str(intra_params_dir.resolve()),
            "platform": simulation.context.getPlatform().getName(),
            "precision": args.precision,
            "temperature_k": args.temperature_k,
            "pressure_atm": args.pressure_atm,
            "dt_fs": args.dt_fs,
            "cutoff_nm": args.cutoff_nm,
            "max_drude_distance_nm": args.max_drude_distance_nm,
            "integrator": args.integrator,
            "nvt_steps": args.nvt_steps,
            "npt_steps": args.npt_steps,
            "short_range_s12": args.short_range_s12,
            "initial_energy_kj_mol": initial_energy,
            "post_nvt_energy_kj_mol": nvt_energy,
            "final_energy_kj_mol": final_energy,
            "status": status,
            "error": error_message,
            "last_report": last_row,
            "force_count": system.getNumForces(),
            "num_real_atoms": num_real_atoms,
            "num_system_particles": system.getNumParticles(),
            "trajectory_enabled": wrote_traj,
            "force_inventory": force_inventory(system),
            "elapsed_ns_day_final": elapsed_ns_per_day(wall_start, simulation.currentStep, args.dt_fs),
        },
    )

    with open(output_dir / f"{args.prefix}_final.pdb", "w", encoding="ascii") as handle:
        app.PDBFile.writeFile(pdb.topology, final_state.getPositions()[: len(list(pdb.topology.atoms()))], handle)

    print(f"Platform: {simulation.context.getPlatform().getName()}")
    print(f"Status: {status}")
    if error_message is not None:
        print(error_message)
    print(f"Last report: {last_row}")


if __name__ == "__main__":
    main()
