from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from openmm import DrudeNoseHooverIntegrator, MonteCarloBarostat, Platform, Vec3
from openmm.app import DCDReporter, ForceField, Modeller, PME, Simulation, StateDataReporter, Topology
from openmm.unit import (
    atmospheres,
    femtoseconds,
    kelvin,
    kilojoules_per_mole,
    nanometer,
    picosecond,
)


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
OUTPUT = ROOT / "output" / "bulk_water_npt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Drude bulk-water NPT baseline.")
    parser.add_argument("--ffxml", default=str(INPUTS / "charmm_polar_2023.xml"))
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--box-edge-nm", type=float, default=2.0)
    parser.add_argument("--cutoff-nm", type=float, default=0.8)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-atm", type=float, default=1.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--max-drude-distance-nm", type=float, default=0.02)
    parser.add_argument("--nvt-steps", type=int, default=1000)
    parser.add_argument("--npt-steps", type=int, default=2000)
    parser.add_argument("--minimize-max-iter", type=int, default=100)
    parser.add_argument("--report-interval", type=int, default=500)
    parser.add_argument("--traj-interval", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--prefix", default="bulk_water")
    return parser.parse_args()


def get_platform(name: str, precision: str):
    try:
        platform = Platform.getPlatformByName(name)
        props = {"Precision": precision} if name == "CUDA" else {}
        return platform, props
    except Exception:
        return None, {}


def build_modeller(forcefield: ForceField, box_edge_nm: float) -> Modeller:
    modeller = Modeller(Topology(), [])
    modeller.addSolvent(forcefield, model="swm4ndp", boxSize=Vec3(box_edge_nm, box_edge_nm, box_edge_nm) * nanometer)
    return modeller


def build_integrator(args: argparse.Namespace) -> DrudeNoseHooverIntegrator:
    integrator = DrudeNoseHooverIntegrator(
        args.temperature_k * kelvin,
        1 / picosecond,
        1 * kelvin,
        20 / picosecond,
        args.dt_fs * femtoseconds,
    )
    integrator.setMaxDrudeDistance(args.max_drude_distance_nm * nanometer)
    return integrator


def read_last_row(csv_path: Path) -> dict[str, str] | None:
    with open(csv_path, encoding="ascii") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forcefield = ForceField(args.ffxml)
    modeller = build_modeller(forcefield, args.box_edge_nm)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=args.cutoff_nm * nanometer,
        constraints=None,
        rigidWater=True,
    )

    platform, properties = get_platform(args.platform, args.precision)
    nvt_integrator = build_integrator(args)
    if platform is None:
        simulation = Simulation(modeller.topology, system, nvt_integrator)
    else:
        simulation = Simulation(modeller.topology, system, nvt_integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)

    initial_state = simulation.context.getState(getEnergy=True)
    initial_energy = initial_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    simulation.minimizeEnergy(maxIterations=args.minimize_max_iter)
    simulation.context.setVelocitiesToTemperature(args.temperature_k * kelvin)
    if args.nvt_steps > 0:
        simulation.step(args.nvt_steps)

    state = simulation.context.getState(getPositions=True, getVelocities=True)
    system.addForce(MonteCarloBarostat(args.pressure_atm * atmospheres, args.temperature_k * kelvin, 25))
    npt_integrator = build_integrator(args)
    if platform is None:
        simulation = Simulation(modeller.topology, system, npt_integrator)
    else:
        simulation = Simulation(modeller.topology, system, npt_integrator, platform, properties)
    simulation.context.setPositions(state.getPositions())
    simulation.context.setVelocities(state.getVelocities())
    simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())

    state_csv = output_dir / f"{args.prefix}_state.csv"
    traj_dcd = output_dir / f"{args.prefix}.dcd"
    simulation.reporters.append(
        StateDataReporter(
            str(state_csv),
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
    simulation.reporters.append(DCDReporter(str(traj_dcd), args.traj_interval))
    if args.npt_steps > 0:
        simulation.step(args.npt_steps)

    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = final_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    last_row = read_last_row(state_csv)

    with open(output_dir / f"{args.prefix}_summary.json", "w", encoding="ascii") as f:
        json.dump(
            {
                "ffxml": str(Path(args.ffxml).resolve()),
                "platform": simulation.context.getPlatform().getName(),
                "num_particles": system.getNumParticles(),
                "num_residues": sum(1 for _ in modeller.topology.residues()),
                "initial_energy_kj_mol": initial_energy,
                "final_energy_kj_mol": final_energy,
                "box_edge_nm": args.box_edge_nm,
                "cutoff_nm": args.cutoff_nm,
                "temperature_k": args.temperature_k,
                "pressure_atm": args.pressure_atm,
                "dt_fs": args.dt_fs,
                "nvt_steps": args.nvt_steps,
                "npt_steps": args.npt_steps,
                "max_drude_distance_nm": args.max_drude_distance_nm,
                "last_report": last_row,
            },
            f,
            indent=2,
        )

    with open(output_dir / f"{args.prefix}_final.pdb", "w", encoding="ascii") as f:
        from openmm.app import PDBFile

        PDBFile.writeFile(modeller.topology, final_state.getPositions(), f)

    print(f"Platform: {simulation.context.getPlatform().getName()}")
    print(f"Initial energy: {initial_energy:.6f} kJ/mol")
    print(f"Final energy: {final_energy:.6f} kJ/mol")
    if last_row is not None:
        print(f"Last density: {float(last_row['Density (g/mL)']):.6f} g/mL")


if __name__ == "__main__":
    main()
