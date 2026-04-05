from __future__ import annotations

import argparse
import json
from pathlib import Path

from openmm import DrudeSCFIntegrator, Platform
from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, Simulation
from openmm.unit import femtoseconds, kilojoules_per_mole


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
OUTPUT = ROOT / "output" / "water_dimer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short Drude water dimer sanity check.")
    parser.add_argument("--pdb", default=str(INPUTS / "water_dimer.pdb"))
    parser.add_argument("--ffxml", default=str(INPUTS / "charmm_polar_2023.xml"))
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--precision", default="double")
    parser.add_argument("--minimize-max-iter", type=int, default=100)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--prefix", default="water_dimer")
    return parser.parse_args()


def get_platform(name: str, precision: str):
    try:
        platform = Platform.getPlatformByName(name)
        props = {"Precision": precision} if name == "CUDA" else {}
        return platform, props
    except Exception:
        return None, {}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb = PDBFile(args.pdb)
    forcefield = ForceField(args.ffxml)
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addExtraParticles(forcefield)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        rigidWater=True,
    )
    integrator = DrudeSCFIntegrator(1e-10 * femtoseconds)
    integrator.setMinimizationErrorTolerance(1e-12)

    platform, properties = get_platform(args.platform, args.precision)
    if platform is None:
        simulation = Simulation(modeller.topology, system, integrator)
    else:
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)

    initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        kilojoules_per_mole
    )
    simulation.minimizeEnergy(maxIterations=args.minimize_max_iter)
    integrator.step(1)
    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = final_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    with open(output_dir / f"{args.prefix}_summary.json", "w", encoding="ascii") as f:
        json.dump(
            {
                "ffxml": str(Path(args.ffxml).resolve()),
                "pdb": str(Path(args.pdb).resolve()),
                "platform": simulation.context.getPlatform().getName(),
                "num_particles": system.getNumParticles(),
                "initial_energy_kj_mol": initial_energy,
                "final_energy_kj_mol": final_energy,
            },
            f,
            indent=2,
        )

    with open(output_dir / f"{args.prefix}_final.pdb", "w", encoding="ascii") as f:
        PDBFile.writeFile(modeller.topology, final_state.getPositions(), f)

    print(f"Platform: {simulation.context.getPlatform().getName()}")
    print(f"Initial energy: {initial_energy:.6f} kJ/mol")
    print(f"Final energy: {final_energy:.6f} kJ/mol")


if __name__ == "__main__":
    main()
