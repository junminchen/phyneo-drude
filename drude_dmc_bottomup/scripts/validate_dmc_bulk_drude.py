from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from common import (
    DEFAULT_DMC_BULK_PDB,
    DEFAULT_DMC_ITP,
    DEFAULT_DRUDE_MODEL,
    OUTPUT,
    create_drude_espol_system,
    density_from_state,
    elapsed_ns_per_day,
    load_drude_model,
    make_positions_with_drudes,
    read_last_csv_row,
    resolve_platform,
    temperature_from_state,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posterior-only DMC bulk Drude validation.")
    parser.add_argument("--pdb", default=str(DEFAULT_DMC_BULK_PDB))
    parser.add_argument("--param-file", default=str(DEFAULT_DRUDE_MODEL))
    parser.add_argument("--itp", default=str(DEFAULT_DMC_ITP))
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-atm", type=float, default=1.0)
    parser.add_argument("--dt-fs", type=float, default=0.25)
    parser.add_argument("--nvt-steps", type=int, default=100)
    parser.add_argument("--npt-steps", type=int, default=200)
    parser.add_argument("--barostat-frequency", type=int, default=25)
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(OUTPUT / "bulk_validation"))
    parser.add_argument("--prefix", default="dmc_bulk_drude")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb = app.PDBFile(args.pdb)
    model = load_drude_model(Path(args.param_file))
    system, metadata = create_drude_espol_system(
        pdb.topology,
        model,
        nonbonded_method="PME",
        cutoff_nm=args.cutoff_nm,
        include_bonded=True,
        itp_path=Path(args.itp),
    )
    system.addForce(mm.CMMotionRemover())
    system.addForce(mm.MonteCarloBarostat(args.pressure_atm * unit.atmospheres, args.temperature_k * unit.kelvin, args.barostat_frequency))

    integrator = mm.DrudeNoseHooverIntegrator(
        args.temperature_k * unit.kelvin,
        1 / unit.picosecond,
        1 * unit.kelvin,
        20 / unit.picosecond,
        args.dt_fs * unit.femtoseconds,
    )
    integrator.setMaxDrudeDistance(0.02 * unit.nanometer)
    platform, properties = resolve_platform(args.platform, args.precision)
    if platform is None:
        simulation = app.Simulation(pdb.topology, system, integrator)
    else:
        simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    if pdb.topology.getPeriodicBoxVectors() is not None:
        simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    real_positions_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    simulation.context.setPositions(make_positions_with_drudes(real_positions_nm, metadata["drude_system_indices"]))
    mm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=50)

    csv_path = output_dir / f"{args.prefix}_state.csv"
    with open(csv_path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time_ps", "potential_kj_mol", "temperature_k", "density_g_ml", "speed_ns_day"])
        start = mm.openmm.Platform.getOpenMMVersion()  # keep linter quiet
        del start
    import time

    wall_start = time.time()
    total_steps = args.nvt_steps + args.npt_steps
    phase_switch = args.nvt_steps
    while simulation.currentStep < total_steps:
        step_block = min(args.report_interval, total_steps - simulation.currentStep)
        simulation.step(step_block)
        state = simulation.context.getState(getEnergy=True)
        current_step = simulation.currentStep
        density = density_from_state(system, state)
        temperature = temperature_from_state(system, state)
        speed = elapsed_ns_per_day(wall_start, current_step, args.dt_fs)
        with open(csv_path, "a", encoding="ascii", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    current_step,
                    f"{current_step * args.dt_fs * 1.0e-3:.6f}",
                    f"{state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole):.8f}",
                    f"{temperature:.6f}",
                    f"{density:.8f}",
                    f"{speed:.4f}",
                ]
            )
        if current_step == phase_switch:
            pass

    last_row = read_last_csv_row(csv_path)
    write_json(
        output_dir / f"{args.prefix}_summary.json",
        {
            "param_file": str(Path(args.param_file).resolve()),
            "pdb": str(Path(args.pdb).resolve()),
            "itp": str(Path(args.itp).resolve()),
            "platform": simulation.context.getPlatform().getName(),
            "temperature_k": args.temperature_k,
            "pressure_atm": args.pressure_atm,
            "dt_fs": args.dt_fs,
            "nvt_steps": args.nvt_steps,
            "npt_steps": args.npt_steps,
            "last_report": last_row,
        },
    )
    print(f"Platform: {simulation.context.getPlatform().getName()}")
    print(f"Last report: {last_row}")


if __name__ == "__main__":
    main()
