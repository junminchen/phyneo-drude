from __future__ import annotations

import copy
import csv
import json
import math
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
OUTPUT = ROOT / "output"

DEFAULT_RAW_DIMER_PICKLE = (
    ROOT.parent / "1_training_slater_nb" / "data_dimer.pickle"
)
DEFAULT_DMC_PDB = INPUTS / "structures" / "DMC.pdb"
DEFAULT_DMC_DIMER_PDB = INPUTS / "structures" / "dimer_001_DMC_DMC.pdb"
DEFAULT_DMC_BULK_PDB = INPUTS / "structures" / "dmc_100mol_box.pdb"
DEFAULT_DMC_ITP = INPUTS / "params_results" / "DMC.itp"
DEFAULT_DMC_JSON = INPUTS / "params_results" / "DMC.json"
DEFAULT_MONOMER_TEMPLATE = INPUTS / "targets" / "monomer_targets.template.json"
DEFAULT_DIMER_TARGETS = INPUTS / "targets" / "dmc_dimer_batch000_targets.npz"
DEFAULT_DRUDE_MODEL = INPUTS / "dmc_drude_initial.json"

DRUDE_CHARGE_FACTOR = 3000.0
DEFAULT_THOLE = 1.3
DEFAULT_DRUDE_MASS_DA = 0.4
MIN_POLARIZABILITY = 1.0e-8
E_NM_TO_DEBYE = 48.03204255928332
DA_PER_NM3_TO_G_PER_ML = 1.66053906660e-3
ONE_4PI_EPS0 = 138.935456

DMC_ATOM_GROUP_ORDER = (
    "ester_carbon",
    "ester_oxygen",
    "carbonyl_carbon",
    "carbonyl_oxygen",
    "methyl_hydrogen",
)


@dataclass(frozen=True)
class ResidueTemplate:
    atom_names: list[str]
    bonds: list[tuple[int, int, float, float]]
    angles: list[tuple[int, int, int, float, float]]
    torsions: list[tuple[int, int, int, int, int, float, float]]


def normalize_atom_label(name: str) -> str:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", name.strip())
    if match is None:
        return name.strip().upper()
    prefix, digits = match.groups()
    return f"{prefix.upper()}{int(digits)}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def resolve_platform(name: str, precision: str = "mixed"):
    try:
        platform = mm.Platform.getPlatformByName(name)
        properties = {"Precision": precision} if name == "CUDA" else {}
        return platform, properties
    except Exception:
        return None, {}


def _patch_legacy_jax_pickle() -> None:
    import jax._src.core as core

    current = core.ShapedArray.__init__
    if getattr(current, "__name__", "") == "patched_shaped_array_init":
        return

    original = current

    def patched_shaped_array_init(self, shape, dtype, weak_type=False, **kwargs):
        return original(self, shape, dtype, weak_type, sharding=kwargs.get("sharding", None))

    patched_shaped_array_init.__name__ = "patched_shaped_array_init"
    core.ShapedArray.__init__ = patched_shaped_array_init


def load_legacy_pickle(path: Path):
    _patch_legacy_jax_pickle()
    with open(path, "rb") as handle:
        return pickle.load(handle)


def extract_dmc_dimer_targets(
    raw_pickle: Path,
    config_key: str = "conf_001_DMC_DMC",
    batch_key: str = "000",
) -> dict[str, np.ndarray]:
    data = load_legacy_pickle(raw_pickle)
    batch = data[config_key][batch_key]
    targets = {
        "shift_angstrom": np.asarray(batch["shift"], dtype=float),
        "lr_es_kj_mol": np.asarray(batch["lr_es"], dtype=float),
        "lr_pol_kj_mol": np.asarray(batch["lr_pol"], dtype=float),
        "es_kj_mol": np.asarray(batch["es"], dtype=float),
        "pol_kj_mol": np.asarray(batch["pol"], dtype=float),
        "posA_angstrom": np.asarray(batch["posA"], dtype=float),
        "posB_angstrom": np.asarray(batch["posB"], dtype=float),
        "weights": np.asarray(batch["wts"], dtype=float),
    }
    targets["target_lr_espol_kj_mol"] = targets["lr_es_kj_mol"] + targets["lr_pol_kj_mol"]
    targets["target_total_espol_kj_mol"] = targets["es_kj_mol"] + targets["pol_kj_mol"]
    return targets


def save_target_bundle(targets: dict[str, np.ndarray], npz_path: Path, csv_path: Path) -> None:
    ensure_dir(npz_path.parent)
    np.savez(npz_path, **targets)
    with open(csv_path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shift_angstrom",
                "lr_es_kj_mol",
                "lr_pol_kj_mol",
                "target_lr_espol_kj_mol",
                "es_kj_mol",
                "pol_kj_mol",
                "target_total_espol_kj_mol",
                "weight",
            ]
        )
        for row in zip(
            targets["shift_angstrom"],
            targets["lr_es_kj_mol"],
            targets["lr_pol_kj_mol"],
            targets["target_lr_espol_kj_mol"],
            targets["es_kj_mol"],
            targets["pol_kj_mol"],
            targets["target_total_espol_kj_mol"],
            targets["weights"],
        ):
            writer.writerow([f"{float(value):.10f}" for value in row])


def generate_initial_drude_model(
    dmc_json_path: Path = DEFAULT_DMC_JSON,
    pdb_path: Path = DEFAULT_DMC_PDB,
    output_path: Path = DEFAULT_DRUDE_MODEL,
    *,
    drude_charge_factor: float = DRUDE_CHARGE_FACTOR,
    default_thole: float = DEFAULT_THOLE,
) -> dict:
    data = load_json(dmc_json_path)
    pdb = app.PDBFile(str(pdb_path))
    atom_names = [atom.name for atom in pdb.topology.atoms()]
    payload = {
        "description": "Initial DMC Drude espol model generated from DMC.json alpha/charge.",
        "residue_name": "DMC",
        "drude_charge_factor": float(drude_charge_factor),
        "default_thole": float(default_thole),
        "atom_order": atom_names,
        "atoms": [],
    }
    for index, name in enumerate(atom_names):
        charge = float(data["charge"][index])
        alpha = float(data["alpha"][index])
        drude_charge = -math.sqrt(max(alpha, 0.0) * drude_charge_factor) if alpha > 0.0 else 0.0
        payload["atoms"].append(
            {
                "index": index + 1,
                "name": name,
                "charge": charge,
                "alpha": alpha,
                "pol_damping": float(data.get("pol_damping", data["alpha"])[index]),
                "thole": float(default_thole),
                "drude_charge": drude_charge,
                "polarizable": bool(alpha > MIN_POLARIZABILITY),
            }
        )
    write_json(output_path, payload)
    return payload


def load_drude_model(path: Path) -> dict:
    return load_json(path)


def dmc_atom_group(name: str) -> str:
    raw = name.strip().upper()
    normalized = normalize_atom_label(name)
    if raw in {"C00", "C05"} or normalized in {"C0", "C5"}:
        return "ester_carbon"
    if raw in {"O01", "O04"} or normalized in {"O1", "O4"}:
        return "ester_oxygen"
    if raw == "C02" or normalized == "C2":
        return "carbonyl_carbon"
    if raw == "O03" or normalized == "O3":
        return "carbonyl_oxygen"
    if raw.startswith("H") or normalized.startswith("H"):
        return "methyl_hydrogen"
    raise ValueError(f"Unsupported DMC atom label for grouped fitting: {name}")


def scale_drude_model(
    base_model: dict,
    *,
    alpha_scale: float = 1.0,
    drude_charge_scale: float = 1.0,
    thole_scale: float = 1.0,
    thole_group_scales: dict[str, float] | None = None,
    charge_group_deltas: dict[str, float] | None = None,
) -> dict:
    model = copy.deepcopy(base_model)
    thole_group_scales = thole_group_scales or {}
    charge_group_deltas = charge_group_deltas or {}
    for atom in model["atoms"]:
        group = dmc_atom_group(atom["name"])
        atom["alpha"] = float(atom["alpha"]) * alpha_scale
        atom["drude_charge"] = float(atom["drude_charge"]) * drude_charge_scale
        atom["thole"] = float(atom["thole"]) * thole_scale * float(thole_group_scales.get(group, 1.0))
        atom["charge"] = float(atom["charge"]) + float(charge_group_deltas.get(group, 0.0))
        atom["polarizable"] = bool(atom["alpha"] > MIN_POLARIZABILITY and abs(atom["drude_charge"]) > 1.0e-8)
    metadata = model.setdefault("fit_metadata", {})
    metadata["thole_group_scales"] = {group: float(thole_group_scales.get(group, 1.0)) for group in DMC_ATOM_GROUP_ORDER}
    metadata["charge_group_deltas"] = {group: float(charge_group_deltas.get(group, 0.0)) for group in DMC_ATOM_GROUP_ORDER}
    return model


def model_atom_map(model: dict) -> dict[str, dict]:
    return {entry["name"]: entry for entry in model["atoms"]}


def parse_itp_sections(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line.strip("[] ").lower()
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(line)
    return sections


def load_dmc_residue_template(itp_path: Path = DEFAULT_DMC_ITP) -> ResidueTemplate:
    sections = parse_itp_sections(itp_path)
    atoms = []
    for line in sections["atoms"]:
        fields = line.split()
        atoms.append(normalize_atom_label(fields[4]))
    bonds = []
    for line in sections["bonds"]:
        ai, aj, _, r0, k = line.split()[:5]
        bonds.append((int(ai) - 1, int(aj) - 1, float(r0), float(k)))
    angles = []
    for line in sections["angles"]:
        ai, aj, ak, _, theta_deg, k = line.split()[:6]
        angles.append((int(ai) - 1, int(aj) - 1, int(ak) - 1, math.radians(float(theta_deg)), float(k)))
    torsions = []
    for line in sections["dihedrals"]:
        ai, aj, ak, al, _, phase_deg, k, periodicity = line.split()[:8]
        torsions.append(
            (
                int(ai) - 1,
                int(aj) - 1,
                int(ak) - 1,
                int(al) - 1,
                int(periodicity),
                math.radians(float(phase_deg)),
                float(k),
            )
        )
    return ResidueTemplate(atom_names=atoms, bonds=bonds, angles=angles, torsions=torsions)


def add_template_bonded_forces(system: mm.System, topology: app.Topology, template: ResidueTemplate) -> None:
    residues = list(topology.residues())
    bond_force = mm.HarmonicBondForce()
    angle_force = mm.HarmonicAngleForce()
    torsion_force = mm.PeriodicTorsionForce()
    for residue in residues:
        atoms = list(residue.atoms())
        if len(atoms) != len(template.atom_names):
            raise ValueError(f"Residue {residue.name} atom count mismatch: {len(atoms)} != {len(template.atom_names)}")
        actual = [normalize_atom_label(atom.name) for atom in atoms]
        if actual != template.atom_names:
            raise ValueError(f"Residue {residue.name} atom order mismatch: {actual} != {template.atom_names}")
        offset = atoms[0].index
        for ai, aj, r0, k in template.bonds:
            bond_force.addBond(offset + ai, offset + aj, r0 * unit.nanometer, k * unit.kilojoule_per_mole / unit.nanometer**2)
        for ai, aj, ak, theta, k in template.angles:
            angle_force.addAngle(offset + ai, offset + aj, offset + ak, theta * unit.radian, k * unit.kilojoule_per_mole / unit.radian**2)
        for ai, aj, ak, al, periodicity, phase, k in template.torsions:
            torsion_force.addTorsion(offset + ai, offset + aj, offset + ak, offset + al, periodicity, phase * unit.radian, k * unit.kilojoule_per_mole)
    system.addForce(bond_force)
    system.addForce(angle_force)
    system.addForce(torsion_force)


def shortest_path_pairs(num_atoms: int, bonds: Iterable[tuple[int, int]], max_distance: int) -> dict[int, set[tuple[int, int]]]:
    adjacency = [set() for _ in range(num_atoms)]
    for atom1, atom2 in bonds:
        adjacency[atom1].add(atom2)
        adjacency[atom2].add(atom1)
    by_shell = {shell: set() for shell in range(1, max_distance + 1)}
    for start in range(num_atoms):
        visited = {start}
        frontier = {start}
        for distance in range(1, max_distance + 1):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(adjacency[node] - visited)
            for node in next_frontier:
                by_shell[distance].add(tuple(sorted((start, node))))
            visited.update(next_frontier)
            frontier = next_frontier
    return by_shell


def _nonbonded_method_from_name(name: str):
    normalized = name.strip().lower()
    if normalized == "nocutoff":
        return mm.NonbondedForce.NoCutoff
    if normalized == "cutoffnonperiodic":
        return mm.NonbondedForce.CutoffNonPeriodic
    if normalized == "cutoffperiodic":
        return mm.NonbondedForce.CutoffPeriodic
    if normalized == "pme":
        return mm.NonbondedForce.PME
    raise ValueError(f"Unsupported nonbonded method: {name}")


def create_drude_espol_system(
    topology: app.Topology,
    model: dict,
    *,
    nonbonded_method: str = "NoCutoff",
    cutoff_nm: float = 1.0,
    include_bonded: bool = False,
    itp_path: Path = DEFAULT_DMC_ITP,
    field_vector: tuple[float, float, float] | None = None,
    drude_mass_da: float = DEFAULT_DRUDE_MASS_DA,
) -> tuple[mm.System, dict]:
    atom_map = model_atom_map(model)
    real_atoms = list(topology.atoms())
    real_bonds = [(bond[0].index, bond[1].index) for bond in topology.bonds()]
    adjacency = [set() for _ in range(len(real_atoms))]
    for atom1, atom2 in real_bonds:
        adjacency[atom1].add(atom2)
        adjacency[atom2].add(atom1)
    system = mm.System()
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(_nonbonded_method_from_name(nonbonded_method))
    if nonbonded_method.strip().lower() != "nocutoff":
        nonbonded.setCutoffDistance(cutoff_nm * unit.nanometer)
    if nonbonded_method.strip().lower() == "pme":
        nonbonded.setEwaldErrorTolerance(1.0e-4)
    drude_force = mm.DrudeForce()

    particle_charges: list[float] = []
    drude_system_indices: dict[int, int] = {}
    drude_force_indices: dict[int, int] = {}
    parent_positions: list[int] = []

    for atom in real_atoms:
        param = atom_map[atom.name]
        mass = atom.element.mass if atom.element is not None else 12.0 * unit.dalton
        alpha = float(param["alpha"])
        drude_charge = float(param["drude_charge"]) if param.get("polarizable", True) else 0.0
        if alpha > MIN_POLARIZABILITY and abs(drude_charge) > 1.0e-8:
            core_mass = mass - drude_mass_da * unit.dalton
            if core_mass <= 0 * unit.dalton:
                raise ValueError(f"Drude mass too large for atom {atom.name}")
        else:
            core_mass = mass
        system.addParticle(core_mass)
        core_charge = float(param["charge"]) - drude_charge
        nonbonded.addParticle(core_charge, 0.1 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
        particle_charges.append(core_charge)

    for atom in real_atoms:
        param = atom_map[atom.name]
        alpha = float(param["alpha"])
        drude_charge = float(param["drude_charge"]) if param.get("polarizable", True) else 0.0
        if alpha <= MIN_POLARIZABILITY or abs(drude_charge) <= 1.0e-8:
            continue
        system_index = system.addParticle(drude_mass_da * unit.dalton)
        force_index = drude_force.addParticle(
            system_index,
            atom.index,
            -1,
            -1,
            -1,
            drude_charge,
            alpha,
            0.0,
            0.0,
        )
        nonbonded.addParticle(drude_charge, 0.1 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
        particle_charges.append(drude_charge)
        drude_system_indices[atom.index] = system_index
        drude_force_indices[atom.index] = force_index
        parent_positions.append(atom.index)
        for excluded_real_atom in sorted({atom.index} | adjacency[atom.index]):
            nonbonded.addException(
                system_index,
                excluded_real_atom,
                0.0 * unit.elementary_charge**2,
                0.1 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
                replace=True,
            )

    shells = shortest_path_pairs(len(real_atoms), real_bonds, max_distance=2)
    for shell in (1, 2):
        for atom1, atom2 in sorted(shells[shell]):
            nonbonded.addException(
                atom1,
                atom2,
                0.0 * unit.elementary_charge**2,
                0.1 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
                replace=True,
            )
            if atom1 in drude_force_indices and atom2 in drude_force_indices:
                thole = float(atom_map[real_atoms[atom1].name]["thole"]) + float(atom_map[real_atoms[atom2].name]["thole"])
                drude_force.addScreenedPair(drude_force_indices[atom1], drude_force_indices[atom2], thole)

    if field_vector is not None:
        field = mm.CustomExternalForce("-q*(Ex*x+Ey*y+Ez*z)")
        field.addGlobalParameter("Ex", float(field_vector[0]))
        field.addGlobalParameter("Ey", float(field_vector[1]))
        field.addGlobalParameter("Ez", float(field_vector[2]))
        field.addPerParticleParameter("q")
        for index, charge in enumerate(particle_charges):
            field.addParticle(index, [charge])
        system.addForce(field)

    system.addForce(nonbonded)
    system.addForce(drude_force)

    if include_bonded:
        template = load_dmc_residue_template(itp_path)
        add_template_bonded_forces(system, topology, template)

    metadata = {
        "num_real_atoms": len(real_atoms),
        "num_particles": system.getNumParticles(),
        "particle_charges": particle_charges,
        "drude_system_indices": drude_system_indices,
    }
    return system, metadata


def make_positions_with_drudes(real_positions_nm: np.ndarray, drude_system_indices: dict[int, int]) -> list[mm.Vec3]:
    total_particles = real_positions_nm.shape[0] + len(drude_system_indices)
    positions = [None] * total_particles
    for index, coords in enumerate(real_positions_nm):
        positions[index] = mm.Vec3(float(coords[0]), float(coords[1]), float(coords[2])) * unit.nanometer
    for parent_index, drude_index in drude_system_indices.items():
        coords = real_positions_nm[parent_index]
        positions[drude_index] = mm.Vec3(float(coords[0]), float(coords[1]), float(coords[2])) * unit.nanometer
    return positions


def create_scf_context(system: mm.System, platform_name: str = "Reference", precision: str = "mixed"):
    integrator = mm.DrudeSCFIntegrator(1.0e-10 * unit.femtoseconds)
    integrator.setMinimizationErrorTolerance(1.0e-12)
    platform, properties = resolve_platform(platform_name, precision)
    if platform is None:
        context = mm.Context(system, integrator)
    else:
        context = mm.Context(system, integrator, platform, properties)
    return integrator, context


def relax_drude_positions(
    context: mm.Context,
    integrator,
    positions: list[mm.Vec3],
    *,
    box_vectors: tuple[mm.Vec3, mm.Vec3, mm.Vec3] | None = None,
) -> tuple[float, np.ndarray]:
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    integrator.step(1)
    state = context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return float(energy), np.asarray(pos, dtype=float)


def dipole_from_positions(charges: list[float], positions_nm: np.ndarray) -> np.ndarray:
    dipole = np.zeros(3, dtype=float)
    for charge, coords in zip(charges, positions_nm):
        dipole += float(charge) * np.asarray(coords, dtype=float)
    return dipole


def evaluate_monomer_response(
    topology: app.Topology,
    real_positions_nm: np.ndarray,
    model: dict,
    platform: str,
    precision: str,
    *,
    field_strength_internal: float = 1.0,
) -> dict[str, np.ndarray]:
    zero_system, zero_meta = create_drude_espol_system(topology, model, field_vector=None)
    zero_integrator, zero_context = create_scf_context(zero_system, platform, precision)
    try:
        zero_energy, zero_positions = relax_drude_positions(
            zero_context,
            zero_integrator,
            make_positions_with_drudes(real_positions_nm, zero_meta["drude_system_indices"]),
        )
        del zero_energy
        zero_dipole = dipole_from_positions(zero_meta["particle_charges"], zero_positions)
    finally:
        del zero_context, zero_integrator

    polarizability_raw = np.zeros((3, 3), dtype=float)
    axes = np.eye(3)
    for axis_index, axis in enumerate(axes):
        plus_system, plus_meta = create_drude_espol_system(
            topology,
            model,
            field_vector=tuple(axis * field_strength_internal),
        )
        minus_system, minus_meta = create_drude_espol_system(
            topology,
            model,
            field_vector=tuple(-axis * field_strength_internal),
        )
        plus_integrator, plus_context = create_scf_context(plus_system, platform, precision)
        minus_integrator, minus_context = create_scf_context(minus_system, platform, precision)
        try:
            _, plus_positions = relax_drude_positions(
                plus_context,
                plus_integrator,
                make_positions_with_drudes(real_positions_nm, plus_meta["drude_system_indices"]),
            )
            _, minus_positions = relax_drude_positions(
                minus_context,
                minus_integrator,
                make_positions_with_drudes(real_positions_nm, minus_meta["drude_system_indices"]),
            )
            dipole_plus = dipole_from_positions(plus_meta["particle_charges"], plus_positions)
            dipole_minus = dipole_from_positions(minus_meta["particle_charges"], minus_positions)
            polarizability_raw[:, axis_index] = (dipole_plus - dipole_minus) / (2.0 * field_strength_internal)
        finally:
            del plus_context, plus_integrator
            del minus_context, minus_integrator

    polarizability_nm3 = polarizability_raw * ONE_4PI_EPS0
    return {
        "dipole_e_nm": zero_dipole,
        "dipole_debye": zero_dipole * E_NM_TO_DEBYE,
        "polarizability_tensor_nm3": polarizability_nm3,
        "polarizability_tensor_angstrom3": polarizability_nm3 * 1000.0,
    }


def read_last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with open(path, encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def density_from_state(system: mm.System, state: mm.State) -> float:
    total_mass_da = 0.0
    for index in range(system.getNumParticles()):
        total_mass_da += system.getParticleMass(index).value_in_unit(unit.dalton)
    volume_nm3 = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
    return float(total_mass_da / volume_nm3 * DA_PER_NM3_TO_G_PER_ML)


def temperature_from_state(system: mm.System, state: mm.State) -> float:
    kinetic = state.getKineticEnergy()
    dof = 3 * system.getNumParticles()
    return float((2 * kinetic / (dof * unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin))


def elapsed_ns_per_day(start_time: float, current_step: int, dt_fs: float) -> float:
    elapsed_s = max(time.time() - start_time, 1.0e-9)
    elapsed_ns = current_step * dt_fs * 1.0e-6
    return elapsed_ns / elapsed_s * 86400.0
