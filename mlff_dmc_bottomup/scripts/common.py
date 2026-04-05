from __future__ import annotations

import csv
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import openmm.app as app
import openmm.unit as unit


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
OUTPUT = ROOT / "output"

SOURCE_ROOT = ROOT.parent / "drude_dmc_bottomup"
SOURCE_RAW_DIMER_PICKLE = ROOT.parent.parent / "1_training_slater_nb" / "data_dimer.pickle"

DEFAULT_DMC_PDB = INPUTS / "structures" / "DMC.pdb"
DEFAULT_DMC_DIMER_PDB = INPUTS / "structures" / "dimer_001_DMC_DMC.pdb"
DEFAULT_DMC_JSON = INPUTS / "params_results" / "DMC.json"
DEFAULT_MONOMER_TARGETS = INPUTS / "targets" / "monomer_targets.json"
DEFAULT_FULL_TARGETS = INPUTS / "targets" / "dmc_dimer_batch000_full_targets.npz"

ONE_4PI_EPS0 = 138.935456
E_NM_TO_DEBYE = 48.03204255928332
MAX_GRAPH_DISTANCE = 6
MIN_POSITIVE = 1.0e-8

ATOM_GROUP_ORDER = (
    "ester_carbon",
    "ester_oxygen",
    "carbonyl_carbon",
    "carbonyl_oxygen",
    "methyl_hydrogen",
)

MODEL_HEAD_ORDER = (
    "charge_delta",
    "alpha_log_scale",
    "thole_log_scale",
    "c6_log_scale",
    "rep_eps_log_scale",
    "rep_lamb_log_scale",
    "ct_eps_log_scale",
    "ct_lamb_log_scale",
    "sr_es_log_scale",
    "sr_pol_log_scale",
)


@dataclass(frozen=True)
class MonomerGraph:
    atom_names: tuple[str, ...]
    atom_name_index: np.ndarray
    group_index: np.ndarray
    atomic_numbers: np.ndarray
    degrees: np.ndarray
    positions_nm: np.ndarray
    graph_distance: np.ndarray
    bonds: tuple[tuple[int, int], ...]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def normalize_atom_label(name: str) -> str:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", name.strip())
    if match is None:
        return name.strip().upper()
    prefix, digits = match.groups()
    return f"{prefix.upper()}{int(digits)}"


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
    raise ValueError(f"Unsupported DMC atom label: {name}")


def element_atomic_number(symbol: str) -> int:
    return {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "P": 15,
        "S": 16,
        "CL": 17,
    }[symbol.strip().upper()]


def graph_distances(num_atoms: int, bonds: list[tuple[int, int]]) -> np.ndarray:
    dist = np.full((num_atoms, num_atoms), MAX_GRAPH_DISTANCE, dtype=np.int32)
    np.fill_diagonal(dist, 0)
    adjacency = [[] for _ in range(num_atoms)]
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)
        dist[i, j] = 1
        dist[j, i] = 1
    for start in range(num_atoms):
        queue = [start]
        seen = {start}
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                dist[start, neighbor] = min(dist[start, current] + 1, MAX_GRAPH_DISTANCE)
                queue.append(neighbor)
    return np.minimum(dist, MAX_GRAPH_DISTANCE)


def load_monomer_graph(pdb_path: Path = DEFAULT_DMC_PDB) -> MonomerGraph:
    pdb = app.PDBFile(str(pdb_path))
    atoms = list(pdb.topology.atoms())
    bonds = tuple(
        sorted((bond.atom1.index, bond.atom2.index))
        for bond in pdb.topology.bonds()
    )
    atom_names = tuple(atom.name for atom in atoms)
    unique_names = {name: index for index, name in enumerate(atom_names)}
    positions_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    atomic_numbers = np.asarray([element_atomic_number(atom.element.symbol) for atom in atoms], dtype=np.int32)
    degree = np.zeros(len(atoms), dtype=np.int32)
    for i, j in bonds:
        degree[i] += 1
        degree[j] += 1
    return MonomerGraph(
        atom_names=atom_names,
        atom_name_index=np.asarray([unique_names[name] for name in atom_names], dtype=np.int32),
        group_index=np.asarray([ATOM_GROUP_ORDER.index(dmc_atom_group(name)) for name in atom_names], dtype=np.int32),
        atomic_numbers=atomic_numbers,
        degrees=degree,
        positions_nm=positions_nm,
        graph_distance=graph_distances(len(atoms), list(bonds)),
        bonds=bonds,
    )


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


def extract_full_dimer_targets(
    raw_pickle: Path = SOURCE_RAW_DIMER_PICKLE,
    config_key: str = "conf_001_DMC_DMC",
    batch_key: str = "000",
) -> dict[str, np.ndarray]:
    data = load_legacy_pickle(raw_pickle)
    batch = data[config_key][batch_key]
    targets = {
        "shift_angstrom": np.asarray(batch["shift"], dtype=float),
        "posA_angstrom": np.asarray(batch["posA"], dtype=float),
        "posB_angstrom": np.asarray(batch["posB"], dtype=float),
        "weights": np.asarray(batch["wts"], dtype=float),
        "lr_es_kj_mol": np.asarray(batch["lr_es"], dtype=float),
        "lr_pol_kj_mol": np.asarray(batch["lr_pol"], dtype=float),
        "lr_disp_kj_mol": np.asarray(batch["lr_disp"], dtype=float),
        "sr_es_total_kj_mol": np.asarray(batch["es"], dtype=float) - np.asarray(batch["lr_es"], dtype=float),
        "sr_pol_total_kj_mol": np.asarray(batch["pol"], dtype=float) - np.asarray(batch["lr_pol"], dtype=float),
        "exchange_kj_mol": np.asarray(batch["ex"], dtype=float),
        "sr_disp_kj_mol": np.asarray(batch["disp"], dtype=float),
        "ct_like_kj_mol": np.asarray(batch["dhf"], dtype=float),
        "total_nonbonded_kj_mol": np.asarray(batch["tot_full"], dtype=float),
    }
    targets["dispersion_total_kj_mol"] = targets["sr_disp_kj_mol"] + targets["lr_disp_kj_mol"]
    targets["target_lr_espol_kj_mol"] = targets["lr_es_kj_mol"] + targets["lr_pol_kj_mol"]
    targets["target_sr_espol_kj_mol"] = targets["sr_es_total_kj_mol"] + targets["sr_pol_total_kj_mol"]
    targets["target_total_espol_kj_mol"] = targets["target_lr_espol_kj_mol"] + targets["target_sr_espol_kj_mol"]
    return targets


def save_full_target_bundle(targets: dict[str, np.ndarray], npz_path: Path, csv_path: Path) -> None:
    ensure_dir(npz_path.parent)
    np.savez(npz_path, **targets)
    fields = [
        "shift_angstrom",
        "lr_es_kj_mol",
        "lr_pol_kj_mol",
        "lr_disp_kj_mol",
        "sr_es_total_kj_mol",
        "sr_pol_total_kj_mol",
        "exchange_kj_mol",
        "sr_disp_kj_mol",
        "ct_like_kj_mol",
        "dispersion_total_kj_mol",
        "target_total_espol_kj_mol",
        "total_nonbonded_kj_mol",
        "weights",
    ]
    with open(csv_path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        rows = zip(*(targets[field] for field in fields))
        for row in rows:
            writer.writerow([f"{float(value):.10f}" for value in row])


def load_base_nonbonded_parameters(path: Path = DEFAULT_DMC_JSON) -> dict[str, np.ndarray]:
    data = load_json(path)
    alpha = np.asarray(data["alpha"], dtype=float)
    params = {
        "charge": np.asarray(data["charge"], dtype=float),
        "alpha": alpha,
        "thole": np.full_like(alpha, 1.3),
        "c6": np.asarray(data["C6"], dtype=float),
        "rep_eps": np.asarray(data["eps"], dtype=float),
        "rep_lamb": np.asarray(data["lamb"], dtype=float),
        "ct_eps": np.asarray(data["ct_eps"], dtype=float),
        "ct_lamb": np.asarray(data["ct_lamb"], dtype=float),
        "pol_damping": np.asarray(data["pol_damping"], dtype=float),
    }
    params["total_charge"] = np.asarray(np.sum(params["charge"]), dtype=float)
    return params


def load_monomer_targets(path: Path = DEFAULT_MONOMER_TARGETS) -> dict:
    data = load_json(path)
    return {
        "dipole_debye": np.asarray(data["dipole_debye"], dtype=float),
        "isotropic_polarizability_nm3": float(data["isotropic_polarizability_nm3"]),
    }


def one_hot(indices: np.ndarray, size: int) -> np.ndarray:
    return np.eye(size, dtype=float)[indices]


def build_feature_matrix(graph: MonomerGraph) -> np.ndarray:
    features = np.concatenate(
        [
            one_hot(graph.atom_name_index, len(graph.atom_names)),
            one_hot(graph.group_index, len(ATOM_GROUP_ORDER)),
            one_hot(np.clip(graph.degrees, 0, 4), 5),
            (graph.atomic_numbers[:, None].astype(float) / 10.0),
        ],
        axis=1,
    )
    return features.astype(np.float32)


def init_attention_model(
    rng: jax.Array,
    input_dim: int,
    num_pair_bins: int,
    hidden_dim: int = 48,
    num_heads: int = 4,
    num_layers: int = 2,
) -> dict:
    keys = iter(jax.random.split(rng, 5 + num_layers * 8))

    def rand(shape, scale=0.08):
        return jax.random.normal(next(keys), shape, dtype=jnp.float32) * scale

    params = {
        "input_proj_w": rand((input_dim, hidden_dim)),
        "input_proj_b": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "pair_bias": rand((num_pair_bins, num_heads), scale=0.03),
        "layers": [],
        "head_w": rand((hidden_dim, len(MODEL_HEAD_ORDER)), scale=0.05),
        "head_b": jnp.zeros((len(MODEL_HEAD_ORDER),), dtype=jnp.float32),
    }
    head_dim = hidden_dim // num_heads
    for _ in range(num_layers):
        params["layers"].append(
            {
                "wq": rand((hidden_dim, hidden_dim)),
                "wk": rand((hidden_dim, hidden_dim)),
                "wv": rand((hidden_dim, hidden_dim)),
                "wo": rand((hidden_dim, hidden_dim)),
                "w1": rand((hidden_dim, hidden_dim * 2)),
                "b1": jnp.zeros((hidden_dim * 2,), dtype=jnp.float32),
                "w2": rand((hidden_dim * 2, hidden_dim)),
                "b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
                "scale": jnp.asarray(1.0 / math.sqrt(float(head_dim)), dtype=jnp.float32),
            }
        )
    return params


def _gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x)


def encode_graph(params: dict, features: jax.Array, pair_bins: jax.Array) -> jax.Array:
    hidden = _gelu(features @ params["input_proj_w"] + params["input_proj_b"])
    num_heads = params["pair_bias"].shape[1]
    head_dim = hidden.shape[-1] // num_heads
    pair_bias = params["pair_bias"][pair_bins]
    for layer in params["layers"]:
        q = (hidden @ layer["wq"]).reshape(hidden.shape[0], num_heads, head_dim)
        k = (hidden @ layer["wk"]).reshape(hidden.shape[0], num_heads, head_dim)
        v = (hidden @ layer["wv"]).reshape(hidden.shape[0], num_heads, head_dim)
        scores = jnp.einsum("ihd,jhd->hij", q, k) * layer["scale"]
        scores = scores + jnp.transpose(pair_bias, (2, 0, 1))
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("hij,jhd->ihd", weights, v).reshape(hidden.shape[0], -1)
        hidden = hidden + _gelu(attended @ layer["wo"])
        ff = _gelu(hidden @ layer["w1"] + layer["b1"]) @ layer["w2"] + layer["b2"]
        hidden = hidden + _gelu(ff)
    return hidden


def predict_parameters(
    model_params: dict,
    graph: MonomerGraph,
    base_params: dict[str, np.ndarray],
) -> dict[str, jax.Array]:
    features = jnp.asarray(build_feature_matrix(graph))
    pair_bins = jnp.asarray(np.clip(graph.graph_distance, 0, MAX_GRAPH_DISTANCE))
    base = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in base_params.items()}
    hidden = encode_graph(model_params, features, pair_bins)
    raw = hidden @ model_params["head_w"] + model_params["head_b"]
    charge_delta = 0.08 * jnp.tanh(raw[:, 0])
    charge = base["charge"] + charge_delta
    charge = charge - (jnp.sum(charge) - base["total_charge"]) / charge.shape[0]
    alpha = jnp.maximum(base["alpha"] * jnp.exp(0.6 * jnp.tanh(raw[:, 1])), MIN_POSITIVE)
    thole = jnp.clip(base["thole"] * jnp.exp(0.5 * jnp.tanh(raw[:, 2])), 0.2, 3.5)
    c6 = jnp.maximum(base["c6"] * jnp.exp(0.8 * jnp.tanh(raw[:, 3])), MIN_POSITIVE)
    rep_eps = jnp.maximum(base["rep_eps"] * jnp.exp(0.8 * jnp.tanh(raw[:, 4])), MIN_POSITIVE)
    rep_lamb = jnp.maximum(base["rep_lamb"] * jnp.exp(0.3 * jnp.tanh(raw[:, 5])), 0.1)
    ct_eps = jnp.maximum(base["ct_eps"] * jnp.exp(0.8 * jnp.tanh(raw[:, 6])), MIN_POSITIVE)
    ct_lamb = jnp.maximum(base["ct_lamb"] * jnp.exp(0.3 * jnp.tanh(raw[:, 7])), 0.1)
    sr_es_scale = jnp.exp(0.6 * jnp.tanh(raw[:, 8]))
    sr_pol_scale = jnp.exp(0.6 * jnp.tanh(raw[:, 9]))
    return {
        "charge": charge,
        "alpha": alpha,
        "thole": thole,
        "c6": c6,
        "rep_eps": rep_eps,
        "rep_lamb": rep_lamb,
        "ct_eps": ct_eps,
        "ct_lamb": ct_lamb,
        "sr_es_scale": sr_es_scale,
        "sr_pol_scale": sr_pol_scale,
    }


def dimer_distance_tensors(targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    delta_ang = targets["posA_angstrom"][:, :, None, :] - targets["posB_angstrom"][:, None, :, :]
    distance_ang = np.linalg.norm(delta_ang, axis=-1)
    distance_ang = np.maximum(distance_ang, 1.0e-4)
    distance_nm = distance_ang * 0.1
    return {
        "distance_ang": distance_ang.astype(np.float32),
        "distance_nm": distance_nm.astype(np.float32),
    }


def evaluate_decomposed_terms(
    predicted: dict[str, jax.Array],
    distances_nm: jax.Array,
    distances_ang: jax.Array,
) -> dict[str, jax.Array]:
    q_a = predicted["charge"][None, :, None]
    q_b = predicted["charge"][None, None, :]
    alpha_a = predicted["alpha"][None, :, None]
    alpha_b = predicted["alpha"][None, None, :]
    thole = 0.5 * (predicted["thole"][None, :, None] + predicted["thole"][None, None, :])
    c6_ij = jnp.sqrt(predicted["c6"][None, :, None] * predicted["c6"][None, None, :])
    rep_eps_ij = jnp.sqrt(predicted["rep_eps"][None, :, None] * predicted["rep_eps"][None, None, :])
    rep_lamb_ij = 0.5 * (predicted["rep_lamb"][None, :, None] + predicted["rep_lamb"][None, None, :])
    ct_eps_ij = jnp.sqrt(predicted["ct_eps"][None, :, None] * predicted["ct_eps"][None, None, :])
    ct_lamb_ij = 0.5 * (predicted["ct_lamb"][None, :, None] + predicted["ct_lamb"][None, None, :])
    sr_es_ij = jnp.sqrt(predicted["sr_es_scale"][None, :, None] * predicted["sr_es_scale"][None, None, :])
    sr_pol_ij = jnp.sqrt(predicted["sr_pol_scale"][None, :, None] * predicted["sr_pol_scale"][None, None, :])

    inv_r_nm = 1.0 / distances_nm
    inv_r4_nm = inv_r_nm**4
    inv_r6_nm = inv_r_nm**6
    damp_short = jnp.exp(-0.35 * thole * distances_ang)
    damp_disp = 1.0 - jnp.exp(-2.0 * distances_nm)

    lr_es_pair = ONE_4PI_EPS0 * q_a * q_b * inv_r_nm
    lr_pol_pair = -0.5 * ONE_4PI_EPS0 * (alpha_a * (q_b**2) + alpha_b * (q_a**2)) * inv_r4_nm * (1.0 - damp_short)
    sr_es_pair = 0.20 * ONE_4PI_EPS0 * q_a * q_b * inv_r_nm * damp_short * sr_es_ij
    sr_pol_pair = -0.15 * ONE_4PI_EPS0 * jnp.sqrt(alpha_a * alpha_b) * inv_r6_nm * damp_short * sr_pol_ij
    exchange_pair = rep_eps_ij * jnp.exp(-rep_lamb_ij * distances_ang)
    sr_disp_pair = -40.0 * c6_ij * inv_r6_nm * damp_disp * damp_short
    lr_disp_pair = -40.0 * c6_ij * inv_r6_nm * (1.0 - damp_short)
    ct_like_pair = -ct_eps_ij * jnp.exp(-ct_lamb_ij * distances_ang)

    terms = {
        "lr_es_kj_mol": jnp.sum(lr_es_pair, axis=(1, 2)),
        "lr_pol_kj_mol": jnp.sum(lr_pol_pair, axis=(1, 2)),
        "sr_es_total_kj_mol": jnp.sum(sr_es_pair, axis=(1, 2)),
        "sr_pol_total_kj_mol": jnp.sum(sr_pol_pair, axis=(1, 2)),
        "exchange_kj_mol": jnp.sum(exchange_pair, axis=(1, 2)),
        "sr_disp_kj_mol": jnp.sum(sr_disp_pair, axis=(1, 2)),
        "lr_disp_kj_mol": jnp.sum(lr_disp_pair, axis=(1, 2)),
        "ct_like_kj_mol": jnp.sum(ct_like_pair, axis=(1, 2)),
    }
    terms["dispersion_total_kj_mol"] = terms["sr_disp_kj_mol"] + terms["lr_disp_kj_mol"]
    terms["target_lr_espol_kj_mol"] = terms["lr_es_kj_mol"] + terms["lr_pol_kj_mol"]
    terms["target_sr_espol_kj_mol"] = terms["sr_es_total_kj_mol"] + terms["sr_pol_total_kj_mol"]
    terms["target_total_espol_kj_mol"] = terms["target_lr_espol_kj_mol"] + terms["target_sr_espol_kj_mol"]
    terms["total_nonbonded_kj_mol"] = (
        terms["target_total_espol_kj_mol"]
        + terms["exchange_kj_mol"]
        + terms["dispersion_total_kj_mol"]
        + terms["ct_like_kj_mol"]
    )
    return terms


def evaluate_monomer_properties(predicted: dict[str, jax.Array], graph: MonomerGraph) -> dict[str, jax.Array]:
    positions_nm = jnp.asarray(graph.positions_nm, dtype=jnp.float32)
    dipole_debye = jnp.sum(predicted["charge"][:, None] * positions_nm, axis=0) * E_NM_TO_DEBYE
    isotropic_polar_nm3 = jnp.sum(predicted["alpha"])
    return {
        "dipole_debye": dipole_debye,
        "isotropic_polarizability_nm3": isotropic_polar_nm3,
    }


def parameter_summary(predicted: dict[str, np.ndarray], atom_names: tuple[str, ...]) -> list[dict]:
    summary = []
    for idx, name in enumerate(atom_names):
        row = {"index": idx + 1, "name": name}
        for key, values in predicted.items():
            row[key] = float(np.asarray(values)[idx])
        summary.append(row)
    return summary
