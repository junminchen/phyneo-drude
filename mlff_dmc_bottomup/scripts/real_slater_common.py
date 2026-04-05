from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    ATOM_GROUP_ORDER,
    DEFAULT_DMC_PDB,
    E_NM_TO_DEBYE,
    MIN_POSITIVE,
    ONE_4PI_EPS0,
    ROOT,
    build_feature_matrix,
    encode_graph,
    load_json,
    load_monomer_graph,
)


DEFAULT_BASE_DRUDE_MODEL_JSON = ROOT.parent / "drude_dmc_bottomup" / "output" / "fit_joint" / "dmc_drude_joint_lr_espol_model.json"
DEFAULT_BASE_SLATER_XML = ROOT.parent / "drude_dmc_bottomup" / "inputs" / "mpid" / "phyneo_ecl.xml"
DEFAULT_S12_NM = 0.169
SHORT_RANGE_DIELECTRIC = 1389.35455846

REAL_SLATER_HEAD_ORDER = (
    "charge_delta",
    "alpha_log_scale",
    "thole_log_scale",
    "slater_b_log_scale",
    "slater_ex_a_log_scale",
    "slater_sr_es_a_log_scale",
    "slater_sr_pol_a_log_scale",
    "slater_sr_disp_a_log_scale",
    "slater_dhf_a_log_scale",
    "disp_c6_log_scale",
    "disp_c8_log_scale",
    "disp_c10_log_scale",
)


def _load_dmc_type_map(xml_path: Path) -> dict[str, str]:
    root = ET.parse(xml_path).getroot()
    residues = root.find("Residues")
    if residues is None:
        raise ValueError(f"Missing <Residues> in {xml_path}")
    for residue in residues.findall("Residue"):
        if residue.attrib.get("name") != "DMC":
            continue
        return {atom.attrib["name"]: atom.attrib["type"] for atom in residue.findall("Atom")}
    raise ValueError(f"Residue DMC not found in {xml_path}")


def _force_section_atom_map(root: ET.Element, section_name: str) -> dict[str, dict[str, float]]:
    section = root.find(section_name)
    if section is None:
        raise ValueError(f"Missing <{section_name}> in XML")
    out: dict[str, dict[str, float]] = {}
    for atom in section.findall("Atom"):
        out[atom.attrib["type"]] = {
            key: float(value)
            for key, value in atom.attrib.items()
            if key != "type"
        }
    return out


def _load_drude_atom_map(path: Path) -> dict[str, dict]:
    payload = load_json(path)
    atoms = payload["atoms"]
    return {atom["name"]: atom for atom in atoms}


def load_base_real_slater_parameters(
    drude_model_json: Path = DEFAULT_BASE_DRUDE_MODEL_JSON,
    xml_path: Path = DEFAULT_BASE_SLATER_XML,
    pdb_path: Path = DEFAULT_DMC_PDB,
) -> dict[str, np.ndarray]:
    graph = load_monomer_graph(pdb_path)
    root = ET.parse(xml_path).getroot()
    type_map = _load_dmc_type_map(xml_path)
    drude_atoms = _load_drude_atom_map(drude_model_json)

    ex_map = _force_section_atom_map(root, "SlaterExForce")
    sr_es_map = _force_section_atom_map(root, "SlaterSrEsForce")
    sr_pol_map = _force_section_atom_map(root, "SlaterSrPolForce")
    sr_disp_map = _force_section_atom_map(root, "SlaterSrDispForce")
    dhf_map = _force_section_atom_map(root, "SlaterDhfForce")
    damp_map = _force_section_atom_map(root, "SlaterDampingForce")

    charges = []
    alpha = []
    thole = []
    slater_b = []
    ex_a = []
    sr_es_a = []
    sr_pol_a = []
    sr_disp_a = []
    dhf_a = []
    c6 = []
    c8 = []
    c10 = []
    for atom_name in graph.atom_names:
        atom_type = type_map[atom_name]
        drude = drude_atoms[atom_name]
        charges.append(float(drude["charge"]))
        alpha.append(float(drude["alpha"]))
        thole.append(float(drude["thole"]))
        slater_b.append(float(ex_map[atom_type]["B"]))
        ex_a.append(float(ex_map[atom_type]["A"]))
        sr_es_a.append(float(sr_es_map[atom_type]["A"]))
        sr_pol_a.append(float(sr_pol_map[atom_type]["A"]))
        sr_disp_a.append(float(sr_disp_map[atom_type]["A"]))
        dhf_a.append(float(dhf_map[atom_type]["A"]))
        c6.append(float(damp_map[atom_type]["C6"]))
        c8.append(float(damp_map[atom_type]["C8"]))
        c10.append(float(damp_map[atom_type]["C10"]))

    charges_np = np.asarray(charges, dtype=np.float32)
    return {
        "charge": charges_np,
        "alpha": np.asarray(alpha, dtype=np.float32),
        "thole": np.asarray(thole, dtype=np.float32),
        "slater_b": np.asarray(slater_b, dtype=np.float32),
        "slater_ex_a": np.asarray(ex_a, dtype=np.float32),
        "slater_sr_es_a": np.asarray(sr_es_a, dtype=np.float32),
        "slater_sr_pol_a": np.asarray(sr_pol_a, dtype=np.float32),
        "slater_sr_disp_a": np.asarray(sr_disp_a, dtype=np.float32),
        "slater_dhf_a": np.asarray(dhf_a, dtype=np.float32),
        "disp_c6": np.asarray(c6, dtype=np.float32),
        "disp_c8": np.asarray(c8, dtype=np.float32),
        "disp_c10": np.asarray(c10, dtype=np.float32),
        "total_charge": np.asarray(np.sum(charges_np), dtype=np.float32),
    }


def predict_real_slater_parameters(
    model_params: dict,
    graph,
    base_params: dict[str, np.ndarray],
) -> dict[str, jax.Array]:
    features = jnp.asarray(build_feature_matrix(graph))
    pair_bins = jnp.asarray(np.clip(graph.graph_distance, 0, 6))
    hidden = encode_graph(model_params, features, pair_bins)
    raw = hidden @ model_params["head_w"] + features @ model_params["head_local_w"] + model_params["head_b"]
    base = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in base_params.items()}

    charge = base["charge"] + 0.12 * jnp.tanh(raw[:, 0])
    charge = charge - (jnp.sum(charge) - base["total_charge"]) / charge.shape[0]
    alpha = jnp.maximum(base["alpha"] * jnp.exp(0.50 * jnp.tanh(raw[:, 1])), MIN_POSITIVE)
    thole = jnp.clip(base["thole"] * jnp.exp(0.40 * jnp.tanh(raw[:, 2])), 0.2, 4.0)
    slater_b = jnp.maximum(base["slater_b"] * jnp.exp(0.25 * jnp.tanh(raw[:, 3])), MIN_POSITIVE)
    slater_ex_a = jnp.maximum(base["slater_ex_a"] * jnp.exp(0.90 * jnp.tanh(raw[:, 4])), MIN_POSITIVE)
    slater_sr_es_a = jnp.maximum(base["slater_sr_es_a"] * jnp.exp(0.90 * jnp.tanh(raw[:, 5])), MIN_POSITIVE)
    slater_sr_pol_a = jnp.maximum(base["slater_sr_pol_a"] * jnp.exp(0.90 * jnp.tanh(raw[:, 6])), MIN_POSITIVE)
    slater_sr_disp_a = jnp.maximum(base["slater_sr_disp_a"] * jnp.exp(0.90 * jnp.tanh(raw[:, 7])), MIN_POSITIVE)
    slater_dhf_a = jnp.maximum(base["slater_dhf_a"] * jnp.exp(0.90 * jnp.tanh(raw[:, 8])), MIN_POSITIVE)
    disp_c6 = jnp.maximum(base["disp_c6"] * jnp.exp(0.70 * jnp.tanh(raw[:, 9])), MIN_POSITIVE)
    disp_c8 = jnp.maximum(base["disp_c8"] * jnp.exp(0.70 * jnp.tanh(raw[:, 10])), MIN_POSITIVE)
    disp_c10 = jnp.maximum(base["disp_c10"] * jnp.exp(0.70 * jnp.tanh(raw[:, 11])), MIN_POSITIVE)
    return {
        "charge": charge,
        "alpha": alpha,
        "thole": thole,
        "slater_b": slater_b,
        "slater_ex_a": slater_ex_a,
        "slater_sr_es_a": slater_sr_es_a,
        "slater_sr_pol_a": slater_sr_pol_a,
        "slater_sr_disp_a": slater_sr_disp_a,
        "slater_dhf_a": slater_dhf_a,
        "disp_c6": disp_c6,
        "disp_c8": disp_c8,
        "disp_c10": disp_c10,
    }


def _pair_sqrt(param: jax.Array) -> jax.Array:
    return jnp.sqrt(param[:, None] * param[None, :])


def _slater_prefactor(a_i: jax.Array, a_j: jax.Array, br: jax.Array) -> jax.Array:
    return a_i * a_j * (1.0 + br + br * br / 3.0) * jnp.exp(-br)


def _tt_x(br: jax.Array) -> jax.Array:
    return br - (2.0 * br * br + 3.0 * br) / (br * br + 3.0 * br + 3.0)


def _tt_poly(x: jax.Array, order: int) -> jax.Array:
    out = jnp.ones_like(x)
    term = jnp.ones_like(x)
    for n in range(1, order + 1):
        term = term * x / float(n)
        out = out + term
    return out


def evaluate_real_slater_terms(
    predicted: dict[str, jax.Array],
    distances_nm: jax.Array,
    s12_nm: float = DEFAULT_S12_NM,
) -> dict[str, jax.Array]:
    distances_nm = jnp.maximum(distances_nm, 1.0e-6)
    distances_ang = distances_nm * 10.0
    inv_r_nm = 1.0 / distances_nm
    inv_r2_nm = inv_r_nm * inv_r_nm
    inv_r4_nm = inv_r2_nm * inv_r2_nm
    inv_r6_nm = inv_r4_nm * inv_r2_nm
    inv_r8_nm = inv_r4_nm * inv_r4_nm
    inv_r10_nm = inv_r8_nm * inv_r2_nm

    q_a = predicted["charge"][None, :, None]
    q_b = predicted["charge"][None, None, :]
    alpha_a = predicted["alpha"][None, :, None]
    alpha_b = predicted["alpha"][None, None, :]
    thole = 0.5 * (predicted["thole"][None, :, None] + predicted["thole"][None, None, :])
    b_pair = _pair_sqrt(predicted["slater_b"])
    br = b_pair[None, :, :] * distances_nm
    damp_short = jnp.exp(-0.35 * thole * distances_ang)

    lr_es_pair = ONE_4PI_EPS0 * q_a * q_b * inv_r_nm
    lr_pol_pair = -0.5 * ONE_4PI_EPS0 * (alpha_a * (q_b**2) + alpha_b * (q_a**2)) * inv_r4_nm * (1.0 - damp_short)

    qqtt_pair = -0.1 * SHORT_RANGE_DIELECTRIC * q_a * q_b * jnp.exp(-br) * (1.0 + br) * inv_r_nm
    slater_sr_es_pair = -_slater_prefactor(
        predicted["slater_sr_es_a"][None, :, None],
        predicted["slater_sr_es_a"][None, None, :],
        br,
    )
    slater_sr_pol_pair = -_slater_prefactor(
        predicted["slater_sr_pol_a"][None, :, None],
        predicted["slater_sr_pol_a"][None, None, :],
        br,
    )
    slater_ex_pair = _slater_prefactor(
        predicted["slater_ex_a"][None, :, None],
        predicted["slater_ex_a"][None, None, :],
        br,
    ) + (s12_nm * inv_r_nm) ** 12
    slater_sr_disp_pair = -_slater_prefactor(
        predicted["slater_sr_disp_a"][None, :, None],
        predicted["slater_sr_disp_a"][None, None, :],
        br,
    )
    slater_dhf_pair = -_slater_prefactor(
        predicted["slater_dhf_a"][None, :, None],
        predicted["slater_dhf_a"][None, None, :],
        br,
    )

    x = _tt_x(br)
    exp_neg_x = jnp.exp(-x)
    c6_pair = _pair_sqrt(predicted["disp_c6"])[None, :, :]
    c8_pair = _pair_sqrt(predicted["disp_c8"])[None, :, :]
    c10_pair = _pair_sqrt(predicted["disp_c10"])[None, :, :]
    damped_dispersion_pair = (
        -(1.0 - exp_neg_x * _tt_poly(x, 6)) * c6_pair * inv_r6_nm
        -(1.0 - exp_neg_x * _tt_poly(x, 8)) * c8_pair * inv_r8_nm
        -(1.0 - exp_neg_x * _tt_poly(x, 10)) * c10_pair * inv_r10_nm
    )

    terms = {
        "lr_es_kj_mol": jnp.sum(lr_es_pair, axis=(1, 2)),
        "lr_pol_kj_mol": jnp.sum(lr_pol_pair, axis=(1, 2)),
        "qqtt_kj_mol": jnp.sum(qqtt_pair, axis=(1, 2)),
        "slater_sr_es_kj_mol": jnp.sum(slater_sr_es_pair, axis=(1, 2)),
        "slater_sr_pol_kj_mol": jnp.sum(slater_sr_pol_pair, axis=(1, 2)),
        "slater_ex_kj_mol": jnp.sum(slater_ex_pair, axis=(1, 2)),
        "damped_dispersion_kj_mol": jnp.sum(damped_dispersion_pair, axis=(1, 2)),
        "slater_sr_disp_kj_mol": jnp.sum(slater_sr_disp_pair, axis=(1, 2)),
        "slater_dhf_kj_mol": jnp.sum(slater_dhf_pair, axis=(1, 2)),
    }
    terms["sr_es_total_kj_mol"] = terms["qqtt_kj_mol"] + terms["slater_sr_es_kj_mol"]
    terms["sr_pol_total_kj_mol"] = terms["slater_sr_pol_kj_mol"]
    terms["exchange_kj_mol"] = terms["slater_ex_kj_mol"]
    terms["dispersion_total_kj_mol"] = terms["damped_dispersion_kj_mol"] + terms["slater_sr_disp_kj_mol"]
    terms["ct_like_kj_mol"] = terms["slater_dhf_kj_mol"]
    terms["target_lr_espol_kj_mol"] = terms["lr_es_kj_mol"] + terms["lr_pol_kj_mol"]
    terms["target_total_espol_kj_mol"] = (
        terms["target_lr_espol_kj_mol"]
        + terms["sr_es_total_kj_mol"]
        + terms["sr_pol_total_kj_mol"]
    )
    terms["total_nonbonded_kj_mol"] = (
        terms["target_total_espol_kj_mol"]
        + terms["exchange_kj_mol"]
        + terms["dispersion_total_kj_mol"]
        + terms["ct_like_kj_mol"]
    )
    return terms


def evaluate_real_monomer_properties(predicted: dict[str, jax.Array], graph) -> dict[str, jax.Array]:
    positions_nm = jnp.asarray(graph.positions_nm, dtype=jnp.float32)
    dipole_debye = jnp.sum(predicted["charge"][:, None] * positions_nm, axis=0) * E_NM_TO_DEBYE
    isotropic_polar_nm3 = jnp.sum(predicted["alpha"])
    return {
        "dipole_debye": dipole_debye,
        "isotropic_polarizability_nm3": isotropic_polar_nm3,
    }


def load_predicted_real_slater_parameters_from_model_json(path: Path) -> dict[str, np.ndarray]:
    payload = load_json(path)
    keys = (
        "charge",
        "alpha",
        "thole",
        "slater_b",
        "slater_ex_a",
        "slater_sr_es_a",
        "slater_sr_pol_a",
        "slater_sr_disp_a",
        "slater_dhf_a",
        "disp_c6",
        "disp_c8",
        "disp_c10",
    )
    return {
        key: np.asarray([float(atom[key]) for atom in payload["atom_parameters"]], dtype=np.float32)
        for key in keys
    }


def physical_term_mapping() -> dict[str, str]:
    return {
        "lr_es_kj_mol": "Drude long-range electrostatics",
        "lr_pol_kj_mol": "Drude long-range polarization",
        "qqtt_kj_mol": "QqTtDampingForce",
        "slater_sr_es_kj_mol": "SlaterSrEsForce",
        "sr_es_total_kj_mol": "QqTtDampingForce + SlaterSrEsForce",
        "slater_sr_pol_kj_mol": "SlaterSrPolForce",
        "sr_pol_total_kj_mol": "SlaterSrPolForce",
        "slater_ex_kj_mol": "SlaterExForce",
        "exchange_kj_mol": "SlaterExForce",
        "damped_dispersion_kj_mol": "ADMPDispPmeForce + SlaterDampingForce",
        "slater_sr_disp_kj_mol": "SlaterSrDispForce",
        "dispersion_total_kj_mol": "ADMPDispPmeForce + SlaterDampingForce + SlaterSrDispForce",
        "slater_dhf_kj_mol": "SlaterDhfForce",
        "ct_like_kj_mol": "SlaterDhfForce",
        "total_nonbonded_kj_mol": "lr_es + lr_pol + sr_es_total + sr_pol_total + exchange + dispersion_total + ct_like",
    }
