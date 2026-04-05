from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from common import (
    DEFAULT_DMC_JSON,
    DEFAULT_DMC_PDB,
    DEFAULT_FULL_TARGETS,
    DEFAULT_MONOMER_TARGETS,
    OUTPUT,
    ensure_dir,
    evaluate_decomposed_terms,
    evaluate_monomer_properties,
    init_attention_model,
    load_base_nonbonded_parameters,
    load_monomer_graph,
    load_monomer_targets,
    parameter_summary,
    predict_parameters,
    dimer_distance_tensors,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight 2D-attention MLFF for DMC full nonbonded terms.")
    parser.add_argument("--dmc-pdb", default=str(DEFAULT_DMC_PDB))
    parser.add_argument("--base-param-json", default=str(DEFAULT_DMC_JSON))
    parser.add_argument("--target-npz", default=str(DEFAULT_FULL_TARGETS))
    parser.add_argument("--monomer-target-file", default=str(DEFAULT_MONOMER_TARGETS))
    parser.add_argument("--stage", choices=["espol", "full_nonbonded"], default="full_nonbonded")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--output-dir", default=str(OUTPUT / "train_full_nonbonded"))
    parser.add_argument("--prefix", default="dmc_mlff_full_nonbonded")
    return parser.parse_args()


def stage_term_names(stage: str) -> tuple[str, ...]:
    if stage == "espol":
        return ("lr_es_kj_mol", "lr_pol_kj_mol")
    return (
        "lr_es_kj_mol",
        "lr_pol_kj_mol",
        "sr_es_total_kj_mol",
        "sr_pol_total_kj_mol",
        "exchange_kj_mol",
        "dispersion_total_kj_mol",
        "ct_like_kj_mol",
    )


def write_curve_csv(path: Path, shifts: np.ndarray, targets: dict[str, np.ndarray], predicted: dict[str, np.ndarray], term_names: tuple[str, ...]) -> None:
    fields = ["shift_angstrom"]
    for term in term_names:
        fields.extend([f"target_{term}", f"predicted_{term}", f"error_{term}"])
    with open(path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index, shift in enumerate(shifts):
            row = [f"{float(shift):.8f}"]
            for term in term_names:
                target = float(targets[term][index])
                pred = float(predicted[term][index])
                row.extend([f"{target:.8f}", f"{pred:.8f}", f"{pred - target:.8f}"])
            writer.writerow(row)


def plot_terms_png(path: Path, shifts: np.ndarray, targets: dict[str, np.ndarray], predicted: dict[str, np.ndarray], term_names: tuple[str, ...]) -> None:
    order = np.argsort(shifts)
    x = shifts[order]
    ncols = 2
    nrows = int(np.ceil(len(term_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis, term in zip(axes.flat, term_names):
        axis.plot(x, targets[term][order], marker="o", linewidth=1.8, label="SAPT target")
        axis.plot(x, predicted[term][order], marker="s", linewidth=1.8, label="MLFF")
        axis.set_title(term)
        axis.set_xlabel("Scan Coordinate (angstrom)")
        axis.set_ylabel("Energy (kJ/mol)")
        axis.legend()
    for axis in axes.flat[len(term_names) :]:
        axis.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    graph = load_monomer_graph(Path(args.dmc_pdb))
    base_params = load_base_nonbonded_parameters(Path(args.base_param_json))
    targets_np = {key: np.asarray(value) for key, value in np.load(args.target_npz).items()}
    monomer_targets = load_monomer_targets(Path(args.monomer_target_file))
    distance_tensors = dimer_distance_tensors(targets_np)
    term_names = stage_term_names(args.stage)

    target_norm = {
        term: float(max(np.sqrt(np.mean(np.square(targets_np[term]))), 1.0))
        for term in term_names
    }
    targets = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in targets_np.items()}
    distances_nm = jnp.asarray(distance_tensors["distance_nm"], dtype=jnp.float32)
    distances_ang = jnp.asarray(distance_tensors["distance_ang"], dtype=jnp.float32)
    monomer_dipole = jnp.asarray(monomer_targets["dipole_debye"], dtype=jnp.float32)
    monomer_iso = jnp.asarray(monomer_targets["isotropic_polarizability_nm3"], dtype=jnp.float32)
    base_params_jax = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in base_params.items()}

    rng = jax.random.PRNGKey(args.seed)
    model_params = init_attention_model(
        rng,
        input_dim=int(build_feature_dim(graph)),
        num_pair_bins=7,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(model_params)
    history: list[dict[str, float]] = []

    def loss_fn(params):
        predicted = predict_parameters(params, graph, base_params)
        terms = evaluate_decomposed_terms(predicted, distances_nm, distances_ang)
        monomer = evaluate_monomer_properties(predicted, graph)
        components = {}
        value = jnp.asarray(0.0, dtype=jnp.float32)
        for term in term_names:
            term_rmse = jnp.sqrt(jnp.mean(jnp.square(terms[term] - targets[term]))) / target_norm[term]
            components[f"{term}_rmse"] = term_rmse
            value = value + term_rmse
        dipole_rmse = jnp.linalg.norm(monomer["dipole_debye"] - monomer_dipole) / jnp.maximum(jnp.linalg.norm(monomer_dipole), 1.0e-6)
        iso_polar_rmse = jnp.abs(monomer["isotropic_polarizability_nm3"] - monomer_iso) / jnp.maximum(jnp.abs(monomer_iso), 1.0e-6)
        components["monomer_dipole_rmse"] = dipole_rmse
        components["monomer_iso_polar_rmse"] = iso_polar_rmse
        value = value + 0.35 * dipole_rmse + 0.35 * iso_polar_rmse

        charge_reg = jnp.mean(jnp.square(predicted["charge"] - base_params_jax["charge"])) / 0.01
        alpha_reg = jnp.mean(jnp.square(jnp.log(predicted["alpha"] / base_params_jax["alpha"])))
        thole_reg = jnp.mean(jnp.square(jnp.log(predicted["thole"] / base_params_jax["thole"])))
        short_reg = (
            jnp.mean(jnp.square(jnp.log(predicted["rep_eps"] / base_params_jax["rep_eps"])))
            + jnp.mean(jnp.square(jnp.log(predicted["rep_lamb"] / base_params_jax["rep_lamb"])))
            + jnp.mean(jnp.square(jnp.log(predicted["c6"] / base_params_jax["c6"])))
            + jnp.mean(jnp.square(jnp.log(predicted["ct_eps"] / base_params_jax["ct_eps"])))
            + jnp.mean(jnp.square(jnp.log(predicted["ct_lamb"] / base_params_jax["ct_lamb"])))
        )
        regularization = 0.10 * charge_reg + 0.05 * alpha_reg + 0.05 * thole_reg + 0.02 * short_reg
        components["regularization"] = regularization
        value = value + regularization
        return value, {"components": components, "predicted": predicted, "terms": terms, "monomer": monomer}

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    for step in range(args.steps):
        (loss_value, aux), grads = value_and_grad(model_params)
        updates, opt_state = optimizer.update(grads, opt_state, model_params)
        model_params = optax.apply_updates(model_params, updates)
        if step % args.log_interval == 0 or step == args.steps - 1:
            record = {"step": float(step), "loss": float(loss_value)}
            for key, item in aux["components"].items():
                record[key] = float(item)
            history.append(record)
            print(json.dumps(record, sort_keys=True))

    final_loss, final_aux = loss_fn(model_params)
    predicted_np = {key: np.asarray(value) for key, value in final_aux["predicted"].items()}
    terms_np = {key: np.asarray(value) for key, value in final_aux["terms"].items()}
    monomer_np = {key: np.asarray(value) for key, value in final_aux["monomer"].items()}

    model_json = {
        "description": "JAX 2D-attention DMC MLFF parameter prediction output.",
        "stage": args.stage,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "atom_parameters": parameter_summary(predicted_np, graph.atom_names),
        "loss_components": {key: float(value) for key, value in final_aux["components"].items()},
    }
    summary_json = {
        "description": "Training summary for the DMC 2D-attention MLFF bottom-up model.",
        "stage": args.stage,
        "seed": args.seed,
        "steps": args.steps,
        "final_loss": float(final_loss),
        "term_rmse_normalized": {key: float(final_aux["components"][f"{key}_rmse"]) for key in term_names},
        "monomer_targets": {
            "dipole_debye": np.asarray(monomer_targets["dipole_debye"], dtype=float).tolist(),
            "isotropic_polarizability_nm3": float(monomer_targets["isotropic_polarizability_nm3"]),
        },
        "monomer_predicted": {
            "dipole_debye": monomer_np["dipole_debye"].tolist(),
            "isotropic_polarizability_nm3": float(monomer_np["isotropic_polarizability_nm3"]),
        },
        "training_history": history,
    }

    prefix = args.prefix
    write_json(output_dir / f"{prefix}_model.json", model_json)
    write_json(output_dir / f"{prefix}_summary.json", summary_json)
    np.savez(output_dir / f"{prefix}_weights.npz", **flatten_params(model_params))
    write_curve_csv(output_dir / f"{prefix}_curves.csv", targets_np["shift_angstrom"], targets_np, terms_np, term_names)
    plot_terms_png(output_dir / f"{prefix}_curves.png", targets_np["shift_angstrom"], targets_np, terms_np, term_names)


def build_feature_dim(graph) -> int:
    return len(graph.atom_names) + 5 + 5 + 1


def flatten_params(params: dict, prefix: str = "") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=f"{name}/"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                flat.update(flatten_params(item, prefix=f"{name}/{index}/"))
        else:
            flat[name] = np.asarray(value)
    return flat


if __name__ == "__main__":
    main()
