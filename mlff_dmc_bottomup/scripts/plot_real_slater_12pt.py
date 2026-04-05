from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import DEFAULT_FULL_TARGETS, ROOT, dimer_distance_tensors, write_json
from real_slater_common import (
    load_predicted_real_slater_parameters_from_model_json,
    evaluate_real_slater_terms,
    physical_term_mapping,
)


DEFAULT_MODEL_JSON = ROOT / "output" / "train_real_slater_12pt" / "dmc_mlff_real_slater_12pt_model.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "plot_real_slater_12pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 12-point DMC-DMC scans using the real PhyNEO Slater/QqTt/dispersion functional form.")
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--target-npz", default=str(DEFAULT_FULL_TARGETS))
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="dmc_mlff_real_slater_12pt")
    return parser.parse_args()


def write_curve_csv(path: Path, shifts: np.ndarray, subset: dict[str, np.ndarray], predicted: dict[str, np.ndarray]) -> None:
    fields = [
        "shift_angstrom",
        "target_lr_espol_kj_mol",
        "predicted_lr_espol_kj_mol",
        "target_total_nonbonded_kj_mol",
        "predicted_total_nonbonded_kj_mol",
        "predicted_qqtt_kj_mol",
        "predicted_slater_sr_es_kj_mol",
        "predicted_sr_es_total_kj_mol",
        "predicted_slater_sr_pol_kj_mol",
        "predicted_exchange_kj_mol",
        "predicted_damped_dispersion_kj_mol",
        "predicted_slater_sr_disp_kj_mol",
        "predicted_dispersion_total_kj_mol",
        "predicted_ct_like_kj_mol",
    ]
    with open(path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index, shift in enumerate(shifts):
            row = [
                shift,
                subset["target_lr_espol_kj_mol"][index],
                predicted["target_lr_espol_kj_mol"][index],
                subset["total_nonbonded_kj_mol"][index],
                predicted["total_nonbonded_kj_mol"][index],
                predicted["qqtt_kj_mol"][index],
                predicted["slater_sr_es_kj_mol"][index],
                predicted["sr_es_total_kj_mol"][index],
                predicted["slater_sr_pol_kj_mol"][index],
                predicted["exchange_kj_mol"][index],
                predicted["damped_dispersion_kj_mol"][index],
                predicted["slater_sr_disp_kj_mol"][index],
                predicted["dispersion_total_kj_mol"][index],
                predicted["ct_like_kj_mol"][index],
            ]
            writer.writerow([f"{float(value):.8f}" for value in row])


def plot_curve(path: Path, shifts: np.ndarray, subset: dict[str, np.ndarray], predicted: dict[str, np.ndarray]) -> None:
    order = np.argsort(shifts)
    x = shifts[order]

    target_espol = subset["target_lr_espol_kj_mol"][order]
    pred_espol = predicted["target_lr_espol_kj_mol"][order]
    target_total = subset["total_nonbonded_kj_mol"][order]
    pred_total = predicted["total_nonbonded_kj_mol"][order]

    qqtt = predicted["qqtt_kj_mol"][order]
    slater_sr_es = predicted["slater_sr_es_kj_mol"][order]
    sr_pol = predicted["sr_pol_total_kj_mol"][order]
    exchange = predicted["exchange_kj_mol"][order]
    dispersion = predicted["dispersion_total_kj_mol"][order]
    ct_like = predicted["ct_like_kj_mol"][order]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    ax00.plot(x, target_espol, marker="o", linewidth=1.8, label="SAPT target (lr es+pol)")
    ax00.plot(x, pred_espol, marker="s", linewidth=1.8, label="Real-Slater MLFF")
    ax00.set_ylabel("Interaction Energy (kJ/mol)")
    ax00.set_title("DMC-DMC 12-point Long-Range es+pol")
    ax00.legend()

    ax10.plot(x, pred_espol - target_espol, marker="s", linewidth=1.6, label="MLFF - target")
    ax10.axhline(0.0, color="black", linewidth=1.0)
    ax10.set_xlabel("Scan Coordinate (angstrom)")
    ax10.set_ylabel("Error (kJ/mol)")
    ax10.legend()

    ax01.plot(x, target_total, marker="o", linewidth=1.8, label="SAPT total nonbonded")
    ax01.plot(x, pred_total, marker="s", linewidth=1.8, label="Real-Slater MLFF total")
    ax01.set_ylabel("Interaction Energy (kJ/mol)")
    ax01.set_title("DMC-DMC 12-point Total Nonbonded")
    ax01.legend()

    ax11.plot(x, qqtt, linewidth=1.5, label="QqTt")
    ax11.plot(x, slater_sr_es, linewidth=1.5, label="SlaterSrEs")
    ax11.plot(x, sr_pol, linewidth=1.5, label="SlaterSrPol")
    ax11.plot(x, exchange, linewidth=1.5, label="SlaterEx")
    ax11.plot(x, dispersion, linewidth=1.5, label="Disp total")
    ax11.plot(x, ct_like, linewidth=1.5, label="SlaterDhf")
    ax11.set_xlabel("Scan Coordinate (angstrom)")
    ax11.set_ylabel("Energy (kJ/mol)")
    ax11.set_title("Predicted Force-Term Decomposition")
    ax11.legend(ncol=2, fontsize=9)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = {key: np.asarray(value) for key, value in np.load(args.target_npz).items()}
    mask = np.asarray(bundle["batch_index"], dtype=int) == args.batch_index
    if not np.any(mask):
        raise ValueError(f"batch_index={args.batch_index} not present in {args.target_npz}")
    subset = {
        key: value[mask]
        for key, value in bundle.items()
        if getattr(value, "ndim", 0) >= 1 and value.shape[0] == mask.shape[0]
    }
    shifts = np.asarray(subset["shift_angstrom"], dtype=float)

    predicted_params = load_predicted_real_slater_parameters_from_model_json(Path(args.model_json))
    tensors = dimer_distance_tensors(subset)
    predicted = {
        key: np.asarray(value, dtype=float)
        for key, value in evaluate_real_slater_terms(predicted_params, tensors["distance_nm"]).items()
    }

    csv_path = output_dir / f"{args.prefix}.csv"
    png_path = output_dir / f"{args.prefix}.png"
    json_path = output_dir / f"{args.prefix}_summary.json"
    write_curve_csv(csv_path, shifts, subset, predicted)
    plot_curve(png_path, shifts, subset, predicted)

    summary = {
        "description": "12-point DMC-DMC scan using the real PhyNEO Slater/QqTt/dispersion functional form.",
        "model_json": str(Path(args.model_json).resolve()),
        "target_npz": str(Path(args.target_npz).resolve()),
        "batch_index": int(args.batch_index),
        "num_points": int(shifts.shape[0]),
        "physical_term_mapping": physical_term_mapping(),
        "lr_espol_rmse_kj_mol": float(np.sqrt(np.mean(np.square(predicted["target_lr_espol_kj_mol"] - subset["target_lr_espol_kj_mol"])))),
        "total_nonbonded_rmse_kj_mol": float(np.sqrt(np.mean(np.square(predicted["total_nonbonded_kj_mol"] - subset["total_nonbonded_kj_mol"])))),
    }
    write_json(json_path, summary)
    print(json_path)
    print(png_path)


if __name__ == "__main__":
    main()
