from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    DEFAULT_FULL_TARGETS,
    ROOT,
    evaluate_decomposed_terms,
    dimer_distance_tensors,
    load_predicted_parameters_from_model_json,
    write_json,
)


DEFAULT_MODEL_JSON = ROOT / "output" / "train_600pts_v2" / "dmc_mlff_600pts_v2_model.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "plot_12pt_espol"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the 12-point DMC-DMC lr es+pol and total nonbonded scans for the fitted MLFF model.")
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--target-npz", default=str(DEFAULT_FULL_TARGETS))
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="dmc_mlff_12pt_lr_espol")
    return parser.parse_args()


def write_curve_csv(
    path: Path,
    shifts: np.ndarray,
    target_espol: np.ndarray,
    predicted_espol: np.ndarray,
    target_total: np.ndarray,
    predicted_total: np.ndarray,
) -> None:
    with open(path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shift_angstrom",
                "target_lr_espol_kj_mol",
                "predicted_lr_espol_kj_mol",
                "lr_espol_error_kj_mol",
                "target_total_nonbonded_kj_mol",
                "predicted_total_nonbonded_kj_mol",
                "total_nonbonded_error_kj_mol",
            ]
        )
        for row in zip(
            shifts,
            target_espol,
            predicted_espol,
            predicted_espol - target_espol,
            target_total,
            predicted_total,
            predicted_total - target_total,
        ):
            writer.writerow([f"{float(value):.8f}" for value in row])


def plot_curve(
    path: Path,
    shifts: np.ndarray,
    target_espol: np.ndarray,
    predicted_espol: np.ndarray,
    target_total: np.ndarray,
    predicted_total: np.ndarray,
) -> None:
    order = np.argsort(shifts)
    x = shifts[order]
    target_espol_ord = target_espol[order]
    pred_espol_ord = predicted_espol[order]
    err_espol_ord = pred_espol_ord - target_espol_ord
    target_total_ord = target_total[order]
    pred_total_ord = predicted_total[order]
    err_total_ord = pred_total_ord - target_total_ord

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    ax00.plot(x, target_espol_ord, marker="o", linewidth=1.8, label="SAPT target (lr es+pol)")
    ax00.plot(x, pred_espol_ord, marker="s", linewidth=1.8, label="MLFF 600pt fit")
    ax00.set_ylabel("Interaction Energy (kJ/mol)")
    ax00.set_title("DMC-DMC 12-point Long-Range es+pol")
    ax00.legend()

    ax10.plot(x, err_espol_ord, marker="s", linewidth=1.6, label="MLFF - target")
    ax10.axhline(0.0, color="black", linewidth=1.0)
    ax10.set_xlabel("Scan Coordinate (angstrom)")
    ax10.set_ylabel("Error (kJ/mol)")
    ax10.legend()

    ax01.plot(x, target_total_ord, marker="o", linewidth=1.8, label="SAPT total nonbonded")
    ax01.plot(x, pred_total_ord, marker="s", linewidth=1.8, label="MLFF total nonbonded")
    ax01.set_ylabel("Interaction Energy (kJ/mol)")
    ax01.set_title("DMC-DMC 12-point Total Nonbonded")
    ax01.legend()

    ax11.plot(x, err_total_ord, marker="s", linewidth=1.6, label="MLFF - target")
    ax11.axhline(0.0, color="black", linewidth=1.0)
    ax11.set_xlabel("Scan Coordinate (angstrom)")
    ax11.set_ylabel("Error (kJ/mol)")
    ax11.legend()

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
    subset = {key: value[mask] for key, value in bundle.items() if getattr(value, "ndim", 0) >= 1 and value.shape[0] == mask.shape[0]}

    predicted_params = load_predicted_parameters_from_model_json(Path(args.model_json))
    tensors = dimer_distance_tensors(subset)
    terms = evaluate_decomposed_terms(predicted_params, tensors["distance_nm"], tensors["distance_ang"])
    predicted_espol = np.asarray(terms["target_lr_espol_kj_mol"], dtype=float)
    target_espol = np.asarray(subset["target_lr_espol_kj_mol"], dtype=float)
    predicted_total = np.asarray(terms["total_nonbonded_kj_mol"], dtype=float)
    target_total = np.asarray(subset["total_nonbonded_kj_mol"], dtype=float)
    shifts = np.asarray(subset["shift_angstrom"], dtype=float)

    espol_rmse = float(np.sqrt(np.mean(np.square(predicted_espol - target_espol))))
    espol_mae = float(np.mean(np.abs(predicted_espol - target_espol)))
    espol_max_abs = float(np.max(np.abs(predicted_espol - target_espol)))
    total_rmse = float(np.sqrt(np.mean(np.square(predicted_total - target_total))))
    total_mae = float(np.mean(np.abs(predicted_total - target_total)))
    total_max_abs = float(np.max(np.abs(predicted_total - target_total)))

    csv_path = output_dir / f"{args.prefix}.csv"
    png_path = output_dir / f"{args.prefix}.png"
    json_path = output_dir / f"{args.prefix}_summary.json"

    write_curve_csv(csv_path, shifts, target_espol, predicted_espol, target_total, predicted_total)
    plot_curve(png_path, shifts, target_espol, predicted_espol, target_total, predicted_total)
    write_json(
        json_path,
        {
            "description": "12-point DMC-DMC lr es+pol and total nonbonded scans for the fitted MLFF model.",
            "model_json": str(Path(args.model_json).resolve()),
            "target_npz": str(Path(args.target_npz).resolve()),
            "batch_index": int(args.batch_index),
            "num_points": int(shifts.shape[0]),
            "lr_espol_rmse_kj_mol": espol_rmse,
            "lr_espol_mae_kj_mol": espol_mae,
            "lr_espol_max_abs_kj_mol": espol_max_abs,
            "total_nonbonded_rmse_kj_mol": total_rmse,
            "total_nonbonded_mae_kj_mol": total_mae,
            "total_nonbonded_max_abs_kj_mol": total_max_abs,
        },
    )
    print(json_path)
    print(png_path)


if __name__ == "__main__":
    main()
