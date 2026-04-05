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
    parser = argparse.ArgumentParser(description="Plot the 12-point DMC-DMC lr es+pol scan for the fitted MLFF model.")
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--target-npz", default=str(DEFAULT_FULL_TARGETS))
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="dmc_mlff_12pt_lr_espol")
    return parser.parse_args()


def write_curve_csv(path: Path, shifts: np.ndarray, target: np.ndarray, predicted: np.ndarray) -> None:
    with open(path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shift_angstrom", "target_lr_espol_kj_mol", "predicted_lr_espol_kj_mol", "error_kj_mol"])
        for row in zip(shifts, target, predicted, predicted - target):
            writer.writerow([f"{float(value):.8f}" for value in row])


def plot_curve(path: Path, shifts: np.ndarray, target: np.ndarray, predicted: np.ndarray) -> None:
    order = np.argsort(shifts)
    x = shifts[order]
    target_ord = target[order]
    pred_ord = predicted[order]
    err_ord = pred_ord - target_ord

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax0.plot(x, target_ord, marker="o", linewidth=1.8, label="SAPT target (lr es+pol)")
    ax0.plot(x, pred_ord, marker="s", linewidth=1.8, label="MLFF 600pt fit")
    ax0.set_ylabel("Interaction Energy (kJ/mol)")
    ax0.set_title("DMC-DMC 12-point Long-Range es+pol Scan")
    ax0.legend()

    ax1.plot(x, err_ord, marker="s", linewidth=1.6, label="MLFF - target")
    ax1.axhline(0.0, color="black", linewidth=1.0)
    ax1.set_xlabel("Scan Coordinate (angstrom)")
    ax1.set_ylabel("Error (kJ/mol)")
    ax1.legend()

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
    predicted = np.asarray(terms["target_lr_espol_kj_mol"], dtype=float)
    target = np.asarray(subset["target_lr_espol_kj_mol"], dtype=float)
    shifts = np.asarray(subset["shift_angstrom"], dtype=float)

    rmse = float(np.sqrt(np.mean(np.square(predicted - target))))
    mae = float(np.mean(np.abs(predicted - target)))
    max_abs = float(np.max(np.abs(predicted - target)))

    csv_path = output_dir / f"{args.prefix}.csv"
    png_path = output_dir / f"{args.prefix}.png"
    json_path = output_dir / f"{args.prefix}_summary.json"

    write_curve_csv(csv_path, shifts, target, predicted)
    plot_curve(png_path, shifts, target, predicted)
    write_json(
        json_path,
        {
            "description": "12-point DMC-DMC lr es+pol scan for the fitted MLFF model.",
            "model_json": str(Path(args.model_json).resolve()),
            "target_npz": str(Path(args.target_npz).resolve()),
            "batch_index": int(args.batch_index),
            "num_points": int(shifts.shape[0]),
            "rmse_kj_mol": rmse,
            "mae_kj_mol": mae,
            "max_abs_kj_mol": max_abs,
        },
    )
    print(json_path)
    print(png_path)


if __name__ == "__main__":
    main()
