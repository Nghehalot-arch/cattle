from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def row_key(row: dict[str, str]) -> str:
    if row.get("sequence"):
        return row["sequence"]
    return f"{row['date']}/{row['sequence_num']}"


def read_predictions(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row_key(row)] = {
                "temperature_f": float(row["temperature_f"]),
                "prediction_f": float(row["prediction_f"]),
            }
    return rows


def metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    errors = np.asarray([row["error_f"] for row in rows], dtype=np.float32)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors * errors)),
        "rmse": float(math.sqrt(np.mean(errors * errors))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Average temperature predictions from multiple models.")
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    prediction_sets = [read_predictions(path) for path in args.inputs]
    if args.weights:
        if len(args.weights) != len(prediction_sets):
            raise RuntimeError("--weights must have the same count as --inputs.")
        weights = np.asarray(args.weights, dtype=np.float32)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise RuntimeError("--weights must sum to a positive value.")
        weights = weights / weight_sum
    else:
        weights = np.full((len(prediction_sets),), 1.0 / len(prediction_sets), dtype=np.float32)
    shared_keys = sorted(set.intersection(*(set(rows) for rows in prediction_sets)))
    if not shared_keys:
        raise RuntimeError("No shared prediction keys across inputs.")

    output_rows = []
    for key in shared_keys:
        truths = [rows[key]["temperature_f"] for rows in prediction_sets]
        if len({round(value, 6) for value in truths}) != 1:
            raise RuntimeError(f"Truth mismatch for {key}")
        preds = [rows[key]["prediction_f"] for rows in prediction_sets]
        pred = float(np.dot(weights, np.asarray(preds, dtype=np.float32)))
        truth = truths[0]
        output_rows.append(
            {
                "sequence": key,
                "temperature_f": truth,
                "prediction_f": pred,
                "error_f": pred - truth,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "temperature_f", "prediction_f", "error_f"])
        writer.writeheader()
        writer.writerows(output_rows)

    result = {
        "inputs": [str(path) for path in args.inputs],
        "weights": [float(value) for value in weights],
        "sample_count": len(output_rows),
        "test": metrics(output_rows),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Saved:", args.output_dir)
    print("MAE:", result["test"]["mae"])
    print("RMSE:", result["test"]["rmse"])


if __name__ == "__main__":
    main()
