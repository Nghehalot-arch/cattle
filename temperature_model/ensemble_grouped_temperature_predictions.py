from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from common import write_csv


def sequence_key(row: dict[str, str]) -> str:
    return f"{row['date']}/{row['sequence_num']}"


def read_predictions(path: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouping = row.get("grouping", "")
            fold = row.get("fold", "")
            key = sequence_key(row)
            rows[(grouping, fold, key)] = {
                "grouping": grouping,
                "fold": fold,
                "date": row["date"],
                "sequence_num": row["sequence_num"],
                "cow_tag": row.get("cow_tag", ""),
                "temperature_f": float(row["temperature_f"]),
                "prediction_f": float(row["prediction_f"]),
            }
    return rows


def read_prediction_spec(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, object]]:
    merged = {}
    for path in paths:
        for key, row in read_predictions(path).items():
            if key in merged:
                raise RuntimeError(f"Duplicate prediction key {key} while reading {path}")
            merged[key] = row
    return merged


def metric_dict(rows: list[dict[str, object]]) -> dict[str, float]:
    truth = np.asarray([float(row["temperature_f"]) for row in rows], dtype=np.float32)
    pred = np.asarray([float(row["prediction_f"]) for row in rows], dtype=np.float32)
    errors = pred - truth
    mse = float(np.mean(errors * errors))
    metrics = {
        "mae": float(np.mean(np.abs(errors))),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
    }
    if len(rows) > 1:
        ss_res = float(np.sum(errors * errors))
        ss_tot = float(np.sum((truth - float(np.mean(truth))) ** 2))
        if ss_tot > 1e-9:
            metrics["r2"] = float(1.0 - ss_res / ss_tot)
    return metrics


def normalized_weight_grid(model_count: int, step: float):
    units = int(round(1.0 / step))
    for weights in product(range(units + 1), repeat=model_count):
        if sum(weights) != units:
            continue
        yield np.asarray(weights, dtype=np.float32) / units


def ensemble_rows(named_predictions, weights):
    common_keys = sorted(set.intersection(*[set(rows) for _, rows in named_predictions]))
    output = []
    for key in common_keys:
        base = named_predictions[0][1][key]
        truth = float(base["temperature_f"])
        preds = np.asarray([rows[key]["prediction_f"] for _, rows in named_predictions], dtype=np.float32)
        pred = float(np.dot(weights, preds))
        output.append(
            {
                "grouping": base["grouping"],
                "fold": base["fold"],
                "date": base["date"],
                "sequence_num": base["sequence_num"],
                "cow_tag": base.get("cow_tag", ""),
                "temperature_f": truth,
                "prediction_f": pred,
                "error_f": pred - truth,
            }
        )
    return output


def summarize_by_grouping(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for grouping in sorted({str(row["grouping"]) for row in rows}):
        grouping_rows = [row for row in rows if row["grouping"] == grouping]
        fold_metrics = []
        for fold in sorted({str(row["fold"]) for row in grouping_rows}, key=lambda value: int(value)):
            fold_rows = [row for row in grouping_rows if str(row["fold"]) == fold]
            fold_metrics.append(metric_dict(fold_rows))
        summaries.append(
            {
                "grouping": grouping,
                "folds": len(fold_metrics),
                "mae_mean": float(np.mean([row["mae"] for row in fold_metrics])),
                "mae_std": float(np.std([row["mae"] for row in fold_metrics])),
                "mse_mean": float(np.mean([row["mse"] for row in fold_metrics])),
                "rmse_mean": float(np.mean([row["rmse"] for row in fold_metrics])),
                "rmse_std": float(np.std([row["rmse"] for row in fold_metrics])),
            }
        )
    return summaries


def parse_named_path(value: str) -> tuple[str, list[Path]]:
    if "=" in value:
        name, path = value.split("=", 1)
        paths = [Path(item) for item in path.split("+")]
        return name, paths
    paths = [Path(item) for item in value.split("+")]
    return paths[0].parent.name, paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep weighted ensembles over grouped CV prediction CSV files.")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="name=path or name=path1+path2+path3. Repeat for each model.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--step", default=0.05, type=float)
    parser.add_argument("--weights", nargs="*", type=float)
    args = parser.parse_args()

    named_predictions = [(name, read_prediction_spec(paths)) for name, paths in map(parse_named_path, args.prediction)]
    if len(named_predictions) < 2:
        raise RuntimeError("Need at least two prediction files for an ensemble.")

    if args.weights:
        weights = np.asarray(args.weights, dtype=np.float32)
        if len(weights) != len(named_predictions):
            raise RuntimeError("--weights count must match prediction count.")
        weights = weights / float(weights.sum())
        rows = ensemble_rows(named_predictions, weights)
        weight_rows = [{"rank": 1, "weights": ",".join(f"{float(w):.3f}" for w in weights), **metric_dict(rows)}]
    else:
        weight_rows = []
        best_rows = None
        best_mae = float("inf")
        for weights in normalized_weight_grid(len(named_predictions), args.step):
            rows = ensemble_rows(named_predictions, weights)
            if not rows:
                continue
            metrics = metric_dict(rows)
            weight_rows.append({"weights": ",".join(f"{float(w):.3f}" for w in weights), **metrics})
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_rows = rows
        weight_rows.sort(key=lambda row: row["mae"])
        for rank, row in enumerate(weight_rows, start=1):
            row["rank"] = rank
        rows = best_rows or []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ensemble_predictions.csv", rows)
    write_csv(args.output_dir / "weight_sweep.csv", weight_rows)
    summary = summarize_by_grouping(rows)
    write_csv(args.output_dir / "validation_summary.csv", summary)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": [name for name, _ in named_predictions],
                "sample_count": len(rows),
                "best_weights": weight_rows[0]["weights"],
                "overall": metric_dict(rows),
                "validation": summary,
            },
            f,
            indent=2,
        )
    print("Best weights:", weight_rows[0]["weights"])
    print("Overall:", metric_dict(rows))
    for row in summary:
        print(
            f"{row['grouping']} CV:",
            f"MAE={row['mae_mean']:.3f}",
            f"MSE={row['mse_mean']:.3f}",
            f"RMSE={row['rmse_mean']:.3f}",
        )


if __name__ == "__main__":
    main()
