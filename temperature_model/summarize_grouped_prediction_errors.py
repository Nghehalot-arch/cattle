from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import write_csv


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in ("temperature_f", "prediction_f", "error_f"):
                parsed[key] = float(parsed[key])
            parsed["abs_error_f"] = abs(float(parsed["error_f"]))
            rows.append(parsed)
    return rows


def rmse(errors: list[float]) -> float:
    values = np.asarray(errors, dtype=np.float32)
    return float(math.sqrt(float(np.mean(values * values))))


def sequence_key(row: dict[str, object]) -> str:
    return f"{row['date']}/{row['sequence_num']}"


def summarize_sequences(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[sequence_key(row)].append(row)

    summaries = []
    for key, items in grouped.items():
        errors = [float(row["error_f"]) for row in items]
        abs_errors = [abs(error) for error in errors]
        first = items[0]
        summaries.append(
            {
                "sequence": key,
                "cow_tag": first.get("cow_tag", ""),
                "temperature_f": first["temperature_f"],
                "prediction_count": len(items),
                "mae_f": float(np.mean(abs_errors)),
                "rmse_f": rmse(errors),
                "max_abs_error_f": float(np.max(abs_errors)),
                "mean_error_f": float(np.mean(errors)),
                "groupings": ";".join(sorted({str(row.get("grouping", "")) for row in items})),
            }
        )
    summaries.sort(key=lambda row: row["mae_f"], reverse=True)
    return summaries


def summarize_folds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("grouping", ""), row.get("fold", ""))].append(row)

    summaries = []
    for (grouping, fold), items in grouped.items():
        errors = [float(row["error_f"]) for row in items]
        abs_errors = [abs(error) for error in errors]
        summaries.append(
            {
                "grouping": grouping,
                "fold": fold,
                "sample_count": len(items),
                "mae_f": float(np.mean(abs_errors)),
                "rmse_f": rmse(errors),
                "max_abs_error_f": float(np.max(abs_errors)),
                "test_sequences": ";".join(sequence_key(row) for row in items),
            }
        )
    summaries.sort(key=lambda row: (str(row["grouping"]), -float(row["mae_f"])))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize high-error sequences and folds from grouped CV predictions.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top", default=10, type=int)
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    sequence_rows = summarize_sequences(rows)
    fold_rows = summarize_folds(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sequence_error_summary.csv", sequence_rows)
    write_csv(args.output_dir / "fold_error_summary.csv", fold_rows)

    print("Top sequence errors:")
    for row in sequence_rows[: args.top]:
        print(
            f"{row['sequence']} cow={row['cow_tag']} "
            f"MAE={row['mae_f']:.3f} RMSE={row['rmse_f']:.3f} "
            f"truth={float(row['temperature_f']):.1f}"
        )


if __name__ == "__main__":
    main()
