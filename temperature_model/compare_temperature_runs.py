from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_RUNS = [
    "raw_baseline_v1",
    "detected_roi_v1",
    "combined_v1",
    "detected_roi_filtered_v1",
    "combined_filtered_v1",
    "detected_roi_filtered_80_v1",
    "combined_filtered_80_v1",
]


def load_summary(run_dir: Path) -> dict[str, str]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return {}

    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare saved temperature regression runs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/temperature_outputs"),
        help="Directory containing temperature run output folders.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=DEFAULT_RUNS,
        help="Run folder names to compare.",
    )
    args = parser.parse_args()

    rows = []
    for run_name in args.runs:
        run_dir = args.base_dir / run_name
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text())
        summary = load_summary(run_dir)
        test = metrics.get("random_forest_test", {})
        kfold = metrics.get("kfold_random_forest", {})
        rows.append(
            {
                "run": run_name,
                "samples": metrics.get("sample_count"),
                "features": metrics.get("feature_count"),
                "kept_frames": summary.get("frame_detection_count", "-"),
                "quality_skipped": summary.get("skipped_quality", "-"),
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "kfold_mae": kfold.get("mae_mean"),
                "kfold_rmse": kfold.get("rmse_mean"),
            }
        )

    if not rows:
        print("No metrics found.")
        return

    headers = [
        "run",
        "samples",
        "features",
        "kept_frames",
        "quality_skipped",
        "test_mae",
        "test_rmse",
        "kfold_mae",
        "kfold_rmse",
    ]
    widths = {
        header: max(len(header), *(len(fmt(row[header])) for row in rows))
        for header in headers
    }

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(
            " | ".join(
                fmt(row[header]).ljust(widths[header]) for header in headers
            )
        )


if __name__ == "__main__":
    main()
