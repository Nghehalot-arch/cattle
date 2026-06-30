from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    if value in (None, ""):
        return "-"
    return f"{float(value):.{digits}f}"


def group_count(value):
    return len([item for item in value.split("|") if item])


def temperature_range(rows, column):
    if not rows:
        return ""
    values = [float(row[column]) for row in rows]
    return f"{min(values):.1f}-{max(values):.1f}"


def grouped_fold_rows(run_root, grouping):
    fold_rows = read_csv(run_root / f"cv_{grouping}_folds.csv")
    prediction_rows = read_csv(run_root / f"cv_{grouping}_predictions.csv")
    predictions_by_fold = defaultdict(list)
    for row in prediction_rows:
        predictions_by_fold[int(row["fold"])].append(row)

    output = []
    for row in fold_rows:
        fold = int(row["fold"])
        predictions = predictions_by_fold[fold]
        output.append(
            {
                "evaluation": f"{grouping}_grouped",
                "fold": fold,
                "status": "evaluated",
                "train_group_count": group_count(row["train_groups"]),
                "test_group_count": group_count(row["test_groups"]),
                "test_sample_count": len(predictions),
                "test_groups": row["test_groups"].replace("|", ", "),
                "actual_temperature_range_f": temperature_range(predictions, "temperature_f"),
                "predicted_temperature_range_f": temperature_range(predictions, "prediction_f"),
                "mae_f": float(row["mae"]),
                "rmse_f": float(row["rmse"]),
                "r2": float(row["r2"]),
                "reason": "",
            }
        )
    return output


def paired_fold_rows(audit_path):
    output = []
    for row in read_csv(audit_path):
        output.append(
            {
                "evaluation": "paired_rgb_thermal",
                "fold": int(row["fold"].split("_")[-1]),
                "status": row["status"],
                "train_group_count": len(row["train_folders"].split()),
                "test_group_count": len(row["test_folders"].split()),
                "test_sample_count": int(row["test_pairs"]),
                "test_groups": row["test_folders"].replace(" ", ", "),
                "actual_temperature_range_f": "",
                "predicted_temperature_range_f": "",
                "mae_f": "",
                "rmse_f": "",
                "r2": "",
                "train_labeled_pairs": int(row["train_labeled_pairs"]),
                "val_labeled_pairs": int(row["val_labeled_pairs"]),
                "test_labeled_pairs": int(row["test_labeled_pairs"]),
                "reason": row["reason"],
            }
        )
    return output


def missing_video_rows(path):
    output = []
    for row in read_csv(path):
        output.append(
            {
                "video": f"{row['date']}/{row['sequence_num']}",
                "date": row["date"],
                "sequence_num": row["sequence_num"],
                "cow_tag": row["cow_tag"],
                "ground_truth_temperature_f": float(row["temperature_f"]),
                "raw_frame_count": int(row["raw_frame_count"]),
                "expected_raw_file_pattern": (
                    f"thermal_raw/{row['date']}/{row['sequence_num']}_Video_Frame_*.tiff"
                ),
                "reason": "Rectal label exists, but the complete raw thermal video/frame sequence is missing.",
            }
        )
    return output


def markdown_table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" if index == 0 else "---:" for index in range(len(headers))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")


def main():
    parser = argparse.ArgumentParser(description="Create readable fold and missing-video result tables.")
    parser.add_argument(
        "--locked-run",
        default="data/temperature_outputs/best_roi_gradient_boosting_k20_v1",
        type=Path,
    )
    parser.add_argument(
        "--paired-audit",
        default="data/temperature_outputs/paired_rgb_thermal_5fold_eval_v1/fold_audit.csv",
        type=Path,
    )
    parser.add_argument(
        "--missing-videos",
        default="data/temperature_outputs/temperature_data_audit_v1/missing_labeled_raw.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="data/temperature_outputs/temperature_performance_report_v1",
        type=Path,
    )
    args = parser.parse_args()

    missing = missing_video_rows(args.missing_videos)
    paired = paired_fold_rows(args.paired_audit)
    sequence = grouped_fold_rows(args.locked_run, "sequence")
    cow = grouped_fold_rows(args.locked_run, "cow")
    date = grouped_fold_rows(args.locked_run, "date")
    all_folds = paired + sequence + cow + date

    write_csv(args.output_dir / "missing_labeled_videos.csv", missing)
    write_csv(args.output_dir / "fold_results_table.csv", all_folds)

    lines = [
        "# Fold Results and Missing Videos",
        "",
        "## Missing labeled videos",
        "",
        f"The metadata contains 29 rectal-temperature labels. Twenty-one have raw TIFF frames and "
        f"{len(missing)} do not. Every missing row below has zero raw frames.",
        "",
    ]
    markdown_table(
        lines,
        ["Video", "Cow", "Ground truth (F)", "Expected raw file pattern"],
        [
            [
                row["video"],
                row["cow_tag"],
                fmt(row["ground_truth_temperature_f"], 1),
                f"`{row['expected_raw_file_pattern']}`",
            ]
            for row in missing
        ],
    )

    lines.extend(["", "## Paired RGB/thermal folds", ""])
    markdown_table(
        lines,
        ["Fold", "Train labels", "Val labels", "Test labels", "Status", "Reason"],
        [
            [
                row["fold"],
                row["train_labeled_pairs"],
                row["val_labeled_pairs"],
                row["test_labeled_pairs"],
                row["status"],
                row["reason"],
            ]
            for row in paired
        ],
    )

    for title, rows in (
        ("Sequence-grouped temperature folds", sequence),
        ("Cow-grouped temperature folds", cow),
        ("Date-grouped temperature folds", date),
    ):
        lines.extend(["", f"## {title}", ""])
        markdown_table(
            lines,
            ["Fold", "Test groups", "Actual range (F)", "Predicted range (F)", "MAE (F)", "RMSE (F)", "R2"],
            [
                [
                    row["fold"],
                    row["test_groups"],
                    row["actual_temperature_range_f"],
                    row["predicted_temperature_range_f"],
                    fmt(row["mae_f"]),
                    fmt(row["rmse_f"]),
                    fmt(row["r2"]),
                ]
                for row in rows
            ],
        )

    lines.extend(
        [
            "",
            "## Reading the table",
            "",
            "- Sequence grouping tests unseen videos.",
            "- Cow grouping keeps the same cow out of both training and testing.",
            "- Date grouping tests transfer to a different collection day.",
            "- Negative R2 means the fold performed worse than predicting that fold's average temperature.",
            "- The paired RGB/thermal folds remain blocked because labeled samples are not present in both train and test.",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "fold_results_table.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved:", args.output_dir / "missing_labeled_videos.csv")
    print("Saved:", args.output_dir / "fold_results_table.csv")
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
