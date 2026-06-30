from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def yes_no(value):
    return "yes" if value else "no"


def prediction_quality(abs_error):
    if abs_error <= 0.5:
        return "within_0.5F"
    if abs_error <= 1.0:
        return "within_1.0F"
    if abs_error <= 2.0:
        return "within_2.0F"
    return "over_2.0F"


def enrich_predictions(rows, evaluation):
    enriched = []
    for row in rows:
        truth = float(row["temperature_f"])
        prediction = float(row["prediction_f"])
        error = prediction - truth
        abs_error = abs(error)
        enriched.append(
            {
                "evaluation": evaluation,
                "fold": row.get("fold", ""),
                "date": row["date"],
                "sequence_num": row["sequence_num"],
                "cow_tag": row.get("cow_tag", ""),
                "actual_temperature_f": truth,
                "predicted_temperature_f": prediction,
                "signed_error_f": error,
                "absolute_error_f": abs_error,
                "within_0_5_f": yes_no(abs_error <= 0.5),
                "within_1_0_f": yes_no(abs_error <= 1.0),
                "within_2_0_f": yes_no(abs_error <= 2.0),
                "error_band": prediction_quality(abs_error),
            }
        )
    return enriched


def prediction_summary(rows):
    errors = [float(row["signed_error_f"]) for row in rows]
    abs_errors = [float(row["absolute_error_f"]) for row in rows]
    return {
        "samples": len(rows),
        "mean_signed_error_f": sum(errors) / len(errors),
        "mae_f": sum(abs_errors) / len(abs_errors),
        "rmse_f": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "within_0_5_f_pct": 100.0 * sum(error <= 0.5 for error in abs_errors) / len(errors),
        "within_1_0_f_pct": 100.0 * sum(error <= 1.0 for error in abs_errors) / len(errors),
        "within_2_0_f_pct": 100.0 * sum(error <= 2.0 for error in abs_errors) / len(errors),
        "worst_absolute_error_f": max(abs_errors),
    }


def main():
    parser = argparse.ArgumentParser(description="Create consolidated cattle temperature performance tables.")
    parser.add_argument(
        "--locked-run",
        default="data/temperature_outputs/best_roi_gradient_boosting_k20_v1",
        type=Path,
    )
    parser.add_argument(
        "--paired-fold-run",
        default="data/temperature_outputs/paired_rgb_thermal_5fold_eval_v1",
        type=Path,
    )
    parser.add_argument(
        "--article-style-comparison",
        default="data/temperature_outputs/detected_roi_filtered_80_frame_model_compare_v1/comparison.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="data/temperature_outputs/temperature_performance_report_v1",
        type=Path,
    )
    args = parser.parse_args()

    locked_metrics = read_json(args.locked_run / "metrics.json")
    paired_summary = read_json(args.paired_fold_run / "summary.json")
    article_rows = read_csv(args.article_style_comparison)
    article_best = min(article_rows, key=lambda row: float(row["test_mae"]))
    article_style_mae = float(article_best["test_mae"])

    prediction_sets = {
        "locked_holdout": read_csv(args.locked_run / "holdout_predictions.csv"),
        "sequence_grouped_cv": read_csv(args.locked_run / "cv_sequence_predictions.csv"),
        "cow_grouped_cv": read_csv(args.locked_run / "cv_cow_predictions.csv"),
        "date_grouped_cv": read_csv(args.locked_run / "cv_date_predictions.csv"),
    }
    enriched_sets = {
        name: enrich_predictions(rows, name)
        for name, rows in prediction_sets.items()
    }
    all_predictions = []
    for rows in enriched_sets.values():
        all_predictions.extend(rows)
    write_csv(args.output_dir / "temperature_predictions.csv", all_predictions)
    write_csv(args.output_dir / "sequence_cv_temperature_predictions.csv", enriched_sets["sequence_grouped_cv"])

    validation_by_group = {
        row["grouping"]: row
        for row in locked_metrics["validation"]
    }
    performance_rows = [
        {
            "evaluation": "paired_rgb_thermal_5fold",
            "model": "paired image-stat baseline",
            "samples": "",
            "folds": paired_summary["folds"],
            "trained_folds": paired_summary["trained_folds"],
            "mae_f": "",
            "mae_std_f": "",
            "rmse_f": "",
            "rmse_std_f": "",
            "gap_vs_local_article_style_mae_f": "",
            "generalization_value": "not_evaluable",
            "notes": "No fold has labeled train and test data with at least two train temperatures.",
        },
        {
            "evaluation": "locked_roi_holdout",
            "model": "gradient_boosting_top20",
            "samples": len(prediction_sets["locked_holdout"]),
            "folds": 1,
            "trained_folds": 1,
            "mae_f": locked_metrics["holdout"]["mae"],
            "mae_std_f": "",
            "rmse_f": locked_metrics["holdout"]["rmse"],
            "rmse_std_f": "",
            "gap_vs_local_article_style_mae_f": locked_metrics["holdout"]["mae"] - article_style_mae,
            "generalization_value": "useful_but_small",
            "notes": "Four held-out sequences.",
        },
    ]

    for grouping, evaluation in (
        ("sequence", "locked_roi_sequence_5fold"),
        ("cow", "locked_roi_cow_5fold"),
        ("date", "locked_roi_date_3fold"),
    ):
        row = validation_by_group[grouping]
        performance_rows.append(
            {
                "evaluation": evaluation,
                "model": "gradient_boosting_top20",
                "samples": locked_metrics["sample_count"],
                "folds": row["folds"],
                "trained_folds": row["folds"],
                "mae_f": row["mae_mean"],
                "mae_std_f": row["mae_std"],
                "rmse_f": row["rmse_mean"],
                "rmse_std_f": row["rmse_std"],
                "gap_vs_local_article_style_mae_f": row["mae_mean"] - article_style_mae,
                "generalization_value": "honest_grouped_evaluation",
                "notes": f"Test groups are held out by {grouping}.",
            }
        )

    performance_rows.append(
        {
            "evaluation": "local_article_style_frame_split",
            "model": article_best["model"],
            "samples": article_best["sample_count"],
            "folds": 5,
            "trained_folds": 5,
            "mae_f": article_best["test_mae"],
            "mae_std_f": "",
            "rmse_f": article_best["test_rmse"],
            "rmse_std_f": "",
            "gap_vs_local_article_style_mae_f": 0.0,
            "generalization_value": "optimistic_frame_level",
            "notes": "Frames from the same sequence may appear in train and test.",
        }
    )
    write_csv(args.output_dir / "performance_table.csv", performance_rows)

    summaries = {
        name: prediction_summary(rows)
        for name, rows in enriched_sets.items()
    }
    write_csv(
        args.output_dir / "prediction_summary.csv",
        [{"evaluation": name, **summary} for name, summary in summaries.items()],
    )

    sequence_rows = sorted(
        enriched_sets["sequence_grouped_cv"],
        key=lambda row: float(row["absolute_error_f"]),
        reverse=True,
    )
    worst_rows = sequence_rows[:5]

    report_lines = [
        "# Temperature Model Performance",
        "",
        "## Evaluation summary",
        "",
        "| Evaluation | Samples | Folds | MAE (F) | RMSE (F) | Meaning |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in performance_rows:
        report_lines.append(
            f"| {row['evaluation']} | {row['samples'] or '-'} | {row['folds']} | "
            f"{fmt(row['mae_f'])} | {fmt(row['rmse_f'])} | {row['generalization_value']} |"
        )

    report_lines.extend(
        [
            "",
            "## Sequence-grouped output temperatures",
            "",
            "| Date/sequence | Cow | Actual (F) | Predicted (F) | Abs error (F) | Band |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(
        enriched_sets["sequence_grouped_cv"],
        key=lambda item: (item["date"], int(item["sequence_num"])),
    ):
        report_lines.append(
            f"| {row['date']}/{row['sequence_num']} | {row['cow_tag']} | "
            f"{fmt(row['actual_temperature_f'], 1)} | {fmt(row['predicted_temperature_f'], 2)} | "
            f"{fmt(row['absolute_error_f'], 2)} | {row['error_band']} |"
        )

    sequence_summary = summaries["sequence_grouped_cv"]
    report_lines.extend(
        [
            "",
            "## Working range",
            "",
            f"- Sequence 5-fold MAE: {fmt(sequence_summary['mae_f'])} F",
            f"- Sequence 5-fold RMSE: {fmt(sequence_summary['rmse_f'])} F",
            f"- Predictions within 0.5 F: {fmt(sequence_summary['within_0_5_f_pct'], 1)}%",
            f"- Predictions within 1.0 F: {fmt(sequence_summary['within_1_0_f_pct'], 1)}%",
            f"- Predictions within 2.0 F: {fmt(sequence_summary['within_2_0_f_pct'], 1)}%",
            f"- Mean signed bias: {fmt(sequence_summary['mean_signed_error_f'])} F",
            "",
            "## Largest sequence errors",
            "",
            "| Date/sequence | Actual (F) | Predicted (F) | Signed error (F) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in worst_rows:
        report_lines.append(
            f"| {row['date']}/{row['sequence_num']} | {fmt(row['actual_temperature_f'], 1)} | "
            f"{fmt(row['predicted_temperature_f'], 2)} | {fmt(row['signed_error_f'], 2)} |"
        )

    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The paired RGB/thermal 5-fold pipeline is implemented, but currently produces no valid temperature metrics.",
            "- The locked ROI model is working and produces held-out temperatures for all 21 labeled sequences.",
            "- Cow-grouped validation is the strongest honest grouped result; date-grouped validation is the weakest.",
            "- The frame-level article-style result is much lower, but it is optimistic because sequence frames can cross splits.",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "performance_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("Saved:", args.output_dir / "performance_report.md")
    print("Saved:", args.output_dir / "performance_table.csv")
    print("Saved:", args.output_dir / "temperature_predictions.csv")
    print("Saved:", args.output_dir / "sequence_cv_temperature_predictions.csv")


if __name__ == "__main__":
    main()
