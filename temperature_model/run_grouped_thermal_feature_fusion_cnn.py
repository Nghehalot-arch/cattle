from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
SKLEARN_COMPAT = Path(__file__).resolve().parent / "_sklearn_compat"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if SKLEARN_COMPAT.exists():
    sys.path.insert(0, str(SKLEARN_COMPAT))

import numpy as np
from sklearn.model_selection import GroupKFold

from common import write_csv
from train_thermal_feature_fusion_cnn import (
    add_anchor_predictions,
    apply_frame_filter,
    load_labeled_sequences,
    load_split,
    read_feature_rows,
    read_frame_filter,
    read_selected_features,
    set_seed,
    train_once,
)


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


def sample_group(sample, grouping: str) -> str:
    if grouping == "sequence":
        return sample.key
    if grouping == "cow":
        return sample.cow_tag
    if grouping == "date":
        return sample.date
    raise KeyError(grouping)


def read_feature_limit(path: Path | None, feature_rows: dict[str, dict[str, object]], limit: int | None) -> list[str]:
    feature_names = read_selected_features(path, feature_rows)
    if limit:
        feature_names = feature_names[:limit]
    return feature_names


def run_one_split(args, train_samples, test_samples, feature_rows, feature_names, seed: int):
    set_seed(seed)
    return train_once(args, train_samples, test_samples, feature_rows, feature_names)


def with_prediction_metadata(predictions, fold: int | None, grouping: str | None, cow_by_key: dict[str, str]):
    rows = []
    for row in predictions:
        date, sequence_num = str(row["sequence"]).split("/")
        output = {
            "date": date,
            "sequence_num": sequence_num,
            "cow_tag": cow_by_key.get(str(row["sequence"]), ""),
            "temperature_f": float(row["temperature_f"]),
            "prediction_f": float(row["prediction_f"]),
            "error_f": float(row["error_f"]),
        }
        if grouping is not None:
            output = {"grouping": grouping, "fold": int(fold), **output}
        rows.append(output)
    return rows


def run_forced_holdout(args, samples, feature_rows, feature_names, cow_by_key):
    if not args.split_metrics:
        return None
    train_keys, test_keys = load_split(args.split_metrics)
    train_samples = [sample for sample in samples if sample.key in train_keys]
    test_samples = [sample for sample in samples if sample.key in test_keys]
    if not train_samples or not test_samples:
        raise RuntimeError("Forced split did not match samples.")
    _, predictions, _, _ = run_one_split(args, train_samples, test_samples, feature_rows, feature_names, args.seed)
    rows = with_prediction_metadata(predictions, None, None, cow_by_key)
    return {
        "train_videos": [sample.key for sample in train_samples],
        "test_videos": [sample.key for sample in test_samples],
        "metrics": metric_dict(rows),
        "predictions": rows,
    }


def run_grouped_validation(args, samples, feature_rows, feature_names, cow_by_key):
    validation = []
    fold_rows_all = []
    prediction_rows_all = []
    for grouping in args.groupings:
        groups = np.asarray([sample_group(sample, grouping) for sample in samples])
        unique_groups = sorted(set(groups))
        if len(unique_groups) < 2:
            continue
        folds = min(args.max_folds, len(unique_groups))
        cv = GroupKFold(n_splits=folds)
        group_fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(samples)), groups=groups), start=1):
            fold_seed = args.seed + fold + (1000 * len(validation))
            train_samples = [samples[int(idx)] for idx in train_idx]
            test_samples = [samples[int(idx)] for idx in test_idx]
            _, predictions, _, _ = run_one_split(args, train_samples, test_samples, feature_rows, feature_names, fold_seed)
            prediction_rows = with_prediction_metadata(predictions, fold, grouping, cow_by_key)
            metrics = metric_dict(prediction_rows)
            test_groups = sorted(set(groups[test_idx]))
            group_fold_metrics.append({"fold": fold, "test_groups": test_groups, **metrics})
            fold_rows_all.append(
                {
                    "grouping": grouping,
                    "fold": fold,
                    "test_groups": ";".join(test_groups),
                    "test_sequences": ";".join(sample.key for sample in test_samples),
                    **metrics,
                }
            )
            prediction_rows_all.extend(prediction_rows)
            print(
                f"{grouping} fold={fold} "
                f"MAE={metrics['mae']:.3f} MSE={metrics['mse']:.3f} RMSE={metrics['rmse']:.3f}"
            )

        validation.append(
            {
                "grouping": grouping,
                "group_count": len(unique_groups),
                "folds": folds,
                "mae_mean": float(np.mean([row["mae"] for row in group_fold_metrics])),
                "mae_std": float(np.std([row["mae"] for row in group_fold_metrics])),
                "mse_mean": float(np.mean([row["mse"] for row in group_fold_metrics])),
                "rmse_mean": float(np.mean([row["rmse"] for row in group_fold_metrics])),
                "rmse_std": float(np.std([row["rmse"] for row in group_fold_metrics])),
                "fold_metrics": group_fold_metrics,
            }
        )
    return validation, fold_rows_all, prediction_rows_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the raw thermal + ROI feature fusion CNN with sequence/cow/date grouped CV."
    )
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--features", default="data/temperature_outputs/detected_article_otsu_fusion_v1/features.csv", type=Path)
    parser.add_argument("--selected-features", type=Path)
    parser.add_argument("--feature-limit", type=int)
    parser.add_argument("--frame-filter-csv", type=Path)
    parser.add_argument("--frame-score-column", default="frontal_score")
    parser.add_argument("--frame-candidate-limit", type=int)
    parser.add_argument("--min-filtered-frames", default=1, type=int)
    parser.add_argument("--anchor-model", type=Path)
    parser.add_argument("--anchor-schema", type=Path)
    parser.add_argument("--anchor-feature-name", default="roi_anchor_prediction")
    parser.add_argument("--split-metrics", default="data/temperature_outputs/thermal_cnn_absolute_quick_lr1e3_v1/metrics.json", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/grouped_thermal_feature_fusion_cnn_v1", type=Path)
    parser.add_argument("--groupings", nargs="*", default=["sequence", "cow", "date"], choices=["sequence", "cow", "date"])
    parser.add_argument("--max-folds", default=5, type=int)
    parser.add_argument("--max-frames", default=8, type=int)
    parser.add_argument("--image-size", default=96, type=int)
    parser.add_argument("--normalize", choices=("absolute", "percentile"), default="absolute")
    parser.add_argument("--thermal-min", default=15.0, type=float)
    parser.add_argument("--thermal-max", default=45.0, type=float)
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--dropout", default=0.25, type=float)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--log-period", default=50, type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    samples, missing = load_labeled_sequences(args.metadata, args.raw_zip)
    dropped_frame_filter = []
    if args.frame_filter_csv:
        frame_filter = read_frame_filter(args.frame_filter_csv, args.frame_score_column)
        samples, dropped_frame_filter = apply_frame_filter(
            samples,
            frame_filter,
            candidate_limit=args.frame_candidate_limit,
            min_frames=args.min_filtered_frames,
        )
    feature_rows = read_feature_rows(args.features)
    if args.anchor_model:
        if not args.anchor_schema:
            raise RuntimeError("--anchor-schema is required when --anchor-model is used.")
        add_anchor_predictions(feature_rows, args.anchor_model, args.anchor_schema, args.anchor_feature_name)
    samples = [sample for sample in samples if sample.key in feature_rows]
    if len(samples) < 5:
        raise RuntimeError(f"Need at least 5 samples with raw frames and ROI features, found {len(samples)}")

    feature_names = read_feature_limit(args.selected_features, feature_rows, args.feature_limit)
    if args.anchor_model and args.anchor_feature_name not in feature_names:
        feature_names.append(args.anchor_feature_name)
    cow_by_key = {sample.key: sample.cow_tag for sample in samples}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdout = run_forced_holdout(args, samples, feature_rows, feature_names, cow_by_key)
    if holdout is not None:
        write_csv(args.output_dir / "holdout_predictions.csv", holdout["predictions"])

    validation, fold_rows, prediction_rows = run_grouped_validation(
        args,
        samples,
        feature_rows,
        feature_names,
        cow_by_key,
    )
    write_csv(args.output_dir / "validation_summary.csv", [{key: value for key, value in row.items() if key != "fold_metrics"} for row in validation])
    write_csv(args.output_dir / "cv_folds.csv", fold_rows)
    write_csv(args.output_dir / "cv_predictions.csv", prediction_rows)
    for grouping in args.groupings:
        grouping_predictions = [row for row in prediction_rows if row["grouping"] == grouping]
        if grouping_predictions:
            write_csv(args.output_dir / f"cv_{grouping}_predictions.csv", grouping_predictions)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "usable_labeled_videos": len(samples),
                "missing_raw_labeled_videos": len(missing),
                "features": str(args.features),
                "selected_features": str(args.selected_features) if args.selected_features else None,
                "feature_count": len(feature_names),
                "frame_filter_csv": str(args.frame_filter_csv) if args.frame_filter_csv else None,
                "dropped_frame_filter": dropped_frame_filter,
                "feature_names": feature_names,
                "holdout": holdout,
                "validation": validation,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
            f,
            indent=2,
        )

    print("Saved:", args.output_dir)
    if holdout is not None:
        print(
            "Holdout:",
            f"MAE={holdout['metrics']['mae']:.3f}",
            f"MSE={holdout['metrics']['mse']:.3f}",
            f"RMSE={holdout['metrics']['rmse']:.3f}",
        )
    for row in validation:
        print(
            f"{row['grouping']} CV:",
            f"MAE={row['mae_mean']:.3f}",
            f"MSE={row['mse_mean']:.3f}",
            f"RMSE={row['rmse_mean']:.3f}",
        )


if __name__ == "__main__":
    main()
