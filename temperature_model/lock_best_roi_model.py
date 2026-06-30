from __future__ import annotations

import argparse
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

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from common import ID_COLUMNS, read_feature_csv, write_csv


def record_key(record: dict[str, object]) -> str:
    return f"{record['date']}/{record['sequence_num']}"


def load_split(path: Path) -> tuple[set[str], set[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data["train_videos"]), set(data["test_videos"])


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if len(y_true) > 1:
        result["r2"] = float(r2_score(y_true, y_pred))
    return result


def make_model(select_k: int, seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        SelectKBest(score_func=f_regression, k=select_k),
        GradientBoostingRegressor(
            random_state=seed,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
        ),
    )


def build_matrix(records: list[dict[str, object]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[record.get(name, np.nan) for name in feature_names] for record in records],
        dtype=np.float32,
    )
    y = np.asarray([record["temperature_f"] for record in records], dtype=np.float32)
    return x, y


def selected_feature_rows(model, feature_names: list[str]) -> list[dict[str, object]]:
    selector = model.named_steps["selectkbest"]
    regressor = model.named_steps["gradientboostingregressor"]
    mask = selector.get_support()
    selected = [name for name, keep in zip(feature_names, mask) if keep]
    scores = selector.scores_
    rows = []
    for rank, (feature, importance) in enumerate(
        sorted(zip(selected, regressor.feature_importances_), key=lambda item: item[1], reverse=True),
        start=1,
    ):
        feature_index = feature_names.index(feature)
        rows.append(
            {
                "rank": rank,
                "feature": feature,
                "f_score": float(scores[feature_index]) if np.isfinite(scores[feature_index]) else "",
                "model_importance": float(importance),
            }
        )
    return rows


def write_predictions(
    path: Path,
    records: list[dict[str, object]],
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    rows = []
    for idx, truth, pred in zip(indices, y_true, y_pred):
        record = records[int(idx)]
        rows.append(
            {
                "date": record["date"],
                "sequence_num": record["sequence_num"],
                "cow_tag": record["cow_tag"],
                "temperature_f": float(truth),
                "prediction_f": float(pred),
                "error_f": float(pred - truth),
            }
        )
    write_csv(path, rows)


def grouped_cv(
    records: list[dict[str, object]],
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    group_name: str,
    groups: np.ndarray,
    select_k: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object] | None:
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return None

    n_splits = min(5, len(unique_groups))
    fold_metrics = []
    prediction_rows = []
    cv = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
        model = make_model(min(select_k, len(feature_names)), seed + fold)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        metrics = metric_dict(y[test_idx], pred)
        metrics.update(
            {
                "grouping": group_name,
                "fold": fold,
                "train_groups": "|".join(sorted(set(groups[train_idx]))),
                "test_groups": "|".join(sorted(set(groups[test_idx]))),
            }
        )
        fold_metrics.append(metrics)
        for idx, truth, value in zip(test_idx, y[test_idx], pred):
            record = records[int(idx)]
            prediction_rows.append(
                {
                    "grouping": group_name,
                    "fold": fold,
                    "date": record["date"],
                    "sequence_num": record["sequence_num"],
                    "cow_tag": record["cow_tag"],
                    "temperature_f": float(truth),
                    "prediction_f": float(value),
                    "error_f": float(value - truth),
                }
            )

    write_csv(output_dir / f"cv_{group_name}_folds.csv", fold_metrics)
    write_csv(output_dir / f"cv_{group_name}_predictions.csv", prediction_rows)
    return {
        "grouping": group_name,
        "group_count": len(unique_groups),
        "folds": n_splits,
        "mae_mean": float(np.mean([row["mae"] for row in fold_metrics])),
        "mae_std": float(np.std([row["mae"] for row in fold_metrics])),
        "mse_mean": float(np.mean([row["mse"] for row in fold_metrics])),
        "rmse_mean": float(np.mean([row["rmse"] for row in fold_metrics])),
        "rmse_std": float(np.std([row["rmse"] for row in fold_metrics])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock the best keypoint ROI temperature model.")
    parser.add_argument("--features", default="data/temperature_outputs/detected_roi_filtered_80_v1/features.csv", type=Path)
    parser.add_argument("--split-metrics", default="data/temperature_outputs/thermal_cnn_absolute_quick_lr1e3_v1/metrics.json", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/best_roi_gradient_boosting_v1", type=Path)
    parser.add_argument("--select-k", default=10, type=int)
    parser.add_argument("--seed", default=1337, type=int)
    args = parser.parse_args()

    records = read_feature_csv(args.features)
    feature_names = [key for key in sorted(records[0]) if key not in ID_COLUMNS]
    x, y = build_matrix(records, feature_names)

    train_groups, test_groups = load_split(args.split_metrics)
    keys = np.asarray([record_key(record) for record in records])
    train_idx = np.asarray([idx for idx, key in enumerate(keys) if key in train_groups], dtype=int)
    test_idx = np.asarray([idx for idx, key in enumerate(keys) if key in test_groups], dtype=int)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Split did not match ROI feature rows.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    select_k = min(args.select_k, len(feature_names))

    holdout_model = make_model(select_k, args.seed)
    holdout_model.fit(x[train_idx], y[train_idx])
    holdout_pred = holdout_model.predict(x[test_idx])
    holdout_metrics = metric_dict(y[test_idx], holdout_pred)
    joblib.dump(holdout_model, args.output_dir / "model_holdout.joblib")
    write_predictions(args.output_dir / "holdout_predictions.csv", records, test_idx, y[test_idx], holdout_pred)
    holdout_selected = selected_feature_rows(holdout_model, feature_names)
    write_csv(args.output_dir / "selected_features_holdout.csv", holdout_selected)

    full_model = make_model(select_k, args.seed)
    full_model.fit(x, y)
    joblib.dump(full_model, args.output_dir / "model_full.joblib")
    full_selected = selected_feature_rows(full_model, feature_names)
    write_csv(args.output_dir / "selected_features_full.csv", full_selected)

    validation = []
    group_sets = {
        "sequence": keys,
        "cow": np.asarray([str(record["cow_tag"]) for record in records]),
        "date": np.asarray([str(record["date"]) for record in records]),
    }
    for group_name, groups in group_sets.items():
        summary = grouped_cv(records, x, y, feature_names, group_name, groups, select_k, args.seed, args.output_dir)
        if summary:
            validation.append(summary)
    write_csv(args.output_dir / "validation_summary.csv", validation)

    schema = {
        "features_file": str(args.features),
        "feature_names": feature_names,
        "id_columns": sorted(ID_COLUMNS),
        "selected_feature_count": select_k,
        "model_type": "GradientBoostingRegressor",
        "model_params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 2,
            "random_state": args.seed,
        },
    }
    with (args.output_dir / "feature_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    metrics = {
        "sample_count": len(records),
        "feature_count": len(feature_names),
        "selected_feature_count": select_k,
        "split_metrics": str(args.split_metrics),
        "holdout_train_videos": sorted(set(keys[train_idx])),
        "holdout_test_videos": sorted(set(keys[test_idx])),
        "holdout": holdout_metrics,
        "validation": validation,
        "selected_features_holdout": [row["feature"] for row in holdout_selected],
        "selected_features_full": [row["feature"] for row in full_selected],
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", args.output_dir)
    print("Holdout MAE:", holdout_metrics["mae"])
    print("Holdout RMSE:", holdout_metrics["rmse"])
    for row in validation:
        print(f"{row['grouping']} CV MAE={row['mae_mean']:.3f} RMSE={row['rmse_mean']:.3f}")


if __name__ == "__main__":
    main()
