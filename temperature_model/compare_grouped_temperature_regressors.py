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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import ID_COLUMNS, read_feature_csv, write_csv


def metric_dict(y_true, y_pred):
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if len(y_true) > 1:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    return metrics


def group_id(record):
    return f"{record['date']}/{record['sequence_num']}"


def load_forced_split(path: Path):
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    return set(split["train_videos"]), set(split["test_videos"])


def make_selector(feature_count, select_k):
    if not select_k:
        return []
    return [SelectKBest(score_func=f_regression, k=min(select_k, feature_count))]


def model_specs(feature_count, select_k, seed):
    selector = make_selector(feature_count, select_k)
    return {
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            *selector,
            RandomForestRegressor(n_estimators=500, random_state=seed, min_samples_leaf=1),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            *selector,
            ExtraTreesRegressor(n_estimators=500, random_state=seed, min_samples_leaf=1),
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            *selector,
            GradientBoostingRegressor(random_state=seed, n_estimators=300, learning_rate=0.03, max_depth=2),
        ),
        "svr_rbf": make_pipeline(
            SimpleImputer(strategy="median"),
            *selector,
            StandardScaler(),
            SVR(C=10.0, epsilon=0.05, gamma="scale"),
        ),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            *selector,
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "knn": make_pipeline(
            SimpleImputer(strategy="median"),
            *selector,
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=5, weights="distance"),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare temperature regressors with cattle sequence grouped train/test splits."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--select-k", default=40, type=int)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument(
        "--split-metrics",
        type=Path,
        help="Optional metrics.json with train_videos and test_videos keys.",
    )
    args = parser.parse_args()

    records = read_feature_csv(args.features)
    feature_names = [key for key in sorted(records[0]) if key not in ID_COLUMNS]
    x = np.asarray(
        [[record.get(name, np.nan) for name in feature_names] for record in records],
        dtype=np.float32,
    )
    y = np.asarray([record["temperature_f"] for record in records], dtype=np.float32)
    groups = np.asarray([group_id(record) for record in records])
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 5:
        raise RuntimeError(f"Need at least 5 groups for grouped evaluation, found {len(unique_groups)}")

    if args.split_metrics:
        train_groups, test_groups = load_forced_split(args.split_metrics)
        train_idx = np.asarray([idx for idx, group in enumerate(groups) if group in train_groups])
        test_idx = np.asarray([idx for idx, group in enumerate(groups) if group in test_groups])
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError("Forced split did not match feature rows.")
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(x, y, groups))

    rows = []
    all_metrics = {}
    best_name = None
    best_model = None
    best_mae = float("inf")
    for name, model in model_specs(len(feature_names), args.select_k, args.seed).items():
        model.fit(x[train_idx], y[train_idx])
        test_metrics = metric_dict(y[test_idx], model.predict(x[test_idx]))

        fold_metrics = []
        cv = GroupKFold(n_splits=min(5, len(unique_groups)))
        for fold, (fold_train, fold_test) in enumerate(cv.split(x, y, groups), start=1):
            fold_model = model_specs(len(feature_names), args.select_k, args.seed + fold)[name]
            fold_model.fit(x[fold_train], y[fold_train])
            metrics = metric_dict(y[fold_test], fold_model.predict(x[fold_test]))
            metrics["fold"] = fold
            metrics["test_groups"] = sorted(set(groups[fold_test]))
            fold_metrics.append(metrics)

        summary = {
            "model": name,
            "sample_count": len(records),
            "group_count": len(unique_groups),
            "feature_count": len(feature_names),
            "selected_feature_count": min(args.select_k, len(feature_names)),
            "train_groups": len(set(groups[train_idx])),
            "test_groups": len(set(groups[test_idx])),
            "test_mae": test_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_rmse": test_metrics["rmse"],
            "kfold_mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
            "kfold_mse_mean": float(np.mean([m["mse"] for m in fold_metrics])),
            "kfold_rmse_mean": float(np.mean([m["rmse"] for m in fold_metrics])),
        }
        rows.append(summary)
        all_metrics[name] = {"test": test_metrics, "group_kfold": fold_metrics}
        print(
            f"{name}: grouped test MAE={summary['test_mae']:.3f}, "
            f"grouped test RMSE={summary['test_rmse']:.3f}, "
            f"grouped kfold MAE={summary['kfold_mae_mean']:.3f}"
        )
        if summary["test_mae"] < best_mae:
            best_mae = summary["test_mae"]
            best_name = name
            best_model = model

    rows.sort(key=lambda row: row["test_mae"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "comparison.csv", rows)
    best_predictions = best_model.predict(x[test_idx])
    prediction_rows = []
    for record, pred in zip([records[idx] for idx in test_idx], best_predictions):
        prediction_rows.append(
            {
                "date": record["date"],
                "sequence_num": record["sequence_num"],
                "cow_tag": record.get("cow_tag", ""),
                "temperature_f": record["temperature_f"],
                "prediction_f": float(pred),
                "error_f": float(pred - record["temperature_f"]),
            }
        )
    write_csv(args.output_dir / "predictions.csv", prediction_rows)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_name,
                "sample_count": len(records),
                "group_count": len(unique_groups),
                "split_metrics": str(args.split_metrics) if args.split_metrics else None,
                "holdout_train_groups": sorted(set(groups[train_idx])),
                "holdout_test_groups": sorted(set(groups[test_idx])),
                "models": all_metrics,
            },
            f,
            indent=2,
        )
    joblib.dump(best_model, args.output_dir / "best_temperature_model.joblib")
    print("Best by grouped holdout MAE:", best_name)
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
