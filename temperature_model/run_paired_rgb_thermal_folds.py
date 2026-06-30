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
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SPLITS = ("train", "val", "test")
ID_COLUMNS = {
    "pair_id",
    "split",
    "folder",
    "frame_id",
    "rgb_file",
    "thermal_file",
    "rgb_original_file",
    "thermal_original_file",
    "rgb_original_id",
    "thermal_original_id",
    "label_available",
    "raw_date",
    "raw_sequence_num",
    "cow_tag",
    "temperature_f",
    "mapping_mean_score",
}


class MeanRegressor:
    def fit(self, x, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_, dtype=np.float32)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def labeled(rows):
    return [row for row in rows if row.get("label_available") == "1" and row.get("temperature_f")]


def sort_key(value):
    try:
        return int(value)
    except ValueError:
        return value


def metric_dict(y_true, y_pred):
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if len(y_true) > 1:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    return metrics


def image_stats(path, prefix):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    features = {
        f"{prefix}_height": float(array.shape[0]),
        f"{prefix}_width": float(array.shape[1]),
        f"{prefix}_aspect": float(array.shape[1] / max(array.shape[0], 1)),
    }
    channel_names = ("r", "g", "b")
    for idx, channel_name in enumerate(channel_names):
        values = array[:, :, idx].reshape(-1)
        features[f"{prefix}_{channel_name}_mean"] = float(np.mean(values))
        features[f"{prefix}_{channel_name}_std"] = float(np.std(values))
        features[f"{prefix}_{channel_name}_min"] = float(np.min(values))
        features[f"{prefix}_{channel_name}_max"] = float(np.max(values))
        features[f"{prefix}_{channel_name}_p10"] = float(np.percentile(values, 10))
        features[f"{prefix}_{channel_name}_p50"] = float(np.percentile(values, 50))
        features[f"{prefix}_{channel_name}_p90"] = float(np.percentile(values, 90))

    gray = array.mean(axis=2).reshape(-1)
    features[f"{prefix}_gray_mean"] = float(np.mean(gray))
    features[f"{prefix}_gray_std"] = float(np.std(gray))
    features[f"{prefix}_gray_p10"] = float(np.percentile(gray, 10))
    features[f"{prefix}_gray_p50"] = float(np.percentile(gray, 50))
    features[f"{prefix}_gray_p90"] = float(np.percentile(gray, 90))
    return features


def row_features(row, source_root):
    rgb_path = source_root / row["rgb_file"]
    thermal_path = source_root / row["thermal_file"]
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)
    if not thermal_path.exists():
        raise FileNotFoundError(thermal_path)

    features = {}
    features.update(image_stats(rgb_path, "rgb"))
    features.update(image_stats(thermal_path, "thermal"))
    features["frame_id_numeric"] = float(int(row["frame_id"]))
    score = row.get("mapping_mean_score") or ""
    features["mapping_mean_score"] = float(score) if score else np.nan
    return features


def feature_matrix(rows, source_root):
    records = []
    for row in rows:
        record = {
            "temperature_f": float(row["temperature_f"]),
            "folder": row["folder"],
            "frame_id": row["frame_id"],
            "raw_date": row.get("raw_date", ""),
            "raw_sequence_num": row.get("raw_sequence_num", ""),
            "cow_tag": row.get("cow_tag", ""),
        }
        record.update(row_features(row, source_root))
        records.append(record)

    feature_names = [key for key in sorted(records[0]) if key not in {"temperature_f", "folder", "frame_id", "raw_date", "raw_sequence_num", "cow_tag"}]
    x = np.asarray([[record.get(name, np.nan) for name in feature_names] for record in records], dtype=np.float32)
    y = np.asarray([record["temperature_f"] for record in records], dtype=np.float32)
    return records, feature_names, x, y


def model_specs(seed):
    return {
        "train_mean": MeanRegressor(),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, random_state=seed, min_samples_leaf=1),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=300, random_state=seed, min_samples_leaf=1),
        ),
        "scaled_extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ExtraTreesRegressor(n_estimators=300, random_state=seed, min_samples_leaf=1),
        ),
    }


def fold_status(split_rows, args):
    train_labeled = labeled(split_rows["train"])
    val_labeled = labeled(split_rows["val"])
    test_labeled = labeled(split_rows["test"])
    train_temperatures = {row["temperature_f"] for row in train_labeled}
    test_temperatures = {row["temperature_f"] for row in test_labeled}

    reasons = []
    if len(train_labeled) < args.min_train_labels:
        reasons.append(f"train has {len(train_labeled)} labeled rows, needs {args.min_train_labels}")
    if len(test_labeled) < args.min_test_labels:
        reasons.append(f"test has {len(test_labeled)} labeled rows, needs {args.min_test_labels}")
    if len(train_temperatures) < args.min_unique_train_temperatures:
        reasons.append(
            f"train has {len(train_temperatures)} unique temperature values, needs {args.min_unique_train_temperatures}"
        )
    if len(test_temperatures) < args.min_unique_test_temperatures:
        reasons.append(
            f"test has {len(test_temperatures)} unique temperature values, needs {args.min_unique_test_temperatures}"
        )
    return train_labeled, val_labeled, test_labeled, reasons


def audit_row(fold_name, split_rows, status, reason):
    row = {"fold": fold_name, "status": status, "reason": reason}
    for split in SPLITS:
        rows = split_rows[split]
        labeled_rows = labeled(rows)
        row[f"{split}_pairs"] = len(rows)
        row[f"{split}_labeled_pairs"] = len(labeled_rows)
        row[f"{split}_folders"] = " ".join(sorted({item["folder"] for item in rows}, key=sort_key))
        row[f"{split}_labeled_folders"] = " ".join(sorted({item["folder"] for item in labeled_rows}, key=sort_key))
        row[f"{split}_unique_temperatures"] = len({item["temperature_f"] for item in labeled_rows})
    return row


def train_fold(fold_name, train_rows, val_rows, test_rows, source_root, output_dir, seed):
    train_records, feature_names, x_train, y_train = feature_matrix(train_rows, source_root)
    test_records, _, x_test, y_test = feature_matrix(test_rows, source_root)
    val_records = []
    x_val = y_val = None
    if val_rows:
        val_records, _, x_val, y_val = feature_matrix(val_rows, source_root)

    metric_rows = []
    prediction_rows = []
    for model_name, model in model_specs(seed).items():
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        test_metrics = metric_dict(y_test, test_pred)
        row = {
            "fold": fold_name,
            "model": model_name,
            "feature_count": len(feature_names),
            "train_labeled_pairs": len(train_rows),
            "val_labeled_pairs": len(val_rows),
            "test_labeled_pairs": len(test_rows),
            "test_mae": test_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_rmse": test_metrics["rmse"],
        }
        if "r2" in test_metrics:
            row["test_r2"] = test_metrics["r2"]

        if x_val is not None and len(val_rows) > 0:
            val_pred = model.predict(x_val)
            val_metrics = metric_dict(y_val, val_pred)
            row["val_mae"] = val_metrics["mae"]
            row["val_mse"] = val_metrics["mse"]
            row["val_rmse"] = val_metrics["rmse"]
            if "r2" in val_metrics:
                row["val_r2"] = val_metrics["r2"]
        metric_rows.append(row)

        for record, pred in zip(test_records, test_pred):
            prediction_rows.append(
                {
                    "fold": fold_name,
                    "model": model_name,
                    "folder": record["folder"],
                    "frame_id": record["frame_id"],
                    "raw_date": record["raw_date"],
                    "raw_sequence_num": record["raw_sequence_num"],
                    "cow_tag": record["cow_tag"],
                    "temperature_f": record["temperature_f"],
                    "prediction_f": float(pred),
                    "error_f": float(pred - record["temperature_f"]),
                }
            )

    write_csv(output_dir / f"{fold_name}_predictions.csv", prediction_rows)
    return metric_rows


def main():
    parser = argparse.ArgumentParser(
        description="Audit and evaluate paired RGB/thermal 5-fold manifests when labels are available."
    )
    parser.add_argument("--fold-root", default="datasets/keypoints/paired_rgb_thermal_5fold", type=Path)
    parser.add_argument("--source-root", default="datasets/keypoints", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/paired_rgb_thermal_5fold_eval_v1", type=Path)
    parser.add_argument("--min-train-labels", default=5, type=int)
    parser.add_argument("--min-test-labels", default=1, type=int)
    parser.add_argument("--min-unique-train-temperatures", default=2, type=int)
    parser.add_argument("--min-unique-test-temperatures", default=1, type=int)
    parser.add_argument("--seed", default=1337, type=int)
    args = parser.parse_args()

    fold_dirs = sorted([path for path in args.fold_root.glob("fold_*") if path.is_dir()], key=lambda path: sort_key(path.name.split("_")[-1]))
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under {args.fold_root}")

    audit_rows = []
    all_metric_rows = []
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        annotations = fold_dir / "annotations"
        split_rows = {split: read_csv(annotations / f"pairs_{split}.csv") for split in SPLITS}
        train_rows, val_rows, test_rows, reasons = fold_status(split_rows, args)
        if reasons:
            reason = "; ".join(reasons)
            audit_rows.append(audit_row(fold_name, split_rows, "skipped", reason))
            print(f"{fold_name}: skipped - {reason}")
            continue

        print(f"{fold_name}: training paired image-stat baseline")
        audit_rows.append(audit_row(fold_name, split_rows, "trained", ""))
        metric_rows = train_fold(
            fold_name,
            train_rows,
            val_rows,
            test_rows,
            args.source_root,
            args.output_dir,
            args.seed,
        )
        all_metric_rows.extend(metric_rows)

    audit_fields = [
        "fold",
        "status",
        "reason",
        "train_pairs",
        "train_labeled_pairs",
        "train_folders",
        "train_labeled_folders",
        "train_unique_temperatures",
        "val_pairs",
        "val_labeled_pairs",
        "val_folders",
        "val_labeled_folders",
        "val_unique_temperatures",
        "test_pairs",
        "test_labeled_pairs",
        "test_folders",
        "test_labeled_folders",
        "test_unique_temperatures",
    ]
    write_csv(args.output_dir / "fold_audit.csv", audit_rows, audit_fields)
    if all_metric_rows:
        write_csv(args.output_dir / "model_metrics.csv", all_metric_rows)

    completed = [row for row in audit_rows if row["status"] == "trained"]
    summary = {
        "fold_root": str(args.fold_root),
        "source_root": str(args.source_root),
        "folds": len(fold_dirs),
        "trained_folds": len(completed),
        "skipped_folds": len(fold_dirs) - len(completed),
        "min_train_labels": args.min_train_labels,
        "min_test_labels": args.min_test_labels,
        "min_unique_train_temperatures": args.min_unique_train_temperatures,
        "min_unique_test_temperatures": args.min_unique_test_temperatures,
        "end_goal": (
            "Train and evaluate one paired RGB/thermal temperature model per fold, then report "
            "mean/std MAE and RMSE across folds once each fold has labeled train and test data."
        ),
    }
    if all_metric_rows:
        by_model = {}
        for model_name in sorted({row["model"] for row in all_metric_rows}):
            rows = [row for row in all_metric_rows if row["model"] == model_name]
            by_model[model_name] = {
                "folds": len(rows),
                "test_mae_mean": float(np.mean([row["test_mae"] for row in rows])),
                "test_mae_std": float(np.std([row["test_mae"] for row in rows])),
                "test_rmse_mean": float(np.mean([row["test_rmse"] for row in rows])),
                "test_rmse_std": float(np.std([row["test_rmse"] for row in rows])),
            }
        summary["models"] = by_model
    write_json(args.output_dir / "summary.json", summary)

    print("Saved audit:", args.output_dir / "fold_audit.csv")
    if all_metric_rows:
        print("Saved metrics:", args.output_dir / "model_metrics.csv")
    else:
        print("No folds had enough labels for training/evaluation.")
    print("Saved summary:", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
