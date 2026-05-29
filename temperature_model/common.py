import csv
import json
import math
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline


try:
    cv2.setLogLevel(0)
except AttributeError:
    pass


RAW_NAME_RE = re.compile(r"thermal_raw/([^/]+)/(\d+)_Video_Frame_(\d+)\.tiff$")
ID_COLUMNS = {
    "date",
    "sequence_num",
    "cow_tag",
    "temperature_f",
    "frame_id",
    "raw_frame_count",
    "sampled_frame_count",
    "detected_frame_count",
}


def load_temperature_metadata(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("temperature_f"):
                continue
            rows.append(
                {
                    "date": row["date"],
                    "sequence_num": row["sequence_num"],
                    "cow_tag": row["cow_tag"],
                    "temperature_f": float(row["temperature_f"]),
                }
            )
    return rows


def index_raw_zip(raw_zip):
    index = defaultdict(list)
    with zipfile.ZipFile(raw_zip) as zf:
        for name in zf.namelist():
            match = RAW_NAME_RE.match(name)
            if not match:
                continue
            date, sequence_num, frame_id = match.groups()
            index[(date, sequence_num)].append((int(frame_id), name))
    for key in index:
        index[key].sort(key=lambda item: item[0])
    return index


def choose_evenly_spaced(items, max_items):
    if len(items) <= max_items:
        return items
    positions = np.linspace(0, len(items) - 1, max_items).round().astype(int)
    return [items[int(pos)] for pos in positions]


def read_tiff_array(zf, name):
    with zf.open(name) as f:
        data = f.read()
    try:
        image = Image.open(BytesIO(data))
        array = np.asarray(image, dtype=np.float32)
    except Exception:
        encoded = np.frombuffer(data, dtype=np.uint8)
        array = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if array is None:
            return None
        array = array.astype(np.float32)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def normalize_thermal_for_detector(array, width=2560, height=1440):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        scaled = np.zeros_like(array, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1, 99])
        scaled = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    resized = cv2.resize(scaled, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def describe_values(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    p95 = np.percentile(values, 95)
    top5 = values[values >= p95]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(p95),
        "p99": float(np.percentile(values, 99)),
        "top5_mean": float(np.mean(top5)),
    }


def rect_values(array, rect):
    height, width = array.shape
    x0, y0, x1, y1 = rect
    left = max(0, min(width - 1, int(round(x0))))
    top = max(0, min(height - 1, int(round(y0))))
    right = max(left + 1, min(width, int(round(x1))))
    bottom = max(top + 1, min(height, int(round(y1))))
    return array[top:bottom, left:right].reshape(-1)


def circle_values(array, center, radius):
    height, width = array.shape
    x, y = center
    left = max(0, int(math.floor(x - radius)))
    right = min(width, int(math.ceil(x + radius + 1)))
    top = max(0, int(math.floor(y - radius)))
    bottom = min(height, int(math.ceil(y + radius + 1)))
    if left >= right or top >= bottom:
        return np.asarray([], dtype=np.float32)
    yy, xx = np.ogrid[top:bottom, left:right]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    return array[top:bottom, left:right][mask]


def prefixed_stats(prefix, values):
    return {f"{prefix}_{key}": value for key, value in describe_values(values).items()}


def aggregate_feature_rows(rows):
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    aggregate = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))
        aggregate[f"{key}_min"] = float(np.min(values))
        aggregate[f"{key}_max"] = float(np.max(values))
        aggregate[f"{key}_p95"] = float(np.percentile(values, 95))
    return aggregate


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_feature_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if key in {"date", "sequence_num", "cow_tag"}:
                    parsed[key] = value
                elif value == "":
                    parsed[key] = np.nan
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def metric_dict(y_true, y_pred):
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if len(y_true) > 1:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    return metrics


def make_random_forest_pipeline(feature_count, seed, select_k=None):
    steps = [SimpleImputer(strategy="median")]
    if select_k:
        steps.append(SelectKBest(score_func=f_regression, k=min(select_k, feature_count)))
    steps.append(
        RandomForestRegressor(n_estimators=500, random_state=seed, min_samples_leaf=1)
    )
    return make_pipeline(*steps)


def selected_feature_names(model, feature_names):
    if "selectkbest" not in model.named_steps:
        return feature_names
    mask = model.named_steps["selectkbest"].get_support()
    return [name for name, keep in zip(feature_names, mask) if keep]


def train_random_forest(records, output_dir, seed=1337, test_size=0.2, select_k=None):
    if len(records) < 5:
        raise RuntimeError(f"Need at least 5 records to train/evaluate, found {len(records)}")
    feature_names = [key for key in sorted(records[0]) if key not in ID_COLUMNS]
    x = np.asarray([[record.get(name, np.nan) for name in feature_names] for record in records], dtype=np.float32)
    y = np.asarray([record["temperature_f"] for record in records], dtype=np.float32)

    train_idx, test_idx = train_test_split(
        np.arange(len(records)),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    model = make_random_forest_pipeline(len(feature_names), seed, select_k=select_k)
    model.fit(x[train_idx], y[train_idx])
    test_pred = model.predict(x[test_idx])
    mean_pred = np.full_like(y[test_idx], y[train_idx].mean(), dtype=np.float32)

    folds = min(5, len(records))
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics = []
    for fold, (fold_train, fold_test) in enumerate(cv.split(x), start=1):
        fold_model = make_random_forest_pipeline(
            len(feature_names),
            seed + fold,
            select_k=select_k,
        )
        fold_model.fit(x[fold_train], y[fold_train])
        metrics = metric_dict(y[fold_test], fold_model.predict(x[fold_test]))
        metrics["fold"] = fold
        fold_metrics.append(metrics)

    metrics = {
        "sample_count": len(records),
        "feature_count": len(feature_names),
        "selected_feature_count": int(min(select_k, len(feature_names))) if select_k else len(feature_names),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "random_forest_test": metric_dict(y[test_idx], test_pred),
        "train_mean_baseline_test": metric_dict(y[test_idx], mean_pred),
        "kfold_random_forest": {
            "folds": folds,
            "mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
            "mae_std": float(np.std([m["mae"] for m in fold_metrics])),
            "mse_mean": float(np.mean([m["mse"] for m in fold_metrics])),
            "rmse_mean": float(np.mean([m["rmse"] for m in fold_metrics])),
            "rmse_std": float(np.std([m["rmse"] for m in fold_metrics])),
            "fold_metrics": fold_metrics,
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "temperature_random_forest.joblib")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": [records[int(i)]["date"] + "/" + records[int(i)]["sequence_num"] for i in train_idx],
                "test": [records[int(i)]["date"] + "/" + records[int(i)]["sequence_num"] for i in test_idx],
            },
            f,
            indent=2,
        )

    predictions = []
    for idx, pred in zip(test_idx, test_pred):
        row = records[int(idx)]
        predictions.append(
            {
                "date": row["date"],
                "sequence_num": row["sequence_num"],
                "cow_tag": row["cow_tag"],
                "temperature_f": row["temperature_f"],
                "prediction_f": float(pred),
                "error_f": float(pred - row["temperature_f"]),
            }
        )
    write_csv(output_dir / "predictions.csv", predictions)

    rf = model.named_steps["randomforestregressor"]
    importance_feature_names = selected_feature_names(model, feature_names)
    importances = sorted(
        zip(importance_feature_names, rf.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    write_csv(
        output_dir / "feature_importance.csv",
        [{"feature": feature, "importance": float(score)} for feature, score in importances],
    )
    return metrics
