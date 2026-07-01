from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
SKLEARN_COMPAT = Path(__file__).resolve().parent / "_sklearn_compat"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if SKLEARN_COMPAT.exists():
    sys.path.insert(0, str(SKLEARN_COMPAT))

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold

from common import write_csv
from train_thermal_feature_fusion_cnn import (
    SequenceSample,
    add_anchor_predictions,
    build_feature_stats,
    choose_evenly_spaced,
    load_labeled_sequences,
    load_split,
    normalize_frame,
    read_feature_rows,
    read_selected_features,
    read_tiff_array,
    set_seed,
)


ROI_NAMES = [
    "left_eye",
    "right_eye",
    "left_nostril",
    "right_nostril",
    "forehead",
    "left_ear",
    "right_ear",
]
QUALITY_COLUMNS = [
    "detection_score",
    "frontal_score",
    "face_area_frac",
    "face_aspect",
    "center_offset_x",
    "center_offset_y",
    "required_kp_score_min",
    "required_kp_score_mean",
    "eye_y_diff_norm",
    "nostril_y_diff_norm",
    "eye_width_norm",
    "nostril_width_norm",
    "muzzle_center_offset",
    "muzzle_symmetry",
    "lower_face_order_ok",
]


def parse_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_detection_rows(path: Path) -> dict[tuple[str, str], list[dict[str, object]]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, object] = {
                "date": row["date"],
                "sequence_num": row["sequence_num"],
                "cow_tag": row.get("cow_tag", ""),
                "temperature_f": parse_float(row.get("temperature_f")),
                "frame_id": int(parse_float(row.get("frame_id"), -1)),
            }
            if parsed["frame_id"] < 0:
                continue
            for name in QUALITY_COLUMNS:
                parsed[name] = parse_float(row.get(name), 0.0)
            for roi_name in ROI_NAMES:
                for suffix in ("x0", "y0", "x1", "y1"):
                    key = f"article_{roi_name}_{suffix}"
                    parsed[key] = parse_float(row.get(key))
            rows_by_key.setdefault((row["date"], row["sequence_num"]), []).append(parsed)
    for key in rows_by_key:
        rows_by_key[key].sort(key=lambda row: int(row["frame_id"]))
    return rows_by_key


def sequence_key(date: str, sequence_num: str) -> str:
    return f"{date}/{sequence_num}"


def sample_group(sample: SequenceSample, grouping: str) -> str:
    if grouping == "sequence":
        return sample.key
    if grouping == "cow":
        return sample.cow_tag
    if grouping == "date":
        return sample.date
    raise KeyError(grouping)


def select_detection_rows(
    rows: list[dict[str, object]],
    max_frames: int,
    score_column: str,
    candidate_limit: int | None,
    selection: str,
) -> list[dict[str, object]]:
    candidates = list(rows)
    if candidate_limit:
        candidates = sorted(candidates, key=lambda row: parse_float(row.get(score_column), 0.0), reverse=True)[
            :candidate_limit
        ]
    if selection == "top":
        candidates = sorted(candidates, key=lambda row: parse_float(row.get(score_column), 0.0), reverse=True)
        return sorted(candidates[:max_frames], key=lambda row: int(row["frame_id"]))
    candidates = sorted(candidates, key=lambda row: int(row["frame_id"]))
    return choose_evenly_spaced(candidates, max_frames)


def filter_samples_with_detections(
    samples: list[SequenceSample],
    detections: dict[tuple[str, str], list[dict[str, object]]],
    min_frames: int,
) -> tuple[list[SequenceSample], list[dict[str, object]]]:
    kept = []
    dropped = []
    for sample in samples:
        rows = detections.get((sample.date, sample.sequence_num), [])
        if len(rows) < min_frames:
            dropped.append({"sequence": sample.key, "available_detection_frames": len(rows)})
            continue
        kept.append(sample)
    return kept, dropped


def read_feature_limit(path: Path | None, feature_rows: dict[str, dict[str, object]], limit: int | None) -> list[str]:
    feature_names = read_selected_features(path, feature_rows)
    if limit:
        feature_names = feature_names[:limit]
    return feature_names


def build_quality_stats(
    samples: list[SequenceSample],
    detections: dict[tuple[str, str], list[dict[str, object]]],
    max_frames: int,
    score_column: str,
    candidate_limit: int | None,
    selection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for sample in samples:
        selected = select_detection_rows(
            detections[(sample.date, sample.sequence_num)],
            max_frames,
            score_column,
            candidate_limit,
            selection,
        )
        rows.extend([[parse_float(row.get(name), math.nan) for name in QUALITY_COLUMNS] for row in selected])
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.size == 0:
        matrix = np.zeros((1, len(QUALITY_COLUMNS)), dtype=np.float32)
    median = np.nanmedian(matrix, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    matrix = np.where(np.isfinite(matrix), matrix, median)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return median, mean, std


def normalize_array(array: np.ndarray, size: int, mode: str, thermal_min: float, thermal_max: float) -> np.ndarray:
    if array.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    if mode == "percentile":
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            scaled = np.zeros_like(array, dtype=np.float32)
        else:
            low, high = np.percentile(finite, [1, 99])
            scaled = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    else:
        scaled = np.clip((array - thermal_min) / max(thermal_max - thermal_min, 1e-6), 0, 1)
    image = Image.fromarray((scaled * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def crop_roi(array: np.ndarray, row: dict[str, object], roi_name: str, size: int, mode: str, thermal_min: float, thermal_max: float):
    height, width = array.shape[:2]
    x0 = parse_float(row.get(f"article_{roi_name}_x0"))
    y0 = parse_float(row.get(f"article_{roi_name}_y0"))
    x1 = parse_float(row.get(f"article_{roi_name}_x1"))
    y1 = parse_float(row.get(f"article_{roi_name}_y1"))
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return np.zeros((size, size), dtype=np.float32)
    ix0 = max(0, min(width - 1, int(math.floor(min(x0, x1)))))
    ix1 = max(0, min(width, int(math.ceil(max(x0, x1)))))
    iy0 = max(0, min(height - 1, int(math.floor(min(y0, y1)))))
    iy1 = max(0, min(height, int(math.ceil(max(y0, y1)))))
    if ix1 <= ix0 + 1 or iy1 <= iy0 + 1:
        return np.zeros((size, size), dtype=np.float32)
    return normalize_array(array[iy0:iy1, ix0:ix1], size, mode, thermal_min, thermal_max)


class MultiRoiQualityDataset:
    def __init__(
        self,
        samples,
        raw_zip,
        detections,
        feature_rows,
        feature_names,
        feature_median,
        feature_mean,
        feature_std,
        quality_median,
        quality_mean,
        quality_std,
        max_frames,
        image_size,
        roi_size,
        normalize,
        thermal_min,
        thermal_max,
        score_column,
        frame_candidate_limit,
        frame_selection,
    ):
        self.samples = samples
        self.raw_zip = raw_zip
        self.detections = detections
        self.feature_rows = feature_rows
        self.feature_names = feature_names
        self.feature_median = feature_median
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.quality_median = quality_median
        self.quality_mean = quality_mean
        self.quality_std = quality_std
        self.max_frames = max_frames
        self.image_size = image_size
        self.roi_size = roi_size
        self.normalize = normalize
        self.thermal_min = thermal_min
        self.thermal_max = thermal_max
        self.score_column = score_column
        self.frame_candidate_limit = frame_candidate_limit
        self.frame_selection = frame_selection
        self.tensor_cache = self._preload_tensors()
        self.feature_cache = self._preload_features()

    def __len__(self):
        return len(self.samples)

    def _selected_rows(self, sample):
        return select_detection_rows(
            self.detections[(sample.date, sample.sequence_num)],
            self.max_frames,
            self.score_column,
            self.frame_candidate_limit,
            self.frame_selection,
        )

    def _preload_tensors(self):
        cache = {}
        with zipfile.ZipFile(self.raw_zip) as zf:
            for sample in self.samples:
                frame_map = {frame_id: zip_name for frame_id, zip_name in sample.frames}
                full_frames = []
                roi_frames = []
                quality_rows = []
                for row in self._selected_rows(sample):
                    frame_id = int(row["frame_id"])
                    zip_name = frame_map.get(frame_id)
                    if not zip_name:
                        continue
                    array = read_tiff_array(zf, zip_name)
                    if array is None:
                        continue
                    full_frames.append(
                        normalize_frame(
                            array,
                            self.image_size,
                            self.normalize,
                            self.thermal_min,
                            self.thermal_max,
                        )
                    )
                    roi_frames.append(
                        np.stack(
                            [
                                crop_roi(
                                    array,
                                    row,
                                    roi_name,
                                    self.roi_size,
                                    self.normalize,
                                    self.thermal_min,
                                    self.thermal_max,
                                )
                                for roi_name in ROI_NAMES
                            ]
                        ).astype(np.float32)
                    )
                    quality = np.asarray([parse_float(row.get(name), math.nan) for name in QUALITY_COLUMNS], dtype=np.float32)
                    quality = np.where(np.isfinite(quality), quality, self.quality_median)
                    quality_rows.append(((quality - self.quality_mean) / self.quality_std).astype(np.float32))
                if not full_frames:
                    raise RuntimeError(f"No readable selected frames for {sample.key}")
                while len(full_frames) < self.max_frames:
                    full_frames.append(full_frames[-1])
                    roi_frames.append(roi_frames[-1])
                    quality_rows.append(quality_rows[-1])
                cache[sample.key] = {
                    "full": np.stack(full_frames[: self.max_frames]).astype(np.float32),
                    "rois": np.stack(roi_frames[: self.max_frames]).astype(np.float32),
                    "quality": np.stack(quality_rows[: self.max_frames]).astype(np.float32),
                }
        return cache

    def _preload_features(self):
        cache = {}
        for sample in self.samples:
            feature_row = self.feature_rows[sample.key]
            features = np.asarray([feature_row.get(name, np.nan) for name in self.feature_names], dtype=np.float32)
            features = np.where(np.isfinite(features), features, self.feature_median)
            cache[sample.key] = ((features - self.feature_mean) / self.feature_std).astype(np.float32)
        return cache

    def __getitem__(self, idx):
        import torch

        sample = self.samples[idx]
        tensors = self.tensor_cache[sample.key]
        return (
            torch.from_numpy(tensors["full"]).unsqueeze(1).float(),
            torch.from_numpy(tensors["rois"]).unsqueeze(2).float(),
            torch.from_numpy(tensors["quality"]).float(),
            torch.from_numpy(self.feature_cache[sample.key]).float(),
            torch.tensor(sample.temperature_f, dtype=torch.float32),
            sample.key,
        )


class MultiRoiQualityFusionRegressor:
    def __new__(cls, feature_dim: int, quality_dim: int, dropout: float):
        import torch
        import torch.nn as nn

        def make_encoder(out_dim: int):
            return nn.Sequential(
                nn.Conv2d(1, 12, 3, stride=2, padding=1),
                nn.BatchNorm2d(12),
                nn.ReLU(inplace=True),
                nn.Conv2d(12, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, out_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.full_encoder = make_encoder(64)
                self.eye_encoder = make_encoder(32)
                self.nostril_encoder = make_encoder(32)
                self.forehead_encoder = make_encoder(32)
                self.ear_encoder = make_encoder(32)
                self.quality_encoder = nn.Sequential(
                    nn.Linear(quality_dim, 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(32, 32),
                    nn.ReLU(inplace=True),
                )
                frame_dim = 64 + (32 * 4) + 32
                self.frame_attention = nn.Sequential(
                    nn.Linear(frame_dim, 48),
                    nn.ReLU(inplace=True),
                    nn.Linear(48, 1),
                )
                self.feature_encoder = nn.Sequential(
                    nn.Linear(feature_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                )
                self.head = nn.Sequential(
                    nn.Linear(frame_dim + 32, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(64, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 1),
                )

            def encode_frame_group(self, encoder, crops):
                batch, frame_count, roi_count, channels, height, width = crops.shape
                encoded = encoder(crops.reshape(batch * frame_count * roi_count, channels, height, width))
                encoded = encoded.reshape(batch, frame_count, roi_count, -1).mean(dim=2)
                return encoded

            def forward(self, full_frames, roi_frames, quality, features):
                batch, frame_count, channels, height, width = full_frames.shape
                full = self.full_encoder(full_frames.reshape(batch * frame_count, channels, height, width))
                full = full.reshape(batch, frame_count, -1)
                eyes = self.encode_frame_group(self.eye_encoder, roi_frames[:, :, 0:2])
                nostrils = self.encode_frame_group(self.nostril_encoder, roi_frames[:, :, 2:4])
                forehead = self.encode_frame_group(self.forehead_encoder, roi_frames[:, :, 4:5])
                ears = self.encode_frame_group(self.ear_encoder, roi_frames[:, :, 5:7])
                quality_embedding = self.quality_encoder(quality)
                frame_embedding = torch.cat([full, eyes, nostrils, forehead, ears, quality_embedding], dim=2)
                attention = torch.softmax(self.frame_attention(frame_embedding).squeeze(2), dim=1)
                pooled = torch.sum(frame_embedding * attention.unsqueeze(2), dim=1)
                feature_embedding = self.feature_encoder(features)
                return self.head(torch.cat([pooled, feature_embedding], dim=1)).squeeze(1)

        return _Model()


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


def mae_rmse(truth, pred):
    errors = np.asarray(pred, dtype=np.float32) - np.asarray(truth, dtype=np.float32)
    return float(np.mean(np.abs(errors))), float(math.sqrt(np.mean(errors * errors)))


def train_once(args, train_samples, test_samples, detections, feature_rows, feature_names):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    feature_median, feature_mean, feature_std = build_feature_stats(train_samples, feature_rows, feature_names)
    quality_median, quality_mean, quality_std = build_quality_stats(
        train_samples,
        detections,
        args.max_frames,
        args.frame_score_column,
        args.frame_candidate_limit,
        args.frame_selection,
    )
    train_ds = MultiRoiQualityDataset(
        train_samples,
        args.raw_zip,
        detections,
        feature_rows,
        feature_names,
        feature_median,
        feature_mean,
        feature_std,
        quality_median,
        quality_mean,
        quality_std,
        args.max_frames,
        args.image_size,
        args.roi_size,
        args.normalize,
        args.thermal_min,
        args.thermal_max,
        args.frame_score_column,
        args.frame_candidate_limit,
        args.frame_selection,
    )
    test_ds = MultiRoiQualityDataset(
        test_samples,
        args.raw_zip,
        detections,
        feature_rows,
        feature_names,
        feature_median,
        feature_mean,
        feature_std,
        quality_median,
        quality_mean,
        quality_std,
        args.max_frames,
        args.image_size,
        args.roi_size,
        args.normalize,
        args.thermal_min,
        args.thermal_max,
        args.frame_score_column,
        args.frame_candidate_limit,
        args.frame_selection,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MultiRoiQualityFusionRegressor(len(feature_names), len(QUALITY_COLUMNS), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    y_mean = float(np.mean([sample.temperature_f for sample in train_samples]))
    y_std = float(np.std([sample.temperature_f for sample in train_samples]) or 1.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for full_frames, roi_frames, quality, features, y, _ in train_loader:
            full_frames = full_frames.to(device)
            roi_frames = roi_frames.to(device)
            quality = quality.to(device)
            features = features.to(device)
            y = ((y.to(device) - y_mean) / y_std).float()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(full_frames, roi_frames, quality, features), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.log_period == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_loss={float(np.mean(losses)):.4f}")

    model.eval()
    predictions = []
    with torch.no_grad():
        for full_frames, roi_frames, quality, features, y, keys in test_loader:
            pred = (
                model(
                    full_frames.to(device),
                    roi_frames.to(device),
                    quality.to(device),
                    features.to(device),
                )
                .cpu()
                .numpy()[0]
                * y_std
                + y_mean
            )
            truth = float(y.numpy()[0])
            predictions.append(
                {
                    "sequence": keys[0],
                    "temperature_f": truth,
                    "prediction_f": float(pred),
                    "error_f": float(pred - truth),
                }
            )
    mae, rmse = mae_rmse([row["temperature_f"] for row in predictions], [row["prediction_f"] for row in predictions])
    return model, predictions, {"mae": mae, "rmse": rmse}, {
        "feature_names": feature_names,
        "feature_median": feature_median.tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "quality_columns": QUALITY_COLUMNS,
        "quality_median": quality_median.tolist(),
        "quality_mean": quality_mean.tolist(),
        "quality_std": quality_std.tolist(),
        "roi_names": ROI_NAMES,
        "y_mean": y_mean,
        "y_std": y_std,
    }


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


def run_forced_holdout(args, samples, detections, feature_rows, feature_names, cow_by_key):
    if not args.split_metrics:
        return None
    train_keys, test_keys = load_split(args.split_metrics)
    train_samples = [sample for sample in samples if sample.key in train_keys]
    test_samples = [sample for sample in samples if sample.key in test_keys]
    if not train_samples or not test_samples:
        raise RuntimeError("Forced split did not match samples.")
    _, predictions, _, _ = train_once(args, train_samples, test_samples, detections, feature_rows, feature_names)
    rows = with_prediction_metadata(predictions, None, None, cow_by_key)
    return {
        "train_videos": [sample.key for sample in train_samples],
        "test_videos": [sample.key for sample in test_samples],
        "metrics": metric_dict(rows),
        "predictions": rows,
    }


def run_grouped_validation(args, samples, detections, feature_rows, feature_names, cow_by_key):
    validation = []
    fold_rows_all = []
    prediction_rows_all = []
    for grouping_index, grouping in enumerate(args.groupings):
        groups = np.asarray([sample_group(sample, grouping) for sample in samples])
        unique_groups = sorted(set(groups))
        if len(unique_groups) < 2:
            continue
        folds = min(args.max_folds, len(unique_groups))
        cv = GroupKFold(n_splits=folds)
        group_fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(samples)), groups=groups), start=1):
            fold_seed = args.seed + fold + (1000 * grouping_index)
            set_seed(fold_seed)
            train_samples = [samples[int(idx)] for idx in train_idx]
            test_samples = [samples[int(idx)] for idx in test_idx]
            _, predictions, _, _ = train_once(args, train_samples, test_samples, detections, feature_rows, feature_names)
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


def write_checkpoint(args, output_dir: Path, model, state: dict[str, object]) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "multi_roi_quality_fusion_cnn",
            "state_dict": model.state_dict(),
            "state": state,
            "args": vars(args),
        },
        output_dir / "multi_roi_quality_fusion_cnn.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped CV for a multi-ROI quality-gated thermal fusion CNN.")
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--features", default="data/temperature_outputs/detected_article_otsu_fusion_v1/features.csv", type=Path)
    parser.add_argument("--selected-features", type=Path)
    parser.add_argument("--feature-limit", type=int)
    parser.add_argument("--frame-detections", default="data/temperature_outputs/article_otsu_roi_v1/frame_detections.csv", type=Path)
    parser.add_argument("--frame-score-column", default="frontal_score")
    parser.add_argument("--frame-candidate-limit", type=int)
    parser.add_argument("--frame-selection", choices=("diverse", "top"), default="diverse")
    parser.add_argument("--min-detection-frames", default=1, type=int)
    parser.add_argument("--anchor-model", type=Path)
    parser.add_argument("--anchor-schema", type=Path)
    parser.add_argument("--anchor-feature-name", default="roi_anchor_prediction")
    parser.add_argument("--split-metrics", default="data/temperature_outputs/thermal_cnn_absolute_quick_lr1e3_v1/metrics.json", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/multi_roi_quality_cnn_grouped_v1", type=Path)
    parser.add_argument("--groupings", nargs="*", default=["sequence", "cow", "date"], choices=["sequence", "cow", "date"])
    parser.add_argument("--max-folds", default=5, type=int)
    parser.add_argument("--max-frames", default=8, type=int)
    parser.add_argument("--image-size", default=96, type=int)
    parser.add_argument("--roi-size", default=48, type=int)
    parser.add_argument("--normalize", choices=("absolute", "percentile"), default="absolute")
    parser.add_argument("--thermal-min", default=15.0, type=float)
    parser.add_argument("--thermal-max", default=45.0, type=float)
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--log-period", default=50, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--train-all", action="store_true", help="Train on all usable samples for deployment.")
    args = parser.parse_args()

    set_seed(args.seed)
    samples, missing = load_labeled_sequences(args.metadata, args.raw_zip)
    detections = read_detection_rows(args.frame_detections)
    samples, dropped_detection_filter = filter_samples_with_detections(samples, detections, args.min_detection_frames)
    feature_rows = read_feature_rows(args.features)
    if args.anchor_model:
        if not args.anchor_schema:
            raise RuntimeError("--anchor-schema is required when --anchor-model is used.")
        add_anchor_predictions(feature_rows, args.anchor_model, args.anchor_schema, args.anchor_feature_name)
    samples = [sample for sample in samples if sample.key in feature_rows]
    if len(samples) < 5:
        raise RuntimeError(f"Need at least 5 samples with raw frames, detections, and ROI features, found {len(samples)}")

    feature_names = read_feature_limit(args.selected_features, feature_rows, args.feature_limit)
    if args.anchor_model and args.anchor_feature_name not in feature_names:
        feature_names.append(args.anchor_feature_name)
    cow_by_key = {sample.key: sample.cow_tag for sample in samples}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_all:
        model, predictions, metrics, state = train_once(args, samples, samples, detections, feature_rows, feature_names)
        rows = with_prediction_metadata(predictions, None, None, cow_by_key)
        write_checkpoint(args, args.output_dir, model, state)
        write_csv(args.output_dir / "predictions.csv", rows)
        with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "multi_roi_quality_fusion_cnn",
                    "usable_labeled_videos": len(samples),
                    "missing_raw_labeled_videos": len(missing),
                    "feature_count": len(feature_names),
                    "roi_names": ROI_NAMES,
                    "quality_columns": QUALITY_COLUMNS,
                    "frame_detections": str(args.frame_detections),
                    "dropped_detection_filter": dropped_detection_filter,
                    "evaluation": "train_fit_not_holdout",
                    "train_videos": [sample.key for sample in samples],
                    "test": metrics,
                    "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                },
                f,
                indent=2,
            )
        print("Saved:", args.output_dir)
        print("Feature count:", len(feature_names))
        print("Train-fit MAE:", metrics["mae"])
        print("Train-fit RMSE:", metrics["rmse"])
        return

    holdout = run_forced_holdout(args, samples, detections, feature_rows, feature_names, cow_by_key)
    if holdout is not None:
        write_csv(args.output_dir / "holdout_predictions.csv", holdout["predictions"])

    validation, fold_rows, prediction_rows = run_grouped_validation(
        args,
        samples,
        detections,
        feature_rows,
        feature_names,
        cow_by_key,
    )
    write_csv(
        args.output_dir / "validation_summary.csv",
        [{key: value for key, value in row.items() if key != "fold_metrics"} for row in validation],
    )
    write_csv(args.output_dir / "cv_folds.csv", fold_rows)
    write_csv(args.output_dir / "cv_predictions.csv", prediction_rows)
    for grouping in args.groupings:
        grouping_predictions = [row for row in prediction_rows if row["grouping"] == grouping]
        if grouping_predictions:
            write_csv(args.output_dir / f"cv_{grouping}_predictions.csv", grouping_predictions)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "multi_roi_quality_fusion_cnn",
                "usable_labeled_videos": len(samples),
                "missing_raw_labeled_videos": len(missing),
                "features": str(args.features),
                "selected_features": str(args.selected_features) if args.selected_features else None,
                "feature_count": len(feature_names),
                "roi_names": ROI_NAMES,
                "quality_columns": QUALITY_COLUMNS,
                "frame_detections": str(args.frame_detections),
                "dropped_detection_filter": dropped_detection_filter,
                "feature_names": feature_names,
                "holdout": holdout,
                "validation": validation,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
            f,
            indent=2,
        )

    print("Saved:", args.output_dir)
    for row in validation:
        print(
            f"{row['grouping']} CV:",
            f"MAE={row['mae_mean']:.3f}",
            f"MSE={row['mse_mean']:.3f}",
            f"RMSE={row['rmse_mean']:.3f}",
        )


if __name__ == "__main__":
    main()
