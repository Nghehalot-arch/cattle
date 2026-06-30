from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
SKLEARN_COMPAT = Path(__file__).resolve().parent / "_sklearn_compat"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if SKLEARN_COMPAT.exists():
    sys.path.insert(0, str(SKLEARN_COMPAT))

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import numpy as np
from PIL import Image

RAW_NAME_RE = __import__("re").compile(r"thermal_raw/([^/]+)/(\d+)_Video_Frame_(\d+)\.tiff$")
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


@dataclass
class SequenceSample:
    date: str
    sequence_num: str
    cow_tag: str
    temperature_f: float
    frames: list[tuple[int, str]]

    @property
    def key(self) -> str:
        return f"{self.date}/{self.sequence_num}"


def index_raw_zip(raw_zip: Path):
    index = {}
    with zipfile.ZipFile(raw_zip) as zf:
        for name in zf.namelist():
            match = RAW_NAME_RE.match(name)
            if not match:
                continue
            date, sequence_num, frame_id = match.groups()
            index.setdefault((date, sequence_num), []).append((int(frame_id), name))
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
        import cv2

        try:
            cv2.setLogLevel(0)
        except AttributeError:
            pass
        encoded = np.frombuffer(data, dtype=np.uint8)
        array = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if array is None:
            return None
        array = array.astype(np.float32)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def normalize_frame(array: np.ndarray, size: int, mode: str, thermal_min: float, thermal_max: float) -> np.ndarray:
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


def load_labeled_sequences(metadata_path: Path, raw_zip: Path) -> tuple[list[SequenceSample], list[dict]]:
    raw_index = index_raw_zip(raw_zip)
    samples = []
    missing = []
    with metadata_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("temperature_f"):
                continue
            frames = raw_index.get((row["date"], row["sequence_num"]), [])
            if not frames:
                missing.append(row)
                continue
            samples.append(
                SequenceSample(
                    date=row["date"],
                    sequence_num=row["sequence_num"],
                    cow_tag=row["cow_tag"],
                    temperature_f=float(row["temperature_f"]),
                    frames=frames,
                )
            )
    return samples, missing


def load_split(path: Path) -> tuple[set[str], set[str]]:
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    return set(split["train_videos"]), set(split["test_videos"])


def read_feature_rows(path: Path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = f"{row['date']}/{row['sequence_num']}"
            parsed = {}
            for name, value in row.items():
                if name in {"date", "sequence_num", "cow_tag"}:
                    parsed[name] = value
                elif value == "":
                    parsed[name] = np.nan
                else:
                    parsed[name] = float(value)
            rows[key] = parsed
    return rows


def read_selected_features(path: Path | None, feature_rows: dict[str, dict[str, object]]) -> list[str]:
    if path is None:
        first = next(iter(feature_rows.values()))
        return [name for name in sorted(first) if name not in ID_COLUMNS]
    names = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["feature"])
    return names


def add_anchor_predictions(
    feature_rows: dict[str, dict[str, object]],
    model_path: Path,
    schema_path: Path,
    feature_name: str,
) -> None:
    import joblib

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    anchor_features = schema["feature_names"]
    keys = sorted(feature_rows)
    matrix = np.asarray(
        [[feature_rows[key].get(name, np.nan) for name in anchor_features] for key in keys],
        dtype=np.float32,
    )
    model = joblib.load(model_path)
    predictions = model.predict(matrix)
    for key, prediction in zip(keys, predictions):
        feature_rows[key][feature_name] = float(prediction)


def build_feature_stats(samples, feature_rows, feature_names):
    matrix = np.asarray(
        [[feature_rows[sample.key].get(name, np.nan) for name in feature_names] for sample in samples],
        dtype=np.float32,
    )
    median = np.nanmedian(matrix, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    matrix = np.where(np.isfinite(matrix), matrix, median)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return median, mean, std


class FusionDataset:
    def __init__(
        self,
        samples,
        raw_zip,
        feature_rows,
        feature_names,
        feature_median,
        feature_mean,
        feature_std,
        max_frames,
        image_size,
        normalize,
        thermal_min,
        thermal_max,
    ):
        self.samples = samples
        self.raw_zip = raw_zip
        self.feature_rows = feature_rows
        self.feature_names = feature_names
        self.feature_median = feature_median
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.max_frames = max_frames
        self.image_size = image_size
        self.normalize = normalize
        self.thermal_min = thermal_min
        self.thermal_max = thermal_max
        self.frame_cache = self._preload_frames()
        self.feature_cache = self._preload_features()

    def __len__(self):
        return len(self.samples)

    def _load_sample_frames(self, sample):
        chosen = choose_evenly_spaced(sample.frames, self.max_frames)
        frames = []
        with zipfile.ZipFile(self.raw_zip) as zf:
            for _, zip_name in chosen:
                array = read_tiff_array(zf, zip_name)
                if array is None:
                    continue
                frames.append(
                    normalize_frame(
                        array,
                        self.image_size,
                        self.normalize,
                        self.thermal_min,
                        self.thermal_max,
                    )
                )
        if not frames:
            raise RuntimeError(f"No readable frames for {sample.key}")
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
        return np.stack(frames[: self.max_frames]).astype(np.float32)

    def _preload_frames(self):
        return {sample.key: self._load_sample_frames(sample) for sample in self.samples}

    def _preload_features(self):
        cache = {}
        for sample in self.samples:
            feature_row = self.feature_rows[sample.key]
            features = np.asarray(
                [feature_row.get(name, np.nan) for name in self.feature_names],
                dtype=np.float32,
            )
            features = np.where(np.isfinite(features), features, self.feature_median)
            cache[sample.key] = ((features - self.feature_mean) / self.feature_std).astype(np.float32)
        return cache

    def __getitem__(self, idx):
        import torch

        sample = self.samples[idx]
        feature_row = self.feature_rows[sample.key]
        return (
            torch.from_numpy(self.frame_cache[sample.key]).unsqueeze(1).float(),
            torch.from_numpy(self.feature_cache[sample.key]).float(),
            torch.tensor(sample.temperature_f, dtype=torch.float32),
            sample.key,
        )


class ThermalFeatureFusionRegressor:
    def __new__(cls, feature_dim: int, dropout: float):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(1, 16, 5, stride=2, padding=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 96, 3, stride=2, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.feature_encoder = nn.Sequential(
                    nn.Linear(feature_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                )
                self.head = nn.Sequential(
                    nn.Linear(96 + 32, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(64, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 1),
                )

            def forward(self, frames, features):
                batch, frame_count, channels, height, width = frames.shape
                encoded = self.image_encoder(frames.reshape(batch * frame_count, channels, height, width))
                encoded = encoded.reshape(batch, frame_count, -1).mean(dim=1)
                feature_embedding = self.feature_encoder(features)
                return self.head(torch.cat([encoded, feature_embedding], dim=1)).squeeze(1)

        return _Model()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def mae_rmse(truth, pred):
    errors = np.asarray(pred, dtype=np.float32) - np.asarray(truth, dtype=np.float32)
    return float(np.mean(np.abs(errors))), float(math.sqrt(np.mean(errors * errors)))


def train_once(args, train_samples, test_samples, feature_rows, feature_names):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    feature_median, feature_mean, feature_std = build_feature_stats(train_samples, feature_rows, feature_names)
    train_ds = FusionDataset(
        train_samples,
        args.raw_zip,
        feature_rows,
        feature_names,
        feature_median,
        feature_mean,
        feature_std,
        args.max_frames,
        args.image_size,
        args.normalize,
        args.thermal_min,
        args.thermal_max,
    )
    test_ds = FusionDataset(
        test_samples,
        args.raw_zip,
        feature_rows,
        feature_names,
        feature_median,
        feature_mean,
        feature_std,
        args.max_frames,
        args.image_size,
        args.normalize,
        args.thermal_min,
        args.thermal_max,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ThermalFeatureFusionRegressor(len(feature_names), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    y_mean = float(np.mean([sample.temperature_f for sample in train_samples]))
    y_std = float(np.std([sample.temperature_f for sample in train_samples]) or 1.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for frames, features, y, _ in train_loader:
            frames = frames.to(device)
            features = features.to(device)
            y = ((y.to(device) - y_mean) / y_std).float()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(frames, features), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.log_period == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_loss={float(np.mean(losses)):.4f}")

    model.eval()
    predictions = []
    with torch.no_grad():
        for frames, features, y, keys in test_loader:
            pred = model(frames.to(device), features.to(device)).cpu().numpy()[0] * y_std + y_mean
            truth = float(y.numpy()[0])
            predictions.append(
                {
                    "sequence": keys[0],
                    "temperature_f": truth,
                    "prediction_f": float(pred),
                    "error_f": float(pred - truth),
                }
            )
    mae, rmse = mae_rmse(
        [row["temperature_f"] for row in predictions],
        [row["prediction_f"] for row in predictions],
    )
    return model, predictions, {"mae": mae, "rmse": rmse}, {
        "feature_names": feature_names,
        "feature_median": feature_median.tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a raw thermal CNN fused with keypoint ROI temperature features.")
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--features", default="data/temperature_outputs/detected_roi_filtered_80_v1/features.csv", type=Path)
    parser.add_argument("--selected-features", type=Path)
    parser.add_argument("--anchor-model", type=Path)
    parser.add_argument("--anchor-schema", type=Path)
    parser.add_argument("--anchor-feature-name", default="roi_anchor_prediction")
    parser.add_argument("--split-metrics", default="data/temperature_outputs/thermal_cnn_absolute_quick_lr1e3_v1/metrics.json", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/thermal_feature_fusion_cnn_v1", type=Path)
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
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train on all usable samples. Metrics are fit diagnostics, not held-out accuracy.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    samples, missing = load_labeled_sequences(args.metadata, args.raw_zip)
    feature_rows = read_feature_rows(args.features)
    if args.anchor_model:
        if not args.anchor_schema:
            raise RuntimeError("--anchor-schema is required when --anchor-model is used.")
        add_anchor_predictions(feature_rows, args.anchor_model, args.anchor_schema, args.anchor_feature_name)
    samples = [sample for sample in samples if sample.key in feature_rows]
    if len(samples) < 5:
        raise RuntimeError(f"Need at least 5 samples with raw frames and ROI features, found {len(samples)}")

    if args.train_all:
        train_samples = list(samples)
        test_samples = list(samples)
        evaluation = "train_fit_not_holdout"
    else:
        train_keys, test_keys = load_split(args.split_metrics)
        train_samples = [sample for sample in samples if sample.key in train_keys]
        test_samples = [sample for sample in samples if sample.key in test_keys]
        evaluation = "heldout"
        if not train_samples or not test_samples:
            raise RuntimeError("Split did not match samples.")
    feature_names = read_selected_features(args.selected_features, feature_rows)
    if args.anchor_model and args.anchor_feature_name not in feature_names:
        feature_names.append(args.anchor_feature_name)

    model, predictions, metrics, state = train_once(args, train_samples, test_samples, feature_rows, feature_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "state": state,
            "args": vars(args),
        },
        args.output_dir / "thermal_feature_fusion_cnn.pt",
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "usable_labeled_videos": len(samples),
                "missing_raw_labeled_videos": len(missing),
                "feature_count": len(feature_names),
                "evaluation": evaluation,
                "train_videos": [sample.key for sample in train_samples],
                "test_videos": [sample.key for sample in test_samples],
                "test": metrics,
            },
            f,
            indent=2,
        )
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "temperature_f", "prediction_f", "error_f"])
        writer.writeheader()
        writer.writerows(predictions)
    print("Saved:", args.output_dir)
    print("Feature count:", len(feature_names))
    print("Test MAE:", metrics["mae"])
    print("Test RMSE:", metrics["rmse"])


if __name__ == "__main__":
    main()
