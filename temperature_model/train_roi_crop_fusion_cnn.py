from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
from PIL import Image

from common import ID_COLUMNS, choose_evenly_spaced, index_raw_zip, read_feature_csv, read_tiff_array


ROI_NAMES = [
    "face_bbox",
    "left_eye",
    "right_eye",
    "muzzle",
    "left_nostril",
    "right_nostril",
    "mouth",
    "nostrils_box",
    "lower_face",
] + [f"kp{index + 1:02d}" for index in range(13)]


def sequence_key(row: dict[str, object]) -> str:
    return f"{row['date']}/{row['sequence_num']}"


def load_split(path: Path) -> tuple[set[str], set[str]]:
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    return set(split["train_videos"]), set(split["test_videos"])


def read_coordinate_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
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


def normalize_crop(array: np.ndarray, crop_size: int, thermal_min: float, thermal_max: float) -> np.ndarray:
    scaled = np.clip((array - thermal_min) / max(thermal_max - thermal_min, 1e-6), 0, 1)
    image = Image.fromarray((scaled * 255).astype(np.uint8)).resize((crop_size, crop_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def crop_rect(array: np.ndarray, rect, crop_size: int, thermal_min: float, thermal_max: float) -> np.ndarray:
    height, width = array.shape
    x0, y0, x1, y1 = rect
    left = max(0, min(width - 1, int(round(x0))))
    top = max(0, min(height - 1, int(round(y0))))
    right = max(left + 1, min(width, int(round(x1))))
    bottom = max(top + 1, min(height, int(round(y1))))
    return normalize_crop(array[top:bottom, left:right], crop_size, thermal_min, thermal_max)


def keypoint(row: dict[str, object], name: str):
    return (
        float(row[f"kp_{name}_x"]),
        float(row[f"kp_{name}_y"]),
        float(row[f"kp_{name}_score"]),
    )


def roi_rect(row: dict[str, object], roi_name: str):
    x0 = float(row["bbox_x0"])
    y0 = float(row["bbox_y0"])
    x1 = float(row["bbox_x1"])
    y1 = float(row["bbox_y1"])
    face_w = max(1.0, x1 - x0)
    face_h = max(1.0, y1 - y0)
    crop_side = max(4.0, min(face_w, face_h) * 0.18)

    if roi_name == "face_bbox":
        return x0, y0, x1, y1

    if roi_name.startswith("kp") and len(roi_name) == 4:
        x = float(row[f"{roi_name}_x"])
        y = float(row[f"{roi_name}_y"])
        return x - crop_side, y - crop_side, x + crop_side, y + crop_side

    if roi_name in {"left_eye", "right_eye", "muzzle", "left_nostril", "right_nostril", "mouth"}:
        x, y, _ = keypoint(row, roi_name)
        return x - crop_side, y - crop_side, x + crop_side, y + crop_side

    if roi_name == "nostrils_box":
        points = [keypoint(row, "left_nostril"), keypoint(row, "right_nostril")]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (
            min(xs) - face_w * 0.08,
            min(ys) - face_h * 0.08,
            max(xs) + face_w * 0.08,
            max(ys) + face_h * 0.08,
        )

    if roi_name == "lower_face":
        points = [
            keypoint(row, "muzzle"),
            keypoint(row, "left_nostril"),
            keypoint(row, "right_nostril"),
            keypoint(row, "mouth"),
        ]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (
            min(xs) - face_w * 0.12,
            min(ys) - face_h * 0.12,
            max(xs) + face_w * 0.12,
            max(ys) + face_h * 0.12,
        )

    raise KeyError(roi_name)


def load_tabular_features(path: Path | None):
    if not path:
        return {}, []
    rows = read_feature_csv(path)
    feature_names = [key for key in sorted(rows[0]) if key not in ID_COLUMNS]
    return {sequence_key(row): row for row in rows}, feature_names


class RoiCropFusionDataset:
    def __init__(
        self,
        keys,
        coordinate_rows,
        raw_index,
        raw_zip,
        tabular_rows,
        tabular_feature_names,
        tabular_mean,
        tabular_std,
        max_frames,
        crop_size,
        thermal_min,
        thermal_max,
    ):
        self.keys = keys
        self.coordinate_rows = coordinate_rows
        self.raw_index = raw_index
        self.raw_zip = raw_zip
        self.tabular_rows = tabular_rows
        self.tabular_feature_names = tabular_feature_names
        self.tabular_mean = tabular_mean
        self.tabular_std = tabular_std
        self.max_frames = max_frames
        self.crop_size = crop_size
        self.thermal_min = thermal_min
        self.thermal_max = thermal_max

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        import torch

        key = self.keys[index]
        rows = choose_evenly_spaced(self.coordinate_rows[key], self.max_frames)
        date, sequence_num = key.split("/")
        frame_map = {frame_id: zip_name for frame_id, zip_name in self.raw_index[(date, sequence_num)]}
        frames = []
        with zipfile.ZipFile(self.raw_zip) as zf:
            for row in rows:
                frame_id = int(row["frame_id"])
                array = read_tiff_array(zf, frame_map[frame_id])
                crops = [
                    crop_rect(array, roi_rect(row, roi_name), self.crop_size, self.thermal_min, self.thermal_max)
                    for roi_name in ROI_NAMES
                ]
                frames.append(np.stack(crops))

        if not frames:
            raise RuntimeError(f"No ROI crop frames for {key}")
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
        crop_tensor = torch.from_numpy(np.stack(frames[: self.max_frames])).float()

        tabular = np.zeros((0,), dtype=np.float32)
        if self.tabular_feature_names:
            feature_row = self.tabular_rows[key]
            tabular = np.asarray(
                [feature_row.get(name, np.nan) for name in self.tabular_feature_names],
                dtype=np.float32,
            )
            tabular = np.where(np.isfinite(tabular), tabular, self.tabular_mean)
            tabular = (tabular - self.tabular_mean) / self.tabular_std

        y = float(rows[0]["temperature_f"])
        return crop_tensor.unsqueeze(2), torch.from_numpy(tabular).float(), torch.tensor(y, dtype=torch.float32), key


class RoiCropFusionRegressor:
    def __new__(cls, roi_count: int, tabular_dim: int):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 32, 5, stride=2, padding=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.tabular = (
                    nn.Sequential(nn.Linear(tabular_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.15))
                    if tabular_dim
                    else None
                )
                head_dim = roi_count * 128 + (64 if tabular_dim else 0)
                self.head = nn.Sequential(
                    nn.Linear(head_dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                    nn.Linear(128, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 1),
                )

            def forward(self, crops, tabular):
                batch, frames, rois, channels, height, width = crops.shape
                encoded = self.encoder(crops.reshape(batch * frames * rois, channels, height, width))
                encoded = encoded.reshape(batch, frames, rois, -1).mean(dim=1)
                parts = [encoded.reshape(batch, -1)]
                if self.tabular is not None:
                    parts.append(self.tabular(tabular))
                return self.head(torch.cat(parts, dim=1)).squeeze(1)

        return _Model()


def mae_rmse(truth, pred):
    errors = np.asarray(pred, dtype=np.float32) - np.asarray(truth, dtype=np.float32)
    return float(np.mean(np.abs(errors))), float(math.sqrt(np.mean(errors * errors)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multi-ROI thermal crop CNN fused with keypoint ROI features.")
    parser.add_argument("--roi-coordinates", required=True, type=Path)
    parser.add_argument("--sequence-features", default="data/temperature_outputs/detected_roi_filtered_80_v1/features.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--split-metrics", default="data/temperature_outputs/thermal_cnn_absolute_quick_lr1e3_v1/metrics.json", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/roi_crop_fusion_cnn_v1", type=Path)
    parser.add_argument("--max-frames", default=16, type=int)
    parser.add_argument("--crop-size", default=64, type=int)
    parser.add_argument("--thermal-min", default=15.0, type=float)
    parser.add_argument("--thermal-max", default=45.0, type=float)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--log-period", default=20, type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.roi_coordinates.exists():
        raise RuntimeError(
            "ROI coordinate file not found. Regenerate ROI features with "
            "extract_detected_roi_features.py --write-roi-coordinates first."
        )

    set_seed(args.seed)
    rows = read_coordinate_rows(args.roi_coordinates)
    by_sequence = defaultdict(list)
    for row in rows:
        if np.isfinite(float(row.get("temperature_f", np.nan))):
            by_sequence[sequence_key(row)].append(row)

    train_keys, test_keys = load_split(args.split_metrics)
    keys = sorted(by_sequence)
    train_keys = [key for key in keys if key in train_keys]
    test_keys = [key for key in keys if key in test_keys]
    if not train_keys or not test_keys:
        raise RuntimeError("Split did not match ROI coordinate rows.")

    tabular_rows, tabular_feature_names = load_tabular_features(args.sequence_features)
    if tabular_feature_names:
        train_tabular = np.asarray(
            [
                [tabular_rows[key].get(name, np.nan) for name in tabular_feature_names]
                for key in train_keys
                if key in tabular_rows
            ],
            dtype=np.float32,
        )
        tabular_mean = np.nanmedian(train_tabular, axis=0)
        tabular_mean = np.where(np.isfinite(tabular_mean), tabular_mean, 0.0).astype(np.float32)
        train_tabular = np.where(np.isfinite(train_tabular), train_tabular, tabular_mean)
        tabular_std = train_tabular.std(axis=0)
        tabular_std = np.where(tabular_std > 1e-6, tabular_std, 1.0).astype(np.float32)
    else:
        tabular_mean = np.zeros((0,), dtype=np.float32)
        tabular_std = np.ones((0,), dtype=np.float32)

    raw_index = index_raw_zip(args.raw_zip)
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    train_ds = RoiCropFusionDataset(
        train_keys,
        by_sequence,
        raw_index,
        args.raw_zip,
        tabular_rows,
        tabular_feature_names,
        tabular_mean,
        tabular_std,
        args.max_frames,
        args.crop_size,
        args.thermal_min,
        args.thermal_max,
    )
    test_ds = RoiCropFusionDataset(
        test_keys,
        by_sequence,
        raw_index,
        args.raw_zip,
        tabular_rows,
        tabular_feature_names,
        tabular_mean,
        tabular_std,
        args.max_frames,
        args.crop_size,
        args.thermal_min,
        args.thermal_max,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = RoiCropFusionRegressor(len(ROI_NAMES), len(tabular_feature_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    y_train = [float(by_sequence[key][0]["temperature_f"]) for key in train_keys]
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) or 1.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for crops, tabular, y, _ in train_loader:
            crops = crops.to(device)
            tabular = tabular.to(device)
            y = ((y.to(device) - y_mean) / y_std).float()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(crops, tabular), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.log_period == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_loss={float(np.mean(losses)):.4f}")

    model.eval()
    predictions = []
    with torch.no_grad():
        for crops, tabular, y, keys_batch in test_loader:
            pred = model(crops.to(device), tabular.to(device)).cpu().numpy()[0] * y_std + y_mean
            truth = float(y.numpy()[0])
            predictions.append(
                {
                    "sequence": keys_batch[0],
                    "temperature_f": truth,
                    "prediction_f": float(pred),
                    "error_f": float(pred - truth),
                }
            )
    mae, rmse = mae_rmse(
        [row["temperature_f"] for row in predictions],
        [row["prediction_f"] for row in predictions],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "roi_names": ROI_NAMES,
            "tabular_feature_names": tabular_feature_names,
            "tabular_mean": tabular_mean,
            "tabular_std": tabular_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "args": vars(args),
        },
        args.output_dir / "roi_crop_fusion_cnn.pt",
    )
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "temperature_f", "prediction_f", "error_f"])
        writer.writeheader()
        writer.writerows(predictions)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_videos": train_keys,
                "test_videos": test_keys,
                "roi_names": ROI_NAMES,
                "tabular_feature_count": len(tabular_feature_names),
                "test": {"mae": mae, "rmse": rmse},
            },
            f,
            indent=2,
        )
    print("Saved:", args.output_dir)
    print("Test MAE:", mae)
    print("Test RMSE:", rmse)


if __name__ == "__main__":
    main()
