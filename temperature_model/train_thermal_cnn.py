from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
from PIL import Image

RAW_NAME_RE = re.compile(r"thermal_raw/([^/]+)/(\d+)_Video_Frame_(\d+)\.tiff$")


def index_raw_zip(raw_zip):
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


class ThermalSequenceDataset:
    def __init__(self, samples, raw_zip, max_frames, image_size, normalize, thermal_min, thermal_max):
        self.samples = samples
        self.raw_zip = raw_zip
        self.max_frames = max_frames
        self.image_size = image_size
        self.normalize = normalize
        self.thermal_min = thermal_min
        self.thermal_max = thermal_max

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch

        sample = self.samples[idx]
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
        x = torch.from_numpy(np.stack(frames[: self.max_frames])).unsqueeze(1)
        y = torch.tensor(sample.temperature_f, dtype=torch.float32)
        return x, y, sample.key


class SequenceCnnRegressor:
    def __new__(cls):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 16, 5, stride=2, padding=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.head = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.15),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                batch, frames, channels, height, width = x.shape
                encoded = self.encoder(x.reshape(batch * frames, channels, height, width))
                encoded = encoded.reshape(batch, frames, -1).mean(dim=1)
                return self.head(encoded).squeeze(1)

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


def split_samples(samples, test_size, seed):
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    test_count = max(1, int(round(len(ordered) * test_size)))
    return ordered[test_count:], ordered[:test_count]


def mae_rmse(truth, pred):
    errors = [p - t for t, p in zip(truth, pred)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    return mae, rmse


def train_once(args, train_samples, test_samples):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    class NullWriter:
        def add_scalar(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            print(f"TensorBoard disabled: {exc}")
            SummaryWriter = None
    else:
        SummaryWriter = None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = ThermalSequenceDataset(
        train_samples,
        args.raw_zip,
        args.max_frames,
        args.image_size,
        args.normalize,
        args.thermal_min,
        args.thermal_max,
    )
    test_ds = ThermalSequenceDataset(
        test_samples,
        args.raw_zip,
        args.max_frames,
        args.image_size,
        args.normalize,
        args.thermal_min,
        args.thermal_max,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = SequenceCnnRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    writer = (
        SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
        if SummaryWriter is not None
        else NullWriter()
    )

    y_mean = float(np.mean([sample.temperature_f for sample in train_samples]))
    y_std = float(np.std([sample.temperature_f for sample in train_samples]) or 1.0)

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, y, _ in train_loader:
                x = x.to(device)
                y = ((y.to(device) - y_mean) / y_std).float()
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(losses))
            writer.add_scalar("loss/train", train_loss, epoch)
            if epoch == 1 or epoch % args.log_period == 0 or epoch == args.epochs:
                print(f"epoch={epoch:04d} train_loss={train_loss:.4f}")
    finally:
        writer.flush()

    model.eval()
    predictions = []
    with torch.no_grad():
        for x, y, keys in test_loader:
            pred = model(x.to(device)).cpu().numpy()[0] * y_std + y_mean
            predictions.append(
                {
                    "sequence": keys[0],
                    "temperature_f": float(y.numpy()[0]),
                    "prediction_f": float(pred),
                    "error_f": float(pred - y.numpy()[0]),
                }
            )
    mae, rmse = mae_rmse(
        [row["temperature_f"] for row in predictions],
        [row["prediction_f"] for row in predictions],
    )
    writer.add_scalar("metrics/test_mae_f", mae, args.epochs)
    writer.add_scalar("metrics/test_rmse_f", rmse, args.epochs)
    writer.close()
    return model, predictions, {"mae": mae, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser(description="Train a sequence-level CNN on raw thermal TIFF videos.")
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/thermal_cnn_v1", type=Path)
    parser.add_argument("--max-frames", default=16, type=int)
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--normalize", choices=("absolute", "percentile"), default="absolute")
    parser.add_argument("--thermal-min", default=15.0, type=float)
    parser.add_argument("--thermal-max", default=45.0, type=float)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--log-period", default=25, type=int)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    samples, missing = load_labeled_sequences(args.metadata, args.raw_zip)
    if len(samples) < 5:
        raise RuntimeError(f"Need at least 5 labeled videos with raw frames, found {len(samples)}")

    train_samples, test_samples = split_samples(samples, args.test_size, args.seed)
    model, predictions, metrics = train_once(args, train_samples, test_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(model.state_dict(), args.output_dir / "thermal_sequence_cnn.pt")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "usable_labeled_videos": len(samples),
                "missing_raw_labeled_videos": len(missing),
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
    print("Usable labeled videos:", len(samples))
    print("Missing raw labeled videos:", len(missing))
    print("Test MAE:", metrics["mae"])
    print("Test RMSE:", metrics["rmse"])


if __name__ == "__main__":
    main()
