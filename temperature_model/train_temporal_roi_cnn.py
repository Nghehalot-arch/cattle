from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np


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
AGG_SUFFIXES = ("_mean", "_std", "_min", "_max", "_p95")


def sequence_key(row: dict[str, object]) -> str:
    return f"{row['date']}/{row['sequence_num']}"


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key in {"date", "sequence_num", "cow_tag"}:
                    parsed[key] = value
                elif value == "":
                    parsed[key] = np.nan
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def load_split(path: Path) -> tuple[set[str], set[str]]:
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    return set(split["train_videos"]), set(split["test_videos"])


def selected_base_feature_names(path: Path | None, available_frame_features: list[str]) -> list[str]:
    if path is None:
        return available_frame_features
    selected = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["feature"]
            for suffix in AGG_SUFFIXES:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            selected.append(name)
    selected = [name for name in selected if name in available_frame_features]
    if not selected:
        raise RuntimeError(f"No selected frame features from {path} matched frame feature columns.")
    return list(dict.fromkeys(selected))


def selected_sequence_feature_names(path: Path | None, available_sequence_features: list[str]) -> list[str]:
    if path is None:
        return []
    selected = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["feature"]
            if name in available_sequence_features:
                selected.append(name)
    if not selected:
        raise RuntimeError(f"No selected sequence features from {path} matched aggregate feature columns.")
    return list(dict.fromkeys(selected))


def choose_evenly_spaced(rows: list[dict[str, object]], max_items: int) -> list[dict[str, object]]:
    if len(rows) <= max_items:
        return rows
    positions = np.linspace(0, len(rows) - 1, max_items).round().astype(int)
    return [rows[int(pos)] for pos in positions]


def fit_feature_stats(sequences, keys: list[str], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    for key in keys:
        for row in sequences[key]:
            frames.append([row.get(name, np.nan) for name in feature_names])
    matrix = np.asarray(frames, dtype=np.float32)
    median = np.nanmedian(matrix, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    matrix = np.where(np.isfinite(matrix), matrix, median)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return median, mean, std


def fit_tabular_stats(rows_by_key, keys: list[str], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not feature_names:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    matrix = np.asarray(
        [[rows_by_key[key].get(name, np.nan) for name in feature_names] for key in keys],
        dtype=np.float32,
    )
    median = np.nanmedian(matrix, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    matrix = np.where(np.isfinite(matrix), matrix, median)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return median, mean, std


class TemporalRoiDataset:
    def __init__(
        self,
        keys,
        frame_rows_by_key,
        sequence_rows_by_key,
        frame_feature_names,
        tabular_feature_names,
        frame_median,
        frame_mean,
        frame_std,
        tabular_median,
        tabular_mean,
        tabular_std,
        max_frames,
        augment,
    ):
        self.keys = keys
        self.frame_rows_by_key = frame_rows_by_key
        self.sequence_rows_by_key = sequence_rows_by_key
        self.frame_feature_names = frame_feature_names
        self.tabular_feature_names = tabular_feature_names
        self.frame_median = frame_median
        self.frame_mean = frame_mean
        self.frame_std = frame_std
        self.tabular_median = tabular_median
        self.tabular_mean = tabular_mean
        self.tabular_std = tabular_std
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self):
        return len(self.keys)

    def _select_rows(self, rows):
        if not self.augment or len(rows) <= self.max_frames:
            return choose_evenly_spaced(rows, self.max_frames)
        if len(rows) > self.max_frames:
            positions = sorted(random.sample(range(len(rows)), self.max_frames))
            return [rows[pos] for pos in positions]
        return rows

    def __getitem__(self, index):
        import torch

        key = self.keys[index]
        rows = self._select_rows(self.frame_rows_by_key[key])
        features = np.asarray(
            [[row.get(name, np.nan) for name in self.frame_feature_names] for row in rows],
            dtype=np.float32,
        )
        features = np.where(np.isfinite(features), features, self.frame_median)
        features = (features - self.frame_mean) / self.frame_std
        if len(rows) < self.max_frames:
            pad = np.repeat(features[-1:, :], self.max_frames - len(rows), axis=0)
            features = np.concatenate([features, pad], axis=0)
        features = features[: self.max_frames]

        tabular = np.zeros((0,), dtype=np.float32)
        if self.tabular_feature_names:
            sequence_row = self.sequence_rows_by_key[key]
            tabular = np.asarray(
                [sequence_row.get(name, np.nan) for name in self.tabular_feature_names],
                dtype=np.float32,
            )
            tabular = np.where(np.isfinite(tabular), tabular, self.tabular_median)
            tabular = (tabular - self.tabular_mean) / self.tabular_std

        y = float(rows[0]["temperature_f"])
        return (
            torch.from_numpy(features.T).float(),
            torch.from_numpy(tabular).float(),
            torch.tensor(y, dtype=torch.float32),
            key,
        )


class TemporalRoiCnnRegressor:
    def __new__(cls, frame_feature_count: int, tabular_feature_count: int, dropout: float):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.temporal = nn.Sequential(
                    nn.Conv1d(frame_feature_count, 32, kernel_size=5, padding=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                )
                self.tabular = (
                    nn.Sequential(
                        nn.Linear(tabular_feature_count, 32),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                    )
                    if tabular_feature_count
                    else None
                )
                head_dim = 128 + (32 if tabular_feature_count else 0)
                self.head = nn.Sequential(
                    nn.Linear(head_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(64, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 1),
                )

            def forward(self, frames, tabular):
                encoded = self.temporal(frames)
                avg_pool = encoded.mean(dim=2)
                max_pool = encoded.max(dim=2).values
                parts = [avg_pool, max_pool]
                if self.tabular is not None:
                    parts.append(self.tabular(tabular))
                return self.head(__import__("torch").cat(parts, dim=1)).squeeze(1)

        return _Model()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def mae_mse_rmse(truth, pred) -> dict[str, float]:
    errors = np.asarray(pred, dtype=np.float32) - np.asarray(truth, dtype=np.float32)
    mse = float(np.mean(errors * errors))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a temporal CNN over per-frame ROI/keypoint thermal features.")
    parser.add_argument("--frame-features", default="data/temperature_outputs/detected_roi_filtered_80_v1/frame_features.csv", type=Path)
    parser.add_argument("--sequence-features", default="data/temperature_outputs/detected_roi_filtered_80_v1/features.csv", type=Path)
    parser.add_argument("--selected-features", default="data/temperature_outputs/best_roi_gradient_boosting_v1/selected_features_holdout.csv", type=Path)
    parser.add_argument("--use-all-frame-features", action="store_true")
    parser.add_argument("--no-tabular-features", action="store_true")
    parser.add_argument("--split-metrics", default="data/temperature_outputs/thermal_cnn_absolute_quick_lr1e3_v1/metrics.json", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/temporal_roi_cnn_v1", type=Path)
    parser.add_argument("--max-frames", default=64, type=int)
    parser.add_argument("--epochs", default=350, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--dropout", default=0.20, type=float)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--log-period", default=50, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    frame_rows = read_rows(args.frame_features)
    sequence_rows = read_rows(args.sequence_features)

    by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frame_rows:
        if np.isfinite(float(row.get("temperature_f", np.nan))):
            by_key[sequence_key(row)].append(row)
    for rows in by_key.values():
        rows.sort(key=lambda row: float(row["frame_id"]))

    sequence_by_key = {sequence_key(row): row for row in sequence_rows if np.isfinite(float(row.get("temperature_f", np.nan)))}
    available_keys = sorted(set(by_key) & set(sequence_by_key))
    train_split, test_split = load_split(args.split_metrics)
    train_keys = [key for key in available_keys if key in train_split]
    test_keys = [key for key in available_keys if key in test_split]
    if not train_keys or not test_keys:
        raise RuntimeError("Split did not match frame feature rows.")

    available_frame_features = [name for name in sorted(frame_rows[0]) if name not in ID_COLUMNS]
    available_sequence_features = [name for name in sorted(sequence_rows[0]) if name not in ID_COLUMNS]
    if args.use_all_frame_features:
        frame_feature_names = available_frame_features
    else:
        frame_feature_names = selected_base_feature_names(args.selected_features, available_frame_features)
    tabular_feature_names = (
        []
        if args.no_tabular_features
        else selected_sequence_feature_names(args.selected_features, available_sequence_features)
    )

    frame_median, frame_mean, frame_std = fit_feature_stats(by_key, train_keys, frame_feature_names)
    tabular_median, tabular_mean, tabular_std = fit_tabular_stats(sequence_by_key, train_keys, tabular_feature_names)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    train_ds = TemporalRoiDataset(
        train_keys,
        by_key,
        sequence_by_key,
        frame_feature_names,
        tabular_feature_names,
        frame_median,
        frame_mean,
        frame_std,
        tabular_median,
        tabular_mean,
        tabular_std,
        args.max_frames,
        augment=not args.no_augment,
    )
    test_ds = TemporalRoiDataset(
        test_keys,
        by_key,
        sequence_by_key,
        frame_feature_names,
        tabular_feature_names,
        frame_median,
        frame_mean,
        frame_std,
        tabular_median,
        tabular_mean,
        tabular_std,
        args.max_frames,
        augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TemporalRoiCnnRegressor(len(frame_feature_names), len(tabular_feature_names), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    y_train = [float(by_key[key][0]["temperature_f"]) for key in train_keys]
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) or 1.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for frames, tabular, y, _ in train_loader:
            frames = frames.to(device)
            tabular = tabular.to(device)
            y = ((y.to(device) - y_mean) / y_std).float()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(frames, tabular), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.log_period == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_loss={float(np.mean(losses)):.4f}")

    model.eval()
    predictions = []
    with torch.no_grad():
        for frames, tabular, y, keys in test_loader:
            pred = model(frames.to(device), tabular.to(device)).cpu().numpy()[0] * y_std + y_mean
            truth = float(y.numpy()[0])
            predictions.append(
                {
                    "sequence": keys[0],
                    "temperature_f": truth,
                    "prediction_f": float(pred),
                    "error_f": float(pred - truth),
                }
            )
    metrics = mae_mse_rmse(
        [row["temperature_f"] for row in predictions],
        [row["prediction_f"] for row in predictions],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "frame_feature_names": frame_feature_names,
            "tabular_feature_names": tabular_feature_names,
            "frame_median": frame_median.tolist(),
            "frame_mean": frame_mean.tolist(),
            "frame_std": frame_std.tolist(),
            "tabular_median": tabular_median.tolist(),
            "tabular_mean": tabular_mean.tolist(),
            "tabular_std": tabular_std.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        args.output_dir / "temporal_roi_cnn.pt",
    )
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "temperature_f", "prediction_f", "error_f"])
        writer.writeheader()
        writer.writerows(predictions)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "usable_labeled_videos": len(available_keys),
                "train_videos": train_keys,
                "test_videos": test_keys,
                "frame_feature_count": len(frame_feature_names),
                "tabular_feature_count": len(tabular_feature_names),
                "frame_features": frame_feature_names,
                "tabular_features": tabular_feature_names,
                "test": metrics,
            },
            f,
            indent=2,
        )
    print("Saved:", args.output_dir)
    print("Frame feature count:", len(frame_feature_names))
    print("Tabular feature count:", len(tabular_feature_names))
    print("Test MAE:", metrics["mae"])
    print("Test RMSE:", metrics["rmse"])


if __name__ == "__main__":
    main()
