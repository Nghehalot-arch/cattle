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


ID_COLUMNS = {"date", "sequence_num", "cow_tag", "temperature_f", "frame_id"}
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


def read_selected_bases(path: Path, available: list[str]) -> list[str]:
    selected = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["feature"]
            for suffix in AGG_SUFFIXES:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            if name in available:
                selected.append(name)
    if not selected:
        raise RuntimeError(f"No selected features from {path} matched frame feature columns.")
    return list(dict.fromkeys(selected))


def split_indices(rows, test_size: float, seed: int, grouped: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if not grouped:
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_size)))
        return indices[test_count:], indices[:test_count]

    groups = sorted({sequence_key(row) for row in rows})
    rng.shuffle(groups)
    test_count = max(1, int(round(len(groups) * test_size)))
    test_groups = set(groups[:test_count])
    train_idx = [index for index, row in enumerate(rows) if sequence_key(row) not in test_groups]
    test_idx = [index for index, row in enumerate(rows) if sequence_key(row) in test_groups]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


class FrameFeatureDataset:
    def __init__(self, rows, indices, feature_names, median, mean, std, noise_std: float):
        self.rows = rows
        self.indices = list(indices)
        self.feature_names = feature_names
        self.median = median
        self.mean = mean
        self.std = std
        self.noise_std = noise_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        import torch

        row = self.rows[self.indices[item]]
        values = np.asarray([row.get(name, np.nan) for name in self.feature_names], dtype=np.float32)
        values = np.where(np.isfinite(values), values, self.median)
        values = (values - self.mean) / self.std
        if self.noise_std > 0:
            values = values + np.random.normal(0.0, self.noise_std, values.shape).astype(np.float32)
        return (
            torch.from_numpy(values[None, :]).float(),
            torch.tensor(float(row["temperature_f"]), dtype=torch.float32),
            sequence_key(row),
            int(row["frame_id"]),
        )


class FrameRoiFeatureCnn:
    def __new__(cls, feature_count: int, dropout: float):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=5, padding=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(64, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.head(self.encoder(x).squeeze(-1)).squeeze(1)

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


def metric_dict(truth, pred) -> dict[str, float]:
    errors = np.asarray(pred, dtype=np.float32) - np.asarray(truth, dtype=np.float32)
    mse = float(np.mean(errors * errors))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an article-style frame-level CNN over ROI thermal features.")
    parser.add_argument("--frame-features", default="data/temperature_outputs/detected_roi_filtered_80_v1/frame_features.csv", type=Path)
    parser.add_argument("--selected-features", default="data/temperature_outputs/best_roi_gradient_boosting_v1/selected_features_holdout.csv", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/frame_roi_feature_cnn_v1", type=Path)
    parser.add_argument("--use-all-features", action="store_true")
    parser.add_argument("--grouped-by-sequence", action="store_true")
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--epochs", default=180, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--dropout", default=0.15, type=float)
    parser.add_argument("--noise-std", default=0.02, type=float)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--log-period", default=30, type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    rows = [row for row in read_rows(args.frame_features) if np.isfinite(float(row.get("temperature_f", np.nan)))]
    if len(rows) < 20:
        raise RuntimeError(f"Need at least 20 labeled frame rows, found {len(rows)}")

    available = [name for name in sorted(rows[0]) if name not in ID_COLUMNS]
    feature_names = available if args.use_all_features else read_selected_bases(args.selected_features, available)
    train_idx, test_idx = split_indices(rows, args.test_size, args.seed, args.grouped_by_sequence)

    x_train = np.asarray(
        [[rows[index].get(name, np.nan) for name in feature_names] for index in train_idx],
        dtype=np.float32,
    )
    median = np.nanmedian(x_train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    train_ds = FrameFeatureDataset(rows, train_idx, feature_names, median, mean, std, args.noise_std)
    test_ds = FrameFeatureDataset(rows, test_idx, feature_names, median, mean, std, 0.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = FrameRoiFeatureCnn(len(feature_names), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    y_train = np.asarray([float(rows[index]["temperature_f"]) for index in train_idx], dtype=np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() or 1.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, y, _, _ in train_loader:
            features = features.to(device)
            y = ((y.to(device) - y_mean) / y_std).float()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.log_period == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_loss={float(np.mean(losses)):.4f}")

    frame_predictions = []
    model.eval()
    with torch.no_grad():
        for features, y, keys, frame_ids in test_loader:
            preds = model(features.to(device)).cpu().numpy() * y_std + y_mean
            for key, frame_id, truth, pred in zip(keys, frame_ids.numpy(), y.numpy(), preds):
                frame_predictions.append(
                    {
                        "sequence": key,
                        "frame_id": int(frame_id),
                        "temperature_f": float(truth),
                        "prediction_f": float(pred),
                        "error_f": float(pred - truth),
                    }
                )

    by_sequence = defaultdict(list)
    for row in frame_predictions:
        by_sequence[row["sequence"]].append(row)
    sequence_predictions = []
    for key, items in sorted(by_sequence.items()):
        truth = float(items[0]["temperature_f"])
        pred = float(np.mean([item["prediction_f"] for item in items]))
        sequence_predictions.append(
            {
                "sequence": key,
                "temperature_f": truth,
                "prediction_f": pred,
                "error_f": pred - truth,
                "test_frame_count": len(items),
            }
        )

    frame_metrics = metric_dict(
        [row["temperature_f"] for row in frame_predictions],
        [row["prediction_f"] for row in frame_predictions],
    )
    sequence_metrics = metric_dict(
        [row["temperature_f"] for row in sequence_predictions],
        [row["prediction_f"] for row in sequence_predictions],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "median": median.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        args.output_dir / "frame_roi_feature_cnn.pt",
    )
    with (args.output_dir / "frame_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "frame_id", "temperature_f", "prediction_f", "error_f"])
        writer.writeheader()
        writer.writerows(frame_predictions)
    with (args.output_dir / "sequence_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sequence", "temperature_f", "prediction_f", "error_f", "test_frame_count"],
        )
        writer.writeheader()
        writer.writerows(sequence_predictions)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluation": "grouped_by_sequence" if args.grouped_by_sequence else "article_style_frame_split",
                "frame_count": len(rows),
                "train_frame_count": int(len(train_idx)),
                "test_frame_count": int(len(test_idx)),
                "train_sequence_count": len({sequence_key(rows[index]) for index in train_idx}),
                "test_sequence_count": len({sequence_key(rows[index]) for index in test_idx}),
                "feature_count": len(feature_names),
                "features": feature_names,
                "frame_test": frame_metrics,
                "sequence_average_test": sequence_metrics,
            },
            f,
            indent=2,
        )
    print("Saved:", args.output_dir)
    print("Evaluation:", "grouped_by_sequence" if args.grouped_by_sequence else "article_style_frame_split")
    print("Feature count:", len(feature_names))
    print("Frame MAE:", frame_metrics["mae"])
    print("Frame RMSE:", frame_metrics["rmse"])
    print("Sequence-average MAE:", sequence_metrics["mae"])
    print("Sequence-average RMSE:", sequence_metrics["rmse"])


if __name__ == "__main__":
    main()
