from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))


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


def read_rows(path: Path) -> list[dict[str, object]]:
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


def sequence_key(row: dict[str, object]) -> str:
    return f"{row['date']}/{row['sequence_num']}"


def load_split(path: Path) -> tuple[set[str], set[str]]:
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    return set(split["train_videos"]), set(split["test_videos"])


class MlpRegressor:
    def __new__(cls, input_dim: int, dropout: float):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = min(128, max(16, input_dim // 4))
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, max(8, hidden // 2)),
                    nn.ReLU(inplace=True),
                    nn.Linear(max(8, hidden // 2), 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(1)

        return _Model()


def metrics(y_true, y_pred):
    errors = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(np.mean(errors * errors))),
    }


def write_predictions(path: Path, rows, y_true, y_pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sequence", "temperature_f", "prediction_f", "error_f"],
        )
        writer.writeheader()
        for row, truth, pred in zip(rows, y_true, y_pred):
            writer.writerow(
                {
                    "sequence": sequence_key(row),
                    "temperature_f": float(truth),
                    "prediction_f": float(pred),
                    "error_f": float(pred - truth),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch MLP on keypoint ROI temperature features."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--split-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--dropout", default=0.15, type=float)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--log-period", default=100, type=int)
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = read_rows(args.features)
    train_keys, test_keys = load_split(args.split_metrics)
    train_rows = [row for row in rows if sequence_key(row) in train_keys]
    test_rows = [row for row in rows if sequence_key(row) in test_keys]
    if not train_rows or not test_rows:
        raise RuntimeError("Split did not match feature rows.")

    feature_names = [key for key in sorted(rows[0]) if key not in ID_COLUMNS]
    x_train = np.asarray(
        [[row.get(name, np.nan) for name in feature_names] for row in train_rows],
        dtype=np.float32,
    )
    x_test = np.asarray(
        [[row.get(name, np.nan) for name in feature_names] for row in test_rows],
        dtype=np.float32,
    )
    y_train = np.asarray([row["temperature_f"] for row in train_rows], dtype=np.float32)
    y_test = np.asarray([row["temperature_f"] for row in test_rows], dtype=np.float32)

    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_test = np.where(np.isfinite(x_test), x_test, med)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    y_mean = float(y_train.mean())
    y_std = float(y_train.std() or 1.0)
    y_train_scaled = (y_train - y_mean) / y_std

    model = MlpRegressor(x_train.shape[1], args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train_scaled)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.log_period == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_loss={float(loss.detach()):.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test)).numpy() * y_std + y_mean

    result = metrics(y_test, pred)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "roi_feature_mlp.pt")
    write_predictions(args.output_dir / "predictions.csv", test_rows, y_test, pred)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_rows": len(rows),
                "feature_count": len(feature_names),
                "train_videos": [sequence_key(row) for row in train_rows],
                "test_videos": [sequence_key(row) for row in test_rows],
                "test": result,
            },
            f,
            indent=2,
        )
    print("Saved:", args.output_dir)
    print("Feature rows:", len(rows))
    print("Feature count:", len(feature_names))
    print("Test MAE:", result["mae"])
    print("Test RMSE:", result["rmse"])


if __name__ == "__main__":
    main()
