from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
SKLEARN_COMPAT = Path(__file__).resolve().parent / "_sklearn_compat"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if SKLEARN_COMPAT.exists():
    sys.path.insert(0, str(SKLEARN_COMPAT))

import joblib
import numpy as np
import torch

from train_thermal_feature_fusion_cnn import (
    ThermalFeatureFusionRegressor,
    choose_evenly_spaced,
    filter_frames_for_key,
    index_raw_zip,
    normalize_frame,
    read_frame_filter,
    read_tiff_array,
)

if SKLEARN_COMPAT.exists():
    sys.path.insert(0, str(SKLEARN_COMPAT))


def sequence_key(date: str, sequence_num: str) -> str:
    return f"{date}/{sequence_num}"


def read_feature_row(path: Path, date: str, sequence_num: str) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["date"] != date or row["sequence_num"] != sequence_num:
                continue
            parsed = {}
            for key, value in row.items():
                if key in {"date", "sequence_num", "cow_tag"}:
                    parsed[key] = value
                elif value == "":
                    parsed[key] = np.nan
                else:
                    parsed[key] = float(value)
            return parsed
    raise RuntimeError(f"No feature row found for {date}/{sequence_num}")


def load_raw_frame_tensor(raw_zip: Path, date: str, sequence_num: str, model_args: dict[str, object]) -> torch.Tensor:
    raw_index = index_raw_zip(raw_zip)
    frames = raw_index.get((date, sequence_num), [])
    if not frames:
        raise RuntimeError(f"No raw TIFF frames found for {date}/{sequence_num}")

    frame_filter_csv = model_args.get("frame_filter_csv")
    if frame_filter_csv and str(frame_filter_csv) != "None":
        frame_filter = read_frame_filter(Path(frame_filter_csv), str(model_args.get("frame_score_column", "frontal_score")))
        candidate_limit = model_args.get("frame_candidate_limit")
        candidate_limit = int(candidate_limit) if candidate_limit not in {None, "None"} else None
        filtered_frames = filter_frames_for_key(frames, date, sequence_num, frame_filter, candidate_limit)
        if filtered_frames:
            frames = filtered_frames

    max_frames = int(model_args["max_frames"])
    image_size = int(model_args["image_size"])
    normalize = str(model_args.get("normalize", "absolute"))
    thermal_min = float(model_args.get("thermal_min", 15.0))
    thermal_max = float(model_args.get("thermal_max", 45.0))

    selected = choose_evenly_spaced(frames, max_frames)
    arrays = []
    with zipfile.ZipFile(raw_zip) as zf:
        for _, zip_name in selected:
            array = read_tiff_array(zf, zip_name)
            if array is None:
                continue
            arrays.append(normalize_frame(array, image_size, normalize, thermal_min, thermal_max))
    if not arrays:
        raise RuntimeError(f"No readable raw TIFF frames for {date}/{sequence_num}")
    while len(arrays) < max_frames:
        arrays.append(arrays[-1])
    return torch.from_numpy(np.stack(arrays[:max_frames])).unsqueeze(0).unsqueeze(2).float()


def classical_prediction(model_dir: Path, row: dict[str, object]) -> float:
    model = joblib.load(model_dir / "model_full.joblib")
    with (model_dir / "feature_schema.json").open("r", encoding="utf-8") as f:
        schema = json.load(f)
    feature_names = schema["feature_names"]
    x = np.asarray([[row.get(name, np.nan) for name in feature_names]], dtype=np.float32)
    return float(model.predict(x)[0])


def fusion_prediction(model_dir: Path, raw_zip: Path, date: str, sequence_num: str, row: dict[str, object]) -> float:
    checkpoint = torch.load(model_dir / "thermal_feature_fusion_cnn.pt", map_location="cpu", weights_only=False)
    state = checkpoint["state"]
    model_args = checkpoint["args"]
    feature_names = state["feature_names"]
    feature_median = np.asarray(state["feature_median"], dtype=np.float32)
    feature_mean = np.asarray(state["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(state["feature_std"], dtype=np.float32)

    features = np.asarray([row.get(name, np.nan) for name in feature_names], dtype=np.float32)
    features = np.where(np.isfinite(features), features, feature_median)
    features = ((features - feature_mean) / feature_std).reshape(1, -1)

    frames = load_raw_frame_tensor(raw_zip, date, sequence_num, model_args)
    model = ThermalFeatureFusionRegressor(len(feature_names), float(model_args.get("dropout", 0.15)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        scaled = float(model(frames, torch.from_numpy(features).float()).numpy()[0])
    return scaled * float(state["y_std"]) + float(state["y_mean"])


def write_output(path: Path | None, rows: list[dict[str, object]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict cattle temperature without rectal input using the deployed ROI + fusion system."
    )
    parser.add_argument("--date", required=True)
    parser.add_argument("--sequence-num", required=True)
    parser.add_argument("--features-csv", default="data/temperature_outputs/detected_roi_filtered_80_v1/features.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--roi-model-dir", default="data/temperature_outputs/best_roi_gradient_boosting_v1", type=Path)
    parser.add_argument(
        "--fusion-model-dirs",
        nargs="*",
        type=Path,
        default=[
            Path("data/temperature_outputs/deployment_fusion_cnn_f8_full_v1"),
            Path("data/temperature_outputs/deployment_fusion_cnn_f12_full_v1"),
        ],
    )
    parser.add_argument(
        "--ensemble-weights",
        nargs="*",
        type=float,
        help="Optional weights for ROI prediction followed by each fusion model prediction.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show-truth-if-present", action="store_true")
    args = parser.parse_args()

    row = read_feature_row(args.features_csv, args.date, args.sequence_num)
    predictions = []
    predictions.append(("roi_gradient_boosting_full", classical_prediction(args.roi_model_dir, row)))
    for model_dir in args.fusion_model_dirs:
        predictions.append((model_dir.name, fusion_prediction(model_dir, args.raw_zip, args.date, args.sequence_num, row)))

    if args.ensemble_weights:
        if len(args.ensemble_weights) != len(predictions):
            raise RuntimeError(
                "--ensemble-weights must match ROI plus fusion model count "
                f"({len(predictions)} weights required)."
            )
        weights = np.asarray(args.ensemble_weights, dtype=np.float32)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise RuntimeError("--ensemble-weights must sum to a positive value.")
        weights = weights / weight_sum
        ensemble = float(np.dot(weights, np.asarray([prediction for _, prediction in predictions], dtype=np.float32)))
        ensemble_name = "ensemble_weighted"
    else:
        weights = np.full((len(predictions),), 1.0 / len(predictions), dtype=np.float32)
        ensemble = float(np.mean([prediction for _, prediction in predictions]))
        ensemble_name = "ensemble_mean"
    output_rows = [
        {
            "sequence": sequence_key(args.date, args.sequence_num),
            "model": name,
            "prediction_f": prediction,
        }
        for name, prediction in predictions
    ]
    output_rows.append(
        {
            "sequence": sequence_key(args.date, args.sequence_num),
            "model": ensemble_name,
            "prediction_f": ensemble,
        }
    )
    for item, weight in zip(output_rows[: len(predictions)], weights):
        item["ensemble_weight"] = float(weight)

    if args.show_truth_if_present and np.isfinite(float(row.get("temperature_f", np.nan))):
        truth = float(row["temperature_f"])
        for item in output_rows:
            item["temperature_f"] = truth
            item["error_f"] = float(item["prediction_f"]) - truth

    write_output(args.output, output_rows)
    for item in output_rows:
        print(f"{item['sequence']} {item['model']} prediction_f={float(item['prediction_f']):.3f}")


if __name__ == "__main__":
    main()
