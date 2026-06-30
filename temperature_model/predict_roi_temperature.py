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

from common import (
    aggregate_feature_rows,
    choose_evenly_spaced,
    index_raw_zip,
    normalize_thermal_for_detector,
    read_feature_csv,
    read_tiff_array,
)


def load_package(model_dir: Path, model_name: str):
    model = joblib.load(model_dir / model_name)
    with (model_dir / "feature_schema.json").open("r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, schema["feature_names"]


def predict_record(model, feature_names: list[str], record: dict[str, object]) -> float:
    x = np.asarray([[record.get(name, np.nan) for name in feature_names]], dtype=np.float32)
    return float(model.predict(x)[0])


def print_or_write(rows: list[dict[str, object]], output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(
            "{}/{} predicted_temperature_f={:.3f}".format(
                row["date"],
                row["sequence_num"],
                float(row["prediction_f"]),
            )
        )


def predict_from_features(args, model, feature_names):
    records = read_feature_csv(args.features_csv)
    if args.date:
        records = [record for record in records if record["date"] == args.date]
    if args.sequence_num:
        records = [record for record in records if record["sequence_num"] == args.sequence_num]
    if not records:
        raise RuntimeError("No matching feature rows found.")

    rows = []
    for record in records:
        pred = predict_record(model, feature_names, record)
        row = {
            "date": record["date"],
            "sequence_num": record["sequence_num"],
            "cow_tag": record.get("cow_tag", ""),
            "prediction_f": pred,
        }
        if np.isfinite(float(record.get("temperature_f", np.nan))):
            truth = float(record["temperature_f"])
            row["temperature_f"] = truth
            row["error_f"] = pred - truth
        rows.append(row)
    return rows


def extract_raw_features(args):
    from extract_detected_roi_features import (
        best_instance,
        detection_quality,
        passes_quality_filter,
        roi_features,
        scale_detection,
        setup_predictor,
    )

    raw_index = index_raw_zip(args.raw_zip)
    key = (args.date, args.sequence_num)
    raw_frames = raw_index.get(key, [])
    if not raw_frames:
        raise RuntimeError(f"No raw TIFF frames found for {args.date}/{args.sequence_num}")

    predictor = setup_predictor(args.config_file, args.weights, args.score_threshold)
    detector_size = (args.detector_width, args.detector_height)
    sampled = choose_evenly_spaced(raw_frames, args.max_frames)
    per_frame_features = []
    skipped_unreadable = 0
    skipped_no_detection = 0
    skipped_quality = 0

    with zipfile.ZipFile(args.raw_zip) as zf:
        for _, zip_name in sampled:
            array = read_tiff_array(zf, zip_name)
            if array is None:
                skipped_unreadable += 1
                continue

            detector_image = normalize_thermal_for_detector(
                array,
                width=args.detector_width,
                height=args.detector_height,
            )
            detection = best_instance(predictor(detector_image))
            if detection is None:
                skipped_no_detection += 1
                continue

            bbox, keypoints = scale_detection(detection, array.shape, detector_size)
            quality = detection_quality(array.shape, bbox, keypoints, detection["score"])
            if not passes_quality_filter(quality, args):
                skipped_quality += 1
                continue

            features = roi_features(
                array,
                bbox,
                keypoints,
                args.min_keypoint_score,
                include_all_keypoints=args.include_all_keypoints,
                include_surrounding_ring=args.include_surrounding_ring,
            )
            if args.include_quality_features:
                features.update({f"quality_{key}": value for key, value in quality.items()})
            if features:
                per_frame_features.append(features)

    if not per_frame_features:
        raise RuntimeError(
            "No usable ROI frames after detection/quality filtering "
            f"(unreadable={skipped_unreadable}, no_detection={skipped_no_detection}, quality={skipped_quality})."
        )

    record = {
        "date": args.date,
        "sequence_num": args.sequence_num,
        "cow_tag": args.cow_tag or "",
        "raw_frame_count": len(raw_frames),
        "sampled_frame_count": len(sampled),
        "detected_frame_count": len(per_frame_features),
    }
    record.update(aggregate_feature_rows(per_frame_features))
    return record


def predict_from_raw(args, model, feature_names):
    record = extract_raw_features(args)
    pred = predict_record(model, feature_names, record)
    return [
        {
            "date": record["date"],
            "sequence_num": record["sequence_num"],
            "cow_tag": record.get("cow_tag", ""),
            "prediction_f": pred,
            "raw_frame_count": record["raw_frame_count"],
            "sampled_frame_count": record["sampled_frame_count"],
            "detected_frame_count": record["detected_frame_count"],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict cattle temperature from ROI features or raw thermal TIFF video.")
    parser.add_argument("--model-dir", default="data/temperature_outputs/best_roi_gradient_boosting_v1", type=Path)
    parser.add_argument("--model-name", default="model_full.joblib")
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--date")
    parser.add_argument("--sequence-num")
    parser.add_argument("--cow-tag")
    parser.add_argument("--output", type=Path)

    parser.add_argument("--config-file", default="configs/CattleKeypoints/keypoints_rcnn_R_50_FPN_thermal.yaml", type=Path)
    parser.add_argument("--weights", default="data/train_outputs/thermal_clean_v1/model_final.pth", type=Path)
    parser.add_argument("--max-frames", default=80, type=int)
    parser.add_argument("--detector-width", default=2560, type=int)
    parser.add_argument("--detector-height", default=1440, type=int)
    parser.add_argument("--score-threshold", default=0.0, type=float)
    parser.add_argument("--min-keypoint-score", default=0.0, type=float)
    parser.add_argument("--min-detection-score", default=0.0, type=float)
    parser.add_argument("--min-required-keypoint-score", default=0.0, type=float)
    parser.add_argument("--min-face-area-frac", default=0.02, type=float)
    parser.add_argument("--max-face-area-frac", default=0.75, type=float)
    parser.add_argument("--min-face-aspect", default=0.45, type=float)
    parser.add_argument("--max-face-aspect", default=2.8, type=float)
    parser.add_argument("--max-eye-y-diff", default=0.20, type=float)
    parser.add_argument("--max-nostril-y-diff", default=0.20, type=float)
    parser.add_argument("--max-muzzle-center-offset", default=0.35, type=float)
    parser.add_argument("--max-muzzle-symmetry", default=0.55, type=float)
    parser.add_argument("--min-frontal-score", default=0.25, type=float)
    parser.add_argument("--require-lower-face-order", action="store_true")
    parser.add_argument("--no-quality-filter", dest="quality_filter", action="store_false")
    parser.add_argument("--no-quality-features", dest="include_quality_features", action="store_false")
    parser.add_argument("--include-all-keypoints", action="store_true")
    parser.add_argument("--include-surrounding-ring", action="store_true")
    parser.set_defaults(quality_filter=True, include_quality_features=True)
    args = parser.parse_args()

    model, feature_names = load_package(args.model_dir, args.model_name)
    if args.features_csv:
        rows = predict_from_features(args, model, feature_names)
    else:
        if not args.date or not args.sequence_num:
            raise RuntimeError("Raw prediction requires --date and --sequence-num.")
        rows = predict_from_raw(args, model, feature_names)
    print_or_write(rows, args.output)


if __name__ == "__main__":
    main()
