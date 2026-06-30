from __future__ import annotations

import argparse
import csv
import site
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def preload_detector_torch_stack() -> None:
    """Prefer the active environment's Torch stack before common.py adds vendor paths."""
    try:
        user_site = str(Path(site.getusersitepackages()).resolve()).lower()
    except Exception:
        user_site = ""
    if user_site:
        filtered = []
        for path in sys.path:
            try:
                resolved = str(Path(path).resolve()).lower()
            except Exception:
                resolved = path.lower()
            if resolved != user_site:
                filtered.append(path)
        sys.path[:] = filtered

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ModuleNotFoundError:
        return


preload_detector_torch_stack()

from common import (
    aggregate_feature_rows,
    choose_evenly_spaced,
    describe_values,
    index_raw_zip,
    load_temperature_metadata,
    normalize_thermal_for_detector,
    prefixed_stats,
    read_tiff_array,
    write_csv,
)
from extract_detected_roi_features import (
    ALL_KEYPOINT_INDEX,
    best_instance,
    detection_quality,
    passes_quality_filter,
    roi_coordinate_record,
    scale_detection,
    setup_predictor,
)


KP = {
    "left_ear_base": 0,
    "left_ear_middle": 1,
    "left_ear_tip": 2,
    "poll": 3,
    "right_ear_base": 4,
    "right_ear_middle": 5,
    "right_ear_tip": 6,
    "left_eye": 7,
    "right_eye": 8,
    "muzzle": 9,
    "left_nostril": 10,
    "right_nostril": 11,
    "mouth": 12,
}


FEATURE_ID_COLUMNS = [
    "date",
    "sequence_num",
    "cow_tag",
    "temperature_f",
    "raw_frame_count",
    "sampled_frame_count",
    "detected_frame_count",
]


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values[np.isfinite(values)]


def write_empty_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def clipped_rect(array: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = array.shape
    left = max(0, min(width - 1, int(round(x0))))
    top = max(0, min(height - 1, int(round(y0))))
    right = max(left + 1, min(width, int(round(x1))))
    bottom = max(top + 1, min(height, int(round(y1))))
    return array[top:bottom, left:right], (left, top, right, bottom)


def centered_box(point: np.ndarray, width: float, height: float) -> tuple[float, float, float, float]:
    x, y = float(point[0]), float(point[1])
    return x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0


def top_center_box(point: np.ndarray, width: float, height: float) -> tuple[float, float, float, float]:
    x, y = float(point[0]), float(point[1])
    return x - width / 2.0, y, x + width / 2.0, y + height


def top_corner_box(point: np.ndarray, width: float, height: float, direction: str) -> tuple[float, float, float, float]:
    x, y = float(point[0]), float(point[1])
    if direction == "left":
        return x - width, y, x, y + height
    if direction == "right":
        return x, y, x + width, y + height
    raise ValueError(f"Unknown direction: {direction}")


def patch_to_otsu_mask(patch: np.ndarray, low_clip_c: float | None = None) -> tuple[np.ndarray, float | None]:
    if patch.size == 0:
        return np.zeros_like(patch, dtype=bool), None

    work = patch.astype(np.float32, copy=True)
    finite = np.isfinite(work)
    if not finite.any():
        return np.zeros_like(patch, dtype=bool), None

    if low_clip_c is not None:
        work[finite] = np.maximum(work[finite], low_clip_c)

    values = work[finite]
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return finite.copy(), low

    scaled = np.zeros_like(work, dtype=np.uint8)
    scaled[finite] = np.clip((work[finite] - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    threshold_u8, mask_u8 = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_c = low + (float(threshold_u8) / 255.0) * (high - low)
    mask = (mask_u8 > 0) & finite
    return mask, threshold_c


def otsu_region_features(prefix: str, patch: np.ndarray, low_clip_c: float | None = None) -> dict[str, float]:
    features: dict[str, float] = {}
    patch_values = finite_values(patch.reshape(-1))
    if patch_values.size == 0:
        return features

    mask, threshold_c = patch_to_otsu_mask(patch, low_clip_c=low_clip_c)
    masked_values = finite_values(patch[mask])
    if masked_values.size == 0:
        masked_values = patch_values

    features.update(prefixed_stats(prefix, masked_values))
    patch_stats = describe_values(patch_values)
    for key in ("mean", "max", "p95", "top5_mean"):
        if key in patch_stats:
            features[f"{prefix}_patch_{key}"] = patch_stats[key]
    features[f"{prefix}_pixel_count"] = float(masked_values.size)
    features[f"{prefix}_patch_pixel_count"] = float(patch_values.size)
    features[f"{prefix}_otsu_fraction"] = float(masked_values.size / max(patch_values.size, 1))
    if threshold_c is not None:
        features[f"{prefix}_otsu_threshold_c"] = float(threshold_c)
    return features


def concat_valid_patches(patches: list[np.ndarray], axis: int = 0) -> np.ndarray | None:
    patches = [patch for patch in patches if patch.size]
    if not patches:
        return None
    try:
        return np.concatenate(patches, axis=axis)
    except ValueError:
        flat = [patch.reshape(-1) for patch in patches]
        return np.concatenate(flat, axis=0).reshape(-1, 1)


def add_box_features(
    features: dict[str, float],
    boxes: dict[str, tuple[np.ndarray, tuple[int, int, int, int], float | None]],
    prefix: str,
    names: list[str],
    axis: int = 0,
    low_clip_c: float | None = None,
) -> None:
    patches = [boxes[name][0] for name in names if name in boxes]
    merged = concat_valid_patches(patches, axis=axis)
    if merged is not None:
        features.update(otsu_region_features(prefix, merged, low_clip_c=low_clip_c))
    for name in names:
        if name in boxes:
            features.update(otsu_region_features(f"{prefix}_{name}", boxes[name][0], low_clip_c=low_clip_c))


def article_otsu_boxes(array: np.ndarray, bbox: np.ndarray, keypoints: np.ndarray, args) -> dict[str, tuple[np.ndarray, tuple[int, int, int, int], float | None]]:
    face_x0, face_y0, face_x1, face_y1 = [float(value) for value in bbox]
    face_w = max(1.0, face_x1 - face_x0)
    face_h = max(1.0, face_y1 - face_y0)

    eye_size = args.eye_box_size or max(8.0, min(face_w, face_h) * args.eye_box_scale)
    nostril_w = args.nostril_box_width or max(8.0, face_w * args.nostril_box_width_scale)
    nostril_h = args.nostril_box_height or max(8.0, face_h * args.nostril_box_height_scale)
    forehead_w = args.forehead_box_width or max(8.0, face_w * args.forehead_box_width_scale)
    forehead_h = args.forehead_box_height or max(8.0, face_h * args.forehead_box_height_scale)
    ear_w = args.ear_box_width or max(8.0, face_w * args.ear_box_width_scale)
    ear_h = args.ear_box_height or max(8.0, face_h * args.ear_box_height_scale)

    specs = {
        "left_eye": (*centered_box(keypoints[KP["left_eye"]], eye_size, eye_size), args.eye_low_clip_c),
        "right_eye": (*centered_box(keypoints[KP["right_eye"]], eye_size, eye_size), args.eye_low_clip_c),
        "left_nostril": (*top_center_box(keypoints[KP["left_nostril"]], nostril_w, nostril_h), None),
        "right_nostril": (*top_center_box(keypoints[KP["right_nostril"]], nostril_w, nostril_h), None),
        "forehead": (*top_center_box(keypoints[KP["poll"]], forehead_w, forehead_h), None),
        "left_ear": (*top_corner_box(keypoints[KP["left_ear_middle"]], ear_w, ear_h, "left"), None),
        "right_ear": (*top_corner_box(keypoints[KP["right_ear_base"]], ear_w, ear_h, "right"), None),
    }

    boxes = {}
    for name, (x0, y0, x1, y1, low_clip_c) in specs.items():
        patch, rect = clipped_rect(array, x0, y0, x1, y1)
        boxes[name] = (patch, rect, low_clip_c)
    return boxes


def article_otsu_features(array: np.ndarray, bbox: np.ndarray, keypoints: np.ndarray, args) -> tuple[dict[str, float], dict[str, tuple[int, int, int, int]]]:
    boxes = article_otsu_boxes(array, bbox, keypoints, args)
    features: dict[str, float] = {}

    add_box_features(features, boxes, "article_eyes", ["left_eye", "right_eye"], axis=0, low_clip_c=args.eye_low_clip_c)
    add_box_features(features, boxes, "article_nostrils", ["left_nostril", "right_nostril"], axis=0)
    add_box_features(features, boxes, "article_forehead", ["forehead"], axis=0)
    add_box_features(features, boxes, "article_ears", ["left_ear", "right_ear"], axis=0)
    add_box_features(
        features,
        boxes,
        "article_all",
        ["left_eye", "right_eye", "left_nostril", "right_nostril", "forehead", "left_ear", "right_ear"],
        axis=0,
    )

    rects = {name: rect for name, (_, rect, _) in boxes.items()}
    return features, rects


def add_article_rects(record: dict[str, object], rects: dict[str, tuple[int, int, int, int]]) -> None:
    for name, (x0, y0, x1, y1) in rects.items():
        record[f"article_{name}_x0"] = x0
        record[f"article_{name}_y0"] = y0
        record[f"article_{name}_x1"] = x1
        record[f"article_{name}_y1"] = y1


def extract_features(args):
    metadata_rows = load_temperature_metadata(args.metadata)
    raw_index = index_raw_zip(args.raw_zip)
    predictor = setup_predictor(args.config_file, args.weights, args.score_threshold, args.device)

    sequence_records = []
    frame_feature_records = []
    frame_detection_records = []
    frame_coordinate_records = []
    skipped_unreadable = 0
    skipped_no_detection = 0
    skipped_quality = 0

    if args.dates:
        date_filter = set(args.dates)
        metadata_rows = [row for row in metadata_rows if row["date"] in date_filter]
    if args.sequences:
        sequence_filter = set(args.sequences)
        metadata_rows = [row for row in metadata_rows if row["sequence_num"] in sequence_filter]
    if args.limit_sequences:
        metadata_rows = metadata_rows[: args.limit_sequences]

    detector_size = (args.detector_width, args.detector_height)
    with zipfile.ZipFile(args.raw_zip) as zf:
        for row in metadata_rows:
            key = (row["date"], row["sequence_num"])
            raw_frames = raw_index.get(key, [])
            if not raw_frames:
                continue

            sampled = choose_evenly_spaced(raw_frames, args.max_frames)
            per_frame_features = []
            for frame_id, zip_name in sampled:
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

                features, rects = article_otsu_features(array, bbox, keypoints, args)
                if not features:
                    continue
                if args.include_quality_features:
                    features.update({f"quality_{key}": value for key, value in quality.items()})

                per_frame_features.append(features)
                frame_feature_records.append(
                    {
                        "date": row["date"],
                        "sequence_num": row["sequence_num"],
                        "cow_tag": row["cow_tag"],
                        "temperature_f": row["temperature_f"],
                        "frame_id": frame_id,
                        **features,
                    }
                )
                detection_record = {
                    "date": row["date"],
                    "sequence_num": row["sequence_num"],
                    "cow_tag": row["cow_tag"],
                    "temperature_f": row["temperature_f"],
                    "frame_id": frame_id,
                    "feature_count": len(features),
                    **quality,
                }
                add_article_rects(detection_record, rects)
                frame_detection_records.append(detection_record)
                if args.write_roi_coordinates:
                    coordinate_record = roi_coordinate_record(row, frame_id, bbox, keypoints, quality)
                    add_article_rects(coordinate_record, rects)
                    frame_coordinate_records.append(coordinate_record)

            if not per_frame_features:
                continue

            record = {
                "date": row["date"],
                "sequence_num": row["sequence_num"],
                "cow_tag": row["cow_tag"],
                "temperature_f": row["temperature_f"],
                "raw_frame_count": len(raw_frames),
                "sampled_frame_count": len(sampled),
                "detected_frame_count": len(per_frame_features),
            }
            record.update(aggregate_feature_rows(per_frame_features))
            sequence_records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sequence_records:
        write_csv(args.output_dir / "features.csv", sequence_records)
    else:
        write_empty_csv(args.output_dir / "features.csv", FEATURE_ID_COLUMNS)
    if frame_feature_records:
        write_csv(args.output_dir / "frame_features.csv", frame_feature_records)
    if frame_detection_records:
        write_csv(args.output_dir / "frame_detections.csv", frame_detection_records)
    if frame_coordinate_records:
        write_csv(args.output_dir / "frame_roi_coordinates.csv", frame_coordinate_records)

    summary = {
        "sequence_count": len(sequence_records),
        "frame_detection_count": len(frame_detection_records),
        "skipped_unreadable": skipped_unreadable,
        "skipped_no_detection": skipped_no_detection,
        "skipped_quality": skipped_quality,
        "max_frames": args.max_frames,
        "eye_box_size": args.eye_box_size or "",
        "eye_box_scale": args.eye_box_scale,
        "nostril_box_width": args.nostril_box_width or "",
        "nostril_box_height": args.nostril_box_height or "",
        "forehead_box_width": args.forehead_box_width or "",
        "forehead_box_height": args.forehead_box_height or "",
        "ear_box_width": args.ear_box_width or "",
        "ear_box_height": args.ear_box_height or "",
    }
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract article-faithful local Otsu ROI temperature features from raw thermal TIFF frames."
    )
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--config-file", default="configs/CattleKeypoints/keypoints_rcnn_R_50_FPN_thermal.yaml", type=Path)
    parser.add_argument("--weights", default="data/train_outputs/thermal_clean_v1/model_final.pth", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/article_otsu_roi_v1", type=Path)
    parser.add_argument("--max-frames", default=80, type=int)
    parser.add_argument("--limit-sequences", type=int)
    parser.add_argument("--dates", nargs="+", help="Optional date folder filters, for example 02_13.")
    parser.add_argument("--sequences", nargs="+", help="Optional sequence number filters, for example 0027.")
    parser.add_argument("--detector-width", default=2560, type=int)
    parser.add_argument("--detector-height", default=1440, type=int)
    parser.add_argument("--score-threshold", default=0.0, type=float)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--min-keypoint-score", default=0.0, type=float)

    parser.add_argument("--eye-box-size", type=float, help="Fixed eye square size in raw thermal pixels.")
    parser.add_argument("--eye-box-scale", default=0.16, type=float)
    parser.add_argument("--eye-low-clip-c", default=16.0, type=float)
    parser.add_argument("--nostril-box-width", type=float)
    parser.add_argument("--nostril-box-height", type=float)
    parser.add_argument("--nostril-box-width-scale", default=0.28, type=float)
    parser.add_argument("--nostril-box-height-scale", default=0.22, type=float)
    parser.add_argument("--forehead-box-width", type=float)
    parser.add_argument("--forehead-box-height", type=float)
    parser.add_argument("--forehead-box-width-scale", default=0.30, type=float)
    parser.add_argument("--forehead-box-height-scale", default=0.24, type=float)
    parser.add_argument("--ear-box-width", type=float)
    parser.add_argument("--ear-box-height", type=float)
    parser.add_argument("--ear-box-width-scale", default=0.22, type=float)
    parser.add_argument("--ear-box-height-scale", default=0.30, type=float)

    parser.add_argument("--quality-filter", dest="quality_filter", action="store_true")
    parser.add_argument("--no-quality-filter", dest="quality_filter", action="store_false")
    parser.add_argument("--include-quality-features", dest="include_quality_features", action="store_true")
    parser.add_argument("--no-quality-features", dest="include_quality_features", action="store_false")
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
    parser.add_argument("--write-roi-coordinates", action="store_true")
    parser.set_defaults(quality_filter=True, include_quality_features=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = extract_features(args)
    print("Saved:", args.output_dir)
    print("Sequences with article Otsu ROI features:", summary["sequence_count"])
    print("Accepted sampled frames:", summary["frame_detection_count"])
    print("Skipped unreadable frames:", summary["skipped_unreadable"])
    print("Skipped no-detection frames:", summary["skipped_no_detection"])
    print("Skipped quality-filter frames:", summary["skipped_quality"])


if __name__ == "__main__":
    main()
