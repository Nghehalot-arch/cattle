import argparse
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (
    aggregate_feature_rows,
    choose_evenly_spaced,
    circle_values,
    index_raw_zip,
    load_temperature_metadata,
    normalize_thermal_for_detector,
    prefixed_stats,
    read_tiff_array,
    rect_values,
    write_csv,
)


KEYPOINT_INDEX = {
    "left_eye": 7,
    "right_eye": 8,
    "muzzle": 9,
    "left_nostril": 10,
    "right_nostril": 11,
    "mouth": 12,
}

QUALITY_KEYPOINTS = [
    "left_eye",
    "right_eye",
    "muzzle",
    "left_nostril",
    "right_nostril",
    "mouth",
]

ALL_KEYPOINT_INDEX = {f"kp{index + 1:02d}": index for index in range(13)}


def setup_predictor(config_file, weights, threshold, device=None):
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        import register_cattle_datasets  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Detectron2 is required to run keypoint detection. "
            "Use an environment with Detectron2 installed, or predict from an already extracted features.csv."
        ) from exc

    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    if device:
        cfg.MODEL.DEVICE = device
    cfg.freeze()
    return DefaultPredictor(cfg)


def best_instance(outputs):
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        return None
    scores = instances.scores.numpy()
    index = int(np.argmax(scores))
    return {
        "score": float(scores[index]),
        "bbox": instances.pred_boxes.tensor.numpy()[index],
        "keypoints": instances.pred_keypoints.numpy()[index],
    }


def scale_detection(detection, raw_shape, detector_size):
    raw_h, raw_w = raw_shape
    det_w, det_h = detector_size
    sx = raw_w / det_w
    sy = raw_h / det_h

    bbox = detection["bbox"].astype(np.float32).copy()
    bbox[[0, 2]] *= sx
    bbox[[1, 3]] *= sy

    keypoints = detection["keypoints"].astype(np.float32).copy()
    keypoints[:, 0] *= sx
    keypoints[:, 1] *= sy
    return bbox, keypoints


def add_keypoint_circle(features, array, keypoints, name, radius, min_keypoint_score):
    index = KEYPOINT_INDEX[name]
    x, y, score = keypoints[index]
    if score < min_keypoint_score:
        return
    features.update(prefixed_stats(name, circle_values(array, (x, y), radius)))


def add_indexed_keypoint_circle(features, array, keypoints, name, index, radius, min_keypoint_score):
    x, y, score = keypoints[index]
    if score < min_keypoint_score:
        return
    features.update(prefixed_stats(name, circle_values(array, (x, y), radius)))


def rect_ring_values(array, inner_rect, margin_x, margin_y):
    height, width = array.shape
    x0, y0, x1, y1 = inner_rect
    outer = (
        x0 - margin_x,
        y0 - margin_y,
        x1 + margin_x,
        y1 + margin_y,
    )
    ix0 = max(0, min(width - 1, int(round(x0))))
    iy0 = max(0, min(height - 1, int(round(y0))))
    ix1 = max(ix0 + 1, min(width, int(round(x1))))
    iy1 = max(iy0 + 1, min(height, int(round(y1))))
    ox0 = max(0, min(width - 1, int(round(outer[0]))))
    oy0 = max(0, min(height - 1, int(round(outer[1]))))
    ox1 = max(ox0 + 1, min(width, int(round(outer[2]))))
    oy1 = max(oy0 + 1, min(height, int(round(outer[3]))))
    patch = array[oy0:oy1, ox0:ox1]
    if patch.size == 0:
        return np.asarray([], dtype=np.float32)
    yy, xx = np.ogrid[oy0:oy1, ox0:ox1]
    inner_mask = (xx >= ix0) & (xx < ix1) & (yy >= iy0) & (yy < iy1)
    values = patch[~inner_mask]
    return values.reshape(-1)


def roi_features(
    array,
    bbox,
    keypoints,
    min_keypoint_score,
    include_all_keypoints=False,
    include_surrounding_ring=False,
):
    features = {}
    x0, y0, x1, y1 = bbox
    face_w = max(1.0, x1 - x0)
    face_h = max(1.0, y1 - y0)
    radius = max(2.0, min(face_w, face_h) * 0.045)

    features.update(prefixed_stats("face_bbox", rect_values(array, bbox)))
    if include_surrounding_ring:
        features.update(
            prefixed_stats(
                "face_surround",
                rect_ring_values(array, bbox, face_w * 0.30, face_h * 0.30),
            )
        )

    for name in KEYPOINT_INDEX:
        add_keypoint_circle(features, array, keypoints, name, radius, min_keypoint_score)
    if include_all_keypoints:
        for name, index in ALL_KEYPOINT_INDEX.items():
            add_indexed_keypoint_circle(features, array, keypoints, name, index, radius, min_keypoint_score)

    nostril_points = []
    for name in ("left_nostril", "right_nostril"):
        x, y, score = keypoints[KEYPOINT_INDEX[name]]
        if score >= min_keypoint_score:
            nostril_points.append((x, y))
    if nostril_points:
        xs = [p[0] for p in nostril_points]
        ys = [p[1] for p in nostril_points]
        margin_x = face_w * 0.08
        margin_y = face_h * 0.08
        nostril_rect = (min(xs) - margin_x, min(ys) - margin_y, max(xs) + margin_x, max(ys) + margin_y)
        features.update(prefixed_stats("nostrils_box", rect_values(array, nostril_rect)))

    lower_points = []
    for name in ("muzzle", "left_nostril", "right_nostril", "mouth"):
        x, y, score = keypoints[KEYPOINT_INDEX[name]]
        if score >= min_keypoint_score:
            lower_points.append((x, y))
    if lower_points:
        xs = [p[0] for p in lower_points]
        ys = [p[1] for p in lower_points]
        lower_rect = (
            min(xs) - face_w * 0.12,
            min(ys) - face_h * 0.12,
            max(xs) + face_w * 0.12,
            max(ys) + face_h * 0.12,
        )
        features.update(prefixed_stats("lower_face", rect_values(array, lower_rect)))

    return features


def distance(left, right):
    return float(np.linalg.norm(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)))


def detection_quality(array_shape, bbox, keypoints, detection_score):
    raw_h, raw_w = array_shape
    x0, y0, x1, y1 = bbox
    face_w = max(1.0, float(x1 - x0))
    face_h = max(1.0, float(y1 - y0))
    face_area_frac = float((face_w * face_h) / max(1.0, raw_w * raw_h))
    face_aspect = float(face_w / face_h)
    center_offset_x = float(abs(((x0 + x1) / 2.0) - (raw_w / 2.0)) / raw_w)
    center_offset_y = float(abs(((y0 + y1) / 2.0) - (raw_h / 2.0)) / raw_h)

    kp = {}
    kp_scores = []
    for name in QUALITY_KEYPOINTS:
        x, y, score = keypoints[KEYPOINT_INDEX[name]]
        kp[name] = (float(x), float(y), float(score))
        kp_scores.append(float(score))

    left_eye = kp["left_eye"]
    right_eye = kp["right_eye"]
    left_nostril = kp["left_nostril"]
    right_nostril = kp["right_nostril"]
    muzzle = kp["muzzle"]
    mouth = kp["mouth"]

    eye_y_diff_norm = abs(left_eye[1] - right_eye[1]) / face_h
    nostril_y_diff_norm = abs(left_nostril[1] - right_nostril[1]) / face_h
    eye_width_norm = distance(left_eye[:2], right_eye[:2]) / face_w
    nostril_width_norm = distance(left_nostril[:2], right_nostril[:2]) / face_w
    nostril_center_x = (left_nostril[0] + right_nostril[0]) / 2.0
    nostril_center_y = (left_nostril[1] + right_nostril[1]) / 2.0
    muzzle_center_offset = abs(muzzle[0] - nostril_center_x) / face_w

    muzzle_to_left = distance(muzzle[:2], left_nostril[:2])
    muzzle_to_right = distance(muzzle[:2], right_nostril[:2])
    muzzle_symmetry = abs(muzzle_to_left - muzzle_to_right) / max(muzzle_to_left + muzzle_to_right, 1.0)

    eyes_above_nostrils = float(((left_eye[1] + right_eye[1]) / 2.0) < nostril_center_y)
    nostrils_above_mouth = float(nostril_center_y < mouth[1])
    lower_face_order_ok = float(eyes_above_nostrils and nostrils_above_mouth)

    penalties = [
        min(1.0, eye_y_diff_norm / 0.18),
        min(1.0, nostril_y_diff_norm / 0.18),
        min(1.0, muzzle_center_offset / 0.28),
        min(1.0, muzzle_symmetry / 0.45),
        0.0 if lower_face_order_ok else 1.0,
    ]
    frontal_score = float(max(0.0, 1.0 - np.mean(penalties)))

    return {
        "detection_score": float(detection_score),
        "face_area_frac": face_area_frac,
        "face_aspect": face_aspect,
        "center_offset_x": center_offset_x,
        "center_offset_y": center_offset_y,
        "required_kp_score_min": float(np.min(kp_scores)),
        "required_kp_score_mean": float(np.mean(kp_scores)),
        "eye_y_diff_norm": float(eye_y_diff_norm),
        "nostril_y_diff_norm": float(nostril_y_diff_norm),
        "eye_width_norm": float(eye_width_norm),
        "nostril_width_norm": float(nostril_width_norm),
        "muzzle_center_offset": float(muzzle_center_offset),
        "muzzle_symmetry": float(muzzle_symmetry),
        "lower_face_order_ok": lower_face_order_ok,
        "frontal_score": frontal_score,
    }


def passes_quality_filter(quality, args):
    if not args.quality_filter:
        return True
    checks = [
        quality["detection_score"] >= args.min_detection_score,
        quality["required_kp_score_min"] >= args.min_required_keypoint_score,
        quality["face_area_frac"] >= args.min_face_area_frac,
        quality["face_area_frac"] <= args.max_face_area_frac,
        quality["face_aspect"] >= args.min_face_aspect,
        quality["face_aspect"] <= args.max_face_aspect,
        quality["eye_y_diff_norm"] <= args.max_eye_y_diff,
        quality["nostril_y_diff_norm"] <= args.max_nostril_y_diff,
        quality["muzzle_center_offset"] <= args.max_muzzle_center_offset,
        quality["muzzle_symmetry"] <= args.max_muzzle_symmetry,
        quality["frontal_score"] >= args.min_frontal_score,
    ]
    if args.require_lower_face_order:
        checks.append(bool(quality["lower_face_order_ok"]))
    return all(checks)


def roi_coordinate_record(row, frame_id, bbox, keypoints, quality):
    record = {
        "date": row["date"],
        "sequence_num": row["sequence_num"],
        "cow_tag": row.get("cow_tag", ""),
        "temperature_f": row.get("temperature_f", ""),
        "frame_id": frame_id,
        "bbox_x0": float(bbox[0]),
        "bbox_y0": float(bbox[1]),
        "bbox_x1": float(bbox[2]),
        "bbox_y1": float(bbox[3]),
    }
    for name, value in quality.items():
        record[f"quality_{name}"] = value
    for name, index in KEYPOINT_INDEX.items():
        x, y, score = keypoints[index]
        record[f"kp_{name}_x"] = float(x)
        record[f"kp_{name}_y"] = float(y)
        record[f"kp_{name}_score"] = float(score)
    for name, index in ALL_KEYPOINT_INDEX.items():
        x, y, score = keypoints[index]
        record[f"{name}_x"] = float(x)
        record[f"{name}_y"] = float(y)
        record[f"{name}_score"] = float(score)
    return record


def extract_features(args):
    metadata_rows = load_temperature_metadata(args.metadata)
    raw_index = index_raw_zip(args.raw_zip)
    predictor = setup_predictor(args.config_file, args.weights, args.score_threshold)

    sequence_records = []
    frame_feature_records = []
    frame_records = []
    frame_coordinate_records = []
    skipped_unreadable = 0
    skipped_no_detection = 0
    skipped_quality = 0

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

                features = roi_features(
                    array,
                    bbox,
                    keypoints,
                    args.min_keypoint_score,
                    include_all_keypoints=args.include_all_keypoints,
                    include_surrounding_ring=args.include_surrounding_ring,
                )
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
                frame_records.append(
                    {
                        "date": row["date"],
                        "sequence_num": row["sequence_num"],
                        "cow_tag": row["cow_tag"],
                        "temperature_f": row["temperature_f"],
                        "frame_id": frame_id,
                        "feature_count": len(features),
                        **quality,
                    }
                )
                if args.write_roi_coordinates:
                    frame_coordinate_records.append(roi_coordinate_record(row, frame_id, bbox, keypoints, quality))

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
    write_csv(args.output_dir / "features.csv", sequence_records)
    if frame_feature_records:
        write_csv(args.output_dir / "frame_features.csv", frame_feature_records)
    if frame_records:
        write_csv(args.output_dir / "frame_detections.csv", frame_records)
    if frame_coordinate_records:
        write_csv(args.output_dir / "frame_roi_coordinates.csv", frame_coordinate_records)

    summary = {
        "sequence_count": len(sequence_records),
        "frame_detection_count": len(frame_records),
        "skipped_unreadable": skipped_unreadable,
        "skipped_no_detection": skipped_no_detection,
        "skipped_quality": skipped_quality,
    }
    with open(args.output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract CattleFever-style ROI temperature features from raw TIFF frames using detected thermal keypoints."
    )
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--config-file", default="configs/CattleKeypoints/keypoints_rcnn_R_50_FPN_thermal.yaml", type=Path)
    parser.add_argument("--weights", default="data/train_outputs/thermal_clean_v1/model_final.pth", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/detected_roi_v1", type=Path)
    parser.add_argument("--max-frames", default=30, type=int)
    parser.add_argument("--detector-width", default=2560, type=int)
    parser.add_argument("--detector-height", default=1440, type=int)
    parser.add_argument("--score-threshold", default=0.0, type=float)
    parser.add_argument("--min-keypoint-score", default=0.0, type=float)
    parser.add_argument("--quality-filter", action="store_true")
    parser.add_argument("--include-quality-features", action="store_true")
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
    parser.add_argument("--include-all-keypoints", action="store_true")
    parser.add_argument("--include-surrounding-ring", action="store_true")
    args = parser.parse_args()

    summary = extract_features(args)
    print("Saved:", args.output_dir)
    print("Sequences with ROI features:", summary["sequence_count"])
    print("Detected sampled frames:", summary["frame_detection_count"])
    print("Skipped unreadable frames:", summary["skipped_unreadable"])
    print("Skipped no-detection frames:", summary["skipped_no_detection"])
    print("Skipped quality-filter frames:", summary["skipped_quality"])


if __name__ == "__main__":
    main()
