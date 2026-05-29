import argparse
import csv
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import register_cattle_datasets  # noqa: F401
from common import (
    choose_evenly_spaced,
    index_raw_zip,
    load_temperature_metadata,
    load_temperature_metadata,
    normalize_thermal_for_detector,
    read_tiff_array,
)


KEYPOINT_INDEX = {
    "left_eye": 7,
    "right_eye": 8,
    "muzzle": 9,
    "left_nostril": 10,
    "right_nostril": 11,
    "mouth": 12,
}

SKELETON = [
    (0, 1),
    (1, 2),
    (1, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (2, 7),
    (6, 8),
    (7, 9),
    (8, 9),
    (9, 10),
    (9, 11),
    (10, 12),
    (11, 12),
]

ROI_COLORS = {
    "face_bbox": (255, 80, 80),
    "left_eye": (80, 255, 255),
    "right_eye": (80, 255, 255),
    "muzzle": (80, 220, 80),
    "left_nostril": (0, 165, 255),
    "right_nostril": (0, 165, 255),
    "mouth": (255, 0, 255),
    "nostrils_box": (0, 255, 255),
    "lower_face": (255, 255, 0),
}


def setup_predictor(config_file, weights, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
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


def thermal_display(array):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        scaled = np.zeros_like(array, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1, 99])
        scaled = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)


def raw_to_display_point(point, scale):
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def draw_rect(image, rect, scale, color, label):
    x0, y0, x1, y1 = rect
    p0 = raw_to_display_point((x0, y0), scale)
    p1 = raw_to_display_point((x1, y1), scale)
    cv2.rectangle(image, p0, p1, color, 2)
    cv2.putText(image, label, (p0[0], max(18, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_circle(image, center, radius, scale, color, label):
    point = raw_to_display_point(center, scale)
    cv2.circle(image, point, max(2, int(round(radius * scale))), color, 2)
    cv2.putText(image, label, (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def roi_geometry(bbox, keypoints, min_keypoint_score):
    x0, y0, x1, y1 = bbox
    face_w = max(1.0, x1 - x0)
    face_h = max(1.0, y1 - y0)
    radius = max(2.0, min(face_w, face_h) * 0.045)

    rois = [{"type": "rect", "name": "face_bbox", "rect": (x0, y0, x1, y1)}]

    for name, index in KEYPOINT_INDEX.items():
        x, y, score = keypoints[index]
        if score >= min_keypoint_score:
            rois.append({"type": "circle", "name": name, "center": (x, y), "radius": radius})

    nostril_points = []
    for name in ("left_nostril", "right_nostril"):
        x, y, score = keypoints[KEYPOINT_INDEX[name]]
        if score >= min_keypoint_score:
            nostril_points.append((x, y))
    if nostril_points:
        xs = [p[0] for p in nostril_points]
        ys = [p[1] for p in nostril_points]
        rois.append(
            {
                "type": "rect",
                "name": "nostrils_box",
                "rect": (
                    min(xs) - face_w * 0.08,
                    min(ys) - face_h * 0.08,
                    max(xs) + face_w * 0.08,
                    max(ys) + face_h * 0.08,
                ),
            }
        )

    lower_points = []
    for name in ("muzzle", "left_nostril", "right_nostril", "mouth"):
        x, y, score = keypoints[KEYPOINT_INDEX[name]]
        if score >= min_keypoint_score:
            lower_points.append((x, y))
    if lower_points:
        xs = [p[0] for p in lower_points]
        ys = [p[1] for p in lower_points]
        rois.append(
            {
                "type": "rect",
                "name": "lower_face",
                "rect": (
                    min(xs) - face_w * 0.12,
                    min(ys) - face_h * 0.12,
                    max(xs) + face_w * 0.12,
                    max(ys) + face_h * 0.12,
                ),
            }
        )

    return rois


def draw_keypoints(image, keypoints, scale, min_keypoint_score):
    points = []
    for index, (x, y, score) in enumerate(keypoints):
        if score < min_keypoint_score:
            points.append(None)
            continue
        point = raw_to_display_point((x, y), scale)
        points.append(point)
        cv2.circle(image, point, 4, (255, 255, 255), -1)
        cv2.putText(image, str(index + 1), (point[0] + 4, point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    for start, end in SKELETON:
        if start < len(points) and end < len(points) and points[start] and points[end]:
            cv2.line(image, points[start], points[end], (180, 180, 180), 1)


def add_header(image, text):
    header = np.full((54, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, text, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2)
    return np.vstack([header, image])


def load_pair_lookup(paired_root):
    pair_lookup = {}
    annotations = paired_root / "annotations"
    for split in ("train", "val", "test", "demo"):
        path = annotations / f"pairs_{split}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pair_lookup[(row["folder"], row["frame_id"])] = {
                    "split": split,
                    "rgb_file": row["rgb_file"],
                    "thermal_file": row["thermal_file"],
                }
    return pair_lookup


def main():
    parser = argparse.ArgumentParser(description="Visualize detected raw thermal temperature ROIs.")
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--paired-root", default="datasets/keypoints/paired_rgb_thermal", type=Path)
    parser.add_argument("--config-file", default="configs/CattleKeypoints/keypoints_rcnn_R_50_FPN_thermal.yaml", type=Path)
    parser.add_argument("--weights", default="data/train_outputs/thermal_clean_v1/model_final.pth", type=Path)
    parser.add_argument("--output-dir", default="data/outtest/temperature_roi_check", type=Path)
    parser.add_argument("--max-sequences", default=8, type=int)
    parser.add_argument("--frames-per-sequence", default=2, type=int)
    parser.add_argument("--score-threshold", default=0.0, type=float)
    parser.add_argument("--min-keypoint-score", default=0.0, type=float)
    parser.add_argument("--display-scale", default=3.0, type=float)
    args = parser.parse_args()

    metadata_rows = load_temperature_metadata(args.metadata)
    raw_index = index_raw_zip(args.raw_zip)
    pair_lookup = load_pair_lookup(args.paired_root)
    predictor = setup_predictor(args.config_file, args.weights, args.score_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    detector_size = (2560, 1440)
    with zipfile.ZipFile(args.raw_zip) as zf:
        for row in metadata_rows[: args.max_sequences]:
            key = (row["date"], row["sequence_num"])
            raw_frames = raw_index.get(key, [])
            if not raw_frames:
                continue
            for frame_id, zip_name in choose_evenly_spaced(raw_frames, args.frames_per_sequence):
                array = read_tiff_array(zf, zip_name)
                if array is None:
                    continue

                detector_image = normalize_thermal_for_detector(array, width=detector_size[0], height=detector_size[1])
                detection = best_instance(predictor(detector_image))
                if detection is None:
                    continue
                bbox, keypoints = scale_detection(detection, array.shape, detector_size)

                display = thermal_display(array)
                display = cv2.resize(
                    display,
                    (int(round(display.shape[1] * args.display_scale)), int(round(display.shape[0] * args.display_scale))),
                    interpolation=cv2.INTER_NEAREST,
                )
                draw_keypoints(display, keypoints, args.display_scale, args.min_keypoint_score)

                for roi in roi_geometry(bbox, keypoints, args.min_keypoint_score):
                    color = ROI_COLORS[roi["name"]]
                    if roi["type"] == "rect":
                        draw_rect(display, roi["rect"], args.display_scale, color, roi["name"])
                    else:
                        draw_circle(display, roi["center"], roi["radius"], args.display_scale, color, roi["name"])

                pair = pair_lookup.get((str(int(row["sequence_num"])), f"{frame_id:05d}"))
                paired_text = "paired=no"
                if pair:
                    paired_text = f"paired=yes split={pair['split']}"
                header = (
                    f"{row['date']}/{row['sequence_num']} frame={frame_id} "
                    f"cow={row['cow_tag']} temp_f={row['temperature_f']} {paired_text}"
                )
                display = add_header(display, header)

                out_name = f"{row['date']}_{row['sequence_num']}_{frame_id:05d}.jpg"
                cv2.imwrite(str(args.output_dir / out_name), display)
                written += 1

    print("Saved ROI visualizations:", args.output_dir)
    print("Images written:", written)


if __name__ == "__main__":
    main()
