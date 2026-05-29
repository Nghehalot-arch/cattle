import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


KEYPOINT_NAMES = [
    "left_ear_base",
    "left_ear_middle",
    "left_ear_tip",
    "poll",
    "right_ear_base",
    "right_ear_middle",
    "right_ear_tip",
    "left_eye",
    "right_eye",
    "muzzle",
    "left_nostril",
    "right_nostril",
    "mouth",
]

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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def annotations_by_file(coco):
    image_id_to_file = {image["id"]: image["file_name"] for image in coco["images"]}
    result = {}
    for annotation in coco["annotations"]:
        file_name = image_id_to_file[annotation["image_id"]]
        result.setdefault(file_name, []).append(annotation)
    return result


def draw_annotation(image, annotation, color):
    bbox = annotation.get("bbox")
    if bbox:
        x, y, w, h = [int(round(v)) for v in bbox]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

    keypoints = annotation.get("keypoints", [])
    points = []
    for index in range(0, len(keypoints), 3):
        x, y, visible = keypoints[index : index + 3]
        if visible <= 0:
            points.append(None)
            continue
        point = (int(round(x)), int(round(y)))
        points.append(point)
        cv2.circle(image, point, 7, color, -1)
        label = str(len(points))
        cv2.putText(image, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for start, end in SKELETON:
        if start < len(points) and end < len(points) and points[start] and points[end]:
            cv2.line(image, points[start], points[end], color, 2)


def resize_to_height(image, height):
    scale = height / image.shape[0]
    width = int(round(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_header(image, text):
    header = np.full((58, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
    return np.vstack([header, image])


def main():
    parser = argparse.ArgumentParser(description="Visualize synchronized RGB/thermal paired split side by side.")
    parser.add_argument("--paired-root", default="datasets/keypoints/paired_rgb_thermal", type=Path)
    parser.add_argument("--split", default="demo", choices=["train", "val", "test", "demo"])
    parser.add_argument("--output-dir", default="data/outtest/paired_rgb_thermal_demo", type=Path)
    parser.add_argument("--limit", default=30, type=int)
    args = parser.parse_args()

    annotations_dir = args.paired_root / "annotations"
    rgb_coco = load_json(annotations_dir / f"rgb_{args.split}.json")
    thermal_coco = load_json(annotations_dir / f"thermal_{args.split}.json")
    rgb_annotations = annotations_by_file(rgb_coco)
    thermal_annotations = annotations_by_file(thermal_coco)

    pairs_path = annotations_dir / f"pairs_{args.split}.csv"
    with open(pairs_path, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs[: args.limit]:
        rgb_path = args.paired_root / "rgb" / f"{args.split}_imgs" / pair["rgb_file"]
        thermal_path = args.paired_root / "thermal" / f"{args.split}_imgs" / pair["thermal_file"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        thermal = cv2.imread(str(thermal_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        if thermal is None:
            raise FileNotFoundError(thermal_path)

        for annotation in rgb_annotations.get(pair["rgb_file"], []):
            draw_annotation(rgb, annotation, (0, 220, 0))
        for annotation in thermal_annotations.get(pair["thermal_file"], []):
            draw_annotation(thermal, annotation, (0, 0, 255))

        rgb = resize_to_height(rgb, 720)
        thermal = resize_to_height(thermal, 720)
        rgb = add_header(rgb, f"RGB {pair['folder']}/{pair['frame_id']}")
        thermal = add_header(thermal, f"Thermal {pair['folder']}/{pair['frame_id']}")

        combined = np.hstack([rgb, thermal])
        out_name = f"{int(pair['pair_id']):04d}_{pair['folder']}_{pair['frame_id']}.jpg"
        cv2.imwrite(str(args.output_dir / out_name), combined)

    print("Saved visualizations:", args.output_dir)
    print("Images written:", min(args.limit, len(pairs)))


if __name__ == "__main__":
    main()
