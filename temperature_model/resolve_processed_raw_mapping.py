import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


RAW_NAME_RE = re.compile(r"thermal_raw/([^/]+)/(\d+)_Video_Frame_(\d+)\.tiff$")


def read_tiff(zf, name):
    data = zf.read(name)
    try:
        array = np.asarray(Image.open(BytesIO(data)), dtype=np.float32)
    except Exception:
        encoded = np.frombuffer(data, dtype=np.uint8)
        array = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if array is None:
            return None
        array = array.astype(np.float32)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def normalize(array):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    scaled = np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def corr_score(raw_array, jpg_path):
    jpg = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
    if jpg is None:
        return None
    raw = normalize(raw_array)
    raw = cv2.resize(raw, (jpg.shape[1], jpg.shape[0]), interpolation=cv2.INTER_LINEAR)

    raw_edge = cv2.Canny(raw, 40, 120).astype(np.float32).reshape(-1)
    jpg_edge = cv2.Canny(jpg, 40, 120).astype(np.float32).reshape(-1)
    raw_flat = raw.astype(np.float32).reshape(-1)
    jpg_flat = jpg.astype(np.float32).reshape(-1)

    scores = []
    for left, right in [(raw_flat, jpg_flat), (raw_edge, jpg_edge)]:
        if left.std() > 0 and right.std() > 0:
            scores.append(float(np.corrcoef(left, right)[0, 1]))
            scores.append(float(np.corrcoef(255 - left, right)[0, 1]))
    if not scores:
        return None
    return max(scores)


def raw_index(raw_zip):
    index = defaultdict(dict)
    with zipfile.ZipFile(raw_zip) as zf:
        for name in zf.namelist():
            match = RAW_NAME_RE.match(name)
            if not match:
                continue
            date, sequence_num, frame_id = match.groups()
            index[(date, sequence_num)][int(frame_id)] = name
    return index


def load_metadata(metadata):
    with open(metadata, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_thermal_frames(annotation_json):
    with open(annotation_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    frames = defaultdict(list)
    for image in coco["images"]:
        frames[str(image["folder"])].append(int(str(image["frame_id"])))
    return {folder: sorted(set(values)) for folder, values in frames.items()}


def evenly_sample(values, limit):
    if len(values) <= limit:
        return values
    positions = np.linspace(0, len(values) - 1, limit).round().astype(int)
    return [values[int(pos)] for pos in positions]


def main():
    parser = argparse.ArgumentParser(
        description="Estimate which raw thermal date/sequence maps to each processed thermal folder."
    )
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--thermal-json", default="data/annotations/thermal_keypoints.json", type=Path)
    parser.add_argument("--thermal-root", default="datasets/keypoints/thermal", type=Path)
    parser.add_argument("--output", default="data/temperature_outputs/processed_raw_mapping.csv", type=Path)
    parser.add_argument("--samples", default=20, type=int)
    args = parser.parse_args()

    rows = load_metadata(args.metadata)
    rows_by_sequence = defaultdict(list)
    for row in rows:
        rows_by_sequence[str(int(row["sequence_num"]))].append(row)

    raw = raw_index(args.raw_zip)
    processed_frames = load_thermal_frames(args.thermal_json)
    output_rows = []

    with zipfile.ZipFile(args.raw_zip) as zf:
        for folder, frames in sorted(processed_frames.items(), key=lambda item: int(item[0])):
            candidates = rows_by_sequence.get(folder, [])
            for row in candidates:
                key = (row["date"], row["sequence_num"])
                raw_frames = raw.get(key, {})
                shared = sorted(set(frames) & set(raw_frames))
                scores = []
                for frame_id in evenly_sample(shared, args.samples):
                    raw_array = read_tiff(zf, raw_frames[frame_id])
                    if raw_array is None:
                        continue
                    jpg_path = args.thermal_root / folder / f"{frame_id:05d}.jpg"
                    score = corr_score(raw_array, jpg_path)
                    if score is not None and np.isfinite(score):
                        scores.append(score)

                output_rows.append(
                    {
                        "folder": folder,
                        "date": row["date"],
                        "sequence_num": row["sequence_num"],
                        "cow_tag": row["cow_tag"],
                        "temperature_f": row["temperature_f"],
                        "shared_frames": len(shared),
                        "scored_frames": len(scores),
                        "mean_score": float(np.mean(scores)) if scores else "",
                        "median_score": float(np.median(scores)) if scores else "",
                        "max_score": float(np.max(scores)) if scores else "",
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print("Saved:", args.output)
    for row in output_rows:
        print(
            "folder={folder} candidate={date}/{sequence_num} temp={temperature_f} "
            "shared={shared_frames} scored={scored_frames} mean={mean_score}".format(**row)
        )


if __name__ == "__main__":
    main()
