from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


RAW_NAME_RE = re.compile(r"thermal_raw/([^/]+)/(\d+)_Video_Frame_(\d+)\.tiff$")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def raw_counts(raw_zip: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    with zipfile.ZipFile(raw_zip) as zf:
        for name in zf.namelist():
            match = RAW_NAME_RE.match(name)
            if match:
                date, sequence_num, _ = match.groups()
                counts[(date, sequence_num)] += 1
    return counts


def annotation_folder_counts(path: Path) -> Counter[str]:
    if not path.exists():
        return Counter()
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    return Counter(str(image.get("folder")) for image in coco.get("images", []))


def pair_counts(paired_root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for split in ("train", "val", "test", "demo"):
        path = paired_root / "annotations" / f"pairs_{split}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[str(row["folder"])] += 1
    return counts


def load_mapping(path: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[str(row["folder"])].append(row)
    return rows


def mapping_for_label(mapping_rows: list[dict[str, str]], date: str, sequence_num: str) -> dict[str, str] | None:
    for row in mapping_rows:
        if row["date"] == date and row["sequence_num"] == sequence_num:
            return row
    return None


def best_temperature_mapping(mapping_rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [
        row
        for row in mapping_rows
        if row.get("temperature_f") and row.get("mean_score")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["mean_score"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit every rectal-temperature label against RGB, thermal JPG, paired, and raw TIFF availability."
    )
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--rgb-json", default="data/annotations/rgb_keypoints.json", type=Path)
    parser.add_argument("--thermal-json", default="data/annotations/thermal_keypoints.json", type=Path)
    parser.add_argument("--paired-root", default="datasets/keypoints/paired_rgb_thermal", type=Path)
    parser.add_argument("--processed-mapping", default="data/temperature_outputs/processed_raw_mapping.csv", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/rectal_label_modality_audit_v1", type=Path)
    args = parser.parse_args()

    raw = raw_counts(args.raw_zip)
    rgb = annotation_folder_counts(args.rgb_json)
    thermal = annotation_folder_counts(args.thermal_json)
    paired = pair_counts(args.paired_root)
    mappings = load_mapping(args.processed_mapping)

    all_rows = []
    labeled_rows = []
    with args.metadata.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            folder = str(int(row["sequence_num"]))
            key = (row["date"], row["sequence_num"])
            mapping_rows = mappings.get(folder, [])
            direct_mapping = mapping_for_label(mapping_rows, row["date"], row["sequence_num"])
            best_mapping = best_temperature_mapping(mapping_rows)
            has_temp = bool(row.get("temperature_f"))
            raw_frame_count = raw.get(key, 0)
            rgb_count = rgb.get(folder, 0)
            thermal_count = thermal.get(folder, 0)
            paired_count = paired.get(folder, 0)
            manifest_row = {
                "date": row["date"],
                "sequence_num": row["sequence_num"],
                "folder": folder,
                "cow_tag": row["cow_tag"],
                "temperature_f": row.get("temperature_f", ""),
                "has_rectal_label": "yes" if has_temp else "no",
                "rectal_temp_role": "ground_truth_label" if has_temp else "",
                "has_raw_tiff": "yes" if raw_frame_count else "no",
                "raw_tiff_frames": raw_frame_count,
                "has_rgb_keypoint_image": "yes" if rgb_count else "no",
                "rgb_annotated_images_by_folder": rgb_count,
                "has_thermal_keypoint_image": "yes" if thermal_count else "no",
                "thermal_annotated_images_by_folder": thermal_count,
                "has_paired_rgb_thermal": "yes" if paired_count else "no",
                "paired_rgb_thermal_images_by_folder": paired_count,
                "processed_mapping_candidates": len(mapping_rows),
                "direct_processed_mapping_score": direct_mapping.get("mean_score", "") if direct_mapping else "",
                "best_processed_mapping_date": best_mapping.get("date", "") if best_mapping else "",
                "best_processed_mapping_sequence": best_mapping.get("sequence_num", "") if best_mapping else "",
                "best_processed_mapping_temp_f": best_mapping.get("temperature_f", "") if best_mapping else "",
                "best_processed_mapping_score": best_mapping.get("mean_score", "") if best_mapping else "",
                "safe_for_temperature_training": "yes" if has_temp and raw_frame_count else "no",
                "safe_for_detector_training": "yes" if rgb_count or thermal_count else "no",
                "notes": "",
            }
            if has_temp and not raw_frame_count:
                manifest_row["notes"] = "rectal label exists but no raw TIFF for calibrated thermal temperature model"
            elif has_temp and (rgb_count or thermal_count) and not direct_mapping:
                manifest_row["notes"] = "image folder has annotations but date-specific label mapping is ambiguous"
            all_rows.append(manifest_row)
            if has_temp:
                labeled_rows.append(manifest_row)

    write_csv(args.output_dir / "all_metadata_modalities.csv", all_rows)
    write_csv(args.output_dir / "rectal_labeled_modalities.csv", labeled_rows)
    usable_temperature = [row for row in labeled_rows if row["safe_for_temperature_training"] == "yes"]
    missing_temperature = [row for row in labeled_rows if row["safe_for_temperature_training"] == "no"]
    write_csv(args.output_dir / "usable_temperature_labels.csv", usable_temperature)
    write_csv(args.output_dir / "unusable_rectal_labels.csv", missing_temperature)

    summary = {
        "metadata_rows": len(all_rows),
        "rectal_labeled_rows": len(labeled_rows),
        "usable_temperature_labels_raw_tiff": len(usable_temperature),
        "unusable_rectal_labels_missing_raw_tiff": len(missing_temperature),
        "rgb_annotated_images": sum(rgb.values()),
        "thermal_annotated_images": sum(thermal.values()),
        "paired_rgb_thermal_images": sum(paired.values()),
        "raw_tiff_frames": sum(raw.values()),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", args.output_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
