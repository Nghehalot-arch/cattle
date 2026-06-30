from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")
PAIR_COLUMNS = [
    "pair_id",
    "split",
    "folder",
    "frame_id",
    "rgb_file",
    "thermal_file",
    "rgb_original_file",
    "thermal_original_file",
    "rgb_original_id",
    "thermal_original_id",
    "label_available",
    "raw_date",
    "raw_sequence_num",
    "cow_tag",
    "temperature_f",
    "mapping_mean_score",
]
LABEL_COLUMNS = [
    "folder",
    "raw_date",
    "raw_sequence_num",
    "cow_tag",
    "temperature_f",
    "mapping_mean_score",
]


def sort_key(value):
    try:
        return int(value)
    except ValueError:
        return value


def load_coco(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pair_key(image):
    return str(image.get("folder")), str(image.get("frame_id")).zfill(5)


def safe_name(prefix, folder, frame_id):
    return f"{prefix}_{folder}_{frame_id}.jpg"


def annotation_index(coco):
    by_image_id = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image_id[annotation["image_id"]].append(annotation)
    return by_image_id


def read_temperature_mapping(path, min_score):
    if not path.exists():
        return {}

    best_by_folder = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("temperature_f") or not row.get("mean_score"):
                continue
            score = float(row["mean_score"])
            if score < min_score:
                continue
            folder = str(row["folder"])
            previous = best_by_folder.get(folder)
            if previous is None or score > float(previous["mean_score"]):
                best_by_folder[folder] = row
    return best_by_folder


def label_for_folder(folder, labels_by_folder):
    label = labels_by_folder.get(folder)
    if not label:
        return {
            "label_available": "0",
            "raw_date": "",
            "raw_sequence_num": "",
            "cow_tag": "",
            "temperature_f": "",
            "mapping_mean_score": "",
        }
    return {
        "label_available": "1",
        "raw_date": label.get("date", ""),
        "raw_sequence_num": label.get("sequence_num", ""),
        "cow_tag": label.get("cow_tag", ""),
        "temperature_f": label.get("temperature_f", ""),
        "mapping_mean_score": label.get("mean_score", ""),
    }


def build_split_coco(coco, images, annotations_by_image_id, output_names):
    out_images = []
    out_annotations = []
    next_ann_id = 0

    for new_image_id, image in enumerate(images):
        original_image_id = image["id"]
        new_image = dict(image)
        new_image["id"] = new_image_id
        new_image["original_id"] = original_image_id
        new_image["file_name"] = output_names[original_image_id]
        out_images.append(new_image)

        for annotation in annotations_by_image_id.get(original_image_id, []):
            new_annotation = dict(annotation)
            new_annotation["id"] = next_ann_id
            new_annotation["image_id"] = new_image_id
            new_annotation["original_id"] = annotation.get("id")
            out_annotations.append(new_annotation)
            next_ann_id += 1

    return {
        "images": out_images,
        "annotations": out_annotations,
        "categories": coco.get("categories", []),
    }


def copy_images(source_root, fold_root, split, modality, images, output_names):
    img_dir = fold_root / modality / f"{split}_imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        source = source_root / image["file_name"]
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, img_dir / output_names[image["id"]])


def prepare_output(output_root, keep_existing):
    resolved = output_root.resolve()
    parent = output_root.parent.resolve()
    if output_root.exists() and not keep_existing:
        if resolved == parent or parent not in resolved.parents:
            raise RuntimeError(f"Refusing to clean unsafe output path: {resolved}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def make_fold_buckets(pairs, folds, seed):
    by_folder = defaultdict(list)
    for pair in pairs:
        by_folder[pair["folder"]].append(pair)

    if len(by_folder) < folds:
        raise RuntimeError(f"Need at least {folds} folders for grouped folds, found {len(by_folder)}.")

    folders = list(by_folder)
    random.Random(seed).shuffle(folders)
    folders.sort(
        key=lambda folder: (
            -sum(1 for pair in by_folder[folder] if pair["label"]["label_available"] == "1"),
            -len(by_folder[folder]),
            sort_key(folder),
        )
    )

    buckets = [[] for _ in range(folds)]
    bucket_pair_counts = [0 for _ in range(folds)]
    bucket_label_counts = [0 for _ in range(folds)]

    for folder in folders:
        labeled = sum(1 for pair in by_folder[folder] if pair["label"]["label_available"] == "1")
        index = min(
            range(folds),
            key=lambda item: (bucket_label_counts[item], bucket_pair_counts[item], item),
        )
        buckets[index].append(folder)
        bucket_pair_counts[index] += len(by_folder[folder])
        bucket_label_counts[index] += labeled

    return buckets


def split_pairs_for_fold(pairs, buckets, fold_index):
    test_folders = set(buckets[fold_index])
    val_folders = set(buckets[(fold_index + 1) % len(buckets)])
    split_to_pairs = {split: [] for split in SPLITS}

    for pair in pairs:
        if pair["folder"] in test_folders:
            split_to_pairs["test"].append(pair)
        elif pair["folder"] in val_folders:
            split_to_pairs["val"].append(pair)
        else:
            split_to_pairs["train"].append(pair)
    return split_to_pairs


def assert_no_overlap(split_to_pairs):
    seen = {}
    for split, pairs in split_to_pairs.items():
        for pair in pairs:
            key = pair["key"]
            if key in seen:
                raise RuntimeError(f"Pair {key} appears in both {seen[key]} and {split}")
            seen[key] = split


def unique_labels(pair_rows):
    seen = set()
    labels = []
    for row in pair_rows:
        if row["label_available"] != "1":
            continue
        key = (row["folder"], row["raw_date"], row["raw_sequence_num"])
        if key in seen:
            continue
        seen.add(key)
        labels.append({column: row[column] for column in LABEL_COLUMNS})
    return labels


def output_name_map(images, modality, copy_files):
    if copy_files:
        return {
            image["id"]: safe_name(modality, str(image["folder"]), str(image["frame_id"]).zfill(5))
            for image in images
        }
    return {image["id"]: image["file_name"] for image in images}


def main():
    parser = argparse.ArgumentParser(
        description="Create 5-fold synchronized RGB/thermal train/val/test manifests with temperature labels."
    )
    parser.add_argument("--rgb-json", default="data/annotations/rgb_keypoints.json", type=Path)
    parser.add_argument("--thermal-json", default="data/annotations/thermal_keypoints.json", type=Path)
    parser.add_argument("--source-root", default="datasets/keypoints", type=Path)
    parser.add_argument("--mapping", default="data/temperature_outputs/processed_raw_mapping.csv", type=Path)
    parser.add_argument("--output-root", default="datasets/keypoints/paired_rgb_thermal_5fold", type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--min-score", default=0.15, type=float)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into each fold. By default JSON/CSV manifests point to the original image paths.",
    )
    args = parser.parse_args()

    if args.folds < 2:
        raise RuntimeError("--folds must be at least 2.")

    rgb_coco = load_coco(args.rgb_json)
    thermal_coco = load_coco(args.thermal_json)
    labels_by_folder = read_temperature_mapping(args.mapping, args.min_score)

    rgb_by_key = {pair_key(image): image for image in rgb_coco["images"]}
    thermal_by_key = {pair_key(image): image for image in thermal_coco["images"]}
    shared_keys = sorted(set(rgb_by_key) & set(thermal_by_key), key=lambda key: (int(key[0]), int(key[1])))

    pairs = []
    for key in shared_keys:
        folder, frame_id = key
        pair = {
            "key": key,
            "folder": folder,
            "frame_id": frame_id,
            "rgb_image": rgb_by_key[key],
            "thermal_image": thermal_by_key[key],
        }
        pair["label"] = label_for_folder(folder, labels_by_folder)
        pairs.append(pair)

    buckets = make_fold_buckets(pairs, args.folds, args.seed)
    prepare_output(args.output_root, args.keep_existing)

    rgb_annotations = annotation_index(rgb_coco)
    thermal_annotations = annotation_index(thermal_coco)
    full_summary = {
        "folds": args.folds,
        "seed": args.seed,
        "min_score": args.min_score,
        "grouping": "folder",
        "copy_images": args.copy_images,
        "source_root": str(args.source_root),
        "paired_images": len(pairs),
        "labeled_paired_images": sum(1 for pair in pairs if pair["label"]["label_available"] == "1"),
        "labeled_folders": sorted(labels_by_folder, key=sort_key),
        "buckets": buckets,
        "folds_detail": {},
    }

    for fold_index in range(args.folds):
        fold_name = f"fold_{fold_index + 1}"
        fold_root = args.output_root / fold_name
        annotations_dir = fold_root / "annotations"
        split_to_pairs = split_pairs_for_fold(pairs, buckets, fold_index)
        assert_no_overlap(split_to_pairs)

        fold_summary = {}
        for split, split_pairs in split_to_pairs.items():
            rgb_images = [pair["rgb_image"] for pair in split_pairs]
            thermal_images = [pair["thermal_image"] for pair in split_pairs]
            rgb_names = output_name_map(rgb_images, "rgb", args.copy_images)
            thermal_names = output_name_map(thermal_images, "thermal", args.copy_images)

            if args.copy_images:
                copy_images(args.source_root, fold_root, split, "rgb", rgb_images, rgb_names)
                copy_images(args.source_root, fold_root, split, "thermal", thermal_images, thermal_names)

            rgb_split = build_split_coco(rgb_coco, rgb_images, rgb_annotations, rgb_names)
            thermal_split = build_split_coco(thermal_coco, thermal_images, thermal_annotations, thermal_names)
            write_json(annotations_dir / f"rgb_{split}.json", rgb_split)
            write_json(annotations_dir / f"thermal_{split}.json", thermal_split)

            pair_rows = []
            for pair_id, pair in enumerate(split_pairs):
                rgb_image = pair["rgb_image"]
                thermal_image = pair["thermal_image"]
                row = {
                    "pair_id": pair_id,
                    "split": split,
                    "folder": pair["folder"],
                    "frame_id": pair["frame_id"],
                    "rgb_file": rgb_names[rgb_image["id"]],
                    "thermal_file": thermal_names[thermal_image["id"]],
                    "rgb_original_file": rgb_image["file_name"],
                    "thermal_original_file": thermal_image["file_name"],
                    "rgb_original_id": rgb_image["id"],
                    "thermal_original_id": thermal_image["id"],
                }
                row.update(pair["label"])
                pair_rows.append(row)

            write_csv(annotations_dir / f"pairs_{split}.csv", pair_rows, PAIR_COLUMNS)
            write_csv(
                annotations_dir / f"labeled_pairs_{split}.csv",
                [row for row in pair_rows if row["label_available"] == "1"],
                PAIR_COLUMNS,
            )
            write_csv(annotations_dir / f"labels_{split}.csv", unique_labels(pair_rows), LABEL_COLUMNS)

            fold_summary[split] = {
                "pairs": len(pair_rows),
                "labeled_pairs": sum(1 for row in pair_rows if row["label_available"] == "1"),
                "folders": sorted({row["folder"] for row in pair_rows}, key=sort_key),
                "labeled_folders": sorted(
                    {row["folder"] for row in pair_rows if row["label_available"] == "1"},
                    key=sort_key,
                ),
                "rgb_images": len(rgb_split["images"]),
                "thermal_images": len(thermal_split["images"]),
                "rgb_annotations": len(rgb_split["annotations"]),
                "thermal_annotations": len(thermal_split["annotations"]),
            }

        write_json(annotations_dir / "split_summary.json", fold_summary)
        full_summary["folds_detail"][fold_name] = fold_summary

    write_json(args.output_root / "fold_summary.json", full_summary)
    print("Saved 5-fold paired RGB/thermal manifests:", args.output_root)
    print("Copy images:", args.copy_images)
    print("Paired images:", full_summary["paired_images"])
    print("Labeled paired images:", full_summary["labeled_paired_images"])
    for fold_name, fold_summary in full_summary["folds_detail"].items():
        parts = [
            f"{split}={values['pairs']} pairs/{values['labeled_pairs']} labeled"
            for split, values in fold_summary.items()
        ]
        print(f"{fold_name}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
